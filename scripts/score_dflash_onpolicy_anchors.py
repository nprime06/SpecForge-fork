#!/usr/bin/env python3
"""Score DFlash anchors visited by the speculative decoding loop.

This script is meant for validation/mining experiments, not serving benchmarks.
It runs deterministic DFlash speculative decoding over prompts, records every
visited anchor, and writes compact JSONL rows with accept length plus draft and
target logprob diagnostics.
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from specforge.modeling.draft.dflash import DFlashDraftModel, extract_context_feature


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    flat = (logits / temperature).view(-1, vocab_size)
    probs = torch.softmax(flat, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def logprob_of(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    logits_f = logits.float()
    gathered = logits_f.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return gathered - torch.logsumexp(logits_f, dim=-1)


def as_float_list(tensor: torch.Tensor, digits: int = 6) -> list[float]:
    values = tensor.detach().float().cpu().tolist()
    return [round(float(v), digits) for v in values]


def normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    messages = []
    for message in raw_messages or []:
        role = message.get("role", message.get("from", "user"))
        content = message.get("content", message.get("value", ""))
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant", "model"}:
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            role = "user"
        if content:
            messages.append({"role": role, "content": str(content)})
    return messages


def prompt_from_example(example: dict[str, Any], tokenizer) -> str:
    for key in ("messages", "conversations"):
        if key in example and isinstance(example[key], list):
            messages = normalize_messages(example[key])
            assistant_indices = [
                i for i, message in enumerate(messages) if message["role"] == "assistant"
            ]
            if assistant_indices:
                messages = messages[: assistant_indices[-1]]
            if messages and getattr(tokenizer, "chat_template", None):
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            if messages:
                return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    for key in ("prompt", "instruction", "question", "input", "text"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            if getattr(tokenizer, "chat_template", None):
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": value}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return value

    raise ValueError(f"could not infer prompt field from keys={sorted(example.keys())}")


def iter_dataset(args) -> Iterable[dict[str, Any]]:
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=args.streaming,
    )
    return dataset


def encode_prompt(tokenizer, prompt: str, device: torch.device, max_prompt_tokens: int):
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    if input_ids.shape[1] > max_prompt_tokens:
        input_ids = input_ids[:, -max_prompt_tokens:]
    return input_ids.to(device)


@torch.inference_mode()
def score_prompt(
    *,
    dflash: DFlashDraftModel,
    target,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_index: int,
    source_index: Any,
    args,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + args.max_new_tokens
    block_size = dflash.block_size
    device = input_ids.device
    eos_token_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_token_ids.add(int(tokenizer.eos_token_id))
    for token_id in getattr(tokenizer, "additional_special_tokens_ids", []) or []:
        eos_token_ids.add(int(token_id))

    output_ids = torch.full(
        (1, max_length + block_size),
        int(dflash.mask_token_id),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    t0 = time.perf_counter()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    first_token = sample(output.logits, args.temperature)
    output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token
    target_hidden = extract_context_feature(output.hidden_states, dflash.target_layer_ids)

    rows: list[dict[str, Any]] = []
    start = num_input_tokens
    step_index = 0
    while start < max_length:
        step_index += 1
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        noise_embedding = target.model.embed_tokens(block_output_ids)

        draft_hidden = dflash(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[
                :, past_key_values_draft.get_seq_length() : start + block_size
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        draft_logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :])
        past_key_values_draft.crop(start)
        proposed = sample(draft_logits, args.temperature)
        block_output_ids[:, 1:] = proposed

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = sample(output.logits, args.temperature)
        target_next = posterior[:, :-1]
        accept_mask = block_output_ids[:, 1:] == target_next
        accepted_suffix = int(accept_mask.cumprod(dim=1).sum(dim=1)[0].item())
        step_advance = accepted_suffix + 1

        target_verify_logits = output.logits[:, :-1, :]
        draft_logp_proposed = logprob_of(draft_logits, proposed)[0]
        draft_logp_target_next = logprob_of(draft_logits, target_next)[0]
        target_logp_proposed = logprob_of(target_verify_logits, proposed)[0]
        target_logp_target_next = logprob_of(target_verify_logits, target_next)[0]
        dpace_proxy = float(torch.exp(draft_logp_target_next).prod().detach().cpu())

        rows.append(
            {
                "prompt_index": prompt_index,
                "source_index": source_index,
                "step_index": step_index,
                "anchor_pos": int(start),
                "block_size": int(block_size),
                "accepted_suffix": int(accepted_suffix),
                "step_advance_tokens": int(step_advance),
                "draft_tokens": block_output_ids[0, 1:].detach().cpu().tolist(),
                "target_next_tokens": target_next[0].detach().cpu().tolist(),
                "accepted_by_pos": accept_mask[0].detach().cpu().int().tolist(),
                "draft_logp_proposed": as_float_list(draft_logp_proposed),
                "draft_logp_target_next": as_float_list(draft_logp_target_next),
                "target_logp_proposed": as_float_list(target_logp_proposed),
                "target_logp_target_next": as_float_list(target_logp_target_next),
                "draft_ce_target_next_mean": round(
                    float((-draft_logp_target_next).mean().detach().cpu()), 6
                ),
                "target_ce_proposed_mean": round(
                    float((-target_logp_proposed).mean().detach().cpu()), 6
                ),
                "dpace_accept_proxy": round(dpace_proxy, 8),
            }
        )

        output_ids[:, start : start + step_advance] = block_output_ids[
            :, :step_advance
        ]
        output_ids[:, start + step_advance] = posterior[:, accepted_suffix]
        start += step_advance
        past_key_values_target.crop(start)
        target_hidden = extract_context_feature(
            output.hidden_states, dflash.target_layer_ids
        )[:, :step_advance, :]

        generated = output_ids[0, num_input_tokens:start].detach().cpu().tolist()
        if eos_token_ids and any(token_id in eos_token_ids for token_id in generated):
            break

    elapsed = time.perf_counter() - t0
    summary = {
        "prompt_index": prompt_index,
        "source_index": source_index,
        "prompt_tokens": int(num_input_tokens),
        "generated_tokens": int(max(0, start - num_input_tokens)),
        "steps": len(rows),
        "elapsed_sec": round(elapsed, 6),
    }
    return rows, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--dflash-checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-interval", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DFlash on-policy scoring expects CUDA")
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16

    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    if world_size > 1:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    target.eval()

    dflash = DFlashDraftModel.from_pretrained(
        args.dflash_checkpoint,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    dflash.eval()
    if getattr(dflash, "mask_token_id", None) is None:
        dflash.mask_token_id = getattr(dflash.config, "dflash_config", {}).get(
            "mask_token_id"
        )
    if dflash.mask_token_id is None:
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        dflash.mask_token_id = tokenizer.mask_token_id
    if dflash.mask_token_id is None:
        raise ValueError("could not determine DFlash mask_token_id")

    shard_path = output_dir / f"anchors_rank{rank:03d}.jsonl"
    summary_path = output_dir / f"summaries_rank{rank:03d}.jsonl"
    hist = Counter()
    processed = 0
    skipped = 0
    started = time.time()

    with shard_path.open("a") as anchor_f, summary_path.open("a") as summary_f:
        for global_index, example in enumerate(iter_dataset(args)):
            if args.max_prompts is not None and global_index >= args.max_prompts:
                break
            if global_index % world_size != rank:
                continue
            try:
                prompt = prompt_from_example(example, tokenizer)
                input_ids = encode_prompt(
                    tokenizer, prompt, device, args.max_prompt_tokens
                )
                rows, summary = score_prompt(
                    dflash=dflash,
                    target=target,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    prompt_index=global_index,
                    source_index=example.get("source_index", global_index),
                    args=args,
                )
            except Exception as exc:
                skipped += 1
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "prompt_index": global_index,
                            "error": repr(exc),
                        }
                    ),
                    flush=True,
                )
                continue

            for row in rows:
                anchor_f.write(json.dumps(row) + "\n")
                hist[row["step_advance_tokens"]] += 1
            summary["rank"] = rank
            summary_f.write(json.dumps(summary) + "\n")
            anchor_f.flush()
            summary_f.flush()
            processed += 1

            if processed % max(1, args.log_interval) == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "processed": processed,
                            "skipped": skipped,
                            "anchors": sum(hist.values()),
                            "prompts_per_sec": round(processed / elapsed, 4),
                            "step_advance_hist": dict(sorted(hist.items())),
                        }
                    ),
                    flush=True,
                )

    final = {
        "rank": rank,
        "processed": processed,
        "skipped": skipped,
        "anchors": sum(hist.values()),
        "step_advance_hist": dict(sorted(hist.items())),
    }
    (output_dir / f"final_rank{rank:03d}.json").write_text(json.dumps(final, indent=2))
    if world_size > 1:
        dist.barrier()
    if is_main_process(rank):
        print(json.dumps({"done": True, "output_dir": str(output_dir)}), flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
