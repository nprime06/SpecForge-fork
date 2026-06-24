#!/usr/bin/env python3
"""Direct DFlash checkpoint acceptance smoke/eval.

This intentionally avoids the SGLang EAGLE3 benchmark path: it loads the target
HF model and the DFlash draft checkpoint directly, runs greedy speculative
decode, and reports the accepted block lengths from the DFlash verifier loop.
"""

import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


PROMPTS = [
    "Explain why speculative decoding can speed up transformer inference in three concise paragraphs.",
    "Write a Python function that computes the longest increasing subsequence length and explain the complexity.",
    "A startup has 18 engineers and wants to ship a reliable eval platform in 6 weeks. Draft a practical plan.",
    "Solve step by step: if a train travels 135 miles in 2.25 hours, what is its average speed in mph?",
    "Give a careful comparison between beam search and nucleus sampling for assistant-style chat models.",
    "Turn this into a warm but professional email: I need the report by Friday or the launch slips.",
    "In a multi-turn support conversation, ask one clarifying question before proposing a fix for intermittent timeouts.",
    "Summarize the tradeoffs of storing activation caches offline for draft-model training.",
]


def load_dflash_module(repo_root: Path):
    path = repo_root / "specforge" / "modeling" / "draft" / "dflash.py"
    spec = importlib.util.spec_from_file_location("dflash_local", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_prompt(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return text


@torch.inference_mode()
def target_generate(target, input_ids, max_new_tokens, eos_token_id):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = target.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        use_cache=True,
    )
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


@torch.inference_mode()
def spec_generate_with_stats(
    dflash,
    target,
    sample_fn,
    extract_context_feature,
    input_ids,
    max_new_tokens,
    stop_token_ids,
    temperature,
):
    dflash.eval()
    target.eval()

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = dflash.block_size
    device = input_ids.device

    output_ids = torch.full(
        (1, max_length + block_size),
        dflash.mask_token_id,
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    torch.cuda.synchronize()
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
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample_fn(
        output.logits, temperature
    )
    target_hidden = extract_context_feature(output.hidden_states, dflash.target_layer_ids)

    steps = []
    start = input_ids.shape[1]
    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        noise_embedding = target.model.embed_tokens(block_output_ids)

        draft_logits = target.lm_head(
            dflash(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size + 1 :, :]
        )
        past_key_values_draft.crop(start)
        block_output_ids[:, 1:] = sample_fn(draft_logits, temperature)

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )

        posterior = sample_fn(output.logits, temperature)
        accepted_suffix = (
            (block_output_ids[:, 1:] == posterior[:, :-1])
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )
        step_advance = accepted_suffix + 1
        output_ids[:, start : start + step_advance] = block_output_ids[
            :, :step_advance
        ]
        output_ids[:, start + step_advance] = posterior[:, accepted_suffix]
        start += step_advance
        past_key_values_target.crop(start)
        target_hidden = extract_context_feature(
            output.hidden_states, dflash.target_layer_ids
        )[:, :step_advance, :]
        steps.append(
            {
                "accepted_draft_suffix": int(accepted_suffix),
                "step_advance_tokens": int(step_advance),
            }
        )

        if stop_token_ids is not None:
            generated = output_ids[:, num_input_tokens:start]
            if torch.isin(
                generated, torch.tensor(stop_token_ids, device=device)
            ).any().item():
                break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != dflash.mask_token_id]
    if stop_token_ids is not None:
        stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_indices = torch.isin(output_ids[0][num_input_tokens:], stop_tensor).nonzero(
            as_tuple=True
        )[0]
        if stop_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]

    return output_ids, steps, elapsed


def summarize(values):
    if not values:
        return {"count": 0}
    values = list(values)
    values_sorted = sorted(values)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "p50": values_sorted[len(values_sorted) // 2],
        "p90": values_sorted[int(0.9 * (len(values_sorted) - 1))],
        "max": max(values),
        "hist": dict(sorted(Counter(values).items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument(
        "--draft-repo", default="qwen3-30b-dflash/qwen3-30b-a3b-dflash-smokes"
    )
    parser.add_argument("--draft-subfolder", default="step_00010000")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    dflash_mod = load_dflash_module(repo_root)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model, trust_remote_code=True, use_fast=True
    )
    eos = tokenizer.eos_token_id
    stop_token_ids = [eos] if eos is not None else None

    print("loading target", args.target_model, flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()
    target.requires_grad_(False)

    print("loading draft", args.draft_repo, args.draft_subfolder, flush=True)
    dflash = dflash_mod.DFlashDraftModel.from_pretrained(
        args.draft_repo,
        subfolder=args.draft_subfolder,
        dtype=torch.bfloat16,
    ).to("cuda")
    dflash.eval()
    dflash.requires_grad_(False)

    results = []
    prompts = PROMPTS[: args.num_prompts]
    for i, prompt in enumerate(prompts):
        rendered = build_prompt(tokenizer, prompt)
        input_ids = tokenizer(rendered, return_tensors="pt").input_ids.to("cuda")
        print(
            f"prompt={i} input_tokens={input_ids.shape[1]} max_new={args.max_new_tokens}",
            flush=True,
        )

        spec_ids, steps, spec_elapsed = spec_generate_with_stats(
            dflash=dflash,
            target=target,
            sample_fn=dflash_mod.sample,
            extract_context_feature=dflash_mod.extract_context_feature,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=args.temperature,
        )
        spec_new = spec_ids.shape[1] - input_ids.shape[1]

        target_ids, target_elapsed = target_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos,
        )
        target_new = target_ids.shape[1] - input_ids.shape[1]

        suffix = [s["accepted_draft_suffix"] for s in steps]
        advance = [s["step_advance_tokens"] for s in steps]
        row = {
            "prompt_index": i,
            "input_tokens": int(input_ids.shape[1]),
            "spec_new_tokens": int(spec_new),
            "target_new_tokens": int(target_new),
            "spec_elapsed_s": spec_elapsed,
            "target_elapsed_s": target_elapsed,
            "spec_tokens_per_s": spec_new / spec_elapsed if spec_elapsed > 0 else None,
            "target_tokens_per_s": target_new / target_elapsed
            if target_elapsed > 0
            else None,
            "num_verify_steps": len(steps),
            "accepted_draft_suffix": summarize(suffix),
            "step_advance_tokens": summarize(advance),
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        results.append(row)

    all_suffix = [
        v
        for row in results
        for v, count in row["accepted_draft_suffix"].get("hist", {}).items()
        for _ in range(count)
    ]
    all_advance = [
        v
        for row in results
        for v, count in row["step_advance_tokens"].get("hist", {}).items()
        for _ in range(count)
    ]
    total_spec_tokens = sum(r["spec_new_tokens"] for r in results)
    total_target_tokens = sum(r["target_new_tokens"] for r in results)
    total_spec_s = sum(r["spec_elapsed_s"] for r in results)
    total_target_s = sum(r["target_elapsed_s"] for r in results)

    summary = {
        "target_model": args.target_model,
        "draft_repo": args.draft_repo,
        "draft_subfolder": args.draft_subfolder,
        "block_size": int(dflash.block_size),
        "target_layer_ids": list(map(int, dflash.target_layer_ids)),
        "num_prompts": len(results),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "total_spec_new_tokens": total_spec_tokens,
        "total_target_new_tokens": total_target_tokens,
        "total_spec_elapsed_s": total_spec_s,
        "total_target_elapsed_s": total_target_s,
        "spec_tokens_per_s": total_spec_tokens / total_spec_s
        if total_spec_s > 0
        else None,
        "target_tokens_per_s": total_target_tokens / total_target_s
        if total_target_s > 0
        else None,
        "accepted_draft_suffix": summarize(all_suffix),
        "step_advance_tokens": summarize(all_advance),
        "results": results,
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
