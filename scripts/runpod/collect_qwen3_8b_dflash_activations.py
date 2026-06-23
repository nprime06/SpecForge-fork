#!/usr/bin/env python3
"""Fast prefill-only DFlash activation collector for Qwen3-8B.

This script intentionally avoids logits and generation. It loads one target
model copy on the visible GPU, captures selected transformer layer outputs via
forward hooks, and writes chunked torch files containing activation records.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--dataset", default="JingweiSong/perfectblend-qwen3-8b-regen-no-thinking"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="/workspace/hf-cache")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--capture-layers", default="1,12,22,33")
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--chunk-records", type=int, default=64)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--stop-after-tokens", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    conversations = row.get("conversations") or row.get("messages")
    if not conversations:
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            return [{"role": "user", "content": text}]
        return None

    messages = []
    for msg in conversations:
        role = msg.get("role") or msg.get("from")
        content = msg.get("content") or msg.get("value")
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant", "model"}:
            role = "assistant"
        elif role in {"system"}:
            role = "system"
        else:
            continue
        if isinstance(content, list):
            content = "\n".join(str(x) for x in content)
        if content is None:
            content = ""
        messages.append({"role": role, "content": str(content)})
    return messages or None


def tokenize_with_assistant_mask(tokenizer, messages, max_length: int) -> dict[str, torch.Tensor] | None:
    def fallback_template_ids(prefix_messages) -> torch.Tensor:
        if not prefix_messages:
            return torch.empty(0, dtype=torch.long)
        text = tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        return encoded["input_ids"][0].to(torch.long)

    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=max_length,
            add_generation_prompt=False,
        )
        input_ids = encoded["input_ids"][0].to(torch.long)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask[0].to(torch.long)
        assistant_mask = encoded.get("assistant_masks")
        if assistant_mask is None:
            assistant_mask = encoded.get("assistant_tokens_mask")
        if assistant_mask is None:
            raise ValueError("chat template did not return an assistant mask")
        loss_mask = assistant_mask[0].to(torch.long)
        if loss_mask.sum().item() == 0:
            raise ValueError("chat template returned an empty assistant mask")
    except Exception:
        # Fallback for chat templates without assistant mask support. This is
        # less exact but keeps collection moving; the manifest records it.
        prompt_ids = fallback_template_ids(messages)
        input_ids = prompt_ids
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.zeros_like(input_ids)
        cursor = 0
        for idx, msg in enumerate(messages):
            prefix = fallback_template_ids(messages[:idx])
            upto = fallback_template_ids(messages[: idx + 1])
            start = min(prefix.numel(), input_ids.numel())
            end = min(upto.numel(), input_ids.numel())
            if msg["role"] == "assistant" and end > start:
                loss_mask[start:end] = 1
            cursor = end
            if cursor >= input_ids.numel():
                break

    if input_ids.numel() < 2:
        return None
    loss_mask[-1] = 0
    if loss_mask.sum().item() == 0:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
    }


def flush_chunk(
    output_dir: Path,
    shard_index: int,
    part_index: int,
    records: list[dict[str, Any]],
) -> Path:
    shard_dir = output_dir / f"shard_{shard_index:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"part_{part_index:06d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"records": records}, tmp)
    os.replace(tmp, path)
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_ids = [int(x) for x in args.capture_layers.split(",") if x.strip()]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    ).eval()
    core_model = getattr(model, "model")
    if args.compile:
        core_model = torch.compile(core_model, mode="reduce-overhead")

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for layer_id in layer_ids:
        layer = core_model.layers[layer_id]

        def make_hook(idx):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[idx] = hidden.detach()

            return hook

        hooks.append(layer.register_forward_hook(make_hook(layer_id)))

    manifest = {
        "format": "qwen3_8b_dflash_chunked_v1",
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "max_length": args.max_length,
        "selected_layer_ids": layer_ids,
        "hidden_size": model.config.hidden_size,
        "stored_hidden_dtype": "bfloat16",
        "logits": False,
        "generation": False,
        "use_cache": False,
        "num_shards": args.num_shards,
        "created_unix": time.time(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    ds = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    total = len(ds)
    indices = range(args.shard_index, total, args.num_shards)
    if args.limit_rows is not None:
        indices = list(indices)[: args.limit_rows]

    pending = []
    pending_tokens = 0
    chunk = []
    part_index = 0
    rows_seen = 0
    rows_written = 0
    tokens_written = 0
    loss_tokens_written = 0
    start_time = time.time()

    def run_batch(batch_items):
        nonlocal chunk, part_index, rows_written, tokens_written, loss_tokens_written
        max_len = max(x["input_ids"].numel() for x in batch_items)
        bsz = len(batch_items)
        input_ids = torch.full(
            (bsz, max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device="cuda",
        )
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device="cuda")
        loss_masks = []
        lengths = []
        source_indices = []
        for row_idx, item in enumerate(batch_items):
            seq_len = item["input_ids"].numel()
            input_ids[row_idx, :seq_len] = item["input_ids"].to("cuda")
            attention_mask[row_idx, :seq_len] = 1
            loss_masks.append(item["loss_mask"])
            lengths.append(seq_len)
            source_indices.append(item["source_index"])

        captured.clear()
        with torch.inference_mode():
            core_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        torch.cuda.synchronize()

        missing = [idx for idx in layer_ids if idx not in captured]
        if missing:
            raise RuntimeError(f"missing captured layers: {missing}")

        layer_stack = torch.stack([captured[idx].to(torch.bfloat16).cpu() for idx in layer_ids], dim=2)
        input_cpu = input_ids.cpu()
        attn_cpu = attention_mask.cpu()

        for i, seq_len in enumerate(lengths):
            record = {
                "input_ids": input_cpu[i, :seq_len].clone(),
                "attention_mask": attn_cpu[i, :seq_len].clone(),
                "loss_mask": loss_masks[i].cpu().clone(),
                "selected_hidden_states": layer_stack[i, :seq_len].contiguous().clone(),
                "selected_layer_ids": layer_ids,
                "hidden_size": model.config.hidden_size,
                "source_index": int(source_indices[i]),
            }
            chunk.append(record)
            rows_written += 1
            tokens_written += int(seq_len)
            loss_tokens_written += int(record["loss_mask"].sum().item())
            if len(chunk) >= args.chunk_records:
                flush_chunk(output_dir, args.shard_index, part_index, chunk)
                part_index += 1
                chunk = []

        del input_ids, attention_mask, input_cpu, attn_cpu, layer_stack
        captured.clear()

    progress_total = len(indices) if isinstance(indices, list) else None
    pbar = tqdm(indices, total=progress_total, desc=f"collect shard {args.shard_index}/{args.num_shards}")
    for source_index in pbar:
        row = ds[int(source_index)]
        if row.get("status") not in (None, "success"):
            rows_seen += 1
            continue
        messages = normalize_messages(row)
        if not messages:
            rows_seen += 1
            continue
        encoded = tokenize_with_assistant_mask(tokenizer, messages, args.max_length)
        rows_seen += 1
        if encoded is None:
            continue
        encoded["source_index"] = int(source_index)
        seq_len = int(encoded["input_ids"].numel())
        if pending and pending_tokens + seq_len > args.max_batch_tokens:
            run_batch(pending)
            pending = []
            pending_tokens = 0
            elapsed = max(time.time() - start_time, 1e-6)
            pbar.set_postfix(
                rows=rows_written,
                mtok=f"{tokens_written / 1e6:.3f}",
                tok_s=f"{tokens_written / elapsed:.0f}",
            )
            if args.stop_after_tokens is not None and tokens_written >= args.stop_after_tokens:
                break
        pending.append(encoded)
        pending_tokens += seq_len

    if pending and (
        args.stop_after_tokens is None or tokens_written < args.stop_after_tokens
    ):
        run_batch(pending)
    if chunk:
        flush_chunk(output_dir, args.shard_index, part_index, chunk)

    stats = {
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "tokens_written": tokens_written,
        "loss_tokens_written": loss_tokens_written,
        "elapsed_sec": time.time() - start_time,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    stats_path = output_dir / f"stats_shard_{args.shard_index:02d}.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    for hook in hooks:
        hook.remove()
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
