#!/usr/bin/env python3
"""Score RedHatAI DFlash anchors from existing rollout token caches.

This is a correctness-first validation script for
RedHatAI/Qwen3-8B-speculator.dflash.  It reuses existing rollout token records
from the chunked SpecForge cache, but re-extracts the RedHat DFlash auxiliary
target hidden states live because that checkpoint expects layers
[2, 10, 18, 26, 34].
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from speculators.models.dflash import DFlashDraftModel


def list_chunk_files(cache_dir: Path) -> list[Path]:
    files = sorted(cache_dir.glob("shard_*/part_*.pt"))
    if not files:
        raise FileNotFoundError(f"no shard_*/part_*.pt files under {cache_dir}")
    return files


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"unsupported chunk format in {path}")


def choose_anchors(
    loss_mask: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
    anchors_per_record: int,
    mode: str,
    rng: random.Random,
) -> list[int]:
    max_anchor = seq_len - block_size
    if max_anchor <= 0:
        return []
    valid = (loss_mask[: max_anchor + 1] > 0).nonzero(as_tuple=True)[0].tolist()
    if not valid:
        return []
    if mode == "all":
        return valid
    if mode == "stride":
        stride = max(1, math.ceil(len(valid) / anchors_per_record))
        return valid[::stride][:anchors_per_record]
    if mode == "random":
        count = min(anchors_per_record, len(valid))
        return sorted(rng.sample(valid, count))
    raise ValueError(f"unknown anchor mode: {mode}")


def load_causal_lm(path: str, *, device: torch.device):
    kwargs = {
        "device_map": None,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, **kwargs
        )
    return model.eval().to(device)


@torch.inference_mode()
def target_forward_hidden(target, input_ids: torch.Tensor):
    try:
        return target(
            input_ids,
            use_cache=False,
            output_hidden_states=True,
            logits_to_keep=1,
        )
    except TypeError:
        return target(input_ids, use_cache=False, output_hidden_states=True)


@torch.inference_mode()
def target_verify_logits(target, verify_ids: torch.Tensor, block_size: int) -> torch.Tensor:
    try:
        out = target(verify_ids, use_cache=False, logits_to_keep=block_size)
        logits = out.logits[0]
        return logits[: block_size - 1]
    except TypeError:
        out = target(verify_ids, use_cache=False)
        anchor = verify_ids.shape[1] - block_size
        return out.logits[0, anchor : anchor + block_size - 1]


def logprob_of(logits: torch.Tensor, token_ids: torch.Tensor) -> list[float]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    values = log_probs.gather(-1, token_ids.view(-1, 1)).squeeze(-1)
    return [round(float(v), 6) for v in values.detach().cpu().tolist()]


def prefix_accept_len(proposed: torch.Tensor, target_greedy: torch.Tensor) -> tuple[int, list[int]]:
    matches = (proposed == target_greedy).detach().cpu().int().tolist()
    accepted = 0
    for match in matches:
        if match:
            accepted += 1
        else:
            break
    return accepted + 1, matches


@torch.inference_mode()
def score_record(
    *,
    target,
    draft: DFlashDraftModel,
    record: dict[str, Any],
    record_index: int,
    chunk_path: str,
    anchors: list[int],
    device: torch.device,
) -> list[dict[str, Any]]:
    block_size = int(draft.config.block_size)
    layer_ids = list(draft.config.aux_hidden_state_layer_ids)
    input_ids = torch.as_tensor(record["input_ids"], dtype=torch.long, device=device)
    seq_len = int(input_ids.numel())
    input_ids_b = input_ids.unsqueeze(0)

    output = target_forward_hidden(target, input_ids_b)
    hidden_tuple = output.hidden_states
    aux_hidden = torch.cat([hidden_tuple[i] for i in layer_ids], dim=-1).to(torch.bfloat16)
    verifier_last = hidden_tuple[-1].to(torch.bfloat16)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    lengths = torch.tensor([seq_len], dtype=torch.long, device=device)

    rows: list[dict[str, Any]] = []
    source_index = record.get("source_index")
    for anchor in anchors:
        loss_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
        loss_mask[0, anchor] = True
        draft_tokens, _loss, _metrics = draft(
            hidden_states=aux_hidden,
            input_ids=input_ids_b,
            loss_mask=loss_mask,
            verifier_last_hidden_states=verifier_last,
            lengths=lengths,
            position_ids=position_ids,
        )
        draft_ids = draft_tokens[0, 1:block_size].long()
        if getattr(draft, "use_draft_vocab", False):
            # Speculators stores d2t as an offset:
            # target_token_id = draft_vocab_id + d2t[draft_vocab_id].
            proposed = draft_ids + draft.d2t[draft_ids].long()
        else:
            proposed = draft_ids

        verify_ids = torch.cat([input_ids[: anchor + 1], proposed], dim=0).unsqueeze(0)
        verify_logits = target_verify_logits(target, verify_ids, block_size)
        target_greedy = torch.argmax(verify_logits.float(), dim=-1).long()
        accept_len, matches = prefix_accept_len(proposed, target_greedy)

        rows.append(
            {
                "score_mode": "redhat_live_target_greedy_verify",
                "record_index": record_index,
                "source_index": source_index,
                "chunk_path": chunk_path,
                "seq_len": seq_len,
                "anchor_pos": int(anchor),
                "block_size": block_size,
                "speculative_tokens": block_size - 1,
                "accept_len": int(accept_len),
                "accepted_spec_tokens": int(accept_len - 1),
                "draft_vocab_ids": draft_ids.detach().cpu().tolist(),
                "draft_target_ids": proposed.detach().cpu().tolist(),
                "target_greedy_ids": target_greedy.detach().cpu().tolist(),
                "accepted_by_pos": matches,
                "target_logp_draft_by_pos": logprob_of(verify_logits, proposed),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--anchors-per-record", type=int, default=8)
    parser.add_argument("--anchor-mode", choices=["random", "stride", "all"], default="stride")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-interval", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this scorer")
    device = torch.device("cuda:0")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "anchors.jsonl"
    summary_path = output_dir / "summary.json"

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    target = load_causal_lm(args.target_model_path, device=device)
    draft = DFlashDraftModel.from_pretrained(
        args.draft_checkpoint,
        torch_dtype=torch.bfloat16,
    ).eval().to(device)
    draft.config.max_anchors = 1
    block_size = int(draft.config.block_size)
    if getattr(draft, "use_draft_vocab", False) and getattr(draft, "d2t", None) is None:
        raise RuntimeError("draft uses a reduced vocab but has no d2t mapping")

    files = list_chunk_files(Path(args.cache_dir))
    processed_records = 0
    skipped_records = 0
    total_anchors = 0
    hist: Counter[int] = Counter()
    t0 = time.perf_counter()

    with rows_path.open("w") as out:
        for chunk_path in files:
            for record in load_records(chunk_path):
                if args.max_records is not None and processed_records >= args.max_records:
                    break
                input_ids = torch.as_tensor(record["input_ids"], dtype=torch.long)
                loss_mask = torch.as_tensor(record.get("loss_mask"), dtype=torch.bool)
                anchors = choose_anchors(
                    loss_mask,
                    seq_len=int(input_ids.numel()),
                    block_size=block_size,
                    anchors_per_record=args.anchors_per_record,
                    mode=args.anchor_mode,
                    rng=rng,
                )
                if not anchors:
                    skipped_records += 1
                    continue
                rows = score_record(
                    target=target,
                    draft=draft,
                    record=record,
                    record_index=processed_records,
                    chunk_path=str(chunk_path),
                    anchors=anchors,
                    device=device,
                )
                for row in rows:
                    out.write(json.dumps(row) + "\n")
                    hist[int(row["accept_len"])] += 1
                out.flush()
                total_anchors += len(rows)
                processed_records += 1
                if processed_records % args.log_interval == 0:
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    print(
                        json.dumps(
                            {
                                "processed_records": processed_records,
                                "skipped_records": skipped_records,
                                "anchors": total_anchors,
                                "records_per_sec": round(processed_records / elapsed, 4),
                                "hist": dict(sorted(hist.items())),
                            }
                        ),
                        flush=True,
                    )
            if args.max_records is not None and processed_records >= args.max_records:
                break

    mean_accept = (
        sum(k * v for k, v in hist.items()) / sum(hist.values()) if hist else 0.0
    )
    summary = {
        "done": True,
        "cache_dir": args.cache_dir,
        "target_model_path": args.target_model_path,
        "draft_checkpoint": args.draft_checkpoint,
        "output_dir": str(output_dir),
        "processed_records": processed_records,
        "skipped_records": skipped_records,
        "anchors": total_anchors,
        "mean_accept_len": mean_accept,
        "hist": dict(sorted(hist.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
