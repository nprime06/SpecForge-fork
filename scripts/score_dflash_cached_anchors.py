#!/usr/bin/env python3
"""Teacher-forced DFlash anchor scoring from cached target hidden states."""

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from specforge.core.dflash import (
    OnlineDFlashModel,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead


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


def read_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def list_chunk_files(cache_dir: Path) -> list[Path]:
    files = sorted(cache_dir.glob("shard_*/part_*.pt"))
    if not files:
        raise FileNotFoundError(f"no shard_*/part_*.pt files under {cache_dir}")
    return files


def as_float_list(tensor: torch.Tensor, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in tensor.detach().float().cpu().tolist()]


def choose_anchors(
    loss_mask: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
    anchors_per_record: int,
    mode: str,
    rng: random.Random,
) -> torch.Tensor:
    max_anchor = max(seq_len - block_size, 0)
    if max_anchor <= 0:
        return torch.empty(0, dtype=torch.long)
    valid = (loss_mask[: max_anchor + 1] > 0.5).nonzero(as_tuple=True)[0].cpu()
    if valid.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    if mode == "all":
        anchors = valid
    elif mode == "stride":
        stride = max(1, math.ceil(valid.numel() / anchors_per_record))
        anchors = valid[::stride][:anchors_per_record]
    elif mode == "random":
        count = min(anchors_per_record, valid.numel())
        indices = sorted(rng.sample(range(valid.numel()), count))
        anchors = valid[indices]
    else:
        raise ValueError(f"unknown anchor mode: {mode}")
    return anchors.long()


@torch.inference_mode()
def score_record(
    *,
    helper: OnlineDFlashModel,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    loss_mask: torch.Tensor,
    anchors: torch.Tensor,
    source_index: Any,
    record_index: int,
    chunk_path: str,
    args,
) -> list[dict[str, Any]]:
    device = next(helper.parameters()).device
    block_size = helper.block_size
    seq_len = int(input_ids.shape[0])
    if anchors.numel() == 0:
        return []

    input_ids_b = input_ids.to(device=device, dtype=torch.long).unsqueeze(0)
    hidden_b = hidden_states.to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    anchor_positions = anchors.to(device=device, dtype=torch.long).view(1, -1)
    keep_mask = torch.ones_like(anchor_positions, dtype=torch.bool, device=device)

    noise_embedding = helper._create_noise_embed(input_ids_b, anchor_positions, keep_mask)
    context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    draft_position_ids = helper._create_position_ids(anchor_positions)
    full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

    if args.attention_backend == "flex_attention":
        attn_mask = create_dflash_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=keep_mask,
            S=seq_len,
            block_size=block_size,
            device=device,
        )
    else:
        attn_mask = create_dflash_sdpa_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=keep_mask,
            S=seq_len,
            block_size=block_size,
            device=device,
        )

    output_hidden = helper.draft_model(
        position_ids=full_position_ids,
        noise_embedding=noise_embedding,
        target_hidden=hidden_b,
        attention_mask=attn_mask,
    )
    logits = helper.lm_head(output_hidden).view(1, anchors.numel(), block_size, -1)
    score_logits = logits[:, :, 1:, :].float()

    offsets = torch.arange(1, block_size, device=device).view(1, -1)
    label_indices = anchor_positions.view(-1, 1) + offsets
    target_ids = input_ids_b[0].gather(0, label_indices.reshape(-1)).view(
        anchors.numel(), block_size - 1
    )
    flat_logits = score_logits.reshape(-1, score_logits.shape[-1])
    flat_targets = target_ids.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none").view(
        anchors.numel(), block_size - 1
    )
    p_target = torch.exp(-ce)
    argmax_ids = torch.argmax(score_logits[0], dim=-1)
    matches = argmax_ids == target_ids
    prefix_match = matches.cumprod(dim=1).sum(dim=1)
    dpace_proxy = p_target.prod(dim=1)

    rows = []
    for i, anchor in enumerate(anchors.tolist()):
        rows.append(
            {
                "score_mode": "teacher_forced_cache",
                "record_index": record_index,
                "source_index": source_index,
                "chunk_path": chunk_path,
                "anchor_pos": int(anchor),
                "block_size": int(block_size),
                "loss_tokens": int(block_size - 1),
                "teacher_forced_prefix_match": int(prefix_match[i].detach().cpu()),
                "draft_ce_target_mean": round(float(ce[i].mean().detach().cpu()), 6),
                "draft_p_target_geom_mean": round(
                    float(torch.exp(torch.log(p_target[i].clamp_min(1e-45)).mean()).detach().cpu()),
                    8,
                ),
                "dpace_accept_proxy": round(float(dpace_proxy[i].detach().cpu()), 10),
                "draft_ce_by_pos": as_float_list(ce[i]),
                "draft_p_target_by_pos": as_float_list(p_target[i], digits=8),
                "draft_argmax_matches_target": matches[i].detach().cpu().int().tolist(),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--target-model-path", default=None)
    parser.add_argument("--dflash-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--anchors-per-record", type=int, default=64)
    parser.add_argument("--anchor-mode", choices=["random", "stride", "all"], default="random")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--attention-backend",
        choices=["eager", "sdpa", "flex_attention"],
        default="sdpa",
    )
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--embedding-key", type=str, default=None)
    parser.add_argument("--lm-head-key", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("cached DFlash anchor scoring expects CUDA")

    cache_dir = Path(args.cache_dir)
    manifest = read_manifest(cache_dir)
    selected_layer_ids = manifest["selected_layer_ids"]
    hidden_size = int(manifest["hidden_size"])
    target_model_path = args.target_model_path or manifest.get("model") or manifest.get(
        "target_model_path"
    )
    if target_model_path is None:
        raise ValueError("target model path not provided and not present in manifest")

    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
        (output_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2))
    if world_size > 1:
        dist.barrier()

    target_config = AutoConfig.from_pretrained(
        target_model_path, trust_remote_code=args.trust_remote_code
    )
    if int(target_config.hidden_size) != hidden_size:
        raise ValueError(
            f"cache hidden_size={hidden_size} but target hidden_size={target_config.hidden_size}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path, trust_remote_code=args.trust_remote_code
    )
    mask_token_id = args.mask_token_id
    if mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("could not determine mask token id")

    dflash = DFlashDraftModel.from_pretrained(
        args.dflash_checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    dflash.eval()
    dflash.mask_token_id = mask_token_id

    expected_layers = list(getattr(dflash, "target_layer_ids", []))
    if expected_layers and expected_layers != selected_layer_ids:
        raise ValueError(
            f"checkpoint target_layer_ids={expected_layers} but cache has {selected_layer_ids}"
        )
    block_size = args.block_size or int(getattr(dflash, "block_size", 16))

    target_components = TargetEmbeddingsAndHead.from_pretrained(
        target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=str(device),
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )
    helper = OnlineDFlashModel(
        draft_model=dflash,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.anchors_per_record,
    ).to(device)
    helper.eval()

    files = list_chunk_files(cache_dir)
    files = [path for i, path in enumerate(files) if i % world_size == rank]
    rng = random.Random(args.seed + rank)
    shard_path = output_dir / f"cached_anchor_scores_rank{rank:03d}.jsonl"
    summary_path = output_dir / f"cached_anchor_summary_rank{rank:03d}.json"
    processed_records = 0
    scored_anchors = 0
    skipped_records = 0
    prefix_hist = Counter()
    started = time.time()

    with shard_path.open("a") as out:
        for chunk_path in files:
            chunk = torch.load(chunk_path, map_location="cpu", mmap=True)
            records = chunk.get("records", [])
            for local_record_index, record in enumerate(records):
                if args.max_records is not None and processed_records >= args.max_records:
                    break
                input_ids = record["input_ids"].long()
                loss_mask = record["loss_mask"].float().clone()
                hidden = record["selected_hidden_states"].to(torch.bfloat16)
                if hidden.ndim != 3:
                    raise ValueError(
                        f"selected_hidden_states must be [seq,layers,hidden], got {tuple(hidden.shape)}"
                    )
                if hidden.shape[1] != len(selected_layer_ids):
                    raise ValueError(
                        f"hidden layer axis {hidden.shape[1]} != manifest layers {selected_layer_ids}"
                    )
                seq_len = min(input_ids.shape[0], loss_mask.shape[0], hidden.shape[0])
                input_ids = input_ids[:seq_len]
                loss_mask = loss_mask[:seq_len]
                hidden = hidden[:seq_len].flatten(start_dim=1)
                if seq_len > 0:
                    loss_mask[-1] = 0
                anchors = choose_anchors(
                    loss_mask,
                    seq_len=seq_len,
                    block_size=block_size,
                    anchors_per_record=args.anchors_per_record,
                    mode=args.anchor_mode,
                    rng=rng,
                )
                if anchors.numel() == 0:
                    skipped_records += 1
                    continue
                rows = score_record(
                    helper=helper,
                    input_ids=input_ids,
                    hidden_states=hidden,
                    loss_mask=loss_mask,
                    anchors=anchors,
                    source_index=record.get("source_index"),
                    record_index=processed_records,
                    chunk_path=str(chunk_path),
                    args=args,
                )
                for row in rows:
                    out.write(json.dumps(row) + "\n")
                    prefix_hist[row["teacher_forced_prefix_match"]] += 1
                out.flush()
                processed_records += 1
                scored_anchors += len(rows)
                if processed_records % max(1, args.log_interval) == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "rank": rank,
                                "processed_records": processed_records,
                                "scored_anchors": scored_anchors,
                                "skipped_records": skipped_records,
                                "anchors_per_sec": round(scored_anchors / elapsed, 2),
                                "prefix_match_hist": dict(sorted(prefix_hist.items())),
                            }
                        ),
                        flush=True,
                    )
            if args.max_records is not None and processed_records >= args.max_records:
                break

    summary = {
        "rank": rank,
        "processed_records": processed_records,
        "scored_anchors": scored_anchors,
        "skipped_records": skipped_records,
        "prefix_match_hist": dict(sorted(prefix_hist.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    if world_size > 1:
        dist.barrier()
    if is_main_process(rank):
        print(json.dumps({"done": True, "output_dir": str(output_dir)}), flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
