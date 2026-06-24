#!/usr/bin/env python3
# coding=utf-8
"""Retrospective DFlash checkpoint eval.

Scores one or more offline DFlash checkpoints after training and optionally logs
the metrics to W&B. The cache metrics reuse the offline trainer's loss/accuracy
path, while acceptance metrics run a direct greedy verifier simulation.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_dflash_acceptance import (  # noqa: E402
    PROMPTS,
    build_prompt,
    spec_generate_with_stats,
    summarize,
)
from scripts.train_dflash_offline import (  # noqa: E402
    DFlashHiddenStateDataset,
    _collect_files,
    batch_to_device,
    collate_batch,
    validate_manifests,
)
from specforge.core.dflash import OnlineDFlashModel  # noqa: E402
from specforge.modeling.draft import dflash as dflash_mod  # noqa: E402
from specforge.modeling.draft.dflash import DFlashDraftModel  # noqa: E402
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--cache-path", action="append", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-root", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-anchors", type=int, default=256)
    parser.add_argument("--anchor-sampling", choices=["uniform", "hard_position"], default="uniform")
    parser.add_argument("--attention-backend", choices=["eager", "sdpa", "flex_attention"], default="sdpa")
    parser.add_argument("--loss-type", default="dflash")
    parser.add_argument("--dpace-alpha", type=float, default=0.5)
    parser.add_argument("--loss-decay-gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-cache-eval", action="store_true")
    parser.add_argument("--skip-accept-eval", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default="online")
    return parser.parse_args()


def discover_checkpoints(args) -> list[Path]:
    checkpoints = [Path(path) for path in args.checkpoint]
    for root in args.checkpoint_root:
        checkpoints.extend(Path(root).glob("step_*"))
    checkpoints = [path for path in checkpoints if path.is_dir()]
    checkpoints = sorted(set(checkpoints), key=lambda path: (checkpoint_step(path), str(path)))
    if not checkpoints:
        raise FileNotFoundError("no checkpoint directories found")
    return checkpoints


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


@torch.no_grad()
def evaluate_cache(model, dataloader, device, block_size: int):
    model.eval()
    losses = []
    accuracies = []
    loss_tokens = []
    skipped = 0
    for batch in dataloader:
        if batch["loss_mask"].sum().item() < 2 * block_size:
            skipped += 1
            continue
        input_ids, hidden_states, loss_mask = batch_to_device(batch, device)
        loss, accuracy = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )
        losses.append(float(loss.detach().cpu()))
        accuracies.append(float(accuracy.detach().cpu()))
        loss_tokens.append(float(loss_mask.sum().detach().cpu()))
    if not losses:
        return {"eval/skipped": skipped, "eval/batches": 0}
    return {
        "eval/loss": sum(losses) / len(losses),
        "eval/accuracy": sum(accuracies) / len(accuracies),
        "eval/loss_tokens": sum(loss_tokens),
        "eval/skipped": skipped,
        "eval/batches": len(losses),
    }


@torch.no_grad()
def evaluate_acceptance(draft, target, tokenizer, args, device):
    eos = tokenizer.eos_token_id
    stop_token_ids = [eos] if eos is not None else None
    rows = []
    prompts = PROMPTS[: args.num_prompts]
    for i, prompt in enumerate(prompts):
        rendered = build_prompt(tokenizer, prompt)
        input_ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
        output_ids, steps, elapsed = spec_generate_with_stats(
            dflash=draft,
            target=target,
            sample_fn=dflash_mod.sample,
            extract_context_feature=dflash_mod.extract_context_feature,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=args.temperature,
        )
        suffix = [step["accepted_draft_suffix"] for step in steps]
        advance = [step["step_advance_tokens"] for step in steps]
        rows.append(
            {
                "prompt_index": i,
                "input_tokens": int(input_ids.shape[1]),
                "spec_new_tokens": int(output_ids.shape[1] - input_ids.shape[1]),
                "num_verify_steps": len(steps),
                "elapsed_s": elapsed,
                "accepted_draft_suffix": summarize(suffix),
                "step_advance_tokens": summarize(advance),
            }
        )

    all_suffix = [
        int(value)
        for row in rows
        for value, count in row["accepted_draft_suffix"].get("hist", {}).items()
        for _ in range(count)
    ]
    all_advance = [
        int(value)
        for row in rows
        for value, count in row["step_advance_tokens"].get("hist", {}).items()
        for _ in range(count)
    ]
    total_new = sum(row["spec_new_tokens"] for row in rows)
    total_elapsed = sum(row["elapsed_s"] for row in rows)
    suffix_summary = summarize(all_suffix)
    advance_summary = summarize(all_advance)
    return {
        "accept/prompts": len(rows),
        "accept/spec_new_tokens": total_new,
        "accept/spec_elapsed_s": total_elapsed,
        "accept/spec_tokens_per_s": total_new / total_elapsed if total_elapsed > 0 else None,
        "accept/accepted_suffix_mean": suffix_summary.get("mean"),
        "accept/accepted_suffix_p50": suffix_summary.get("p50"),
        "accept/accepted_suffix_p90": suffix_summary.get("p90"),
        "accept/accepted_suffix_max": suffix_summary.get("max"),
        "accept/step_advance_mean": advance_summary.get("mean"),
        "accept/step_advance_p50": advance_summary.get("p50"),
        "accept/step_advance_p90": advance_summary.get("p90"),
        "accept/step_advance_max": advance_summary.get("max"),
        "accept/accepted_suffix_hist": suffix_summary.get("hist", {}),
        "accept/step_advance_hist": advance_summary.get("hist", {}),
        "accept/results": rows,
    }


def flatten_for_wandb(metrics: dict):
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            flat[key] = value
    return flat


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DFlash checkpoint eval expects CUDA")

    checkpoints = discover_checkpoints(args)
    selected_layer_ids, hidden_size = validate_manifests(args.cache_path)
    target_config = AutoConfig.from_pretrained(args.target_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
    mask_token_id = tokenizer.mask_token_id
    if target_config.hidden_size != hidden_size:
        raise ValueError(
            f"cache hidden_size={hidden_size} but target hidden_size={target_config.hidden_size}"
        )

    dataloader = None
    target_components = None
    if not args.skip_cache_eval:
        eval_files = _collect_files(
            args.cache_path,
            args.max_eval_samples,
            args.seed,
            shuffle_files=True,
        )
        dataloader = DataLoader(
            DFlashHiddenStateDataset(eval_files, args.max_length),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )
        target_components = TargetEmbeddingsAndHead.from_pretrained(
            args.target_model_path,
            device=str(device),
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    target = None
    if not args.skip_accept_eval:
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model_path,
            dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).eval()
        target.requires_grad_(False)

    run = None
    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={**vars(args), "checkpoints": [str(path) for path in checkpoints]},
        )
        print(f"wandb_url={run.url}", flush=True)

    summaries = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        print(f"evaluating checkpoint={checkpoint} step={step}", flush=True)
        draft = DFlashDraftModel.from_pretrained(
            str(checkpoint), torch_dtype=torch.bfloat16
        ).to(device)
        draft.eval()
        draft.requires_grad_(False)

        metrics = {
            "checkpoint/path": str(checkpoint),
            "checkpoint/step": step,
            "block_size": int(draft.block_size),
            "target_layer_count": len(draft.target_layer_ids),
        }

        if dataloader is not None and target_components is not None:
            torch.manual_seed(args.seed + max(step, 0))
            cache_model = OnlineDFlashModel(
                draft_model=draft,
                target_lm_head=target_components.lm_head,
                target_embed_tokens=target_components.embed_tokens,
                block_size=int(draft.block_size),
                mask_token_id=mask_token_id,
                attention_backend=args.attention_backend,
                num_anchors=args.num_anchors,
                loss_decay_gamma=args.loss_decay_gamma,
                loss_type=args.loss_type,
                dpace_alpha=args.dpace_alpha,
                anchor_sampling=args.anchor_sampling,
            ).to(device)
            metrics.update(evaluate_cache(cache_model, dataloader, device, int(draft.block_size)))
            del cache_model

        if target is not None:
            metrics.update(evaluate_acceptance(draft, target, tokenizer, args, device))

        summaries.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if run is not None:
            run.log(flatten_for_wandb(metrics), step=step if step >= 0 else None)

        del draft
        torch.cuda.empty_cache()

    output = {"checkpoints": summaries}
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2, sort_keys=True))
        print(f"wrote {out}", flush=True)

    if run is not None:
        run.summary.update({"evaluated_checkpoints": len(summaries)})
        run.finish()


if __name__ == "__main__":
    main()
