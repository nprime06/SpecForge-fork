# DFlash On-Policy Anchor Validation

This branch is for testing whether DFlash undertrains the anchors it actually
visits at inference time, and whether posttraining on those anchors improves
acceptance length.

## Question

DFlash training currently samples random valid anchors from target-regenerated
offline sequences. Inference does not visit anchors uniformly: each next anchor
is determined by the model's own accepted draft prefix length. If poor
acceptance clusters in specific regimes, random anchors may underweight the
states that matter most for real speculative decoding.

The experiment has two phases:

1. Score anchor hardness for many candidate positions.
2. If hardness is non-uniform, posttrain on a mixture of random anchors and
   on-policy hard anchors.

## Anchor Scores

For a block size of 16, each anchor at position `t` scores draft behavior for
positions `t + 1` through `t + 15`. Position `t` is the real anchor token and is
not part of the loss.

### Teacher-Forced Score

This is the cheap first pass. It uses cached target hidden states and one
DFlash checkpoint, without running target verification.

For each anchor:

- run DFlash on cached target hidden states for the chosen block;
- compare draft logits against the cached target-generated tokens;
- record per-position cross entropy and probability on the target token;
- compute prefix argmax match length against the cached tokens;
- compute a D-PACE-style proxy from the cumulative product of target-token
  probabilities.

This answers whether an anchor is locally easy for the draft under teacher
forcing. It should be batched across many anchors from the same sequence.

Suggested fields:

```json
{
  "record_id": 123,
  "source_index": 456,
  "anchor_pos": 812,
  "block_size": 16,
  "score_mode": "teacher_forced",
  "loss_tokens": 15,
  "draft_ce_mean": 1.73,
  "draft_ce_by_pos": [0.2, 0.4, 2.1],
  "draft_p_target_by_pos": [0.82, 0.67, 0.12],
  "teacher_forced_prefix_match": 2,
  "dpace_accept_proxy": 0.054
}
```

### Exact Greedy Verifier Score

This is the slower confirmation pass. It matches deterministic speculative
decoding at `temperature=0`.

For each anchor:

- let DFlash greedily propose the draft suffix;
- run the target model verifier on the anchor plus proposed suffix;
- greedily sample the target posterior;
- compute accepted suffix length as the first mismatch prefix length.

This is closer to real inference but requires target forward passes. Run it on
a stratified subset selected from the teacher-forced pass.

## Why Not Score One Anchor At A Time?

One-anchor scoring will be too slow. The validation path should use batched
anchor scoring:

- load a chunk of cached sequences;
- choose many candidate anchors per sequence;
- flatten `(sequence, anchor)` pairs into a batch of DFlash draft blocks;
- reuse the cached target hidden states for all anchors;
- write scores in append-only shards.

For exact verifier scoring, batch anchors by similar context length where
possible. The target verifier is the expensive part, so the first exact pass
should be small and stratified:

- easy anchors by teacher-forced score;
- medium anchors;
- very hard anchors;
- anchors sampled from actual DFlash rollout positions.

## On-Policy Anchor Mining

After the score pass, collect anchors from the inference process itself:

1. Start from the prompt prefix.
2. Run DFlash for a block.
3. Verify with the target.
4. Record the visited anchor and accepted suffix length.
5. Advance by `accepted_suffix + 1`.
6. Repeat until max length or EOS.

Store only compact anchor rows, not another hidden-state cache:

```json
{
  "record_id": 123,
  "anchor_pos": 812,
  "accepted_suffix": 2,
  "step_advance_tokens": 3,
  "draft_ce_mean": 1.73,
  "dpace_accept_proxy": 0.054,
  "selection_reason": "on_policy_visit"
}
```

If anchors refer to positions inside the existing cached target-generated
sequences, posttraining can reuse the hidden-state cache. If speculative
rollouts create new token sequences, target hidden states for those generated
contexts must be collected separately.

## Posttraining

Add an explicit-anchor path to DFlash training:

- `OnlineDFlashModel.forward(..., anchor_positions=None, block_keep_mask=None)`;
- if anchors are omitted, keep the existing random-anchor behavior;
- if anchors are provided, train on exactly those blocks;
- optionally return per-anchor diagnostics for validation scripts.

Initial mixture:

- 70% random anchors;
- 30% on-policy or hard anchors.

Aggressive follow-up:

- 50% random anchors;
- 50% hard anchors.

Avoid 100% hard anchors until acceptance evaluations show no regression on
ordinary anchors. Some hard anchors may be irreducible target entropy rather
than useful learning signal.

## Success Criteria

The validation pass should answer:

- Does hardness vary substantially across anchors?
- Does teacher-forced hardness predict exact greedy acceptance length?
- Are on-policy visited anchors harder than random anchors?
- Are hard anchors clustered by sequence position, response region, entropy, or
  turn type?

The posttraining pass should be judged by acceptance length, not just loss:

- mean and percentile `step_advance_tokens`;
- histogram of `accepted_suffix`;
- acceptance by position in block;
- regression on random-anchor validation slices.

## Implementation Order

1. Add a chunked-cache dataset reader for the RunPod Qwen3-8B cache format.
2. Add batched teacher-forced anchor scoring.
3. Add exact greedy verifier scoring for a smaller stratified subset.
4. Add explicit-anchor DFlash training support.
5. Add anchor-table posttraining with random/hard mixture sampling.
6. Run acceptance evaluation before and after posttraining.
