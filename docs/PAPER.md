# The paper matrix: pre-registration and how to run it

Claim under test: **a linear readout at the last token of a wrong number the model has just written predicts whether the model's final answer will follow that number, beyond everything observable about the situation, and the direction is causally used.**
Modelled on "How Much is Left?" (Merzouk et al.): same vocabulary - floors, a ladder of predictors, selectivity controls, a sampling-noise ceiling, a cross-dataset transfer matrix, steering with a quality check, and an event analysis on natural occurrences.

## Cells
| cell | model | dataset | status |
|---|---|---|---|
| qwen-gsm8k | Qwen2.5-7B-Instruct | GSM8K | main + inject + robust run exist; only stoch / natural are added |
| qwen-math | Qwen2.5-7B-Instruct | MATH (integer-answer subset) | gen + states exist from the main run; steps -> inject -> robust -> extras are new |
| r1-gsm8k, r1-math | DeepSeek-R1-Distill-Qwen-7B | GSM8K, MATH | everything is new (shared gen+states, no constrained prompt, no deletion edit) |

## Metrics (fixed before the runs)
- Label `flip` (final answer after a greedy continuation of the edited prefix differs from the unedited final answer); continuous `prop` = log10(1+|final - original final|). Unfinished continuations (no end-of-sequence within the budget) are excluded and counted - the analogue of the paper's naturally-terminated filter. For R1 this exclusion can be large because greedy decoding of long reasoning can loop; it is reported per cell, never hidden.
- Ladder (iid and easy->hard shift): constant floor; position; surface text + hashed bag of words; early layer; the model's own surprise; **pre-edit state carried forward (the analogue of their exact countdown)**; injection-time probe. Reported: AUROC, R2, Brier skill, MAE / constant-median-MAE, TPR at 10% FPR.
- Controls: shuffled labels (global and within-problem), fold seeds 0-2 (note: the fold seed only changes hyper-parameter selection, so small variance there is expected and is *not* the paper's three training seeds).
- Sampling ceiling: for 500-1000 test rows, 6-8 sampled continuations of the same edited prefix; oracle that knows the model's own follow probability (greedy and leave-one-out); `captured_vs_floor` = (AUROC_probe - 0.5)/(AUROC_oracle - 0.5).
- Transfer: probe frozen on one dataset, evaluated on the other (same model), against constant and the target's best observable baseline.
- Steering at the injected token (C8): as in docs/ROBUST.md plus quality check (NLL of steered continuations under the unsteered model) and a paired problem-clustered sign-flip test against alpha = 0.
- Natural errors: the model's own two-operand arithmetic slips (deterministic scan), probe trained on injected rows of training problems only, applied at the last token of the slip; `matters` = final answer changes when the slip is corrected. N1: AUROC for `matters` against relative position and relative size of the slip. N2: slip vs matched correct equations of the same solution.
- Prompt-end reference (their Table 4/5 analogue, `lenref`): total-length probe at the end of the prompt with floor, surface ridge, shuffled labels and sampling floor, on the same models/datasets - a sanity anchor for the pipeline.

## Cross-cell rule
- **REPLICATES (diagnostic)**: every cell reaches ROBUST (diagnostic) - C1-C7 pass.
- **CAUSAL REPLICATES**: additionally C8 SUPPORTED in all cells but at most one, and in at least one.
- **MIXED / NOT REPLICATED**: otherwise, with the failing cells listed. A failing cell is reported, not dropped.
Natural-error (N1/N2), ceiling and transfer results are reported with their intervals but do not enter the verdict; they decide how the claim is worded.

## How to run (login node)
```
cd ~/probe_lab_linear
bash cluster/fetch_model.sh                                  # once, login node: caches DeepSeek-R1-Distill-Qwen-7B (compute nodes are offline)
SMOKE=1 CELLS="r1-gsm8k qwen-math" bash cluster/matrix.sh    # ~1-2 h dry run, 40 problems per dataset, writes *_psmoke dirs; look at it before the real run
bash cluster/matrix.sh                                       # all four cells; independent chains, resumable (resubmit after a time-out)
bash cluster/matrix.sh paper                                 # when the cell-analyze jobs are done -> $SCRATCH/psuite/outputs/paper/PAPER_REPORT.md
```
Knobs (environment): `R1_LIMIT` (problems per dataset for R1, default 500), `R1_GEN_H`, `R1_STEPS_H`, `R1_STEER_H`, `QM_LIMIT` (800), `QM_STEPS_H`, `R1_ROBUST_N`, `STOCH_N`. Upper bound of node-hours if every budget is used: about 45 (R1 about 28, Qwen-MATH about 10, extras for Qwen-GSM8K about 3); budgets are caps and a cut-off leaves a random subset of problems, never a biased one.

## Known risks (check in the smoke run)
- R1-Distill emits long `<think>` text; line-based step splitting and answer parsing were built for Qwen's step-by-step prose. The smoke run prints the finished rate and the number of editable numeric steps per solution; if fewer than about 2 per solution survive, tell me before spending the full budget.
- R1 greedy decoding can loop; the exclusion rate then dominates the cell's interpretation.
- Qwen-MATH uses the integer-answer subset only (deterministic ground truth).
