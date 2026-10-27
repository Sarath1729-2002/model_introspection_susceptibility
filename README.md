# model_introspection_probe_suite

Linear probes for what a frozen LLM's residual stream knows about its **own continuation**, in the quantities a monitor would act on:
will it run away, how long, will it obey a constraint, is its answer already settled, how much is left. Every label is a deterministic function of greedy-decoded text
(plus deterministic verifiers for the constraints), the baselines are explicit ladders, and the scorecard in `REPORT.md` applies pre-declared strict rules (`docs/PROTOCOL.md`).

Model: Qwen2.5-7B-Instruct. Data: GSM8K, MATH (integer answers), ARC (multiple choice), synthetic arithmetic, each passed through a ground-truth audit (admitted only at >= 99.5% pass).
No model is trained: the only fitted objects are closed-form ridge probes.

## Stages (all resumable; one process per GPU; whole-node jobs)
`setup.sh` (login node, once) -> `states` (prompt-end states for the plain and the constrained prompt) -> `gen` (greedy plain, greedy constrained, 4 sampled plain solutions)
-> `trace` (hidden states + forced-answer log-probs at 6 checkpoints along the greedy plain solution) -> `analyze` (blocks P, C, K, tables, one per GPU) -> `report`.

## Run it
```
bash cluster/setup.sh                       # once. Reuses ~/venvs/susc and the cached model if they exist. Refuses to rebuild an existing problem set.
MODE=gate bash cluster/submit.sh            # optional dry run on 64 problems per dataset
MODE=full bash cluster/submit.sh            # states -> gen -> trace -> analyze -> report, chained with --dependency=afterok
```
Recovery: `MODE=resume` (gen hit its time budget), `MODE=rescue RESCUE_SHARD=<n>` (one shard crashed: its leftover chunks run on all 4 GPUs, then trace -> analyze -> report), `MODE=analyze`, `MODE=report`.
Never edit on the cluster and never pull while jobs run.

## Look at the results
```
cat <RUN_DIR>/analysis/REPORT.md
cat <RUN_DIR>/analysis/scorecard.csv
```
where `<RUN_DIR>` is `$SCRATCH/psuite/outputs/qwen7b-psuite`.

## Layout
`src/psuite`: `bench.py` (datasets + ground-truth audit), `verify.py` (constraints and their verifiers), `prompts.py`, `hf_util.py`, `gen.py`, `states.py`, `trace.py`, `table.py` (labels + baseline ladders),
`probes.py`, `study.py`, `report.py`, `core.py`/`ridge.py` (probe machinery), `cli.py`. `tests/`: verifier tests, a tiny-model end-to-end test that checks the forced log-probs and the
causality of the stored states against direct forwards, and planted-signal tests of the analysis (including a signal that does NOT generalise and must not be scored SUPPORTED).


## Uncertainty POC (added)
Linear probes for answer-distribution uncertainty (pass rate / answer entropy / majority share from rollouts, at the prompt end and after 48 / 96 tokens of the greedy solution), with split-half label reliability, positive controls and a decision rule written in advance.
`docs/UNCERTAINTY_POC.md` has the design and the rule. Run after the main study: `MODE=unc bash cluster/submit.sh` (smoke: `UNC_N=40 MODE=unc ...`), read `$RUN_DIR/unc_n400/analysis/UNC_REPORT.md`.

## Mega run (added)
One overnight GSM8K-only job with four independent threads (causal step importance by intervention, uncertainty on 2000 unseen problems, steering test of the length direction, corrected re-analysis), corrected CV protocol and pre-registered rules: see `docs/MEGA.md`.
`MODE=mega-smoke bash cluster/submit.sh` (dry run) then `MODE=mega bash cluster/submit.sh`; report: `$RUN_DIR/mega/analysis/MEGA_REPORT.md`.

## Inject run (added)
Error propagation at injection time (probe on the state at the last token of a number injected into the model's own chain of thought, vs the model's own surprise and the pre-edit state) + steering at large doses. Design and pre-registered rule: `docs/INJECT.md`. After the mega run: `MODE=inject-smoke` / `MODE=inject bash cluster/submit.sh`; report `$RUN_DIR/mega/inject/analysis/INJECT_REPORT.md`.


## Robust run (after the inject run)
`MODE=robust-smoke` then `MODE=robust bash cluster/submit.sh`: robustness checks C1-C7 of `I[num]:flip` (subsets, demeaned label, bag-of-words baselines, shuffled control, paired vs the pre-edit state, AUROC)
plus a causal steering test at the injected token (C8). Pre-registration and decision rules: `docs/ROBUST.md`. Report: `$MEGA_DIR/robust/analysis/ROBUST_REPORT.md`.


## Paper matrix (model x dataset) - see docs/PAPER.md
`cluster/matrix.sh` runs the cells (Qwen2.5-7B and DeepSeek-R1-Distill-Qwen-7B x GSM8K and MATH), adds the sampling-noise ceiling (`stoch`), the natural-slip test (`natural`), the prompt-end length reference (`lenref`) and the transfer matrix (`matrix`), and `psuite.paper` assembles `PAPER_REPORT.md`. `docs/INTERIM_REPORT.md` holds the finished Qwen/GSM8K cell.
