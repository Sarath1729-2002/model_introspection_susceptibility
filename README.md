# model_introspection_susceptibility

**Can we read, from a frozen LLM's hidden states, that it is about to defer to a wrong answer it was handed?**  Linear probes (regression labels only) and monitors for
*hint adoption / sycophancy susceptibility* in Qwen2.5-7B-Instruct, on five problem sets whose ground truth is deterministic and audited. Probing and monitoring only - no steering
(that is the next study; this one is designed to tell us whether it is worth doing and along which direction).

Read first: [`docs/PROTOCOL.md`](docs/PROTOCOL.md) (hypotheses and decision rules, declared before the run) and [`docs/MENTOR_BRIEF.md`](docs/MENTOR_BRIEF.md) (what this buys us and where it can go).

## What gets measured
| stage | what | where |
|---|---|---|
| audit | every dataset must pass deterministic ground-truth checks (replayed GSM8K calculator annotations, single-boxed-integer MATH, re-executed synthetic code, ...) | `bench.py` |
| behaviour | 7 prompt conditions per problem (`clean`, 4 wrong-hint framings, `placebo`, correct-hint), 8 sampled solutions each | `sample.py` |
| states | last-prompt-token hidden state (all layers) + forced-answer log P(gold) / log P(wrong), per condition | `states.py` |
| trace | hidden states at 5 points along one sampled solution (before it starts ... just before the final `\boxed`) | `trace.py` |
| probes | A prospective (clean state -> susceptibility) · B in-context (hinted state -> will it follow) · C delta-state · D leave-one-framing-out · E geometry/specificity | `probes.py` |
| causal-by-design | placebo/correct-hint contrasts, randomised hint offset/position, generator-set difficulty, belief->behaviour decomposition, difficulty-adjusted contrast, monitoring operating points | `causal.py` |
| trace monitoring | probe vs text-so-far baselines along the solution, incl. *silent adoption* (hint value never written) | `monitor.py` |
| report | `analysis/REPORT.md` + figures + CSVs, headline bullets generated from the numbers | `report.py` |

Datasets: GSM8K · MATH (integer-answer subset) · synthetic arithmetic chains · synthetic Python programs (tier A) · ARC-Challenge (tier B, expert key; analysed separately).
Splits: iid (by problem) · difficulty shift (easy -> hard within each dataset) · leave-one-dataset-out.

## Run on TamIA (account aip-oberman, whole nodes)
```bash
git pull                                   # on the cluster, never edit there
bash cluster/setup.sh                      # LOGIN node, once: venv, model, GSM8K/MATH/ARC download, synthetic sets, ground-truth audit -> problems.jsonl
MODE=gate bash cluster/submit.sh           # optional ~1h dry run on 48 problems/dataset  -> ${RUN_DIR}-gate/analysis/REPORT.md
MODE=full bash cluster/submit.sh           # the study (chained with afterok): states -> sample -> trace -> analyze (4 GPUs in parallel) -> report
MODE=resume bash cluster/submit.sh         # only if the sampling time budget cut the run short
MODE=analyze bash cluster/submit.sh        # re-run analysis + report on whatever exists
```
Timing (4xH100, one node): states ~1 h, sampling ~6-7 h (budget `SAMPLE_BUDGET_H=8.5`, then it stops cleanly; datasets are interleaved so a cut run still covers all of them),
trace ~0.5 h, analysis ~1-1.5 h. Total ~9-10 h + queue. For a hard 12 h ceiling set `SAMPLE_BUDGET_H=7`.
Knobs (env): `N_SAMPLES=8 SIZES='{"gsm8k":2400,"math":1800,"arc":1500,"arith":1000,"prog":1000}' DATASETS=gsm8k,math MODEL_ID=... SAMPLE_BUDGET_H=8.5`.
Two nodes at once (optional): node 1 `DATASETS=gsm8k,arith,prog MODE=part bash cluster/submit.sh`, node 2 `DATASETS=math,arc MODE=part bash cluster/submit.sh` (states + sampling for those sets only; chunk files are
dataset-keyed so they never collide), then once both finish `MODE=resume bash cluster/submit.sh` (sampling returns instantly, then trace -> analysis -> report on everything).
Output: `$RUN_DIR/analysis/REPORT.md` (+ `fig*.png`, `*.csv`, `*.pkl`), `$RUN_DIR/ground_truth_audit.json`.
Notifications: `SLACK_WEBHOOK_URL` / `NTFY_TOPIC` in `~/.bashrc` (see `cluster/config.sh`).

## Tests
`pytest tests` - tiny random Qwen2: prompt-end state equals a direct forward, resumability, time-budget stop, trace position 0 equals the pre-generation state, audit catches corrupted labels, leave-dataset-out really excludes the held-out set.
`tests/test_signal.py` plants a known linear signal in fabricated states and checks the analysis **recovers it, passes against observable baselines, transfers across datasets, rises along the trace, and returns no PASS when the signal is removed.**

## Layout
```
src/susc/  bench prompts textutil | sample states trace sched hf_util | core ridge table | probes causal monitor | study report cli
cluster/   config.sh setup.sh submit.sh run.sbatch
docs/      PROTOCOL.md MENTOR_BRIEF.md
tests/
```
