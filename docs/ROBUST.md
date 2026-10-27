# ROBUST run: is `I[num]:flip` real, and is the direction causal? (pre-registration)

Written BEFORE the run. The decision rules below are fixed; the report applies them mechanically.

## Question
The inject run found that the hidden state at the last token of a wrong number written into the model's own chain of thought predicts whether the final answer will FOLLOW the error (`flip`):
R2 0.615 (iid) / 0.590 (shift), gain +0.109 / +0.129 over the best observable baseline, and 0.615 vs 0.480 for the natural pre-edit state. Open caveats: `flip` also fires when a step is
simply decorative-vs-load-bearing, the last numeric step flips ~85% (position could carry it), between-problem variance, one model, correlational only.

## Part A - offline checks (no generation; `psuite.robust`)
Same corrected protocol as every other result (grouped K-fold choice of layer and lambda over the training pool, problem-clustered bootstrap, iid and easy->hard shift), against the inject baselines
PLUS hashed bag-of-words of the question and of the edited prefix (`Bbow-question`, `Bbow-prefix`, `Bbow+obs`).

| check | what | passes when |
|---|---|---|
| C1 | drop the last numeric step of every problem | generic SUPPORTED (CI lower bound of gain over best observable > 0 and R2 > 0.05, iid AND shift) |
| C2 | label demeaned within problem (rows of problems with > 2 edits) | generic SUPPORTED |
| C3 | variant n1 alone and n2 alone | both SUPPORTED |
| C4 | all rows, extended baselines | generic SUPPORTED |
| C5 | whole pipeline on shuffled labels (3 seeds) | max test R2 <= 0.02 |
| C6 | paired: R2(injection-time state) - R2(natural pre-edit state), clustered bootstrap, iid | CI lower bound > 0 |
| C7 | AUROC difference vs best observable, clustered bootstrap | CI lower bound > 0 in iid AND shift |

Also reported without a decision: TPR at 10% FPR, AUROC by step index and last-vs-other, CV R2 by layer for `I[num]`, `I[line]`, `Pre[nat]`.

## Part B - causal test (`psuite.istear`)
Prompt = chat prompt + the edited solution cut right after the last character of the injected number; so the last prompt token is exactly where the probe reads. A forward hook adds
`alpha * s * unit_direction` to the residual stream of one layer at that single position during prefill only (nothing is added while generating). The model then continues greedily (max 384 new tokens);
`flip` = final answer differs from the answer of its unedited solution. Every condition, alpha = 0 included, uses this protocol, so effects are paired per row (about 3000 TEST-problem rows, deterministic sample).

* Directions: the flip-probe direction, the flip mean-difference direction (flip=1 minus flip=0), 3 random norm-matched directions. `s` = sd of the probe projection over the training rows.
* Layers: the CV-best layer for `I[num]:flip` and the layer nearest 12 (never the last stored state: it is post-final-norm and a layer hook cannot write there).
* Doses: +-8 and +-16 sd first (priority order, so a cut budget leaves the informative conditions), then +-2, +-4. **Never 32:** the earlier steering run showed every direction, random ones too, breaks the model there.
* Slope: OLS of per-row flip on alpha over |alpha| <= 16, cluster bootstrap over problems.

Per layer, SUPPORTED needs ALL of: slope CI lower bound > 0; slope > 2x the largest |random-direction slope|; effect > 0 at +8 and +16 and < 0 at -8 and -16;
finished rate and parsable-answer rate within 0.05 of the alpha=0 baseline at |alpha| <= 8. PARTIAL = sign and slope CI hold but random-ratio or integrity fails.
NEGATIVE = slope CI inside +-0.003 flip-rate per sd. Overall status = best layer. A condition counts only if every shard produced it. Accuracy is reported but not gated
(a steer that makes the model follow the error lowers accuracy by construction).
C8 = SUPPORTED.

## Verdict
* **ROBUST + CAUSAL**: C1-C8. **ROBUST (diagnostic)**: C1-C7 only (the signal is real and not an artefact of position / text, but the direction is not a lever at these doses, or C8 was not evaluated). **FRAGILE**: any of C1-C7 fails, failed checks listed.

## What a result would and would not mean
* ROBUST (diagnostic): a monitor can read, at the moment an error is written, whether the model will follow it, beyond free text, position and the model's own surprise. It does not say the model "knows" it is wrong.
* C8 passing would say the direction is used, not only correlated; it would not say the model deliberately recovers. A null at C8 is informative, not a failure of the build.
* Single model (Qwen2.5-7B-Instruct), GSM8K only, greedy decoding, edits are +1 / +10% of a step result. No R1-Distill or second-model replication in this run.

## Running
```
MODE=robust-smoke bash cluster/submit.sh      # ~1 h, writes to $MEGA_DIR/robust_smoke (does not touch real results)
MODE=robust       bash cluster/submit.sh      # prep -> steer (afterany), offline (independent), analyze (afterany steer + offline)
MODE=robust-analyze bash cluster/submit.sh    # rebuild the report from what exists
```
Report: `$MEGA_DIR/robust/analysis/ROBUST_REPORT.md` (+ `robust_scorecard.csv`, `robust_results.csv`). Needs `$MEGA_DIR/steps`, `$MEGA_DIR/inject`, `$MEGA_DIR/steer/dirs.npz`.
