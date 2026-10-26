# Inject run - does the state right after a wrong number know the error will propagate?

Pre-registered before any data. Runs AFTER the mega run (it re-uses the mega steps outcomes and the length probe's directions). GSM8K only, Qwen2.5-7B-Instruct.

## Question and monitor decision
A model writes a wrong intermediate number into its own chain of thought. Either the final answer follows the error (the CoT is load-bearing) or the model silently recovers (the CoT is not what decides the answer).
If a linear probe on the hidden state at the moment of injection predicts which, a monitor can decide per instance whether reading the written chain tells it anything about the answer (trust / escalate / replace the CoT monitor).
Prior art (to re-check before writing): Phan et al. 2026 (arXiv 2603.16331) read recovery AFTER generation, at answer tokens; this run asks whether it is readable at injection time. The literature report disputes whether error-handling probes track final correctness at all (arXiv 2608.07528), so a null is informative.

## Labels (deterministic, from the intervention, no new generation)
For every edited step of the mega steps thread (n1: result c -> c+1; n2: c -> c+max(10, c//10)) the model continued greedily from the edited prefix.
`flip` = final answer differs from the unedited solution's answer (no parsable answer counts as a flip). `prop` = log10(1 + |final - original final|), how far the error travelled (NaN if unparsable). `flip_step` = mean of the two variants.
Noise floor: the steps thread's null regeneration (unedited prefix) flip rate; the thread is INCONCLUSIVE if it exceeds 0.15 or the label std < 0.1.

## Probe inputs
HA = hidden state (layers 4,8,...,28) at the LAST TOKEN OF THE INJECTED NUMBER, from one forward over prompt + edited prefix. Nothing after it exists. **Primary.**
HB = state at the end of the edited line (secondary). Pre[nat] = the natural state of the same step BEFORE the edit (from the steps thread) as a probe family, for the "did the injection add anything" comparison.
Ridge, (layer, lambda) by grouped 5-fold CV over the training pool, bootstrap over problems, iid and easy->hard shift (cv_eval.py).

## Baselines (all observable at injection time; the verdict is against the best of them, chosen on TEST - conservative for the probe)
Bpos (step index, token position) - Bedit (variant, edit size, relative size) - **Bsurprise** (the model's own NLL / min log-prob / first-token entropy of the injected number, of the original number at the same place, and their difference: free "does the model notice the error" signals) -
Btext (step text, question, edited-prefix features) - **Bobs-all** (everything above) - Hpost8 (early layer at the injection position) - **Hpre8, Hpre16 (the natural pre-edit state)**.
Oracle (shown, never decides): how often the result is reused later, relative step position, tokens remaining.

## Decision rule
Headline row: **I[num]:flip**. SUPPORTED = probe R2 > 0.05 and CI lower bound of the gain over the best observable baseline > 0 in iid AND shift. PARTIAL = one regime. NEGATIVE = gain CI upper bound < 0.05 everywhere. INCONCLUSIVE otherwise.
I[num]:prop, I[line]:*, Pre[nat]:* are secondary, reported with the same rule but not the headline (multiplicity: 9 status rows, one pre-declared headline).
Interpretation guide: SUPPORTED on I[num] but not Pre[nat] = the injection moment carries information the pre-edit state did not. SUPPORTED on both = it was already readable before the error (mega steps result, +0.026 over layer 8). NEGATIVE with Bsurprise winning = the model's output distribution already tells a monitor everything the probe does.

## Second part: steering at large doses
The mega steering test found probe slope 0.007 log-tokens/sd (+4 sd = +3% tokens), only 1.75x the largest random direction: statistically present, practically tiny, doses too small. Same directions (from mega/steer/dirs.npz), |alpha| in {4, 8, 16, 32} sd, 4 random norm-matched controls.
Same rule (probe slope CI > 0, > 1.5x largest random slope) with accuracy damage <= 0.05 for |alpha| <= 8 (not 2). A dose table compares probe / mean-difference / mean |random| per dose. At very large alpha any perturbation breaks the model; read the table, not only the slope.

## Run
```
MODE=inject-smoke bash cluster/submit.sh     # optional, ~30 min (writes complete chunks into mega/inject - harmless - and steer_big_smoke)
MODE=inject bash cluster/submit.sh           # forwards only for the inject stage (~20-40 min on 4 GPUs) + steering job, analysis after both
cat /scratch/s/sarath02/psuite/outputs/qwen7b-psuite/mega/inject/analysis/INJECT_REPORT.md
```
Needs `mega/steps` and `mega/steer/dirs.npz` from MODE=mega.

## Honest expectations
The most likely outcomes: (a) Bsurprise or Hpre16 already explains the flip (NEGATIVE/INCONCLUSIVE) - a clean, reportable "the signal is not new"; (b) a small positive over everything (+0.02-0.05 R2): real but modest; (c) a clear positive. I would put (c) below one in four.
Known limits: one model; edits are +1 / +10% on numeric results only; `flip` conflates "followed the error" and "number was decorative" (reuse is the oracle control: 96% of results are reused later); the step thread's decode may differ slightly from the original batch generation (noise floor reported).
