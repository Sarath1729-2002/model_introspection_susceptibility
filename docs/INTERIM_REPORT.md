# Interim report - Qwen2.5-7B-Instruct on GSM8K (the only cell finished so far)

Status: **interim**. One model, one dataset, injected errors. The matrix (MATH, DeepSeek-R1-Distill, natural errors, sampling ceiling, transfer) is built and queued by `cluster/matrix.sh`; those sections of `PAPER_REPORT.md` are what will turn this into a claim that is not about one model.

## Claim
A ridge probe on the hidden state at the last token of a wrong number the model has just written (a +1 or +10% edit of a number in its own greedy solution) predicts whether the final answer will follow the wrong number, and the probe direction causally shifts that behaviour.

## What was measured (31,918 injected rows, 2,323 problems; labels are exact: final answer after greedy continuation differs from the unedited one)
| predictor | iid R2 | shift R2 | iid AUROC | shift AUROC |
|---|---|---|---|---|
| best observable baseline (early layer, post-edit) | 0.506 | 0.461 | 0.905 | 0.890 |
| injection-time probe (best layer 20) | 0.615 | 0.590 | 0.947 | 0.940 |
| pre-edit natural state (carried forward) | 0.480 | - | - | - |

- Gain of the injection-time state over the pre-edit state (paired, problem-clustered): +0.135 R2 [0.119, 0.153].
- TPR at 10% FPR: 0.841 vs 0.764. Last-step AUROC 0.988 vs 0.917; non-last steps 0.936 vs 0.889.
- Layer profile (CV R2): L4 0.469, L8 0.497, L12 0.522, L16 0.569, L20 0.616, L24 0.603, L28 0.578.
- Robustness checks C1-C7 (drop last step, demeaned label, each edit variant, bag-of-words baselines, shuffled labels <= 0.02 R2, paired pre-edit comparison, AUROC gains in both regimes): all pass.

## Causal test (C8, steering at the injected token, 3000 rows, base follow rate 0.496)
- Slope per sd of the probe projection: 0.0048 [0.0044, 0.0053] (layer 12), 0.0050 (layer 20); the largest of three random directions about 0.002.
- Effect at +-16 sd: layer 12 +0.060 / -0.100, layer 20 +0.089 / -0.067; random directions +-0.02-0.04.
- Integrity (finished and parsable rates) within 0.004-0.008 of baseline. Accuracy moves almost exactly as minus the change in follow rate.

Verdict on this cell: ROBUST + CAUSAL under the pre-registered rules in `docs/ROBUST.md`.

## What this does not yet show
1. **One model, one dataset.** The matrix addresses this.
2. **Injected, not natural, errors.** The natural-slip test addresses this; its outcome is open.
3. **`flip` mixes "the chain recovered" with "this step was not load-bearing".** Last-step and within-problem checks reduce this; they do not remove it.
4. **The AUROC gain over the best observable baseline is modest (about +0.04-0.05 absolute)**; the R2 gain (about +0.11) is larger. Do not oversell the headline number.
5. **The mean-difference direction is as strong or stronger than the ridge direction** (slope 0.005-0.007), and random directions are not null at high dose (+-0.02-0.04). The causal claim is about a direction in this subspace, not about the ridge solution being special.
6. The paper's sampling-noise ceiling for this label is not measured yet (the greedy label is one draw); `stoch` measures it.

## How it maps to "How Much is Left?"
constant floor -> base rate; exact countdown -> pre-edit state carried forward; TF-IDF surface ridge -> hashed bag of words; shuffled controls -> global and within-problem; sampling floor -> `stoch`; transfer matrix -> `matrix`; steering with NLL and paired tests -> `istear`; retraction event analysis -> natural slips; prompt-end length probe -> `lenref` (sanity anchor).
