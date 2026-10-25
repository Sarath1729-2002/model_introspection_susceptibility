# Reading hint-adoption off a model's hidden states - proposal and direction

**One line.** Train linear probes on a frozen LLM to predict, *before* it writes anything, whether it will adopt an answer someone hands it - and keep monitoring that signal while it reasons.
The study is built so that a positive result cannot be explained away by difficulty, number-copying, text-visible cues or one dataset's quirks, and so that a negative result is still informative.
(Probing and monitoring only. Steering is deliberately left for the follow-up this study is designed to justify.)

## Why this is mainline
- **Monitoring is the live safety use of probes.** Linear monitors for defection, misaligned personas and reward hacking are being pursued by several labs; the recurring weakness is *out-of-distribution reliability*
  (task-specific truthfulness probes do not transfer; leave-one-dataset-out costs 8-16 AUC points in the literature I checked). Our design bakes in leave-dataset-out and difficulty-shift tests from the start.
- **Sycophancy / deferring to unverified claims is a concrete, measurable failure** with objective ground truth (the planted answer is wrong by construction), which most "deception" probe work lacks.
- **It sits on the CoT-monitorability question**: if a model adopts a hint *without ever writing it down*, a text monitor is blind; an activation monitor might not be. The "silent adoption" analysis tests exactly that.

## What I already know (pilot, GSM8K, 4,016 problems, 7 framings, 16 samples each)
- Behaviour: adoption is strongly framing-dependent (authority cues > social pressure); the model is not merely copying numbers.
- A probe for the *forced-answer* belief gap (no chain of thought) works in all seven framings; a probe for *written-solution* adoption is modest and fails under easy->hard shift;
  the latent "susceptibility from the clean prompt" probe is real but small (R2 ~ 0.18 iid) and does not survive shift. So the honest state is: **promising but fragile, on one dataset.**
- Steering along a probe direction moves adoption, but random directions also move it, so those results are not specific. Hence: no steering until the probes are characterised.

## What this study adds (and why each piece is there)
| addition | answers |
|---|---|
| 5 audited problem sets (GSM8K, MATH-int, synthetic arithmetic, synthetic code, ARC) | is it one dataset's artifact? (leave-dataset-out) - and a deterministic-ground-truth audit that rejects datasets that fail it |
| **placebo** prompt (wrong value shown as a Problem ID) | is "susceptibility" just copying a salient number? |
| **correct-hint** prompt | is it a general "trust the hint" factor (credulity) or specific to being wrong? Tests whether one direction serves both |
| randomised offset / option position; generator-set difficulty | design-based dose-response: causal effects of *prompt* features, no activation intervention |
| difficulty-adjusted contrast (strata of dataset x clean accuracy) | does the probe read more than "this problem is hard"? |
| forced (no-CoT) belief labels next to sampled behaviour | separates what the model believes from what it writes; mediation-style decomposition |
| trace monitoring at 5 points + silent adoption | can we flag a trajectory *before* the wrong answer is written, and where text monitors are blind? |
| monitoring operating points (recall of adopters at 5-50% flag budgets) | the deployment-relevant number, vs observable and oracle baselines |
| probe geometry (cosine between susceptibility / difficulty / credulity / copying directions) | picks the direction(s) to steer later, and predicts which controls will matter |

## How to read the outcome (decision tree)
1. **H1-H4 pass broadly (pooled, >=3 datasets, shift, difficulty-adjusted):** a genuine pre-generation susceptibility signal. Next: second model family + scale (Qwen2.5 1.5B/7B/32B, Llama-3.1-8B) to claim generality; then the steering/patching study with the geometry result choosing the direction and difficulty/random/credulity directions as controls.
2. **Passes on belief (forced gap) but not on written behaviour / not across datasets:** the signal is real but task-bound - a useful negative-with-controls result about monitor reliability; still reportable, and points to training the probe on mixed-dataset data.
3. **Trace monitoring beats text and silent-adoption AUROC > 0.5:** the strongest positive: an activation monitor that sees what a CoT monitor cannot. Worth a short paper on its own.
4. **Nothing beats the observable baselines:** then "susceptibility" is mostly difficulty/uncertainty readable from text; we stop probing for it and say so. (The controls are what make this a conclusion rather than a shrug.)

## Honest novelty and demand
- Probing for sycophancy and hint-influence exists (to my knowledge from abstracts; Anthropic's work on hint-use faithfulness in reasoning models is the closest I recall - **not re-verified here**). I am not claiming a new idea; the contribution would be
  the *control design* (placebo, correct-hint, difficulty-adjusted, leave-dataset-out, silent adoption), the deterministic multi-dataset audit, and the trace-monitoring view.
- Demand is real (monitor reliability under distribution shift is an open complaint) but the field moves fast; a clean, well-controlled negative or mixed result with released code is still publishable as a workshop/short paper, a clean positive is a stronger paper.
- Risk: R2 for sampled labels is capped by sampling noise (reported); the forced-gap labels are the clean ones. One model only until replication.

## What it costs
~9-10 h on one TamIA node (4xH100): states 1 h, sampling 6-7 h, trace 0.5 h, analysis ~1-1.5 h. Everything resumable; a time budget stops sampling cleanly. Output is a single `REPORT.md` with generated headline bullets, figures, and CSVs.

## Asks for the mentor
1. Is the placebo + correct-hint + difficulty-adjusted control set the right bar, or is there a control you would insist on?
2. Priority after this: second model family vs the steering study vs a trained (mixed-dataset) monitor?
3. Is "silent adoption" the angle to lead with if it works?
