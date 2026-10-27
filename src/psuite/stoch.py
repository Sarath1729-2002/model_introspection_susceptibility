"""SAMPLING-NOISE CEILING for the injection probe (the paper's section A.13: how close is the probe to what ANY predictor could reach?).

The label of a row is the answer after ONE greedy continuation of the edited prefix.  The model's own follow probability p_i = P(final answer follows the error | prefix) is a property of the prefix; a predictor
that knew p_i exactly is the best any prompt-conditioned predictor can be.  We estimate that oracle by drawing n_samp continuations per row (T=0.7, same edited prefix as the steps / inject protocol):
  oracle_sampled   leave-one-out AUROC of p_hat_{-j}(i) against the j-th sampled outcome, pooled over (row, draw)   -> the ceiling for predicting a SAMPLED outcome (the paper's leave-one-out MAD analogue)
  oracle_greedy    AUROC of p_hat_i (all draws) against the GREEDY flip of the same row                             -> the ceiling for our actual label (a lower bound: p_hat itself is noisy)
and compare the probe and the best observable baseline on the same rows: 'closable range captured' = (AUROC_probe - 0.5) / (AUROC_oracle - 0.5) and the same above the best observable.
Resumable per (shard, chunk): <out>/k<shard>_<chunk>.jsonl.
"""
import json
import zlib
from pathlib import Path
from typing import Dict, List

import numpy as np

from .hf_util import generate_ids
from .prompts import chat_prompt, plain_text
from .sched import Budget
from .textutil import parse_answer


def select(irows: List[dict], splits: Dict[int, str], n: int) -> List[dict]:
    """deterministic sample (crc32) of at most n injected rows of TEST-split problems."""
    c = [r for r in irows if splits.get(r["id"]) == "test" and r.get("fin", True)]
    c.sort(key=lambda r: zlib.crc32(f"stoch{r['id']}_{r['k']}_{r['var']}".encode()))
    return c[:n] if n else c


def run(problems, irows, model, tok, out_dir: Path, shard=0, nshards=1, n_rows=1000, n_samp=8, temperature=0.7, top_p=0.95, max_new=512, max_seqs=96, chunk=16, budget_s=0, log=print) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = {p["id"]: p for p in problems}
    rows = [r for r in select(irows, {p["id"]: p["split"] for p in problems}, n_rows) if r["id"] % nshards == shard]
    bud = Budget(budget_s)
    done = 0
    for c in range(0, len(rows), chunk):
        f = out_dir / f"k{shard:02d}_{c // chunk:04d}.jsonl"
        if f.exists():
            continue
        if bud.spent():
            log(f"[stoch shard {shard}] time budget reached - stopping cleanly (resubmit to continue)")
            break
        batch = rows[c:c + chunk]
        res = generate_ids(model, tok, [chat_prompt(tok, plain_text(probs[r["id"]])) + r["ptext"] for r in batch], max_new, n_samp, temperature, top_p, max_seqs)
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r, out in zip(batch, res):
                flips, fins = [], []
                for ids, fin in out:
                    ans = parse_answer(r["ptext"] + tok.decode(ids, skip_special_tokens=True), probs[r["id"]]["kind"])
                    flips.append(float(ans != r["ans_orig"]))
                    fins.append(bool(fin))
                fh.write(json.dumps({"id": r["id"], "k": r["k"], "var": r["var"], "greedy_flip": r["flip"], "flips": flips, "fins": fins}) + "\n")
        tmp.rename(f)
        done += len(batch)
        log(f"[stoch shard {shard}] chunk {c // chunk}: {len(batch)} rows x {n_samp} draws ({done} rows this run)")
    return done


def load(out_dir: Path) -> List[dict]:
    rows = []
    for f in sorted(Path(out_dir).glob("k*_*.jsonl")):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def pooled_loo_auroc(flip_mat: np.ndarray, valid: np.ndarray) -> float:
    """flip_mat [n, S] in {0,1}, valid [n, S] bool.  Score of draw j of row i = mean of the OTHER valid draws of row i; pooled AUROC."""
    from .robust import auroc
    s, y = [], []
    for fm, v in zip(flip_mat, valid):
        idx = np.where(v)[0]
        if len(idx) < 3:
            continue
        tot = fm[idx].sum()
        for j in idx:
            s.append((tot - fm[j]) / (len(idx) - 1))
            y.append(fm[j])
    return auroc(np.array(y), np.array(s)) if y else float("nan")


def analyze(run_dir: Path, mega_dir: Path, out_dir: Path, ds: str = "gsm8k", B: int = 300, log=print) -> dict:
    from .inject import prepare
    from .robust import auroc, boot_auroc_delta, regime_fit, expanded_baselines
    rows = load(out_dir)
    if len(rows) < 100:
        return {"status": "n/a", "why": f"only {len(rows)} rows sampled"}
    P, why = prepare(run_dir, mega_dir, ds)
    if P is None:
        return {"status": "n/a", "why": why}
    S = min(len(r["flips"]) for r in rows)
    F = np.array([r["flips"][:S] for r in rows], dtype=float)
    V = np.array([r["fins"][:S] for r in rows], dtype=bool)
    phat = np.array([F[i][V[i]].mean() if V[i].sum() >= 2 else np.nan for i in range(len(rows))])
    df = P["df"]
    key = {(i, k, v): n for n, (i, k, v) in enumerate(zip(df.id, df.k, df["var"]))}
    f = regime_fit(P, expanded_baselines(P), "iid")
    te_pos = {int(t): n for n, t in enumerate(f["test"])}
    sel = [(n, key[(r["id"], r["k"], r["var"])]) for n, r in enumerate(rows) if (r["id"], r["k"], r["var"]) in key and key[(r["id"], r["k"], r["var"])] in te_pos and np.isfinite(phat[n])]
    if len(sel) < 100:
        return {"status": "n/a", "why": f"only {len(sel)} sampled rows are in the iid test split"}
    ni = np.array([a for a, _ in sel])
    tp = np.array([te_pos[b] for _, b in sel])
    yg = np.array([rows[a]["greedy_flip"] for a in ni], dtype=float)
    grp = np.array([rows[a]["id"] for a in ni])
    p_probe, p_obs = f["p"][tp], f["q"][tp]
    au = {"probe": auroc(yg, p_probe), "best_observable": auroc(yg, p_obs), "oracle_greedy": auroc(yg, phat[ni]), "oracle_sampled_loo": pooled_loo_auroc(F[ni], V[ni])}
    d_po = boot_auroc_delta(yg, p_probe, p_obs, grp, B)
    d_og = boot_auroc_delta(yg, phat[ni], p_probe, grp, B)
    out = {"status": "ok", "ds": ds, "n_rows": int(len(ni)), "n_problems": int(len(np.unique(grp))), "n_samp": int(S), "best_observable": f["best"], "auroc": au,
           "share_stochastic_rows": float(((phat[ni] > 0) & (phat[ni] < 1)).mean()), "mean_abs_greedy_vs_phat": float(np.abs(yg - phat[ni]).mean()),
           "captured_vs_floor": float((au["probe"] - 0.5) / (au["oracle_greedy"] - 0.5)) if au["oracle_greedy"] > 0.5 else float("nan"),
           "captured_vs_floor_sampled": float((au["probe"] - 0.5) / (au["oracle_sampled_loo"] - 0.5)) if au["oracle_sampled_loo"] > 0.5 else float("nan"),
           "probe_minus_observable": {"delta": d_po[0], "lo": d_po[1], "hi": d_po[2]}, "oracle_minus_probe": {"delta": d_og[0], "lo": d_og[1], "hi": d_og[2]}}
    log(f"  stoch: probe {au['probe']:.3f} / observable {au['best_observable']:.3f} / oracle(greedy) {au['oracle_greedy']:.3f} / oracle(sampled, LOO) {au['oracle_sampled_loo']:.3f}")
    return out
