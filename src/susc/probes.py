"""The probing experiments.  Each block takes the wide table + state arrays and returns (results, curves); all through core.run_targets.

  A  prospective readout   CLEAN prompt-end state (no hint yet)  ->  how susceptible will the model be?  (+ knowledge-boundary labels)
  B  in-context readout    HINTED prompt-end state               ->  will it follow this hint?  per framing, + placebo + correct-hint prompts
  C  delta-state           hinted minus clean state of the same problem
  D  leave-one-framing-out train on three framings (pooled rows), test on the fourth
  E  geometry / specificity  are the susceptibility, difficulty, credulity and copying probes the same direction?
Scopes: every dataset separately (iid + difficulty-shift), pooled (iid + shift + leave-one-dataset-out).
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from . import core, ridge
from .prompts import FRAMINGS
from .table import ladder

ADOPT = 0.25
TA = ["S", "Sgap", "S_net", "Sgap_net", "uplift", "cred", "p_clean", "ent_clean", "lpg_clean"]
EXCL_A = {t: ["Bdiff(oracle)"] for t in ("uplift", "cred", "p_clean", "ent_clean", "lpg_clean")}


def coarse_layers(L: int) -> List[int]:
    ls = [l for l in range(4, L, 2)]
    if (L - 1) not in ls and L - 1 >= 1:
        ls.append(L - 1)
    return ls or list(range(1, L))


def other_rate(df: pd.DataFrame, f: str) -> np.ndarray:
    return df[[f"follow_{g}" for g in FRAMINGS if g != f]].mean(axis=1).values.astype(np.float32)[:, None]


def add_auroc(results):
    for r in results:
        if r["target"].startswith("follow") or r["target"] == "S":
            y, p = r["_y"], r["_pred"]
            rate = float((y >= ADOPT).mean())
            r["base_rate_adopt"] = rate
            r["auroc_adopt"] = core.auroc(y >= ADOPT, p) if 0.01 < rate < 0.99 else float("nan")
            po = r["_pred_obs"]
            r["auroc_adopt_obs_baseline"] = core.auroc(y >= ADOPT, po) if 0.01 < rate < 0.99 else float("nan")
    return results


def _scopes(df, datasets, min_rows=120):
    for ds in datasets:
        m = (df.ds == ds).values
        if m.sum() >= min_rows:
            yield ds, m


def _run(df, sub, cond, targets, datasets, layers, base_fn, prefix, excl=None, pooled_regimes=None, ds_regimes=("iid", "shift"), ci_for=(), log=print):
    res_all, cur_all = [], []
    for ds, m in _scopes(df, datasets):
        d, s = core.take(df, sub, m)
        res, cur = core.run_targets(d, s, cond, cond, targets, targets, ds_regimes, base_fn(d, s), layers, f"{prefix}[{ds}]", excl, ci_for=ci_for)
        res_all += res
        cur_all.append(cur.assign(scope=ds))
        log(f"  {prefix}[{ds}] done")
    if pooled_regimes:
        res, cur = core.run_targets(df, sub, cond, cond, targets, targets, pooled_regimes, base_fn(df, sub), layers, f"{prefix}[pooled]", excl, ci_for=ci_for)
        res_all += res
        cur_all.append(cur.assign(scope="pooled"))
        log(f"  {prefix}[pooled] done")
    return add_auroc(res_all), (pd.concat([c for c in cur_all if len(c)]) if cur_all else pd.DataFrame())


def pooled_regs(datasets, lodo=True):
    return ["iid", "shift"] + ([f"lodo:{ds}" for ds in datasets] if lodo else [])


# ----------------------------------------------------------------------------- A
def block_A(df, sub, datasets, layers, log=print):
    return _run(df, sub, "clean", TA, datasets, layers, lambda d, s: ladder(d, s, "clean", False), "A", EXCL_A, pooled_regs(datasets), ci_for=("S",), log=log)


# ----------------------------------------------------------------------------- B
def block_B(df, sub, datasets, layers, framings=FRAMINGS, lodo_for=("auth", "pressure"), log=print):
    res_all, cur_all = [], []
    for f in framings:
        tg = [f"follow_{f}", f"gap_{f}"]
        oth = lambda d: other_rate(d, f)                                                   # noqa: E731
        res, cur = _run(df, sub, f, tg, datasets, layers, lambda d, s: ladder(d, s, f, True, other=oth(d)), f"B[{f}]", None,
                        pooled_regs(datasets, lodo=f in lodo_for), log=log)
        for r in res:
            r["framing"] = f
        res_all += res
        cur_all.append(cur.assign(framing=f))
    return res_all, pd.concat(cur_all)


def block_B_controls(df, sub, datasets, layers, log=print):
    """placebo prompt (wrong value as an ID): does the state predict copying?   correct-hint prompt: does it predict credulity?"""
    res1, cur1 = _run(df, sub, "placebo", ["follow_plc", "gap_plc"], datasets, layers, lambda d, s: ladder(d, s, "placebo", True), "B[placebo]", None,
                      pooled_regs(datasets, lodo=False), log=log)
    res2, cur2 = _run(df, sub, "cor", ["uplift", "acc_cor"], datasets, layers, lambda d, s: ladder(d, s, "cor", True, shown="gold"), "B[cor]",
                      {"uplift": ["Bdiff(oracle)", "Bhint+gold+difficulty(oracle)"]}, pooled_regs(datasets, lodo=False), log=log)
    for r in res1:
        r["framing"] = "placebo"
    for r in res2:
        r["framing"] = "cor"
    return res1 + res2, pd.concat([cur1.assign(framing="placebo"), cur2.assign(framing="cor")])


# ----------------------------------------------------------------------------- C
def block_C(df, sub, datasets, layers, framings=("auth", "pressure"), log=print):
    res_all, cur_all = [], []
    for f in framings:
        name = f"delta_{f}"
        sub2 = dict(sub)
        sub2[name] = (sub[f].astype(np.float32) - sub["clean"].astype(np.float32)).astype(np.float16)
        tg = [f"follow_{f}", f"gap_{f}"]
        res, cur = _run(df, sub2, name, tg, datasets, layers, lambda d, s: ladder(d, s, f, True, other=other_rate(d, f), early=()), f"C[{f}]", None,
                        ["iid", "shift"], log=log)
        for r in res:
            r["framing"] = f
        res_all += res
        cur_all.append(cur.assign(framing=f))
    return res_all, pd.concat(cur_all)


# ----------------------------------------------------------------------------- D
def block_D(df, sub, datasets, layers, framings=FRAMINGS, log=print):
    """train on the hinted states of the other framings (rows pooled), test on the held-out framing, on held-out problems."""
    rows = []
    scopes = [(ds, m) for ds, m in _scopes(df, datasets)] + [("pooled", np.ones(len(df), dtype=bool))]
    for scope, m in scopes:
        d, s = core.take(df, sub, m)
        fit, val, test = core.make_splits(d, "iid")
        if len(fit) < core.MIN_ROWS or len(test) < core.MIN_ROWS:
            continue
        bases = {g: ladder(d, s, g, True, other=other_rate(d, g)) for g in framings}
        for target in ("follow", "gap"):
            for f in framings:
                tr = [g for g in framings if g != f]
                pool = lambda idx, col: np.concatenate([d[f"{col}_{g}"].values[idx] for g in tr])[:, None]        # noqa: E731
                yfit, yval, ytest = pool(fit, target), pool(val, target), d[f"{target}_{f}"].values[test][:, None]
                best = None
                for l in layers:
                    Xg = lambda g: s[g][:, l, :].astype(np.float32)                                              # noqa: E731
                    Xf = np.concatenate([Xg(g)[fit] for g in tr]); Xv = np.concatenate([Xg(g)[val] for g in tr])   # noqa: E702
                    r = core.ridge_eval(Xf, Xv, Xg(f)[test], yfit, yval, ytest)[0]
                    if best is None or (not np.isnan(r["val_r2"]) and r["val_r2"] > best["val_r2"]):
                        best = {**r, "layer": l}
                names = [n for n in bases[f] if n in ("Bhint(visible)", "H8(early-layer)", "Bother(oracle)", "Bhint+gold+difficulty(oracle)")]
                bl = {}
                for n in names:
                    Xf = np.concatenate([bases[g][n][fit] for g in tr]); Xv = np.concatenate([bases[g][n][val] for g in tr])   # noqa: E702
                    bl[n] = core.ridge_eval(Xf, Xv, bases[f][n][test], yfit, yval, ytest)[0]
                y = ytest[:, 0]
                key = lambda k: -1e9 if np.isnan(bl[k]["test_r2"]) else bl[k]["test_r2"]                          # noqa: E731
                obs = [k for k in bl if "oracle" not in k]
                bn_o = max(obs, key=key)
                bn_a = max(bl, key=key)
                do, lo_o, hi_o = core.boot_delta(y, best["pred"], bl[bn_o]["pred"], B=500)
                da, lo_a, hi_a = core.boot_delta(y, best["pred"], bl[bn_a]["pred"], B=500)
                rows.append({"scope": scope, "target": target, "held_out": f, "probe_layer": best["layer"], "probe_r2": core.r2(y, best["pred"]),
                             "spearman": core.spearman(y, best["pred"]), "mean_label": float(y.mean()), "best_observable": bn_o, "obs_r2": bl[bn_o]["test_r2"],
                             "delta_vs_obs": do, "obs_lo": lo_o, "obs_hi": hi_o, "best_baseline": bn_a, "oracle_r2": bl[bn_a]["test_r2"],
                             "delta_vs_oracle": da, "oracle_lo": lo_a, "oracle_hi": hi_a, "usable": bool(d[f"follow_{f}"].mean() >= 0.02)})
        log(f"  D[{scope}] done")
    out = pd.DataFrame(rows)
    if len(out):
        out["verdict_obs"] = [core.verdict(a, b, c) for a, b, c in zip(out.delta_vs_obs, out.obs_lo, out.probe_r2)]
    return out


# ----------------------------------------------------------------------------- E
GEO = ["S", "p_clean", "cred", "uplift", "S_net", "follow_plc", "gap_plc", "Sgap"]


def block_E(df, sub, layers, log=print):
    """Probes for several labels fitted at ONE layer (best validation layer for S), pooled over datasets.
    -> cosine similarity of the (raw-space) weight vectors and the cross-label rank correlation of their test predictions."""
    fit, val, test = core.make_splits(df, "iid")
    Y = df[GEO].values.astype(np.float64)
    X = lambda l: sub["clean"][:, l, :].astype(np.float32)                                        # noqa: E731
    best_l, best_v = layers[-1], -9
    for l in layers:
        r = core.ridge_eval(X(l)[fit], X(l)[val], X(l)[test], Y[fit][:, :1], Y[val][:, :1], Y[test][:, :1])[0]
        if not np.isnan(r["val_r2"]) and r["val_r2"] > best_v:
            best_l, best_v = l, r["val_r2"]
    Xl = core._t(X(best_l))
    fm = ridge.fit_layer(Xl[fit], Y[fit])
    W, P = [], []
    for t in range(len(GEO)):
        pv = fm.predict(Xl[val], t)
        ri = ridge.pick_ratio(pv, Y[val, t])
        w = (fm.W[t][ri] / fm.sd).cpu().numpy()
        W.append(w / max(np.linalg.norm(w), 1e-12))
        P.append(fm.predict(Xl[test], t)[:, ri].cpu().numpy())
    W = np.stack(W)
    cos = pd.DataFrame(W @ W.T, index=GEO, columns=GEO)
    sp = pd.DataFrame([[core.spearman(P[i], P[j]) for j in range(len(GEO))] for i in range(len(GEO))], index=GEO, columns=GEO)
    lab = pd.DataFrame([[core.spearman(Y[test][:, i], Y[test][:, j]) for j in range(len(GEO))] for i in range(len(GEO))], index=GEO, columns=GEO)
    cross = pd.DataFrame([[core.spearman(Y[test][:, j], P[i]) for j in range(len(GEO))] for i in range(len(GEO))], index=[f"probe[{g}]" for g in GEO],
                         columns=[f"label[{g}]" for g in GEO])
    return {"layer": int(best_l), "cosine": cos, "pred_spearman": sp, "label_spearman": lab, "probe_vs_label": cross}
