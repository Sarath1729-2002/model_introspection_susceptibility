"""ROBUSTNESS of the headline result  I[num]:flip  (probe on the state at the last token of an injected wrong number predicts whether the final answer follows the error)  + the scorecard C1-C8.
Offline part (no generation): every check re-uses the corrected protocol of cv_eval (grouped K-fold selection of layer and lambda over the training pool, problem-clustered bootstrap, iid and easy->hard shift)
against an EXPANDED set of observable baselines (the inject baselines + hashed bag-of-words of the question and of the edited prefix).
  C1  drop the last numeric step of every problem (it flips ~85%: 'position' could carry the result)
  C2  label demeaned within problem (removes all between-problem variance)
  C3  each edit variant alone (n1: c->c+1, n2: c->c+max(10,c//10))
  C4  all rows against the bag-of-words baselines
  C5  shuffled-label control (the whole pipeline on permuted labels must give R2 ~ 0)
  C6  paired: injection-time state vs the natural pre-edit state of the same step (the 'did the injection add anything' question), clustered bootstrap
  C7  AUROC (not only R2): probe vs best observable, clustered CI, iid and shift
  C8  causal: steering along the probe direction at the injected token moves follow-vs-recover (istear.py)
Everything is pre-registered in docs/ROBUST.md.  Overall: ROBUST + CAUSAL (C1-C8), ROBUST (diagnostic) (C1-C7), else FRAGILE with the failed checks listed.
"""
import json
import zlib
from pathlib import Path
from typing import Dict, List

import numpy as np

SHUF_MAX = 0.02
HASH_DIM = 256


def hashed_bow(texts, dim: int = HASH_DIM) -> np.ndarray:
    """log1p counts of hashed lower-case word tokens (no vocabulary, no leakage between rows), l2-free so ridge standardises it."""
    import re
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+|\d+", t.lower()):
            X[i, zlib.crc32(w.encode()) % dim] += 1.0
    return np.log1p(X)


def auroc(y, s) -> float:
    import pandas as pd
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = ~np.isnan(y) & ~np.isnan(s)
    y, s = y[m], s[m]
    n1, n0 = int((y >= 0.5).sum()), int((y < 0.5).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().values
    return float((r[y >= 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tpr_at_fpr(y, s, fpr: float = 0.10) -> float:
    y, s = np.asarray(y, float), np.asarray(s, float)
    neg = s[y < 0.5]
    if (y >= 0.5).sum() == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(neg, 1 - fpr)
    return float((s[y >= 0.5] > thr).mean())


def boot_auroc_delta(y, pa, pb, groups, B: int = 500, seed: int = 0):
    """paired cluster bootstrap of AUROC(a) - AUROC(b).  -> (delta, lo95, hi95)"""
    y, pa, pb, groups = (np.asarray(x) for x in (y, pa, pb, groups))
    ug, inv = np.unique(groups, return_inverse=True)
    rows = [np.where(inv == i)[0] for i in range(len(ug))]
    rng = np.random.RandomState(seed)
    d = np.empty(B)
    for b in range(B):
        ix = np.concatenate([rows[i] for i in rng.randint(0, len(ug), len(ug))])
        d[b] = auroc(y[ix], pa[ix]) - auroc(y[ix], pb[ix])
    return auroc(y, pa) - auroc(y, pb), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def _sub(P, mask, baselines):
    """restrict states / frame / baselines to the rows in mask."""
    return P["df"][mask].reset_index(drop=True), P["HA"][mask], {k: v[mask] for k, v in baselines.items()}


def expanded_baselines(P) -> Dict[str, np.ndarray]:
    bl = dict(P["bl"])
    qb = hashed_bow(P["q"])
    pb = hashed_bow(list(P["df"].ptext))
    bl["Bbow-question"] = qb
    bl["Bbow-prefix"] = pb
    bl["Bbow+obs"] = np.concatenate([P["obs"], qb, pb], 1)
    return bl


def _block(df, HA, ly, bl, y, name, prefix, log, B):
    from . import cv_eval
    return cv_eval.run_block(df.split.values, df.d1.values, df.id.values, lambda li: HA[:, li, :].astype(np.float32), list(range(len(ly))), bl, y, [name], prefix=prefix, log=log, B=B)


def subset_checks(P, log=print, B: int = 500) -> dict:
    """C1-C4 (and the label-sd sanity).  -> {'rows': [result dicts], 'status': {check: dict}}"""
    from . import mega
    df, ly = P["df"], P["ly"]
    bl = expanded_baselines(P)
    kmax = df.groupby("id").k.transform("max").values
    flip = df.flip.values.astype(np.float64)
    dm = flip - df.assign(f=flip).groupby("id").f.transform("mean").values
    cnt = df.groupby("id").id.transform("size").values
    specs = {
        "C4": ("R[all+bow]", np.ones(len(df), bool), flip),
        "C1": ("R[no-last-step]", df.k.values != kmax, flip),
        "C2": ("R[demeaned]", cnt > 2, dm),
        "C3a": ("R[n1-only]", (df["var"] == "n1").values, flip),
        "C3b": ("R[n2-only]", (df["var"] == "n2").values, flip),
    }
    rows, status = [], {}
    for chk, (pfx, mask, y) in specs.items():
        d, H, b = _sub(P, mask, bl)
        yy = y[mask][:, None]
        if len(d) < 300 or d.id.nunique() < 40:
            status[chk] = {"status": "n/a", "why": f"only {len(d)} rows / {d.id.nunique()} problems"}
            continue
        sd = float(np.nanstd(yy))
        res = _block(d, H, ly, b, yy, "flip", pfx, log, B)
        for r in res:
            r["probe_layer"] = ly[r["probe_layer"]]
        rows += res
        st = mega.generic_status(mega.rows_df(res), pfx, "flip", "" if sd >= 0.1 or chk == "C2" else f"label std {sd:.3f} < 0.1")
        st["n_rows"], st["n_problems"], st["label_sd"] = int(len(d)), int(d.id.nunique()), sd
        status[chk] = st
    status["C3"] = _both(status.get("C3a"), status.get("C3b"))
    return {"rows": rows, "status": status}


def _both(a, b) -> dict:
    if not a or not b or a["status"] == "n/a" or b["status"] == "n/a":
        return {"status": "n/a", "why": "a variant has too few rows"}
    ok = a["status"] == b["status"] == "SUPPORTED"
    return {"status": "SUPPORTED" if ok else (a["status"] if a["status"] != "SUPPORTED" else b["status"]), "why": f"n1: {a['status']}, n2: {b['status']}"}


def shuffled_control(P, seeds=(0, 1, 2)) -> dict:
    """the whole pipeline (layer + lambda by grouped CV, fit on the pool, evaluate on test) on label vectors permuted over rows: R2 must be ~0, otherwise the pipeline fits noise."""
    from . import cv_eval
    from .core import r2
    df, HA, ly = P["df"], P["HA"], P["ly"]
    pool, test = cv_eval.split_indices(df.split.values, df.d1.values, "iid")
    out = []
    for s in seeds:
        y = np.random.RandomState(100 + s).permutation(df.flip.values.astype(np.float64))[:, None]
        res = cv_eval.cv_probe(lambda li: HA[:, li, :].astype(np.float32), list(range(len(ly))), y, pool, test, df.id.values)[0]
        out.append(float(r2(y[test, 0], res["pred"])))
    return {"r2_by_seed": out, "max": float(max(out)), "ok": bool(max(out) <= SHUF_MAX)}


def curves(P) -> dict:
    """CV R2 by layer (best lambda) for the injection-time state, the end-of-line state and the natural pre-edit state, iid."""
    from . import cv_eval
    df, HA, HB, ly, sly, pre = P["df"], P["HA"], P["HB"], P["ly"], P["sly"], P["pre"]
    pool, test = cv_eval.split_indices(df.split.values, df.d1.values, "iid")
    y = df.flip.values.astype(np.float64)[:, None]
    out = {}
    for nm, H in (("I[num]", HA), ("I[line]", HB)):
        r = cv_eval.cv_probe(lambda li, H=H: H[:, li, :].astype(np.float32), list(range(len(ly))), y, pool, test, df.id.values)[0]
        out[nm] = {int(ly[i]): v for i, v in r["curve"].items()}
    pl = [l for l in sly if l % 4 == 0 or l == sly[-1]]
    r = cv_eval.cv_probe(lambda li: pre(pl[li]), list(range(len(pl))), y, pool, test, df.id.values)[0]
    out["Pre[nat]"] = {int(pl[i]): v for i, v in r["curve"].items()}
    return out


def diagnostics(P, bl, B: int = 500, log=print) -> dict:
    """one fit per regime with test predictions: R2, AUROC, TPR@10%FPR of the probe and of the best observable baseline, AUROC difference CI, by-step-index and last-vs-other strata, and the paired
    I[num] vs Pre[nat] comparison.  -> {'regimes': {regime: {...}}, 'strata': [...], 'paired': {...}}"""
    from . import cv_eval
    from .core import r2
    df, HA, ly, sly, pre = P["df"], P["HA"], P["ly"], P["sly"], P["pre"]
    y1 = df.flip.values.astype(np.float64)
    Y = y1[:, None]
    g = df.id.values
    pl = [l for l in sly if l % 4 == 0 or l == sly[-1]]
    kmax = df.groupby("id").k.transform("max").values
    out, strata, paired = {}, [], {}
    for rg in ("iid", "shift"):
        pool, test = cv_eval.split_indices(df.split.values, df.d1.values, rg)
        if len(pool) < 60 or len(test) < 30:
            continue
        pr = cv_eval.cv_probe(lambda li: HA[:, li, :].astype(np.float32), list(range(len(ly))), Y, pool, test, g)[0]
        bp = {n: cv_eval.cv_probe(lambda _l, F=F: F, [0], Y, pool, test, g)[0] for n, F in bl.items() if "oracle" not in n}
        yt = y1[test]
        b2 = {n: r2(yt, v["pred"]) for n, v in bp.items()}
        best = max(b2, key=lambda n: -1e9 if np.isnan(b2[n]) else b2[n])
        p, q = pr["pred"], bp[best]["pred"]
        d, lo, hi = boot_auroc_delta(yt, p, q, g[test], B)
        pre_r = cv_eval.cv_probe(lambda li: pre(pl[li]), list(range(len(pl))), Y, pool, test, g)[0]
        dr, lor, hir = cv_eval.boot_delta_cluster(yt, p, pre_r["pred"], g[test], B)
        da, loa, hia = boot_auroc_delta(yt, p, pre_r["pred"], g[test], B)
        out[rg] = {"n_test": int(len(test)), "n_test_groups": int(len(np.unique(g[test]))), "probe_layer": int(ly[pr["layer"]]), "probe_r2": float(r2(yt, p)), "probe_auroc": auroc(yt, p),
                   "probe_tpr10": tpr_at_fpr(yt, p), "best_observable": best, "obs_r2": float(b2[best]), "obs_auroc": auroc(yt, q), "obs_tpr10": tpr_at_fpr(yt, q), "auroc_delta": d, "auroc_lo": lo, "auroc_hi": hi,
                   "pre_r2": float(r2(yt, pre_r["pred"])), "pre_auroc": auroc(yt, pre_r["pred"]), "r2_delta_vs_pre": dr, "r2_delta_vs_pre_lo": lor, "r2_delta_vs_pre_hi": hir,
                   "auroc_delta_vs_pre": da, "auroc_delta_vs_pre_lo": loa, "auroc_delta_vs_pre_hi": hia}
        if rg == "iid":
            kt, last = df.k.values[test], (df.k.values == kmax)[test]
            for nm, m in [("last step", last), ("not last step", ~last)] + [(f"step k={k}", kt == k) for k in sorted(set(kt))[:6]]:
                if m.sum() >= 30 and len(set(yt[m] >= 0.5)) == 2:
                    strata.append({"stratum": nm, "n": int(m.sum()), "flip_rate": float(yt[m].mean()), "probe_auroc": auroc(yt[m], p[m]), "obs_auroc": auroc(yt[m], q[m]), "pre_auroc": auroc(yt[m], pre_r["pred"][m])})
        log(f"  diagnostics {rg}: probe AUROC {out[rg]['probe_auroc']:.3f} vs {best} {out[rg]['obs_auroc']:.3f} (delta {d:+.3f} [{lo:+.3f},{hi:+.3f}])")
    return {"regimes": out, "strata": strata}


# ----------------------------------------------------------------------------- the 'How Much is Left?' evaluation vocabulary, applied to error propagation
def regime_fit(P, bl, rg: str = "iid", y_col: str = "flip", seed: int = 0):
    """one CV-selected probe fit and every OBSERVABLE baseline for a regime.  -> dict(pool, test, y, p (probe test pred), q (best observable test pred), best, layer, bp {name: pred}) or None."""
    from . import cv_eval
    from .core import r2
    df, HA, ly = P["df"], P["HA"], P["ly"]
    pool, test = cv_eval.split_indices(df.split.values, df.d1.values, rg)
    if len(pool) < 60 or len(test) < 30:
        return None
    y = df[y_col].values.astype(np.float64)
    Y, g = y[:, None], df.id.values
    pr = cv_eval.cv_probe(lambda li: HA[:, li, :].astype(np.float32), list(range(len(ly))), Y, pool, test, g, 5, seed)[0]
    bp = {n: cv_eval.cv_probe(lambda _l, F=F: F, [0], Y, pool, test, g, 5, seed)[0]["pred"] for n, F in bl.items() if "oracle" not in n}
    b2 = {n: r2(y[test], v) for n, v in bp.items()}
    best = max(b2, key=lambda n: -1e9 if np.isnan(b2[n]) else b2[n])
    return {"pool": pool, "test": test, "y": y, "p": pr["pred"], "q": bp[best], "best": best, "layer": int(ly[pr["layer"]]), "bp": bp, "cv_r2": pr["cv_r2"]}


def brier_skill(y, p, base) -> float:
    p = np.clip(np.asarray(p, float), 0, 1)
    y = np.asarray(y, float)
    return float(1 - ((p - y) ** 2).mean() / ((base - y) ** 2).mean())


def mae_ratio(y, p, med) -> float:
    """MAE of p over MAE of the constant-median predictor (the paper's floor-normalised error; < 1 beats the floor)."""
    y = np.asarray(y, float)
    return float(np.abs(np.asarray(p, float) - y).mean() / np.abs(med - y).mean())


def paper_metrics(P, bl, log=print, seeds=(0, 1, 2)) -> dict:
    """the paper's evaluation vocabulary for the headline label:
      floor            constant predictor (median of the pool label; base rate for Brier)
      ladder           floor < position-only < surface text (+ hashed bag-of-words) < early layer < pre-edit state carried forward ('exact countdown' analogue) < injection-time probe
      controls         shuffled labels (global; within problem) through the whole pipeline
      seeds            CV fold seeds 0,1,2: mean +- sd of R2 / AUROC
      prop             the continuous label log10(1+|final - original final|): MAE and MAE / constant-median MAE
    -> {'ladder': [rows], 'controls': [...], 'seeds': [...], 'prop': [...]}"""
    from . import cv_eval
    from .core import r2
    df, HA, ly, sly, pre = P["df"], P["HA"], P["ly"], P["sly"], P["pre"]
    pl = [l for l in sly if l % 4 == 0 or l == sly[-1]]
    g = df.id.values
    out = {"ladder": [], "controls": [], "seeds": [], "prop": []}
    for rg in ("iid", "shift"):
        f = regime_fit(P, bl, rg)
        if f is None:
            continue
        pool, test, y = f["pool"], f["test"], f["y"]
        yt = y[test]
        base = float(y[pool].mean())
        med = float(np.median(y[pool]))
        pre_r = cv_eval.cv_probe(lambda li: pre(pl[li]), list(range(len(pl))), y[:, None], pool, test, g)[0]["pred"]
        rungs = [("floor: constant", np.full(len(test), base))] + [(n, f["bp"][k]) for n, k in (("position only", "Bpos"), ("surface text (+bag-of-words)", "Bbow+obs"), ("early layer (L8, post-injection)", "Hpost8(early-layer)"),
                                                                                                     ("model's own surprise + edit", "Bsurprise")) if k in f["bp"]]
        rungs += [("pre-edit state carried forward (countdown analogue)", pre_r), ("injection-time probe", f["p"])]
        for name, pred in rungs:
            out["ladder"].append({"regime": rg, "predictor": name, "auroc": auroc(yt, pred) if "floor" not in name else 0.5, "r2": float(r2(yt, pred)) if "floor" not in name else 0.0,
                                  "brier_skill": brier_skill(yt, pred, base), "mae_over_floor": 1.0 if "floor" in name else mae_ratio(yt, np.clip(pred, 0, 1), med),
                                  "tpr10": tpr_at_fpr(yt, pred) if "floor" not in name else 0.1, "n_test": int(len(test))})
        log(f"  paper ladder {rg}: probe AUROC {out['ladder'][-1]['auroc']:.3f}")
    # controls (iid): shuffled labels through the whole pipeline
    pool, test = cv_eval.split_indices(df.split.values, df.d1.values, "iid")
    y = df.flip.values.astype(np.float64)
    real = cv_eval.cv_probe(lambda li: HA[:, li, :].astype(np.float32), list(range(len(ly))), y[:, None], pool, test, g)[0]
    out["controls"].append({"control": "real labels", "seed": 0, "test_r2": float(r2(y[test], real["pred"])), "test_auroc": auroc(y[test], real["pred"])})
    for kind in ("global", "within-problem"):
        for s in seeds:
            rng = np.random.RandomState(500 + s)
            if kind == "global":
                ys = rng.permutation(y)
            else:
                ys = y.copy()
                for gi in np.unique(g):
                    ix = np.where(g == gi)[0]
                    ys[ix] = y[rng.permutation(ix)]
            r = cv_eval.cv_probe(lambda li: HA[:, li, :].astype(np.float32), list(range(len(ly))), ys[:, None], pool, test, g)[0]
            out["controls"].append({"control": f"shuffled labels ({kind})", "seed": s, "test_r2": float(r2(ys[test], r["pred"])), "test_r2_vs_real_labels": float(r2(y[test], r["pred"])), "test_auroc_vs_real_labels": auroc(y[test], r["pred"])})
    # fold-seed stability
    rs, au = [], []
    for s in seeds:
        f = regime_fit(P, bl, "iid", seed=s)
        rs.append(float(r2(f["y"][f["test"]], f["p"])))
        au.append(auroc(f["y"][f["test"]], f["p"]))
    out["seeds"] = [{"metric": "test R2", "mean": float(np.mean(rs)), "sd": float(np.std(rs)), "values": rs}, {"metric": "test AUROC", "mean": float(np.mean(au)), "sd": float(np.std(au)), "values": au}]
    # continuous propagation label, in the paper's units (MAE vs constant-median floor)
    if "prop" in df:
        for rg in ("iid", "shift"):
            pool, test = cv_eval.split_indices(df.split.values, df.d1.values, rg)
            yp = df.prop.values.astype(np.float64)
            m = np.isfinite(yp)
            pool, test = pool[m[pool]], test[m[test]]
            if len(pool) < 60 or len(test) < 30:
                continue
            res = cv_eval.cv_probe(lambda li: HA[:, li, :].astype(np.float32), list(range(len(ly))), yp[:, None], pool, test, g)[0]
            medv = float(np.median(yp[pool]))
            bpr = cv_eval.cv_probe(lambda _l: P["obs"], [0], yp[:, None], pool, test, g)[0]["pred"]
            out["prop"].append({"regime": rg, "n_test": int(len(test)), "floor_mae": float(np.abs(yp[test] - medv).mean()), "probe_mae": float(np.abs(yp[test] - res["pred"]).mean()),
                                "probe_over_floor": float(np.abs(yp[test] - res["pred"]).mean() / np.abs(yp[test] - medv).mean()), "observables_mae": float(np.abs(yp[test] - bpr).mean()),
                                "observables_over_floor": float(np.abs(yp[test] - bpr).mean() / np.abs(yp[test] - medv).mean()), "probe_r2": float(r2(yp[test], res["pred"]))})
    return out


def offline(run_dir: Path, mega_dir: Path, log=print, B: int = 500, name: str = "robust", ds: str = "gsm8k") -> dict:
    """everything that needs no generation.  Results are cached in <mega>/<name>/offline.json."""
    from . import mega
    from .inject import prepare
    P, why = prepare(run_dir, mega_dir, ds)
    if P is None:
        return {"status": "n/a", "why": why}
    log(f"robust offline: {len(P['df'])} rows from {P['df'].id.nunique()} problems")
    sc = subset_checks(P, log, B)
    bl = expanded_baselines(P)
    res = {"status": "ok", "n_rows": int(len(P["df"])), "n_problems": int(P["df"].id.nunique()), "checks": sc["status"], "rows": mega.rows_df(sc["rows"]).to_dict("records")}
    res["shuffled"] = shuffled_control(P)
    log(f"  shuffled-label R2 by seed {res['shuffled']['r2_by_seed']}")
    res["curves"] = curves(P)
    res["diag"] = diagnostics(P, bl, B, log)
    res["paper"] = paper_metrics(P, bl, log)
    res["meta"] = {"ds": ds, "n_unfinished_excluded": int(P.get("n_unfinished", 0)), "n_rows_before_filter": int(P.get("n_rows_all", 0)), "base_flip_rate": float(P["df"].flip.mean()), "n_problems_by_split": P["df"].groupby("split").id.nunique().to_dict()}
    out = Path(mega_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "offline.json").write_text(json.dumps(res, indent=1, default=float))
    return res


# ----------------------------------------------------------------------------- scorecard
def scorecard(off: dict, steer_res: dict = None) -> List[dict]:
    """-> [{'check', 'what', 'ok' (True/False/None=not evaluable), 'detail'}], C1-C8, rules as pre-registered."""
    rows = []
    ck = off.get("checks", {}) if off.get("status") == "ok" else {}

    def sup(key, what, label):
        s = ck.get(key)
        if not s or s["status"] == "n/a":
            rows.append({"check": label, "what": what, "ok": None, "detail": "not evaluable" + (f" ({s['why']})" if s else "")})
            return
        d = s["why"]
        if "iid_delta" in s:
            d += f"; iid R2 {s['iid_probe_r2']:.3f} vs {s['iid_obs_r2']:.3f} (gain {s['iid_delta']:+.3f}, CI lower {s['iid_lo']:+.3f})"
            if "shift_delta" in s:
                d += f"; shift gain {s['shift_delta']:+.3f} (CI lower {s['shift_lo']:+.3f})"
        rows.append({"check": label, "what": what, "ok": s["status"] == "SUPPORTED", "detail": f"{s['status']}: {d}"})
    sup("C1", "last numeric step of each problem dropped", "C1")
    sup("C2", "label demeaned within problem", "C2")
    sup("C3", "each edit variant alone (n1 and n2)", "C3")
    sup("C4", "all rows vs bag-of-words baselines added", "C4")
    sh = off.get("shuffled")
    rows.append({"check": "C5", "what": "shuffled-label control", "ok": None if not sh else sh["ok"], "detail": "not run" if not sh else f"R2 by seed {', '.join(f'{x:.3f}' for x in sh['r2_by_seed'])} (must be <= {SHUF_MAX})"})
    dg = off.get("diag", {}).get("regimes", {}) if off.get("status") == "ok" else {}
    if "iid" in dg:
        r = dg["iid"]
        ok = r["r2_delta_vs_pre_lo"] > 0
        rows.append({"check": "C6", "what": "injection-time state beats the natural pre-edit state (paired)", "ok": bool(ok),
                     "detail": f"iid R2 {r['probe_r2']:.3f} vs {r['pre_r2']:.3f}: gain {r['r2_delta_vs_pre']:+.3f} [{r['r2_delta_vs_pre_lo']:+.3f}, {r['r2_delta_vs_pre_hi']:+.3f}]"})
    else:
        rows.append({"check": "C6", "what": "injection-time state beats the natural pre-edit state (paired)", "ok": None, "detail": "not evaluable"})
    if "iid" in dg and "shift" in dg:
        a, b = dg["iid"], dg["shift"]
        ok = a["auroc_lo"] > 0 and b["auroc_lo"] > 0
        rows.append({"check": "C7", "what": "AUROC gain over best observable, iid and shift", "ok": bool(ok),
                     "detail": f"iid {a['probe_auroc']:.3f} vs {a['obs_auroc']:.3f} ({a['auroc_delta']:+.3f} [{a['auroc_lo']:+.3f}, {a['auroc_hi']:+.3f}]); shift {b['probe_auroc']:.3f} vs {b['obs_auroc']:.3f} "
                               f"({b['auroc_delta']:+.3f} [{b['auroc_lo']:+.3f}, {b['auroc_hi']:+.3f}])"})
    else:
        rows.append({"check": "C7", "what": "AUROC gain over best observable, iid and shift", "ok": None, "detail": "not evaluable"})
    if steer_res and steer_res.get("status") not in (None, "n/a"):
        rows.append({"check": "C8", "what": "causal: steering the probe direction at the injected token moves follow-vs-recover", "ok": steer_res["status"] == "SUPPORTED", "detail": f"{steer_res['status']}: {steer_res['why']}"})
    else:
        rows.append({"check": "C8", "what": "causal: steering the probe direction at the injected token moves follow-vs-recover", "ok": None, "detail": "steering not run / not finished" + ("" if not steer_res else f": {steer_res.get('why', '')}")})
    return rows


def overall(rows: List[dict]) -> dict:
    byc = {r["check"]: r["ok"] for r in rows}
    diag = [byc.get(f"C{i}") for i in range(1, 8)]
    failed = [f"C{i}" for i in range(1, 9) if byc.get(f"C{i}") is False]
    unknown = [f"C{i}" for i in range(1, 9) if byc.get(f"C{i}") is None]
    if all(x is True for x in diag):
        if byc.get("C8") is True:
            return {"verdict": "ROBUST + CAUSAL", "why": "C1-C8 all pass"}
        return {"verdict": "ROBUST (diagnostic)", "why": "C1-C7 pass; " + ("C8 failed (the direction is a correlate, not a lever at these doses)" if byc.get("C8") is False else "C8 not evaluated")}
    if failed:
        return {"verdict": "FRAGILE", "why": "failed: " + ", ".join(failed) + (f"; not evaluable: {', '.join(unknown)}" if unknown else "")}
    return {"verdict": "INCOMPLETE", "why": "not evaluable: " + ", ".join(unknown)}


RULE = """Pre-registered (docs/ROBUST.md).  Headline object: I[num]:flip.  C1-C4 need the generic SUPPORTED status (probe beats the best OBSERVABLE baseline, CI lower bound > 0 and R2 > 0.05, in iid AND shift) on the stated subset / label.
C5: shuffled-label R2 <= 0.02.  C6: paired CI lower bound of (R2 injection-time - R2 natural pre-edit) > 0 in iid.  C7: AUROC-difference CI lower bound > 0 vs the best observable in iid and shift.
C8: steering SUPPORTED at >= 1 layer (slope CI > 0 over |alpha| <= 16 sd, > 2x the largest random-direction slope, correct sign at +-8 and +-16, integrity within 0.05 at |alpha| <= 8).
ROBUST + CAUSAL = C1-C8, ROBUST (diagnostic) = C1-C7, anything else FRAGILE (failed checks listed)."""


def write_report(run_dir: Path, mega_dir: Path, log=print, B: int = 500, recompute: bool = False, name: str = "robust", ds: str = "gsm8k") -> Path:
    import traceback
    import pandas as pd
    from . import core, istear, mega
    mega_dir = Path(mega_dir)
    outd = mega_dir / name / "analysis"
    outd.mkdir(parents=True, exist_ok=True)
    off = None
    f = mega_dir / name / "offline.json"
    if f.exists() and not recompute:
        off = json.loads(f.read_text())
    else:
        try:
            off = offline(run_dir, mega_dir, log, B, name, ds)
        except Exception:                                                                    # noqa: BLE001
            log(traceback.format_exc())
            off = {"status": "error", "why": traceback.format_exc()}
    steer_res, steer_err = None, ""
    try:
        steer_res = istear.analyze(mega_dir / name / "steer")
    except Exception:                                                                        # noqa: BLE001
        steer_err = traceback.format_exc()
        log(steer_err)
    rows = scorecard(off, steer_res)
    ov = overall(rows)
    head = ["# Robustness + causal test of the injection probe (GSM8K, Qwen2.5-7B-Instruct)", "", RULE, "", f"## Verdict: **{ov['verdict']}** - {ov['why']}", "",
            core.md(pd.DataFrame([{"check": r["check"], "what": r["what"], "pass": {True: "PASS", False: "FAIL", None: "n/a"}[r["ok"]], "detail": r["detail"]} for r in rows]))]
    body = []
    if off.get("status") == "ok":
        body.append(f"## Offline checks\n\n{off['n_rows']} injected rows from {off['n_problems']} problems.\n")
        rdf = pd.DataFrame(off["rows"])
        body.append(mega.tab(rdf))
        rdf.to_csv(outd / "robust_results.csv", index=False)
        dg = off["diag"]["regimes"]
        if dg:
            body.append("### AUROC / TPR at 10% FPR (one fit per regime)\n\n" + core.md(pd.DataFrame([{"regime": k, **{a: b for a, b in v.items()}} for k, v in dg.items()])))
        if off["diag"]["strata"]:
            body.append("### Strata (iid test)\n\n" + core.md(pd.DataFrame(off["diag"]["strata"])))
        cu = pd.DataFrame(off["curves"]).rename_axis("layer").reset_index().sort_values("layer")
        body.append("### CV R2 by layer (iid, best lambda)\n\n" + core.md(cu))
    else:
        body.append(f"## Offline checks\n\n**{off.get('status')}**: {off.get('why', '')}\n")
    if steer_res and steer_res.get("status") != "n/a":
        s = f"## Causal test (steering at the injected token)\n\n**{steer_res['status']}**: {steer_res['why']}\n\n{steer_res['n']} injected rows finished in every condition; baseline flip rate {steer_res['base_flip']:.3f}\n\n"
        s += core.md(pd.DataFrame([{"family": k, "slope_flip_per_sd": v[0], "lo95": v[1], "hi95": v[2]} for k, v in steer_res["slopes"].items()]))
        s += "\nPer layer:\n\n" + core.md(pd.DataFrame([{"layer": L, **{k: v for k, v in d.items() if k != "slope"}} for L, d in steer_res["layers"].items()]))
        s += "\nDose-response:\n\n" + core.md(steer_res["dose"]) + "\nPer condition:\n\n" + core.md(steer_res["table"])
        body.append(s)
    else:
        body.append("## Causal test (steering at the injected token)\n\n" + (f"**n/a**: {steer_res['why']}\n" if steer_res else f"not available\n\n```\n{steer_err}\n```\n" if steer_err else "not run\n"))
    pd.DataFrame(rows).to_csv(outd / "robust_scorecard.csv", index=False)
    p = outd / "ROBUST_REPORT.md"
    p.write_text("\n".join(head) + "\n" + "\n".join(body))
    log(f"wrote {p}")
    return p
