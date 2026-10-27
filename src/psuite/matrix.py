"""CROSS-CELL analyses: the cross-dataset transfer matrix of the paper (their section 5), for error propagation.

A cell = (model, dataset).  For two datasets of the SAME model (identical hidden size / layer list) the flip probe is fitted on every injected row of dataset A's iid pool (layer and lambda by grouped CV on A),
frozen, and evaluated on dataset B's iid test rows: AUROC, Spearman, raw Brier skill (intercept NOT re-fitted, so it also reports calibration drift).  Reference rows on B: B's own probe and B's best OBSERVABLE baseline
(fitted on B), plus the constant floor (AUROC 0.5).  A transferred probe 'counts' if it clears B's best observable baseline AUROC, the strict bar (the paper's bar is the constant floor; both are reported).
"""
from typing import Dict, List

import numpy as np

from . import cv_eval, ridge
from .core import spearman
from .robust import auroc, brier_skill, regime_fit


def transfer(cells: Dict[str, dict], log=print) -> List[dict]:
    """cells: {name: P} from inject.prepare, all from the same model.  -> rows (train, eval, ...)."""
    rows, fits, own = [], {}, {}
    for a, P in cells.items():
        df, ly = P["df"], P["ly"]
        f = regime_fit(P, P["bl"], "iid")
        if f is None:
            log(f"  transfer: cell {a} too small for an iid split")
            continue
        y = f["y"]
        res = cv_eval.cv_probe(lambda li: P["HA"][:, li, :].astype(np.float32), list(range(len(ly))), y[:, None], f["pool"], f["test"], df.id.values)[0]
        li, ri = res["layer"], ridge.RATIOS.index(res["ratio"])
        X = P["HA"][:, li, :].astype(np.float32)
        fits[a] = (int(ly[li]), ri, ridge.fit_layer(cv_eval._t(X[f["pool"]]), y[f["pool"]][:, None]), float(y[f["pool"]].mean()))
        own[a] = {"auroc": auroc(y[f["test"]], f["p"]), "obs_auroc": auroc(y[f["test"]], f["q"]), "best": f["best"], "base": float(y[f["test"]].mean()), "n": int(len(f["test"]))}
    for a, (layer, ri, fit, _) in fits.items():
        for b, P in cells.items():
            if b not in own:
                continue
            if a == b:
                rows.append({"train": a, "eval": b, "layer": layer, "auroc": own[b]["auroc"], "spearman": float("nan"), "brier_skill_raw": float("nan"), "eval_best_observable": own[b]["best"],
                             "eval_observable_auroc": own[b]["obs_auroc"], "beats_floor": True, "beats_best_observable": own[b]["auroc"] > own[b]["obs_auroc"], "n_eval": own[b]["n"], "diagonal": True})
                continue
            if layer not in P["ly"]:
                continue
            df = P["df"]
            f = regime_fit(P, P["bl"], "iid")
            te = f["test"]
            X = P["HA"][te, P["ly"].index(layer), :].astype(np.float32)
            pred = fit.predict(cv_eval._t(X), 0).cpu().numpy()[:, ri]
            yb = df.flip.values.astype(np.float64)[te]
            au = auroc(yb, pred)
            rows.append({"train": a, "eval": b, "layer": layer, "auroc": au, "spearman": float(spearman(yb, pred)), "brier_skill_raw": brier_skill(yb, pred, float(yb.mean())),
                         "eval_best_observable": own[b]["best"], "eval_observable_auroc": own[b]["obs_auroc"], "beats_floor": bool(au > 0.5), "beats_best_observable": bool(au > own[b]["obs_auroc"]),
                         "n_eval": int(len(te)), "diagonal": False})
            log(f"  transfer {a}->{b}: AUROC {au:.3f} (B's own probe {own[b]['auroc']:.3f}, B's best observable {own[b]['obs_auroc']:.3f})")
    return rows
