"""REFERENCE ROW: the paper's prompt-end claim ("total response length is linearly decodable from the prompt's last hidden state"), re-measured on OUR cells with THEIR vocabulary
(Table 4/5, A.12, A.13), so that the error-propagation results sit on a pipeline that reproduces the published phenomenon and uses the same floors.

 target            T = number of tokens of the greedy solution (naturally terminated only; the exclusion rate is reported, as in their A.7)
 probe             ridge (single layer, layer and lambda by grouped CV on the training pool) on the last prompt token
 floors            constant median (MAE = mean absolute deviation about the eval median); surface ridge (question length features + hashed bag-of-words, the TF-IDF analogue); shuffled labels
 sampling floor    leave-one-out spread of the sampled solutions of the same prompt (T=0.7) - what no prompt-conditioned predictor can remove; and the distance between the greedy T and the median of the sampled draws
 reported          MAE (tokens), MAE / constant-median MAE, share of the closable range captured, Spearman
"""
from pathlib import Path

import numpy as np


def loo_mad(draws) -> float:
    d = np.asarray(draws, float)
    if len(d) < 3:
        return float("nan")
    return float(np.mean([abs(d[j] - np.median(np.delete(d, j))) for j in range(len(d))]))


def analyze(run_dir: Path, ds: str = "gsm8k", log=print) -> dict:
    from . import cv_eval, table
    from .core import spearman
    try:
        df, sub = table.build(run_dir, [ds], need_states=True, names=("plain",), log=log)
    except RuntimeError as e:
        return {"status": "n/a", "why": str(e)}
    G = table.load_gen(run_dir, [ds])
    fin = df.runaway.values == 0
    n_all, n_fin = len(df), int(fin.sum())
    X = sub["plain"]
    L = X.shape[1]
    layers = sorted(set(list(range(0, L, 4)) + [L - 1]))
    T = df["T"].values.astype(np.float64)
    pool = np.where(np.isin(df.split.values, ["fit", "val"]) & fin)[0]
    test = np.where((df.split.values == "test") & fin)[0]
    if len(pool) < 100 or len(test) < 50:
        return {"status": "n/a", "why": f"pool {len(pool)} / test {len(test)} naturally terminated problems"}
    g = df.id.values
    y = T[:, None]
    pr = cv_eval.cv_probe(lambda l: X[:, l, :].astype(np.float32), layers, y, pool, test, g)[0]
    yt = T[test]
    med = float(np.median(yt))
    floor = float(np.abs(yt - med).mean())
    surf = np.concatenate([np.array([table.question_features(q) for q in df._q], dtype=np.float32), np.stack([table.bow(q) for q in df._q])], 1)
    sr = cv_eval.cv_probe(lambda _l: surf, [0], y, pool, test, g)[0]
    rng = np.random.RandomState(0)
    ysh = rng.permutation(T)[:, None]
    sh = cv_eval.cv_probe(lambda l: X[:, l, :].astype(np.float32), layers, ysh, pool, test, g)[0]
    mae = lambda p: float(np.abs(yt - np.asarray(p)).mean())          # noqa: E731
    sp = [G[i] for i in df.id.values[test]]
    floors = [loo_mad([n for n, f in zip(r["spread_n"], r["spread_fin"]) if f]) for r in sp]
    meds = [np.median([n for n, f in zip(r["spread_n"], r["spread_fin"]) if f]) if sum(r["spread_fin"]) >= 2 else np.nan for r in sp]
    samp_floor = float(np.nanmean(floors)) if np.isfinite(floors).any() else float("nan")
    oracle = float(np.nanmean(np.abs(yt - np.array(meds)))) if np.isfinite(meds).any() else float("nan")
    out = {"status": "ok", "ds": ds, "n_problems": n_all, "n_terminated": n_fin, "excluded_share": float(1 - n_fin / n_all), "n_test": int(len(test)), "probe_layer": int(layers[pr["layer"]]),
           "floor_mae": floor, "probe_mae": mae(pr["pred"]), "probe_over_floor": mae(pr["pred"]) / floor, "surface_mae": mae(sr["pred"]), "surface_over_floor": mae(sr["pred"]) / floor,
           "shuffled_mae": mae(sh["pred"]), "shuffled_over_floor": mae(sh["pred"]) / floor, "probe_spearman": float(spearman(yt, pr["pred"])),
           "sampling_floor_loo_mad": samp_floor, "greedy_vs_sampled_median_mae": oracle}
    ref = out["greedy_vs_sampled_median_mae"]
    if np.isfinite(ref) and floor > ref:
        out["closable_captured_probe"] = (floor - out["probe_mae"]) / (floor - ref)
        out["closable_captured_surface"] = (floor - out["surface_mae"]) / (floor - ref)
    log(f"  lenref {ds}: probe MAE {out['probe_mae']:.1f} / floor {floor:.1f} (ratio {out['probe_over_floor']:.2f}); surface ratio {out['surface_over_floor']:.2f}; excluded {out['excluded_share']:.1%}")
    return out
