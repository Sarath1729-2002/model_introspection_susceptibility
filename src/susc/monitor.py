"""Monitoring along the reasoning trace (analysis side of trace.py).

Question: as the model writes its solution, can a linear probe on the residual stream tell - before the final answer is written - whether this
trajectory will END on the planted wrong answer (or end wrong at all)?  And how much earlier / better is it than reading the text written so far?
Labels are deterministic (the trajectory's own parsed answer vs the planted value / gold).  Splits are by problem.
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import core
from .bench import read_problems
from .prompts import TRACE_CONDS
from .table import bow, ds_onehot, hint_oracle, hint_visible, question_features
from .trace import POS_NAMES

HINTED = ("auth", "pressure")


def load_trace(run_dir: Path, datasets=None, conds=TRACE_CONDS):
    run_dir = Path(run_dir)
    ids, H, FR, ln, layers = [], {c: [] for c in conds}, {c: [] for c in conds}, {c: [] for c in conds}, None
    for f in sorted(glob.glob(str(run_dir / "trace" / "t*.npz"))):
        if ".tmp" in f:
            continue
        z = np.load(f)
        if not all(f"h_{c}" in z.files for c in conds):
            continue
        ids.append(z["ids"])
        layers = z["layers"]
        for c in conds:
            H[c].append(z[f"h_{c}"]), FR[c].append(z[f"frac_{c}"]), ln[c].append(z[f"len_{c}"])       # noqa: E702
    if not ids:
        return None
    ids = np.concatenate(ids)
    o = np.argsort(ids)
    out = {"ids": ids[o], "layers": layers}
    for c in conds:
        out[c] = {"h": np.concatenate(H[c])[o], "frac": np.concatenate(FR[c])[o], "len": np.concatenate(ln[c])[o]}
    return out


def load_trace_samples(run_dir: Path, ids, conds=TRACE_CONDS):
    want = set(int(i) for i in ids)
    out = {}
    for f in sorted(glob.glob(str(Path(run_dir) / "samples" / "s*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if r["cond"] in conds and r["id"] in want:
                out[(r["id"], r["cond"])] = (r["answers"][0], r["t0"])
    return out


def partial_text(t0: str, pos: int) -> str:
    if pos == 0:
        return ""
    if pos == 4:
        i = t0.rfind("\\boxed")
        return t0[:i] if i >= 0 else t0
    return t0[: int((0.25, 0.5, 0.75)[pos - 1] * len(t0))]


def _mentions(text: str, val: str, kind: str) -> float:
    if kind == "int":
        return float(re.search(rf"(?<![\d.,]){re.escape(val)}(?![\d.]|,\d)", text) is not None)
    return float(re.search(rf"(?<![A-Za-z]){val}(?![A-Za-z])", text) is not None)


def text_baselines(df: pd.DataFrame, texts, pos: int):
    Bq = np.concatenate([np.array([question_features(q) for q in df._q], dtype=np.float32), ds_onehot(df), hint_visible(df)], 1)
    pt = [partial_text(t, pos) for t in texts]
    feats = np.array([[np.log1p(len(t)), np.log1p(len(re.findall(r"\d+", t))), t.count("="), _mentions(t, h, k)]
                      for t, h, k in zip(pt, df.hint, df.kind)], dtype=np.float32)
    Btext = np.concatenate([Bq, feats, np.stack([bow(t, 256) for t in pt])], 1)
    gold_m = np.array([[_mentions(t, g, k)] for t, g, k in zip(pt, df.gold, df.kind)], dtype=np.float32)
    return {"Bq+hint": Bq, "Btext(partial solution)": Btext, "Btext+gold(oracle)": np.concatenate([Btext, gold_m, hint_oracle(df)], 1)}, feats[:, 3]


def analyse(run_dir: Path, datasets=None, log=print):
    tr = load_trace(run_dir, datasets)
    if tr is None:
        return None
    probs = {p["id"]: p for p in read_problems(run_dir, datasets)}
    ids = [int(i) for i in tr["ids"] if int(i) in probs]
    sel = np.searchsorted(tr["ids"], ids)
    S = load_trace_samples(run_dir, ids)
    ids = [i for i in ids if all((i, c) in S for c in TRACE_CONDS)]
    sel = np.searchsorted(tr["ids"], ids)
    rows = []
    for i in ids:
        p = probs[i]
        r = {"id": i, "ds": p["ds"], "kind": p["kind"], "split": p["split"], "d1": p["d1"], "gold": p["gold"], "hint": p["hint"], "hint_off": p["hint_off"], "_q": p["question"], "ntok": 0}
        for c in TRACE_CONDS:
            a, _ = S[(i, c)]
            r[f"y_follow_{c}"] = float(a == p["hint"])
            r[f"y_correct_{c}"] = float(a == p["gold"])
        rows.append(r)
    df = pd.DataFrame(rows)
    nl = tr[TRACE_CONDS[0]]["h"].shape[2]
    out = []
    scopes = [(ds, (df.ds == ds).values) for ds in df.ds.unique() if (df.ds == ds).sum() >= 200] + [("pooled", np.ones(len(df), bool))]
    for cond in TRACE_CONDS:
        targets = ([f"y_follow_{cond}"] if cond in HINTED else []) + [f"y_correct_{cond}"]
        Hc = tr[cond]["h"][sel]
        texts_all = [S[(i, cond)][1] for i in ids]
        for pos in range(5):
            for scope, m in scopes:
                d = df[m].reset_index(drop=True)
                sub = {"s": Hc[m][:, pos]}
                base, ment = text_baselines(d, [t for t, k in zip(texts_all, m) if k], pos)
                ment_of = dict(zip(d.id.values, ment))
                res, _ = core.run_targets(d, sub, "s", "s", targets, targets, ["iid"], base, list(range(nl)), "T", B=300)
                for r in res:
                    y, p = r["_y"], r["_pred"]
                    rate = float(y.mean())
                    silent = np.array([ment_of[i] == 0 for i in r["_test_ids"]])          # the hint value has NOT been written in the solution so far
                    ys, ps_, pt_ = y[silent], p[silent], r["_pred_obs"][silent]
                    out.append({"scope": scope, "cond": cond, "target": r["target"], "pos": POS_NAMES[pos], "pos_i": pos, "probe_layer": int(tr["layers"][r["probe_layer"]]),
                                "n_test": r["n_test"], "pos_rate": rate, "probe_auroc": core.auroc(y, p), "probe_r2": r["probe_test_r2"],
                                "text_auroc": core.auroc(y, r["_pred_obs"]), "best_text_baseline": r["best_observable"], "oracle_auroc": core.auroc(y, r["_pred_oracle"]),
                                "delta_r2_vs_text": r["delta_vs_obs"], "lo": r["obs_lo"], "hi": r["obs_hi"],
                                "silent_n": int(silent.sum()), "silent_pos": int(ys.sum()), "silent_probe_auroc": core.auroc(ys, ps_) if ys.sum() >= 5 and (1 - ys).sum() >= 5 else float("nan"),
                                "silent_text_auroc": core.auroc(ys, pt_) if ys.sum() >= 5 and (1 - ys).sum() >= 5 else float("nan")})
            log(f"  trace {cond} {POS_NAMES[pos]} done")
    return pd.DataFrame(out)
