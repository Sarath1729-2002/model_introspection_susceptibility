"""Load problems + samples + states into one wide per-problem table (rows aligned with the state arrays).

Label glossary (all per problem; 'wrong' = the planted wrong answer, 'gold' = ground truth):
  p_clean, ent_clean, none_clean   sampled accuracy / answer entropy / parse-failure rate with NO hint
  lpg_clean, lph_clean             forced-answer log P(gold) / log P(wrong value), no reasoning
  follow_<f>   fraction of sampled solutions ending on the wrong value under framing f      (behaviour)
  gap_<f>      forced log P(wrong) - log P(gold) under framing f, clipped to +-20 nats      (belief, no sampling noise)
  follow_plc / gap_plc   same with the PLACEBO prompt (wrong value shown as an ID, not as a claim)  -> token copying / priming
  acc_cor / uplift       correct-hint condition: sampled accuracy; forced log P(gold) gain over clean   -> credulity
  S, Sgap      mean of follow / gap over the four wrong-hint framings   (latent susceptibility)
  S_net, Sgap_net   S - follow_plc,  Sgap - gap_plc    (susceptibility beyond copying the number)
  cred = acc_cor - p_clean      harm = p_clean - mean_f acc_f
"""
import glob
import json
import re
import zlib
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .bench import read_problems
from .prompts import CONDS, FRAMINGS

KEEP = ("n", "n_gold", "n_wrong", "n_none", "ent", "mean_chars")


def load_samples(run_dir: Path, datasets=None) -> Dict[tuple, dict]:
    out = {}
    for f in sorted(glob.glob(str(Path(run_dir) / "samples" / "s*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if datasets is None or r["ds"] in datasets:
                out[(r["id"], r["cond"])] = {k: r[k] for k in KEEP}
    return out


def load_states(run_dir: Path, state_conds: Iterable[str], lp_conds: Iterable[str] = CONDS):
    """-> {'ids', 'lp': {cond: {lpg,lph,ntok}}, 'states': {cond: [N,L+1,D] fp16}} for the problems present in every file.
    Hidden states are read only for `state_conds` (npz arrays are lazy), forced log-probs for all of `lp_conds`."""
    state_conds, lp_conds = list(state_conds), list(lp_conds)
    idl, lp, stt = [], {c: [] for c in lp_conds}, {c: [] for c in state_conds}
    for f in sorted(glob.glob(str(Path(run_dir) / "states" / "x*.npz"))):
        if ".tmp" in f:
            continue
        z = np.load(f)
        if not all(f"{c}_lpg" in z.files for c in lp_conds):
            continue
        idl.append(z["ids"])
        for c in lp_conds:
            lp[c].append({k: z[f"{c}_{k}"] for k in ("lpg", "lph", "ntok")})
        for c in state_conds:
            stt[c].append(z[f"{c}_states"])
    if not idl:
        return None
    ids = np.concatenate(idl)
    o = np.argsort(ids)
    return {"ids": ids[o], "lp": {c: {k: np.concatenate([d[k] for d in lp[c]])[o] for k in ("lpg", "lph", "ntok")} for c in lp_conds},
            "states": {c: np.concatenate(stt[c])[o] for c in state_conds}}


def question_features(q: str) -> list:
    nums = re.findall(r"\d+(?:\.\d+)?", q)
    return [np.log1p(len(q)), np.log1p(len(q.split())), np.log1p(len(nums)), np.log1p(len(re.findall(r"[.?!]", q))),
            np.log1p(max([float(x) for x in nums], default=0)), np.log1p(q.count("\n"))]


def bow(q: str, dim=512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for w in re.findall(r"[a-z]+|\d+", q.lower()):
        v[zlib.crc32(w.encode()) % dim] += 1
    return np.log1p(v)


def build(run_dir: Path, datasets=None, state_conds: Iterable[str] = (), need_states=True, log=print):
    """-> (df, sub).  sub[cond] = fp16 states [N, L+1, D] aligned with df rows (only for state_conds).  Only problems complete in every
    condition (samples AND forced log-probs) are kept.  need_states=False builds the behaviour-only table (no GPU data needed)."""
    run_dir = Path(run_dir)
    probs = {p["id"]: p for p in read_problems(run_dir, datasets)}
    S = load_samples(run_dir, datasets)
    st = load_states(run_dir, state_conds) if need_states else {"ids": np.array(sorted(probs))}
    if st is None:
        raise RuntimeError("no states yet")
    ids = [int(i) for i in st["ids"] if int(i) in probs and all((int(i), c) in S for c in CONDS)]
    if not ids:
        raise RuntimeError("no problem is complete in every condition (samples + states)")
    sel = np.searchsorted(st["ids"], ids) if need_states else None
    rows = []
    for i in ids:
        p = probs[i]
        r = {"id": i, "ds": p["ds"], "kind": p["kind"], "tier": p["tier"], "split": p["split"], "d1": p["d1"], "gold": p["gold"], "hint": p["hint"],
             "hint_off": p["hint_off"], "_q": p["question"]}
        if p["kind"] == "int":
            r["rel_off"] = abs(p["hint_off"]) / max(abs(int(p["gold"])), 1)
            r["gold_pos"] = float("nan")
            r["hint_pos"] = float("nan")
        else:
            r["rel_off"] = abs(p["hint_off"])
            r["gold_pos"] = "ABCD".index(p["gold"])
            r["hint_pos"] = "ABCD".index(p["hint"])
        c0 = S[(i, "clean")]
        r.update(n_samples=c0["n"], p_clean=c0["n_gold"] / c0["n"], ent_clean=c0["ent"], none_clean=c0["n_none"] / c0["n"],
                 spont_clean=c0["n_wrong"] / c0["n"], chars_clean=c0["mean_chars"])
        for f in FRAMINGS:
            s = S[(i, f)]
            r[f"follow_{f}"], r[f"acc_{f}"], r[f"none_{f}"], r[f"chars_{f}"] = s["n_wrong"] / s["n"], s["n_gold"] / s["n"], s["n_none"] / s["n"], s["mean_chars"]
        s = S[(i, "placebo")]
        r.update(follow_plc=s["n_wrong"] / s["n"], acc_plc=s["n_gold"] / s["n"], none_plc=s["n_none"] / s["n"])
        s = S[(i, "cor")]
        r.update(acc_cor=s["n_gold"] / s["n"], none_cor=s["n_none"] / s["n"])
        rows.append(r)
    df = pd.DataFrame(rows)
    df["S"] = df[[f"follow_{f}" for f in FRAMINGS]].mean(axis=1)
    df["S_net"] = df.S - df.follow_plc
    df["cred"] = df.acc_cor - df.p_clean
    df["harm"] = df.p_clean - df[[f"acc_{f}" for f in FRAMINGS]].mean(axis=1)
    df["adopts"] = (df.S >= 0.25).astype(float)
    sub = {}
    if need_states:
        lp = st["lp"]
        df["lpg_clean"] = np.clip(lp["clean"]["lpg"][sel], -30, 0)
        df["lph_clean"] = np.clip(lp["clean"]["lph"][sel], -30, 0)
        df["ntok"] = lp["clean"]["ntok"][sel]
        for f in FRAMINGS:
            df[f"gap_{f}"] = np.clip(lp[f]["lph"][sel] - lp[f]["lpg"][sel], -20, 20)
        df["gap_plc"] = np.clip(lp["placebo"]["lph"][sel] - lp["placebo"]["lpg"][sel], -20, 20)
        df["gap_clean"] = np.clip(lp["clean"]["lph"][sel] - lp["clean"]["lpg"][sel], -20, 20)
        df["uplift"] = np.clip(np.clip(lp["cor"]["lpg"][sel], -30, 0) - df["lpg_clean"], -20, 20)
        df["Sgap"] = df[[f"gap_{f}" for f in FRAMINGS]].mean(axis=1)
        df["Sgap_net"] = df.Sgap - df.gap_plc
        sub = {c: a[sel] for c, a in st["states"].items()}
    else:
        df["ntok"] = 0
    log(f"table: {len(df)} problems {df.ds.value_counts().to_dict()}")
    return df, sub


def ceilings(df: pd.DataFrame) -> pd.DataFrame:
    """R^2 ceiling implied by binomial sampling noise (n samples per problem) for the sampled labels.  Forced-answer labels have none."""
    n = df.n_samples.values.astype(float)
    rows = []

    def add(name, var, noise, mean):
        rows.append({"label": name, "mean": mean, "var": var, "noise_var": noise, "r2_ceiling": float(max(0.0, 1 - noise / max(var, 1e-12)))})
    for f in FRAMINGS:
        p = df[f"follow_{f}"].values
        add(f"follow_{f}", p.var(), np.mean(p * (1 - p) / np.maximum(n - 1, 1)), p.mean())
    P = np.stack([df[f"follow_{f}"].values for f in FRAMINGS], 1)
    add("S", df.S.var(), np.mean((P * (1 - P) / np.maximum(n[:, None] - 1, 1)).sum(1)) / len(FRAMINGS) ** 2, df.S.mean())
    p = df.p_clean.values
    add("p_clean", p.var(), np.mean(p * (1 - p) / np.maximum(n - 1, 1)), p.mean())
    pc = df.acc_cor.values
    add("cred", df.cred.var(), np.mean(p * (1 - p) / np.maximum(n - 1, 1)) + np.mean(pc * (1 - pc) / np.maximum(n - 1, 1)), df.cred.mean())
    q = df.follow_plc.values
    add("S_net", df.S_net.var(), np.mean((P * (1 - P) / np.maximum(n[:, None] - 1, 1)).sum(1)) / len(FRAMINGS) ** 2 + np.mean(q * (1 - q) / np.maximum(n - 1, 1)), df.S_net.mean())
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- baselines
def hint_visible(df: pd.DataFrame, shown: str = "hint") -> np.ndarray:
    """Features of the value shown in a hinted prompt that are READABLE from it (no gold).  shown='gold' for the correct-hint prompt."""
    rows = []
    for q, kind, h in zip(df._q, df.kind, df[shown]):
        if kind == "int":
            hv = float(int(h))
            nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", q)]
            rows.append([np.log1p(abs(hv)), len(str(abs(int(h)))), float(hv < 0), np.log1p(abs(hv)) - np.log1p(max([abs(x) for x in nums], default=0.0)),
                         float(hv in nums), float(hv == sum(nums)), float(len(nums) > 1 and hv == nums[0] * nums[1]), 0, 0, 0, 0])
        else:
            rows.append([0] * 7 + [float(h == c) for c in "ABCD"])
    return np.array(rows, dtype=np.float32)


def hint_oracle(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for kind, g, h, off in zip(df.kind, df.gold, df.hint, df.hint_off):
        if kind == "int":
            gv, hv = float(int(g)), float(int(h))
            rows.append([np.log1p(abs(gv)), np.log1p(abs(hv - gv)), np.sign(hv - gv), (hv - gv) / max(abs(gv), 1), 0, 0, 0, 0, 0])
        else:
            rows.append([0, 0, np.sign(off), off, 1] + [float(g == c) for c in "ABCD"])
    return np.array(rows, dtype=np.float32)


def ds_onehot(df: pd.DataFrame) -> np.ndarray:
    names = ["gsm8k", "math", "arith", "prog", "arc"]
    return np.stack([(df.ds.values == n).astype(np.float32) for n in names], 1)


def d1_rank(df: pd.DataFrame) -> np.ndarray:
    r = df.groupby("ds").d1.rank(pct=True).values.astype(np.float32)
    return np.stack([r, r ** 2], 1)


def ladder(df: pd.DataFrame, sub: Dict[str, np.ndarray], cond: str, hinted: bool, early=(4, 8), other: Optional[np.ndarray] = None,
           shown: str = "hint") -> Dict[str, np.ndarray]:
    """Baseline ladder.  Names containing 'oracle' use information the model cannot have at the prompt end."""
    oh = ds_onehot(df)
    Bq = np.array([question_features(q) + [np.log1p(n)] for q, n in zip(df._q, df.ntok)], dtype=np.float32)
    Bq = np.concatenate([Bq, oh], 1)
    Bbow = np.concatenate([np.stack([bow(q) for q in df._q]), oh], 1)
    Bd1 = np.concatenate([Bq, d1_rank(df)], 1)
    out = {"Bq": Bq, "Bbow": Bbow, "Bd1(oracle)": Bd1}
    diff = df[["p_clean", "ent_clean", "lpg_clean"]].values.astype(np.float32)
    out["Bdiff(oracle)"] = np.concatenate([Bd1, diff, diff ** 2], 1)
    if hinted:
        hv, ho = hint_visible(df, shown), hint_oracle(df)
        out["Bhint(visible)"] = np.concatenate([Bq, hv], 1)
        out["Bhint+gold(oracle)"] = np.concatenate([Bq, hv, ho, d1_rank(df)], 1)
        out["Bhint+gold+difficulty(oracle)"] = np.concatenate([out["Bhint+gold(oracle)"], diff, diff ** 2], 1)
        if other is not None:
            out["Bother(oracle)"] = np.concatenate([other, other ** 2, Bd1], 1)
    for l in early:
        if cond in sub and l < sub[cond].shape[1]:
            out[f"H{l}(early-layer)"] = sub[cond][:, l, :].astype(np.float32)
    return out
