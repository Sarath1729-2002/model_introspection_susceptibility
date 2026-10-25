"""Causal-by-design analyses.  NO intervention on the model's activations - every 'treatment' is a change to the INPUT that WE randomise or control:

  * hint present / absent / placebo / correct          within-problem contrasts (same problem, same weights, only the prompt differs)
  * the wrong value's offset from gold, and its option position   assigned by a seeded RNG independent of everything else (randomised)
  * difficulty in the synthetic sets                     set by the generator (k operations / loop iterations), not measured
What these identify: causal effects of PROMPT features on behaviour/belief.  What they do not: any claim that a probe direction is causally
used by the model (that is the steering study, deferred).  Mediation-style decompositions below are observational given the randomised input.
"""
from typing import List

import numpy as np
import pandas as pd

from . import core
from .prompts import FRAMINGS

ALL = "all"


def _scopes(df: pd.DataFrame):
    out = [(ds, df[df.ds == ds]) for ds in df.ds.unique()]
    return out + [(ALL, df)]


def _ci(x):
    m, lo, hi = core.boot_ci(x, B=500)
    return m, lo, hi


def treatment_effects(df: pd.DataFrame) -> pd.DataFrame:
    """per scope x condition: adoption, accuracy, parse failures and the within-problem effect vs clean / vs placebo."""
    rows = []
    for scope, d in _scopes(df):
        base_acc = d.p_clean.values
        for f in FRAMINGS:
            fo, ac = d[f"follow_{f}"].values, d[f"acc_{f}"].values
            m, lo, hi = _ci(fo)
            da, dlo, dhi = _ci(base_acc - ac)
            ad, alo, ahi = _ci(fo - d.follow_plc.values)
            rows.append({"scope": scope, "n": len(d), "cond": f, "follow": m, "follow_lo": lo, "follow_hi": hi, "adopt_net_of_placebo": ad,
                         "net_lo": alo, "net_hi": ahi, "acc": ac.mean(), "acc_drop_vs_clean": da, "drop_lo": dlo, "drop_hi": dhi,
                         "frac_problems_any_adoption": float((fo > 0).mean()), "parse_fail": d[f"none_{f}"].mean(), "gap_nats": d[f"gap_{f}"].mean()})
        m, lo, hi = _ci(d.follow_plc.values)
        rows.append({"scope": scope, "n": len(d), "cond": "placebo", "follow": m, "follow_lo": lo, "follow_hi": hi, "acc": d.acc_plc.mean(),
                     "acc_drop_vs_clean": (d.p_clean - d.acc_plc).mean(), "frac_problems_any_adoption": float((d.follow_plc > 0).mean()),
                     "parse_fail": d.none_plc.mean(), "gap_nats": d.gap_plc.mean()})
        g, glo, ghi = _ci(d.acc_cor.values - d.p_clean.values)
        rows.append({"scope": scope, "n": len(d), "cond": "cor(+gain)", "follow": float("nan"), "acc": d.acc_cor.mean(), "acc_drop_vs_clean": -g,
                     "drop_lo": -ghi, "drop_hi": -glo, "parse_fail": d.none_cor.mean()})
        rows.append({"scope": scope, "n": len(d), "cond": "clean", "follow": d.spont_clean.mean(), "acc": d.p_clean.mean(), "parse_fail": d.none_clean.mean(),
                     "gap_nats": d.gap_clean.mean() if "gap_clean" in d else float("nan")})
    return pd.DataFrame(rows)


def hint_distance(df: pd.DataFrame) -> pd.DataFrame:
    """The wrong value's offset from gold (integers) / its option position (MCQ) is randomised -> design-based dose-response of adoption S."""
    rows = []

    def add(scope, factor, groups):
        for lv, g in groups:
            if len(g) >= 20:
                m, lo, hi = _ci(g.S.values)
                rows.append({"scope": scope, "factor": factor, "level": float(lv), "n": len(g), "S": m, "lo": lo, "hi": hi,
                             "gap_nats": g.Sgap.mean() if "Sgap" in g else float("nan")})
    ints = df[df.kind == "int"]
    for scope, d in [(ds, g) for ds, g in ints.groupby("ds")] + ([("int-pooled", ints)] if ints.ds.nunique() > 1 else []):
        add(scope, "abs_offset", d.groupby(d.hint_off.abs()))
        add(scope, "offset_sign", d.groupby(np.sign(d.hint_off)))
    a = df[df.kind == "mcq"]
    if len(a):
        add("arc", "hint_option_position", a.groupby("hint_pos"))
        add("arc", "gold_option_position", a.groupby("gold_pos"))
        add("arc", "signed_position_offset", a.groupby("hint_off"))
    return pd.DataFrame(rows)


def difficulty_response(df: pd.DataFrame) -> pd.DataFrame:
    """susceptibility vs difficulty; the synthetic sets have difficulty set by the generator (k) -> a manipulated variable."""
    rows = []
    for scope, d in _scopes(df):
        if scope == ALL:
            continue
        sp, lo, hi = core.boot_stat(lambda i: core.spearman(d.d1.values[i], d.S.values[i]), len(d), B=300)
        sp2 = core.spearman(d.p_clean.values, d.S.values)
        rows.append({"scope": scope, "design_controlled": scope in ("arith", "prog"), "n": len(d), "spearman(d1,S)": sp, "lo": lo, "hi": hi,
                     "spearman(p_clean,S)": sp2, "mean_S": d.S.mean(), "mean_p_clean": d.p_clean.mean()})
    return pd.DataFrame(rows)


def difficulty_bins(df: pd.DataFrame, nb=5) -> pd.DataFrame:
    rows = []
    for ds, d in df.groupby("ds"):
        u = np.unique(d.d1)
        grp = d.d1.rank(method="dense").astype(int) if len(u) <= nb else pd.qcut(d.d1.rank(method="first"), nb, labels=False)
        for g, h in d.groupby(grp):
            rows.append({"ds": ds, "bin": int(g), "d1_mean": h.d1.mean(), "n": len(h), "S": h.S.mean(), "p_clean": h.p_clean.mean(), "acc_hint": h[[f"acc_{f}" for f in FRAMINGS]].mean(axis=1).mean()})
    return pd.DataFrame(rows)


def belief_to_behaviour(df: pd.DataFrame) -> pd.DataFrame:
    """How much of the behavioural adoption is carried by the forced (no-CoT) belief shift?  Observational given the randomised hint.
    R^2 of follow_f ~ shift (gap_f - gap_clean), vs follow_f ~ difficulty only, vs both."""
    rows = []
    for scope, d in _scopes(df):
        for f in FRAMINGS:
            y = d[f"follow_{f}"].values
            shift = (d[f"gap_{f}"] - d.gap_clean).values
            dif = d[["p_clean", "lpg_clean"]].values
            def r2_of(X):
                X = np.column_stack([np.ones(len(X)), X])
                w = np.linalg.lstsq(X, y, rcond=None)[0]
                return core.r2(y, X @ w)
            a, b, c = r2_of(dif), r2_of(shift[:, None]), r2_of(np.column_stack([dif, shift]))
            rows.append({"scope": scope, "framing": f, "n": len(d), "R2[difficulty]": a, "R2[belief shift]": b, "R2[both]": c, "added_by_shift": c - a,
                         "spearman(shift,follow)": core.spearman(y, shift)})
    return pd.DataFrame(rows)


def credulity_link(df: pd.DataFrame) -> pd.DataFrame:
    """Is susceptibility to a WRONG hint the same latent as credulity toward a CORRECT hint (hint-trust), or is it difficulty/knowledge?"""
    pairs = [("S", "cred"), ("S", "uplift"), ("S_net", "uplift"), ("S", "p_clean"), ("cred", "p_clean"), ("S", "follow_plc"), ("Sgap", "uplift"), ("S", "ent_clean")]
    rows = []
    for scope, d in _scopes(df):
        for a, b in pairs:
            if a in d and b in d:
                v, lo, hi = core.boot_stat(lambda i: core.spearman(d[a].values[i], d[b].values[i]), len(d), B=300)
                rows.append({"scope": scope, "a": a, "b": b, "spearman": v, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- uses held-out probe predictions
def monitor_curves(df: pd.DataFrame, res: dict, budgets=(0.05, 0.1, 0.2, 0.3, 0.5)) -> pd.DataFrame:
    """Deployment view: flag the top-b fraction of held-out problems by predicted susceptibility; how many of the problems that truly adopt
    (S >= 0.25) are caught, and how much more adoption do flagged problems show?  probe vs best observable baseline vs best oracle."""
    ids = res["_test_ids"]
    d = df.set_index("id").loc[ids]
    S, adopts = d.S.values, (d.S.values >= 0.25)
    rows = []
    for name, pred in (("probe", res["_pred"]), ("best observable baseline", res["_pred_obs"]), ("best oracle baseline", res["_pred_oracle"])):
        order = np.argsort(-pred)
        for b in budgets:
            k = max(1, int(round(b * len(pred))))
            top = order[:k]
            rows.append({"scorer": name, "budget": b, "recall_of_adopters": float(adopts[top].sum() / max(adopts.sum(), 1)),
                         "lift_in_mean_S": float(S[top].mean() / max(S.mean(), 1e-9)), "precision": float(adopts[top].mean()),
                         "base_rate": float(adopts.mean()), "auroc": core.auroc(adopts, pred)})
    return pd.DataFrame(rows)


def stratified_contrast(df: pd.DataFrame, res: dict, strata_cols=("ds",), n_bins=3) -> dict:
    """Difficulty-adjusted contrast: within strata of (dataset x clean-accuracy tercile), compare realised S in the top vs bottom half by predicted S.
    A probe that only reads difficulty gives ~0 here."""
    ids = res["_test_ids"]
    d = df.set_index("id").loc[ids].reset_index()
    d["pred"] = res["_pred"]
    d["bin"] = d.groupby("ds").p_clean.transform(lambda x: pd.qcut(x.rank(method="first"), n_bins, labels=False, duplicates="drop"))
    d["stratum"] = d.ds.astype(str) + "_" + d.bin.astype(str)
    def stat(ix):
        t = d.iloc[ix]
        num, w = 0.0, 0.0
        for _, g in t.groupby("stratum"):
            if len(g) < 6:
                continue
            hi = g.pred >= g.pred.median()
            if hi.all() or (~hi).all():
                continue
            num += (g.S[hi].mean() - g.S[~hi].mean()) * len(g)
            w += len(g)
        return num / w if w else float("nan")
    v, lo, hi = core.boot_stat(stat, len(d), B=300)
    return {"adjusted_diff_S_top_minus_bottom_half": v, "lo": lo, "hi": hi, "n": len(d)}
