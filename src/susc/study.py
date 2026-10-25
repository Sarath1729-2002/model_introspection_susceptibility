"""Driver for the analysis stage.  Blocks are independent (each loads only the states it needs) so the cluster runs them in parallel on
the four GPUs:   A  |  B1 (auth,user)  |  B2 (weak,pressure + placebo + correct-hint)  |  CDE (delta-state, leave-one-framing-out, geometry)  |  T (trace)
Each block writes  analysis/<block>.pkl ; report.py reads all pickles + the table-level causal analyses."""
import pickle
from pathlib import Path

import pandas as pd

from . import causal, core, monitor, probes, table
from .bench import DS_ORDER
from .prompts import CONDS, FRAMINGS

BLOCKS = ("A", "B1", "B2", "CDE", "T", "tables")


def _datasets(run_dir, datasets):
    from .bench import read_problems
    have = {p["ds"] for p in read_problems(run_dir)}
    return [d for d in DS_ORDER if d in have and (datasets is None or d in datasets)]


def _save(out: Path, name: str, obj):
    with open(out / f"{name}.pkl", "wb") as fh:
        pickle.dump(obj, fh)


def compute(run_dir, block: str, datasets=None, layers=None, log=print):
    run_dir = Path(run_dir)
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    dss = _datasets(run_dir, datasets)
    if block == "T":
        res = monitor.analyse(run_dir, dss, log)
        _save(out, "T", {"trace": res})
        if res is not None:
            res.to_csv(out / "trace_results.csv", index=False)
        return res
    if block == "tables":
        df, _ = table.build(run_dir, dss, state_conds=(), log=log)
        _save(out, "tables", {"df": df.drop(columns=["_q"]), "ceil": table.ceilings(df)})
        df.drop(columns=["_q"]).to_csv(out / "problem_table.csv", index=False)
        return df
    need = {"A": ("clean",), "B1": ("clean", "auth", "user"), "B2": ("clean", "weak", "pressure", "placebo", "cor"),
            "CDE": ("clean", "auth", "pressure") + FRAMINGS}[block]
    df, sub = table.build(run_dir, dss, state_conds=tuple(dict.fromkeys(need)), log=log)
    L = next(iter(sub.values())).shape[1]
    full = layers or list(range(1, L))
    coarse = layers or probes.coarse_layers(L)
    log(f"block {block}: n={len(df)} datasets={dss} layers full={full[0]}..{full[-1]} coarse={coarse}")
    if block == "A":
        res, cur = probes.block_A(df, sub, dss, full, log)
        _save(out, "A", {"res": res, "cur": cur})
        core.strip(res).to_csv(out / "A_results.csv", index=False)
    elif block in ("B1", "B2"):
        fr = ("auth", "user") if block == "B1" else ("weak", "pressure")
        res, cur = probes.block_B(df, sub, dss, coarse, framings=fr, log=log)
        if block == "B2":
            r2, c2 = probes.block_B_controls(df, sub, dss, coarse, log)
            res, cur = res + r2, pd.concat([cur, c2])
        _save(out, block, {"res": res, "cur": cur})
        core.strip(res).to_csv(out / f"{block}_results.csv", index=False)
    elif block == "CDE":
        resC, curC = probes.block_C(df, sub, dss, coarse, log=log)
        lofo = probes.block_D(df, sub, dss, coarse, log=log)
        geo = probes.block_E(df, sub, coarse, log)
        _save(out, "CDE", {"C": resC, "curC": curC, "D": lofo, "E": geo})
        core.strip(resC).to_csv(out / "C_results.csv", index=False)
        lofo.to_csv(out / "D_lofo.csv", index=False)
    return df
