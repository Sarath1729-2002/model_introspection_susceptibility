"""THE PAPER-STYLE REPORT.  One document over all (model, dataset) cells, written in the vocabulary of 'How Much is Left?' (floors, ladder of predictors, selectivity controls, sampling ceiling,
cross-dataset transfer matrix, steering with a quality check), plus the natural-error test.

analyze_cell(...)  per cell, offline (needs the cell's outputs): robust.offline (C1-C7 + paper metrics), istear.analyze (C8), stoch.analyze, natural.analyze, lenref.analyze -> <mega>/<name>/analysis/cell.json
write(specs, out_dir) over cells -> PAPER_REPORT.md, paper_*.csv, figures.  specs = [{"name","model","ds","run_dir","mega_dir"}, ...]
Cross-cell rule (fixed in docs/PAPER.md before the runs):  REPLICATES  = every cell reaches ROBUST (diagnostic) or better;  CAUSAL REPLICATES = C8 SUPPORTED in all but at most one cell (and in at least one);  otherwise MIXED / NOT REPLICATED with the failing cells listed.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import core


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _clean(o.to_dict("records"))
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def analyze_cell(run_dir: Path, mega_dir: Path, ds: str, name: str = "robust", log=print, recompute: bool = False, B: int = 500) -> dict:
    import traceback
    from . import istear, lenref, natural, robust, stoch
    run_dir, mega_dir = Path(run_dir), Path(mega_dir)
    base = mega_dir / name
    out = {"ds": ds}
    for key, fn in (("offline", lambda: json.loads((base / "offline.json").read_text()) if ((base / "offline.json").exists() and not recompute) else robust.offline(run_dir, mega_dir, log, B, name, ds)),
                    ("steer", lambda: istear.analyze(base / "steer")),
                    ("stoch", lambda: stoch.analyze(run_dir, mega_dir, base / "stoch", ds, log=log) if list((base / "stoch").glob("k*_*.jsonl")) else {"status": "n/a", "why": "stage not run"}),
                    ("natural", lambda: natural.analyze(run_dir, mega_dir, base / "natural", ds, log=log) if list((base / "natural").glob("n*_*.jsonl")) else {"status": "n/a", "why": "stage not run"}),
                    ("lenref", lambda: lenref.analyze(run_dir, ds, log))):
        try:
            out[key] = _clean(fn())
        except Exception:                                                                   # noqa: BLE001  one failing analysis must not kill the cell
            log(traceback.format_exc())
            out[key] = {"status": "error", "why": traceback.format_exc()[-600:]}
    (base / "analysis").mkdir(parents=True, exist_ok=True)
    (base / "analysis" / "cell.json").write_text(json.dumps(out, indent=1))
    try:
        robust.write_report(run_dir, mega_dir, log, B, False, name, ds)
    except Exception:                                                                       # noqa: BLE001
        log(traceback.format_exc())
    return out


# ----------------------------------------------------------------------------- the report
def _md(rows, cols=None, fmt="{:.3f}"):
    df = pd.DataFrame(rows)
    df = df.apply(lambda c: c.map(lambda v: v.replace("|", "/").replace("\n", " ") if isinstance(v, str) else v))     # '|' inside a cell would break the table
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return core.md(df, fmt=fmt) if len(df) else "(none)\n"


def _grid(cells: Dict[str, dict]):
    from . import robust
    cards = {}
    for n, c in cells.items():
        off = c.get("offline", {})
        cards[n] = robust.scorecard(off, c.get("steer") if c.get("steer", {}).get("status") not in (None, "n/a", "error") else None)
    return cards


def cross_cell(cards: Dict[str, list]) -> dict:
    from . import robust
    ov = {n: robust.overall(r) for n, r in cards.items()}
    diag = [n for n, o in ov.items() if o["verdict"] in ("ROBUST (diagnostic)", "ROBUST + CAUSAL")]
    caus = [n for n, o in ov.items() if o["verdict"] == "ROBUST + CAUSAL"]
    c8 = [n for n, r in cards.items() if {x["check"]: x["ok"] for x in r}.get("C8") is True]
    bad = [n for n in cards if n not in diag]
    n = len(cards)
    if not n:
        return {"verdict": "NO CELLS", "why": "", "overall": ov}
    if not bad and len(c8) >= max(1, n - 1):
        v, why = "CAUSAL REPLICATES", f"every cell reaches at least ROBUST (diagnostic); C8 SUPPORTED in {len(c8)}/{n} cells"
    elif not bad:
        v, why = "REPLICATES (diagnostic)", f"every cell reaches ROBUST (diagnostic); C8 SUPPORTED in only {len(c8)}/{n} cells ({', '.join(c8) or 'none'})"
    else:
        v, why = "MIXED" if len(diag) >= 1 else "NOT REPLICATED", f"ROBUST (diagnostic) or better in {len(diag)}/{n} cells; failing: {', '.join(bad)}"
    return {"verdict": v, "why": why, "overall": ov, "diag": diag, "c8": c8}


def load_cells(specs: List[dict]) -> Dict[str, dict]:
    cells = {}
    for s in specs:
        f = Path(s["mega_dir"]) / s.get("robust", "robust") / "analysis" / "cell.json"
        if f.exists():
            cells[s["name"]] = {**json.loads(f.read_text()), "spec": s}
    return cells


def _fig_ladder(cells, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(cells)
    fig, ax = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6), sharey=True, squeeze=False)
    for a, n in zip(ax[0], names):
        lad = [r for r in cells[n].get("offline", {}).get("paper", {}).get("ladder", []) if r["regime"] == "iid"]
        if not lad:
            a.set_title(f"{n}\n(no data)")
            continue
        a.barh([r["predictor"].split(" (")[0] for r in lad][::-1], [r["auroc"] for r in lad][::-1], color=["#2563eb" if "probe" in r["predictor"] else "#9ca3af" for r in lad][::-1])
        a.set_xlim(0.45, 1.0)
        a.set_title(n, fontsize=10)
        a.set_xlabel("AUROC (iid test)")
    fig.tight_layout()
    fig.savefig(out / "fig_ladder.png", dpi=140)
    plt.close(fig)


def _fig_dose(cells, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in cells if cells[n].get("steer", {}).get("dose")]
    if not names:
        return
    fig, ax = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4), sharey=True, squeeze=False)
    for a, n in zip(ax[0], names):
        d = pd.DataFrame(cells[n]["steer"]["dose"])
        for L, g in d.groupby("layer"):
            g = g.sort_values("alpha")
            a.plot(g.alpha, g.probe_d_flip, marker="o", label=f"probe L{int(L)}")
            a.plot(g.alpha, g.diff_d_flip, marker="x", ls="--", label=f"mean-diff L{int(L)}")
            a.plot(g.alpha, g.rand_mean_abs_d_flip, marker=".", ls=":", color="gray")
        a.axhline(0, color="k", lw=0.5)
        a.set_title(n, fontsize=10)
        a.set_xlabel("alpha (sd of probe projection)")
    ax[0][0].set_ylabel("change in follow-the-error rate")
    ax[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_dose.png", dpi=140)
    plt.close(fig)


def write(specs: List[dict], out_dir: Path, log=print, transfer: bool = True) -> Path:
    import traceback
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells(specs)
    cards = _grid(cells)
    cc = cross_cell(cards)
    L = ["# Error propagation is linearly readable at the moment a wrong number is written - and the direction is causally used", "",
         "Pre-registered cross-cell rule (docs/PAPER.md): REPLICATES = every cell reaches ROBUST (diagnostic) or better; CAUSAL REPLICATES = additionally C8 SUPPORTED in all but at most one cell (and at least one).", "",
         f"## Verdict: **{cc['verdict']}** - {cc['why']}", ""]
    if cards:
        grid = []
        for chk in [f"C{i}" for i in range(1, 9)]:
            row = {"check": chk, "what": next((x["what"] for r in cards.values() for x in r if x["check"] == chk), "")}
            for n, r in cards.items():
                ok = {x["check"]: x["ok"] for x in r}.get(chk)
                row[n] = {True: "PASS", False: "FAIL", None: "n/a"}[ok]
            grid.append(row)
        grid.append({"check": "overall", "what": "", **{n: o["verdict"] for n, o in cc["overall"].items()}})
        L += [_md(grid), ""]
    # 1 setup
    meta = [{"cell": n, **{k: v for k, v in c.get("offline", {}).get("meta", {}).items() if k != "n_problems_by_split"}, "rows": c.get("offline", {}).get("n_rows"), "problems": c.get("offline", {}).get("n_problems")} for n, c in cells.items()]
    L += ["## 1. Data", "", "Rows = injected wrong numbers (n1: c -> c+1, n2: c -> c+max(10, c//10)) in the model's own greedy solution; label `flip` = the final answer after a greedy continuation differs from the unedited answer. "
          "Continuations that did not terminate within the budget are excluded (right-censored, as in the paper's naturally-terminated filter); the exclusion count is reported.", "", _md(meta)]
    # 2 ladder
    lad = [{"cell": n, **r} for n, c in cells.items() for r in c.get("offline", {}).get("paper", {}).get("ladder", [])]
    L += ["## 2. The ladder of predictors (paper Table 1/6 vocabulary)", "",
          "floor = constant (AUROC 0.5, Brier-skill 0, MAE ratio 1 by construction). `pre-edit state carried forward` is the analogue of their exact-countdown predictor: what the model's own state already said at the end of the step, before the error was written. "
          "`mae_over_floor` = MAE of the clipped prediction over MAE of the constant-median predictor (< 1 beats the floor).", "", _md(lad, ["cell", "regime", "predictor", "auroc", "r2", "brier_skill", "mae_over_floor", "tpr10", "n_test"])]
    # 3 checks C1-C7 detail
    det = [{"cell": n, "check": x["check"], "result": {True: "PASS", False: "FAIL", None: "n/a"}[x["ok"]], "detail": x["detail"]} for n, r in cards.items() for x in r]
    L += ["## 3. Robustness checks C1-C8 in detail", "", _md(det), ""]
    # 4 controls / seeds
    ctl = [{"cell": n, **r} for n, c in cells.items() for r in c.get("offline", {}).get("paper", {}).get("controls", [])]
    sd = [{"cell": n, **{k: v for k, v in r.items() if k != "values"}} for n, c in cells.items() for r in c.get("offline", {}).get("paper", {}).get("seeds", [])]
    L += ["## 4. Selectivity controls and seed stability (paper A.12)", "", "Shuffled labels go through the whole pipeline (layer and lambda by CV). `within-problem` keeps each problem's label multiset, so a probe that only knows which problems flip a lot would still score.", "",
          _md(ctl), "Fold-seed stability (CV folds seeds 0, 1, 2):", "", _md(sd)]
    prop = [{"cell": n, **r} for n, c in cells.items() for r in c.get("offline", {}).get("paper", {}).get("prop", [])]
    L += ["## 5. The continuous label: how far the error propagates (paper units: MAE vs the constant-median floor)", "", _md(prop), ""]
    # 6 ceiling
    st = [{"cell": n, **{k: v for k, v in c["stoch"].items() if k in ("n_rows", "n_samp", "best_observable", "share_stochastic_rows", "captured_vs_floor", "captured_vs_floor_sampled")}, **{f"auroc_{k}": v for k, v in c["stoch"].get("auroc", {}).items()}}
          for n, c in cells.items() if c.get("stoch", {}).get("status") == "ok"]
    L += ["## 6. Sampling-noise ceiling (paper A.13)", "", "The label is one greedy continuation; the model's own follow probability is estimated from sampled continuations of the same edited prefix. `oracle_*` is a predictor that knows that probability. "
          "`captured_vs_floor` = (AUROC_probe - 0.5) / (AUROC_oracle - 0.5).", "", _md(st) if st else "(stage not run for these cells)\n"]
    # 7 transfer
    tr_rows = []
    if transfer:
        try:
            from . import matrix
            from .inject import prepare
            by_model = {}
            for s in specs:
                if s["name"] in cells:
                    by_model.setdefault(s["model"], []).append(s)
            for mdl, ss in by_model.items():
                if len(ss) < 2:
                    continue
                Ps = {}
                for s in ss:
                    P, why = prepare(Path(s["run_dir"]), Path(s["mega_dir"]), s["ds"])
                    if P is not None:
                        Ps[s["ds"]] = P
                if len(Ps) >= 2:
                    tr_rows += [{"model": mdl, **r} for r in matrix.transfer(Ps, log)]
        except Exception:                                                                    # noqa: BLE001
            log(traceback.format_exc())
    L += ["## 7. Cross-dataset transfer (paper section 5)", "", "A probe fitted on one dataset's injected rows, frozen, evaluated on the other dataset's iid test rows (same model). Floors: constant (AUROC 0.5) and the target dataset's best observable baseline. "
          "`brier_skill_raw` keeps the source intercept, so it also shows calibration drift.", "", _md(tr_rows) if tr_rows else "(needs two datasets of the same model)\n"]
    if tr_rows:
        pd.DataFrame(tr_rows).to_csv(out_dir / "paper_transfer.csv", index=False)
    # 8 natural
    nat = []
    for n, c in cells.items():
        z = c.get("natural", {})
        if z.get("status") != "ok":
            nat.append({"cell": n, "status": z.get("status", "n/a"), "why": z.get("why", "")})
            continue
        for scope in ("unseen_problems", "all_problems"):
            q = z.get(scope, {})
            if "probe" not in q:
                nat.append({"cell": n, "scope": scope, "n_slips": q.get("n_slips"), "note": q.get("note", "")})
                continue
            row = {"cell": n, "scope": scope, "n_slips": q["n_slips"], "n_solutions": q["n_solutions"], "share_slips_that_matter": q["share_matters"], "probe_auroc": q["probe"][0], "probe_lo": q["probe"][1], "probe_hi": q["probe"][2]}
            for bn, bv in q["baselines"].items():
                row[f"auroc[{bn}]"] = bv[0]
            sv = q.get("slip_vs_control", {})
            row.update({"slip_vs_ctrl_cohen_d": sv.get("cohen_d"), "slip_vs_ctrl_p": sv.get("p_signflip"), "slip_vs_ctrl_auroc": sv.get("auroc_slip_vs_ctrl")})
            nat.append(row)
    L += ["## 8. Natural errors (the model's own arithmetic slips)", "", "Slips are found deterministically (a complete integer two-operand equation whose stated result is wrong). `matters` = the final answer changes when the slip is corrected and the model continues. "
          "The injection-trained probe is applied to the state at the last token of the slip and compared with relative position and relative size of the slip.", "", _md(nat) if nat else "(stage not run)\n"]
    # 9 steering
    stw = []
    for n, c in cells.items():
        z = c.get("steer", {})
        if z.get("status") in (None, "n/a", "error"):
            continue
        tab = pd.DataFrame(z["table"])
        bl = z.get("layers", {})
        bestL = min(bl, key=lambda k: {"SUPPORTED": 0, "PARTIAL": 1, "INCONCLUSIVE": 2, "NEGATIVE": 3}[bl[k]["status"]]) if bl else None
        if bestL is None:
            continue
        base = float(tab[tab.cond == "base"].flip_rate.iloc[0])
        for r in tab[(tab.family == f"probe_L{bestL}") | (tab.cond == "base")].sort_values("alpha").to_dict("records"):
            stw.append({"cell": n, "layer": bestL, "alpha": r["alpha"], "follow_rate": r["flip_rate"], "ratio_vs_alpha0": r["flip_rate"] / base if base else np.nan, "d_follow": r["d_flip_vs_base"],
                        "non_terminated": 1 - r["finished"], "unparsable": 1 - r["parsed"], "nll_under_unsteered": r.get("nll_unsteered"), "p_signflip": r.get("p_signflip")})
    L += ["## 9. Steering at the injected token (paper Table 2 / A.17 vocabulary)", "", "Dose in sd of the probe projection, added at the last token of the injected number, prefill only. `ratio_vs_alpha0` = follow rate / alpha=0 follow rate; `nll_under_unsteered` = quality check "
          "(mean per-token NLL of the steered continuation under the unsteered model); `p_signflip` = paired problem-clustered sign-flip test vs alpha=0.", "", _md(stw) if stw else "(not run)\n"]
    slopes = [{"cell": n, "family": k, "slope_per_sd": v[0], "lo95": v[1], "hi95": v[2]} for n, c in cells.items() for k, v in (c.get("steer", {}).get("slopes") or {}).items()]
    L += ["Slopes (flip rate per sd over |alpha| <= 16; random directions are the selectivity control):", "", _md(slopes, fmt="{:.4f}")]
    # 10 reference
    ref = [{"cell": n, **{k: v for k, v in c["lenref"].items() if k not in ("status",)}} for n, c in cells.items() if c.get("lenref", {}).get("status") == "ok"]
    L += ["## 10. Reference: the paper's own claim re-measured on these cells (prompt-end total length, their Table 4/5, A.12, A.13)", "",
          "Sanity anchor: if this pipeline reproduces the published phenomenon on the same models and datasets with the same floors, differences elsewhere are not pipeline artefacts.", "", _md(ref) if ref else "(gen+states not available)\n"]
    # 11 layers
    cu = []
    for n, c in cells.items():
        for nm, d in c.get("offline", {}).get("curves", {}).items():
            cu += [{"cell": n, "state": nm, "layer": int(k), "cv_r2": v} for k, v in d.items()]
    if cu:
        L += ["## 11. CV R2 by layer", "", core.md(pd.DataFrame(cu).pivot_table(index=["cell", "layer"], columns="state", values="cv_r2").reset_index().assign(cell=lambda d: d.cell.str.replace("|", "/", regex=False))), ""]
    L += ["## 12. What this does and does not show", "",
          "- Shows: a linear readout at the last token of a wrong number predicts whether the final answer follows it, beyond position, text, the model's own surprise, an early layer and the pre-edit state; "
          "the effect is checked by shuffled labels, subsets and a causal intervention, in the cells above.",
          "- Does not show: that the model's unperturbed computation consults the direction (steering shows causal usability, as in the paper); that the label separates 'recovered' from 'step not load-bearing' (the within-problem and last-step checks reduce this, not remove it); "
          "anything about models or datasets outside the cells; anything about errors of other kinds than +1 / +10% number edits except through the natural-error test.", ""]
    _fig_ladder(cells, out_dir)
    _fig_dose(cells, out_dir)
    p = out_dir / "PAPER_REPORT.md"
    p.write_text("\n".join(L))
    pd.DataFrame(lad).to_csv(out_dir / "paper_ladder.csv", index=False)
    pd.DataFrame(det).to_csv(out_dir / "paper_scorecard.csv", index=False)
    log(f"wrote {p}")
    return p
