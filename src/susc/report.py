"""REPORT.md + figures from the analysis pickles.  Headline bullets are generated from the numbers (nothing is hand-written)."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import causal, core, probes
from .core import md, strip
from .prompts import FRAMINGS

SHOW = ["prefix", "target", "regime", "probe_layer", "probe_test_r2", "probe_spearman", "best_observable", "obs_r2", "delta_vs_obs", "obs_lo", "obs_hi",
        "verdict_obs", "best_baseline", "delta_vs_oracle", "oracle_lo", "verdict_oracle"]


def _load(out: Path, name):
    f = out / f"{name}.pkl"
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)


def _pick(res, prefix=None, target=None, regime=None, framing=None):
    for r in res:
        if (prefix is None or r["prefix"] == prefix) and (target is None or r["target"] == target) and (regime is None or r["regime"] == regime) \
                and (framing is None or r.get("framing") == framing):
            return r
    return None


def _line(label, r):
    if r is None:
        return f"- {label}: (not available)"
    v1 = core.verdict(r["delta_vs_obs"], r["obs_lo"], r["probe_test_r2"])
    v2 = core.verdict(r["delta_vs_oracle"], r["oracle_lo"], r["probe_test_r2"])
    return (f"- **{label}**: probe R2 = {r['probe_test_r2']:.3f} (layer {r['probe_layer']}, n_test={r['n_test']}); best observable baseline "
            f"`{r['best_observable']}` R2 = {r['obs_r2']:.3f}; gap {r['delta_vs_obs']:+.3f} [{r['obs_lo']:+.3f}, {r['obs_hi']:+.3f}] -> **{v1}**; "
            f"vs best oracle (`{r['best_baseline']}`) gap {r['delta_vs_oracle']:+.3f} [{r['oracle_lo']:+.3f}, {r['oracle_hi']:+.3f}] -> {v2}")


def _ok(r, oracle=False):
    if r is None:
        return False
    return (core.verdict(r["delta_vs_oracle"], r["oracle_lo"], r["probe_test_r2"]) if oracle else core.verdict(r["delta_vs_obs"], r["obs_lo"], r["probe_test_r2"])) == "PASS"


def scorecard(A, Bres, CDE, tab, trace, dss) -> pd.DataFrame:
    """The pre-declared hypotheses (docs/PROTOCOL.md) evaluated mechanically from the numbers."""
    rows = []

    def add(h, claim, ev, status):
        rows.append({"hypothesis": h, "claim": claim, "evidence": ev, "status": status})
    if A:
        rA = A["res"]
        h1 = _pick(rA, "A[pooled]", "S", "iid")
        add("H1", "S readable from the CLEAN state", f"pooled iid R2={h1['probe_test_r2']:.3f}, gap vs observable {h1['delta_vs_obs']:+.3f} [{h1['obs_lo']:+.3f},{h1['obs_hi']:+.3f}]" if h1 else "n/a",
            "SUPPORTED" if _ok(h1) else "not supported")
        per = {d: _ok(_pick(rA, f"A[{d}]", "S", "iid")) for d in dss}
        lod = {d: _ok(_pick(rA, "A[pooled]", "S", f"lodo:{d}")) for d in dss}
        need = min(3, len(dss))
        if not dss:
            add("H2", "generalises across datasets", "no per-dataset results (too few problems per dataset, e.g. a gate run)", "n/a")
        else:
            ev = (f"per-dataset PASS {sum(per.values())}/{len(dss)} ({', '.join(d for d, v in per.items() if v)}); "
                  f"leave-dataset-out PASS {sum(lod.values())}/{len(dss)} ({', '.join(d for d, v in lod.items() if v)})")
            st = "SUPPORTED" if sum(per.values()) >= need and sum(lod.values()) >= need else ("partly" if sum(per.values()) + sum(lod.values()) > 0 else "not supported")
            add("H2", "generalises across datasets", ev, st)
        h3 = _pick(rA, "A[pooled]", "S", "shift")
        add("H3", "survives easy->hard difficulty shift", f"pooled shift R2={h3['probe_test_r2']:.3f}, gap {h3['delta_vs_obs']:+.3f} [{h3['obs_lo']:+.3f},{h3['obs_hi']:+.3f}]" if h3 else "n/a",
            "SUPPORTED" if _ok(h3) else "not supported")
        if tab is not None and h1 is not None:
            sc = causal.stratified_contrast(tab["df"], h1)
            no_loss = h1["delta_vs_oracle"] >= 0
            add("H4", "more than difficulty", f"difficulty-adjusted contrast {sc['adjusted_diff_S_top_minus_bottom_half']:+.3f} [{sc['lo']:+.3f},{sc['hi']:+.3f}]; vs best oracle gap {h1['delta_vs_oracle']:+.3f} [{h1['oracle_lo']:+.3f},{h1['oracle_hi']:+.3f}]",
                "SUPPORTED" if (sc["lo"] > 0 and no_loss) else ("partly" if sc["lo"] > 0 or no_loss else "not supported"))
        sn = _pick(rA, "A[pooled]", "S_net", "iid")
    else:
        sn = None
    if Bres:
        a = _pick(Bres, "B[auth][pooled]", "follow_auth", "iid")
        pl = _pick(Bres, "B[placebo][pooled]", "follow_plc", "iid")
        if sn is not None and a is not None and pl is not None:
            add("H5", "more than copying", f"S_net R2={sn['probe_test_r2']:.3f} (PASS={_ok(sn)}); follow_plc probe R2={pl['probe_test_r2']:.3f} vs follow_auth {a['probe_test_r2']:.3f}",
                "SUPPORTED" if (_ok(sn) and pl["probe_test_r2"] < a["probe_test_r2"]) else ("partly" if _ok(sn) else "not supported"))
        g = {f: _pick(Bres, f"B[{f}][pooled]", f"gap_{f}", "iid") for f in ("auth", "pressure")}
        if all(g.values()):
            nds = {f: sum(_ok(_pick(Bres, f"B[{f}][{d}]", f"gap_{f}", "iid")) for d in dss) for f in g}
            allok = all(_ok(x) for x in g.values()) and all(v >= min(3, len(dss)) for v in nds.values())
            add("H6", "forced-belief gap readout replicates", f"pooled R2 auth={g['auth']['probe_test_r2']:.3f}, pressure={g['pressure']['probe_test_r2']:.3f}; datasets PASS auth {nds['auth']}/{len(dss)}, pressure {nds['pressure']}/{len(dss)}",
                "SUPPORTED" if allok else ("partly" if any(_ok(x) for x in g.values()) else "not supported"))
    if trace is not None and len(trace):
        t = trace[(trace.scope == "pooled") & trace.target.str.startswith("y_follow") & trace.pos.isin(["p50", "p75", "pend"])]
        win = int(((t.probe_auroc > t.text_auroc) & (t.lo > 0)).sum())
        te = trace[(trace.scope == "pooled") & trace.target.str.startswith("y_follow") & (trace.pos == "pend")]
        sil = te[(te.silent_pos >= 30)]
        silent_ok = bool((sil.silent_probe_auroc > 0.55).any()) if len(sil) else False
        add("H7", "trace monitoring adds over text; works on silent adoption", f"probe beats text with CI>0 in {win}/{len(t)} (cond x position) cells; silent AUROC at pend: " +
            (", ".join(f"{r.cond}={r.silent_probe_auroc:.2f} (n+={int(r.silent_pos)})" for r in te.itertuples()) or "n/a"),
            "SUPPORTED" if (win >= 2 and silent_ok) else ("partly" if (win >= 1 or silent_ok) else "not supported"))
    if tab is not None:
        cr = causal.credulity_link(tab["df"])
        c = cr[(cr.scope == "all") & (cr.a == "S") & (cr.b == "cred")]
        if len(c):
            c = c.iloc[0]
            cos = f"; cosine(S-probe, uplift-probe)={CDE['E']['cosine'].loc['S', 'uplift']:.2f}" if CDE else ""
            add("H8", "one 'trust the hint' latent?", f"Spearman(S, cred)={c.spearman:+.2f} [{c.lo:+.2f},{c.hi:+.2f}]{cos}", "SUPPORTED" if c.lo > 0 else "not supported")
    return pd.DataFrame(rows)


def _plots(out, A, B, C, T, tab, trace):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    made = []
    if A and len(A["cur"]):
        cur = A["cur"]
        cur = cur[(cur.target == "S") & (cur.regime == "iid")]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for sc, g in cur.groupby("scope"):
            ax.plot(g.layer, g.test_r2, label=sc, lw=2.2 if sc == "pooled" else 1.2)
            if sc == "pooled" and "r2_lo" in g:
                ax.fill_between(g.layer, g.r2_lo, g.r2_hi, alpha=0.15)
        ax.axhline(0, c="k", lw=0.6)
        ax.set_xlabel("layer"), ax.set_ylabel("held-out R2 of S (clean prompt-end state)"), ax.legend(fontsize=7), ax.set_title("Layer sweep: latent susceptibility S")
        fig.tight_layout(), fig.savefig(out / "fig1_layer_sweep_S.png", dpi=130), plt.close(fig)
        made.append("fig1_layer_sweep_S.png")
    if tab is not None:
        te = causal.treatment_effects(tab["df"])
        te = te[te.cond.isin(FRAMINGS + ("placebo",))]
        scopes = [s for s in te.scope.unique() if s != "all"]
        fig, ax = plt.subplots(figsize=(8, 4))
        w = 0.8 / max(len(scopes), 1)
        conds = list(FRAMINGS) + ["placebo"]
        for i, sc in enumerate(scopes):
            g = te[te.scope == sc].set_index("cond").reindex(conds)
            ax.bar(np.arange(len(conds)) + i * w, g.follow.values, w, yerr=[g.follow.values - g.follow_lo.values, g.follow_hi.values - g.follow.values], label=sc)
        ax.set_xticks(np.arange(len(conds)) + 0.4 - w / 2), ax.set_xticklabels(conds), ax.set_ylabel("P(solution ends on the planted wrong answer)")
        ax.legend(fontsize=7), ax.set_title("Behavioural adoption by framing (95% CI over problems)")
        fig.tight_layout(), fig.savefig(out / "fig2_adoption.png", dpi=130), plt.close(fig)
        made.append("fig2_adoption.png")
    if trace is not None and len(trace):
        t = trace[(trace.scope == "pooled") & (trace.target.str.startswith("y_follow"))]
        if len(t):
            fig, axs = plt.subplots(1, t.cond.nunique(), figsize=(4.5 * t.cond.nunique(), 3.8), squeeze=False)
            for ax, (cond, g) in zip(axs[0], t.groupby("cond")):
                g = g.sort_values("pos_i")
                ax.plot(g.pos, g.probe_auroc, "o-", label="linear probe (best layer)")
                ax.plot(g.pos, g.text_auroc, "s--", label="best text baseline")
                ax.plot(g.pos, g.oracle_auroc, "^:", label="best oracle baseline")
                ax.axhline(0.5, c="k", lw=0.5), ax.set_title(f"{cond}: will this solution END on the wrong hint?"), ax.set_ylabel("AUROC"), ax.legend(fontsize=6)
            fig.tight_layout(), fig.savefig(out / "fig5_trace_monitor.png", dpi=130), plt.close(fig)
            made.append("fig5_trace_monitor.png")
    if tab is not None:
        bins = causal.difficulty_bins(tab["df"])
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for ds, g in bins.groupby("ds"):
            ax.plot(g.p_clean, g.S, "o-", label=ds)
        ax.set_xlabel("clean accuracy (bin mean)"), ax.set_ylabel("adoption S (bin mean)"), ax.legend(fontsize=7), ax.set_title("Susceptibility vs difficulty (within-dataset bins)")
        fig.tight_layout(), fig.savefig(out / "fig3_difficulty.png", dpi=130), plt.close(fig)
        made.append("fig3_difficulty.png")
    return made


def write(run_dir: Path, log=print) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "analysis"
    A, B1, B2, CDE, T, tab = (_load(out, n) for n in ("A", "B1", "B2", "CDE", "T", "tables"))
    audit = json.load(open(run_dir / "ground_truth_audit.json")) if (run_dir / "ground_truth_audit.json").exists() else []
    Bres = (B1["res"] if B1 else []) + (B2["res"] if B2 else [])
    Bcur = pd.concat([x["cur"] for x in (B1, B2) if x]) if (B1 or B2) else pd.DataFrame()
    trace = T["trace"] if T else None
    L = ["# Hint-adoption susceptibility: probing and monitoring (Qwen2.5-7B-Instruct)\n",
         "Everything below is observational on a frozen model (no activation interventions). Probes are closed-form ridge regressions on frozen hidden states; "
         "layer and lambda are chosen on validation problems; test problems are touched once; CIs are paired bootstraps over test problems. "
         "`verdict_obs` compares against the strongest baseline that uses only what is observable at the prompt end (text features, hint features, the model's own early layers); "
         "`verdict_oracle` against the strongest baseline that also sees gold / measured difficulty / behaviour in other framings. PASS = CI lower bound > 0 and probe R2 > 0.05.\n"]
    # ---- audit
    if audit:
        L.append("## 1. Ground-truth audit (a dataset is admitted only if every check passes on >= 99.5% of problems)\n\n"
                 + md(pd.DataFrame([{**{k: v for k, v in a.items() if k != "fails"}, "fails": json.dumps(a["fails"])} for a in audit])))
    # ---- behaviour
    if tab is not None:
        df = tab["df"]
        te = causal.treatment_effects(df)
        te.to_csv(out / "treatment_effects.csv", index=False)
        sanity = df.groupby("ds").agg(n=("id", "size"), tier=("tier", "first"), clean_acc=("p_clean", "mean"), clean_parse_fail=("none_clean", "mean"),
                                      spontaneous_wrong=("spont_clean", "mean"), S=("S", "mean"), frac_adopting=("adopts", "mean")).reset_index()
        L.append("## 2. Behaviour (labels by sampling; the same problem under every prompt condition)\n\n### 2a. Dataset sanity (G7: the model must do far better than chance with no hint, else adoption is uninterpretable)\n\n"
                 + md(sanity) + "\n### 2b. Treatment effects of the hint (within-problem contrasts; 95% CI over problems)\n\n"
                 + md(te[["scope", "n", "cond", "follow", "follow_lo", "follow_hi", "adopt_net_of_placebo", "net_lo", "net_hi", "acc", "acc_drop_vs_clean", "drop_lo", "drop_hi",
                          "frac_problems_any_adoption", "parse_fail", "gap_nats"]]))
        cl = tab["ceil"]
        L.append("\nFor the `cor(+gain)` row the `acc_drop_vs_clean` columns are NEGATIVE when the correct hint HELPS (gain = -drop).\n")
        L.append("\n### 2c. Label-noise ceilings (max R2 any probe can reach given binomial sampling noise; forced-answer labels have none)\n\n" + md(cl))
    # ---- headline
    dss_present = sorted({r["prefix"][2:-1] for r in (A["res"] if A else []) if r["prefix"].startswith("A[") and r["prefix"] != "A[pooled]"}, key=lambda d: ["gsm8k", "arith", "prog", "math", "arc"].index(d))
    sc_tab = scorecard(A, Bres, CDE, tab, trace, dss_present)
    sc_tab.to_csv(out / "scorecard.csv", index=False)
    L.append("## 3. Scorecard: the pre-declared hypotheses (docs/PROTOCOL.md), evaluated mechanically\n\n" + md(sc_tab) + "\n## 3b. Headline results\n")
    heads = []
    if A:
        rA = A["res"]
        heads += ["**Prospective readout from the CLEAN prompt-end state (no hint present yet)**",
                  _line("S (adoption, sampled) pooled, iid", _pick(rA, "A[pooled]", "S", "iid")),
                  _line("Sgap (adoption, forced no-CoT belief) pooled, iid", _pick(rA, "A[pooled]", "Sgap", "iid")),
                  _line("S_net (beyond placebo copying) pooled, iid", _pick(rA, "A[pooled]", "S_net", "iid")),
                  _line("S pooled, difficulty shift (train easy -> test hard)", _pick(rA, "A[pooled]", "S", "shift"))]
        for ds in ("gsm8k", "math", "arith", "prog", "arc"):
            heads.append(_line(f"S in {ds}, iid", _pick(rA, f"A[{ds}]", "S", "iid")))
        for ds in ("gsm8k", "math", "arith", "prog", "arc"):
            heads.append(_line(f"S leave-dataset-out ({ds} unseen)", _pick(rA, "A[pooled]", "S", f"lodo:{ds}")))
        heads.append(_line("knowledge boundary: p_clean pooled iid", _pick(rA, "A[pooled]", "p_clean", "iid")))
    if Bres:
        heads += ["", "**In-context readout (hinted prompt-end state -> will it follow THIS hint?)**"]
        for f in FRAMINGS:
            heads.append(_line(f"follow[{f}] pooled iid", _pick(Bres, f"B[{f}][pooled]", f"follow_{f}", "iid")))
        for f in FRAMINGS:
            heads.append(_line(f"gap[{f}] pooled iid", _pick(Bres, f"B[{f}][pooled]", f"gap_{f}", "iid")))
        heads.append(_line("copying (placebo prompt) follow_plc pooled iid", _pick(Bres, "B[placebo][pooled]", "follow_plc", "iid")))
        heads.append(_line("credulity (correct-hint prompt) uplift pooled iid", _pick(Bres, "B[cor][pooled]", "uplift", "iid")))
    L.append("\n".join(heads) + "\n")
    # ---- A detail
    if A:
        s = strip(A["res"])
        s.to_csv(out / "A_results.csv", index=False)
        L.append("## 4. Prospective readout (state = clean prompt end) - full table\n\nPer-dataset columns `r2[ds]` are the pooled probe's R2 on each dataset's test rows.\n\n" + md(s[SHOW + [c for c in s.columns if c.startswith("r2[")]]))
    if Bres:
        sb = strip(Bres)
        sb["framing"] = [r.get("framing") for r in Bres]
        sb["auroc_adopt"] = [r.get("auroc_adopt", np.nan) for r in Bres]
        sb["base_rate"] = [r.get("base_rate_adopt", np.nan) for r in Bres]
        sb.to_csv(out / "B_results.csv", index=False)
        L.append("## 5. In-context readout per framing (pooled and per dataset)\n\n" + md(sb[SHOW + ["auroc_adopt", "base_rate"]]))
    if CDE:
        sc = strip(CDE["C"])
        L.append("## 6. Delta-state (hinted minus clean state of the same problem)\n\n" + md(sc[SHOW]))
        D = CDE["D"]
        D.to_csv(out / "D_lofo.csv", index=False)
        L.append("## 7. Leave-one-framing-out transfer (train on three framings, test on the fourth)\n\n" + md(D))
        E = CDE["E"]
        L.append(f"## 8. Geometry and specificity (probes fitted at layer {E['layer']} on pooled data)\n\n"
                 "Cosine similarity of probe weight vectors (raw space):\n\n" + md(E["cosine"].reset_index().rename(columns={"index": ""})) +
                 "\nRank correlation between labels (test rows):\n\n" + md(E["label_spearman"].reset_index().rename(columns={"index": ""})) +
                 "\nRank correlation of each probe's predictions with each label (cross-label specificity):\n\n" + md(E["probe_vs_label"].reset_index().rename(columns={"index": ""})))
    # ---- causal design
    if tab is not None:
        df = tab["df"]
        hd, dr, db, bb, cr = causal.hint_distance(df), causal.difficulty_response(df), causal.difficulty_bins(df), causal.belief_to_behaviour(df), causal.credulity_link(df)
        for n, x in (("hint_distance", hd), ("difficulty_response", dr), ("difficulty_bins", db), ("belief_to_behaviour", bb), ("credulity_link", cr)):
            x.to_csv(out / f"causal_{n}.csv", index=False)
        L.append("## 9. Causal-by-design analyses (inputs randomised or controlled by us; no activation interventions)\n\n"
                 "### 9a. Dose-response to the randomised wrong value (offset from gold / option position)\n\n" + md(hd) +
                 "\n### 9b. Susceptibility vs difficulty (arith/prog difficulty is set by the generator)\n\n" + md(dr) + "\n" + md(db) +
                 "\n### 9c. Does the forced (no-CoT) belief shift carry the behavioural adoption? (observational given the randomised hint)\n\n" + md(bb) +
                 "\n### 9d. Is wrong-hint susceptibility the same latent as credulity toward a correct hint? (Spearman, 95% CI)\n\n" + md(cr))
        if A:
            r = _pick(A["res"], "A[pooled]", "S", "iid")
            if r is not None:
                mc = causal.monitor_curves(df, r)
                sc2 = causal.stratified_contrast(df, r)
                mc.to_csv(out / "monitor_curves.csv", index=False)
                L.append("### 9e. Monitoring operating points (held-out problems; flag the top-b by predicted S)\n\n" + md(mc) +
                         f"\n### 9f. Difficulty-adjusted contrast: within (dataset x clean-accuracy tercile) strata, realised S of the top vs bottom half by predicted S\n\n"
                         f"difference = {sc2['adjusted_diff_S_top_minus_bottom_half']:+.3f} [{sc2['lo']:+.3f}, {sc2['hi']:+.3f}] (n={sc2['n']}). A probe that only reads difficulty would give ~0.\n")
    # ---- trace
    if trace is not None and len(trace):
        trace.to_csv(out / "trace_results.csv", index=False)
        L.append("## 10. Monitoring along the reasoning trace (one sampled solution per problem and condition)\n\n"
                 "p0 = pre-generation state, p25/p50/p75 = after that fraction of the solution, pend = just before the final \\boxed. `text_auroc` = best observable text baseline "
                 "(partial solution, incl. 'is the hint number already mentioned'); `oracle_auroc` additionally sees gold.\n\n"
                 + md(trace[trace.scope == "pooled"][["cond", "target", "pos", "probe_layer", "n_test", "pos_rate", "probe_auroc", "text_auroc", "oracle_auroc", "probe_r2", "delta_r2_vs_text", "lo", "hi"]])
                 + "\n### 10b. Silent adoption: restricted to trajectories whose text so far does NOT contain the hint value (a text monitor has nothing to read there)\n\n"
                 + md(trace[(trace.scope == "pooled") & trace.target.str.startswith("y_follow")][["cond", "pos", "silent_n", "silent_pos", "silent_probe_auroc", "silent_text_auroc"]])
                 + "\nPer dataset:\n\n" + md(trace[trace.scope != "pooled"][["scope", "cond", "target", "pos", "n_test", "pos_rate", "probe_auroc", "text_auroc", "oracle_auroc"]]))
    figs = _plots(out, A, Bres, CDE, T, tab, trace)
    if figs:
        L.append("## Figures\n\n" + "\n".join(f"![{f}]({f})" for f in figs) + "\n")
    L.append("## Limitations and how to read this\n\n"
             "- One model (Qwen2.5-7B-Instruct). A second model is a one-line change (`MODEL_ID`) and is the first replication to run.\n"
             "- Sampled labels have binomial noise (see 2c); forced-answer gaps do not, but measure belief without reasoning.\n"
             "- 'Causal' here means design-based: the hint, its framing, its offset/position and (synthetic) difficulty are assigned by us. Nothing here shows the model USES a probe direction; that needs the steering study (deferred).\n"
             "- MATH is restricted to integer answers (unique surface form) and Qwen2.5 saw MATH-style data in training; ARC is expert-keyed (tier B).\n"
             "- Probe R2 below ~0.05, or CIs that include 0, are reported as such; do not read weak/FAIL rows as evidence of absence at larger scale.\n")
    p = out / "REPORT.md"
    p.write_text("\n".join(L))
    log(f"report -> {p}")
    return p
