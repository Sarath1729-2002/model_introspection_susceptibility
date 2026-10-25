"""Plant a known linear signal in fabricated states/samples (bypassing the model) and check that the ANALYSIS recovers it - and does not
'recover' it where there is none (permuted labels, difficulty-only signal)."""
import json

import numpy as np
import pytest

from susc import bench, causal, core, monitor, report, study
from susc.prompts import CONDS, FRAMINGS, TRACE_CONDS
from test_pipeline import fake_loaders  # noqa: F401

D, NL = 24, 12
LATE = (np.arange(NL) >= 9)[None, :, None].astype(np.float32)          # the signal lives only in layers >= 9: the early-layer baselines (H4, H8) cannot see it


def _loaders(n):
    import pandas as pd
    rng = np.random.RandomState(0)
    gs = {s: [{"question": f"Tom has {a} apples and gets {b} more ({s}{i}).", "answer": f"s <<{a}+{b}={a + b}>>\n#### {a + b}"}
              for i, (a, b) in enumerate(zip(rng.randint(2, 40, n), rng.randint(2, 40, n)))] for s in ("train", "test")}
    mth = {s: pd.DataFrame([{"problem": f"What is {i}+{i} ({s})?", "solution": f"$\\boxed{{{2 * i}}}$", "level": f"Level {1 + i % 5}", "type": "algebra"} for i in range(n)])
           for s in ("train", "test")}
    arc = {s: [{"id": f"{s}{i}", "question": f"Which is largest ({s}{i})?", "choices": {"label": list("ABCD"), "text": list("abcd")}, "answerKey": "ABCD"[i % 4]}
               for i in range(n)] for s in ("train", "validation", "test")}
    return {"gsm8k": lambda: gs, "math": lambda: mth, "arc": lambda split: arc[split]}


def fabricate(tmp, n=160, signal=True, seed=0):
    probs, _ = bench.build(tmp, {"gsm8k": n, "math": n, "arc": n, "arith": n, "prog": n}, loaders=_loaders(n), log=lambda *_: None)
    rng = np.random.RandomState(seed)
    u = rng.randn(D); u /= np.linalg.norm(u)                                      # susceptibility direction  # noqa: E702
    v = rng.randn(D); v /= np.linalg.norm(v)                                      # difficulty direction (independent)  # noqa: E702
    w = rng.randn(D); w /= np.linalg.norm(w)                                      # trace direction  # noqa: E702
    Ns = 8
    by_ds = {}
    for p in probs:
        by_ds.setdefault(p["ds"], []).append(p)
    (tmp / "samples").mkdir(exist_ok=True), (tmp / "states").mkdir(exist_ok=True), (tmp / "trace").mkdir(exist_ok=True)   # noqa: E702
    for ds, ps in by_ds.items():
        N = len(ps)
        z = rng.randn(N)                                                          # latent susceptibility
        dif = rng.randn(N)                                                        # latent difficulty
        X = {c: rng.randn(N, NL, D).astype(np.float32) * 0.5 for c in CONDS}
        for c in CONDS:
            X[c] += LATE * (2.0 * z[:, None, None] * u * (signal) + 1.5 * dif[:, None, None] * v)
        for k, c in enumerate(FRAMINGS):
            X[c] += LATE * 0.7 * (k + 1) * w                                      # hinted prompts look different from clean ones
        sus = 1 / (1 + np.exp(-(1.8 * z - 0.7))) if signal else np.full(N, 0.3)
        pcl = 1 / (1 + np.exp(1.2 * dif - 0.5))
        rows, pack = [], {"ids": np.array([p["id"] for p in ps])}
        for c in CONDS:
            lpg = -2 - dif - 0.3 * rng.randn(N) + (1.0 if c == "cor" else 0.0)
            lph = lpg + 3 * (z if signal else 0) - 4 - 0.3 * rng.randn(N)
            if c in ("clean", "cor"):
                lph = lpg - 5 + 0.2 * rng.randn(N)
            pack[f"{c}_states"] = X[c].astype(np.float16)
            pack[f"{c}_lpg"], pack[f"{c}_lph"], pack[f"{c}_ntok"] = lpg, lph, np.full(N, 30, dtype=np.int32)
            for i, p in enumerate(ps):
                if c == "clean":
                    pw, pg = 0.01, pcl[i]
                elif c == "cor":
                    pw, pg = 0.0, min(0.99, pcl[i] + 0.25 * (sus[i] if signal else 0.3))
                elif c == "placebo":
                    pw, pg = 0.05 * sus[i], pcl[i]
                else:
                    pw, pg = sus[i] * (0.5 + 0.15 * FRAMINGS.index(c)) * (1 if signal else 1), pcl[i] * (1 - sus[i] * 0.3)
                nw = rng.binomial(Ns, min(pw, 0.99))
                ng = rng.binomial(Ns - nw, min(pg, 1.0))
                ans = [p["hint"]] * nw + [p["gold"]] * ng + [None] * (Ns - nw - ng)
                rng.shuffle(ans)
                follow0 = ans[0] == p["hint"]
                said = follow0 and rng.rand() < 0.4                       # most followers adopt SILENTLY: the hint value never appears in their text
                txt = "Let us think. " + ("x" * 20) + (f" maybe {p['hint']} " if said else " ") + "\\boxed{" + str(ans[0]) + "}"
                rows.append({"id": p["id"], "ds": ds, "cond": c, "n": Ns, "n_gold": ng, "n_wrong": nw, "n_none": Ns - nw - ng, "ent": 0.5, "mean_chars": 100.0,
                             "answers": ans, "t0": txt, "hv": None})
        np.savez(tmp / "states" / f"x00_{ds}_0000.npz", **pack)
        with open(tmp / "samples" / f"s00_{ds}_0000.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        # trace: the label is whether sample 0 follows; the signal grows with position (early positions carry little)
        tp = {"ids": pack["ids"], "layers": np.array([4])}
        for c in TRACE_CONDS:
            lab = np.array([r["answers"][0] == r["answers"][0] and r["answers"][0] == ps[j]["hint"] for j, r in enumerate([r for r in rows if r["cond"] == c])], float)
            H = rng.randn(N, 5, 1, D).astype(np.float32) * 0.5
            for pos in range(5):
                H[:, pos, 0] += (pos / 4.0) * 2.5 * lab[:, None] * w
            tp[f"h_{c}"], tp[f"frac_{c}"], tp[f"len_{c}"] = H.astype(np.float16), np.tile([0, .25, .5, .75, 1.0], (N, 1)).astype(np.float32), np.full(N, 40, np.int32)
        np.savez(tmp / "trace" / f"t00_{ds}_0000.npz", **tp)
    return probs


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("planted")
    fabricate(tmp, signal=True)
    for block in ("tables", "A", "B1", "B2", "CDE", "T"):
        study.compute(tmp, block, layers=[2, 6, 9, 10, 11] if block not in ("tables", "T") else None, log=lambda *_: None)
    return tmp


def test_probe_recovers_planted_susceptibility(planted):
    A = report._load(planted / "analysis", "A")
    r = report._pick(A["res"], "A[pooled]", "S", "iid")
    assert r["probe_test_r2"] > 0.35 and r["delta_vs_obs"] > 0.2 and r["obs_lo"] > 0 and core.verdict(r["delta_vs_obs"], r["obs_lo"], r["probe_test_r2"]) == "PASS"
    assert r["auroc_adopt"] > 0.8 and r["auroc_adopt"] > r["auroc_adopt_obs_baseline"]
    for ds in ("gsm8k", "arith", "arc"):
        assert report._pick(A["res"], f"A[{ds}]", "S", "iid")["probe_test_r2"] > 0.2
    lo = report._pick(A["res"], "A[pooled]", "S", "lodo:prog")
    assert lo["probe_test_r2"] > 0.2 and set(lo["_ds"]) == {"prog"}                  # transfers to a dataset never seen in fitting
    # difficulty is a different direction: p_clean is also readable, and it is NOT beaten by the difficulty-only oracle trivially
    assert report._pick(A["res"], "A[pooled]", "p_clean", "iid")["probe_test_r2"] > 0.3
    assert report._pick(A["res"], "A[pooled]", "S", "iid")["probe_layer"] >= 9


def test_in_context_and_lofo(planted):
    B = report._load(planted / "analysis", "B1")["res"]
    r = report._pick(B, "B[auth][pooled]", "gap_auth", "iid", "auth")
    assert r["probe_test_r2"] > 0.3
    D_ = report._load(planted / "analysis", "CDE")["D"]
    assert (D_[D_.scope == "pooled"].spearman > 0.5).all() and (D_[D_.scope == "pooled"].probe_r2 > 0.0).all()   # R2 also pays for the scale shift between framings; rank correlation does not
    E = report._load(planted / "analysis", "CDE")["E"]
    # susceptibility direction ~ credulity/uplift (same planted latent), distinct from difficulty direction
    assert E["cosine"].loc["S", "Sgap"] > 0.3 and abs(E["cosine"].loc["S", "p_clean"]) < 0.5
    assert E["probe_vs_label"].loc["probe[S]", "label[S]"] > 0.4


def test_causal_and_monitoring(planted):
    df = report.pickle.load(open(planted / "analysis" / "tables.pkl", "rb"))["df"]
    te = causal.treatment_effects(df)
    row = te[(te.scope == "all") & (te.cond == "auth")].iloc[0]
    assert row.follow > 0.1 and row.adopt_net_of_placebo > 0 and row.net_lo > 0
    A = report._load(planted / "analysis", "A")
    r = report._pick(A["res"], "A[pooled]", "S", "iid")
    mc = causal.monitor_curves(df, r)
    pr = mc[(mc.scorer == "probe") & (mc.budget == 0.2)].iloc[0]
    assert pr.recall_of_adopters > pr.budget + 0.15 and pr.lift_in_mean_S > 1.3
    sc = causal.stratified_contrast(df, r)
    assert sc["adjusted_diff_S_top_minus_bottom_half"] > 0.05 and sc["lo"] > 0         # reads susceptibility, not only difficulty
    assert causal.credulity_link(df).query("scope=='all' and a=='S' and b=='cred'").spearman.iloc[0] > 0.2


def test_trace_monitoring_grows_with_position(planted):
    T = report._load(planted / "analysis", "T")["trace"]
    g = T[(T.scope == "pooled") & (T.cond == "auth") & (T.target == "y_follow_auth")].sort_values("pos_i")
    assert g.probe_auroc.iloc[-1] > 0.85 and g.probe_auroc.iloc[-1] > g.probe_auroc.iloc[0] + 0.15
    assert g.probe_auroc.iloc[-1] > g.text_auroc.iloc[-1] - 0.02 or g.probe_auroc.iloc[-1] > 0.9


def test_report_renders(planted):
    p = report.write(planted, log=lambda *_: None)
    txt = p.read_text()
    sc = report.pd.read_csv(planted / "analysis" / "scorecard.csv")
    assert {"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"} <= set(sc.hypothesis) and (sc.status.isin(["SUPPORTED", "partly", "not supported"])).all()
    assert sc.set_index("hypothesis").loc["H1", "status"] == "SUPPORTED" and sc.set_index("hypothesis").loc["H7", "status"] in ("SUPPORTED", "partly")
    T = report._load(planted / "analysis", "T")["trace"]
    assert {"silent_n", "silent_probe_auroc"} <= set(T.columns)
    for needle in ("Ground-truth audit", "Scorecard", "Headline results", "PASS", "Monitoring operating points", "Monitoring along the reasoning trace", "Geometry and specificity"):
        assert needle in txt, needle
    assert (planted / "analysis" / "fig5_trace_monitor.png").exists()


def test_no_signal_no_pass(tmp_path):
    fabricate(tmp_path, signal=False, seed=3)
    for block in ("A",):
        study.compute(tmp_path, block, layers=[2, 6, 9, 10, 11], log=lambda *_: None)
    A = report._load(tmp_path / "analysis", "A")
    r = report._pick(A["res"], "A[pooled]", "S", "iid")
    # with S independent of the states the probe must not beat the baselines or reach R2 > 0.05 with a positive CI
    assert core.verdict(r["delta_vs_obs"], r["obs_lo"], r["probe_test_r2"]) != "PASS" and r["probe_test_r2"] < 0.1
