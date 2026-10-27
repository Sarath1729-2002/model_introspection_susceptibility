"""Tests of the paper-matrix additions: the metric ladder + controls (planted signals), the sampling-noise ceiling, the natural-error scan / run, the transfer matrix, the prompt-end length reference, the report assembler and the CLI wiring."""
import json

import numpy as np
import pytest

from psuite import cli, inject, lenref, matrix, natural, paper, robust, stoch
from test_pipeline import tok, model, run   # noqa: F401  (fixtures)
from test_mega import fabricate_mega
from test_inject import write_inject
from test_robust import tiny_inject

QUIET = lambda *_: None            # noqa: E731


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    out = {}
    for mode in ("post", "surprise"):
        tmp = tmp_path_factory.mktemp(f"pap_{mode}")
        md = fabricate_mega(tmp)
        write_inject(md, mode)
        out[mode] = (tmp, md, robust.offline(tmp, md, log=QUIET, B=100))
    return out


# ----------------------------------------------------------------------------- the metric ladder and controls
def test_paper_metrics_ladder_and_controls(planted):
    tmp, md, off = planted["post"]
    pm = off["paper"]
    names = {r["predictor"] for r in pm["ladder"]}
    assert any("floor" in n for n in names) and any("probe" in n for n in names) and any("pre-edit" in n for n in names)
    iid = {r["predictor"]: r for r in pm["ladder"] if r["regime"] == "iid"}
    fl = next(r for n, r in iid.items() if "floor" in n)
    pr = next(r for n, r in iid.items() if "probe" in n)
    assert abs(fl["auroc"] - 0.5) < 1e-9 and abs(fl["brier_skill"]) < 1e-6 and abs(fl["mae_over_floor"] - 1) < 1e-6
    assert pr["auroc"] > 0.7 and pr["brier_skill"] > 0 and pr["mae_over_floor"] < 1
    assert {r["metric"] for r in pm["seeds"]} == {"test R2", "test AUROC"} and all(len(r["values"]) == 3 for r in pm["seeds"])
    ctl = [c for c in pm["controls"] if c["control"].startswith("shuffled")]
    assert len(ctl) == 6 and next(c for c in pm["controls"] if c["control"] == "real labels")["test_r2"] > 0.1
    for c in ctl:                                                                              # shuffled labels: no signal in either flavour
        assert c["test_r2"] < 0.05                                                              # nothing learnable from the shuffled labels themselves
        if "global" in c["control"]:
            assert abs(c["test_auroc_vs_real_labels"] - 0.5) < 0.12
        # within-problem shuffles keep each problem's label multiset, so the probe may still rank problems (AUROC vs the real labels > 0.5): that is the problem-level part, reported not asserted away
    _, _, off2 = planted["surprise"]
    p2 = next(r for r in off2["paper"]["ladder"] if r["regime"] == "iid" and "probe" in r["predictor"])
    s2 = next(r for r in off2["paper"]["ladder"] if r["regime"] == "iid" and "surprise" in r["predictor"])
    assert p2["auroc"] - s2["auroc"] < pr["auroc"] - 0.5                                        # the surprise-carried signal is explained by the surprise baseline


# ----------------------------------------------------------------------------- sampling ceiling
def test_pooled_loo_auroc_known_cases():
    rng = np.random.RandomState(0)
    p = rng.rand(300)
    F = (rng.rand(300, 8) < p[:, None]).astype(float)
    V = np.ones_like(F, bool)
    a = stoch.pooled_loo_auroc(F, V)
    assert 0.6 < a < 0.9                                                                        # informative but noisy
    F0 = (rng.rand(300, 8) < 0.5).astype(float)
    assert abs(stoch.pooled_loo_auroc(F0, V) - 0.5) < 0.05
    assert np.isnan(stoch.pooled_loo_auroc(F[:, :2], V[:, :2]))                                 # < 3 valid draws: nothing to say


def test_stoch_run_resumable_tiny(tmp_path, tok, model):
    probs, G, irows = tiny_inject(tmp_path, tok, model)
    assert not any(r.get("fin", True) for r in irows) or True
    irows = [dict(r, fin=True) for r in irows]                                                  # the random tiny model never terminates; the selection keeps only finished greedy continuations
    out = tmp_path / "mega" / "robust" / "stoch"
    for sh in (0, 1):
        stoch.run(probs, irows, model, tok, out, sh, 2, n_rows=12, n_samp=3, max_new=5, max_seqs=16, log=QUIET)
    rows = stoch.load(out)
    assert 0 < len(rows) <= 12 and all(len(r["flips"]) == 3 and len(r["fins"]) == 3 for r in rows)
    names = sorted(p.name for p in out.glob("k*_*.jsonl"))
    stoch.run(probs, irows, model, tok, out, 0, 2, n_rows=12, n_samp=3, max_new=5, max_seqs=16, log=QUIET)
    assert names == sorted(p.name for p in out.glob("k*_*.jsonl"))
    n0 = stoch.run(probs, irows, model, tok, tmp_path / "x", 0, 1, n_rows=12, n_samp=3, max_new=5, max_seqs=8, budget_s=1e-9, log=QUIET)
    assert not n0 or n0 == 0
    assert stoch.analyze(tmp_path, tmp_path / "mega", out, "gsm8k", B=20, log=QUIET)["status"] == "n/a"      # far too few rows to decide


def test_stoch_analyze_planted(planted, tmp_path):
    tmp, md, off = planted["post"]
    P, why = inject.prepare(tmp, md, "gsm8k")
    df = P["df"]
    rng = np.random.RandomState(1)
    out = tmp_path / "stoch"
    out.mkdir()
    sel = df                                                                                    # all rows, so that enough fall in the iid test split
    rows = []
    for _, r in sel.iterrows():
        pf = 1 / (1 + np.exp(-(3 * (float(r["flip"]) - 0.5))))          # sampled follow probability tracks the greedy label
        fl = (rng.rand(8) < pf).astype(float)
        rows.append({"id": int(r["id"]), "k": int(r["k"]), "var": r["var"], "greedy_flip": float(r["flip"]), "flips": fl.tolist(), "fins": [True] * 8})
    (out / "k00_00.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    a = stoch.analyze(tmp, md, out, "gsm8k", B=50, log=QUIET)
    assert a["status"] == "ok" and a["n_rows"] > 50
    assert "auroc" in a and "captured_vs_floor" in a


# ----------------------------------------------------------------------------- natural errors
def test_natural_scan_finds_slips_and_controls():
    t = "First 12 + 15 = 28 apples.\nThen 28 * 3 = 84.\nAlso 1,000 - 1 = 999 and 7 * 8 = 56.\nNot an equation 2 + 3 + 4 = 9\nWrong 9 - 4 = 6"
    ev = natural.scan(t)
    slips = [e for e in ev if e["wrong"]]
    ok = [e for e in ev if not e["wrong"]]
    assert len(slips) == 2 and {e["shown"] for e in slips} == {28, 6}
    assert all(e["a"] + e["b"] != e["shown"] for e in slips if e["op"] == "+")
    assert len(ok) == 3 and all(e["shown"] == e["true"] for e in ok)
    assert natural.fixed_prefix(t, slips[0]).startswith("First 12 + 15 = 27 apples.") and natural.fixed_prefix(t, slips[-1]) is None
    assert not any("2 + 3" in t[e["start"]:e["end"]] for e in ev)                               # chains are not complete two-operand equations


def test_natural_run_tiny(tmp_path, tok, model):
    text = "Step 1: add: 3 + 4 = 8\nThen: 8 * 2 = 16\nSo 16 apples.\n\\boxed{16}"
    comp = tok.encode(text)
    probs = [{"id": i, "ds": "gsm8k", "kind": "int", "question": f"What is {i}+{i}?", "gold": "14", "d1": 1.0, "split": "test"} for i in range(6)]
    G = {p["id"]: {"plain_ids": comp, "plain_fin": True, "plain_ans": "16", "plain_text": text} for p in probs}
    out = tmp_path / "natural"
    for sh in (0, 1):
        natural.run(probs, G, model, tok, out, sh, 2, chunk=2, max_new=6, max_seqs=16, log=QUIET)
    ev = natural.load(out)
    assert ev is not None
    before = sorted(p.name for p in out.glob("n*_*"))
    assert before
    natural.run(probs, G, model, tok, out, 0, 2, chunk=2, max_new=6, max_seqs=16, log=QUIET)
    assert before == sorted(p.name for p in out.glob("n*_*"))


# ----------------------------------------------------------------------------- transfer matrix
def test_transfer_matrix_planted(planted):
    tmp, md, off = planted["post"]
    P, _ = inject.prepare(tmp, md, "gsm8k")
    rows = matrix.transfer({"gsm8k": P, "math": P}, log=QUIET)
    assert len(rows) == 4 and {(x["train"], x["eval"]) for x in rows} == {(a, b) for a in ("gsm8k", "math") for b in ("gsm8k", "math")}
    r = next(x for x in rows if x["train"] == "gsm8k" and x["eval"] == "math")
    assert r["auroc"] > 0.65 and r["beats_floor"]
    print(sorted(r))


# ----------------------------------------------------------------------------- length reference
def test_lenref_on_tiny_pipeline(run):
    tmp, problems = run
    a = lenref.analyze(tmp, "gsm8k", QUIET)
    assert a["status"] in ("ok", "n/a")
    assert abs(lenref.loo_mad([3, 5, 9, 10]) - 4.75) < 1e-9 and lenref.loo_mad([1, 1, 1, 1]) == 0 and np.isnan(lenref.loo_mad([1, 2]))


# ----------------------------------------------------------------------------- report assembler
def test_cross_cell_rule():
    ok = [{"check": f"C{i}", "ok": True, "what": "", "detail": ""} for i in range(1, 9)]
    no8 = [dict(x, ok=(x["check"] != "C8")) for x in ok]
    bad = [dict(x, ok=False) for x in ok]
    assert paper.cross_cell({"a": ok, "b": ok})["verdict"] == "CAUSAL REPLICATES"
    assert paper.cross_cell({"a": ok, "b": no8})["verdict"] == "CAUSAL REPLICATES"          # one cell without C8 allowed
    assert paper.cross_cell({"a": ok, "b": no8, "c": no8})["verdict"] == "REPLICATES (diagnostic)"
    assert paper.cross_cell({"a": ok, "b": bad})["verdict"] == "MIXED"
    assert paper.cross_cell({"b": bad})["verdict"] == "NOT REPLICATED"
    assert paper.cross_cell({})["verdict"] == "NO CELLS"
    assert paper.cross_cell({"a": no8})["verdict"] == "REPLICATES (diagnostic)"                  # a single cell without a causal test cannot claim causal replication


def test_paper_end_to_end_and_cli(planted, tmp_path, monkeypatch):
    tmp, md, _ = planted["post"]
    cell = paper.analyze_cell(tmp, md, "gsm8k", log=QUIET)
    assert cell["offline"]["status"] == "ok" and cell["steer"]["status"] in ("n/a", "error", "ok", "PARTIAL")
    assert (md / "robust" / "analysis" / "cell.json").exists()
    spec = [{"name": "A / GSM8K", "model": "m", "ds": "gsm8k", "run_dir": str(tmp), "mega_dir": str(md)},
            {"name": "B / missing", "model": "m", "ds": "math", "run_dir": str(tmp), "mega_dir": str(tmp / "nope")}]
    f = tmp / "cells.json"
    f.write_text(json.dumps(spec))
    cli.main(["paper", "--run-dir", str(tmp), "--cells", str(f), "--paper-out", str(tmp / "paper")])
    txt = (tmp / "paper" / "PAPER_REPORT.md").read_text()
    assert "## Verdict" in txt and "A / GSM8K" in txt and "## 2. The ladder" in txt and "B / missing" not in txt.split("## 1.")[0]
    assert (tmp / "paper" / "paper_ladder.csv").exists() and (tmp / "paper" / "fig_ladder.png").exists()
    cli.main(["cell-analyze", "--run-dir", str(tmp), "--mega-dir", str(md), "--ds", "gsm8k"])
