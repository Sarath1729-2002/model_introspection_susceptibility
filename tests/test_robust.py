"""Tests of the robustness + causal follow-up: helpers, the offline checks on planted signals (a signal that exists only after the injection survives every subset check, a signal carried by the
model's own surprise does not), the scorecard logic, the last-position-only steering hook on the tiny model, the injection-steering run / analysis (dose-restriction, collapse excluded, random-direction control) and the CLI wiring."""
import json

import numpy as np
import pytest
import torch

from psuite import cli, inject, istear, robust, steps
from psuite.hf_util import enc
from psuite.textutil import parse_answer
from test_pipeline import tok, model, run   # noqa: F401  (fixtures)
from test_mega import fabricate_mega
from test_inject import write_inject

QUIET = lambda *_: None            # noqa: E731


# ----------------------------------------------------------------------------- helpers
def test_pure_helpers():
    X = robust.hashed_bow(["Tom has 3 apples", "tom has 3 apples!", "completely different"])
    assert X.shape == (3, robust.HASH_DIM) and np.allclose(X[0], X[1]) and not np.allclose(X[0], X[2])
    y = np.array([0, 0, 1, 1, 1, 0.0])
    assert robust.auroc(y, np.array([1, 2, 3, 4, 5, 0.0])) == pytest.approx(1.0)
    assert robust.auroc(y, -np.array([1, 2, 3, 4, 5, 0.0])) == pytest.approx(0.0)
    assert robust.auroc(y, np.zeros(6)) == pytest.approx(0.5)
    assert np.isnan(robust.auroc(np.zeros(4), np.arange(4)))
    assert robust.tpr_at_fpr(y, np.array([1, 2, 3, 4, 5, 0.0])) == pytest.approx(1.0)
    rng = np.random.RandomState(0)
    g = np.repeat(np.arange(60), 5)
    yy = (rng.rand(300) < 0.5).astype(float)
    good, bad = yy + 0.3 * rng.randn(300), rng.randn(300)
    d, lo, hi = robust.boot_auroc_delta(yy, good, bad, g, B=100)
    assert d > 0.3 and lo > 0 and hi >= d - 0.2


def test_scorecard_and_overall():
    sup = {"status": "SUPPORTED", "why": "w", "iid_probe_r2": .6, "iid_obs_r2": .5, "iid_delta": .1, "iid_lo": .05}
    off = {"status": "ok", "checks": {k: dict(sup) for k in ("C1", "C2", "C3", "C4")}, "shuffled": {"r2_by_seed": [0.0, -0.01], "ok": True},
           "diag": {"regimes": {"iid": {"probe_r2": .6, "pre_r2": .48, "r2_delta_vs_pre": .1, "r2_delta_vs_pre_lo": .05, "r2_delta_vs_pre_hi": .2, "probe_auroc": .8, "obs_auroc": .75, "auroc_delta": .05, "auroc_lo": .02, "auroc_hi": .08},
                                  "shift": {"probe_auroc": .8, "obs_auroc": .75, "auroc_delta": .05, "auroc_lo": .01, "auroc_hi": .08}}}}
    rows = robust.scorecard(off, {"status": "SUPPORTED", "why": "ok"})
    assert [r["check"] for r in rows] == [f"C{i}" for i in range(1, 9)] and all(r["ok"] for r in rows)
    assert robust.overall(rows)["verdict"] == "ROBUST + CAUSAL"
    rows = robust.scorecard(off, {"status": "NEGATIVE", "why": "flat"})
    assert robust.overall(rows)["verdict"] == "ROBUST (diagnostic)" and "correlate" in robust.overall(rows)["why"]
    rows = robust.scorecard(off, None)
    assert robust.overall(rows)["verdict"] == "ROBUST (diagnostic)" and "not evaluated" in robust.overall(rows)["why"]
    off["checks"]["C1"] = {"status": "INCONCLUSIVE", "why": "x"}
    rows = robust.scorecard(off, {"status": "SUPPORTED", "why": "ok"})
    ov = robust.overall(rows)
    assert ov["verdict"] == "FRAGILE" and "C1" in ov["why"]
    assert robust.overall(robust.scorecard({"status": "n/a"}, None))["verdict"] == "INCOMPLETE"


# ----------------------------------------------------------------------------- offline checks on planted signals
@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    out = {}
    for mode in ("post", "surprise"):
        tmp = tmp_path_factory.mktemp(f"rob_{mode}")
        md = fabricate_mega(tmp)
        write_inject(md, mode)
        out[mode] = (tmp, md, robust.offline(tmp, md, log=QUIET, B=100))
    return out


def test_post_injection_signal_survives_the_subset_checks(planted):
    tmp, md, off = planted["post"]
    assert off["status"] == "ok"
    ck = off["checks"]
    for k in ("C1", "C2", "C3", "C4"):
        assert ck[k]["status"] == "SUPPORTED", (k, ck[k])
    assert off["shuffled"]["ok"], off["shuffled"]
    dg = off["diag"]["regimes"]
    assert dg["iid"]["probe_auroc"] > dg["iid"]["obs_auroc"] + 0.05 and dg["iid"]["auroc_lo"] > 0 and dg["shift"]["auroc_lo"] > 0
    assert dg["iid"]["r2_delta_vs_pre_lo"] > 0                       # the pre-edit state knows nothing about the planted latent
    rows = robust.scorecard(off, None)
    assert all(r["ok"] for r in rows if r["check"] in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")), rows
    assert (md / "robust" / "offline.json").exists()
    assert off["diag"]["strata"] and {s["stratum"] for s in off["diag"]["strata"]} >= {"last step", "not last step"}


def test_surprise_carried_signal_is_not_robust(planted):
    tmp, md, off = planted["surprise"]
    ck = off["checks"]
    assert ck["C4"]["status"] != "SUPPORTED", ck["C4"]
    assert robust.overall(robust.scorecard(off, None))["verdict"] != "ROBUST (diagnostic)"


def test_report_written_and_reads_cached_offline(planted):
    tmp, md, off = planted["post"]
    p = robust.write_report(tmp, md, log=QUIET)
    txt = p.read_text()
    assert "## Verdict:" in txt and "ROBUST (diagnostic)" in txt and "C8" in txt and "Pre-registered" in txt and "steering not run" in txt
    assert (md / "robust" / "analysis" / "robust_scorecard.csv").exists()


def test_report_without_data_does_not_crash(tmp_path):
    from test_signal import fabricate
    fabricate(tmp_path)
    p = robust.write_report(tmp_path, tmp_path / "mega", log=QUIET)
    assert "INCOMPLETE" in p.read_text() or "FRAGILE" in p.read_text()


# ----------------------------------------------------------------------------- the steering hook (last prompt position, prefill only)
def test_steer_last_hook_only_last_position_and_only_prefill(model):
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    vec = torch.randn(32)
    blk = model.model.layers[1]                                                           # = the producer of hidden_states[2]

    def out(vec_, x=ids):
        cap = {}
        with istear.SteerLast(model, 2, vec_):
            h = blk.register_forward_hook(lambda m, i, o: cap.__setitem__("o", (o[0] if isinstance(o, tuple) else o).detach().clone()))      # registered after ours: sees the steered output
            try:
                with torch.no_grad():
                    final = model.model(input_ids=x).last_hidden_state
            finally:
                h.remove()
        return cap["o"], final
    (b_o, b_f), (s_o, s_f) = out(None), out(vec)
    assert torch.allclose(s_o[0, :-1], b_o[0, :-1], atol=1e-6)                             # earlier positions untouched
    assert torch.allclose(s_o[0, -1], b_o[0, -1] + vec, atol=1e-5)                          # exactly the vector at the last position
    assert not torch.allclose(s_f[0, -1], b_f[0, -1], atol=1e-4) and torch.allclose(s_f[0, :-1], b_f[0, :-1], atol=1e-5)    # it propagates forward from there only
    (o1, f1), (o0, f0) = out(vec, ids[:, -1:]), out(None, ids[:, -1:])                       # T = 1 (a decoding step): nothing is added
    assert torch.allclose(o1, o0, atol=1e-6) and torch.allclose(f1, f0, atol=1e-6)
    assert torch.allclose(out(None)[1], b_f, atol=1e-6)                                    # the hook is gone after the context ends


# ----------------------------------------------------------------------------- injection steering: run on the tiny model
def tiny_inject(tmp_path, tok, model, n=10):
    text = "Step 1: add: 3 + 4 = 7\nThen: 7 * 2 = 14\nSo 14 apples.\n\\boxed{14}"
    comp = tok.encode(text)
    probs = [{"id": i, "ds": "gsm8k", "kind": "int", "question": f"What is {i}+{i}?", "gold": "14", "d1": 1.0, "split": "test"} for i in range(n)]
    G = {p["id"]: {"plain_ids": comp, "plain_fin": True, "plain_ans": "14", "plain_text": text} for p in probs}
    sdir = tmp_path / "mega" / "steps"
    for sh in (0, 1):
        steps.run(probs, G, model, tok, sdir, sh, 2, chunk=3, max_steps=3, max_new=6, max_seqs=16, log=QUIET)
    sr = inject.step_rows(sdir)
    out = tmp_path / "mega" / "inject"
    for sh in (0, 1):
        inject.run(probs, G, sr, model, tok, out, sh, 2, chunk=3, log=QUIET)
    return probs, G, inject.load(out)[0]


def write_dirs(d, keys, layers=(2, 3), n_rand=3, D=32, seed=0):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    arrs = {"layers": np.array(layers)}
    for L in layers:
        u = lambda *s: (lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True))(rng.randn(*s))      # noqa: E731
        arrs.update({f"probe_{L}": u(D), f"diff_{L}": u(D), f"rand_{L}": u(n_rand, D), f"s_{L}": np.array(2.0)})
    np.savez(d / "dirs.npz", **arrs)
    (d / "rows.json").write_text(json.dumps(keys))


def test_istear_run_tiny_model(tmp_path, tok, model):
    probs, G, irows = tiny_inject(tmp_path, tok, model)
    keys = [[r["id"], r["k"], r["var"]] for r in irows][:24]
    d = tmp_path / "mega" / "robust" / "steer"
    write_dirs(d, keys)
    items = istear.build_items(probs, G, irows, keys, tok)
    assert len(items) == len(keys)
    lut = {(r["id"], r["k"], r["var"]): r for r in irows}
    for it in items:                                                                      # the prompt ends exactly with the injected number: the steered position is its last token
        r = lut[(it["id"], it["k"], it["var"])]
        assert it["prompt"].endswith(str(r["new_value"])) and r["ptext"].startswith(it["cut"])
    for sh in (0, 1):
        istear.run(probs, G, irows, model, tok, d, sh, 2, max_new=6, max_seqs=16, log=QUIET)
    names = sorted(p.name for p in d.glob("r*_*.jsonl"))
    cond = istear.conditions([2, 3])
    assert len(names) == 2 * len(cond) and f"r00_{cond[0][0]}.jsonl" in names
    before = names
    istear.run(probs, G, irows, model, tok, d, 0, 2, max_new=6, max_seqs=16, log=QUIET)       # resumable
    assert before == sorted(p.name for p in d.glob("r*_*.jsonl"))
    R, shards = istear.load(d)
    assert len(R) == len(cond) and all(shards[c] == {"00", "01"} for c in R)
    base = R["base"]
    assert len(base) == len(items)
    it = items[3]                                                                          # baseline row = plain greedy continuation of the truncated prompt
    from psuite.hf_util import generate_ids
    ids, fin = generate_ids(model, tok, [it["prompt"]], 6, 1, 0.0, 0.95, 16)[0][0]
    ans = parse_answer(it["cut"] + tok.decode(ids, skip_special_tokens=True), it["kind"])
    assert base[(it["id"], it["k"], it["var"])]["ans"] == ans and base[(it["id"], it["k"], it["var"])]["flip"] == float(ans != it["ans_orig"])
    big = [r for c, rr in R.items() if "probe_L3_a+16" in c for r in rr.values()]
    assert big and set(r["flip"] for r in big) <= {0.0, 1.0}
    a = istear.analyze(d, B=50)
    assert a["status"] == "n/a"                                                           # 24 rows: far too few, must say so instead of deciding


def test_istear_time_budget_stops_cleanly(tmp_path, tok, model):
    probs, G, irows = tiny_inject(tmp_path, tok, model, n=4)
    keys = [[r["id"], r["k"], r["var"]] for r in irows][:6]
    d = tmp_path / "mega" / "robust" / "steer"
    write_dirs(d, keys)
    n = istear.run(probs, G, irows, model, tok, d, 0, 1, max_new=4, max_seqs=8, budget_s=1e-9, log=QUIET)
    assert n == 0 and not list(d.glob("r*_*.jsonl"))


# ----------------------------------------------------------------------------- injection steering: analysis
def fab_istear(out, slope=0.01, collapse=True, n_prob=600, seed=0, drop_shard=False, shards=1, integrity=0.0):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    keys = [(i, k, "n1") for i in range(n_prob) for k in range(3)]
    base_p = np.clip(0.5 + 0.2 * rng.randn(len(keys)), 0.05, 0.95)
    cond = istear.conditions([8, 12])
    if collapse:
        cond = cond + [("probe_L8_a+32", "probe_L8", 8, 32.0), ("probe_L8_a-32", "probe_L8", 8, -32.0)]
    for name, fam, L, a in cond:
        eff = 0.0
        if fam and fam.startswith("probe_L8"):
            eff = slope * a if abs(a) <= 16 else 0.5 * np.sign(a)
        if integrity and fam and fam.startswith("probe_L12"):
            eff = 0.0
        flips = (rng.rand(len(keys)) < np.clip(base_p + eff, 0, 1)).astype(float)
        for sh in range(shards):
            if drop_shard and sh == 1 and name.startswith("diff_L12_a+16"):
                continue
            f = out / f"r{sh:02d}_{name}.jsonl"
            with open(f, "w") as fh:
                for j, (i, k, v) in enumerate(keys):
                    if j % shards == sh:
                        ok = rng.rand() > (integrity if (fam and fam.startswith("probe_L12") and abs(a) <= 8) else 0.0)
                        fh.write(json.dumps({"id": i, "k": k, "var": v, "cond": name, "ntok": 50, "fin": bool(ok), "ans": "1", "flip": float(flips[j]), "correct": float(ok and flips[j] == 0 and rng.rand() > 0.1)}) + "\n")


def test_istear_conditions_priority_and_dose_cap():
    c = istear.conditions([8, 12])
    assert c[0][0] == "base" and len({x[0] for x in c}) == len(c)
    assert len(c) == 1 + 2 * (4 + 3 * 4 + 4) + 2 * (4 + 4)
    assert max(abs(x[3]) for x in c) == 16.0 and min(abs(x[3]) for x in c if x[3] != 0) == 2.0
    first_low = min(i for i, x in enumerate(c) if abs(x[3]) in (2.0, 4.0))
    assert all(abs(x[3]) in (8.0, 16.0) or x[3] == 0 for x in c[:first_low]) and all(abs(x[3]) in (2.0, 4.0) for x in c[first_low:])


def test_istear_analyze_supported_and_collapse_excluded(tmp_path):
    out = tmp_path / "s"
    fab_istear(out, slope=0.012, collapse=True)
    r = istear.analyze(out, B=300)
    assert r["status"] == "SUPPORTED", r["why"]
    assert r["layers"][8]["status"] == "SUPPORTED" and r["layers"][12]["status"] in ("NEGATIVE", "INCONCLUSIVE")
    lo, hi = r["slopes"]["probe_L8"][1], r["slopes"]["probe_L8"][2]
    assert r["slopes"]["probe_L8"][0] == pytest.approx(0.012, abs=0.003) and lo > 0              # the +-32 collapse rows (flip rate +-0.5) did not enter the slope
    assert "probe_L8_a+32" in set(r["table"].cond) and 32.0 in set(r["dose"].alpha) | {32.0}
    assert r["layers"][8]["max_rand"] < 0.006


def test_istear_analyze_null_is_negative_and_incomplete_condition_dropped(tmp_path):
    out = tmp_path / "n"
    fab_istear(out, slope=0.0, collapse=False)
    r = istear.analyze(out, B=300)
    assert r["status"] == "NEGATIVE", (r["status"], r["why"])
    out2 = tmp_path / "d"
    fab_istear(out2, slope=0.012, collapse=False, shards=2, drop_shard=True)
    r2 = istear.analyze(out2, B=100)
    assert "diff_L12_a+16" not in set(r2["table"].cond) and r2["status"] == "SUPPORTED"          # a condition that not every shard produced is not used


def test_istear_integrity_failure_blocks_supported(tmp_path):
    out = tmp_path / "i"
    fab_istear(out, slope=0.0, collapse=False)
    # L8 effect with broken generations at |alpha| <= 8: finished rate drops 20% -> never SUPPORTED
    for f in out.glob("r00_probe_L8_a*"):
        a = float(f.stem.split("_a")[1])
        if abs(a) <= 8.0:
            rows = [json.loads(l) for l in open(f)]
            with open(f, "w") as fh:
                for j, r in enumerate(rows):
                    r["fin"] = j % 5 != 0
                    r["flip"] = float(np.sign(a) > 0) if j % 2 else r["flip"]
                    fh.write(json.dumps(r) + "\n")
    r = istear.analyze(out, B=100)
    assert r["layers"][8]["status"] != "SUPPORTED"


# ----------------------------------------------------------------------------- prep + CLI wiring
def test_istear_prep_and_cli(planted, run, tok, model, monkeypatch):
    tmp, md, _ = planted["post"]
    info = istear.prep(tmp, md, md / "robust_p" / "steer", n_rows=300, log=QUIET)
    z = np.load(md / "robust_p" / "steer" / "dirs.npz")
    layers = [int(x) for x in z["layers"]]
    assert 12 not in layers and len(layers) == 2 and info["n_rows"] == 300                       # the last stored state is post-norm: never a hook target
    for L in layers:
        assert z[f"probe_{L}"].shape == (24,) and abs(np.linalg.norm(z[f"probe_{L}"]) - 1) < 1e-6 and z[f"rand_{L}"].shape == (3, 24) and float(z[f"s_{L}"]) > 0
    keys = json.loads((md / "robust_p" / "steer" / "rows.json").read_text())
    import pandas as pd
    from psuite.bench import read_problems
    split = {p["id"]: p["split"] for p in read_problems(tmp, ["gsm8k"])}
    assert len(keys) == 300 and all(split[k[0]] == "test" for k in keys)                          # only test-split problems are steered
    keys2 = istear.select_rows([{"id": k[0], "k": k[1], "var": k[2]} for k in keys], 300, None, None)
    assert [[r["id"], r["k"], r["var"]] for r in keys2] == keys                                  # deterministic order
    common = ["--run-dir", str(tmp), "--mega-dir", str(md)]
    cli.main(["robust-prep", *common, "--robust-dir", "robust_c", "--robust-n", "100"])
    assert (md / "robust_c" / "steer" / "dirs.npz").exists() and len(json.loads((md / "robust_c" / "steer" / "rows.json").read_text())) == 100
    fab_istear(md / "robust_c" / "steer", slope=0.012, collapse=False)
    cli.main(["robust-offline", *common, "--robust-dir", "robust_c"])
    assert (md / "robust_c" / "offline.json").exists()
    cli.main(["robust-analyze", *common, "--robust-dir", "robust_c"])
    txt = (md / "robust_c" / "analysis" / "ROBUST_REPORT.md").read_text()
    assert "## Verdict:" in txt and "Causal test" in txt and "C8" in txt and "slope_flip_per_sd" in txt
