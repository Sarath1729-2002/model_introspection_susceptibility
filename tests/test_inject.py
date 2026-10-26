"""Tests of the inject run: the injection-time state / surprise extraction on the tiny model (alignment checked against direct forwards), the analysis on planted signals
(a signal that lives ONLY in the post-injection state must be found; a signal that the model's own surprise already carries must NOT be called SUPPORTED), the large-dose steering analysis and the CLI wiring."""
import json

import numpy as np
import pytest
import torch

from psuite import cli, inject, mega, steer, steps
from psuite.hf_util import enc
from psuite.prompts import chat_prompt, plain_text
from test_pipeline import tok, model, run   # noqa: F401  (fixtures)
from test_mega import fabricate_mega, fab_steer
from test_signal import NL, D as FD

QUIET = lambda *_: None            # noqa: E731


def test_pure_helpers(tok):
    assert inject.propagation("14", "12") == pytest.approx(np.log10(3.0))
    assert inject.propagation("1,000", "999") == pytest.approx(np.log10(2.0))
    assert np.isnan(inject.propagation(None, "12")) and np.isnan(inject.propagation("x", "12"))
    ids = tok.encode("ab = 123 cd")
    f, l = inject.num_tokens(tok, ids, 5, 8)                              # char-level tokenizer: exactly the three digits
    assert (f, l) == (5, 7) and tok.decode(ids[f:l + 1]) == "123"


def test_inject_stage_tiny_model(tmp_path, tok, model):
    text = "Step 1: add: 3 + 4 = 7\nThen: 7 * 2 = 14\nSo 14 apples.\n\\boxed{14}"
    comp = tok.encode(text)
    probs = [{"id": i, "ds": "gsm8k", "kind": "int", "question": f"What is {i}+{i}?", "gold": "14", "d1": 1.0, "split": "fit"} for i in range(10)]
    G = {p["id"]: {"plain_ids": comp, "plain_fin": True, "plain_ans": "14", "plain_text": text} for p in probs}
    sdir = tmp_path / "mega" / "steps"
    for sh in (0, 1):
        steps.run(probs, G, model, tok, sdir, sh, 2, chunk=3, max_steps=3, max_new=6, max_seqs=16, log=QUIET)
    sr = inject.step_rows(sdir)
    assert len(sr) == 10 and all("ans_n1" in r and "ans_n2" in r for rs in sr.values() for r in rs)
    out = tmp_path / "mega" / "inject"
    for sh in (0, 1):
        inject.run(probs, G, sr, model, tok, out, sh, 2, chunk=3, log=QUIET)
    files = sorted(p.name for p in out.glob("i*.jsonl"))
    inject.run(probs, G, sr, model, tok, out, 0, 2, chunk=3, log=QUIET)
    assert files and files == sorted(p.name for p in out.glob("i*.jsonl"))                # resumable
    rows, HA, HB, layers = inject.load(out)
    n_steps = sum(len(v) for v in sr.values())
    assert len(rows) == 2 * n_steps and HA.shape == (len(rows), len(layers), 32) == HB.shape
    r = rows[5]
    for key in ("flip", "prop", "flip_step", "inj_nll_sum", "orig_nll_sum", "inj_ent0", "ptext", "line", "delta", "new_value"):
        assert key in r
    assert r["delta"] == r["new_value"] - r["value"] and r["flip"] in (0.0, 1.0)
    # alignment against direct forwards: HA is the state at the LAST DIGIT of the injected number, HB the last token of the edited prefix, the surprise is the NLL of the injected digits
    p = probs[r["id"]]
    pr = enc(tok, chat_prompt(tok, plain_text(p)))
    ids = enc(tok, chat_prompt(tok, plain_text(p)) + r["ptext"])
    nvs = str(r["new_value"])
    pos_last = len(pr) + r["ptext"].rindex(nvs) + len(nvs) - 1                  # char-level tokenizer: one token per character
    with torch.no_grad():
        o = model.model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
        lg = model(input_ids=torch.tensor([ids])).logits[0].float().log_softmax(-1)
    li = len(layers) - 1
    np.testing.assert_allclose(HB[5, li].astype(np.float32), o[int(layers[li])][0, len(ids) - 1].numpy(), atol=3e-2, rtol=3e-2)
    np.testing.assert_allclose(HA[5, li].astype(np.float32), o[int(layers[li])][0, pos_last].numpy(), atol=3e-2, rtol=3e-2)
    a = pos_last - len(nvs) + 1
    nll = -sum(float(lg[j - 1, ids[j]]) for j in range(a, pos_last + 1))
    assert nll == pytest.approx(r["inj_nll_sum"], rel=1e-3, abs=1e-3) and r["inj_n_tok"] == len(nvs)


# ----------------------------------------------------------------------------- analysis on planted signals
def write_inject(mega_dir, mode, seed=11):
    """inject rows for every step row of the fabricated steps thread.  mode 'post': the flip depends on a latent w that lives ONLY in the post-injection state;
    mode 'surprise': the flip depends on the model's own surprise at the injected number (an observable), the state carries nothing extra."""
    rng = np.random.RandomState(seed)
    srows, _, _ = mega.load_steps(mega_dir / "steps")
    ly = np.array([4, 8, 12])
    rows, HA, HB = [], [], []
    for r in srows:
        for v in ("n1", "n2"):
            w = rng.randn()
            sur = rng.randn()
            pf = 1 / (1 + np.exp(-2 * (w if mode == "post" else sur)))
            flip = float(rng.rand() < pf)
            h = rng.randn(len(ly), FD).astype(np.float32)
            if mode == "post":
                h[ly >= 12, 5] += 2.0 * w
            value, delta = r["value"], (1 if v == "n1" else max(10, r["value"] // 10))
            nll = 5.0 + 2.0 * sur + 0.1 * rng.randn()
            rows.append({"id": r["id"], "k": r["k"], "var": v, "value": value, "new_value": value + delta, "delta": delta, "reuse": r["reuse"], "n_steps": r["n_steps"], "t_end": r["t_end"], "n_tok": r["n_tok"],
                         "start": r["start"], "end": r["end"], "ans_orig": "1", "ans_var": "2", "line": f"Step: {value - 3} + 3 = {value}", "ptext": f"Step: {value - 3} + 3 = {value + delta}\n",
                         "flip": flip, "flip_step": flip, "prop": float(flip) * (1 + 0.2 * rng.randn()),
                         "inj_nll_sum": nll, "inj_nll_mean": nll / 2, "inj_minlp": -nll / 2, "inj_lp0": -nll / 3, "inj_ent0": 1.0 + 0.1 * rng.randn(), "inj_n_tok": 2,
                         "orig_nll_sum": 1.0 + 0.1 * rng.randn(), "orig_nll_mean": 0.5, "orig_ent0": 1.0, "orig_lp0": -0.3, "orig_n_tok": 2})
            HA.append(h)
            HB.append(h + 0.1 * rng.randn(*h.shape).astype(np.float32))
    d = mega_dir / "inject"
    d.mkdir(exist_ok=True)
    with open(d / "i00_0000.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.savez(d / "i00_0000.npz", HA=np.stack(HA).astype(np.float16), HB=np.stack(HB).astype(np.float16), layers=ly)


@pytest.fixture(scope="module")
def inj_runs(tmp_path_factory):
    out = {}
    for mode in ("post", "surprise"):
        tmp = tmp_path_factory.mktemp(f"inj_{mode}")
        md = fabricate_mega(tmp)
        write_inject(md, mode)
        out[mode] = (tmp, md, inject.write_report(tmp, md, log=QUIET))
    return out


def test_post_injection_signal_is_found(inj_runs):
    import pandas as pd
    tmp, md, rep = inj_runs["post"]
    st = pd.read_csv(md / "inject" / "analysis" / "inject_status.csv").set_index("label")
    assert st.loc["I[num]:flip", "status"] == "SUPPORTED", st.loc["I[num]:flip"].to_dict()
    assert st.loc["Pre[nat]:flip", "status"] != "SUPPORTED"                  # the pre-edit state knows nothing about this latent
    txt = rep.read_text()
    assert "## Verdict" in txt and "I[num]:flip: SUPPORTED" in txt and "Pre-registered decision" in txt


def test_signal_carried_by_the_models_own_surprise_is_not_supported(inj_runs):
    import pandas as pd
    tmp, md, rep = inj_runs["surprise"]
    st = pd.read_csv(md / "inject" / "analysis" / "inject_status.csv").set_index("label")
    assert st.loc["I[num]:flip", "status"] in ("NEGATIVE", "INCONCLUSIVE"), st.loc["I[num]:flip"].to_dict()
    assert "I[num]:flip: SUPPORTED" not in rep.read_text()


def test_report_without_data_does_not_crash(tmp_path):
    from test_signal import fabricate
    fabricate(tmp_path)
    p = inject.write_report(tmp_path, tmp_path / "mega", log=QUIET)
    txt = p.read_text()
    assert "not enough data" in txt or "n/a" in txt


# ----------------------------------------------------------------------------- large-dose steering
def test_large_dose_dose_table(tmp_path):
    out = tmp_path / "big"
    out.mkdir()
    rng = np.random.RandomState(0)
    n = 120
    base = 5.0 + 0.5 * rng.randn(n)
    ok = rng.rand(n)
    doses = (8.0, 16.0, 32.0)
    al = tuple([-d for d in reversed(doses)] + list(doses))
    conds = steer.conditions(al, al, 3)
    assert len(conds) == 1 + 2 * 6 + 3 * 6
    for name, fam, a in conds:
        eff = 0.02 * a if fam in ("probe", "diff") else 0.0
        with open(out / f"r00_{name}.jsonl", "w") as f:
            for i in range(n):
                f.write(json.dumps({"id": i, "cond": name, "ntok": int(np.exp(base[i] + eff + 0.05 * rng.randn())), "fin": True, "ans": "1", "correct": float(ok[i] > 0.3)}) + "\n")
    r = steer.analyze(out, dmg_alpha=8.0)
    assert r["status"] == "SUPPORTED" and len(r["dose"]) == len(al)
    d = r["dose"].set_index("alpha")
    assert d.loc[32.0, "probe_d_logT"] > d.loc[8.0, "probe_d_logT"] > 0 > d.loc[-8.0, "probe_d_logT"] and d.loc[32.0, "rand_mean_abs_d_logT"] < 0.05


# ----------------------------------------------------------------------------- CLI wiring
def test_inject_cli(run, tok, model, monkeypatch):
    tmp, _ = run
    monkeypatch.setattr("psuite.hf_util.load_model", lambda mid: (model, tok))
    md = tmp / "mega_inj_cli"
    common = ["--run-dir", str(tmp), "--mega-dir", str(md)]
    with pytest.raises(SystemExit):                                                  # no step-thread outcomes yet: fail fast with a clear message
        cli.main(["inject", *common, "--chunk", "4"])
    (md / "steer").mkdir(parents=True, exist_ok=True)
    np.savez(md / "steer" / "dirs.npz", probe=np.eye(32)[0], diff=np.eye(32)[1], rand=np.eye(32)[2:6], layer=2, s=1.0)
    cli.main(["steer-run", *common, "--shard", "0", "--nshards", "1", "--steer-n", "4", "--max-new", "6", "--max-seqs", "16", "--steer-alphas", "8,32", "--steer-nrand", "2", "--steer-out", "steer_big"])
    got = sorted(p.name for p in (md / "steer_big").glob("r00_*.jsonl"))
    assert len(got) == len(steer.conditions((-32.0, -8.0, 8.0, 32.0), (-32.0, -8.0, 8.0, 32.0), 2)) and "r00_probe_a+32.jsonl" in got and "r00_probe_a-8.jsonl" in got
    assert not (md / "steer").glob("r00_*")  or not list((md / "steer").glob("r00_*"))        # results never land in the directions directory
    cli.main(["inject-analyze", *common, "--steer-out", "steer_big"])
    assert (md / "inject" / "analysis" / "INJECT_REPORT.md").exists()
