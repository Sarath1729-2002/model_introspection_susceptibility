import json

import numpy as np
import pytest
import torch

from susc import bench, causal, cli, core, monitor, probes, report, sample, states, study, table, textutil, trace
from susc.hf_util import enc
from susc.prompts import CONDS, FRAMINGS, chat_prompt, user_text

CHARS = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,=+-*/\\{}()$:#?!<>|_'\n;%\"&`[]")


class FakeTok:
    def __init__(self):
        self.vocab = ["<pad>", "<eos>"] + CHARS + ["\n\n"]
        self.idx = {t: i for i, t in enumerate(self.vocab)}
        self.eos_token_id, self.pad_token_id = 1, 0

    def __len__(self):
        return len(self.vocab)

    def encode(self, s, add_special_tokens=False):
        out, i = [], 0
        while i < len(s):
            if s[i:i + 2] == "\n\n":
                out.append(self.idx["\n\n"]); i += 2
            else:
                out.append(self.idx.get(s[i], 2)); i += 1
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[int(i)] for i in ids if not (skip_special_tokens and int(i) in (0, 1)))


@pytest.fixture(scope="module")
def tok():
    return FakeTok()


@pytest.fixture(scope="module")
def model(tok):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=4096, eos_token_id=1, pad_token_id=0, tie_word_embeddings=False)
    return Qwen2ForCausalLM(cfg).eval()


def fake_loaders():
    rng = np.random.RandomState(0)
    gs = {"train": [], "test": []}
    for sp in gs:
        for i in range(60):
            a, b = int(rng.randint(2, 30)), int(rng.randint(2, 30))
            gs[sp].append({"question": f"Tom has {a} apples and gets {b} more ({sp}{i}). How many?", "answer": f"step <<{a}+{b}={a + b}>> \n#### {a + b}"})
    import pandas as pd
    mth = {s: pd.DataFrame([{"problem": f"What is {i}+{i} ({s})?", "solution": f"It is $\\boxed{{{2 * i}}}$.", "level": f"Level {1 + i % 5}", "type": "algebra"} for i in range(80)])
           for s in ("train", "test")}
    arc = {s: [{"id": f"{s}{i}", "question": f"Which is largest ({s}{i})?", "choices": {"label": list("ABCD"), "text": ["a", "b", "c", "d"]}, "answerKey": "ABCD"[i % 4]}
               for i in range(60)] for s in ("train", "validation", "test")}
    return {"gsm8k": lambda: gs, "math": lambda: mth, "arc": lambda split: arc[split]}


def test_textutil():
    assert textutil.parse_answer("so \\boxed{\\$1,250} then \\boxed{12 apples}", "int") == "12"
    assert textutil.parse_answer("\\boxed{(B)}", "mcq") == "B" and textutil.parse_answer("\\boxed{\\frac12}", "int") is None


def test_audit_rejects_bad_ground_truth():
    rows = bench.rows_arith(60, seed=1)
    for i, r in enumerate(rows):
        r["id"], r["kind"] = i, "int"
        r["hint"] = bench.make_hint("int", i, r["gold"])
        r["hint_off"] = bench.hint_distance("int", r["gold"], r["hint"])
    assert bench.audit(rows, log=lambda *_: None)["report"]["admitted"]
    bad = [dict(r) for r in rows]
    for r in bad[:10]:
        r["gold"] = str(int(r["gold"]) + 1)                       # corrupt 10 labels: recomputation from the prompt text must catch them
    rep = bench.audit(bad, log=lambda *_: None)["report"]
    assert rep["n_kept"] == 50 and not rep["admitted"] and "G2_arith_recompute" in rep["fails"]
    # hints: never equal gold, reproducible
    assert all(bench.hint_wrong_int(i, g) != g for i in range(200) for g in (0, 1, 5, 99, 1000))
    assert all(bench.hint_wrong_letter(i, c) != c for i in range(50) for c in "ABCD")
    assert bench.gsm8k_check("a <<2*3=7>>\n#### 7")[1] == "annotation_mismatch" and bench.gsm8k_check("#### 3.5")[1] == "non_integer"
    assert bench.run_code_block(bench.prog_text("mod", {"s0": 1, "n": 3, "a": 2, "m": 7, "c": 2, "b": 1})) == bench.prog_value("mod", {"s0": 1, "n": 3, "a": 2, "m": 7, "c": 2, "b": 1})


def test_download_failure_is_survivable(tmp_path):
    def boom():
        raise RuntimeError("no network")
    problems, reports = bench.build(tmp_path, {"math": 50, "arith": 60}, loaders={"math": boom}, log=lambda *_: None)
    assert {p["ds"] for p in problems} == {"arith"} and [r["dataset"] for r in reports] == ["arith"]


def test_prompts():
    p = {"question": "Q?", "kind": "int", "gold": "5", "hint": "7"}
    assert "7" in user_text(p, "auth") and "5" not in user_text(p, "auth").split("Please")[0].replace("Q?", "")
    assert "(Answer key: 5)" in user_text(p, "cor") and "Problem ID: 7" in user_text(p, "placebo") and "7" not in user_text(p, "clean")
    assert all(user_text(p, c).endswith(user_text(p, "clean").split("\n\n")[-1]) for c in CONDS)


@pytest.fixture(scope="module")
def run(tmp_path_factory, tok, model):
    tmp = tmp_path_factory.mktemp("run")
    sizes = {"gsm8k": 70, "math": 60, "arc": 60, "arith": 70, "prog": 70}
    problems, reports = bench.build(tmp, sizes, loaders=fake_loaders(), log=lambda *_: None)
    assert {r["dataset"] for r in reports} == set(sizes) and all(r["admitted"] for r in reports)
    for sh in (0, 1):
        sample.run(problems, model, tok, tmp, sh, 2, n=4, max_new=6, chunk=30, max_seqs=32, log=lambda *_: None)
        states.run(problems, model, tok, tmp, sh, 2, fwd_bs=5, chunk=40, log=lambda *_: None)
        trace.run(problems, model, tok, tmp, sh, 2, fwd_bs=3, chunk=30, log=lambda *_: None)
    return tmp, problems


def test_sample_states_resumable_and_correct(run, tok, model):
    tmp, problems = run
    before = sorted(p.name for p in (tmp / "samples").iterdir())
    sample.run(problems, model, tok, tmp, 0, 2, n=4, max_new=6, chunk=30, max_seqs=32, log=lambda *_: None)
    assert before == sorted(p.name for p in (tmp / "samples").iterdir())
    # a time budget of ~0 stops cleanly without writing partial files
    n_before = len(list((tmp / "states").iterdir()))
    assert states.run(problems, model, tok, tmp, 0, 2, budget_s=1e-9, log=lambda *_: None) == 0 and n_before == len(list((tmp / "states").iterdir()))
    st = table.load_states(tmp, ("clean", "auth"))
    assert st["states"]["clean"].shape == (len(problems), 5, 32) and (np.diff(st["ids"]) > 0).all()
    assert np.isfinite(st["lp"]["auth"]["lph"]).all() and np.allclose(st["lp"]["cor"]["lph"], st["lp"]["cor"]["lpg"])
    # the stored prompt-end state is the last prompt token's hidden state of a direct forward
    p = problems[7]
    ids = torch.tensor([enc(tok, chat_prompt(tok, user_text(p, "auth")))])
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    ref = torch.stack([h[0, -1] for h in hs]).numpy()
    k = int(np.where(st["ids"] == p["id"])[0][0])
    np.testing.assert_allclose(st["states"]["auth"][k].astype(np.float32), ref, atol=2e-2, rtol=2e-2)
    s = table.load_samples(tmp)
    r = s[(p["id"], "auth")]
    assert r["n"] == 4 and 0 <= r["n_wrong"] <= 4


def test_trace_positions_and_state(run, tok, model):
    tmp, problems = run
    tr = monitor.load_trace(tmp)
    assert tr["clean"]["h"].shape[1] == 5 and tr["clean"]["h"].shape[3] == 32 and (np.diff(tr["ids"]) > 0).all()
    # position 0 equals the pre-generation state
    st = table.load_states(tmp, ("clean",))
    common = np.intersect1d(st["ids"], tr["ids"])
    i = int(common[3])
    a = tr["clean"]["h"][np.where(tr["ids"] == i)[0][0], 0, -1].astype(np.float32)          # last stored layer
    b = st["states"]["clean"][np.where(st["ids"] == i)[0][0], tr["layers"][-1]].astype(np.float32)
    np.testing.assert_allclose(a, b, atol=3e-2, rtol=3e-2)
    assert (tr["clean"]["frac"][:, 0] == 0).all() and (np.diff(tr["clean"]["frac"], axis=1) >= -1e-6).all()


def test_ridge_and_core():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 20).astype(np.float32)
    y = X[:, 0] * 2 + 0.1 * rng.randn(400)
    res = core.ridge_eval(X[:300], X[300:350], X[350:], y[:300, None], y[300:350, None], y[350:, None])
    assert res[0]["test_r2"] > 0.9 and abs(core.auroc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9]) - 1) < 1e-9


def test_analysis_end_to_end(run):
    tmp, problems = run
    dss = ["gsm8k", "math", "arc", "arith", "prog"]
    for block in ("tables", "A", "B1", "B2", "CDE", "T"):
        study.compute(tmp, block, layers=[1, 2, 4] if block not in ("tables", "T") else None, log=lambda *_: None)
    df = report.pickle.load(open(tmp / "analysis" / "tables.pkl", "rb"))["df"]
    assert len(df) == len(problems) and {"S", "S_net", "cred", "uplift", "Sgap", "gap_plc", "adopts"} <= set(df.columns)
    A = report._load(tmp / "analysis", "A")
    regs = {r["regime"] for r in A["res"]}
    assert {"iid", "shift"} <= regs and any(x.startswith("lodo:") for x in regs)
    r = report._pick(A["res"], "A[pooled]", "S", "iid")
    assert r is not None and r["probe_layer"] in (1, 2, 4) and len(r["_pred"]) == r["n_test"]
    # leave-dataset-out really excludes the held-out dataset from fitting
    lo = report._pick(A["res"], "A[pooled]", "S", "lodo:gsm8k")
    assert set(lo["_ds"]) == {"gsm8k"}
    B = [x for n in ("B1", "B2") for x in report._load(tmp / "analysis", n)["res"]]
    assert {x["framing"] for x in B} == set(FRAMINGS) | {"placebo", "cor"}
    cde = report._load(tmp / "analysis", "CDE")
    assert len(cde["D"]) > 0 and cde["E"]["cosine"].shape[0] == len(probes.GEO) and np.all(np.isclose(np.diag(cde["E"]["cosine"].values), 1, atol=1e-4) | np.isclose(np.diag(cde["E"]["cosine"].values), 0, atol=1e-6))   # 0 = degenerate (all-zero) label on the tiny model
    T = report._load(tmp / "analysis", "T")["trace"]
    assert len(T) and {"p0", "pend"} <= set(T.pos)
    te = causal.treatment_effects(df)
    assert {"auth", "placebo", "cor(+gain)", "clean"} <= set(te.cond) and len(causal.hint_distance(df)) > 0
    mc = causal.monitor_curves(df, r)
    assert mc.recall_of_adopters.between(0, 1).all()
    p = report.write(tmp, log=lambda *_: None)
    txt = p.read_text()
    assert "Ground-truth audit" in txt and "Headline results" in txt and "Leave-one-framing-out" in txt and (tmp / "analysis" / "fig1_layer_sweep_S.png").exists()
    # CLI path used on the cluster
    cli.main(["analyze", "--run-dir", str(tmp), "--block", "tables"])
    cli.main(["report", "--run-dir", str(tmp)])
