"""NATURAL ERRORS - does the injection probe transfer to mistakes the model made BY ITSELF?

Everything before this used injected errors.  Here the errors are the model's own arithmetic slips in its greedy solution, found deterministically: a line with  a (+ | - | * | x | \\times | \\cdot) b = c  where a, b, c are plain
integers, the equation is a complete two-operand step (not part of a longer chain) and c differs from the exact value.  Controls are CORRECT equations of the same form in the same solutions.
For every event the hidden state at the LAST TOKEN OF c (exactly where the injection probe reads, nothing after it exists) is stored (one forward per solution).
Label of a slip (an intervention on the natural error): the slip is CORRECTED in the text and the model continues greedily; matters = the final answer differs from the original final answer
(a slip that the chain recovers from anyway has matters = 0; a slip that the final answer carries has matters = 1).  Also stored: whether the original final answer was correct.
Analysis (analyze): the injection-trained flip probe (fitted on injected rows of TRAINING problems only) is applied to the natural states:
  N1  among slips: AUROC of the probe score for `matters`, with problem-clustered CI, against two observable baselines (relative position of the slip in the solution; relative size of the slip)
  N2  slip vs matched CORRECT equations of the same solution: paired difference of probe scores (sign-flip permutation over solutions), Cohen's d, AUROC
Resumable per (shard, chunk): <out>/n<shard>_<chunk>.{npz,jsonl}.
"""
import json
import re
import zlib
from pathlib import Path
from typing import List

import numpy as np

from .hf_util import enc, generate_ids
from .prompts import chat_prompt, plain_text
from .sched import Budget, schedule
from .steps import end_token_index
from .textutil import parse_answer

NUM = r"(\d{1,3}(?:,\d{3})+|\d+)"
OP = r"(\+|-|−|–|\*|×|\\times|\\cdot|·)"
EQ = re.compile(rf"(?<![\w.,]){NUM}\s*{OP}\s*{NUM}\s*=\s*{NUM}(?![\w,]|\.\d)")
PRE_OP = re.compile(r"(?:[+\-−–*×/÷·^(]|\\times|\\cdot|\\div|=)\s*$")
POST_OP = re.compile(r"^\s*(?:[+\-−–*×/÷·^]|\\times|\\cdot|\\div|\.\d|\))")
N_CTRL = 6


def val(s: str) -> int:
    return int(s.replace(",", ""))


def scan(text: str) -> List[dict]:
    """complete integer two-operand equations of a solution.  -> events with char span of the result, shown / true value and `wrong`."""
    ev = []
    for m in EQ.finditer(text):
        if PRE_OP.search(text[max(0, m.start() - 12):m.start()]) or POST_OP.match(text[m.end():m.end() + 12]):
            continue
        a, op, b, c = val(m.group(1)), m.group(2), val(m.group(3)), val(m.group(4))
        t = a + b if op == "+" else (a - b if op in ("-", "−", "–") else a * b)
        ev.append({"start": m.start(4), "end": m.end(4), "a": a, "b": b, "op": op, "shown": c, "true": t, "wrong": c != t, "comma": "," in m.group(4)})
    return ev


def fixed_prefix(text: str, e: dict):
    """text up to and including the line of the slip, with the result corrected, plus a newline; None when the slip is on the last line."""
    le = text.find("\n", e["end"])
    if le < 0:
        return None
    s = f"{e['true']:,}" if e["comma"] and e["true"] >= 0 else str(e["true"])
    return text[:e["start"]] + s + text[e["end"]:le] + "\n"


def pick(text: str):
    ev = scan(text)
    slips = [e for e in ev if e["wrong"]]
    if not slips:
        return [], []
    ok = [e for e in ev if not e["wrong"]]
    ok.sort(key=lambda e: zlib.crc32(f"nat{e['start']}_{e['shown']}".encode()))
    return slips, ok[:N_CTRL]


def run(problems, G, model, tok, out_dir: Path, shard=0, nshards=1, chunk=16, max_new=512, max_seqs=96, budget_s=0, log=print) -> int:
    import torch
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    bud = Budget(budget_s)
    dev = next(model.parameters()).device
    base = getattr(model, "model", model)
    layers, done = None, 0
    probs = [p for p in problems if p["id"] in G and G[p["id"]].get("plain_fin") and G[p["id"]].get("plain_ans") is not None]
    for p in probs:
        p["_text"] = tok.decode(G[p["id"]]["plain_ids"], skip_special_tokens=True)
    probs = [p for p in probs if pick(p["_text"])[0]]                              # only solutions that contain a slip
    probs.sort(key=lambda p: zlib.crc32(f"nat{p['id']}".encode()))
    log(f"[natural shard {shard}] {len(probs)} solutions with at least one arithmetic slip")
    for ds, k, batch in schedule(probs, shard, nshards, chunk):
        f = d / f"n{shard:02d}_{k:04d}.jsonl"
        if f.exists():
            continue
        if bud.spent():
            log(f"[natural shard {shard}] time budget reached - stopping cleanly (resubmit to continue)")
            break
        rows, Hs, jobs = [], [], []
        for p in batch:
            g, text = G[p["id"]], p["_text"]
            comp = g["plain_ids"]
            pr = enc(tok, chat_prompt(tok, plain_text(p)))
            slips, ctrl = pick(text)
            with torch.no_grad():
                hs = base(input_ids=torch.tensor([pr + comp], device=dev), output_hidden_states=True).hidden_states
            if layers is None:
                layers = list(range(4, len(hs), 4))
                if len(hs) - 1 not in layers:
                    layers.append(len(hs) - 1)
            for kind, evs in (("slip", slips), ("ctrl", ctrl)):
                for e in evs:
                    te = end_token_index(tok, comp, e["end"])
                    if te >= len(comp):
                        continue
                    row = {"id": p["id"], "kind": kind, "shown": e["shown"], "true": e["true"], "rel_err": abs(e["shown"] - e["true"]) / max(abs(e["true"]), 1), "t_end": te, "n_tok": len(comp),
                           "frac": te / len(comp), "ans_orig": g["plain_ans"], "orig_correct": float(g["plain_ans"] == p["gold"])}
                    pre = fixed_prefix(text, e) if kind == "slip" else None
                    rows.append(row)
                    Hs.append(torch.stack([hs[l][0, len(pr) + te - 1] for l in layers]).half().cpu().numpy())
                    if pre is not None:
                        jobs.append((len(rows) - 1, p, pre))
        if jobs:
            out = generate_ids(model, tok, [chat_prompt(tok, plain_text(p)) + pre for _, p, pre in jobs], max_new, 1, 0.0, 0.95, max_seqs)
            for (ri, p, pre), res in zip(jobs, out):
                ids, fin = res[0]
                ans = parse_answer(pre + tok.decode(ids, skip_special_tokens=True), p["kind"])
                rows[ri].update({"ans_fix": ans, "fin_fix": bool(fin), "matters": float(ans != rows[ri]["ans_orig"])})
        if rows:
            np.savez(f.with_suffix(".npz"), H=np.stack(Hs), layers=np.array(layers))
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.rename(f)
        done += len(batch)
        log(f"[natural shard {shard}] chunk {k}: {len(batch)} solutions, {len(rows)} events ({done} solutions this run)")
    return done


def load(out_dir: Path):
    rows, H, layers = [], [], None
    for f in sorted(Path(out_dir).glob("n*_*.jsonl")):
        lines = [json.loads(l) for l in open(f) if l.strip()]
        if not lines:
            continue
        z = np.load(f.with_suffix(".npz"))
        rows += lines
        H.append(z["H"])
        layers = z["layers"]
    return rows, (np.concatenate(H) if H else None), layers


def analyze(run_dir: Path, mega_dir: Path, out_dir: Path, ds: str = "gsm8k", B: int = 1000, seed: int = 0, log=print) -> dict:
    from . import cv_eval, ridge
    from .inject import prepare
    from .robust import auroc, regime_fit
    rows, H, nl = load(out_dir)
    if H is None or sum(r["kind"] == "slip" for r in rows) < 20:
        return {"status": "n/a", "why": f"only {sum(r['kind'] == 'slip' for r in rows)} natural slips found"}
    P, why = prepare(run_dir, mega_dir, ds)
    if P is None:
        return {"status": "n/a", "why": why}
    f = regime_fit(P, P["bl"], "iid")
    df, ly = P["df"], list(P["ly"])
    layer = f["layer"]
    li = ly.index(layer)
    nl = list(nl)
    if layer not in nl:
        return {"status": "n/a", "why": f"layer {layer} missing in the natural states"}
    y = f["y"]
    Xp = P["HA"][:, li, :].astype(np.float32)
    res = cv_eval.cv_probe(lambda l: Xp, [0], y[:, None], f["pool"], f["test"], df.id.values)[0]
    ri = ridge.RATIOS.index(res["ratio"])
    fit = ridge.fit_layer(cv_eval._t(Xp[f["pool"]]), y[f["pool"]][:, None])
    score = fit.predict(cv_eval._t(H[:, nl.index(layer), :].astype(np.float32)), 0).cpu().numpy()[:, ri]
    from .bench import read_problems
    split = {p["id"]: p["split"] for p in read_problems(run_dir, [ds])}
    seen = np.array([split.get(r["id"]) in ("fit", "val") for r in rows])
    kind = np.array([r["kind"] for r in rows])
    ids = np.array([r["id"] for r in rows])
    rng = np.random.RandomState(seed)
    out = {"status": "ok", "ds": ds, "layer": layer, "n_slips": int((kind == "slip").sum()), "n_ctrl": int((kind == "ctrl").sum()), "n_solutions": int(len(np.unique(ids))),
           "share_slips_matter": float(np.mean([r["matters"] for r in rows if r["kind"] == "slip" and "matters" in r])) if any("matters" in r for r in rows) else float("nan")}

    def clustered_auc(yv, sv, gv):
        ug, inv = np.unique(gv, return_inverse=True)
        rws = [np.where(inv == i)[0] for i in range(len(ug))]
        bs = []
        for _ in range(B):
            ix = np.concatenate([rws[i] for i in rng.randint(0, len(ug), len(ug))])
            if len(set(yv[ix] >= 0.5)) == 2:
                bs.append(auroc(yv[ix], sv[ix]))
        return auroc(yv, sv), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    for name, use in (("unseen_problems", ~seen), ("all_problems", np.ones(len(rows), bool))):
        sl = (kind == "slip") & use & np.array(["matters" in r for r in rows])
        if sl.sum() < 20 or len(set(np.array([rows[i]["matters"] for i in np.where(sl)[0]]) >= 0.5)) < 2:
            out[name] = {"n_slips": int(sl.sum()), "note": "too few slips / single-class label"}
            continue
        ix = np.where(sl)[0]
        ym = np.array([rows[i]["matters"] for i in ix])
        base = {"relative position": np.array([rows[i]["frac"] for i in ix]), "relative size of the slip": np.log1p(np.array([rows[i]["rel_err"] for i in ix]))}
        n1 = {"n_slips": int(sl.sum()), "n_solutions": int(len(np.unique(ids[ix]))), "share_matters": float(ym.mean()), "probe": clustered_auc(ym, score[ix], ids[ix]),
              "baselines": {k: clustered_auc(ym, v, ids[ix]) for k, v in base.items()}}
        # N2: slips vs matched correct equations of the same solution
        pairs = []
        for i in np.unique(ids[use & (kind == "slip")]):
            a, b = score[(ids == i) & (kind == "slip") & use], score[(ids == i) & (kind == "ctrl") & use]
            if len(a) and len(b):
                pairs.append(a.mean() - b.mean())
        if len(pairs) >= 10:
            pairs = np.array(pairs)
            obs = pairs.mean()
            perm = np.array([(pairs * rng.choice([-1, 1], len(pairs))).mean() for _ in range(5000)])
            sd = np.concatenate([score[(kind == "slip") & use], score[(kind == "ctrl") & use]]).std()
            n1["slip_vs_control"] = {"n_solutions": int(len(pairs)), "mean_diff": float(obs), "cohen_d": float(obs / sd) if sd > 0 else float("nan"), "p_signflip": float((np.abs(perm) >= abs(obs)).mean()),
                                     "auroc_slip_vs_ctrl": auroc((kind[use] == "slip").astype(float), score[use])}
        out[name] = n1
    log(f"  natural {ds}: {out['n_slips']} slips in {out['n_solutions']} solutions; share that matter {out['share_slips_matter']:.2f}")
    return out
