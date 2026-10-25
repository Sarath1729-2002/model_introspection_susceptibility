"""Monitoring along the reasoning trace.  For ONE sampled solution per (problem, condition) [sample 0, text stored by sample.py] we re-run the
model on  prompt + that solution  and read the residual stream at five points:  p0 = last prompt token (the pre-generation state),
p25 / p50 / p75 = after that fraction of the solution's tokens, pend = just BEFORE the final \\boxed (the answer is not yet written).
The label (did this trajectory end on the planted wrong answer?) is deterministic.  Only a layer subset is stored (every 4th)."""
import json
from pathlib import Path

import numpy as np

from .hf_util import enc, hidden_at_positions, pad_id
from .prompts import TRACE_CONDS, chat_prompt, user_text
from .sched import Budget, schedule

FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)
POS_NAMES = ("p0", "p25", "p50", "p75", "pend")
MAX_COMP = 640


def trace_layers(nstates: int):
    ls = [l for l in range(4, nstates, 4)]
    return ls or [nstates - 1]


def positions_for(pr_len: int, comp_ids, text: str, tok):
    Tc = len(comp_ids)
    base = pr_len - 1
    pos = [base]
    for f in (0.25, 0.5, 0.75):
        pos.append(base + min(max(1, int(f * Tc)), Tc))
    i = text.rfind("\\boxed")
    if i >= 0:
        n_pre = len(enc(tok, text[:i]))
        pos.append(base + min(max(1, n_pre), Tc))
    else:
        pos.append(base + Tc)
    return pos


def run(problems, model, tok, out_dir: Path, shard=0, nshards=1, conds=TRACE_CONDS, fwd_bs=4, chunk=64, tag="", budget_s=0, log=print) -> int:
    out_dir = Path(out_dir)
    d = out_dir / "trace"
    d.mkdir(parents=True, exist_ok=True)
    pad = pad_id(tok)
    bud = Budget(budget_s)
    done = 0
    for ds, k, batch in schedule(problems, shard, nshards, chunk):
        f = d / f"t{shard:02d}_{ds}_{k:04d}{tag}.npz"
        sfile = out_dir / "samples" / f"s{shard:02d}_{ds}_{k:04d}{tag}.jsonl"
        if f.exists() or not sfile.exists():
            continue
        if bud.spent():
            log("[trace] time budget reached - stopping cleanly")
            break
        t0 = {}
        for l in open(sfile):
            r = json.loads(l)
            t0[(r["id"], r["cond"])] = r["t0"]
        pack = {"ids": np.array([p["id"] for p in batch])}
        layers = None
        for cond in conds:
            jobs = []
            for bi, p in enumerate(batch):
                pr = enc(tok, chat_prompt(tok, user_text(p, cond)))
                text = t0[(p["id"], cond)]
                comp = enc(tok, text)[:MAX_COMP]
                jobs.append((bi, pr + comp, positions_for(len(pr), comp, text, tok), len(comp), len(pr)))
            H, order = None, sorted(range(len(jobs)), key=lambda j: len(jobs[j][1]))
            lens = np.zeros(len(batch), dtype=np.int32)
            fr = np.zeros((len(batch), len(FRACS)), dtype=np.float32)
            for s in range(0, len(order), fwd_bs):
                sel = [jobs[j] for j in order[s:s + fwd_bs]]
                if layers is None:
                    layers = trace_layers(model.config.num_hidden_layers + 1)
                h = hidden_at_positions(model, [j[1] for j in sel], [j[2] for j in sel], layers, pad)
                if H is None:
                    H = np.zeros((len(batch),) + tuple(h.shape[1:]), dtype=np.float16)
                for r, (bi, seq, pos, Tc, plen) in enumerate(sel):
                    H[bi] = h[r].numpy()
                    lens[bi] = Tc
                    fr[bi] = [(q - (plen - 1)) / max(Tc, 1) for q in pos]
            pack[f"h_{cond}"], pack[f"len_{cond}"], pack[f"frac_{cond}"] = H, lens, fr
        pack["layers"] = np.array(layers)
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, **pack)
        tmp.rename(f)
        done += len(batch)
        log(f"[trace shard {shard}] {ds} chunk {k}: +{len(batch)} ({done} this run)")
    return done
