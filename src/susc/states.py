"""Prompt-end hidden states (every layer) + forced-answer log-probs, per condition.

For each (problem, condition) one forward over  prompt + CUE + "<answer>}"  gives
  * the residual stream at the LAST PROMPT TOKEN (before the cue: the model has seen everything and produced nothing)
  * log P("<answer>}" | prompt + cue)  for answer = gold AND answer = the planted wrong value  (the model's direct, no-CoT belief)
"""
from pathlib import Path

import numpy as np

from .hf_util import enc, hidden_and_logprobs, pad_id
from .prompts import CONDS, CUE, chat_prompt, user_text
from .sched import Budget, schedule


def run(problems, model, tok, out_dir: Path, shard=0, nshards=1, conds=CONDS, fwd_bs=8, chunk=128, tag="", budget_s=0, log=print) -> int:
    d = Path(out_dir) / "states"
    d.mkdir(parents=True, exist_ok=True)
    pad = pad_id(tok)
    cue_ids = enc(tok, CUE)
    bud = Budget(budget_s)
    done = 0
    for ds, k, batch in schedule(problems, shard, nshards, chunk):
        f = d / f"x{shard:02d}_{ds}_{k:04d}{tag}.npz"
        if f.exists():
            continue
        if bud.spent():
            log("[states] time budget reached - stopping cleanly")
            break
        pack = {"ids": np.array([p["id"] for p in batch])}
        for cond in conds:
            jobs = []        # (problem_index, which, seq, state_pos, cont_start)
            for bi, p in enumerate(batch):
                pr = enc(tok, chat_prompt(tok, user_text(p, cond)))
                whichs = [("gold", p["gold"])] + ([] if cond == "cor" else [("hint", p["hint"])])
                for which, ans in whichs:
                    seq = pr + cue_ids + enc(tok, f"{ans}}}")
                    jobs.append((bi, which, seq, len(pr) - 1, len(pr) + len(cue_ids)))
            states = None
            lps = {"gold": np.full(len(batch), np.nan), "hint": np.full(len(batch), np.nan)}
            ntok = np.zeros(len(batch), dtype=np.int32)
            order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][2]))
            for s in range(0, len(order), fwd_bs):
                sel = [jobs[j] for j in order[s:s + fwd_bs]]
                st, lp = hidden_and_logprobs(model, [j[2] for j in sel], [j[3] for j in sel], [j[4] for j in sel], pad)
                for r, (bi, which, seq, sp, cs) in enumerate(sel):
                    lps[which][bi] = lp[r]
                    ntok[bi] = sp + 1
                    if which == "gold":
                        if states is None:
                            states = np.zeros((len(batch),) + tuple(st.shape[1:]), dtype=np.float16)
                        states[bi] = st[r].numpy().astype(np.float16)
            if cond == "cor":
                lps["hint"] = lps["gold"].copy()                 # hint == gold by construction
            pack[f"{cond}_states"] = states
            pack[f"{cond}_lpg"] = lps["gold"]
            pack[f"{cond}_lph"] = lps["hint"]
            pack[f"{cond}_ntok"] = ntok
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, **pack)
        tmp.rename(f)
        done += len(batch)
        log(f"[states shard {shard}] {ds} chunk {k}: +{len(batch)} problems ({done} this run)")
    return done
