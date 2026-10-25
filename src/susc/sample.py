"""Behavioural labels by sampling: n answers per (problem, condition) at T>0.  Resumable chunk files, one process per GPU."""
import json
import math
from collections import Counter
from pathlib import Path
from typing import List

import torch

from .hf_util import generate
from .prompts import CONDS, chat_prompt, hint_value, parse, user_text
from .sched import Budget, schedule

MAX_NEW = {"gsm8k": 512, "math": 768, "arc": 384, "arith": 512, "prog": 512}


def answer_entropy(answers: List) -> float:
    c = Counter(answers)
    n = sum(c.values())
    return -sum(v / n * math.log(v / n) for v in c.values())


def run(problems, model, tok, out_dir: Path, shard=0, nshards=1, conds=CONDS, n=8, max_new: int = 0, temperature=0.7, top_p=0.95,
        chunk=64, max_seqs=192, tag="", budget_s=0, sub="", log=print) -> int:
    d = Path(out_dir) / "samples"
    d.mkdir(parents=True, exist_ok=True)
    bud = Budget(budget_s)
    torch.manual_seed(1234 + 17 * shard)                  # reproducible per shard up to batching nondeterminism
    done = 0
    for ds, k, batch in schedule(problems, shard, nshards, chunk, sub):
        f = d / f"s{shard:02d}_{ds}_{k:04d}{tag}.jsonl"
        if f.exists():
            continue
        if bud.spent():
            log(f"[sample shard {shard}] time budget reached - stopping cleanly (resubmit to continue)")
            break
        mn = max_new or MAX_NEW.get(ds, 512)
        rows = []
        for cond in conds:
            prompts = [chat_prompt(tok, user_text(p, cond)) for p in batch]
            outs = generate(model, tok, prompts, n, mn, temperature, top_p, max_seqs)
            for p, texts in zip(batch, outs):
                ans = [parse(t, p["kind"]) for t in texts]
                rows.append({"id": p["id"], "ds": ds, "cond": cond, "n": n,
                             "n_gold": sum(a == p["gold"] for a in ans), "n_wrong": sum(a == p["hint"] for a in ans), "n_none": sum(a is None for a in ans),
                             "ent": answer_entropy(ans), "mean_chars": sum(len(t) for t in texts) / n, "answers": ans, "t0": texts[0],
                             "hv": hint_value(p, cond)})
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.rename(f)
        done += len(batch)
        log(f"[sample shard {shard}] {ds} chunk {k}: +{len(batch)} problems ({done} this run)")
    return done
