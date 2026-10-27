"""THREAD 5 - error propagation AT INJECTION TIME.  Does the hidden state, right after a wrong number has been written into the model's own chain of thought,
already say whether the final answer will FOLLOW the error or silently RECOVER?

Re-uses the intervention outcomes of the mega steps thread (no new generation): for every edited step (n1: result c -> c+1, n2: c -> c+max(10, c//10)) we already have the final answer
after the model continued greedily.  This stage runs ONE forward over  prompt + edited prefix  and stores
   HA   hidden states (layers every 4 + last) at the LAST TOKEN OF THE INJECTED NUMBER   (primary: the moment of injection, nothing after it exists yet)
   HB   hidden states at the end of the edited line                                       (secondary)
plus FREE observable signals of the model's own error awareness: the surprise (NLL, min log-prob, first-token entropy) of the injected number and the surprise of the original number
at the same place (from one forward over the original solution).  A probe has to beat these.
Labels (deterministic, from the intervention):   flip = final answer differs from the unedited solution's answer (no parsable answer counts as a flip)
                                                  prop = log10(1 + |final answer - original final answer|)   how far the error propagated (NaN when an answer is unparsable)
Resumable chunks  i<shard>_<chunk>.{npz,jsonl}  (jsonl last = completion marker), same discipline as steps.py.
"""
import glob
import json
import re
import zlib
from pathlib import Path
from typing import Dict, List

import numpy as np

from .hf_util import enc
from .prompts import chat_prompt, plain_text
from .sched import Budget, schedule
from .steps import candidate_steps, edit_value, edited_prefix, end_token_index

LAYER_STEP = 4
VARIANTS = ("n1", "n2")


def num_tokens(tok, suffix_ids: List[int], c0: int, c1: int):
    """(first, last) token indices (inside suffix_ids) of the tokens that cover characters [c0, c1)."""
    first = end_token_index(tok, suffix_ids, c0 + 1) - 1
    last = end_token_index(tok, suffix_ids, c1) - 1
    return first, max(last, first)


def edit_geometry(text: str, s: dict, v: str):
    """-> (edited prefix incl. the whole edited line + newline, new number string, char offsets [c0, c1) of the injected number inside that prefix)."""
    pre = edited_prefix(text, s, v)
    nv = edit_value(s["value"], v)
    nvs = f"{nv:,}" if s["comma"] else str(nv)
    c0 = s["start"] + s["span"][0]
    return pre, nvs, c0, c0 + len(nvs)


def span_stats(model, h_last, ids: List[int], a: int, b: int) -> dict:
    """surprise of tokens ids[a..b] (absolute indices) under the model, from the final-norm hidden states h_last [T, D]: NLL sum/mean, min log-prob, entropy and probability at the first token."""
    import torch
    head = model.get_output_embeddings()
    with torch.no_grad():
        pos = torch.arange(a - 1, b, device=h_last.device)
        lp = torch.log_softmax(head(h_last[pos]).float(), dim=-1)
        tgt = torch.tensor(ids[a:b + 1], device=h_last.device)
        tl = lp[torch.arange(len(tgt), device=h_last.device), tgt]
        p0 = lp[0].exp()
    return {"nll_sum": float(-tl.sum()), "nll_mean": float(-tl.mean()), "minlp": float(tl.min()), "lp0": float(tl[0]), "ent0": float(-(p0 * lp[0]).sum()), "n_tok": int(len(tgt))}


def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def propagation(ans_var, ans_orig) -> float:
    a, b = _num(ans_var), _num(ans_orig)
    return float("nan") if np.isnan(a) or np.isnan(b) else float(np.log10(1.0 + abs(a - b)))


def step_rows(steps_dir: Path) -> Dict[int, List[dict]]:
    """rows of the steps thread that have both edit outcomes, grouped by problem."""
    out: Dict[int, List[dict]] = {}
    for f in sorted(glob.glob(str(Path(steps_dir) / "s*.jsonl"))):
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            if "ans_n1" in r and "ans_n2" in r:
                out.setdefault(r["id"], []).append(r)
    return out


def run(problems, G, srows, model, tok, out_dir: Path, shard=0, nshards=1, chunk=16, budget_s=0, log=print) -> int:
    import torch
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    bud = Budget(budget_s)
    dev = next(model.parameters()).device
    base = getattr(model, "model", model)
    layers = None
    done = 0
    probs = [p for p in problems if p["id"] in srows and p["id"] in G and G[p["id"]].get("plain_ids")]
    probs.sort(key=lambda p: zlib.crc32(f"inj{p['id']}".encode()))
    for ds, k, batch in schedule(probs, shard, nshards, chunk):
        f = d / f"i{shard:02d}_{k:04d}.jsonl"
        if f.exists():
            continue
        if bud.spent():
            log(f"[inject shard {shard}] time budget reached - stopping cleanly (resubmit to continue)")
            break
        rows, HA, HB, skipped = [], [], [], 0
        for p in batch:
            comp = G[p["id"]]["plain_ids"]
            text = tok.decode(comp, skip_special_tokens=True)                 # identical to steps.run: the stored offsets refer to this string
            cm = {(c["start"], c["end"]): c for c in candidate_steps(text, 10 ** 9)}
            prompt = chat_prompt(tok, plain_text(p))
            pr = enc(tok, prompt)
            with torch.no_grad():
                h0 = base(input_ids=torch.tensor([pr + comp], device=dev), output_hidden_states=True).hidden_states
            hl0 = h0[-1][0]
            for r in srows[p["id"]]:
                s = cm.get((r["start"], r["end"]))
                if s is None:
                    skipped += 1
                    continue
                c0 = s["start"] + s["span"][0]
                f0, l0 = num_tokens(tok, comp, c0, s["start"] + s["span"][1])
                if l0 >= len(comp):
                    skipped += 1
                    continue
                orig = span_stats(model, hl0, pr + comp, len(pr) + f0, len(pr) + l0)
                for v in VARIANTS:
                    pre = edited_prefix(text, s, v)
                    nv = edit_value(s["value"], v)
                    nvs = f"{nv:,}" if s["comma"] else str(nv)
                    ids = enc(tok, prompt + pre)
                    if ids[:len(pr)] != pr:                                   # prompt/prefix boundary tokenises differently: skip rather than misalign
                        skipped += 1
                        continue
                    suf = ids[len(pr):]
                    fi, la = num_tokens(tok, suf, c0, c0 + len(nvs))
                    if la >= len(suf):
                        skipped += 1
                        continue
                    with torch.no_grad():
                        hs = base(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states
                    if layers is None:
                        layers = list(range(LAYER_STEP, len(hs), LAYER_STEP))
                        if len(hs) - 1 not in layers:
                            layers.append(len(hs) - 1)
                    A, Bp = len(pr) + la, len(ids) - 1
                    HA.append(torch.stack([hs[l][0, A] for l in layers]).half().cpu().numpy())
                    HB.append(torch.stack([hs[l][0, Bp] for l in layers]).half().cpu().numpy())
                    inj = span_stats(model, hs[-1][0], ids, len(pr) + fi, A)
                    rows.append({"id": p["id"], "k": r["k"], "var": v, "value": s["value"], "new_value": nv, "delta": nv - s["value"], "reuse": r["reuse"], "n_steps": r["n_steps"], "t_end": r["t_end"],
                                 "n_tok": r["n_tok"], "start": r["start"], "end": r["end"], "ans_orig": r["ans_orig"], "ans_var": r[f"ans_{v}"], "line": s["line"], "ptext": pre, "flip": float(r[f"flip_{v}"]),
                                 "flip_step": float((r["flip_n1"] + r["flip_n2"]) / 2), "prop": propagation(r[f"ans_{v}"], r["ans_orig"]),
                                 **{f"inj_{a}": b for a, b in inj.items()}, **{f"orig_{a}": b for a, b in orig.items()}})
        if rows:
            np.savez(f.with_suffix(".npz"), HA=np.stack(HA), HB=np.stack(HB), layers=np.array(layers))
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.rename(f)
        done += len(batch)
        log(f"[inject shard {shard}] chunk {k}: {len(batch)} problems, {len(rows)} injected rows ({skipped} skipped) ({done} problems this run)")
    return done


def load(inject_dir: Path):
    rows, A, B, layers = [], [], [], None
    for f in sorted(glob.glob(str(Path(inject_dir) / "i*.jsonl"))):
        lines = [json.loads(l) for l in open(f) if l.strip()]
        if not lines:
            continue
        z = np.load(Path(f).with_suffix(".npz"))
        rows += lines
        A.append(z["HA"])
        B.append(z["HB"])
        layers = z["layers"]
    return rows, (np.concatenate(A) if A else None), (np.concatenate(B) if B else None), layers


# ----------------------------------------------------------------------------- analysis
MIN_STD = 0.1
NOISE_MAX = 0.15


def feature_blocks(df, q_of):
    """observable baselines.  Everything here is available at the moment of injection (the edited prefix and the model's own probabilities); nothing from the continuation."""
    from . import table
    from .mega import step_text_feats
    from .trace import prefix_feats
    pos = np.stack([df.k, np.log1p(df.k), df.t_end / 100.0, np.log1p(df.t_end)], 1).astype(np.float32)
    ed = np.stack([(df["var"] == "n2").astype(float), np.log1p(df.delta.abs()), np.log1p(df.value), (df.delta / df.value.clip(lower=1)).clip(upper=50), np.log10(df.new_value.clip(lower=1))], 1).astype(np.float32)
    stp = np.array([step_text_feats(l, v, q) for l, v, q in zip(df.line, df.value, q_of)], dtype=np.float32)
    qf = np.array([table.question_features(q) for q in q_of], dtype=np.float32)
    pf = np.array([prefix_feats(t) for t in df.ptext], dtype=np.float32)
    sp = np.stack([df.inj_nll_sum, df.inj_nll_mean, df.inj_minlp, df.inj_ent0, df.orig_nll_sum, df.orig_nll_mean, df.orig_ent0, df.inj_nll_sum - df.orig_nll_sum, df.inj_lp0 - df.orig_lp0,
                   df.inj_n_tok], 1).astype(np.float32)
    orc = np.nan_to_num(np.stack([df.reuse, np.log1p(df.reuse), df.k / df.n_steps, np.log1p(df.n_tok - df.t_end)], 1).astype(np.float32))
    return pos, ed, stp, qf, pf, sp, orc


def prepare(run_dir: Path, mega_dir: Path):
    """-> (dict P, None) with the aligned frame, states and baseline feature blocks, or (None, reason) when there is not enough data."""
    import pandas as pd
    from . import mega
    from .bench import read_problems
    rows, HA, HB, layers = load(Path(mega_dir) / "inject")
    if HA is None or len(rows) < 400:
        return None, f"only {len(rows)} injected rows - the inject stage did not produce enough data"
    srows, Hs, slayers = mega.load_steps(Path(mega_dir) / "steps")
    probs = {p["id"]: p for p in read_problems(run_dir, ["gsm8k"])}
    df = pd.DataFrame(rows)
    # natural (pre-edit) state of the same step from the steps thread: the competitor "the state already knew before the error was written"
    idx = {(r["id"], r["k"]): j for j, r in enumerate(srows)}
    j = np.array([idx.get((i, k), -1) for i, k in zip(df.id, df.k)])
    keep = j >= 0
    if keep.sum() < 400:
        return None, f"only {int(keep.sum())} injected rows could be matched to a step-thread state"
    df, HA, HB, j = df[keep].reset_index(drop=True), HA[keep], HB[keep], j[keep]
    df["split"] = [probs[i]["split"] for i in df.id]
    df["d1"] = [probs[i]["d1"] for i in df.id]
    q = [probs[i]["question"] for i in df.id]
    pos, ed, stp, qf, pf, sp, orc = feature_blocks(df, q)
    sly, ly = list(slayers), list(layers)
    pre = lambda l: Hs[j, sly.index(l), :].astype(np.float32)                    # noqa: E731
    obs = np.concatenate([pos, ed, stp, qf, pf, sp], 1)
    bl = {"Bpos": pos, "Bedit": np.concatenate([pos, ed], 1), "Bsurprise": np.concatenate([ed, sp], 1), "Btext": np.concatenate([pos, ed, stp, qf, pf], 1), "Bobs-all": obs,
          "Bfuture(oracle)": np.concatenate([obs, orc], 1)}
    if 8 in ly:
        bl["Hpost8(early-layer)"] = HA[:, ly.index(8), :].astype(np.float32)
    for l in (8, 16):
        if l in sly:
            bl[f"Hpre{l}(natural-state)"] = pre(l)
    return {"df": df, "HA": HA, "HB": HB, "ly": ly, "srows": srows, "Hs": Hs, "sly": sly, "j": j, "q": q, "bl": bl, "obs": obs, "orc": orc, "pre": pre}, None


def analyze(run_dir: Path, mega_dir: Path, log=print) -> dict:
    import pandas as pd
    from . import core, cv_eval, mega
    P, why = prepare(run_dir, mega_dir)
    if P is None:
        return {"status": "n/a", "why": why, "md": "(not enough data)\n", "statuses": []}
    df, HA, HB, ly, srows, sly, bl, obs, orc, pre = (P[k] for k in ("df", "HA", "HB", "ly", "srows", "sly", "bl", "obs", "orc", "pre"))
    targets = ["flip", "prop", "flip_step"]
    Y = df[targets].values.astype(np.float64)
    nulls = [r["flip_null"] for r in srows if "flip_null" in r]
    noise = float(np.mean(nulls)) if nulls else float("nan")
    log(f"inject: {len(df)} rows from {df.id.nunique()} problems; step-thread noise floor {noise:.3f}; label means {dict(zip(targets, np.nanmean(Y, 0).round(3)))}")
    out = []
    for H, pfx in ((HA, "I[num]"), (HB, "I[line]")):
        res = cv_eval.run_block(df.split.values, df.d1.values, df.id.values, lambda li, H=H: H[:, li, :].astype(np.float32), list(range(len(ly))), bl, Y, targets, prefix=pfx, log=log)
        for r in res:
            r["probe_layer"] = ly[r["probe_layer"]]
        out += res
    pl = [l for l in sly if l % 4 == 0 or l == sly[-1]]                         # the pre-injection natural-state probe, for the 'did the injection add anything' comparison
    resp = cv_eval.run_block(df.split.values, df.d1.values, df.id.values, lambda li: pre(pl[li]), list(range(len(pl))), {"Bobs-all": obs, "Bfuture(oracle)": np.concatenate([obs, orc], 1)}, Y, targets, prefix="Pre[nat]", log=log)
    for r in resp:
        r["probe_layer"] = pl[r["probe_layer"]]
    out += resp
    rdf = mega.rows_df(out)
    sts = []
    for pfx in ("I[num]", "I[line]", "Pre[nat]"):
        for t in targets:
            deg = ""
            sd = float(np.nanstd(Y[:, targets.index(t)]))
            if sd < MIN_STD:
                deg = f"label std {sd:.3f} < {MIN_STD}: nearly constant, nothing to probe"
            elif noise > NOISE_MAX:
                deg = f"noise floor {noise:.3f} > {NOISE_MAX}"
            sts.append({"thread": "inject", "label": f"{pfx}:{t}", **mega.generic_status(rdf, pfx, t, deg)})
    by_k = df.groupby("k").agg(n=("id", "size"), flip=("flip", "mean"), prop=("prop", "mean"), surprise_nll=("inj_nll_sum", "mean")).reset_index()
    by_v = df.groupby("var").agg(n=("id", "size"), flip=("flip", "mean"), prop=("prop", "mean"), injected_nll=("inj_nll_sum", "mean"), orig_nll=("orig_nll_sum", "mean")).reset_index()
    md = [f"{len(df)} injected rows (2 edits per step) from {df.id.nunique()} problems. Folds and bootstraps are over problems. Step-thread noise floor (unedited regeneration flips): {noise:.3f}.", "",
          "- label means: " + ", ".join(f"{t} {m:.3f}" for t, m in zip(targets, np.nanmean(Y, 0))) + f"; Spearman(flip, model's own surprise at the injected number) = {core.spearman(df.flip.values, df.inj_nll_sum.values):.3f}",
          f"- mean NLL of the injected number {df.inj_nll_sum.mean():.2f} vs of the original number {df.orig_nll_sum.mean():.2f} (does the model notice the error at all?)", "",
          "By variant:", "", core.md(by_v), "By step index:", "", core.md(by_k), mega.tab(rdf)]
    return {"status": "ok", "md": "\n".join(md) + "\n", "statuses": sts, "results": rdf}


# ----------------------------------------------------------------------------- report
RULE = """Pre-registered decision (docs/INJECT.md): the headline claim is  I[num]:flip  (probe on the state at the last token of the injected number).
SUPPORTED needs a probe R2 > 0.05 with a bootstrap-over-problems CI of the gain over the best OBSERVABLE baseline above 0, in iid AND under easy->hard shift, where the observables include
the model's own surprise at the injected number, step text, position, edit size, an early layer, and the natural pre-edit state of the same step.
PARTIAL = one regime.  NEGATIVE = gain ruled out above 0.05.  Everything else INCONCLUSIVE.  I[line] and Pre[nat] are secondary and reported without a decision."""


def write_report(run_dir: Path, mega_dir: Path, log=print, big_name: str = "steer_big") -> Path:
    import traceback
    import pandas as pd
    from . import core, steer
    outd = Path(mega_dir) / "inject" / "analysis"
    outd.mkdir(parents=True, exist_ok=True)
    secs, sts = [], []
    try:
        a = analyze(run_dir, mega_dir, log)
        secs.append(("Error propagation at injection time", a["md"] if a["status"] == "ok" else f"**{a['status']}**: {a.get('why', '')}\n"))
        sts += a["statuses"]
        if a.get("results") is not None and len(a["results"]):
            a["results"].to_csv(outd / "inject_results.csv", index=False)
    except Exception:                                                                      # noqa: BLE001  one failing analysis must not kill the report
        log(traceback.format_exc())
        secs.append(("Error propagation at injection time", "analysis raised an exception:\n\n```\n" + traceback.format_exc() + "\n```\n"))
        sts.append({"thread": "inject", "label": "(thread crashed)", "status": "INCONCLUSIVE", "why": "analysis raised an exception"})
    big = Path(mega_dir) / big_name
    try:
        r = steer.analyze(big, dmg_alpha=8.0)
        body = f"**{r['status']}**: {r['why']}\n\n"
        if r.get("slopes"):
            body += core.md(pd.DataFrame([{"family": k, "slope_log_tokens_per_sd": v[0], "lo95": v[1], "hi95": v[2]} for k, v in r["slopes"].items()]))
            body += "\nPer condition:\n\n" + core.md(r["table"])
            if r.get("dose") is not None and len(r["dose"]):
                body += "\nDose-response (probe and mean-difference vs the mean |effect| of random directions at the same dose):\n\n" + core.md(r["dose"])
        sts.append({"thread": "steer_big", "label": "length direction at large doses", "status": r["status"], "why": r["why"]})
        secs.append(("Steering at large doses", body))
    except Exception:                                                                      # noqa: BLE001
        log(traceback.format_exc())
        secs.append(("Steering at large doses", "analysis raised an exception:\n\n```\n" + traceback.format_exc() + "\n```\n"))
    sdf = pd.DataFrame(sts)
    if len(sdf):
        sdf.to_csv(outd / "inject_status.csv", index=False)
    head = ["# Inject run - error propagation at injection time (+ steering at large doses), GSM8K, Qwen2.5-7B-Instruct", "", RULE, "", "## Verdict", ""]
    if len(sdf):
        main = sdf[sdf.label == "I[num]:flip"]
        head += [f"- **I[num]:flip: {main.iloc[0].status}** - {main.iloc[0].why}" if len(main) else "- headline row missing"]
        for lab in ("I[num]:prop", "I[line]:flip", "Pre[nat]:flip"):
            m = sdf[sdf.label == lab]
            if len(m):
                head.append(f"- {lab}: {m.iloc[0].status} - {m.iloc[0].why}")
        for m in sdf[sdf.thread == "steer_big"].itertuples():
            head.append(f"- steering at large doses: {m.status} - {m.why}")
        head += ["", core.md(sdf[[c for c in ("thread", "label", "status", "why", "iid_probe_r2", "iid_obs_r2", "iid_delta", "iid_lo", "iid_hi", "shift_probe_r2", "shift_obs_r2", "shift_delta", "shift_lo", "shift_hi") if c in sdf.columns]])]
    txt = "\n".join(head) + "\n" + "".join(f"\n## {t}\n\n{b}\n" for t, b in secs)
    p = outd / "INJECT_REPORT.md"
    p.write_text(txt)
    log(f"wrote {p}")
    return p
