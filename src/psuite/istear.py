"""CAUSAL TEST of the injection probe.  Is the direction the probe reads at the injected number something the model USES to decide follow-vs-recover, or only a correlate?

For a deterministic sample of TEST-problem injected rows (id, step, variant) the prompt is  chat prompt + solution up to and including the last character of the injected number.
A forward hook adds  alpha * s * unit_direction  to the residual stream of ONE layer at the LAST PROMPT TOKEN ONLY (prefill only: nothing is added while generating), i.e. exactly where the probe reads.
The model then continues greedily; flip = its final answer differs from the answer of its unedited solution.  All conditions (incl. alpha = 0) use this same protocol, so effects are paired per row.
Directions (norm-matched, alpha in units of s = sd of the probe projection over the training rows): the PROBE direction of the flip probe, the MEAN-DIFFERENCE direction (flip=1 minus flip=0), and random Gaussian directions.
Two layers: the CV-best probe layer and a mid layer (nearest to 12).  Doses are capped at 16 sd: the earlier steering run showed every direction, random ones included, breaks the model at 32 sd.
Resumable per (condition, shard).  Prior lesson built in: the slope is fitted ONLY over |alpha| <= 16 and integrity (finished rate, parsable-answer rate) is checked at |alpha| <= 8.
"""
import json
import zlib
from pathlib import Path
from typing import Dict, List

import numpy as np

from .hf_util import enc, generate_ids
from .inject import edit_geometry
from .prompts import chat_prompt, plain_text
from .sched import Budget
from .steps import candidate_steps
from .textutil import parse_answer

HI = (8.0, 16.0)
LO = (2.0, 4.0)
N_RAND = 3
SLOPE_MAX = 16.0
INTEGRITY_ALPHA = 8.0


def conditions(layers, n_rand=N_RAND, hi=HI, lo=LO):
    """priority order: a cut-off budget must leave the most informative conditions finished.  -> [(name, family, layer, alpha)]"""
    c = [("base", None, None, 0.0)]

    def add(fam, L, doses):
        c.extend((f"{fam}_L{L}_a{s * a:+g}", f"{fam}_L{L}", L, s * a) for a in doses for s in (-1.0, 1.0))
    for L in layers:
        add("probe", L, hi)
        for j in range(n_rand):
            add(f"rand{j}", L, hi)
        add("diff", L, hi)
    for L in layers:
        add("probe", L, lo)
        add("diff", L, lo)
    return c


class SteerLast:
    """forward hook adding `vec` to the output of decoder layer `layer` (= hidden_states[layer]) at the LAST position of every sequence, during prefill only (sequence length > 1).
    The batch is left-padded, so the last position is the last real token of every prompt."""

    def __init__(self, model, layer: int, vec):
        import torch
        p = next(model.parameters())
        self.vec = None if vec is None else torch.as_tensor(vec, dtype=p.dtype, device=p.device)
        self.model, self.layer, self.h = model, layer, None

    def __enter__(self):
        if self.vec is None:
            return self
        blk = getattr(self.model, "model", self.model).layers[self.layer - 1]

        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] > 1:
                hs = hs.clone()
                hs[:, -1, :] += self.vec
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        self.h = blk.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        if self.h is not None:
            self.h.remove()


def select_rows(rows: List[dict], n: int, split: str = "test", splits: Dict[int, str] = None) -> List[dict]:
    """deterministic sample (crc32 of the row key) of at most n injected rows from problems of the given split."""
    cand = [r for r in rows if split is None or (splits or {}).get(r["id"]) == split]
    cand.sort(key=lambda r: zlib.crc32(f"ist{r['id']}_{r['k']}_{r['var']}".encode()))
    return cand[:n] if n else cand


def prep(run_dir: Path, mega_dir: Path, out_dir: Path, n_rows: int = 3000, log=print) -> dict:
    from . import cv_eval, ridge
    from .core import r2
    from .inject import prepare
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    P, why = prepare(run_dir, mega_dir)
    if P is None:
        raise RuntimeError(f"istear prep: {why}")
    df, HA, ly = P["df"], P["HA"], P["ly"]
    cand = [L for L in ly if L < ly[-1]]                                       # the last stored state is post-final-norm: a decoder-layer hook cannot write there
    y = df.flip.values.astype(np.float64)[:, None]
    pool = np.where(np.isin(df.split.values, ["fit", "val"]))[0]
    test = np.where(df.split.values == "test")[0]
    if len(pool) < 200 or len(test) < 50:
        raise RuntimeError(f"istear prep: pool {len(pool)} / test {len(test)} rows are too few")
    ci = [ly.index(L) for L in cand]
    res = cv_eval.cv_probe(lambda li: HA[:, li, :].astype(np.float32), ci, y, pool, test, df.id.values)[0]
    best = ly[res["layer"]]
    mid = min([L for L in cand if L != best], key=lambda L: (abs(L - 12), L))
    layers = sorted({int(best), int(mid)})
    rng = np.random.RandomState(0)
    info = {"layers": layers, "probe_cv_r2_by_layer": {int(ly[i]): v for i, v in res["curve"].items()}, "cv_layer": int(best), "n_pool": int(len(pool))}
    arrs = {"layers": np.array(layers)}
    for L in layers:
        X = HA[:, ly.index(L), :].astype(np.float32)
        fit = ridge.fit_layer(cv_eval._t(X[pool]), y[pool])
        pr = cv_eval.cv_probe(lambda _l, X=X: X, [0], y, pool, test, df.id.values)[0]                     # CV lambda on this layer only
        ri = ridge.RATIOS.index(pr["ratio"])
        w = (fit.W[0][ri] / fit.sd).cpu().numpy().astype(np.float64)
        probe = w / np.linalg.norm(w)
        diff = X[pool][y[pool, 0] >= 0.5].mean(0) - X[pool][y[pool, 0] < 0.5].mean(0)
        diff = (diff / np.linalg.norm(diff)).astype(np.float64)
        rand = rng.randn(N_RAND, X.shape[1])
        rand /= np.linalg.norm(rand, axis=1, keepdims=True)
        arrs.update({f"probe_{L}": probe, f"diff_{L}": diff, f"rand_{L}": rand, f"s_{L}": np.array(float((X[pool] @ probe).std()))})
        info[f"L{L}"] = {"cv_r2": pr["cv_r2"], "test_r2": r2(y[test, 0], pr["pred"]), "s": float(arrs[f"s_{L}"]), "cos_probe_diff": float(probe @ diff), "cos_probe_rand_max": float(np.abs(rand @ probe).max())}
    sel = select_rows(df[["id", "k", "var"]].to_dict("records"), n_rows, "test", dict(zip(df.id, df.split)))
    np.savez(out_dir / "dirs.npz", **arrs)
    (out_dir / "rows.json").write_text(json.dumps([[int(r["id"]), int(r["k"]), str(r["var"])] for r in sel]))
    info["n_rows"] = len(sel)
    (out_dir / "prep.json").write_text(json.dumps(info, indent=1))
    log(f"istear prep: layers {layers}, {len(sel)} test rows; " + "; ".join(f"L{L}: CV R2 {info[f'L{L}']['cv_r2']:.3f}" for L in layers))
    return info


def build_items(problems, G, inj_rows, rows_sel, tok):
    """rows whose edited step can be located again AND whose truncated prompt tokenises as a prefix of the full edited prompt (so the steered position is the last token of the injected number)."""
    probs = {p["id"]: p for p in problems}
    lut = {(r["id"], r["k"], r["var"]): r for r in inj_rows}
    items = []
    for (i, k, v) in rows_sel:
        r, p = lut.get((i, k, v)), probs.get(i)
        if r is None or p is None or i not in G:
            continue
        text = tok.decode(G[i]["plain_ids"], skip_special_tokens=True)
        cm = {(c["start"], c["end"]): c for c in candidate_steps(text, 10 ** 9)}
        s = cm.get((r["start"], r["end"]))
        if s is None:
            continue
        pre, nvs, c0, c1 = edit_geometry(text, s, v)
        head = chat_prompt(tok, plain_text(p))
        if enc(tok, head + pre)[:len(enc(tok, head + pre[:c1]))] != enc(tok, head + pre[:c1]):
            continue
        items.append({"id": i, "k": k, "var": v, "prompt": head + pre[:c1], "cut": pre[:c1], "ans_orig": r["ans_orig"], "gold": p["gold"], "kind": p["kind"]})
    return items


def run(problems, G, inj_rows, model, tok, out_dir: Path, shard=0, nshards=1, max_new=384, max_seqs=96, budget_s=0, log=print) -> int:
    out_dir = Path(out_dir)
    z = np.load(out_dir / "dirs.npz")
    layers = [int(x) for x in z["layers"]]
    rows_sel = json.loads((out_dir / "rows.json").read_text())
    items = [it for it in build_items(problems, G, inj_rows, rows_sel, tok) if it["id"] % nshards == shard]
    dirs = {}
    for L in layers:
        s = float(z[f"s_{L}"])
        dirs[f"probe_L{L}"] = s * z[f"probe_{L}"]
        dirs[f"diff_L{L}"] = s * z[f"diff_{L}"]
        for j in range(len(z[f"rand_{L}"])):
            dirs[f"rand{j}_L{L}"] = s * z[f"rand_{L}"][j]
    n_rand = len(z[f"rand_{layers[0]}"])
    bud = Budget(budget_s)
    done = 0
    prompts = [it["prompt"] for it in items]
    for name, fam, L, a in conditions(layers, n_rand):
        f = out_dir / f"r{shard:02d}_{name}.jsonl"
        if f.exists():
            continue
        if bud.spent():
            log(f"[istear shard {shard}] time budget reached - stopping cleanly (resubmit to continue)")
            break
        vec = None if fam is None else a * dirs[fam]
        with SteerLast(model, L or 1, vec):
            res = generate_ids(model, tok, prompts, max_new, 1, 0.0, 0.95, max_seqs)
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for it, r in zip(items, res):
                ids, fin = r[0]
                ans = parse_answer(it["cut"] + tok.decode(ids, skip_special_tokens=True), it["kind"])
                fh.write(json.dumps({"id": it["id"], "k": it["k"], "var": it["var"], "cond": name, "ntok": len(ids), "fin": bool(fin), "ans": ans, "flip": float(ans != it["ans_orig"]),
                                     "correct": float(ans == it["gold"])}) + "\n")
        tmp.rename(f)
        done += 1
        log(f"[istear shard {shard}] {name}: {len(items)} rows")
    return done


def load(out_dir: Path):
    out: Dict[str, Dict[tuple, dict]] = {}
    shards: Dict[str, set] = {}
    for f in sorted(Path(out_dir).glob("r*_*.jsonl")):
        sh = f.name[1:3]
        cond = None
        for l in open(f):
            r = json.loads(l)
            cond = r["cond"]
            out.setdefault(cond, {})[(r["id"], r["k"], r["var"])] = r
        if cond is None:                                                          # a shard with no rows for this condition still counts as present
            cond = f.name[4:-6]
            out.setdefault(cond, {})
        shards.setdefault(cond, set()).add(sh)
    return out, shards


def analyze(out_dir: Path, B: int = 1000, seed: int = 0) -> dict:
    """-> {'status', 'why', 'table', 'slopes', 'layers': {L: {...status...}}}.  Pre-registered rule (docs/ROBUST.md): per layer, SUPPORTED needs
    (i) probe slope CI lower bound > 0 over |alpha| <= 16, (ii) probe slope > 2x the largest |random-direction slope| of that layer, (iii) the effect at +8 and +16 is positive and at -8 and -16 negative,
    (iv) integrity at |alpha| <= 8: finished rate and parsable-answer rate within 0.05 of the baseline (accuracy is reported but not gated: a steer that makes the model follow the error lowers accuracy by construction).  PARTIAL = (i) and (iii) hold but (ii) or (iv) fails.  NEGATIVE = slope CI inside +-0.003 flip-rate per sd.  The overall status is the best layer."""
    import pandas as pd
    R, shards = load(out_dir)
    if "base" not in R or not R["base"]:
        return {"status": "n/a", "why": "no baseline condition finished", "table": pd.DataFrame(), "slopes": {}, "layers": {}}
    allsh = set().union(*shards.values())
    ok = [c for c in R if shards.get(c, set()) >= allsh and R[c]]
    R = {c: R[c] for c in ok}
    keys = sorted(set.intersection(*[set(v) for v in R.values()]))
    if len(keys) < 200:
        return {"status": "n/a", "why": f"only {len(keys)} rows finished in every condition", "table": pd.DataFrame(), "slopes": {}, "layers": {}}
    flip = {c: np.array([R[c][k]["flip"] for k in keys]) for c in R}
    cor = {c: np.array([R[c][k]["correct"] for k in keys]) for c in R}
    fin = {c: np.array([float(R[c][k]["fin"]) for k in keys]) for c in R}
    par = {c: np.array([float(R[c][k]["ans"] is not None) for k in keys]) for c in R}
    gid = np.array([k[0] for k in keys])
    ug, inv = np.unique(gid, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    rows = []
    for c in sorted(R):
        fam = c.rsplit("_a", 1)[0] if "_a" in c else "base"
        a = float(c.rsplit("_a", 1)[1]) if "_a" in c else 0.0
        rows.append({"cond": c, "family": fam, "alpha": a, "flip_rate": flip[c].mean(), "d_flip_vs_base": (flip[c] - flip["base"]).mean(), "acc": cor[c].mean(), "d_acc_vs_base": (cor[c] - cor["base"]).mean(),
                     "finished": fin[c].mean(), "d_finished_vs_base": (fin[c] - fin["base"]).mean(), "parsed": par[c].mean(), "d_parsed_vs_base": (par[c] - par["base"]).mean()})
    tab = pd.DataFrame(rows)
    rng = np.random.RandomState(seed)
    boots = [rng.randint(0, len(ug), len(ug)) for _ in range(B)]

    def slope(fam):
        cs = [c for c in R if c.startswith(fam + "_a") and abs(float(c.rsplit("_a", 1)[1])) <= SLOPE_MAX] + ["base"]
        a = np.array([float(c.rsplit("_a", 1)[1]) if "_a" in c else 0.0 for c in cs])
        Y = np.stack([flip[c] for c in cs], 1)
        am = a - a.mean()
        per_row = (Y - Y.mean(1, keepdims=True)) @ am / (am @ am)
        sums = np.bincount(inv, weights=per_row)
        pt = per_row.mean()
        bs = np.array([sums[b].sum() / cnt[b].sum() for b in boots])
        return float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    fams = sorted({r["family"] for r in rows if r["family"] != "base"})
    slopes = {f: slope(f) for f in fams}
    layers = {}
    for L in sorted({int(f.rsplit("_L", 1)[1]) for f in fams}):
        pf = f"probe_L{L}"
        if pf not in slopes:
            continue
        pr = slopes[pf]
        rand = [abs(slopes[f][0]) for f in fams if f.startswith("rand") and f.endswith(f"_L{L}")]
        mr = max(rand) if rand else 0.0
        t = tab[tab.family == pf].set_index("alpha")
        eff = lambda x: float(t.loc[x, "d_flip_vs_base"]) if x in t.index else float("nan")       # noqa: E731
        sign_ok = all(eff(x) > 0 for x in (8.0, 16.0)) and all(eff(x) < 0 for x in (-8.0, -16.0))
        sel = t[t.index.map(abs) <= INTEGRITY_ALPHA]
        dfin = float(sel.d_finished_vs_base.abs().max()) if len(sel) else float("nan")
        dacc = float(sel.d_parsed_vs_base.abs().max()) if len(sel) else float("nan")
        integ = (dfin <= 0.05) and (dacc <= 0.05)
        big = pr[0] > 2 * mr
        if pr[1] > 0 and big and sign_ok and integ:
            st, why = "SUPPORTED", f"slope {pr[0]:.4f} flip-rate/sd [{pr[1]:.4f}, {pr[2]:.4f}] > 2x the largest random slope ({mr:.4f}); sign correct at +-8 and +-16; finished/parsable-answer rate within {max(dfin, dacc):.3f} for |alpha|<=8"
        elif pr[1] > 0 and sign_ok:
            st, why = "PARTIAL", f"slope {pr[0]:.4f} [{pr[1]:.4f}, {pr[2]:.4f}], sign correct, but " + ("not above 2x random (%.4f)" % mr if not big else "integrity fails (finished %.3f, parsable %.3f)" % (dfin, dacc))
        elif pr[1] > -0.003 and pr[2] < 0.003:
            st, why = "NEGATIVE", f"slope {pr[0]:.4f} [{pr[1]:.4f}, {pr[2]:.4f}] is inside +-0.003 flip-rate per sd: the direction does not move follow-vs-recover"
        else:
            st, why = "INCONCLUSIVE", f"slope {pr[0]:.4f} [{pr[1]:.4f}, {pr[2]:.4f}], largest random {mr:.4f}, sign_ok={sign_ok}"
        layers[L] = {"status": st, "why": why, "slope": pr, "max_rand": mr, "sign_ok": sign_ok, "dfin": dfin, "dacc": dacc}
    order = {"SUPPORTED": 0, "PARTIAL": 1, "INCONCLUSIVE": 2, "NEGATIVE": 3}
    if not layers:
        return {"status": "n/a", "why": "no probe family finished", "table": tab, "slopes": slopes, "layers": {}}
    bl = min(layers, key=lambda L: order[layers[L]["status"]])
    dose = []
    for L in layers:
        for a in sorted({r["alpha"] for r in rows if r["family"] == f"probe_L{L}"}):
            g = tab[tab.alpha == a]
            rr = g[g.family.str.startswith("rand") & g.family.str.endswith(f"_L{L}")]
            dose.append({"layer": L, "alpha": a, "probe_d_flip": float(g[g.family == f"probe_L{L}"].d_flip_vs_base.mean()), "diff_d_flip": float(g[g.family == f"diff_L{L}"].d_flip_vs_base.mean()),
                         "rand_mean_abs_d_flip": float(rr.d_flip_vs_base.abs().mean()) if len(rr) else float("nan"), "probe_d_acc": float(g[g.family == f"probe_L{L}"].d_acc_vs_base.mean())})
    return {"status": layers[bl]["status"], "why": f"best layer L{bl}: {layers[bl]['why']}", "table": tab, "slopes": slopes, "layers": layers, "n": len(keys), "base_flip": float(flip["base"].mean()),
            "dose": pd.DataFrame(dose)}
