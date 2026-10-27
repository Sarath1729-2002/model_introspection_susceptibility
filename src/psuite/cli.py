import argparse
import json
from pathlib import Path

DEFAULT_SIZES = {"gsm8k": 2400, "math": 1800, "arc": 1200, "arith": 1000}


def _write_meta(run: Path, a):
    import time
    try:
        import torch
        import transformers
        ver = {"torch": torch.__version__, "transformers": transformers.__version__}
    except Exception:                                   # noqa: BLE001
        ver = {}
    d = run / "meta"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.cmd}_{a.shard:02d}.json").write_text(json.dumps({"args": vars(a), "versions": ver, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "gen", "states", "trace", "analyze", "report", "unc-roll", "unc-analyze", "steps", "steer-prep", "steer-run", "mega-analyze", "inject", "inject-analyze", "robust-prep", "robust-steer", "robust-offline", "robust-analyze", "stoch-run", "natural-run", "cell-analyze", "paper"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n-spread", type=int, default=4, help="sampled plain solutions per problem (length spread, soft runaway rate)")
    ap.add_argument("--max-new", type=int, default=0, help="0 = per-dataset default")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-seqs", type=int, default=96)
    ap.add_argument("--fwd-bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N problems of every dataset")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--budget-s", type=float, default=0)
    ap.add_argument("--sub", default="", help="w/W: take every W-th chunk of this shard (rescue mode)")
    ap.add_argument("--block", default="P", help="analysis block: P, C, K, tables")
    ap.add_argument("--sizes", default="")
    ap.add_argument("--unc-n", type=int, default=400, help="uncertainty POC: problems per dataset (gsm8k, math)")
    ap.add_argument("--n-roll", type=int, default=16, help="uncertainty POC: prompt-end rollouts per problem")
    ap.add_argument("--n-cont", type=int, default=8, help="uncertainty POC: continuations per checkpoint (o48, o96); 0 = prompt-end rollouts only")
    ap.add_argument("--unc-datasets", default="gsm8k,math", help="uncertainty rollouts: datasets")
    ap.add_argument("--unc-skip", type=int, default=0, help="uncertainty rollouts: skip the first N (hash order) problems per dataset - the mega run excludes the 400 problems already analysed")
    ap.add_argument("--mega-dir", default="", help="mega run output dir (default <run-dir>/mega)")
    ap.add_argument("--max-steps", type=int, default=8, help="mega steps thread: numeric steps edited per solution")
    ap.add_argument("--steer-n", type=int, default=0, help="mega steering: only the first N test problems (smoke runs); 0 = all")
    ap.add_argument("--steer-alphas", default="", help="steer-run: comma list of |alpha| doses (applied as +/-); default = the mega run's 1,2,4")
    ap.add_argument("--steer-out", default="steer", help="steer-run: sub-directory of the mega dir for the results (directions are always read from <mega>/steer)")
    ap.add_argument("--steer-nrand", type=int, default=8, help="steer-run: number of random control directions")
    ap.add_argument("--ds", default="gsm8k", help="dataset of a matrix cell (steps / inject / robust-* / stoch / natural / lenref); gsm8k = the original cell")
    ap.add_argument("--no-con", action="store_true", help="gen/states: skip the constrained prompt (matrix cells that only need plain solutions)")
    ap.add_argument("--no-del", action="store_true", help="steps: do not run the line-deletion edit (saves a third of the regenerations)")
    ap.add_argument("--robust-dir", default="robust", help="robust-*: sub-directory of the mega dir (the smoke run uses robust_smoke); the steering results are in <robust-dir>/steer")
    ap.add_argument("--robust-n", type=int, default=3000, help="robust-prep: injected test rows to steer")
    ap.add_argument("--stoch-n", type=int, default=1000, help="stoch-run: injected rows resampled")
    ap.add_argument("--n-samp", type=int, default=8, help="stoch-run: sampled continuations per row")
    ap.add_argument("--cells", default="", help="paper: JSON file with the list of cells [{name,model,ds,run_dir,mega_dir}]")
    ap.add_argument("--paper-out", default="", help="paper: output directory")
    ap.add_argument("--parts", default="steps,unc,steer,suite", help="mega-analyze: threads to analyse")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    dss = tuple(x for x in a.datasets.split(",") if x) or None
    if a.cmd == "data":
        from . import bench
        bench.build(run, {**DEFAULT_SIZES, **(json.loads(a.sizes) if a.sizes else {})})
        return
    if a.cmd == "analyze":
        from . import study
        study.compute(run, a.block, dss)
        return
    if a.cmd == "report":
        from . import report
        report.write(run)
        return
    unc_dir = run / f"unc_n{a.unc_n}"
    mega = Path(a.mega_dir) if a.mega_dir else run / "mega"
    if a.cmd == "mega-analyze":
        from . import mega as mg
        mg.run_all(run, mega, tuple(x for x in a.parts.split(",") if x))
        return
    if a.cmd == "inject-analyze":
        from . import inject
        inject.write_report(run, mega, big_name=a.steer_out if a.steer_out != "steer" else "steer_big")
        return
    if a.cmd == "robust-prep":
        from . import istear
        istear.prep(run, mega, mega / a.robust_dir / "steer", a.robust_n, ds=a.ds)
        return
    if a.cmd in ("robust-offline", "robust-analyze"):
        from . import robust
        if a.cmd == "robust-offline":
            robust.offline(run, mega, name=a.robust_dir, ds=a.ds)
        else:
            robust.write_report(run, mega, name=a.robust_dir, ds=a.ds)
        return
    if a.cmd == "cell-analyze":
        from . import paper
        paper.analyze_cell(run, mega, a.ds, name=a.robust_dir)
        return
    if a.cmd == "paper":
        from . import paper
        specs = json.loads(Path(a.cells).read_text())
        paper.write(specs, Path(a.paper_out) if a.paper_out else Path(a.cells).parent / "paper")
        return
    if a.cmd == "steer-prep":
        from . import steer
        steer.prep(run, mega / "steer")
        return
    if a.cmd == "unc-analyze":
        from . import unc_study
        unc_study.compute(run, unc_dir)
        unc_study.write(unc_dir)
        return
    from .bench import read_problems
    from .hf_util import load_model
    _write_meta(run, a)
    model, tok = load_model(a.model_id)
    problems = read_problems(run, dss)
    if a.limit:
        keep, cnt = [], {}
        for p in problems:
            cnt[p["ds"]] = cnt.get(p["ds"], 0) + 1
            if cnt[p["ds"]] <= a.limit:
                keep.append(p)
        problems = keep
    if a.cmd == "unc-roll":
        from . import table, unc
        G = table.load_gen(run)
        dsn = tuple(x for x in a.unc_datasets.split(",") if x)
        sub = unc.pick_subset(problems, G, a.unc_n, dsn, a.unc_skip)
        out_u = (mega / "unc") if a.mega_dir or a.unc_skip else unc_dir
        unc.run(sub, G, model, tok, out_u, a.shard, a.nshards, n_roll=a.n_roll, n_cont=a.n_cont, cps=unc.CPS if a.n_cont else (), max_seqs=a.max_seqs, max_new=a.max_new, budget_s=a.budget_s)
    elif a.cmd == "steps":
        from . import steps, table
        G = table.load_gen(run, [a.ds])
        steps.run([p for p in problems if p["ds"] == a.ds], G, model, tok, mega / "steps", a.shard, a.nshards, chunk=a.chunk, max_steps=a.max_steps, max_new=a.max_new or 512, max_seqs=a.max_seqs, budget_s=a.budget_s,
                  variants=("n1", "n2") if a.no_del else ("n1", "n2", "del"))
    elif a.cmd == "steer-run":
        from . import steer
        test = [p for p in problems if p["ds"] == a.ds and p["split"] == "test"]
        test = test[:a.steer_n] if a.steer_n else test
        kw = {}
        if a.steer_alphas:
            doses = sorted({abs(float(x)) for x in a.steer_alphas.split(",") if x})
            al = tuple([-d for d in reversed(doses)] + doses)
            kw = {"alphas": al, "rand_alphas": al}
        steer.run(test, model, tok, mega / a.steer_out, a.shard, a.nshards, max_new=a.max_new or 512, max_seqs=a.max_seqs, budget_s=a.budget_s, dirs_dir=mega / "steer", n_rand=a.steer_nrand, **kw)
    elif a.cmd == "inject":
        from . import inject, table
        G = table.load_gen(run, [a.ds])
        sr = inject.step_rows(mega / "steps")
        if not sr:
            raise SystemExit(f"no step-thread outcomes in {mega / 'steps'}: run MODE=mega (steps) first")
        inject.run([p for p in problems if p["ds"] == a.ds], G, sr, model, tok, mega / "inject", a.shard, a.nshards, chunk=a.chunk, budget_s=a.budget_s)
    elif a.cmd == "robust-steer":
        from . import inject, istear, table
        G = table.load_gen(run, [a.ds])
        irows = inject.load(mega / "inject")[0]
        istear.run([p for p in problems if p["ds"] == a.ds], G, irows, model, tok, mega / a.robust_dir / "steer", a.shard, a.nshards, max_new=a.max_new or 384, max_seqs=a.max_seqs, budget_s=a.budget_s)
    elif a.cmd == "stoch-run":
        from . import inject, stoch
        irows = inject.load(mega / "inject")[0]
        stoch.run([p for p in problems if p["ds"] == a.ds], irows, model, tok, mega / a.robust_dir / "stoch", a.shard, a.nshards, n_rows=a.stoch_n, n_samp=a.n_samp,
                  max_new=a.max_new or 512, max_seqs=a.max_seqs, budget_s=a.budget_s)
    elif a.cmd == "natural-run":
        from . import natural, table
        G = table.load_gen(run, [a.ds])
        natural.run([p for p in problems if p["ds"] == a.ds], G, model, tok, mega / a.robust_dir / "natural", a.shard, a.nshards, max_new=a.max_new or 512, max_seqs=a.max_seqs, budget_s=a.budget_s)
    elif a.cmd == "gen":
        from . import gen
        gen.run(problems, model, tok, run, a.shard, a.nshards, chunk=a.chunk, max_new=a.max_new, n_spread=a.n_spread, max_seqs=a.max_seqs, tag=a.tag, budget_s=a.budget_s, sub=a.sub, con=not a.no_con)
    elif a.cmd == "states":
        from . import states
        states.run(problems, model, tok, run, a.shard, a.nshards, fwd_bs=a.fwd_bs * 2, tag=a.tag, budget_s=a.budget_s, sub=a.sub, con=not a.no_con)
    else:
        from . import trace
        trace.run(problems, model, tok, run, a.shard, a.nshards, fwd_bs=a.fwd_bs, chunk=a.chunk, tag=a.tag, budget_s=a.budget_s, sub=a.sub)


if __name__ == "__main__":
    main()
