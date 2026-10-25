import argparse
import json
from pathlib import Path

DEFAULT_SIZES = {"gsm8k": 2400, "math": 1800, "arc": 1500, "arith": 1000, "prog": 1000}


def _csv(s):
    return tuple(x for x in s.split(",") if x)


def _write_meta(run: Path, a):
    """provenance for the report: arguments + library versions, one file per (stage, shard)."""
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
    ap.add_argument("cmd", choices=["data", "sample", "states", "trace", "analyze", "report"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n", type=int, default=8, help="sampled solutions per (problem, condition)")
    ap.add_argument("--max-new", type=int, default=0, help="0 = per-dataset default")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-seqs", type=int, default=192)
    ap.add_argument("--fwd-bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N problems of every dataset")
    ap.add_argument("--datasets", default="", help="comma list; default = all admitted")
    ap.add_argument("--conds", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--budget-s", type=float, default=0, help="stop starting new chunks after this many seconds (clean exit; resumable)")
    ap.add_argument("--block", default="A", help="analysis block: A, B1, B2, CDE, T, tables")
    ap.add_argument("--sizes", default="", help='JSON, e.g. {"gsm8k":2400,"math":1800,"arc":1500,"arith":1000,"prog":1000}')
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    dss = _csv(a.datasets) or None
    if a.cmd == "data":
        from . import bench
        sizes = {**DEFAULT_SIZES, **(json.loads(a.sizes) if a.sizes else {})}
        bench.build(run, sizes)
        return
    if a.cmd == "analyze":
        from . import study
        study.compute(run, a.block, dss)
        return
    if a.cmd == "report":
        from . import report
        report.write(run)
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
    from .prompts import CONDS, TRACE_CONDS
    if a.cmd == "sample":
        from . import sample
        sample.run(problems, model, tok, run, a.shard, a.nshards, conds=_csv(a.conds) or CONDS, n=a.n, max_new=a.max_new, temperature=a.temperature,
                   top_p=a.top_p, chunk=a.chunk, max_seqs=a.max_seqs, tag=a.tag, budget_s=a.budget_s)
    elif a.cmd == "states":
        from . import states
        states.run(problems, model, tok, run, a.shard, a.nshards, conds=_csv(a.conds) or CONDS, fwd_bs=a.fwd_bs, tag=a.tag, budget_s=a.budget_s)
    else:
        from . import trace
        trace.run(problems, model, tok, run, a.shard, a.nshards, conds=_csv(a.conds) or TRACE_CONDS, fwd_bs=max(1, a.fwd_bs // 2), chunk=a.chunk, tag=a.tag,
                  budget_s=a.budget_s)


if __name__ == "__main__":
    main()
