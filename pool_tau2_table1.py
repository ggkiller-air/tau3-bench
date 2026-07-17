#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pool_tau2_table1.py — 把同一方法在 tau2 多个 domain（telecom/retail/airline）的 run
按任务数加权池化成论文 Table-1 的单行（07-11 评审会决议：tau-bench 三域合并统计）。

复用 analyze_cost.py 的 analyze()，逐 domain 出指标后做 pooled 聚合：
  SR        = Σ n_pass / Σ n
  Steps     = Σ (steps_ep · n) / Σ n
  TokenEp   = Σ (prompt+completion) / Σ n
  ETUEp     = Σ (etu_ep · n) / Σ n          （各域用各自 cache rate 折算后再合并）
  Cache%    = Σ (prompt·rate) / Σ prompt    （prompt 加权；有域缺 rate 则整体标 n/a*）
  $/ep      = Σ cost / Σ n
  $/succ    = Σ cost / Σ n_pass

用法（每个 spec = <run_dir>[:cache_rate]，cache_rate 缺省走 analyze_cost 的默认解析链）:
  python pool_tau2_table1.py --label "SchemaFlex" \
      data/simulations/full_telecom_hilo_thru_150:0.939 \
      data/simulations/<retail_run> \
      data/simulations/<airline_run>
  # frontier 臂另加 --price-in 2.50 --price-out 10.0（与 analyze_cost.py 相同语义）

仅读取产物文件，不跑任何评测。
"""
import argparse, importlib.util, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("_analyze_cost", os.path.join(_HERE, "analyze_cost.py"))
_ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ac)


def main():
    ap = argparse.ArgumentParser(description="task-weighted pooling of tau2 domain runs for Table-1")
    ap.add_argument("specs", nargs="+", help="<run_dir>[:cache_rate]")
    ap.add_argument("--label", default="pooled")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--price-in", type=float, default=0.27)
    ap.add_argument("--price-out", type=float, default=1.10)
    ap.add_argument("--price-cached", type=float, default=0.07)
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    args = ap.parse_args()

    rows = []
    for spec in args.specs:
        if ":" in spec and not os.path.isdir(spec):
            run_dir, rate = spec.rsplit(":", 1)
            args.cache_rate = float(rate)
        else:
            run_dir, args.cache_rate = spec, None
        rows.append(_ac.analyze(run_dir, args))

    N = sum(r["n"] for r in rows)
    NP = sum(r["n_pass"] for r in rows)
    prompt = sum(r["prompt_tot"] for r in rows)
    comp = sum(r["completion_tot"] for r in rows)
    steps = sum(r["steps_ep"] * r["n"] for r in rows if r["steps_ep"] is not None) / N
    etu = sum(r["etu_ep"] * r["n"] for r in rows) / N
    cost = sum(r["cost_ep"] * r["n"] for r in rows)
    missing_cache = [r["run"] for r in rows if r["cache_rate"] is None]
    cache = (sum(r["prompt_tot"] * r["cache_rate"] for r in rows if r["cache_rate"] is not None)
             / sum(r["prompt_tot"] for r in rows if r["cache_rate"] is not None)) \
        if any(r["cache_rate"] is not None for r in rows) else None

    print(f"# pooled over {len(rows)} domains ({N} evaluated sims); "
          f"cache missing in: {missing_cache or 'none'}")
    for r in rows:
        print(f"#   {r['run']}: n={r['n']} SR={r['SR']:.3f} tokens from {r['token_source']}")
    print("| method | n | SR | Steps | TokenEp | ETUEp | Cache% | $/ep | $/succ |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    print("| {l} | {n} | {sr:.3f} | {st:.1f} | {tok:,.0f} | {etu:,.0f} | {ca} | ${ce:.4f} | ${cs} |".format(
        l=args.label, n=N, sr=NP / N, st=steps, tok=(prompt + comp) / N, etu=etu,
        ca=(f"{cache:.1%}" + ("*" if missing_cache else "")) if cache is not None else "n/a",
        ce=cost / N, cs=f"{cost / NP:.4f}" if NP else "n/a"))


if __name__ == "__main__":
    main()
