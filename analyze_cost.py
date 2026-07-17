#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_cost.py — 统一计算论文 Table-1 的成本/效率列，适用于任意 tau2 run 目录
（schema_user_agent / llm_agent 等各类 agent 一视同仁）。

用法:
  python analyze_cost.py <run_dir> [<run_dir> ...]                 # markdown 表
  python analyze_cost.py <run_dir> --format json
  python analyze_cost.py <run_dir> --cache-rate 0.6 --price-in 2.50 --price-out 10.0

每个 run 计算:
  n / SR      : 已评测 sim 数(reward 非 None) / reward==1.0 比例
  Steps       : canonical tau2 step_count 的 per-sim 均值(= orchestrator step() 次数,
                --max-steps 就 cap 这个数)。口径:每条非 tool 消息计 1 步,一串连续 tool
                消息(一次 ENV 工具执行)合并计 1 步。四臂同款代码,可直接横向比。
  Time(s/ep)  : 每 sim 的 assistant 消息 generation_time_seconds 求和，再对 sim 取均值。
                ⚠ schema-agent run 只覆盖可见 executor 调用(macro/updater 侧调延迟不进
                results.json)——有 tokens.json sidecar 时会打印该 caveat。
  TokenEp     : (prompt+completion 总量)/n。token 来源规则(逐 run 打印):
                有非空 <run>/tokens.json → 只信 sidecar(含 schema macro/updater、hiagent
                摘要等隐藏侧调、且不重复计数)；否则回退累加 results.json 里
                assistant 消息的 usage.prompt_tokens/completion_tokens。
  Cache%      : 复用 scripts/report.py::get_cache(优先 <run>/cache.json 快照,
                否则读 vLLM /metrics, server-cumulative)。
  ETUEp       : (prompt*(1-hit) + prompt*hit*alpha + completion)/n
  $/ep,$/succ : cache 调价后的每 episode / 每成功 sim 成本
仅读取产物文件，不跑任何评测。
"""
import argparse, importlib.util, json, os, sys

# 复用 scripts/report.py 的 load()/reward()/get_cache()，不重实现
_REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "report.py")
_spec = importlib.util.spec_from_file_location("_tau2_report", _REPORT)
_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_report)
load, reward, get_cache = _report.load, _report.reward, _report.get_cache


def _sim_time(sim):
    """sum of generation_time_seconds over this sim's assistant messages (None if absent)."""
    ts = [m.get("generation_time_seconds") for m in sim.get("messages") or []
          if m.get("role") == "assistant" and m.get("generation_time_seconds") is not None]
    return sum(ts) if ts else None


def _sim_steps(sim):
    """canonical tau2 step_count for this sim = orchestrator step() calls.
    Each step appends exactly one message to the trajectory, EXCEPT an ENV
    tool-execution step appends N consecutive tool results but counts as 1 step.
    So: every non-tool message = 1 step; a run of consecutive tool messages = 1 step.
    Matches what --max-steps caps (max observed == max_steps+1 tail)."""
    steps = 0
    prev = None
    for m in sim.get("messages") or []:
        r = m.get("role")
        if r == "tool":
            if prev != "tool":
                steps += 1
        else:
            steps += 1
        prev = r
    return steps


def _traj_tokens(sims):
    """fallback: sum usage.prompt_tokens/completion_tokens over assistant messages."""
    p = c = 0
    for s in sims:
        for m in s.get("messages") or []:
            if m.get("role") != "assistant":
                continue
            u = m.get("usage") or {}
            p += u.get("prompt_tokens", 0) or 0
            c += u.get("completion_tokens", 0) or 0
    return p, c


def analyze(run_dir, args):
    data, src = load(run_dir)
    rd = os.path.dirname(src)
    sims = data.get("simulations") or []
    info = data.get("info") or {}
    ainfo = info.get("agent_info", {}) or {}
    agent, model = ainfo.get("implementation", "?"), ainfo.get("llm", "?")

    evaluated = [s for s in sims if reward(s) is not None]
    n = len(evaluated)
    n_pass = sum(1 for s in evaluated if reward(s) == 1.0)
    sr = n_pass / n if n else None

    times = [t for t in (_sim_time(s) for s in evaluated) if t is not None]
    time_ep = sum(times) / len(times) if times else None

    stepcounts = [_sim_steps(s) for s in evaluated]
    steps_ep = sum(stepcounts) / len(stepcounts) if stepcounts else None

    # --- token source rule: sidecar(含隐藏侧调) > trajectory usage ---
    tok_path = os.path.join(rd, "tokens.json")
    sidecar = {}
    if os.path.exists(tok_path):
        try:
            sidecar = json.load(open(tok_path)) or {}
        except Exception:
            sidecar = {}
    if sidecar:
        prompt_tot = sum(r.get("prompt", 0) for r in sidecar.values() if isinstance(r, dict))
        comp_tot = sum(r.get("completion", 0) for r in sidecar.values() if isinstance(r, dict))
        tok_src = "sidecar tokens.json (includes hidden side calls: schema macro/updater, hiagent summarize)"
    else:
        prompt_tot, comp_tot = _traj_tokens(evaluated)
        tok_src = "results.json trajectory usage (assistant messages only)"

    # --- cache hit rate ---
    if args.cache_rate is not None:
        cache = {"rate": args.cache_rate, "source": "--cache-rate flag"}
    else:
        cache = get_cache(rd, args.metrics_url)
        # API-served frontier run + no per-run cache.json → local /metrics 无意义
        if cache["source"] != "run" and "gpt" in model.lower():
            cache = {"rate": None, "source": "n/a (non-local model, no cache.json)"}
    hit = cache["rate"] if cache["rate"] is not None else 0.0

    tok_ep = (prompt_tot + comp_tot) / n if n else None
    etu_ep = (prompt_tot * (1 - hit) + prompt_tot * hit * args.alpha + comp_tot) / n if n else None
    cost = (prompt_tot * (1 - hit) * args.price_in + prompt_tot * hit * args.price_cached
            + comp_tot * args.price_out) / 1e6
    cost_ep = cost / n if n else None
    cost_succ = cost / n_pass if n_pass else None

    return {
        "run": os.path.basename(rd) or rd,
        "agent": agent, "model": model,
        "n": n, "n_pass": n_pass, "trials": info.get("num_trials"),
        "SR": sr, "time_ep": time_ep, "steps_ep": steps_ep,
        "token_source": tok_src, "sidecar": bool(sidecar),
        "prompt_tot": prompt_tot, "completion_tot": comp_tot,
        "token_ep": tok_ep, "etu_ep": etu_ep,
        "cache_rate": cache["rate"], "cache_source": cache["source"],
        "cost_ep": cost_ep, "cost_succ": cost_succ,
    }


def _fmt(v, spec="{:.3f}", na="n/a"):
    return na if v is None else spec.format(v)


def main():
    ap = argparse.ArgumentParser(description="Table-1 cost/efficiency columns for tau2 run dirs")
    ap.add_argument("run_dirs", nargs="+", help="run directory (or results.json path)")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="cached-prefix compute discount for ETU (default 0.1)")
    ap.add_argument("--price-in", type=float, default=0.27,
                    help="$ per 1M input tokens (illustrative SLM placeholder, same as "
                         "scripts/report.py; frontier-model runs should be invoked "
                         "separately with e.g. --price-in 2.50 --price-out 10.0)")
    ap.add_argument("--price-out", type=float, default=1.10,
                    help="$ per 1M output tokens (illustrative placeholder)")
    ap.add_argument("--price-cached", type=float, default=0.07,
                    help="$ per 1M cached input tokens (illustrative placeholder)")
    ap.add_argument("--cache-rate", type=float, default=None,
                    help="override prefix-cache hit rate (0-1) when neither <run>/cache.json "
                         "nor live vLLM /metrics is trustworthy, e.g. API-served frontier "
                         "runs where local-vLLM metrics are meaningless")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics",
                    help="vLLM /metrics endpoint for prefix-cache hit rate")
    ap.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = ap.parse_args()

    rows = [analyze(d, args) for d in args.run_dirs]

    if args.format == "json":
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    for r in rows:
        print(f"# {r['run']}: agent={r['agent']} model={r['model']} "
              f"n={r['n']} trials={r['trials']} | tokens from {r['token_source']}")
        if r["sidecar"]:
            print("#   ⚠ Time(s/ep) covers only visible executor calls "
                  "(macro/updater side-call latency is not in results.json)")
    print(f"# prices $/M: in {args.price_in} / out {args.price_out} / cached {args.price_cached} "
          f"(illustrative placeholders) · alpha={args.alpha}")
    print("| run | agent | model | n | SR | Steps | Time(s/ep) | TokenEp | ETUEp | Cache% | $/ep | $/succ |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print("| {run} | {agent} | {model} | {n} | {sr} | {st} | {t} | {tok} | {etu} | {ca} | {ce} | {cs} |".format(
            run=r["run"], agent=r["agent"], model=r["model"], n=r["n"],
            sr=_fmt(r["SR"]), st=_fmt(r["steps_ep"], "{:.1f}"), t=_fmt(r["time_ep"], "{:.1f}"),
            tok=_fmt(r["token_ep"], "{:,.0f}"), etu=_fmt(r["etu_ep"], "{:,.0f}"),
            ca=_fmt(r["cache_rate"], "{:.1%}"),
            ce=_fmt(r["cost_ep"], "${:.4f}"), cs=_fmt(r["cost_succ"], "${:.4f}")))


if __name__ == "__main__":
    main()
