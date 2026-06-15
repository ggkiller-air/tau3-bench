#!/usr/bin/env python3
"""Render a tau2 schema-agent run into a single self-contained HTML report.

Three levels (one file, hash-routed):
  index  : overview + (family × #faults) table; the #faults cell links to →
  group  : card grid of every task in that (family, #faults) bucket; a card →
  task   : the fully-expanded tool-call trace + per-role token panel.

Usage:
    python scripts/report.py data/simulations/<run>            # dir or results.json
    python scripts/report.py data/simulations/<run> -o out.html
Per-role tokens come from a sidecar JSON written by schema_agent when the run set
SCHEMAFLEX_TOKEN_LOG (default looked up at <run>/tokens.json). Missing → shown n/a.
"""
import argparse, json, os, re, collections, html, urllib.request, math


def get_cache(run_dir, url):
    """Prefix-cache hit rate. Prefer a per-run <run>/cache.json {queries,hits}
    (delta snapshot); else read the vLLM /metrics counters (server-cumulative)."""
    p = os.path.join(run_dir, "cache.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            q, h = d.get("queries", 0), d.get("hits", 0)
            return {"rate": h / q if q else None, "queries": q, "hits": h, "source": "run"}
        except Exception:
            pass
    try:
        m = urllib.request.urlopen(url, timeout=2).read().decode()
        q = float(re.search(r"vllm:gpu_prefix_cache_queries_total\S*\s+([0-9.e+]+)", m).group(1))
        h = float(re.search(r"vllm:gpu_prefix_cache_hits_total\S*\s+([0-9.e+]+)", m).group(1))
        return {"rate": h / q if q else None, "queries": q, "hits": h, "source": "server-cumulative"}
    except Exception:
        return {"rate": None, "queries": 0, "hits": 0, "source": "n/a"}


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    with open(path) as f:
        return json.load(f), path


def reward(s):
    ri = s.get("reward_info") or {}
    return ri.get("reward", s.get("reward"))


def scenario(tid):
    b = re.sub(r"\[PERSONA:[^\]]*\]", "", tid)
    return b[b.index("]") + 1:] if "]" in b else b


def family(tid):
    m = re.match(r"\[([^\]]+)\]", tid)
    return m.group(1) if m else "?"


def trace_of(sim):
    """Pair each assistant tool-call with its following tool result."""
    steps, pending = [], None
    for m in sim.get("messages", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]
            pending = {"tool": tc.get("name"), "args": tc.get("arguments"), "result": None}
            steps.append(pending)
        elif m.get("role") == "tool" and pending is not None:
            pending["result"] = m.get("content")
            pending = None
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-o", "--out")
    ap.add_argument("--tokens", help="per-role token sidecar JSON (default <run>/tokens.json)")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics",
                    help="vLLM /metrics endpoint for prefix-cache hit rate")
    ap.add_argument("--price-in", type=float, default=0.27, help="$ per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=1.10, help="$ per 1M output tokens")
    ap.add_argument("--price-cached", type=float, default=0.07, help="$ per 1M cached input tokens")
    args = ap.parse_args()
    data, src = load(args.path)
    run_dir = os.path.dirname(src)
    sims = data.get("simulations") or []
    info = data.get("info") or {}
    ainfo = info.get("agent_info", {}) or {}

    tok_path = args.tokens or os.path.join(run_dir, "tokens.json")
    tokens = {}
    if os.path.exists(tok_path):
        try:
            tokens = json.load(open(tok_path))
        except Exception:
            tokens = {}

    # task descriptions/reasons
    reason = {}
    for t in data.get("tasks") or []:
        us = (t.get("user_scenario") or {}).get("instructions") or {}
        reason[t.get("id")] = us.get("reason_for_call") or (t.get("description") or {}).get("purpose") or ""

    def tok(tid):
        r = tokens.get(tid)
        if not r:
            return {"macro": None, "executor": None, "updater": None, "total": None,
                    "prompt": 0, "completion": 0}
        tot = (r.get("macro", 0) + r.get("executor", 0) + r.get("updater", 0))
        return {"macro": r.get("macro", 0), "executor": r.get("executor", 0),
                "updater": r.get("updater", 0), "total": tot,
                "prompt": r.get("prompt", 0), "completion": r.get("completion", 0)}

    evaluated = [s for s in sims if reward(s) is not None]
    passed = [s for s in evaluated if reward(s) == 1.0]
    infra = [s for s in sims if reward(s) is None]
    avg = sum(reward(s) for s in evaluated) / len(evaluated) if evaluated else 0.0
    # pass^k: group trials by task; pass^k = mean over tasks of C(passes,k)/C(trials,k)
    by_task = collections.defaultdict(list)
    for s in evaluated:
        by_task[s.get("task_id")].append(1 if reward(s) == 1.0 else 0)
    ntrials = max((len(v) for v in by_task.values()), default=1)

    def pass_k(k):
        vals = [math.comb(sum(v), k) / math.comb(len(v), k) for v in by_task.values() if len(v) >= k]
        return (sum(vals) / len(vals)) if vals else None
    passk = {k: pass_k(k) for k in (1, 2, 3, 4)}

    role_tot = {"macro": 0, "executor": 0, "updater": 0, "total": 0}
    for tid in (s.get("task_id") for s in sims):
        tt = tok(tid)
        if tt["total"] is not None:
            for k in role_tot:
                role_tot[k] += tt[k] if k != "total" else tt["total"]
    # run-level prompt/completion split (new sidecar format) + prefix-cache
    prompt_tot = sum(r.get("prompt", 0) for r in tokens.values())
    comp_tot = sum(r.get("completion", 0) for r in tokens.values())
    cache = get_cache(run_dir, args.metrics_url)
    hit = cache["rate"] or 0.0
    # effective (non-cached) input tokens = prompt × (1 − hit_rate)
    eff_input = round(prompt_tot * (1 - hit)) if (prompt_tot and cache["rate"] is not None) else None

    def dollars(p, c):  # cache-adjusted $: cached input is cheaper
        return (p * (1 - hit) * args.price_in + p * hit * args.price_cached + c * args.price_out) / 1e6
    def dollars_nocache(p, c):
        return (p * args.price_in + c * args.price_out) / 1e6

    # group by (family, #faults)
    buckets = collections.defaultdict(list)
    for s in sims:
        tid = s.get("task_id", "")
        buckets[(family(tid), len(scenario(tid).split("|")))].append(s)

    groups = []
    for (fam, nf), ss in sorted(buckets.items()):
        ev = [s for s in ss if reward(s) is not None]
        ps = [s for s in ev if reward(s) == 1.0]
        toks = [tok(s.get("task_id"))["total"] for s in ss if tok(s.get("task_id"))["total"] is not None]
        steps = [len(trace_of(s)) for s in ss]
        tasks = []
        for s in sorted(ss, key=lambda x: x.get("task_id", "")):
            tid = s.get("task_id", "")
            tr = trace_of(s)
            tk = tok(tid)
            tasks.append({
                "id": tid, "reward": reward(s), "term": s.get("termination_reason"),
                "reason": reason.get(tid, ""), "steps": len(tr),
                "tokens": tk, "dollar": dollars(tk["prompt"], tk["completion"]), "trace": tr,
            })
        dols = [t["dollar"] for t in tasks if t["tokens"]["total"] is not None]
        groups.append({
            "family": fam, "faults": nf,
            "n_pass": len(ps), "n_eval": len(ev),
            "rate": (len(ps) / len(ev)) if ev else None,
            "avg_tok": round(sum(toks) / len(toks)) if toks else None,
            "avg_steps": round(sum(steps) / len(steps), 1) if steps else None,
            "avg_dollar": (sum(dols) / len(dols)) if dols else None,
            "tasks": tasks,
        })

    payload = {
        "meta": {
            "run": os.path.basename(run_dir) or src,
            "agent": ainfo.get("implementation", "?"), "model": ainfo.get("llm", "?"),
            "date": data.get("timestamp", ""), "git": info.get("git_commit", ""),
            "max_steps": info.get("max_steps"),
            "total": len(sims), "evaluated": len(evaluated), "infra": len(infra),
            "pass1": (len(passed) / len(evaluated)) if evaluated else 0.0,
            "n_pass": len(passed), "avg_reward": avg, "tokens": role_tot,
            "num_trials": ntrials, "passk": passk,
            "prompt": prompt_tot or None, "completion": comp_tot or None,
            "cache": cache, "eff_input": eff_input,
            "dollar": dollars(prompt_tot, comp_tot) if prompt_tot else None,
            "dollar_nocache": dollars_nocache(prompt_tot, comp_tot) if prompt_tot else None,
            "dollar_per_task": (dollars(prompt_tot, comp_tot) / len(sims)) if (prompt_tot and sims) else None,
            "price": {"in": args.price_in, "out": args.price_out, "cached": args.price_cached},
        },
        "groups": groups,
    }

    out = args.out or os.path.join(run_dir, "report.html")
    with open(out, "w") as f:
        f.write(HTML.replace("/*DATA*/", json.dumps(payload, ensure_ascii=False)))
    print(f"wrote {out}  (Pass^1 {len(passed)}/{len(evaluated)}, tokens {'yes' if tokens else 'MISSING'})")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>tau2 report</title>
<style>
*{box-sizing:border-box}
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f6f8fa;color:#1f2328;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:34px 24px 64px}
a{color:#0969da;text-decoration:none;cursor:pointer} a:hover{text-decoration:underline}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.9em;background:#eff1f3;padding:1px 6px;border-radius:5px}
h1{font-size:22px;font-weight:650;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#818b98;margin:30px 0 12px}
.crumb{margin-bottom:18px;color:#656d76;font-size:13px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:6px 0}
.stat{background:#fff;border:1px solid #d8dee4;border-radius:12px;padding:15px 17px;box-shadow:0 1px 2px rgba(31,35,40,.04)}
.stat .lab{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#818b98;margin-bottom:7px}
.stat .val{font-size:25px;font-weight:680;letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1.1}
.stat .sub{font-size:12px;color:#818b98;font-weight:500;margin-top:5px}
.stat.hi .val{color:#0969da}
.tbl{background:#fff;border:1px solid #d8dee4;border-radius:12px;overflow:hidden;box-shadow:0 1px 2px rgba(31,35,40,.04)}
table{border-collapse:collapse;width:100%}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#818b98;font-weight:600;text-align:left;padding:11px 16px;background:#f6f8fa;border-bottom:1px solid #d8dee4}
td{padding:11px 16px;border-bottom:1px solid #eef1f4;font-size:13.5px;font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:none} tbody tr:hover{background:#f9fafb}
.num{text-align:right} .rate{font-weight:650}
.pill{display:inline-block;background:#eef4fd;color:#0969da;font-weight:600;padding:2px 11px;border-radius:999px;font-size:12.5px}
.pill:hover{background:#ddeafc;text-decoration:none}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px}
.card{background:#fff;border:1px solid #d8dee4;border-radius:12px;padding:16px 18px;cursor:pointer;transition:.12s;box-shadow:0 1px 2px rgba(31,35,40,.04)}
.card:hover{border-color:#0969da;box-shadow:0 5px 16px rgba(9,105,218,.10);transform:translateY(-1px)}
.card h3{font-size:14px;font-weight:600;margin:0 0 4px;display:flex;justify-content:space-between;align-items:center;gap:8px;text-transform:capitalize}
.reason{color:#57606a;font-size:13px;line-height:1.5;margin:8px 0 12px;max-height:60px;overflow:hidden}
.muted{color:#818b98;font-size:12px}
.badge{font-weight:700;font-size:12px;padding:3px 9px;border-radius:7px;font-variant-numeric:tabular-nums}
.ok{background:#dafbe1;color:#116329} .bad{background:#ffebe9;color:#a40e26} .na{background:#eef1f4;color:#818b98}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 6px}
.chip{background:#fff;border:1px solid #d8dee4;border-radius:10px;padding:11px 16px;min-width:92px;box-shadow:0 1px 2px rgba(31,35,40,.04)}
.chip .lab{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#818b98;font-weight:600}
.chip .val{font-size:18px;font-weight:680;font-variant-numeric:tabular-nums;margin-top:4px}
.step{display:flex;gap:12px;margin:0 0 9px}
.step .n{flex:0 0 26px;height:26px;border-radius:50%;background:#eef1f4;color:#656d76;font-size:12px;font-weight:600;display:flex;align-items:center;justify-content:center;margin-top:3px}
.step .body{flex:1;min-width:0;background:#fff;border:1px solid #d8dee4;border-radius:10px;overflow:hidden;box-shadow:0 1px 2px rgba(31,35,40,.03)}
.call{padding:9px 14px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;border-bottom:1px solid #eef1f4;background:#fbfcfd;word-break:break-word}
.call .t{color:#0550ae;font-weight:600} .call .a{color:#818b98}
.res{padding:10px 14px;white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#24292f;max-height:300px;overflow:auto}
</style></head><body><div class="wrap" id="app"></div>
<script>
const D = /*DATA*/;
const esc = s => (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const fam = f => f.replace('_issue','').replace('_',' ');
const badge = r => r==null?'<span class="badge na">infra</span>':`<span class="badge ${r==1?'ok':'bad'}">${r.toFixed(2)}</span>`;
const num = n => n==null?'n/a':n.toLocaleString();
const rateColor = r => r==null?'#818b98':(r>=.9?'#116329':r>=.6?'#9a6700':'#a40e26');
const dol = n => n==null?'n/a':'$'+(n<1?n.toFixed(4):n.toFixed(2));
const pk = v => v==null?'N/A':(v*100).toFixed(0)+'%';

function index(){
  const m=D.meta, t=m.tokens;
  let h=`<h1>tau2 report — ${esc(m.run)}</h1>
  <div class="crumb">agent <code>${esc(m.agent)}</code> · model <code>${esc(m.model)}</code> · ${esc(m.date).slice(0,19)} · max_steps ${m.max_steps}</div>
  <div class="stats">
   <div class="stat hi"><div class="lab">Pass^1</div><div class="val">${(m.pass1*100).toFixed(1)}%</div><div class="sub">${m.n_pass}/${m.evaluated} · ${m.num_trials} trial(s)</div></div>
   <div class="stat"><div class="lab">pass^k</div><div class="val" style="font-size:15px">${[2,3,4].map(k=>'^'+k+': '+pk(m.passk[k])).join(' · ')}</div><div class="sub">N/A if trials &lt; k</div></div>
   <div class="stat"><div class="lab">avg reward</div><div class="val">${m.avg_reward.toFixed(3)}</div></div>
   <div class="stat"><div class="lab">simulations</div><div class="val">${m.total}</div><div class="sub">eval ${m.evaluated} · infra ${m.infra}</div></div>
   <div class="stat"><div class="lab">total tokens</div><div class="val">${num(t.total)}</div><div class="sub">macro ${num(t.macro)} · exec ${num(t.executor)} · upd ${num(t.updater)}</div></div>
   <div class="stat hi"><div class="lab">prefix cache hit</div><div class="val">${m.cache.rate==null?'n/a':(m.cache.rate*100).toFixed(1)+'%'}</div><div class="sub">${esc(m.cache.source)}</div></div>
   <div class="stat"><div class="lab">effective input tok</div><div class="val">${num(m.eff_input)}</div><div class="sub">${m.prompt?('of '+num(m.prompt)+' prompt'):'set SCHEMAFLEX_TOKEN_LOG'} · non-cached</div></div>
   <div class="stat hi"><div class="lab">$ / run (cache-adj)</div><div class="val">${dol(m.dollar)}</div><div class="sub">${dol(m.dollar_per_task)}/task · no-cache ${dol(m.dollar_nocache)}</div></div>
  </div>
  <div class="crumb">price (placeholder, DeepSeek-style): in $${m.price.in}/M · out $${m.price.out}/M · cached $${m.price.cached}/M per 1M tokens</div>
  <h2>By family × #faults</h2>
  <div class="tbl"><table><thead><tr><th>family</th><th>#faults</th><th class="num">pass / eval</th><th class="num">rate</th><th class="num">avg tok/task</th><th class="num">avg steps/task</th><th class="num">avg $/task</th></tr></thead><tbody>`;
  D.groups.forEach((g,i)=>{
    h+=`<tr><td style="text-transform:capitalize">${esc(fam(g.family))}</td>
      <td><a class="pill" href="#g=${i}">${g.faults} faults ›</a></td>
      <td class="num">${g.n_pass}/${g.n_eval}</td>
      <td class="num rate" style="color:${rateColor(g.rate)}">${g.rate==null?'-':(g.rate*100).toFixed(0)+'%'}</td>
      <td class="num">${num(g.avg_tok)}</td><td class="num">${g.avg_steps==null?'-':g.avg_steps}</td><td class="num">${dol(g.avg_dollar)}</td></tr>`;
  });
  return h+`</tbody></table></div>`;
}
function group(i){
  const g=D.groups[i];
  let h=`<div class="crumb"><a href="#">← overview</a></div>
   <h1>${esc(fam(g.family))} · ${g.faults} faults</h1>
   <div class="crumb">${g.n_pass}/${g.n_eval} passed · avg ${num(g.avg_tok)} tok · ${g.avg_steps} steps/task</div>
   <div class="grid">`;
  g.tasks.forEach((t,j)=>{
    h+=`<div class="card" onclick="location.hash='t=${i}-${j}'">
      <h3>${esc(fam(g.family))} ${badge(t.reward)}</h3>
      <div class="reason">${esc(t.reason)}</div>
      <div class="muted">${t.steps} steps · ${num(t.tokens.total)} tok · ${esc(t.term||'')}</div></div>`;
  });
  return h+`</div>`;
}
function task(i,j){
  const g=D.groups[i], t=g.tasks[j], tk=t.tokens;
  let h=`<div class="crumb"><a href="#">overview</a> / <a href="#g=${i}">${esc(fam(g.family))} · ${g.faults}f</a></div>
   <h1>${badge(t.reward)} <code style="font-size:13px">${esc(t.id)}</code></h1>
   <div class="crumb">${esc(t.reason)}</div>
   <div class="chips">
     <div class="chip"><div class="lab">total tok</div><div class="val">${num(tk.total)}</div></div>
     <div class="chip"><div class="lab">macro</div><div class="val">${num(tk.macro)}</div></div>
     <div class="chip"><div class="lab">executor</div><div class="val">${num(tk.executor)}</div></div>
     <div class="chip"><div class="lab">updater</div><div class="val">${num(tk.updater)}</div></div>
     <div class="chip"><div class="lab">$</div><div class="val">${dol(t.dollar)}</div></div>
     <div class="chip"><div class="lab">steps</div><div class="val">${t.steps}</div></div>
     <div class="chip"><div class="lab">termination</div><div class="val" style="font-size:13px">${esc(t.term||'')}</div></div>
   </div><h2>Trace</h2>`;
  if(!t.trace.length) h+=`<div class="muted">no tool calls</div>`;
  t.trace.forEach((s,k)=>{
    const a=typeof s.args==='object'?JSON.stringify(s.args):s.args;
    h+=`<div class="step"><div class="n">${k+1}</div><div class="body">
         <div class="call"><span class="t">${esc(s.tool)}</span><span class="a">(${esc(a||'')})</span></div>
         <div class="res">${esc(s.result)}</div></div></div>`;
  });
  return h;
}
function render(){
  const hsh=location.hash.slice(1);
  let m;
  if((m=hsh.match(/^t=(\d+)-(\d+)/))) document.getElementById('app').innerHTML=task(+m[1],+m[2]);
  else if((m=hsh.match(/^g=(\d+)/))) document.getElementById('app').innerHTML=group(+m[1]);
  else document.getElementById('app').innerHTML=index();
  window.scrollTo(0,0);
}
addEventListener('hashchange',render); render();
</script></body></html>"""


if __name__ == "__main__":
    main()
