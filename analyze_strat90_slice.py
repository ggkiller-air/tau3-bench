"""从全量 telecom run 里切出 telecom_strat90 子集的 pass^1（填 main.tex 表3 Q3 消融用）。

用法：python analyze_strat90_slice.py data/simulations/t1_react_text_full_150
strat90 交集不足 90 时会提示还缺哪些家族（在飞 run 未跑到）。
"""
import json
import re
import sys
from collections import defaultdict

STRAT90 = "data/tau2/domains/telecom/tasks_strat90.json"


def main(run_dir: str) -> None:
    strat_ids = {t["id"] for t in json.load(open(STRAT90))}
    res = json.load(open(f"{run_dir}/results.json"))
    sims = res["simulations"] if isinstance(res, dict) else res
    sub = [s for s in sims if s["task_id"] in strat_ids]

    fam = lambda tid: re.match(r"\[(\w+)\]", tid).group(1)
    stat = defaultdict(lambda: [0, 0])
    for s in sub:
        f = fam(s["task_id"])
        stat[f][1] += 1
        if s["reward_info"]["reward"] == 1.0:
            stat[f][0] += 1

    ok = sum(v[0] for v in stat.values())
    n = sum(v[1] for v in stat.values())
    for f, (o, m) in sorted(stat.items()):
        print(f"{f}: {o}/{m} = {o/m:.3f}")
    print(f"strat90 slice pass^1 = {ok}/{n} = {ok/n:.4f}")

    done_ids = {s["task_id"] for s in sub}
    missing = strat_ids - done_ids
    if missing:
        miss_fam = defaultdict(int)
        for tid in missing:
            miss_fam[fam(tid)] += 1
        print(f"⚠️ 交集不完整，还缺 {len(missing)}/90：{dict(miss_fam)}（run 在飞，跑完再算终值）")


if __name__ == "__main__":
    main(sys.argv[1])
