#!/usr/bin/env bash
# 11/27 Howard「跑完直接跑 60 趟、不用我同意」：等 gmpcdev3 跑完 ⇒ 依決策紀錄的規則選版本 ⇒ 有贏過 H 就以該配置跑 N1 正式 60 條（n1v2）。
cd "$(dirname "$0")/.."
L=evaluation/results/gmpcdev3/gmpcdev3.log; C=evaluation/results/plan1127/chain.log
until grep -q "BATCH DONE" "$L" 2>/dev/null; do sleep 20; done
W=$(python3 - <<'PY'
import csv, glob, os, re, collections, json
S = collections.defaultdict(dict)
for R in ('evaluation/results/gmpcdev2', 'evaluation/results/gmpcdev3'):
    for f in glob.glob(R + '/n1_*_crossing_k*_*.csv'):
        m = re.match(r'(n1_\w+?)_crossing_k(\d+)_(H|U|UF|UE|UFE)\.csv$', os.path.basename(f))
        if not m: continue
        r = list(csv.DictReader(open(f)))[-1]
        ok = r['success'] == 'True'; c = int(float(r['collision_count'] or 0))
        S[m.group(3)][(m.group(1), int(m.group(2)))] = (ok, ok and c == 0, float(r['arrival_time_s']) if ok else 1e9)
cases = sorted(set.intersection(*[set(S[k]) for k in ('H', 'U', 'UF', 'UE', 'UFE')]))
st = {}
for k in ('H', 'U', 'UF', 'UE', 'UFE'):
    x = [S[k][c] for c in cases]
    t = [a[2] for a in x if a[0]]
    st[k] = (sum(a[1] for a in x), sum(a[0] for a in x), -(sum(t) / len(t)) if t else -1e9)
best = max(('U', 'UF', 'UE', 'UFE'), key=lambda k: st[k])
win = best if (st[best] > st['H'] and st[best][1] >= st['H'][1]) else 'H'
open('evaluation/results/plan1127/dev_selection.json', 'w').write(json.dumps(dict(n_cases=len(cases), stats={k: dict(clean=v[0], success=v[1], time_mean=-v[2]) for k, v in st.items()}, winner=win), ensure_ascii=False, indent=1))
print(win)
PY
)
echo "$(date +%F_%T) dev selection -> $W（evaluation/results/plan1127/dev_selection.json）" >> "$C"
echo "- $(date +%F_%T) 開發比較判定（gmpcdev2＋gmpcdev3，同 12 條開發路線）：選定 **$W**（細節 evaluation/results/plan1127/dev_selection.json）。" >> evaluation/results/plan1127/N1_決策紀錄.md
if [ "$W" = "H" ]; then echo "$(date +%F_%T) H 最好 ⇒ 不跑 n1v2" >> "$C"; exit 0; fi
bash mission/tools/wait_clean.sh > /dev/null 2>&1
echo n1v2 > evaluation/results/plan1127/current_batch_dev.txt
echo "$(date +%F_%T) start n1v2 cfg=$W" >> "$C"
MODULE=N1V2 bash evaluation/run_n2_batch.sh n1v2 n1_bigarena,n1_office,n1_warehouse 1 20 "$W" > evaluation/logs/n1v2_batch.out 2>&1 < /dev/null
echo "$(date +%F_%T) n1v2 done" >> "$C"
