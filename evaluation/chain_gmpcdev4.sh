#!/usr/bin/env bash
# 11/28 擴大開發驗證：gmpcdev4（105–114 × 3 布局 × H、UFE）⇒ 以 101–114 全 42 條判定 ⇒ UFE 勝就跑 n1v2（UFE），否則跑 N2 消融。
cd "$(dirname "$0")/.."
C=evaluation/results/plan1127/chain.log
bash mission/tools/wait_clean.sh > /dev/null 2>&1
echo gmpcdev4 > evaluation/results/plan1127/current_batch_dev.txt
echo "$(date +%F_%T) start gmpcdev4" >> "$C"
MODULE=DEV bash evaluation/run_n2_batch.sh gmpcdev4 n1_bigarena,n1_office,n1_warehouse 105 114 H,UFE > evaluation/logs/gmpcdev4_batch.out 2>&1 < /dev/null
W=$(python3 - <<'PY'
import csv, glob, os, re, collections, json
S = collections.defaultdict(dict)
for R in ('evaluation/results/gmpcdev2', 'evaluation/results/gmpcdev3', 'evaluation/results/gmpcdev4'):
    for f in glob.glob(R + '/n1_*_crossing_k*_*.csv'):
        m = re.match(r'(n1_\w+?)_crossing_k(\d+)_(H|UFE)\.csv$', os.path.basename(f))
        if not m: continue
        r = list(csv.DictReader(open(f)))[-1]
        ok = r['success'] == 'True'; c = int(float(r['collision_count'] or 0))
        S[m.group(3)][(m.group(1), int(m.group(2)))] = (ok, ok and c == 0, float(r['arrival_time_s']) if ok else None)
cases = sorted(set(S['H']) & set(S['UFE']))
st = {}
for k in ('H', 'UFE'):
    x = [S[k][c] for c in cases]; t = [a[2] for a in x if a[0]]
    st[k] = (sum(a[1] for a in x), sum(a[0] for a in x), -(sum(t) / len(t)) if t else -1e9)
win = 'UFE' if (st['UFE'] > st['H'] and st['UFE'][1] >= st['H'][1]) else 'H'
json.dump(dict(n_cases=len(cases), stats={k: dict(clean=v[0], success=v[1], time_mean=-v[2]) for k, v in st.items()}, winner=win), open('evaluation/results/plan1127/dev_selection_42.json', 'w'), ensure_ascii=False, indent=1)
print(win)
PY
)
echo "$(date +%F_%T) dev42 selection -> $W" >> "$C"
echo "- $(date +%F_%T) 擴大開發判定（42 條）：選定 **$W**（evaluation/results/plan1127/dev_selection_42.json）。" >> evaluation/results/plan1127/N1_決策紀錄.md
bash mission/tools/wait_clean.sh > /dev/null 2>&1
if [ "$W" = "UFE" ]; then
  echo n1v2 > evaluation/results/plan1127/current_batch_dev.txt
  echo "$(date +%F_%T) start n1v2 cfg=UFE" >> "$C"
  MODULE=N1V2 bash evaluation/run_n2_batch.sh n1v2 n1_bigarena,n1_office,n1_warehouse 1 20 UFE > evaluation/logs/n1v2_batch.out 2>&1 < /dev/null
  echo "$(date +%F_%T) n1v2 done" >> "$C"
else
  echo n2main > evaluation/results/plan1127/current_batch_dev.txt
  echo "$(date +%F_%T) start n2main（A＝N1 原版）" >> "$C"
  bash evaluation/run_n2_batch.sh n2main n1_bigarena,n1_office 1 10 B,C,D,E > evaluation/logs/n2main_batch.out 2>&1 < /dev/null
  echo "$(date +%F_%T) n2main done" >> "$C"
fi
