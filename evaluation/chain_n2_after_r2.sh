#!/usr/bin/env bash
# 11/28 Howard：N1 第二版（n1r2）跑完 ⇒ 不跑 GMPC 改版，直接跑 N2 消融（多房間＋辦公室 × 種子 1–10 × B／C／D／E＝80 趟；A＝N1 最終結果同路線）。
cd "$(dirname "$0")/.."
C=evaluation/results/plan1127/chain.log
while pgrep -f "[r]un_n1_r2.sh" > /dev/null; do sleep 20; done
echo "$(date +%F_%T) n1r2 結束 ⇒ start n2main" >> "$C"
bash mission/tools/wait_clean.sh > /dev/null 2>&1
echo n2main > evaluation/results/plan1127/current_batch_dev.txt
bash evaluation/run_n2_batch.sh n2main n1_bigarena,n1_office 1 10 B,C,D,E > evaluation/logs/n2main_batch.out 2>&1 < /dev/null
echo "$(date +%F_%T) n2main done" >> "$C"
