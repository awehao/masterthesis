#!/usr/bin/env bash
# 11/28 00:27 Howard「為什麼不是跑更新版60趟」：擴大開發 gmpcdev4 停在跑完當前一趟，不等 42 條判定，直接以預先指定的改進版 UFE 跑 N1 正式 60 條（n1v2）。
cd "$(dirname "$0")/.."
C=evaluation/results/plan1127/chain.log
while pgrep -f "[r]un_n2_batch.sh gmpcdev4" > /dev/null; do sleep 10; done
echo "$(date +%F_%T) BATCH STOPPED gmpcdev4（Howard 指示改跑 n1v2）" >> evaluation/results/gmpcdev4/gmpcdev4.log
bash mission/tools/wait_clean.sh > /dev/null 2>&1
echo n1v2 > evaluation/results/plan1127/current_batch_dev.txt
echo "$(date +%F_%T) start n1v2 cfg=UFE（Howard 00:27 指示，不等擴大開發判定）" >> "$C"
MODULE=N1V2 bash evaluation/run_n2_batch.sh n1v2 n1_bigarena,n1_office,n1_warehouse 1 20 UFE > evaluation/logs/n1v2_batch.out 2>&1 < /dev/null
echo "$(date +%F_%T) n1v2 done" >> "$C"
