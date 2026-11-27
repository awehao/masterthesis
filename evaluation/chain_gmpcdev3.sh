#!/usr/bin/env bash
# 開發比較自動接續（11/27）：等 gmpcdev2 跑完 ⇒ 清理 ⇒ 編譯 ammr_wholebody_mpc（任務模式保護）⇒ 開跑 gmpcdev3（UE、UFE，同開發種子 101–104 × 3 布局）。
cd "$(dirname "$0")/.."
L=evaluation/results/gmpcdev2/gmpcdev2.log
until grep -q "BATCH DONE" "$L" 2>/dev/null; do sleep 20; done
echo "$(date +%F_%T) gmpcdev2 done -> rebuild + gmpcdev3" >> evaluation/results/plan1127/chain.log
bash mission/tools/wait_clean.sh > /dev/null 2>&1
env -u VIRTUAL_ENV PATH=/usr/local/bin:/usr/bin:/bin:/opt/ros/jazzy/bin bash -c "source /opt/ros/jazzy/setup.bash && colcon build --packages-select ammr_wholebody_mpc" >> evaluation/results/plan1127/chain.log 2>&1
echo gmpcdev3 > evaluation/results/plan1127/current_batch_dev.txt
MODULE=DEV bash evaluation/run_n2_batch.sh gmpcdev3 n1_bigarena,n1_office,n1_warehouse 101 104 UE,UFE > evaluation/logs/gmpcdev3_batch.out 2>&1 < /dev/null
echo "$(date +%F_%T) gmpcdev3 done" >> evaluation/results/plan1127/chain.log
