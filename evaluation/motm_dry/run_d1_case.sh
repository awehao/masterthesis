#!/usr/bin/env bash
# D1 案例跑法（固定設定；scratchpad 會被清，所以放 repo）：bash evaluation/motm_dry/run_d1_case.sh <輸出前綴> [額外環境變數…]
# 設定＝D1 通過的那組：相位 r−1、規劃 0.42／核對 0.40、池 8、串流每批 4、尾段證書、GC 凍結、收斂接近候選終點起步、鎖定模式。
# 注意：呼叫這支腳本的指令字串不要含節點檔名（run_d1.sh 用 pgrep 檢查殘留節點，會自我比對而拒跑）。
cd "$(dirname "$0")/../.."; OUT="$1"; shift
env "$@" D1_PHASE_OFFSET=-1 DOOR_BASE_CLEAR_M=0.10 DOOR_BASE_CLEAR_PLAN_M=0.12 FAKE_START_Q9=$(cat evaluation/results/ahead_open/cand_end_q9.txt) FAKE_LOCK=1 POOL_N=8 CAND_STREAM_CHUNK=4 TAIL_CERT=1 AHEAD_GC_FREEZE=1 ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-73} \
  timeout 600 bash evaluation/motm_dry/run_d1.sh "$OUT" 120 > "$OUT.run.log" 2>&1; echo "rc=$?"
grep "\[d1\]" "$OUT.world.log" | cut -c1-420
