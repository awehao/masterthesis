#!/usr/bin/env bash
# K1 被動門短測的運行器（由 run_guarded_v2.sh 包 92 °C 熱中止；只起 Isaac，不起任何 ROS 節點、不放機器人）：
#   RUN_ID=door_k1_1 ROS_DOMAIN_ID=96 RUNNER=evaluation/run_door_k1.sh bash evaluation/run_guarded_v2.sh
# 中止收尾：TERM／INT 或異常退出時，先對本次模擬器送 TERM（模擬端保存目前短測的部分紀錄並標 interrupted），
# 最多等 STOP_GRACE_S 秒，逾時才 KILL；只處理本運行器啟動的那一個模擬器 PID。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
DIR="${K1_RUNS_DIR:-$WS/evaluation/runs}/$RUN_ID"; mkdir -p "$DIR"     # K1_RUNS_DIR 只供離線測試
STOP_GRACE_S="${K1_STOP_GRACE_S:-15}"                              # 只供離線測試縮短
FRZ="${K1_FREEZE:-evaluation/results/door/freeze_door_k1_dev_r2.sha256}"   # 只供離線測試指定
sha256sum -c --quiet "$FRZ" >> "$DIR/run.log" 2>&1 || { echo "K1 開發凍結（r2）不符 ⇒ 拒跑" >> "$DIR/run.log"; exit 95; }
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
SP=''
stop_sim(){
  [ -n "$SP" ] || return 0
  if kill -0 "$SP" 2>/dev/null; then
    kill -TERM "$SP" 2>/dev/null
    for _ in $(seq 1 $((STOP_GRACE_S * 10))); do kill -0 "$SP" 2>/dev/null || break; sleep 0.1; done
    if kill -0 "$SP" 2>/dev/null; then
      kill -KILL "$SP" 2>/dev/null
      echo "[$(date +%H:%M:%S)]   模擬器 ${STOP_GRACE_S} s 內未結束 ⇒ KILL PID=$SP" >> "$DIR/run.log"
    else
      echo "[$(date +%H:%M:%S)]   模擬器已收到 TERM 並結束 PID=$SP" >> "$DIR/run.log"
    fi
  fi
}
on_sig(){ echo "[$(date +%H:%M:%S)]   收到 $1 ⇒ 停止模擬器" >> "$DIR/run.log"; stop_sim; exit 143; }
trap 'on_sig TERM' TERM
trap 'on_sig INT' INT
trap 'stop_sim' EXIT
"$ISAAC_PY" -u evaluation/door_k1_sim.py --out "$DIR" > "$DIR/sim.log" 2>&1 &
SP=$!
echo "[$(date +%H:%M:%S)]   起 sim PID=$SP" >> "$DIR/run.log"
wait "$SP"; rc=$?
echo "=== 結束 rc=$rc ===" >> "$DIR/run.log"
SP=''
exit "$rc"
