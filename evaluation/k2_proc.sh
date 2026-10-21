#!/usr/bin/env bash
# K2 運行器用的有界停止：stop_bounded <pid> <grace_s> ⇒ 若仍在跑：TERM → 至多 grace_s 秒 → KILL → wait；
# 結果寫入全域 STOP_RC（程序 rc）與 STOP_HOW（exited／terminated／killed）。只處理呼叫端給的單一 PID（須為本 shell 的子程序）。
stop_bounded(){
  local pid="$1" grace="${2:-10}" i
  STOP_HOW=exited
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null
    STOP_HOW=terminated
    for i in $(seq $((grace * 10))); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null; STOP_HOW=killed; fi
  fi
  wait "$pid" 2>/dev/null; STOP_RC=$?
}
# wait_or_stop <pid> <wait_s> <grace_s>：先等至多 wait_s 秒自然結束，逾時再 stop_bounded；STOP_HOW 會標 terminated／killed
wait_or_stop(){
  local pid="$1" w="$2" grace="$3" i
  for i in $(seq "$w"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  stop_bounded "$pid" "$grace"
}
