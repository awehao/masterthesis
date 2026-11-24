#!/usr/bin/env bash
# 替身完整任務（不開 Isaac；感測為合成觀測）。約 20 分鐘。
# 用法：mission/launch/run_stub.sh <TAG> [KEY=VAL ...]
#   例：mission/launch/run_stub.sh s300d CYL_SEED=2 YS_PRED=0
# 結果：evaluation/results/motm_direct/d2/<TAG>*（_stub.json 成敗、_d2w.json 控制事件、.sup.log 監督）；終端輸出：mission/logs/<TAG>.out
source "$(dirname "$0")/_env.sh"
TAG=${1:?要給 TAG，如 s300d}; shift
bash "$MISSION/tools/wait_clean.sh" || { echo "還有導航／替身程序沒收乾淨：先跑 mission/launch/stop_all.sh"; exit 1; }
load_env task.env stub.env
apply_overrides "$@"
[ -n "$B_PICK_RETURN" ] || export B_PICK_RETURN="$(echo "${D2_START:-1.5,-4.5,1.5708}" | cut -d, -f1-3)"   # 回程終點預設＝替身起點（D2_START 前三個）
export DOOR_QP_DUMP_DIR=$D2OUT/${TAG}_qpfail
cd "$WS"
(setsid timeout 9000 bash evaluation/motm_dry/run_d2_stub.sh evaluation/results/motm_direct/d2/$TAG 900 > "$MISSION/logs/$TAG.out" 2>&1 < /dev/null &)
(setsid bash "$MISSION/tools/stub_end_guard.sh" "$TAG" > "$MISSION/logs/$TAG.guard" 2>&1 < /dev/null &)
(setsid bash "$MISSION/tools/nav_watchdog.sh" "$TAG" 150 "$@" > "$MISSION/logs/$TAG.navwd" 2>&1 < /dev/null &)
echo "$(date +%H:%M) 替身 $TAG 已起跑；看進度：tail -f $D2OUT/$TAG.sup.log"
