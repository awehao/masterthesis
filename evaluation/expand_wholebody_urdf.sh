#!/usr/bin/env bash
# 產生「操作模式」唯一的一份展開 URDF，Isaac 與 robot_state_publisher 共用。
#
# 為什麼需要這支：導航跑批載入的是 /tmp/omni_bot_wb.urdf，那個檔在 repo 裡
# 沒有任何產生器 —— 它是手動展開後留下的，重開機就消失，也無法核對它當初用
# 了哪些參數。本輪用展開比對還原出實際參數為
#     use_arm:=true add_gripper:=true add_arm_camera:=true use_camera:=true
# （與 /tmp/omni_bot_wb.urdf 去註解、正規化空白後逐字元相同）。
#
# 同一時間 omni_bot_dynamic.launch.py 給 robot_state_publisher 的是
#     use_camera:=false use_arm:=true          （add_gripper 取預設 false）
# 兩邊因此差兩個參數：夾爪與底盤相機。TF 樹實測止於 link_eef，沒有 link_tcp。
#
# 這支只產生檔案，不動任何既有啟動流程；導航實驗維持凍結。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(dirname "$HERE")"
SRC="$WS/src/my_omnibot_description/urdf/omni_bot.urdf.xacro"
OUT="${1:-$WS/evaluation/models/omni_bot_manip.urdf}"
ARGS=(use_arm:=true add_gripper:=true add_arm_camera:=true use_camera:=true)

command -v xacro >/dev/null || { echo "找不到 xacro，請先 source ROS 環境"; exit 1; }
mkdir -p "$(dirname "$OUT")"
xacro "$SRC" "${ARGS[@]}" > "$OUT.tmp"
{
  echo "<?xml version=\"1.0\"?>"
  echo "<!-- 由 evaluation/expand_wholebody_urdf.sh 產生，請勿手改 -->"
  echo "<!-- 來源: src/my_omnibot_description/urdf/omni_bot.urdf.xacro -->"
  echo "<!-- 參數: ${ARGS[*]} -->"
  echo "<!-- 來源 sha256: $(sha256sum "$SRC" | cut -c1-16) -->"
  tail -n +2 "$OUT.tmp"
} > "$OUT"
rm -f "$OUT.tmp"
echo "已產生 $OUT"
echo "  參數      ${ARGS[*]}"
echo "  來源 sha  $(sha256sum "$SRC" | cut -c1-16)"
echo "  輸出 sha  $(sha256sum "$OUT" | cut -c1-16)"
