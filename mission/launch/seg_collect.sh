#!/usr/bin/env bash
# Mask R-CNN 訓練資料收集（會開 Isaac；沿過去實跑紀錄擺位＋算手腕相機 RGB／深度／語意標註）。
# 用法：mission/launch/seg_collect.sh <輸出資料夾> <趟號,逗號分隔> [其他參數，如 --every-s 2.0]
#   例：mission/launch/seg_collect.sh evaluation/house_seg_data/v1 53,54,55
source "$(dirname "$0")/_env.sh"
OUT=${1:?輸出資料夾}; RUNS=${2:?趟號}; shift 2
pgrep -f "[i]saac_m8_house" >/dev/null && { echo "已有 Isaac 在跑"; exit 1; }
cd "$WS"
~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_seg_collect.py --runs "$RUNS" --out "$OUT" "$@"
