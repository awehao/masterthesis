#!/usr/bin/env bash
# 比較替身趟次：成敗、時間、路徑長、行人附近停住時間、對每位行人最小距離。
# 用法：mission/launch/compare_walkers.sh s199d s201d ...
source "$(dirname "$0")/_env.sh"
D2_WALKER_SPEED_SCALE=0.75 python3 "$MISSION/tools/compare_walkers.py" "$@"
