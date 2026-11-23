#!/usr/bin/env bash
# Mask R-CNN 訓練（.venv-dl0，GPU；不要和 Isaac 同時跑）。驗證趟／測試趟依趟號切分。
# 用法：mission/launch/seg_train.sh <資料夾,逗號分隔> <輸出資料夾> [--iters 6000 --dev-runs 84 --test-runs 85,86,87]
source "$(dirname "$0")/_env.sh"
DATA=${1:?資料夾}; OUT=${2:?輸出}; shift 2
cd "$WS"
.venv-dl0/bin/python -u evaluation/house_seg.py train --data "$DATA" --out "$OUT" "$@"
