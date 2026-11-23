#!/usr/bin/env bash
# Isaac 趟次登錄＋凍結（起跑前核對程式雜湊；改過程式後要重新登錄）。
# 用法：mission/launch/register_isaac.sh <N> <替身TAG> "本趟說明" ["設定追加"] ["控制追加；分號分隔"]
source "$(dirname "$0")/_env.sh"
python3 "$MISSION/tools/make_registration.py" "$@"
