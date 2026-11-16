#!/usr/bin/env bash
# 過濾後快進推送（本機 main 含 evaluation/runs 40 GB，不能直推）。流程見記憶 reference_push_filtered：
# 固定基底 b9cae11bb434 起濾掉 evaluation/runs，裸 clone 上 filter-repo 後快進推送；不加 force，非快進即出錯。
set -eu
WS="$(cd "$(dirname "$0")/.." && pwd)"
BASE=$(git -C "$WS" rev-parse b9cae11bb434)
TMPD=$(mktemp -d "${PUSH_TMP:-/tmp}/pushf.XXXXXX")
R="$TMPD/r.git"
git clone -q --bare --shared "$WS" "$R"
export GIT_DIR="$R"
git filter-repo --force --refs "$BASE..main" --path evaluation/runs --invert-paths > "$TMPD/filter.log" 2>&1
BIG=$(git rev-list --objects "$BASE..main" | git cat-file --batch-check='%(objecttype) %(objectsize)' | awk '$1=="blob" && $2>100000000' | wc -l)
[ "$BIG" = "0" ] || { echo "有 >100 MB 的檔案：$BIG"; exit 2; }
git push -q git@github.com:awehao/masterthesis.git main:main
H=$(git rev-parse main); RH=$(git ls-remote git@github.com:awehao/masterthesis.git refs/heads/main | cut -f1)
unset GIT_DIR
rm -rf "$TMPD"
echo "pushed; github main=${RH:0:9} local=${H:0:9}; max blob $BIG"
