#!/usr/bin/env bash
# Claude → Codex 審查：把請求排進 Howard 的 Codex 審查對話，等回合結束，取回最後一則回覆原文。
#
#   tools/codex_review.sh <請求.md> [標題]
#
# 流程：請求加上唯一標記 → `codex queue --thread <ID>`（由握有該串寫入鎖的 VS Code 面板／
#       app-server 處理，面板上看得到）→ 盯該串 rollout 紀錄，等標記之後的 task_complete →
#       取該回合最後一則 assistant 訊息。請求與回覆原文都存進 reviews/（可追溯，不改寫）。
# 界線：同一時間只讓一方在本串發起新審查；Codex 在面板端的沙盒／核准設定沿用面板的設定，
#       所以請求內文一律寫明「唯讀審查、不改檔、不啟動模擬」。
# 離開碼：0 取得回覆；2 排入失敗；3 逾時（請求已排入，回覆可能稍後才出現）
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
REQ="${1:?請求檔}"
TITLE="${2:-審查請求}"
THREAD="${CODEX_THREAD:-01a105a3-84d4-7f40-90be-5847b88ffa77}"
TIMEOUT_S="${CODEX_TIMEOUT_S:-2400}"
CODEX="${CODEX_BIN:-$(ls -d "$HOME"/.vscode/extensions/openai.chatgpt-*-linux-x64 2>/dev/null | sort -V | tail -1)/bin/linux-x86_64/codex}"
[ -x "$CODEX" ] || { echo "找不到 codex：$CODEX"; exit 2; }
ROLL=$(find "$HOME/.codex/sessions" -name "rollout-*-${THREAD}.jsonl" 2>/dev/null | head -1)
[ -n "$ROLL" ] || { echo "找不到對話紀錄：$THREAD"; exit 2; }

TS=$(date +%Y%m%d_%H%M%S)
MARK="[Claude 審查請求 $TS]"
OUTDIR="$WS/reviews"; mkdir -p "$OUTDIR"
RQ="$OUTDIR/${TS}_request.md"; RP="$OUTDIR/${TS}_reply.md"
{ echo "$MARK $TITLE"; echo; echo "（由 Claude 經 tools/codex_review.sh 送出。唯讀審查：請不要改檔、不要啟動模擬；回覆會被 Claude 讀回並原文保存。）"; echo; cat "$REQ"; } > "$RQ"
N0=$(wc -l < "$ROLL")

"$CODEX" queue --thread "$THREAD" --message "$(cat "$RQ")" > "$OUTDIR/${TS}_queue.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "排入失敗（rc=$rc）：$(tail -3 "$OUTDIR/${TS}_queue.log")"; exit 2
fi
echo "已排入 $THREAD；等待回覆（最多 ${TIMEOUT_S} s）…"

t0=$(date +%s)
while :; do
  if python3 - "$ROLL" "$N0" "$MARK" "$RP" <<'PY'
import json, sys
roll, n0, mark, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
lines = open(roll, encoding='utf-8').read().splitlines()[n0:]
seen, last = False, None
for l in lines:
    try:
        e = json.loads(l)
    except ValueError:
        continue
    p = e.get('payload') or {}
    if not isinstance(p, dict):
        continue
    if p.get('type') == 'message' and p.get('role') == 'user':
        txt = ''.join(c.get('text', '') for c in (p.get('content') or []) if isinstance(c, dict))
        if mark in txt:
            seen, last = True, None
    elif seen and p.get('type') == 'message' and p.get('role') == 'assistant':
        last = ''.join(c.get('text', '') for c in (p.get('content') or []) if isinstance(c, dict))
    elif seen and p.get('type') == 'task_complete':
        if last is not None:
            open(out, 'w', encoding='utf-8').write(last + '\n')
            sys.exit(0)
sys.exit(1)
PY
  then
    echo "回覆已存：$RP"; echo "-----"; cat "$RP"; exit 0
  fi
  [ $(( $(date +%s) - t0 )) -ge "$TIMEOUT_S" ] && { echo "逾時：請求已排入，回覆尚未完成（之後可再讀 $ROLL）"; exit 3; }
  sleep 10
done
