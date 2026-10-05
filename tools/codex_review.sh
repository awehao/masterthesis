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
# 用法二：tools/codex_review.sh --fetch <TS>   只讀回既有請求（reviews/<TS>_request.md）的回覆，不重新排入
# 離開碼另加：4 回合被中斷（turn_aborted，需重送）
THREAD="${CODEX_THREAD:-01a105a3-84d4-7f40-90be-5847b88ffa77}"
TIMEOUT_S="${CODEX_TIMEOUT_S:-2400}"
OUTDIR="$WS/reviews"; mkdir -p "$OUTDIR"
# 串的紀錄檔可能不只一個（續接時換成 rollout-…-<THREAD>_<新 id>.jsonl），只檢查存在，不預先挑一個
ls "$HOME"/.codex/sessions/*/*/*/rollout-*"${THREAD}"*.jsonl >/dev/null 2>&1 || { echo "找不到對話紀錄：$THREAD"; exit 2; }
if [ "${1:-}" = "--fetch" ]; then
  TS="${2:?--fetch 需要請求時間戳 TS}"
  MARK="[Claude 審查請求 $TS]"
  RP="$OUTDIR/${TS}_reply.md"
  [ -f "$OUTDIR/${TS}_request.md" ] || { echo "找不到請求 $OUTDIR/${TS}_request.md"; exit 2; }
  TIMEOUT_S="${CODEX_TIMEOUT_S:-0}"
else
  REQ="${1:?請求檔}"
  TITLE="${2:-審查請求}"
  CODEX="${CODEX_BIN:-$(ls -d "$HOME"/.vscode/extensions/openai.chatgpt-*-linux-x64 2>/dev/null | sort -V | tail -1)/bin/linux-x86_64/codex}"
  [ -x "$CODEX" ] || { echo "找不到 codex：$CODEX"; exit 2; }
  TS=$(date +%Y%m%d_%H%M%S)
  MARK="[Claude 審查請求 $TS]"
  RQ="$OUTDIR/${TS}_request.md"; RP="$OUTDIR/${TS}_reply.md"
  { echo "$MARK $TITLE"; echo; echo "（由 Claude 經 tools/codex_review.sh 送出。唯讀審查：請不要改檔、不要啟動模擬；回覆會被 Claude 讀回並原文保存。）"; echo; cat "$REQ"; } > "$RQ"
  "$CODEX" queue --thread "$THREAD" --message "$(cat "$RQ")" > "$OUTDIR/${TS}_queue.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "排入失敗（rc=$rc）：$(tail -3 "$OUTDIR/${TS}_queue.log")"; exit 2
  fi
  echo "已排入 $THREAD；等待回覆（最多 ${TIMEOUT_S} s）…"
fi

t0=$(date +%s)
while :; do
  # **每輪重新找這個串的所有紀錄檔**：串會在面板重啟／續接時換成新的 rollout 檔（實測 2026-10-05 11:25：
  # 舊檔停在 11:24、之後寫進 rollout-…-<THREAD>_<新 id>.jsonl）。只盯開始時那一個會永遠等不到回覆。
  python3 - "$THREAD" "$MARK" "$RP" "$HOME/.codex/sessions" <<'PY'
import glob, json, os, sys
thread, mark, out, root = sys.argv[1:5]
files = sorted(glob.glob(os.path.join(root, '**', f'rollout-*{thread}*.jsonl'), recursive=True),
               key=os.path.getmtime, reverse=True)
for f in files:
    seen, last, state = False, None, None
    try:
        fh = open(f, encoding='utf-8')
    except OSError:
        continue
    for l in fh:
        if not seen and mark not in l:
            continue
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
                seen, last, state = True, None, None
        elif seen and p.get('type') == 'message' and p.get('role') == 'assistant':
            last = ''.join(c.get('text', '') for c in (p.get('content') or []) if isinstance(c, dict))
        elif seen and p.get('type') == 'task_complete' and last is not None:
            open(out, 'w', encoding='utf-8').write(last + '\n')
            print(f)
            sys.exit(0)
        elif seen and p.get('type') == 'turn_aborted' and last is None:
            state = 'aborted'
    if seen and state == 'aborted':
        print(f)
        sys.exit(4)       # 標記之後、回覆之前該回合被中斷（例如面板端中斷或重啟）
sys.exit(1)
PY
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "回覆已存：$RP"; echo "-----"; cat "$RP"; exit 0
  fi
  if [ $rc -eq 4 ]; then
    echo "**回合被中斷**：請求已送達但 Codex 該回合在回覆前被中斷（turn_aborted）。需要重送。"; exit 4
  fi
  [ $(( $(date +%s) - t0 )) -ge "$TIMEOUT_S" ] && { echo "逾時：請求已排入，回覆尚未完成（之後可用 --fetch $TS 再取）"; exit 3; }
  sleep 10
done
