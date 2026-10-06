#!/usr/bin/env bash
# XH3 測試資產一次開封（只執行一次；開封紀錄存在即拒跑；失敗保留、不重開）。
#   bash evaluation/xh_open_test.sh
# 步驟：核對 freeze_xh3_model.sha256 → 寫開封紀錄（opened）→ 渲染 6 條 test 序列（run_xh_capture_test.sh）→ 擷取核對 →
#       測試評估（xh_test_eval.py，完整分母、complete 旗標）→ 更新開封紀錄（complete／failed＋各步 rc）。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"; cd "$WS"
FRZ=evaluation/results/vision/freeze_xh3_model_r2.sha256   # r1 被取代（開封腳本 venv 路徑寫錯）
OPEN=evaluation/results/vision/XH_test_opening.json
SEQS="R_x25_150_A_front R_x25_150_B_oblique R_x40_300_A_front R_x40_300_B_oblique N_k40_A_front N_k40_B_oblique"
[ -e "$OPEN" ] && { echo "開封紀錄已存在（$OPEN）⇒ 拒跑：測試只開封一次"; exit 92; }
sha256sum -c --quiet "$FRZ" || { echo "**模型／評估凍結不符** ⇒ 拒跑"; exit 95; }
python3 - "$OPEN" "$FRZ" <<'PY'
import json, sys, hashlib, time
json.dump({'state': 'opened', 'opened_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'freeze': sys.argv[2], 'freeze_sha256': hashlib.sha256(open(sys.argv[2], 'rb').read()).hexdigest(),
           'ckpt': 'evaluation/xh_ckpt/run1/ep11.pth', 'rule': '只開封一次；不據測試結果調整模型、門檻、後端或路徑'},
          open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
PY
bash evaluation/run_xh_capture_test.sh $SEQS; RC_CAP=$?
RC_CHK=-1; RC_EVAL=-1
if [ "$RC_CAP" = 0 ]; then
  python3 evaluation/xh_capture_check.py $SEQS > evaluation/runs/xh3_test_capture_check.log 2>&1; RC_CHK=$?
  .venv-dl0/bin/python evaluation/xh_test_eval.py --ckpt evaluation/xh_ckpt/run1/ep11.pth > evaluation/runs/xh3_test_eval.log 2>&1; RC_EVAL=$?
fi
python3 - "$OPEN" "$RC_CAP" "$RC_CHK" "$RC_EVAL" <<'PY'
import json, sys, time, os
p = sys.argv[1]; d = json.load(open(p))
rc = {'capture': int(sys.argv[2]), 'capture_check': int(sys.argv[3]), 'test_eval': int(sys.argv[4])}
ev = 'evaluation/results/vision/XH3_test_eval.json'
comp = json.load(open(ev)).get('complete') if os.path.exists(ev) else None
d.update({'state': 'complete' if all(v == 0 for v in rc.values()) and comp else 'failed', 'rc': rc, 'eval_complete': comp,
          'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')})
json.dump(d, open(p, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(d, ensure_ascii=False))
PY
