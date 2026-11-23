#!/usr/bin/env python3
"""DL0 封存測試的**一次性開封**（Codex 審查：核對須在任何測試影格被讀取之前；不覆寫；失敗保留、不得當作未開封重來）。

    python3 evaluation/dl0_open_test.py

步驟（任一步失敗即停，紀錄狀態，不重試）：
  0 開封紀錄 results/vision/DL0_test_opening.json 已存在 ⇒ 拒絕（包含先前失敗的情況：須另與 Codex 決定）
  1 寫開封紀錄：開始時間、命令、git HEAD、環境版本（dl0_env_versions.txt）
  2 sha256sum -c 封存清單 seal_dl0_test.sha256 與模型凍結 freeze_dl0_model_v2.sha256 ——**在讀取任何測試影格之前**
  3 系統 python 產生測試標籤快取（dl0_label_cache.py --test；與開發標籤同一環境）
  4 凍結 D1（G0）跑兩條測試 ⇒ runs/dl0_test_*/wrist_v0/d1_detect_test.json（已存在 ⇒ 停）
  5 .venv-dl0：dl0_l1_eval.py --split test ⇒ results/vision/DL0_test_eval.json（G0／G1／L1，完整分母、逐格結果）
  6 寫結束狀態
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
V = os.path.join(HERE, 'results', 'vision')
REC = os.path.join(V, 'DL0_test_opening.json')
CKPT = os.path.join(HERE, 'dl0_ckpt', 'run1', 'ep11.pth')
VENV_PY = os.path.join(WS, '.venv-dl0', 'bin', 'python')
TESTS = ['runs/dl0_test_lateral', 'runs/dl0_test_view']


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def main():
    if os.path.exists(REC):
        print(f'開封紀錄已存在，拒絕再次開封：{REC}')
        return 2
    rec = {'schema': 'dl0_test_opening/1', 'start': now(), 'command': ' '.join([sys.executable] + sys.argv),
           'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=WS, capture_output=True, text=True).stdout.strip(),
           'versions': open(os.path.join(V, 'dl0_env_versions.txt')).read(), 'steps': [], 'status': 'started'}

    def write():
        json.dump(rec, open(REC, 'w'), ensure_ascii=False, indent=1)

    write()

    def step(name, cmd, cwd=WS, env=None):
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        rec['steps'].append({'step': name, 'cmd': ' '.join(cmd), 'rc': r.returncode, 'time': now(),
                             'stdout_tail': r.stdout[-1500:], 'stderr_tail': r.stderr[-1500:]})
        write()
        if r.returncode != 0:
            rec['status'] = f'failed_at {name}'
            rec['end'] = now()
            write()
            print('失敗於', name, r.stdout[-500:], r.stderr[-500:])
            sys.exit(1)

    # 2 核對（讀取任何測試影格之前）
    step('verify_seal', ['sha256sum', '-c', '--quiet', 'evaluation/results/vision/seal_dl0_test.sha256'])
    step('verify_model_freeze', ['sha256sum', '-c', '--quiet', 'evaluation/results/vision/freeze_dl0_model_v2.sha256'])
    for t in TESTS:
        out = os.path.join(HERE, t, 'wrist_v0', 'd1_detect_test.json')
        if os.path.exists(out):
            rec['status'] = f'failed_at precheck（{out} 已存在）'
            write()
            sys.exit(1)
    # 3 測試標籤（系統 python）
    env = dict(os.environ, DL0_OPENING_RECORD=REC)
    step('test_labels', [sys.executable, 'evaluation/dl0_label_cache.py', '--test'], env=env)
    # 4 G0：凍結 D1
    for t in TESTS:
        step(f'G0_{os.path.basename(t)}', [sys.executable, 'evaluation/d1_handle_detect.py', f'evaluation/{t}', '--tag', 'test'])
    # 5 G1／L1
    step('G1_L1_eval', [VENV_PY, 'evaluation/dl0_l1_eval.py', '--ckpt', os.path.relpath(CKPT, WS), '--split', 'test'])
    ev = json.load(open(os.path.join(V, 'DL0_test_eval.json')))
    rec['status'] = 'completed' if ev.get('complete') else 'completed_with_evidence_issues'
    rec['evidence_issues'] = ev.get('evidence_issues')
    rec['end'] = now()
    write()
    print('完成', rec['status'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
