#!/usr/bin/env python3
"""【v2，2026-10-21】xh_capture_gate.py（凍結，不改）的入口複本：判定函式 gate() 直接匯入原版（不複製）；差異只有 --runs-root 明示，
並要求報告記錄的 run_dir 與實際讀取位置一致（防止讀到別的根目錄的報告）。
原：XH 擷取核對閘門：讀 xh_capture_check.py 寫出的 analysis/xh_capture_check.json，依明確條件判定；任一條不通過即結束碼 1。

條件（與 XH2 採集核對一致）：報告存在；資產雜湊＝登錄且模擬器載入該資產；真值欄位逐格齊全；真值對資產幾何最大差 < 0.01 mm；
渲染時間嚴格遞增且影格唯一；位姿與渲染時間差 < 1e-6 s；標註讀取無 EvidenceError；N 資產不得有 handle_center、R 資產必須有。

    python3 evaluation/xh_capture_gate_v2.py --runs-root <根目錄> <序列名> [...]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


sys.path.insert(0, HERE)
from xh_capture_gate import gate                       # noqa: E402  判定條件與原版同一份程式
WS = os.path.abspath(os.path.join(HERE, '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', required=True)
    ap.add_argument('seqs', nargs='+')
    a = ap.parse_args()
    root = os.path.abspath(a.runs_root)
    bad = {}
    for seq in a.seqs:
        D = os.path.join(root, f'xh_{seq}')
        p = os.path.join(D, 'analysis', 'xh_capture_check.json')
        rep = json.load(open(p)) if os.path.exists(p) else None
        w = gate(rep)
        if rep is not None and rep.get('run_dir') != os.path.relpath(D, WS):
            w = w + ['run_dir_mismatch']
        if w:
            bad[seq] = w
    print(json.dumps({'pass': not bad, 'fail': bad, 'runs_root': os.path.relpath(root, WS)}, ensure_ascii=False))
    sys.exit(0 if not bad and a.seqs else 1)

if __name__ == '__main__':
    main()
