#!/usr/bin/env python3
"""XH3 測試資產一次開封編排（取代 xh_open_test.sh）。只執行一次；開封紀錄存在即拒跑；任一步失敗即停止後續、保存原因並以非 0 結束。

    python3 evaluation/xh_open_test.py
步驟：核對 freeze_xh3_model_r3.sha256 → 寫開封紀錄 opened（寫入失敗即停）→ 渲染 6 條 test 序列 → 擷取核對（報告）→ 擷取閘門（明確判定）
     → 測試評估（完整分母、complete）→ 紀錄 complete／failed（含失敗步驟與原因）。SIGINT／SIGTERM 中斷：紀錄 interrupted，不留可繼續的 opened。
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
FRZ = 'evaluation/results/vision/freeze_xh3_model_r3.sha256'
OPEN = os.path.join(HERE, 'results', 'vision', 'XH_test_opening.json')
SEQS = ['R_x25_150_A_front', 'R_x25_150_B_oblique', 'R_x40_300_A_front', 'R_x40_300_B_oblique', 'N_k40_A_front', 'N_k40_B_oblique']


def default_steps():
    py, vpy = sys.executable, os.path.join(WS, '.venv-dl0', 'bin', 'python')
    return [('capture', ['bash', 'evaluation/run_xh_capture_test.sh'] + SEQS),
            ('capture_check', [py, 'evaluation/xh_capture_check.py'] + SEQS),
            ('capture_gate', [py, 'evaluation/xh_capture_gate.py'] + SEQS),
            ('test_eval', [vpy, 'evaluation/xh_test_eval.py', '--ckpt', 'evaluation/xh_ckpt/run1/ep11.pth'])]


def write(rec, path):
    tmp = path + '.tmp'
    json.dump(rec, open(tmp, 'w'), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def run(steps=None, opening=OPEN, freeze=FRZ, cwd=WS, check_freeze=True):
    steps = steps or default_steps()
    if os.path.exists(opening):
        print(f'開封紀錄已存在（{opening}）⇒ 拒跑：測試只開封一次')
        return 92
    if check_freeze and subprocess.run(['sha256sum', '-c', '--quiet', freeze], cwd=cwd).returncode != 0:
        print('**模型／評估凍結不符** ⇒ 拒跑')
        return 95
    rec = {'state': 'opened', 'opened_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'freeze': freeze,
           'freeze_sha256': hashlib.sha256(open(os.path.join(cwd, freeze), 'rb').read()).hexdigest() if check_freeze else None,
           'ckpt': 'evaluation/xh_ckpt/run1/ep11.pth', 'rule': '只開封一次；不據測試結果調整模型、門檻、後端或路徑', 'steps': []}
    try:
        write(rec, opening)
    except OSError as e:
        print(f'開封紀錄寫入失敗：{e} ⇒ 停止')
        return 96

    def on_sig(s, _f):
        rec.update({'state': 'interrupted', 'signal': signal.Signals(s).name, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')})
        write(rec, opening)
        sys.exit(130)
    old = {s: signal.signal(s, on_sig) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for name, cmd in steps:
            t0 = time.time()
            p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            rec['steps'].append({'step': name, 'rc': p.returncode, 'wall_s': round(time.time() - t0, 1),
                                 'stdout_tail': p.stdout[-1500:], 'stderr_tail': p.stderr[-1500:]})
            write(rec, opening)
            if p.returncode != 0:
                rec.update({'state': 'failed', 'failed_step': name, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')})
                write(rec, opening)
                print(f'**步驟 {name} 失敗（rc {p.returncode}）** ⇒ 停止後續；紀錄 failed')
                return 1
        rec.update({'state': 'complete', 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')})
        write(rec, opening)
        print('開封完成：', json.dumps([(s['step'], s['rc']) for s in rec['steps']], ensure_ascii=False))
        return 0
    finally:
        for s, h in old.items():
            signal.signal(s, h)


if __name__ == '__main__':
    sys.exit(run())
