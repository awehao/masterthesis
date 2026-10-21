#!/usr/bin/env python3
"""【attempt2，2026-10-21】XH3 測試開封的**一次例外續跑**編排（Codex 20261021 裁定）。xh_open_test.py（凍結 r5，不改）的複本。

為什麼有 attempt2：attempt1（XH_test_opening.json，保留不動）在擷取第一步、模擬器啟動前因熱取樣器無讀值而 failed（換機：k10temp → coretemp），
沒有渲染任何測試影像。本編排只允許在**機器核對** attempt1 確實如此之後執行一次；attempt2 再失敗即保留並送審，**不產生 attempt3**。

與原版的差異：
  * 起跑前 `check_attempt1()`：attempt1 紀錄 failed@capture、唯一步驟 rc 92；首條目錄只有守護四檔、sim.pid 為空、guard.log 是「沒有有效讀值」、
    溫度列皆 0；其餘五條原目錄不存在；attempt2 根目錄無殘存序列；兩份評估輸出都不存在。任一不符 ⇒ 拒跑 97（不寫 attempt2 紀錄）。
  * attempt2 紀錄 `XH_test_opening_attempt2.json`：attempt=2、prior_attempt（路徑、雜湊、失敗原因、核對結果）、runs_root、test_eval_out、freeze。
  * 所有步驟引用同一份紀錄與同一個根目錄 evaluation/runs/xh_attempt2：擷取（XH_OPENING／XH_FREEZE）、核對 v2、閘門 v2、評估 v2（--opening／--runs-root／--out）。
  * 訊號遮罩、程序群組停止、例外收尾沿用原版（能力界線：只停止本次子程序所在的程序群組）。

    python3 evaluation/xh_open_test_v2.py
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
FRZ = 'evaluation/results/vision/freeze_xh3_model_r6.sha256'
ATT1 = 'evaluation/results/vision/XH_test_opening.json'
OPEN2 = 'evaluation/results/vision/XH_test_opening_attempt2.json'
ROOT = 'evaluation/runs/xh_attempt2'
OUT = 'evaluation/results/vision/XH3_test_eval_attempt2.json'
OUT1 = 'evaluation/results/vision/XH3_test_eval.json'
SEQS = ['R_x25_150_A_front', 'R_x25_150_B_oblique', 'R_x40_300_A_front', 'R_x40_300_B_oblique', 'N_k40_A_front', 'N_k40_B_oblique']
ATT1_FILES = {'guard.log', 'sim.pid', 'thermal.csv', 'thermal.log'}


def default_steps(opening=OPEN2, root=ROOT, out=OUT, freeze=FRZ):
    py, vpy = sys.executable, os.path.join(WS, '.venv-dl0', 'bin', 'python')
    return [('capture', ['bash', 'evaluation/run_xh_capture_test_v2.sh'] + SEQS, {'XH_OPENING': opening, 'XH_FREEZE': freeze}),
            ('capture_check', [py, 'evaluation/xh_capture_check_v2.py', '--runs-root', root] + SEQS, {}),
            ('capture_gate', [py, 'evaluation/xh_capture_gate_v2.py', '--runs-root', root] + SEQS, {}),
            ('test_eval', [vpy, 'evaluation/xh_test_eval_v2.py', '--ckpt', 'evaluation/xh_ckpt/run1/ep11.pth',
                           '--opening', opening, '--runs-root', root, '--out', out], {})]


def check_attempt1(cwd, att1=ATT1, root=ROOT, out=OUT, out1=OUT1):
    """回傳 (拒絕原因清單, 核對紀錄)。只讀，不改任何檔案。"""
    J = lambda p: os.path.join(cwd, p)                   # noqa: E731
    why, ev = [], {}
    try:
        r1 = json.load(open(J(att1)))
        ev['sha256'] = hashlib.sha256(open(J(att1), 'rb').read()).hexdigest()
    except Exception as e:                               # noqa: BLE001
        return [f'attempt1_unreadable:{type(e).__name__}'], ev
    st = r1.get('steps') or []
    ev.update({'state': r1.get('state'), 'failed_step': r1.get('failed_step'), 'steps': [(x.get('step'), x.get('rc')) for x in st]})
    if r1.get('state') != 'failed' or r1.get('failed_step') != 'capture':
        why.append('attempt1_not_failed_at_capture')
    if [(x.get('step'), x.get('rc')) for x in st] != [('capture', 92)]:
        why.append('attempt1_steps_not_single_capture_rc92')
    d0 = J(f'evaluation/runs/xh_{SEQS[0]}')
    if not os.path.isdir(d0):
        why.append('attempt1_first_dir_missing')
    else:
        files = set(os.listdir(d0))
        ev['first_dir_files'] = sorted(files)
        if files != ATT1_FILES:
            why.append('attempt1_first_dir_unexpected_files')
        pid = os.path.join(d0, 'sim.pid')
        if not os.path.isfile(pid) or os.path.getsize(pid) != 0:
            why.append('attempt1_sim_started')
        try:
            if '沒有有效讀值' not in open(os.path.join(d0, 'guard.log')).read():
                why.append('attempt1_guard_reason_mismatch')
            rows = open(os.path.join(d0, 'thermal.csv')).read().splitlines()[1:]
            ev['thermal_rows'] = len(rows)
            if not rows or any(x.split(',')[2] != '0' for x in rows):
                why.append('attempt1_thermal_not_all_zero')
        except OSError as e:
            why.append(f'attempt1_evidence_unreadable:{type(e).__name__}')
    for q in SEQS[1:]:
        if os.path.exists(J(f'evaluation/runs/xh_{q}')):
            why.append(f'attempt1_other_dir_exists:{q}')
    for q in SEQS:
        if os.path.exists(J(f'{root}/xh_{q}')):
            why.append(f'attempt2_residual:{q}')
    for o in (out1, out):
        if os.path.exists(J(o)):
            why.append(f'eval_output_exists:{o}')
    return why, ev


def write(rec, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def run(steps=None, opening=OPEN2, freeze=FRZ, cwd=WS, check_freeze=True, att1=ATT1, root=ROOT, out=OUT, out1=OUT1):
    steps = steps or default_steps(opening, root, out, freeze)
    op_abs = os.path.join(cwd, opening)
    if os.path.exists(op_abs):
        print(f'attempt2 紀錄已存在（{opening}）⇒ 拒跑：只允許一次例外續跑，不產生 attempt3')
        return 92
    why, ev = check_attempt1(cwd, att1, root, out, out1)
    if why:
        print('**attempt1 證據不符「渲染前失敗」** ⇒ 拒跑：', why)
        return 97
    if check_freeze and subprocess.run(['sha256sum', '-c', '--quiet', freeze], cwd=cwd).returncode != 0:
        print('**模型／評估凍結不符** ⇒ 拒跑')
        return 95
    rec = {'state': 'opened', 'attempt': 2, 'opened_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'freeze': freeze,
           'freeze_sha256': hashlib.sha256(open(os.path.join(cwd, freeze), 'rb').read()).hexdigest() if check_freeze else None,
           'prior_attempt': {'path': att1, 'reason': '擷取第一步、模擬器啟動前：熱取樣器無有效讀值（換機 k10temp → coretemp）', **ev},
           'runs_root': root, 'test_eval_out': out,
           'ckpt': 'evaluation/xh_ckpt/run1/ep11.pth', 'rule': '一次例外續跑；不據測試結果調整模型、門檻、後端或路徑；再失敗即保留送審', 'steps': []}
    opening = op_abs
    try:
        write(rec, opening)
    except OSError as e:
        print(f'開封紀錄寫入失敗：{e} ⇒ 停止')
        return 96

    cur = {'p': None, 'step': None, 'stopping': False}

    def kill_tree(p, grace=5.0):
        """有界停止本次子程序與其後代（子程序在自己的程序群組）；不碰其他程序。"""
        if p is None:
            return 'not_running'
        # 父程序已退出不代表整個群組已退出；仍須處理持有管線的孫程序。
        def group_alive():
            p.poll()  # 回收本次直接子程序，避免它的 zombie 維持群組存在。
            try:
                os.killpg(p.pid, 0)
                return True
            except ProcessLookupError:
                return False

        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            p.wait(timeout=1)
            return 'not_running'
        deadline = time.monotonic() + grace
        while group_alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        if group_alive():
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait(timeout=1)
            return 'killed'
        p.wait(timeout=1)
        return 'terminated'

    def on_sig(s, _f):
        if cur['stopping']:
            return
        cur['stopping'] = True
        try:
            how = kill_tree(cur['p'])
        except Exception as e:  # 清理失敗必須記錄，不能冒充已停止。
            how = f'cleanup_failed:{type(e).__name__}: {e}'
        rec.update({'state': 'interrupted', 'signal': signal.Signals(s).name, 'interrupted_step': cur['step'],
                    'child_stop': how, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')})
        try:
            write(rec, opening)
        finally:
            os._exit(128 + s)
    old = {s: signal.signal(s, on_sig) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for name, cmd, extra in steps:
            t0 = time.time()
            cur['step'] = name
            try:
                # 「建立子程序＋登記控制代碼」期間阻擋 SIGINT／SIGTERM：否則訊號落在 Popen 與登記之間時，處理器拿不到
                # 已出生的子程序。恢復遮罩後待決訊號才送達，此時 cur['p'] 已登記。子程序在 exec 前恢復原遮罩，不繼承阻擋。
                prev = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
                try:
                    p = subprocess.Popen(cmd, cwd=cwd, env=dict(os.environ, **extra), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                         start_new_session=True,
                                         preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, prev))
                    cur['p'] = p
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, prev)
                rec['running'] = {'step': name, 'pid': p.pid}
                write(rec, opening)
                out, err = p.communicate()
                rc_ = p.returncode
                # 即使父程序成功退出，也不允許把仍運行的同群組後代帶到下一步。
                cleanup = kill_tree(p)
                if cleanup != 'not_running':
                    rec['child_cleanup'] = cleanup
                    rc_ = rc_ or 1
            except Exception as e:                       # noqa: BLE001  編排例外（例如找不到指令）⇒ failed
                try:
                    cleanup = kill_tree(cur['p'])
                except Exception as stop_e:
                    cleanup = f'cleanup_failed:{type(stop_e).__name__}: {stop_e}'
                rec['steps'].append({'step': name, 'rc': None, 'exception': f'{type(e).__name__}: {e}'})
                rec.update({'state': 'failed', 'failed_step': name, 'child_stop': cleanup,
                            'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')})
                write(rec, opening)
                print(f'**步驟 {name} 編排例外：{type(e).__name__}** ⇒ 停止；紀錄 failed')
                return 1
            finally:
                cur['p'] = None
            p = type('R', (), {'returncode': rc_, 'stdout': out or '', 'stderr': err or ''})()
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
    except Exception as e:
        rec.update({'state': 'failed', 'failed_step': cur['step'],
                    'exception': f'{type(e).__name__}: {e}',
                    'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')})
        try:
            write(rec, opening)
        except OSError as record_error:
            print(f'無法保存 failed 狀態：{record_error}；不得視為未開封重跑', file=sys.stderr)
        return 1
    finally:
        for s, h in old.items():
            signal.signal(s, h)


if __name__ == '__main__':
    sys.exit(run())
