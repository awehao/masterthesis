#!/usr/bin/env python3
"""run_base_residual_probe.sh 離線測試（Codex 20261025_021820 必修②；假 Isaac 程序，B0_RUNS_DIR 指到暫存目錄，不碰 evaluation/runs）。
  P1 假程序不結束 ⇒ TIMEOUT_S 後有界收尾（terminated／killed），非 0、標 incomplete；
  P2 假程序寫完整 probe.json、rc 0 ⇒ 0；P3 寫不完整 probe.json ⇒ 3；P4 已有 probe.json 或啟動標記 ⇒ 91 拒跑；P5 條件名不合法 ⇒ 97。

    python3 evaluation/test_run_base_residual_probe.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
FAIL = []

FAKE = r'''#!/usr/bin/env bash
# 假 Isaac：$MODE＝hang／complete／incomplete；參數同 isaac_base_residual_probe.py
OUT=""; while [ $# -gt 0 ]; do [ "$1" = --out ] && OUT="$2"; shift; done
case "$FAKE_MODE" in
  hang) trap '' TERM; while true; do sleep 1; done ;;
  complete) echo '{"summary": {"complete": true, "class": "small"}}' > "$OUT/probe.json"; echo "[probe] 彙整 ok"; exit 0 ;;
  incomplete) echo '{"summary": {"complete": false, "why": ["signal_15"], "class": null}}' > "$OUT/probe.json"; exit 3 ;;
esac
'''


def ck(n, ok, d=''):
    print(f'  {n:58s} {"ok" if ok else "**錯**"}  {d}')
    if not ok:
        FAIL.append(n)


def run(tmp, rid, mode, cond='k2_ready__zero', timeout='3'):
    fake = os.path.join(tmp, 'fake_isaac.sh')
    open(fake, 'w').write(FAKE)
    os.chmod(fake, 0o755)
    env = dict(os.environ, RUN_ID=rid, COND=cond, ISAAC_PY=fake, FAKE_MODE=mode, TIMEOUT_S=timeout, B0_RUNS_DIR=tmp)
    t0 = time.monotonic()
    p = subprocess.run(['bash', os.path.join(HERE, 'run_base_residual_probe.sh')], cwd=WS, env=env, capture_output=True, text=True, timeout=120)
    return p, time.monotonic() - t0, os.path.join(tmp, rid)


def main():
    tmp = tempfile.mkdtemp(prefix='b0_runner_test_', dir=os.environ.get('TMPDIR', '/tmp'))
    try:
        p, dt, d = run(tmp, 'p1', 'hang')
        log = open(os.path.join(d, 'run.log')).read()
        ck('P1 逾時有界收尾、非 0、incomplete', p.returncode != 0 and ('terminated' in log or 'killed' in log) and 'incomplete' in log and dt < 60,
           f'rc={p.returncode} {dt:.1f}s')
        p, _, d = run(tmp, 'p2', 'complete')
        ck('P2 完整 ⇒ 0', p.returncode == 0, f'rc={p.returncode} {p.stdout[-200:]}')
        p, _, d = run(tmp, 'p3', 'incomplete')
        ck('P3 不完整 ⇒ 3', p.returncode == 3, f'rc={p.returncode}')
        p, _, _ = run(tmp, 'p2', 'complete')
        ck('P4 已有 probe.json ⇒ 91 拒跑', p.returncode == 91, f'rc={p.returncode}')
        os.makedirs(os.path.join(tmp, 'p4b'))
        open(os.path.join(tmp, 'p4b', 'started.marker'), 'w').write('0')
        p, _, _ = run(tmp, 'p4b', 'complete')
        ck('P4 已有啟動標記 ⇒ 91 拒跑', p.returncode == 91, f'rc={p.returncode}')
        p, _, _ = run(tmp, 'p5', 'complete', cond='stow__zero')
        ck('P5 條件名不合法 ⇒ 97', p.returncode == 97, f'rc={p.returncode}')
    finally:
        shutil.rmtree(tmp)                                                            # 只刪本測試自建的暫存目錄
    print('全部通過' if not FAIL else '失敗：' + '、'.join(FAIL))
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
