#!/usr/bin/env python3
"""D1 S4 r6：擷取端故障隔離的假物件反例（Codex 20261005 複核；不需 Isaac）。

    python3 evaluation/test_wrist_live_guard.py
"""
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wrist_live import guarded_step, safe_close  # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


class Fake:
    def __init__(self, step_exc=None, fail_exc=None, close_exc=None):
        self.step_exc, self.fail_exc, self.close_exc = step_exc, fail_exc, close_exc
        self.n_step = self.n_fail = self.n_close = 0

    def on_step(self, world, t, k):
        self.n_step += 1
        if self.step_exc:
            raise self.step_exc

    def fail(self, t, k, e, tb):
        self.n_fail += 1
        if self.fail_exc:
            raise self.fail_exc

    def close(self):
        self.n_close += 1
        if self.close_exc:
            raise self.close_exc


logs = []
# 1 正常
w = Fake()
act, dead = guarded_step(w, None, 1.0, 100, log=logs.append)
check('ok_step_stays_active', act is w and dead is None)
# 2 擷取例外、記錄成功 ⇒ 停用且已記錄
w = Fake(step_exc=TypeError('int'))
act, dead = guarded_step(w, None, 1.0, 100, log=logs.append)
check('step_exc_disables', act is None and dead is w and w.n_fail == 1)
# 3 擷取例外、**故障記錄也失敗** ⇒ 不拋出、仍停用（Codex 反例 1）
w = Fake(step_exc=TypeError('int'), fail_exc=OSError('disk'))
try:
    act, dead = guarded_step(w, None, 1.0, 100, log=logs.append)
    ok = act is None and dead is w
except Exception as e:                                      # noqa: BLE001
    ok, act = False, repr(e)
check('fail_exc_does_not_escape_and_disables', ok, act)
# 4 連記錄函式本身都失敗 ⇒ 仍不拋出
def bad_log(msg):
    raise RuntimeError('log')
w = Fake(step_exc=ValueError('x'), fail_exc=OSError('disk'))
try:
    act, dead = guarded_step(w, None, 1.0, 100, log=bad_log)
    ok = act is None and dead is w
except Exception as e:                                      # noqa: BLE001
    ok = False
check('log_failure_does_not_escape', ok)
# 5 停用後主迴圈不再呼叫（模擬三步）
w = Fake(step_exc=TypeError('int'), fail_exc=OSError('disk'))
cur, done = w, None
for k in range(3):
    if cur is not None:
        cur, d = guarded_step(cur, None, k * 0.01, k, log=logs.append)
        done = d or done
check('disabled_not_called_again', w.n_step == 1 and cur is None and done is w, w.n_step)
# 6 close 失敗不得阻止封存（Codex 反例 2）：照模擬器收尾順序 safe_close → 寫 room_run.json
d = tempfile.mkdtemp()
w = Fake(close_exc=OSError('meta write'))
try:
    r = safe_close(w, log=logs.append)
    json.dump({'steps': []}, open(os.path.join(d, 'room_run.json'), 'w'))
    ok = (r is False) and os.path.exists(os.path.join(d, 'room_run.json'))
except Exception:                                           # noqa: BLE001
    ok = False
check('close_failure_does_not_block_archive', ok)
check('safe_close_none_ok', safe_close(None) is True)
check('safe_close_log_failure_ok', safe_close(Fake(close_exc=OSError('x')), log=bad_log) is False)
# 7 靜態：模擬器只在 wrist_live 路徑內呼叫，且 safe_close 位在 room_run.json 寫出之前
src = open(os.path.join(HERE, 'isaac_drawer_room_sim.py')).read()
i_close = src.index('safe_close(wlive if wlive is not None else _wlive_done)')
i_dump = src.index("out = os.path.join(a.out, 'room_run.json')")
check('static_safe_close_before_room_run_dump', i_close < i_dump)
check('static_safe_close_guarded_by_flag',
      re.search(r"if a\.wrist_live:\n\s+safe_close\(", src) is not None)
check('static_guarded_step_inside_wlive_branch',
      re.search(r"if wlive is not None:\n(?:\s+#.*\n)*\s+wlive, _wl_failed = guarded_step\(", src) is not None)
check('static_no_bare_on_step_call', 'wlive.on_step(' not in src)

print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
