#!/usr/bin/env python3
"""`_u_prev` 的最小重現測試：**no_command 佔位回報不得冒充套用回報**。

缺陷：執行端在設定點還沒建立時，每一步也會發一筆 `/coman/applied_cmd`
（`exec_mode = 4`、欄位零值、`api_applied = False`）。`_on_applied` 不分
模式一律設 `self._applied`，於是 strict 政策的首輪會把 u_prev 標成
`applied` ——「量到的套用值」——而其實什麼都沒套用過，
`--assume-initial-rest` 也因此永遠觸發不到。

數值上兩者都是零（機器人確實靜止），**錯的是標籤**；strict 政策的全部
意義就在標籤，所以這是實質缺陷，不是措辭問題。
"""
from __future__ import annotations

import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 只取 `_u_prev` 的程式碼，不載入 rclpy：把方法從原始碼編譯出來。
SRC = open(os.path.join(HERE, 'wgmpc_wg2_node.py')).read()
i = SRC.index('    def _u_prev(self, theta):')
j = SRC.index('\n    def ', i + 10)
BODY = 'import time\nimport numpy as np\nNU = 9\n' + \
    'def body_to_world(t):\n    return np.eye(9)\n' + \
    'class _Stub:\n' + SRC[i:j]
ns: dict = {}
exec(compile(BODY, 'wgmpc_wg2_node._u_prev', 'exec'), ns)
Stub = ns['_Stub']

N = 0
BAD = []


def mk(applied, **kw):
    s = Stub()
    s._applied = applied
    s._ep_req = kw.get('ep_req')
    s._modified = kw.get('modified')
    s.a = types.SimpleNamespace(
        u_prev_policy=kw.get('policy', 'strict'),
        assume_initial_rest=kw.get('air', False),
        hist_age=0.5)
    return s


def ck(name, s, want_src, want_auth):
    global N
    N += 1
    u, src, auth = s._u_prev(0.0)
    if src != want_src or auth != want_auth:
        BAD.append(f'{name}: 期待 ({want_src}, {want_auth})，'
                   f'得到 ({src}, {auth})')
    return u, src, auth


now = time.monotonic()
H_NO_CMD = {'exec_mode': 4, 'api_applied': False, 'n_recv': 0}
H_NORMAL = {'exec_mode': 0, 'api_applied': True, 'n_recv': 7}
AP_NO_CMD = (tuple([0.0] * 9), now, 1, 0.01, H_NO_CMD)
AP_NORMAL = (tuple([0.1] * 9), now, 9, 0.09, H_NORMAL)

# ---- 缺陷本體：no_command 佔位不得回報為 'applied' ----
ck('no_command 佔位 ＋ strict，無初始靜止授權 ⇒ 不得發布',
   mk(AP_NO_CMD), 'no_valid_applied_report', False)
u, _, _ = ck('no_command 佔位 ＋ strict ＋ 初始靜止 ⇒ 零值，標初始靜止',
             mk(AP_NO_CMD, air=True), 'zeros_initial_rest', True)
assert np.allclose(u, 0.0)

# ---- 真的套用過就必須回 'applied' ----
u, _, _ = ck('正常套用回報', mk(AP_NORMAL), 'applied', True)
assert np.allclose(u, 0.1)
ck('正常套用回報：有初始靜止授權也要以實測為先',
   mk(AP_NORMAL, air=True), 'applied', True)

# ---- 正常模式下的**零命令**仍是合法的套用回報（不可被這次修改誤殺）----
u, _, _ = ck('正常模式零命令仍算 applied',
             mk((tuple([0.0] * 9), now, 9, 0.09, H_NORMAL)), 'applied', True)
assert np.allclose(u, 0.0)
# 逾時保持（exec_mode = 1）也是真的套用過
ck('逾時保持仍算 applied',
   mk((tuple([0.0] * 9), now, 9, 0.09, {'exec_mode': 1})), 'applied', True)
# 閂鎖（3）由迴圈開頭的 _chain_failed 處理，這裡仍視為套用回報
ck('閂鎖回報仍由 applied 分支回報',
   mk((tuple([0.0] * 9), now, 9, 0.09, {'exec_mode': 3})), 'applied', True)

# ---- 完全沒有回報 ----
ck('完全沒有回報 ＋ strict', mk(None), 'no_valid_applied_report', False)
ck('完全沒有回報 ＋ 初始靜止', mk(None, air=True), 'zeros_initial_rest', True)

# ---- 初始靜止只在**還沒有任何命令流動**之前適用 ----
ck('no_command 佔位，但端點已請求過 ⇒ 初始靜止不適用',
   mk(AP_NO_CMD, air=True, ep_req=(tuple([0.0] * 9), now)),
   'no_valid_applied_report', False)
ck('no_command 佔位，但安全層改過命令 ⇒ 初始靜止不適用',
   mk(AP_NO_CMD, air=True, modified=(tuple([0.0] * 9), now)),
   'no_valid_applied_report', False)

# ---- 過期的套用回報 ----
ck('套用回報過期 ＋ strict',
   mk((tuple([0.1] * 9), now - 5.0, 9, 0.09, H_NORMAL)),
   'no_valid_applied_report', False)

# ---- diagnostic 政策：另標來源，不算驗證通過 ----
ck('diagnostic ＋ no_command 佔位 ＋ 端點請求',
   mk(AP_NO_CMD, policy='diagnostic', ep_req=(tuple([0.2] * 9), now)),
   'DIAG:endpoint_requested', False)
ck('diagnostic ＋ 什麼都沒有', mk(None, policy='diagnostic'),
   'DIAG:zeros', False)

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
