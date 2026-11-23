"""E2 模擬的 --mode 選項必須與命令鏈認得的模式一致。

為什麼要這個測試：solver_drawer 已經加進 E2 的 --mode 選項與場景守衛的放行
子樹，卻漏了 MODES_E2。那種漏接不會在任何離線檢查裡出現 —— 它在實跑的
第一步建構命令鏈時才炸，而那時 Isaac 已經載入完場景。
"""
import ast, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wb_cmd_chain_e2 import CmdChainE2, MODES_E2
from wb_wheel_limit import WheelLimitConfig

OK = []
def chk(c, m):
    OK.append(bool(c)); print(f'{"ok  " if c else "FAIL"}  {m}')

# ---- A. 取出 E2 模擬 --mode 的選項清單 ----
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'isaac_wholebody_sim_e2.py')).read()
choices = None
for node in ast.walk(ast.parse(src)):
    if (isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument'
            and node.args and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == '--mode'):
        for kw in node.keywords:
            if kw.arg == 'choices':
                choices = [e.value for e in kw.value.elts]
chk(choices is not None, f'A1 找到 --mode 的 choices：{choices}')
missing = [c for c in (choices or []) if c not in MODES_E2]
chk(not missing, f'A2 每個 --mode 選項都在 MODES_E2 內；缺的是 {missing}')

# ---- B. 每個選項都真的能建構出命令鏈 ----
LO = tuple([-10.0] * 3 + [-2.0] * 6); HI = tuple([10.0] * 3 + [2.0] * 6)
for c in (choices or []):
    try:
        ch = CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0,
                        wheel_ok=lambda *a: (True, ''),
                        joint_lower=LO, joint_upper=HI, joint_margin=0.05,
                        mode=c, wheel_cfg=WheelLimitConfig(arm_rate_max=1.0))
        chk(ch.cfg['mode'] == c,
            f'B 模式 {c!r} 建構成功，且 cfg 保留原名（得 {ch.cfg["mode"]!r}）')
    except Exception as e:
        chk(False, f'B 模式 {c!r} 建構失敗：{type(e).__name__}: {e}')

# ---- C. solver_drawer 的允許分量：底盤與手臂都要放行 ----
ch = CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0,
                wheel_ok=lambda *a: (True, ''),
                joint_lower=LO, joint_upper=HI, joint_margin=0.05,
                mode='solver_drawer', wheel_cfg=WheelLimitConfig(arm_rate_max=1.0))
chk(ch.mode['base'] and ch.mode['arm'],
    'C1 solver_drawer 同時放行底盤與手臂分量（全身同動需要）')
chk(ch.receive([0.01, 0.0, 0.0] + [0.1] * 6, 0.0),
    'C2 九維命令整筆接受，不因模式被切掉分量')
chk(ch.cfg['mode'] == 'solver_drawer',
    'C3 cfg 記的是 solver_drawer，不是用來建構的 sync')

# ---- D. 未知模式仍要拒絕（不是把檢查拿掉） ----
try:
    CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0,
               wheel_ok=lambda *a: (True, ''),
               joint_lower=LO, joint_upper=HI, mode='no_such_mode',
               wheel_cfg=WheelLimitConfig())
    chk(False, 'D1 未知模式應拒絕但沒有')
except ValueError:
    chk(True, 'D1 未知模式仍正確拒絕')

print(f'\n{sum(OK)}/{len(OK)} 通過')
sys.exit(0 if all(OK) else 1)
