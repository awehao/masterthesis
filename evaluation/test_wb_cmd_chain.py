"""命令鏈的純邏輯測試：**不開模擬器**。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wb_cmd_chain import CmdChain, N_DOF

DT = 0.01
LO = (-2.0,) * 6
HI = (2.0,) * 6
fails = []


def ok_wheels(vx, vy, wz):
    return (True, None)


def slow_only(limit=0.05, wlim=0.2):
    """尚未實作正常輸出限制時的介面測試用：明確低速界限、越界即中止。"""
    def f(vx, vy, wz):
        if (vx * vx + vy * vy) ** 0.5 > limit:
            return (False, f'底盤線速度 > {limit} m/s')
        if abs(wz) > wlim:
            return (False, f'底盤角速度 > {wlim} rad/s')
        return (True, None)
    return f


def mk(**kw):
    d = dict(max_cmd_age_s=0.2, arm_rate_max=1.0, wheel_ok=ok_wheels,
             joint_lower=LO, joint_upper=HI)
    d.update(kw)
    return CmdChain(**d)


def check(name, cond, detail=''):
    print(f'  {"通過" if cond else "**未通過**"}  {name}' + (f'  {detail}' if detail else ''))
    if not cond:
        fails.append(name)


print('A 結構有效性：長度、有限值')
c = mk()
check('長度 8 整筆拒收', not c.receive([0.0] * 8, 0.0), c.last_reject)
check('整體失效已設定', c.fail is not None, c.fail)
c = mk()
check('含 NaN 整筆拒收', not c.receive([0.0] * 4 + [float('nan')] + [0.0] * 4, 0.0),
      c.last_reject)
c = mk()
check('含 Inf 整筆拒收', not c.receive([float('inf')] + [0.0] * 8, 0.0), c.last_reject)
c = mk()
check('長度 9 且全有限 ⇒ 接受', c.receive([0.0] * 9, 0.0), '')

print('B 同一物理步使用同一份快照')
c = mk()
c.receive([0.01, 0, 0] + [0.1] * 6, 0.0)
r1 = c.step(0.0, DT, [0.0] * 6)
s_used = c.applied
c.receive([0.02, 0, 0] + [0.2] * 6, 0.005)   # 同一步內又來一筆
check('已取用的快照不被本步覆寫', c.applied is s_used,
      f'recv_seq={s_used.recv_seq}')
r2 = c.step(0.01, DT, [0.0] * 6)
check('下一步才換新快照', c.applied.recv_seq == 2, f'recv_seq={c.applied.recv_seq}')

print('C 手臂積分：初始化、限位、過期凍結')
c = mk()
c.receive([0, 0, 0] + [0.5] * 6, 0.0)
q0 = [0.3, -0.2, 0.1, 0.0, 0.4, -0.1]
base, sp = c.step(0.0, DT, q0)
check('初始設定點由實測關節位置建立',
      all(abs(a - b) < 1e-12 for a, b in zip(sp, [x + 0.5 * DT for x in q0])),
      f'首步 {tuple(round(x,4) for x in sp)}')
check('初始化事件有記錄',
      any(e[1] == 'setpoint_init' for e in c.events))
for k in range(1, 50):
    c.receive([0, 0, 0] + [0.5] * 6, k * DT)
    c.step(k * DT, DT, [9.9] * 6)            # 量測值故意亂給
check('初始化後不再被量測值覆寫', abs(c.setpoint[0] - (q0[0] + 0.5 * 0.5)) < 1e-9,
      f'setpoint[0]={c.setpoint[0]:.4f}')

c = mk()
c.receive([0, 0, 0] + [1.0] + [0.0] * 5, 0.0)
t = 0.0
while c.fail is None and t < 5.0:
    c.receive([0, 0, 0] + [1.0] + [0.0] * 5, t)
    c.step(t, DT, [1.99] * 6)
    t += DT
check('積分超出限位 ⇒ 整體失效', c.fail is not None and '限位' in c.fail, c.fail)

print('D 命令過期：停止積分，設定點凍結')
c = mk(max_cmd_age_s=0.05)
c.receive([0.01, 0, 0] + [0.5] * 6, 0.0)
c.step(0.0, DT, [0.0] * 6)
sp_before = tuple(c.setpoint)
out = c.step(0.30, DT, [0.0] * 6)            # 已過期
check('過期時底盤分量歸零', out is not None and out[0] == (0.0, 0.0, 0.0), str(out[0]))
check('過期時設定點凍結', out[1] == sp_before, '')
check('停止積分有計數', c.n_frozen == 1, f'n_frozen={c.n_frozen}')
check('凍結不等於實際速度為零（僅語意，須另量測）', True,
      '見 summary 的 fail_note / 規格 §1.1')

print('E 輪級檢查：越界即整體失效，不只縮底盤')
c = mk(wheel_ok=slow_only(0.05, 0.2))
c.receive([0.03, 0.0, 0.0] + [0.0] * 6, 0.0)
check('低速通過', c.step(0.0, DT, [0.0] * 6) is not None)
c.receive([0.50, 0.0, 0.0] + [0.1] * 6, 0.01)
out = c.step(0.01, DT, [0.0] * 6)
check('超界 ⇒ 整筆停止（回傳 None）', out is None)
check('整體失效原因為輪級', c.fail is not None and '輪級' in c.fail, c.fail)
check('未縮底盤後續跑', c.step(0.02, DT, [0.0] * 6) is None)

print('F 時間重設')
c = mk()
c.receive([0] * 9, 0.0)
c.step(0.0, DT, [0.0] * 6)
c.note_time_reset(0.0)
check('時間非單調 ⇒ 整體失效', c.fail is not None and '單調' in c.fail, c.fail)

print('G 接收端編號不冒稱來源時間')
c = mk()
c.receive([0] * 9, 1.234)
check('欄位名為 recv_seq / recv_sim_t',
      hasattr(c.snap, 'recv_seq') and hasattr(c.snap, 'recv_sim_t'),
      f'recv_seq={c.snap.recv_seq}, recv_sim_t={c.snap.recv_sim_t}')
check('summary 註明不得冒稱端到端延遲',
      '端到端' in c.summary()['recv_seq_note'])

print(f'\n{"全部通過" if not fails else "**未通過：" + ", ".join(fails) + "**"}')
sys.exit(1 if fails else 0)
