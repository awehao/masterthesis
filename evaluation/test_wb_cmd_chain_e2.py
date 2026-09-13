"""E2 命令鏈的離線測試：不開模擬器、不含 ROS。

驗 E2 相對 E1 的**唯一差異**（輪級限制）確實生效，且 E1 的檢查語意未變。
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from wb_cmd_chain import CmdChain                                    # noqa
from wb_cmd_chain_e2 import CmdChainE2                               # noqa
from wb_wheel_limit import WheelLimitConfig, wheel_matrix            # noqa

LO = [-2.9356, -2.4435, -0.0611, -2.9356, -1.9897, -2.9356]
HI = [2.9356, 2.4435, 2.9356, 2.9356, 1.9897, 2.9356]
DT = 0.01
CFG = WheelLimitConfig()
W = wheel_matrix(CFG.wheel_base_L)

fails = []
def expect(n, c, d=''):
    print(f"  [{'ok' if c else '**FAIL**'}] {n}{(' — ' + d) if d else ''}")
    if not c:
        fails.append(n)


def mk(cls, **kw):
    return cls(max_cmd_age_s=0.2, arm_rate_max=1.0,
               wheel_ok=lambda *v: (True, ''), mode='sync',
               joint_lower=LO, joint_upper=HI, **kw)


def cmd(vx=0.0, vy=0.0, wz=0.0, dq2=0.0):
    v = [0.0] * 9
    v[0], v[1], v[2], v[4] = vx, vy, wz, dq2
    return v


print('A E1 的檢查語意未變')
for cls, name in [(CmdChain, 'E1'), (CmdChainE2, 'E2')]:
    c = mk(cls)
    bad = cmd(0.03); bad[4] = 5.0                 # 手臂速率超限
    c.receive(bad, 1.0)
    c.step(1.0, DT, [0.0] * 6)
    expect(f'A1 {name} 手臂超速仍整筆失效', c.fail is not None,
           str(c.fail)[:44])
    c2 = mk(cls)
    expect(f'A2 {name} 結構不合整筆拒收',
           c2.receive([0.0] * 8, 1.0) is False and c2.summary()['rejected'] == 1)

print('B 低幅度：E2 與 E1 輸出一致（λ=1，不被修改）')
e1, e2 = mk(CmdChain), mk(CmdChainE2)
same = True
for k in range(300):
    t = 1.0 + k * DT
    v = cmd(0.03, dq2=0.05)
    e1.receive(v, t); e2.receive(v, t)
    o1 = e1.step(t, DT, [0.0] * 6)
    o2 = e2.step(t, DT, [0.0] * 6)
    if not (np.allclose(o1[0], o2[0]) and np.allclose(o1[1], o2[1])):
        same = False
expect('B1 本階段幅度下 E2 與 E1 逐筆一致', same)
expect('B2 且 E2 記錄 modified 步數為 0',
       e2.summary()['wheel_limit']['modified_steps'] == 0)

print('C 階躍＋反向：限制**實際觸發**')
e2 = mk(CmdChainE2)
seq, lam_lt1 = [], 0
for k in range(400):
    t = 1.0 + k * DT
    sgn = 1.0 if (k // 50) % 2 == 0 else -1.0     # 每 0.5 s 反向一次
    v = cmd(0.045 * sgn, wz=0.18 * sgn, dq2=0.05 * sgn)
    e2.receive(v, t)
    out = e2.step(t, DT, [0.0] * 6)
    seq.append(np.array(out[0]))
    if e2.last_limit['modified']:
        lam_lt1 += 1
expect('C1 λ<1 的步數 > 0（限制確實被觸發）', lam_lt1 > 0,
       f'{lam_lt1} / 400 步')
acc = max(float(np.max(np.abs(W @ (seq[k + 1] - seq[k])))) / DT
          for k in range(len(seq) - 1))
lim_a = CFG.wheel_a_max * CFG.wheel_radius
expect('C2 全序列輪加速度都在上限內', acc <= lim_a + 1e-6,
       f'{acc:.4f} ≤ {lim_a:.4f} m/s²')
sp = max(float(np.max(np.abs(W @ s))) for s in seq)
expect('C3 全序列輪速都在上限內', sp <= CFG.w_lim + 1e-9,
       f'{sp:.6f} ≤ {CFG.w_lim:.6f}')
expect('C4 未提高既有速度界限',
       max(abs(float(s[0])) for s in seq) <= 0.05 + 1e-9
       and max(abs(float(s[2])) for s in seq) <= 0.2 + 1e-9,
       'vx ≤ 0.05、wz ≤ 0.2（低速介面界限未動）')

print('D 逾時：底盤減速、手臂設定點立即停止積分')
e2 = mk(CmdChainE2)
for k in range(60):                                # 先跑到高輪速
    t = 1.0 + k * DT
    e2.receive(cmd(0.045, wz=0.18, dq2=0.05), t)
    e2.step(t, DT, [0.0] * 6)
sp_before = list(e2.setpoint)
t0 = 1.0 + 60 * DT
outs = []
for k in range(40):                                # 不再餵新命令 → 過期
    t = t0 + 0.5 + k * DT
    outs.append(e2.step(t, DT, [0.0] * 6))
expect('D1 設定點在逾時後完全不變',
       all(np.allclose(o[1], sp_before) for o in outs),
       f'最大變動 {max(max(abs(a-b) for a, b in zip(o[1], sp_before)) for o in outs):.3e} rad')
base_seq = [np.array(o[0]) for o in outs]
expect('D2 底盤確實減到零', float(np.max(np.abs(W @ base_seq[-1]))) <= 1e-9)
d_acc = max(float(np.max(np.abs(W @ (base_seq[k + 1] - base_seq[k])))) / DT
            for k in range(len(base_seq) - 1))
expect('D3 減速每步在加速度上限內', d_acc <= lim_a + 1e-6,
       f'{d_acc:.4f} ≤ {lim_a:.4f}')
# 到零所需的步數（含到達零的那一步）；E1 在此為 1 步瞬間歸零
n_to_zero = next(k + 1 for k, b in enumerate(base_seq)
                 if float(np.max(np.abs(W @ b))) <= 1e-9)
import math as _m
sp_at_to = float(np.max(np.abs(W @ np.array([0.045, 0.0, 0.18]))))
expect('D4 減速步數與加速度預算相符（E1 為 1 步瞬間歸零）',
       n_to_zero == _m.ceil(sp_at_to / CFG.a_lim(DT)) and n_to_zero > 1,
       f'輪速 {sp_at_to:.4f} / 預算 {CFG.a_lim(DT):.4f} → {n_to_zero} 步')
expect('D5 逾時標記不保留耦合方向',
       e2.last_limit['coupling_preserved'] is False
       and e2.summary()['wheel_limit']['timeout_decel_steps'] > 0)

print('E 摘要欄位')
d = e2.summary()['wheel_limit']
need = {'policy', 'frame', 'w_lim_mps', 'modified_steps',
        'timeout_decel_steps', 'stop_unverified_steps', 'last_limit',
        'note', 'not_claimed'}
expect('E1 摘要欄位齊全', need <= set(d), f'缺 {need - set(d)}')
expect('E2 明記不宣稱上游全身安全性不變', '不宣稱上游全身安全性不變' in d['not_claimed'])
expect('E3 明記座標系為本體', '本體座標' in d['frame'])

print(f"\n{'全部通過' if not fails else '**未通過：' + ', '.join(fails) + '**'}")
sys.exit(1 if fails else 0)
