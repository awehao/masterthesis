#!/usr/bin/env python3
"""受控碰觸邏輯測試（假受控對象；接觸以 TCP 高度觸發，只驗流程與結果碼，不代表物理接觸）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_ref as BR                                               # noqa: E402
import bottle_s0_ctrl as C                                            # noqa: E402
import bottle_s0_probe as P                                           # noqa: E402
import test_bottle_s0a as T                                           # noqa: E402

REQ = ('extra0_table', 'extra1_table', 'arm_table', 'arm_bottle')
TS = BR.grasp_pose(0.20, 0.44, 0.68)
QR, TR = C.ready_from_pre(T.BASE, TS)
BAD, N = [], 0


def chk(name, cond, info=''):
    global N
    N += 1
    if not cond:
        BAD.append(f'{name} {info}')


def run(z_contact=0.637, channel='extra1_table', fault=None, t_end=40.0):
    c = P.ProbeController(T.K, T.ID, TS, TR, QR, T.TOP, T.R_B, REQ)
    q = np.r_[T.BASE, QR]
    t, zs, fp = 0.0, [], []
    while t < t_end:
        z = float(T.K.fk(q, 'link_tcp')[2, 3])
        f = max(0.0, (z_contact - z) * 2000.0)
        forb = {k: 0.0 for k in REQ}
        forb[channel] = f
        ov = dict(forbidden=forb)
        if fault:
            ov.update(fault(t))
        out = c.step(t, q, f_fingers=(0.0, 0.0), t_contact=t, T_bottle=T.bottle_T(), t_bottle=t, f_table=1.962, penetration=0.0, **ov)
        q[3:] = q[3:] + T.ALPHA * (out['arm_target'] - q[3:] + T.D_TRUE)
        zs.append(z); fp.append(f)
        t += 0.01
        if c.stop_ok is not None:
            break
    return c, np.array(zs), np.array(fp)


c, z, f = run()
r = c.report()
chk('P1 預期通道觸發', r['stop']['why'] == 'forbidden_contact:extra1_table', str(r['stop']))
chk('P1 結果碼 0', P.probe_exit_code(r) == 0, str(P.probe_exit_code(r)))
chk('P1 停止有效', (r['stop_effect'] or {}).get('ok') is True, str(r['stop_effect']))
chk('P1 下降速度', 0.0 < r['feed_at_stop_m'] < P.FEED_MAX, str(r['feed_at_stop_m']))
chk('P1 觸發後不再下壓（TCP 下降 ≤ 3 mm）', z.min() > 0.637 - 0.003, f'{z.min():.4f}')
print('P1：進給', round(r['feed_at_stop_m'] * 1e3, 1), 'mm、峰值力', round(float(f.max()), 3), 'N、停止', r['stop_effect'])
c, z, f = run(z_contact=0.0)
r = c.report()
chk('P2 未接觸 ⇒ 進給上限停止', r['stop']['why'] == 'probe_no_contact', str(r['stop']))
chk('P2 結果碼 6', P.probe_exit_code(r) == 6)
chk('P2 進給不超上限', z.min() >= 0.68 - P.FEED_MAX - 0.004, f'{z.min():.4f}')
c, z, f = run(channel='arm_table')
r = c.report()
chk('P3 其他通道先觸發 ⇒ 結果碼 5', r['stop']['why'] == 'forbidden_contact:arm_table' and P.probe_exit_code(r) == 5, str(r['stop']))
c, z, f = run(fault=lambda t: dict(forbidden={k: float('nan') for k in REQ}) if t > 7.0 else {})
r = c.report()
chk('P4 通道資料失效 ⇒ 停止、非 0', (r['stop']['why']).startswith('data_stale_forbidden') and P.probe_exit_code(r) == 5, str(r['stop']))
chk('P5 觸發但停止未驗 ⇒ 7', P.probe_exit_code(dict(stop={'why': 'forbidden_contact:extra1_table', 't': 9.0}, t_descend=8.0, stop_effect=None)) == 7)
c, z, f = run(z_contact=2.0)                                        # 進場途中就有接觸力
r = c.report()
chk('P7 下降前預期通道觸發 ⇒ 非 0、probe_pass False', r['stop']['why'] == 'forbidden_contact:extra1_table' and r['t_descend'] is None and r['probe_exit'] == 5 and r['probe_pass'] is False,
    f"{r['stop']} {r['t_descend']} {r['probe_exit']}")
chk('P8 觸發早於 t_descend ⇒ 非 0', P.probe_exit_code(dict(stop={'why': 'forbidden_contact:extra1_table', 't': 7.0}, t_descend=8.0, stop_effect={'ok': True})) == 5)
c, z, f = run()
r = c.report()
chk('P9 正常 probe_pass True、不用 success 鍵', r['probe_pass'] is True and 'success' not in r, str(r.get('probe_exit')))
chk('P6 無阻點 ⇒ 非 0', P.probe_exit_code(dict(stop=None)) == 6)
print(f'受控碰觸邏輯測試 {N} 項：' + ('全過' if not BAD else '失敗 ' + '；'.join(BAD)))
sys.exit(0 if not BAD else 1)
