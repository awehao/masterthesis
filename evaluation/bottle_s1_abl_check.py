#!/usr/bin/env python3
"""單一消融最小驗收：拿掉 TCP 未來參考。開關關閉重現凍結 G；開啟時各列目標相同；名目離線各一次。假受控對象，非物理。
    python3 evaluation/bottle_s1_abl_check.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_groups_check as CK                     # noqa: E402
import bottle_ref as BR                                 # noqa: E402
import test_bottle_s0b as T                             # noqa: E402


def run(flag):
    c = SC.make_lidar(no_tcp_preview=flag)
    errs = []
    orig = c._T_seq

    def wrap(q, t_eff):
        Ts = orig(q, t_eff)
        if flag and c.n_tick == 400:
            assert np.abs(Ts - Ts[0]).max() == 0.0, '消融未生效'
        return Ts
    c._T_seq = wrap
    f = CK.fake(c, 45.0)
    r = CK.summary('消融 no_tcp_preview' if flag else 'G 開關關閉', c, f)
    q = np.array([h['q'] for h in f.hist])
    t = np.array([h['t'] for h in f.hist])
    print('  時域內 TCP 目標變化峰 mm（原序列）', r['preview_range_mm'])
    tc, tl = r['t_close'], r['t_lift_start']
    if tc and tl:
        mk = (t >= tc) & (t <= tl)
        e = np.array([np.linalg.norm(T.K.fk(h['q'], 'link_tcp')[:3, 3] - T.T_G[:3, 3]) for h, m in zip(f.hist, mk) if m])
        print(f'  閉爪至抬升 TCP 對抓取位 max {e.max() * 1e3:.2f} mm')
    return r, f


def main():
    r0, f0 = run(False)
    r1, f1 = run(True)
    print('關閉時事件同既有 G', [e.get('to') for e in r0['events'] if e['what'] == 'phase'] == ['PREGRASP', 'INSERT', 'CLOSE', 'LIFT', 'CARRY'])


if __name__ == '__main__':
    sys.exit(main())
