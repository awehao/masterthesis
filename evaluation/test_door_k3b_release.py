#!/usr/bin/env python3
"""door_k3b_release 純邏輯反例（規格 K3B_release_clear_spec_draft2.md）。
合成時間序（10 ms 一步）：
- t_open 後手指 0.31 s 開到 95 %、承載 0.04 s 內降到 0；
- 確認後以 0.02 m/s 退離（工具沿 a_H 位移）；
- 配對距離由開度與 s 線性給出（只用於邏輯測試，不是幾何）；
- 非手指 7.96 mm 起、底盤 193 mm、自碰 10.8 mm（30_1 HELD 實測量級）。
  K0 帶：s＝0 +0.1°；0.0075～0.03 +0.5°；0.09 起 +5°；s<0 以 0 計；非有限 ⇒ None。
  K1 正常：A → B（t_open＋0.31＋0.5）→ C_CLEAR（s≥0.02 且配對 ≥5 mm）→ 准許恢復（靜 0.5 s 後）→ 讀回 → C_RESTORED → RETREAT_DONE → STOW_READY。
  K2 確認逾時（手指只開 90 %）⇒ confirm_timeout；K3 連續無效 > 0.2 s ⇒ data_invalid；無效步不計入確認持續窗。
  K4 段 A 配對 > 18 N ⇒ pair_force_design_bound；段 B 配對 > 0.1 N ⇒ pair_force_after_confirm。
  K5 段 B 配對模型距離 < 0 ⇒ pair_penetration_near；s≥0.02 但配對 < 5 mm ⇒ 不退出、近區逾時 near_exit_timeout。
  K6 門角帶：s≈0 時門 +0.15° ⇒ door_band；門速預測（+0.0 角＋0.01 rad/s×0.2 s＝0.115°）⇒ door_band；門 −1.2° ⇒ door_band。
  K7 工具偏移：x 3.5 mm、沿退離軸 y 1.5 mm、z 5 mm（s<0.01）、轉角 0.6° 各 ⇒ tool_offset。
  K8 非手指 4.9 mm／預測 4.9 mm ⇒ nonfinger_clearance；底盤 19 mm ⇒ base_clearance；自碰 4 mm ⇒ self_clearance；錨點 21 mm ⇒ base_anchor。
  K9 5 N：退出前讀回成功 ⇒ effort_restored_early；未請求即讀回 ⇒ effort_restored_without_request；讀回失敗 ⇒ effort_readback；
     靜窗未滿 0.5 s 不准許。
  K10 收臂前置：退到終點時門 +2.6°（帶上界 5° 內）⇒ stow_precondition_door；−3.5° ⇒ door_band（下界 −1° 先觸發）。
  K11 缺步（步號跳 2）視為無效；open 兩次 ⇒ open_twice；FAILED 後閂鎖不變。

    python3 evaluation/test_door_k3b_release.py
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3b_release as R                           # noqa: E402

E_REL = np.eye(4)
E_REL[:3, 3] = (-0.001, -0.0007, 0.0)
FO = R.REG['finger_open_m']
Q_REL = math.radians(30.21)


def sample(k, t, s, fq, pf, door_dq=0.0, door_qd=0.0, dx=0.0, dy=0.0, dz=0.0, rot=0.0, nonf=None, base=0.193, selfd=0.0108,
           base_dev=0.00002, pair=None, eff=None, step=None, age=0.01, inv=0):
    E = E_REL.copy()
    if rot:
        c, sn = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        E[:3, :3] = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]) @ E_REL[:3, :3]
    E[:3, 3] = E_REL[:3, 3] + s * R.A_H + np.array([dx, dy, dz])
    if pair is None:
        pair = 0.002 + 0.5 * s + (fq - FO * 0.95) * 0.3                   # 合成配對距離（只供邏輯）
    nf = 0.00796 + s if nonf is None else nonf
    return {'t': t, 'step': k if step is None else step, 'finger_q': [fq, fq], 'pair_force': [pf, pf], 'contacts_age': age,
            'contacts_invalid': inv, 'door_q': Q_REL + math.radians(door_dq), 'door_qd': door_qd, 'E': E.tolist(), 's_cmd': s,
            'pair_d': [pair, pair], 'pair_d_pred': [pair, pair], 'nonfinger_d': nf, 'nonfinger_d_pred': nf, 'self_d': selfd, 'self_d_pred': selfd,
            'base_d': base, 'base_d_pred': base, 'base_dev_m': base_dev, 'base_dev_yaw': 0.0, 'wrist_n': 3.0, 'unexpected_n': 0.0,
            'effort_readback_ok': eff}


def finger(t_rel, frac_final=1.0):
    """合成開度：0.31 s 內由 20 % 線性到 95 %，之後緩升到 frac_final（K2 實錄量級）。"""
    f = 0.2 + 0.75 * t_rel / 0.31 if t_rel < 0.31 else 0.95 + (t_rel - 0.31) * 0.2
    return FO * min(frac_final, f)


def run(mon, t_end=12.0, mod=None, frac_final=1.0, restore=True):
    mon.open(0.0, 0, Q_REL, E_REL)
    s, eff, k = 0.0, None, 0
    for k in range(1, int(t_end / 0.01) + 1):
        t = round(k * 0.01, 4)
        fq = finger(t, frac_final)
        pf = max(0.0, 12.0 * (1 - t / 0.04)) if t < 0.04 else 0.0
        if mon.state in ('B_CONFIRMED', 'C_CLEAR', 'C_RESTORED', 'RETREAT_DONE'):
            s = min(R.REG['s_end_m'], s + 0.02 * 0.01)
        kw = {}
        if mod:
            kw = mod(t, s, mon) or {}
        if restore and mon.state == 'C_CLEAR' and mon.restore_allowed(t):
            eff = True
        args = dict(fq=fq, pf=pf, eff=eff if mon.state == 'C_CLEAR' else None)
        args.update(kw)                                                     # 情境覆蓋優先
        x = sample(k, t, s, **args)
        mon.step(x)
        if mon.state in ('FAILED', 'STOW_READY'):
            break
    return mon


class TK(unittest.TestCase):
    def test_k0_band(self):
        self.assertEqual(R.band_upper_deg(0.0), 0.1)
        self.assertEqual(R.band_upper_deg(-0.001), 0.1)
        self.assertEqual(R.band_upper_deg(0.0074), 0.4)
        self.assertEqual(R.band_upper_deg(0.0075), 0.5)
        self.assertEqual(R.band_upper_deg(0.0299), 0.5)
        self.assertEqual(R.band_upper_deg(0.03), 0.75)
        self.assertEqual(R.band_upper_deg(0.095), 5.0)
        self.assertIsNone(R.band_upper_deg(float('nan')))

    def test_k1_normal(self):
        m = run(R.ReleaseMonitor())
        self.assertEqual(m.state, 'STOW_READY', m.why)
        tt = m.t
        self.assertAlmostEqual(tt['B_CONFIRMED'], 0.81, places=2)
        self.assertGreater(tt['C_CLEAR'], tt['B_CONFIRMED'])
        self.assertGreaterEqual(m.restore_requested - tt['C_CLEAR'], 0.5 - 1e-9)
        self.assertLess(tt['C_RESTORED'], tt['RETREAT_DONE'])
        self.assertEqual([e['to'] for e in m.events if 'to' in e][:5],
                         ['A_OPENING', 'B_CONFIRMED', 'C_CLEAR', 'restore_requested', 'C_RESTORED'])

    def test_k2_k3_confirm_and_invalid(self):
        m = run(R.ReleaseMonitor(), frac_final=0.90)
        self.assertEqual(m.why, 'confirm_timeout')
        m = run(R.ReleaseMonitor(), mod=lambda t, s, mm: {'age': 0.5} if 0.3 <= t < 0.6 else None)
        self.assertEqual(m.why, 'data_invalid')
        # 0.15 s 無效（< 0.2）只延後確認：持續窗重新起算
        m = run(R.ReleaseMonitor(), mod=lambda t, s, mm: {'inv': 1} if 0.5 <= t < 0.65 else None)
        self.assertEqual(m.state, 'STOW_READY', m.why)
        self.assertAlmostEqual(m.t['B_CONFIRMED'], 1.15, places=2)

    def test_k4_force(self):
        m = R.ReleaseMonitor()
        m.open(0.0, 0, Q_REL, E_REL)
        m.step(sample(1, 0.01, 0.0, FO * 0.2, 18.5))
        self.assertEqual(m.why, 'pair_force_design_bound')
        m = run(R.ReleaseMonitor(), mod=lambda t, s, mm: {'pf': 0.3} if mm.state == 'B_CONFIRMED' else None)
        self.assertEqual(m.why, 'pair_force_after_confirm')

    def test_k5_pair_geometry(self):
        m = run(R.ReleaseMonitor(), mod=lambda t, s, mm: {'pair': -0.0001} if mm.state == 'B_CONFIRMED' else None)
        self.assertEqual(m.why, 'pair_penetration_near')
        m = run(R.ReleaseMonitor(), mod=lambda t, s, mm: {'pair': 0.004} if mm.state == 'B_CONFIRMED' else None)
        self.assertEqual(m.why, 'near_exit_timeout')

    def test_k6_door_band(self):
        m = run(R.ReleaseMonitor(), mod=lambda t, s, mm: {'door_dq': 0.15} if t >= 0.2 else None)
        self.assertEqual(m.why, 'door_band')
        m = run(R.ReleaseMonitor(), mod=lambda t, s, mm: {'door_qd': 0.01} if t >= 0.2 else None)
        self.assertEqual(m.why, 'door_band')
        m = run(R.ReleaseMonitor(), mod=lambda t, s, mm: {'door_dq': -1.2} if t >= 0.2 else None)
        self.assertEqual(m.why, 'door_band')

    def test_k7_tool_offset(self):
        for kw in ({'dx': 0.0035}, {'dy': 0.0015}, {'dz': 0.005}, {'rot': 0.6}):
            m = run(R.ReleaseMonitor(), mod=(lambda kk: (lambda t, s, mm: kk if t >= 0.2 else None))(kw))
            self.assertEqual(m.why, 'tool_offset', kw)

    def test_k8_clearances(self):
        for kw, why in (({'nonf': 0.0049}, 'nonfinger_clearance'), ({'base': 0.019}, 'base_clearance'),
                        ({'selfd': 0.004}, 'self_clearance'), ({'base_dev': 0.021}, 'base_anchor')):
            m = run(R.ReleaseMonitor(), mod=(lambda kk: (lambda t, s, mm: kk if t >= 0.2 else None))(kw))
            self.assertEqual(m.why, why)

    def test_k9_effort(self):
        m = run(R.ReleaseMonitor(), mod=lambda t, s, mm: {'eff': True} if mm.state == 'B_CONFIRMED' else None)
        self.assertEqual(m.why, 'effort_restored_early')
        m = run(R.ReleaseMonitor(), restore=False, mod=lambda t, s, mm: {'eff': True} if mm.state == 'C_CLEAR' else None)
        self.assertEqual(m.why, 'effort_restored_without_request')
        m = run(R.ReleaseMonitor(), mod=lambda t, s, mm: {'eff': False} if mm.state == 'C_CLEAR' else None)
        self.assertEqual(m.why, 'effort_readback')
        m = R.ReleaseMonitor()
        m.state, m.quiet_t0 = 'C_CLEAR', 1.0
        self.assertFalse(m.restore_allowed(1.4))
        self.assertTrue(m.restore_allowed(1.5))

    def test_k10_stow(self):
        m = run(R.ReleaseMonitor(), mod=lambda t, s, mm: {'door_dq': 2.6} if s >= 0.09 else None)
        self.assertEqual(m.why, 'stow_precondition_door')
        m = run(R.ReleaseMonitor(), mod=lambda t, s, mm: {'door_dq': -3.5} if s >= 0.09 else None)
        self.assertEqual(m.why, 'door_band')

    def test_k11_misc(self):
        m = R.ReleaseMonitor()
        m.open(0.0, 0, Q_REL, E_REL)
        m.step(sample(1, 0.01, 0.0, FO * 0.2, 5.0))
        m.step(sample(3, 0.03, 0.0, FO * 0.3, 0.0))                          # 缺步 ⇒ 無效
        self.assertEqual(m.n['invalid'], 1)
        self.assertEqual(m.open(0.1, 5, Q_REL, E_REL), 'FAILED')
        self.assertEqual(m.why, 'open_twice')
        m.step(sample(4, 0.04, 0.0, FO, 0.0))
        self.assertEqual(m.state, 'FAILED')


if __name__ == '__main__':
    unittest.main(verbosity=1)
