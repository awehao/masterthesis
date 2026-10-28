#!/usr/bin/env python3
"""換抓＋關門任務端策略 r8（door_regrasp_policy7 的子類；Codex r67）：模式表換 door_regrasp_mode3（到 8）。
  mode 6 CLOSE_INNER 完成 ⇒ 請求 7 PASS_CLOSE（看執行端 gate7＝HoldConfirm 確認且新鮮）；mode 7 參考走完 ⇒ 請求 8 RELEASE_RETREAT（看 gate8）；
  mode 8 參考走完且 exit8 確認且新鮮 ⇒ DONE（不再於 mode 6 DONE）。
  門角監看（取代換抓期間「對解除值 ≤ 0.7°」，不是關掉）：mode 7 ⇒ |q_meas − q_ref7(τ)| ≤ DOOR_TRACK_DEG（2.0°，候選）；mode 8 ⇒ 對 mode 8 提交當下門角 ≤ 0.7°；
    門速上限（door_abort_rate）不變。接觸規則、幾何監看（目標 in）、wall／wrist、求解端、執行端違規等其餘監看照 r7。
  參考執行期 RefRuntime8：mode 7／8 起點銜接改對各自段首門角（不是 q_ref 47.89°），底盤 0.02 m／2°、手臂 0.05 rad 同 mode 3／4；無構型銜接（段首＝前段末）。
"""
from __future__ import annotations

import math

import numpy as np

import door_regrasp_mode3 as RG3
import door_regrasp_policy1 as P1
import door_regrasp_policy3 as P3
import door_regrasp_policy7 as P7

VERSION = 'door_regrasp_policy8/r8'
Meas = P7.Meas
DOOR_TRACK_DEG = 2.0


class RefRuntime8(P3.RefRuntime3):
    def start(self, k, base, arm, q_door_deg, T_tcp):
        if k < 7:
            return super().start(k, base, arm, q_door_deg, T_tcp)
        sg = self.seg[k]
        b0, a0 = np.asarray(sg['base'][0], float), np.asarray(sg['arm'][0], float)
        b, a = np.asarray(base, float), np.asarray(arm, float)
        dq = float(q_door_deg) - float(sg['q_door_deg'][0])
        det = {'k': k, 'base_dev_m': float(np.linalg.norm(b[:2] - b0[:2])), 'yaw_dev_deg': math.degrees(abs(P1.wrap(b[2] - b0[2]))),
               'arm_dev_rad': float(np.max(np.abs(a - a0))), 'door_dev_deg': dq}
        if abs(dq) > self.br['door_deg'] + 1e-12:
            return False, 'bridge_door', det
        if det['base_dev_m'] > self.br['base_m'] or det['yaw_dev_deg'] > self.br['base_deg']:
            return False, 'bridge_base', det
        if det['arm_dev_rad'] > self.br['arm_rad']:
            return False, 'bridge_arm', det
        self.ctx[k] = {}
        return True, None, det


class RegraspPolicy8(P7.RegraspPolicy7):
    RGT = RG3
    ALLOWED_FN = staticmethod(RG3.allowed_finger_classes)
    GEOM_FOR = {2: 'out', 3: 'out', 4: None, 5: None, 6: 'in', 7: 'in', 8: 'in'}
    GATE_FOR = {2: 'gate3', 3: 'gate4', 4: 'gate5', 5: 'gate6', 6: 'gate7', 7: 'gate8', 8: 'exit8'}
    FINAL_MODE = 8
    HOLD_REF = (2, 6)

    def __init__(self, *a, ref_pkg, **kw):
        if '7' not in ref_pkg.get('segments', {}) or '8' not in ref_pkg.get('segments', {}):
            raise ValueError('policy8 需含 mode 7／8 段的參考包（ref v5 起）')
        super().__init__(*a, ref_pkg=ref_pkg, **kw)
        self.ref = RefRuntime8(self.ref.p, self.ref.K, T_b=self.ref.T_b)

    def _monitor_rg(self, m, act):
        kc = self.rg.get('k_committed', 1)
        if kc in (7, 8) and self.rg.get('t_ref0', {}).get(kc) is not None:
            dev0 = self.r['door_dev_deg']
            if kc == 7:
                sg = self.ref.seg[7]
                i = min(int(round(max(0.0, m.t - self.rg['t_ref0'][7]) / self.ref.dt)), len(sg['t']) - 1)
                self.rg['q_rel_deg'], self.r['door_dev_deg'] = float(sg['q_door_deg'][i]), DOOR_TRACK_DEG
            else:
                self.rg.setdefault('q_rel8_deg', math.degrees(m.q_door) if m.q_door is not None else None)
                self.rg['q_rel_deg'] = self.rg['q_rel8_deg']
            try:
                return super()._monitor_rg(m, act)
            finally:
                self.r['door_dev_deg'] = dev0
        return super()._monitor_rg(m, act)

    def rg_report(self):
        r = super().rg_report()
        r['version'] = VERSION
        r['door_track_deg'] = DOOR_TRACK_DEG
        return r
