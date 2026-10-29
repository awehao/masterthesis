#!/usr/bin/env python3
"""任務端 mode 4 重設計 R1–R3（Codex 20261029_145418／151139／154350；參考包 pb1g）。換抓策略 r8 的 mixin：只改「mode 4 已提交」之後的推進，其他模式原樣。

mode 4 子階段（單向）：
  R1        參考段 4（收臂＋轉向＋依朝向前進）；段末之後沿 R2 起點切線以 V_HAND 外推（交棒前不停車）。
            段末前 mission_lead_s 送導航第二段任務（pb1g m4r.R2 折線＋交回點）；段末且手臂距 STOW ≤ arm_tol ⇒ 請求交給導航。
  R1_HO     等執行端控制者為 nav；逾 ho_timeout_s ⇒ 中止 m4r_nav_handover_timeout（全身續管減速，由執行端窗口規則保證）。
  R2        導航持有；求解端任期封存。nav1 第二段 NAV_TIMEOUT／GLIDE_TIMEOUT ⇒ 中止；HANDOVER_READY（mission 2）⇒ 請求交回全身。逾 r2_timeout_s ⇒ 中止。
  R2_HO     等執行端控制者為 wholebody 且任期 ≥ 1；逾 ho_timeout_s ⇒ 中止 m4r_wb_handover_timeout。
  R3        段 41 自交回當下起算：起點偏差 Δ0 依 bridge_r3 准入，參考＝x_ref(τ)＋(1−w(τ/T_b))Δ0（同 RefRuntime3）；
            段末 ⇒ 既有 Gate5（CONFIRMED 且新鮮）＋_extra_ready ⇒ 請求 mode 5；gate_wait_s 逾時照舊中止。
前饋：段 4／41 用參考包的解析底盤速度 base_v（與取樣同一條軌跡）＋銜接項的解析導數，不以差分近似。
監看：每輪先跑既有 RG 監看（狀態新鮮、權威中止、gate 失敗、接觸、門、牆／腕、求解端、幾何）—— 導航段也不停。
輸出：take_m4r_outbox() ⇒ [('mission', dict)｜('handover', dict)]，由節點發 /nav1/mission 與 /handover/request。
量測（節點每輪設）：m.ho_owner、m.ho_tenure（/handover/tenure 的 chain_tenure、chain_tenure_open）、m.nav1（/nav1/state）。
"""
from __future__ import annotations

import math

import numpy as np

import door_regrasp_policy1 as P1
import door_regrasp_policy3 as P3
import door_regrasp_policy8 as P8

VERSION = 'door_regrasp_m4r_policy/1'
SEG_R3 = 41


class RefRuntimeM4R(P8.RefRuntime8):
    def start(self, k, base, arm, q_door_deg, T_tcp):
        if k != SEG_R3:
            return super().start(k, base, arm, q_door_deg, T_tcp)
        sg = self.seg[k]
        br = self.p['m4r']['bridge_r3']
        b0, a0 = np.asarray(sg['base'][0], float), np.asarray(sg['arm'][0], float)
        b, a = np.asarray(base, float), np.asarray(arm, float)
        det = {'k': k, 'base_dev_m': float(np.linalg.norm(b[:2] - b0[:2])), 'yaw_dev_deg': math.degrees(abs(P1.wrap(b[2] - b0[2]))),
               'arm_dev_rad': float(np.max(np.abs(a - a0))), 'door_dev_deg': float(q_door_deg) - float(self.p['q_ref_deg'])}
        if abs(det['door_dev_deg']) > br['door_deg'] + 1e-12:
            return False, 'bridge_door', det
        if det['base_dev_m'] > br['base_m'] or det['yaw_dev_deg'] > br['base_deg']:
            return False, 'bridge_base', det
        if det['arm_dev_rad'] > br['arm_rad']:
            return False, 'bridge_arm', det
        self.ctx[k] = {'d0': np.zeros(9)}
        d0 = np.r_[b, a] - self.cfg(k, 0)
        d0[2] = P1.wrap(d0[2])
        self.ctx[k]['d0'] = d0
        return True, None, dict(det, d0=[float(v) for v in d0])

    def cfg(self, k, i):
        if k != SEG_R3:
            return super().cfg(k, i)
        sg = self.seg[k]
        i = min(max(i, 0), len(sg['t']) - 1)
        x = self._xref(k, i)
        return x + (1.0 - P3.smooth(float(sg['t'][i]) / self.T_b)) * self.ctx[k]['d0']

    def at(self, k, tau):
        if k not in (4, SEG_R3) or 'base_v' not in self.seg[k]:
            return super().at(k, tau)
        sg = self.seg[k]
        n = len(sg['t'])
        i = min(int(round(max(0.0, tau) / self.dt)), n - 1)
        xi = self.cfg(k, i)
        ff = np.asarray(sg['base_v'][i], float).copy()
        u = float(sg['t'][i]) / self.T_b
        if 0.0 < u < 1.0:                                   # 銜接項 (1−w(u))Δ0 的解析導數：−w'(u)/T_b·Δ0
            ff = ff - (6 * u - 6 * u * u) / self.T_b * np.asarray(self.ctx[k]['d0'][:3], float)
        return np.asarray(self.K.fk(xi, self.tcp), float), xi[:3].copy(), ff


REG_M4R = {'mission_lead_s': 3.0, 'arm_tol_rad': 0.05, 'ho_window_steps': 300, 'ho_timeout_s': 5.0, 'r2_timeout_s': 150.0}


def m4r(Base):
    class M4R(Base):
        def __init__(self, *a, ref_pkg, m4r_reg=None, **kw):
            if 'm4r' not in ref_pkg or str(SEG_R3) not in ref_pkg.get('segments', {}):
                raise ValueError('m4r 需 pb1g 參考包（m4r、段 41）')
            super().__init__(*a, ref_pkg=ref_pkg, **kw)
            self.ref = RefRuntimeM4R(self.ref.p, self.ref.K, T_b=self.ref.T_b)
            self.m4cfg = dict(REG_M4R, **(m4r_reg or {}))
            self.m4 = {'sub': None, 't': {}, 'events': [], 'mission_sent': False, 'bridge_r3': None, 't_ref3': None, 'last_T': None}
            self.m4_outbox = []

        def take_m4r_outbox(self):
            out, self.m4_outbox = ([], []) if self.phase == 'ABORTED' else (self.m4_outbox, [])
            return out

        def _m4go(self, sub, m, **kw):
            self.m4['sub'] = sub
            self.m4['t'][sub] = m.t
            self.m4['events'].append(dict({'t': m.t, 'sub': sub}, **kw))

        def _rg_step(self, m, dt):
            k = self.rg['k_target'] if 'k_target' in self.rg else 2
            if not (k == 4 and self.rg.get('k_committed', 1) == 4):
                return super()._rg_step(m, dt)
            act = self._act()
            self.rg['k'] = k
            st = self.w2_last_state
            if self.w2_last_state_t is None or m.t - self.w2_last_state_t > self.w2_state_stale_s:
                return self._rg_abort('rg_state_stale', m, act)
            if isinstance(st, dict) and st.get('aborted'):
                return self._rg_abort(f"rg_authority_aborted:{(st.get('aborted') or {}).get('why')}", m, act)
            gates = (st or {}).get('gates') or {}
            for g, gs in gates.items():
                if isinstance(gs, dict) and gs.get('state') == 'FAILED':
                    return self._rg_abort(f"rg_exec_{g}_failed:{gs.get('why')}", m, act)
            res = self._monitor_rg(m, act)
            if res is not None:
                return res
            c = self.m4cfg
            mp = self.ref.p['m4r']
            if self.m4['sub'] is None:
                self._m4go('R1', m)
            sub = self.m4['sub']
            owner = getattr(m, 'ho_owner', None)
            nav1 = getattr(m, 'nav1', None) or {}
            ten = getattr(m, 'ho_tenure', None) or {}
            if sub in ('R1', 'R1_HO'):
                tau = m.t - self.rg['t_ref0'][4]
                L = self.ref.length_s(4)
                T, b_ref, ff = self.ref.at(4, min(tau, L))
                if tau > L:                                        # 段末外推：沿 R2 起點切線 V_HAND
                    psi = float(mp['handover_nav']['yaw'])
                    v = float(mp['V_HAND'])
                    d = v * (tau - L)
                    T = np.asarray(T, float).copy()
                    T[0, 3] += d * math.cos(psi)
                    T[1, 3] += d * math.sin(psi)
                    b_ref = np.asarray(b_ref, float) + np.array([d * math.cos(psi), d * math.sin(psi), 0.0])
                    ff = np.array([v * math.cos(psi), v * math.sin(psi), 0.0])
                act['target'] = T
                act['coord'] = self._coord(m, b_ref, ff)
                if not self.m4['mission_sent'] and tau >= L - c['mission_lead_s']:
                    self.m4['mission_sent'] = True
                    self.m4_outbox.append(('mission', {'id': 2, 'vertices': mp['R2']['vertices'], 'goal': mp['R2']['goal']}))
                    self.m4['events'].append({'t': m.t, 'mission_sent': True})
                if sub == 'R1' and tau >= L:
                    arm = getattr(m, 'qf', None)
                    stow = np.asarray(self.ref.p['stow'], float)
                    dev = None if arm is None else float(np.max(np.abs(np.asarray(list(arm)[3:9], float) - stow)))
                    if dev is not None and dev <= c['arm_tol_rad']:
                        self.m4_outbox.append(('handover', {'to': 'nav', 'window_steps': int(c['ho_window_steps'])}))
                        self._m4go('R1_HO', m, arm_dev=dev)
                if self.m4['sub'] == 'R1_HO':
                    if owner == 'nav':
                        self._m4go('R2', m)
                    elif m.t - self.m4['t']['R1_HO'] > c['ho_timeout_s']:
                        return self._rg_abort('m4r_nav_handover_timeout', m, act, owner=owner)
                self.m4['last_T'] = np.asarray(act['target'], float)
            elif sub in ('R2', 'R2_HO'):
                act['target'] = self.m4['last_T']                  # 求解端封存；目標保持（不推進）
                seg = nav1.get('seg')
                if nav1.get('mission') == 2 and seg in ('NAV_TIMEOUT', 'GLIDE_TIMEOUT'):
                    return self._rg_abort(f'm4r_nav_{seg.lower()}', m, act, nav1=nav1)
                if sub == 'R2':
                    if m.t - self.m4['t']['R2'] > c['r2_timeout_s']:
                        return self._rg_abort('m4r_r2_timeout', m, act)
                    if nav1.get('mission') == 2 and seg == 'HANDOVER_READY':
                        self.m4_outbox.append(('handover', {'to': 'wholebody', 'window_steps': int(c['ho_window_steps'])}))
                        self._m4go('R2_HO', m, nav1=nav1)
                elif owner == 'wholebody' and int(ten.get('chain_tenure', 0) or 0) >= 1 and ten.get('chain_tenure_open') is True:
                    bp, arm = getattr(m, 'base_pose', None), getattr(m, 'qf', None)
                    ok, why, det = self.ref.start(SEG_R3, bp, None if arm is None else list(arm)[3:9], math.degrees(m.q_door), m.T_tcp)
                    self.m4['bridge_r3'] = det
                    if not ok:
                        return self._rg_abort(f'm4r_r3_{why}', m, act, detail=det)
                    self.m4['t_ref3'] = m.t
                    self._m4go('R3', m, bridge=det)
                elif m.t - self.m4['t']['R2_HO'] > c['ho_timeout_s']:
                    return self._rg_abort('m4r_wb_handover_timeout', m, act, owner=owner, tenure=ten)
            if self.m4['sub'] == 'R3':
                tau3 = m.t - self.m4['t_ref3']
                T, b_ref, ff = self.ref.at(SEG_R3, tau3)
                act['target'] = T
                act['coord'] = self._coord(m, b_ref, ff)
                self.m4['last_T'] = np.asarray(T, float)
            self.rg['last_T'] = np.asarray(act['target'], float)
            act['geom_target'] = self.rg['geom_target']
            gs = gates.get(self.GATE_FOR[4]) if isinstance(gates.get(self.GATE_FOR[4]), dict) else {}
            act['rg'] = {'k': 4, 'state': 'running', 'sub': self.m4['sub'], 'gate': gs.get('state')}
            if self.m4['sub'] == 'R3' and m.t - self.m4['t_ref3'] >= self.ref.length_s(SEG_R3) - 1e-9:
                xr, xwhy = self._extra_ready(4, m)
                if gs.get('state') == 'CONFIRMED' and gs.get('fresh') is True and xr:
                    self.rg['k_target'] = 5
                    self._request(5, m)
                    self.rg['gate_wait_t0'] = None
                    self.m4['events'].append({'t': m.t, 'request_5': True})
                    return act
                self.rg['gate_wait_t0'] = self.rg['gate_wait_t0'] if self.rg['gate_wait_t0'] is not None else m.t
                if m.t - self.rg['gate_wait_t0'] > self.r['gate_wait_s']:
                    return self._rg_abort('rg4_gate_timeout', m, act, gate=gs)
            return act

        def rg_report(self):
            r = super().rg_report()
            r['m4r'] = {'version': VERSION, 'cfg': dict(self.m4cfg), 'sub': self.m4['sub'], 't': dict(self.m4['t']),
                        'events': list(self.m4['events'])[:200], 'bridge_r3': self.m4['bridge_r3']}
            return r
    M4R.__name__ = f'M4R_{Base.__name__}'
    return M4R
