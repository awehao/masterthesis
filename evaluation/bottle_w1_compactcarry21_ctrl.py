#!/usr/bin/env python3
"""緊湊持物 v2.1（Codex 20261101_204716 第 2 項；bottle_w1_compactcarry2_ctrl 的版本化子類，v2 原樣保留）。過渡、姿態、停車與放置同 v2，只改第一桌淨空閘：
1 補實際停止策略的軌跡：以目前實測狀態、手臂設定點 s、在途命令管線與最後命令，照停止後的實際規則逐物理步積分——
  每控制拍命令朝零有界減速（底盤 A_BASE、手臂 A_ARM，同 _arm_decel），管線內已發布命令照常生效，受控對象用 plant_phys_step（同延遲補償模型），
  直到命令歸零、管線排空、手臂一階收斂（再 60 物理步）；每 4 物理步取一個狀態，用 FK 算瓶身＋夾爪點（世界座標），套同一判準（低於桌面頂＋Z_PAD 的點須在桌面投影外擴 EDGE_PAD 之外）。
  兩條停止軌跡：現在就停；再照最後命令走一個控制拍才停（閘在拍與拍之間不再核對）。每條取底盤停止路徑的取樣位姿 × 手臂停止路徑的取樣構型交叉核對，
  不依時間對齊（Shield 提早煞停時底盤停在路徑較前段，組合仍在集合內）。
2 前方距離速度改用已強制的上界：Shield 對命令各軸限幅 vx_max＝vy_max＝0.2775 m/s（run_bottle_ros.sh 的 SH_P），合成 0.3924 m/s；v2 的 0.30 m/s 不是強制上界。
實際底盤命令還經過 Shield；Shield 介入（例如掃描過期）時煞得比 A_BASE 快，實際停止較短——模型軌跡在行進方向上偏保守，但不涵蓋 Shield 改變方向的投影。
口徑：這是取樣幾何檢查——取樣位姿 × 取樣路徑點，加上模型積分的兩條停止軌跡；只涵蓋列入的組合，不是連續安全保證，也不涵蓋模型外的底盤動態或未建模的障礙。
"""
import math
import time

import numpy as np

from ammr_wholebody_mpc import wgmpc_core_sp as SP

import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_w1_compactcarry2_ctrl as C2

V_CMD_MAX = math.hypot(0.2775, 0.2775)
SETTLE_STEPS = 60
SAMPLE_EVERY = 4                                        # 4 物理步取一個狀態（底盤 0.39 m/s 時相鄰約 16 mm，遠小於 EDGE_PAD 0.05 m）


def stop_traj(q, s, pipe, u_last, cfg, extra_ticks=0, phase=0):
    """停止策略的模型軌跡（含起點）：回傳每 SAMPLE_EVERY 物理步的 q。extra_ticks＝先照最後命令走幾個控制拍才開始減速；
    phase＝起點距上一個控制拍的物理步數（閘在控制拍上呼叫 ⇒ 0）。"""
    qp, sp = np.asarray(q, float).copy(), np.asarray(s, float).copy()
    pp = [np.asarray(u, float).copy() for u in pipe]
    u = np.asarray(u_last, float).copy()
    da = np.r_[B.A_BASE, np.full(6, C0.A_ARM)] * cfg.dt
    out, k, quiet = [qp.copy()], 0, 0
    while quiet < SETTLE_STEPS and k < 2000:
        if (k + phase) % C0.NPH == 0 and k >= extra_ticks * C0.NPH:
            u = u + np.clip(-u, -da, da)
        pp.append(u.copy())
        qp, sp = SP.plant_phys_step(qp, sp, pp.pop(0), cfg, 1)
        k += 1
        quiet = quiet + 1 if (not np.any(u) and not any(np.any(x) for x in pp)) else 0
        if k % SAMPLE_EVERY == 0:
            out.append(qp.copy())
    return np.array(out)


class W1PlaceController(C2.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT21'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gate_stop_min, self.gate_stop_n, self.gate_stop_ms = math.inf, 0, []

    def _poses(self, q, t_rem):
        q = np.asarray(q, float)
        P = list(super()._poses(q, 0.0))                     # 實測、在途、預測時域（路徑前方另算，速度改用強制上界）
        if self.cpath is not None and self.csig is not None:
            off = q[:2] - self.cpath.xy(self.csig)[0]
            L = V_CMD_MAX * max(float(t_rem), 0.0)
            sig = self.csig + np.linspace(0.0, L, max(2, int(L / 0.02) + 1))
            pa = self.cpath.xy(sig) + off
            P.extend(np.column_stack([pa, np.full(len(pa), q[2])]))
        return np.array(P, float)

    def _stop_margin(self, q):
        t0 = time.perf_counter()
        try:
            return self._stop_margin_inner(q)
        finally:
            self.gate_stop_ms.append((time.perf_counter() - t0) * 1e3)

    def _stop_margin_inner(self, q):
        """停止軌跡交叉核對：底盤停止路徑上的取樣位姿 × 手臂停止路徑上的取樣構型（不依時間對齊）。
        Shield 介入時底盤煞得較快、停在模型底盤路徑較前段，手臂仍走自己的收斂路徑——這種組合仍在交叉集合內；Shield 改變方向的投影不在內。"""
        Pt = self._pts_tcp()
        m = math.inf
        for extra in (0, 1):
            Q = stop_traj(q, self.s, self.pipe, self.u_last, self.cfg, extra)
            if not np.isfinite(Q).all():
                return None
            self.gate_stop_n = max(self.gate_stop_n, len(Q))
            PB = []
            for qk in Q:
                Tb_ = self.K.fk(np.r_[0.0, 0.0, 0.0, qk[3:]], 'link_tcp')
                PB.append((Tb_[:3, :3] @ Pt.T).T + Tb_[:3, 3])
            mk = C2.clear_margin(np.vstack(PB), Q[:, :3])
            if mk is None:
                return None
            m = min(m, mk)
        return m

    def _gate(self, q, t_rem):
        m, n = super()._gate(q, t_rem)
        if m is None:
            return None, n
        ms = self._stop_margin(q)
        if ms is None:
            return None, n
        if math.isfinite(ms):
            self.gate_stop_min = min(self.gate_stop_min, ms)
        return min(m, ms), n

    def report(self):
        r = super().report()
        g = r['compact_carry']['gate']
        sm = np.array(self.gate_stop_ms) if self.gate_stop_ms else np.zeros(1)
        g.update(v_ahead=V_CMD_MAX, stop_traj=dict(min_margin=None if not math.isfinite(self.gate_stop_min) else self.gate_stop_min, max_states=self.gate_stop_n,
                                                   ms_max=float(sm.max()), ms_mean=float(sm.mean()),
                                                   rule='停止後每控制拍有界減速（A_BASE／A_ARM）、管線照常生效、plant_phys_step；現在停與再走一拍才停兩條'),
                 scope='取樣幾何檢查（取樣位姿 × 路徑點＋模型積分的兩條停止軌跡），不是連續安全保證')
        r['compact_carry']['version'] = 'v2.1'
        return r
