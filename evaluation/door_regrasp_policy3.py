#!/usr/bin/env python3
"""換抓任務端策略 r3（door_regrasp_policy2 的子類；Codex r53 §2–§5）。
§3／§2 接觸：任務端不再用自己的 k_committed 判定接觸規則，改讀**執行端逐物理步摘要**（door_regrasp_contact_window：依該步權威模式分組、
  帶 Gate4 離開閂鎖、第一筆違規）——摘要的 violation 非空 ⇒ 中止 rg_exec_contact_violation；另以同一規則函式 allowed_finger_classes 對 by_mode 複核（一致性）。
  摘要必要欄（src_step、src_t、complete、by_mode、exited3、violation）缺 ⇒ 中止 rg_contact_v3_malformed；年齡＝m.t − src_t（來源模擬時刻，非接收時刻）> stale_s ⇒ 中止。
§4 幾何：進入換抓前須有 v3 幾何後端與 GeometryMonitor，缺 ⇒ 中止 rg_geometry_backend_missing（不先動、不等 gate 逾時）；運行中每輪必算，stop（含輸入無效、過期）即中止。
  配對預測距離改用 v＝max(實測接近率, v_floor)（實測接近率＝前一輪 finger_bar_out 距離減本輪除以時間差，取正），不把 floor 當上界。
§5 參考銜接（RefRuntime3）：mode 4、5 在**構型空間**連續消除起始偏差：x_cmd(τ)＝x_ref'(τ)＋(1 − w(τ/T_b))·Δ0，w(u)＝3u²−2u³（u 夾在 [0,1]），
  Δ0＝起始實測構型 − 參考起點（偏航取最短角）；x_ref' 在 mode 5 為參考構型繞鉸鏈剛體轉 w5·Δq（A 段 w5 0→1、I 段 1）。TCP 目標＝FK(x_cmd)、底盤參考＝x_cmd 底盤、
  前饋＝同一條 x_cmd 的差分（含轉角權重變化與銜接項；不再只旋轉原 ff）。T_b 預設 4.0 s（候選；離線 door_regrasp_bridge_check_v1 以登錄範圍角點核速度、加速度與淨空）。
  mode 5 起點亦核銜接範圍。mode 6 目標＝mode 5 末的 FK(x_cmd)（保持）。
  說明（r53 §5 措辭）：門角修正在**模式開始時取一次**（固定參考），不是每輪跟隨實測把手；閉爪准入（執行端 Gate6）才以當步實測門角的把手位姿判定。
  門角只驗 −1／0／+1° 三個取樣，不稱整個連續區間已驗證。
"""
from __future__ import annotations

import math

import numpy as np

import door_regrasp_mode1 as RGM
import door_regrasp_policy1 as P1
import door_regrasp_policy2 as P2

VERSION = 'door_regrasp_policy3/r3'
Meas = P1.Meas
REG = dict(P1.REG, bridge_T_s=4.0)
FINGERS = P1.FINGERS


def smooth(u):
    u = min(1.0, max(0.0, u))
    return 3 * u * u - 2 * u * u * u


class RefRuntime3(P1.RefRuntime):
    def __init__(self, pkg, kin, T_b=4.0, tcp='link_tcp'):
        super().__init__(pkg)
        self.K, self.T_b, self.tcp = kin, float(T_b), tcp

    def _xref(self, k, i):
        sg = self.seg[k]
        return np.r_[np.asarray(sg['base'][i], float), np.asarray(sg['arm'][i], float)]

    def cfg(self, k, i):
        sg = self.seg[k]
        n = len(sg['t'])
        i = min(max(i, 0), n - 1)
        x = self._xref(k, i)
        c = self.ctx.get(k, {})
        if k == 5:
            a = -self._w5(i) * c['dq_rad']
            ca, sa = math.cos(a), math.sin(a)
            x[0], x[1] = ca * x[0] - sa * x[1], sa * x[0] + ca * x[1]
            x[2] = x[2] + a
        if k in (4, 5):
            x = x + (1.0 - smooth(float(sg['t'][i]) / self.T_b)) * c['d0']
        return x

    def start(self, k, base, arm, q_door_deg, T_tcp):
        ok, why, det = super().start(k, base, arm, q_door_deg, T_tcp)
        if not ok:
            return ok, why, det
        if k in (4, 5):
            x_act = np.r_[np.asarray(base, float), np.asarray(arm, float)]
            if k == 5:
                if det['base_dev_m'] > self.br['base_m'] or det['yaw_dev_deg'] > self.br['base_deg']:
                    return False, 'bridge_base', det
                if det['arm_dev_rad'] > self.br['arm_rad']:
                    return False, 'bridge_arm', det
            self.ctx.setdefault(k, {})
            self.ctx[k]['d0'] = np.zeros(9)                         # cfg 先以零偏差算參考起點
            x0 = self.cfg(k, 0)
            d0 = x_act - x0
            d0[2] = P1.wrap(d0[2])
            self.ctx[k]['d0'] = d0
            det = dict(det, d0=[float(v) for v in d0])
        if k == 6:
            self.ctx[6] = {'T': np.asarray(self.K.fk(self.cfg(5, len(self.seg[5]['t']) - 1), self.tcp), float)}
        return True, None, det

    def at(self, k, tau):
        if k == 3:
            return super().at(k, tau)
        if k == 6:
            return self.ctx[6]['T'], None, np.zeros(3)
        n = len(self.seg[k]['t'])
        i = min(int(round(max(0.0, tau) / self.dt)), n - 1)
        xi = self.cfg(k, i)
        if i >= n - 1:
            ff = np.zeros(3)
        else:
            xj = self.cfg(k, i + 1)
            ff = np.array([xj[0] - xi[0], xj[1] - xi[1], P1.wrap(xj[2] - xi[2])]) / self.dt
        return np.asarray(self.K.fk(xi, self.tcp), float), xi[:3].copy(), ff


class RegraspPolicy3(P2.RegraspPolicy2):
    ALLOWED_FN = staticmethod(RGM.allowed_finger_classes)   # r3 表驅動（關門版換 mode3 規則）
    def __init__(self, *a, kin=None, **kw):
        super().__init__(*a, **kw)
        self.r = dict(REG, **{k: v for k, v in self.r.items() if k in P1.REG})
        if kin is None:
            raise ValueError('policy3 需運動學 kin（RefRuntime3 的 FK）')
        self.ref = RefRuntime3(self.ref.p, kin, T_b=self.r['bridge_T_s'])
        self.pair_prev = None

    def _enter_rg(self, m, dt):
        if self.geom_v3_fn is None or getattr(self, 'geom', None) is None:
            return self._rg_abort('rg_geometry_backend_missing', m, self._act())
        return super()._enter_rg(m, dt)

    def _monitor_rg(self, m, act):
        r = self.r
        if not all(v is not None and math.isfinite(v) for v in (m.state_age, m.contacts_age)) or m.state_age > r['stale_s'] \
                or m.contacts_age > r['stale_s']:
            return self._rg_abort('rg_monitor_stale', m, act)
        c = self.c3
        need = ('src_step', 'src_t', 'complete', 'by_mode', 'exited3', 'violation')
        if not isinstance(c, dict) or any(k not in c for k in need) or not isinstance(c.get('by_mode'), dict) \
                or not isinstance(c.get('src_t'), (int, float)) or not math.isfinite(c['src_t']):
            return self._rg_abort('rg_contact_v3_malformed', m, act)
        if m.t - float(c['src_t']) > r['stale_s'] or c['complete'] is not True:
            return self._rg_abort('rg_contact_v3_stale_or_incomplete', m, act, src_t=c.get('src_t'))
        if c['violation'] is not None:
            return self._rg_abort('rg_exec_contact_violation', m, act, detail=c['violation'])
        for mode_s, links in c['by_mode'].items():
            try:
                mode = int(mode_s)
            except (TypeError, ValueError):
                return self._rg_abort('rg_contact_v3_malformed', m, act)
            ex = (c.get('exited_by_mode') or {}).get(str(mode), c['exited3'] if mode == 3 else False)
            allowed = self.ALLOWED_FN(mode, bool(ex))
            for link, cl in (links or {}).items():
                for cls, f in (cl or {}).items():
                    if not (isinstance(f, (int, float)) and math.isfinite(f)):
                        return self._rg_abort('rg_contact_v3_nonfinite', m, act)
                    if f > r['contact_abort_n'] and (link not in FINGERS or cls not in allowed):
                        return self._rg_abort('rg_contact_not_designated', m, act, mode=mode, link=link, cls=cls, n=f)
        if m.q_door is None or m.qdot_door is None or not math.isfinite(m.q_door) or not math.isfinite(m.qdot_door):
            return self._rg_abort('rg_door_nonfinite', m, act)
        if abs(math.degrees(m.q_door) - self.rg['q_rel_deg']) > r['door_dev_deg'] or abs(m.qdot_door) > r['door_abort_rate']:
            return self._rg_abort('rg_door_moved', m, act, q_deg=math.degrees(m.q_door), qdot=m.qdot_door)
        if m.wall_max is None or not math.isfinite(m.wall_max) or m.wall_max > r['contact_abort_n']:
            return self._rg_abort('rg_wall_contact', m, act, n=m.wall_max)
        if m.wrist_max is None or not math.isfinite(m.wrist_max) or m.wrist_max >= r['wrist_abort_n']:
            return self._rg_abort('rg_wrist_force', m, act, n=m.wrist_max)
        if m.solver_failed:
            return self._rg_abort('rg_solver_failed', m, act, detail=m.solver_failed)
        if m.solver_up and (m.solver_status_age is None or m.solver_status_age > self.c.solver_stale_s):
            return self._rg_abort('rg_solver_status_stale', m, act)
        if getattr(m, 'executor_violation', None) is not None:
            return self._rg_abort('rg_executor_violation', m, act, detail=m.executor_violation)
        if self.geom_v3_fn is None or getattr(self, 'geom', None) is None:
            return self._rg_abort('rg_geometry_backend_missing', m, act)
        dec, det = self._geom_step(m)
        self.geom_last, self.geom_t = (dec, det), m.t
        if dec == 'stop':
            return self._rg_abort(f"rg_geometry:{det.get('why')}", m, act, detail=det)
        return None

    def _geom_step(self, m):
        """幾何判定掛點（子類可換成切換包裝）；本類行為＝設目標後直接呼叫 GeometryMonitor。"""
        self.geom_v3_fn.set_target(self.rg['geom_target'])
        return self.geom.step(m.t, getattr(m, 'qf', None), m.q_door, getattr(m, 'q_finger', None), m.state_age)

    def _pair_clear(self, m):
        fn = self.geom_v3_fn
        last = getattr(fn, 'last', None) if fn is not None else None
        gl = getattr(self, 'geom_last', None)
        if last is None or gl is None or getattr(self, 'geom_t', None) != m.t:
            return False, 'pair_geometry_missing', None
        dec, det = gl
        T = det.get('T_pred_s') if isinstance(det, dict) else None
        d = (last.get('finger_bar_out') or (None,))[0]
        if not (isinstance(T, (int, float)) and math.isfinite(T) and isinstance(d, (int, float)) and math.isfinite(d)):
            return False, 'pair_geometry_invalid', None
        rate = 0.0
        if self.pair_prev is not None and m.t > self.pair_prev[0]:
            rate = max(0.0, (self.pair_prev[1] - float(d)) / (m.t - self.pair_prev[0]))
        self.pair_prev = (m.t, float(d))
        c = self.geom.c
        v = max(rate, c.v_floor['finger'])
        dp = float(d) - v * float(T) - c.ds_err['finger']
        need = c.d_min['finger'] + c.warn_margin_m
        return dp >= need, (None if dp >= need else 'pair_clearance_low'), {'d': float(d), 'd_pred': dp, 'need': need, 'T_pred': float(T),
                                                                             'rate': rate, 'v_used': v}

    def rg_report(self):
        r = super().rg_report()
        r['version'] = VERSION
        r['bridge_T_s'] = self.ref.T_b
        return r
