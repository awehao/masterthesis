#!/usr/bin/env python3
"""瓶子 S1 候選控制器：繞障 → 回到可操作區 → 移動抓取 → 持瓶進出口區（純邏輯、無 Isaac）。規格：規格/瓶子S1_候選規格.md。
以 S0bController 為基礎（判定、停止、求解、放行管沿用），只加 S1 才需要的部分：
  非直線底盤參考（x 單調的路徑 y(x)，偏航 0）；導航段手臂維持備妥姿態（底盤座標任務，與底盤解耦）；展臂由**位置事件**觸發（繞障延誤不會錯過）；
  成功必須實際進入出口區且底盤停下（不沿用 S0b 的持瓶後停住）。障礙來自 obstacles_fn：本版是真值注入 CBF，**不是感知導航**；無 Raw-scan Shield。
底盤速度：導航段 v_nav、接近段 v_c、持瓶段 v_carry，皆為工程設定，不是最佳化結果。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.se2 import from_xytheta                       # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as SP                    # noqa: E402
import bottle_ref as BR                                               # noqa: E402
import bottle_s0b_ctrl as B                                           # noqa: E402
import bottle_s0_ctrl as C0                                           # noqa: E402
import bottle_s1_geom as G                                            # noqa: E402

N, DT = B.N, B.DT
CLEAR_REJECT = 0.005      # 非手指↔瓶拒絕線（m），事前指定，不因任務調整
CLEAR_GUARD = 0.0005      # 觸發線＝拒絕線＋此值；餘量不足時增加幾何或降低插入速度，不動此值與拒絕線
CLEAR_NEAR = 0.25         # TCP 離瓶超過此距離不核（省算）
A_SLEW = 0.4              # 參考速度斜率（m/s²），工程設定
SHIELD_DV = 1e-3          # Shield 改寫超過此值（m/s 範數）視為非微量
SCAN_HZ, OBS_MAX_AGE = 10.0, 0.5   # 同 bottle_nav_chain
N_STILL = 20              # 出口停住須連續低速的物理步數


class DetourPath:
    """y(x)：在 [x1, x2] 間升餘弦側移 A，其餘 y＝0。x 單調，參考沿 x 前進。"""

    def __init__(self, x1, x2, A):
        self.x1, self.x2, self.A = float(x1), float(x2), float(A)

    def y(self, x):
        if x <= self.x1 or x >= self.x2:
            return 0.0
        return self.A * 0.5 * (1.0 - math.cos(2.0 * math.pi * (x - self.x1) / (self.x2 - self.x1)))

    def dy(self, x):
        if x <= self.x1 or x >= self.x2:
            return 0.0
        w = 2.0 * math.pi / (self.x2 - self.x1)
        return self.A * 0.5 * w * math.sin(w * (x - self.x1))


class S1Controller(B.S0bController):
    def __init__(self, K, ident, T_grasp, T_ready_rel, q_ready, table_top_z, bottle_radius, x_start, x_win, path, exit_c, exit_r, obstacles_fn,
                 x_exit_stop=None, v_nav=0.15, v_c=0.03, v_carry=0.10, x_slow=-0.60, q_tuck=None, t_unfold=3.0, t_settle=1.0, x_unfold=None, t_blend=B.T_BLEND, chain=None, scan_fn=None, t_insert=None, no_tcp_preview=False, no_inflight=False, body_boxes=None, d_body=0.05, body_links=(), carry_no_yaw=False, **kw):
        x_stop = exit_c[0] if x_exit_stop is None else x_exit_stop
        super().__init__(K, ident, T_grasp, T_ready_rel, q_ready if q_tuck is None else q_tuck, table_top_z, bottle_radius, x_start, x_win, x_stop, v_c=v_c, obstacles_fn=obstacles_fn,
                         cbf_cfg=dict(B.CBF_MISSION, **kw.pop('cbf_extra', {})), **kw)
        self.path, self.exit_c, self.exit_r = path, np.asarray(exit_c, float), float(exit_r)
        self.v_nav, self.v_carry, self.x_slow = float(v_nav), float(v_carry), float(x_slow)
        self.x_pre = float(x_win[0]) + 0.05
        self.t_blend = float(t_blend)
        if t_insert is not None:
            self.task.t_insert = float(t_insert)               # 插入放慢：TCP 速度低，追蹤落後與停止續行都變小，不動已驗證的抓取幾何
        self.x_blend = self.x_pre - v_c * self.t_blend         # 朝瓶漸變起點：以接近速度倒推，位置事件觸發
        self.st.t_pre, self.st.t_cruise0 = float('inf'), 0.0
        self.carry_no_yaw, self.mpc_nav0 = bool(carry_no_yaw), self.mpc
        self.mpc_noyaw = None if not carry_no_yaw else B.GW.GMPC(B.dataclasses.replace(self.nav, u_min=self.nav.u_min * np.array([1.0, 1.0, 0.0]), u_max=self.nav.u_max * np.array([1.0, 1.0, 0.0])))   # 持瓶段底盤不旋轉：工程假設，圓形底盤的轉向側牆擋不住
        self.q_tuck, self.q_arm_ready, self.t_unfold, self.t_u0 = (None if q_tuck is None else np.array(q_tuck, float)), np.array(q_ready, float), float(t_unfold), None
        self.x_unfold = (self.x_blend - v_c * (self.t_unfold + float(t_settle))) if x_unfold is None else float(x_unfold)   # 展完靜止 t_settle 再朝瓶：d̂ 隨姿態變，停止點預測要用備妥姿態的靜態值；展臂在底盤座標，可在高速段進行
        self.blending, self.mission_done, self.t_exit_in = False, False, None
        self.n_still, self.T_b_now, self.d_static, self.d_p_prev = 0, None, None, None
        self.no_tcp_preview, self.preview, self.no_inflight = bool(no_tcp_preview), {}, bool(no_inflight)
        self.body_boxes, self.d_body, self._grip_corners, self.body_links = body_boxes, float(d_body), None, tuple(body_links)   # S2：三維障礙（模擬真值，LiDAR 看不到）
        self.t_ms = dict(scan=[], shield=[], advance=[])     # 分項耗時：solve_ms 不含這些，也不含物理與渲染
        self.chain, self.scan_fn, self.t_next_scan = chain, scan_fn, 0.0   # 有 chain 時：障礙只來自掃描鏈，底盤命令只經 Shield 寫出
        self.clear_min = dict(meas=float('inf'), pred=float('inf'), t_meas=None, t_pred=None, n=0)

    # ---- 參考
    def _ref_window(self, xs_eff):
        xs = xs_eff + self.v_b * DT * np.arange(N + 1)
        if self.st.phase == 'CARRY':                           # 持瓶段沿錨定的 y、θ
            return super()._ref_window(xs_eff)
        X_ref = np.array([from_xytheta(float(x), self.path.y(float(x)), 0.0) for x in xs])
        xi_ref = np.array([[self.v_b, self.path.dy(float(x)) * self.v_b, 0.0] for x in xs])
        return X_ref, xi_ref

    def _rel_task(self):
        return self.st.phase == 'CARRY' or (self.st.phase == 'CRUISE' and not self.blending)

    def _T_seq(self, q, t_eff):
        if self.st.phase == 'CRUISE' and not self.blending:    # 導航段：手臂在底盤座標（收臂、關節空間展臂、備妥）
            T = np.stack([self._rel_target(t_eff + k * DT) for k in range(N + 1)])
        else:
            T = super()._T_seq(q, t_eff)
        ph = self.st.phase if self.blending or self.st.phase != 'CRUISE' else 'NAV'
        d = float(np.linalg.norm(T[:, :3, 3] - T[0, :3, 3], axis=1).max())
        self.preview[ph] = max(self.preview.get(ph, 0.0), d)   # 記該相位 TCP 目標在時域內實際變化多少；消融時也照原序列記
        return np.stack([T[0]] * (N + 1)) if self.no_tcp_preview else T   # 消融：只拿掉 TCP 未來參考，N、dt、設定點模型、延遲補償、底盤參考與安全預測不變

    def _obb_tcp(self):
        """夾爪（gripper、link_eef、link6、開爪手指）點雲在 TCP 座標的軸對齊包圍盒 8 角點，只算一次。"""
        if self._grip_corners is None:
            q0 = np.r_[0.0, 0.0, 0.0, self.q_arm_ready]
            Ti = np.linalg.inv(self.K.fk(q0, 'link_tcp'))
            W = G.link_world(q0, G.Q_OPEN)
            P = np.vstack([v for k, v in W.items() if k in ('link6', 'link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')])
            Pt = (Ti[:3, :3] @ P.T).T + Ti[:3, 3]
            lo, hi = Pt.min(axis=0), Pt.max(axis=0)
            self._grip_corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        return self._grip_corners

    def _solve(self, q):
        if self.carry_no_yaw:
            self.mpc = self.mpc_noyaw if self.st.phase in ('LIFT', 'CARRY') else self.mpc_nav0
        return super()._solve(q)

    def _arm_extra(self):
        if not self.body_boxes:
            return {}
        bodies = [dict(link='link_tcp', name='grip', pts=self._obb_tcp())]
        if self.judge.T_rel is not None:                       # 抓持後加瓶身包圍盒 8 角（瓶座標 ±r、±r、0–h），用實際抓持的瓶對 TCP
            r_, h_ = self.judge.br, G.BOTTLE['h']
            bodies.append(dict(link='link_tcp', name='bottle', pts=[tuple((self.judge.T_rel @ np.array([x, y, z, 1.0]))[:3]) for x in (-r_, r_) for y in (-r_, r_) for z in (0.0, h_)]))
        bodies += [dict(link=n_, name=n_, pts=link_aabb(n_)) for n_ in self.body_links]
        slope = 0.5
        return dict(body_obst=dict(boxes=self.body_boxes, bodies=bodies, d_safe=self.d_body * math.sqrt(1.0 + slope ** 2), d_act=0.60, slope=slope, stride=2))   # d_safe 補償斜坡：到箱角真實距離＝d_safe／√(1＋slope²)＝d_body；slope 是提前量，不是安全門檻

    def _rel_target(self, tau):
        if self.q_tuck is None:
            return self.task.T_ready_rel
        s_ = 0.0 if self.t_u0 is None else BR.min_jerk((tau - self.t_u0) / self.t_unfold)
        return self.K.fk(np.r_[0.0, 0.0, 0.0, self.q_tuck + (self.q_arm_ready - self.q_tuck) * s_], 'link_tcp')

    def _unfolded(self):
        return self.q_tuck is None or (self.t_u0 is not None and self.t >= self.t_u0 + self.t_unfold)

    def _speed(self, T_tcp, x):
        if x >= self.x_end:
            return max(self.v_b - B.A_REF * 2.0 * DT, 0.0)
        return self._slew(self._v_target(T_tcp, x))

    def _slew(self, v):
        """參考速度斜率限制：起步、減速點、進持瓶段不做階躍（持瓶時階躍會直接傳到瓶子）。啟發式調速要減速時不受限，避免越窗。"""
        da = A_SLEW * 2.0 * DT
        return min(v, self.v_b + da) if self.blending else float(np.clip(v, self.v_b - da, self.v_b + da))

    def _v_target(self, T_tcp, x):
        if self.st.phase == 'CARRY':
            return self.v_carry
        if not self.blending:
            if x >= self.x_blend and not self._unfolded():     # 手臂還沒展完：底盤等手臂，不靠時間賭
                return max(self.v_b - B.A_REF * 2.0 * DT, 0.0)
            v = self.v_nav if x < self.x_slow else self.v_c
            return v / math.sqrt(1.0 + self.path.dy(x) ** 2)   # v 是路徑速度；參考以 x 前進
        return B.heuristic_speed(self.task, self.st, self.t, T_tcp, x, self.x_we, self.v_c)

    # ---- 相位
    def _bottle_clear(self, q9, T_b):
        Pl = np.vstack([P for n_, P in G.link_world(q9, G.Q_OPEN).items() if n_ not in G.C.FINGERS])
        Pb = (T_b[:3, :3].T @ (Pl - T_b[:3, 3]).T).T
        return float(G.cyl_dist(Pb, 0.0, 0.0, 0.0, G.BOTTLE['h'], self.judge.br).min())

    def _stop_rest(self, q):
        """若此刻停止，機構最後停在哪：在途命令走完（qp, sp），加有界減速距離與穩態偏移。回傳 (命令生效時刻構型, 停止點構型)。"""
        qp, sp = q.copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = SP.plant_phys_step(qp, sp, u_in, self.cfg, 1)
        u = self.u_last
        q_rest = qp.copy()
        c_, s_ = math.cos(qp[2]), math.sin(qp[2])
        dx = u[:3] * np.abs(u[:3]) / (2.0 * B.A_BASE)
        q_rest[:3] += np.array([c_ * dx[0] - s_ * dx[1], s_ * dx[0] + c_ * dx[1], dx[2]])
        d_s = np.asarray(self.obs.d if self.d_static is None else self.d_static, float)   # 穩態 q＝s＋d：用展臂前手臂靜止時的 d̂，運動中的 d̂ 在伺服變慢時被追蹤落後汙染（α×0.4 預測誤差 1.4 mm）
        q_rest[3:] = sp + d_s + u[3:] * np.abs(u[3:]) / (2.0 * C0.A_ARM)
        q_next = q_rest.copy()                             # 下一拍才停的停止點：底盤與手臂都照 MPC 計畫走一拍再減速（線性外推會把快到位時當成全速續衝）
        if self.u_arm_plan is not None and len(self.u_arm_plan) > 1 and self.base_plan is not None:
            u1, v1 = self.u_arm_plan[1], self.base_plan[1]
            d_b = DT * v1 + v1 * np.abs(v1) / (2.0 * B.A_BASE)
            q_next[:3] = qp[:3] + np.array([c_ * d_b[0] - s_ * d_b[1], s_ * d_b[0] + c_ * d_b[1], d_b[2]])
            q_next[3:] = sp + DT * u1 + d_s + u1 * np.abs(u1) / (2.0 * C0.A_ARM)
        return qp, q_rest, q_next

    def _clear_check(self, T_tcp, q):
        """抓持前的非手指↔瓶線上拒絕：核實測、命令生效時刻與停止點構型，預測停止點跨線前停。抓持後瓶隨夾爪，不適用。"""
        T_b = self.T_b_now
        if T_b is None or not np.isfinite(T_b).all() or self.judge.t_grasp is not None or self.st.phase not in ('CRUISE', 'PREGRASP', 'INSERT', 'CLOSE'):
            return
        c = T_b[:3, 3] + 0.5 * G.BOTTLE['h'] * T_b[:3, 2]
        if np.linalg.norm(T_tcp[:3, 3] - c) > CLEAR_NEAR:
            return
        d_m = self._bottle_clear(q, T_b)
        qp, q_rest, q_next = self._stop_rest(q)
        d_p = min(self._bottle_clear(qp, T_b), self._bottle_clear(q_rest, T_b))
        cm = self.clear_min
        cm['n'] += 1
        if d_m < cm['meas']:
            cm['meas'], cm['t_meas'] = d_m, self.t
        if d_p < cm['pred']:
            cm['pred'], cm['t_pred'] = d_p, self.t
        d_next = self._bottle_clear(q_next, T_b)
        d_all = min(d_m, d_p, d_next)                      # 實測、在途構型、當下停止點、下一拍停止點全核
        cm['next'] = min(cm.get('next', float('inf')), d_next)
        if d_all < CLEAR_REJECT + CLEAR_GUARD:
            self.judge._stop(self.t, 'clearance_bottle', meas=d_m, pred=d_p, pred_next=d_next)

    def _advance(self, T_tcp, close_done, q):
        t_s = time.perf_counter()
        self._advance_inner(T_tcp, close_done, q)
        self.t_ms['advance'].append((time.perf_counter() - t_s) * 1e3)

    def _advance_inner(self, T_tcp, close_done, q):
        if self.chain is not None and self.chain.age(self.t) > OBS_MAX_AGE:
            self.judge._stop(self.t, 'scan_stale', age=self.chain.age(self.t))   # 比 Shield 限速更保守：看不到環境就不繼續任務；Shield 仍在輸出端
        self._clear_check(T_tcp, q)
        if self.q_tuck is not None and self.t_u0 is None and q[0] >= self.x_unfold:
            self.t_u0 = self.t
            self._ev('unfold_start', x=float(q[0]), y=float(q[1]))
        if self.st.phase == 'CRUISE' and not self.blending and q[0] >= self.x_blend and self._unfolded():
            if self._in_tube(np.r_[max(q[0], self.x_ws), q[1:3]]):
                self.blending = True
                self.d_static = np.array(self.obs.d, float)
                self.st.t_cruise0, self.st.t_pre = self.t, self.t + self.t_blend
                self._ev('blend_start', x=float(q[0]), y=float(q[1]))
            else:
                self.n_gate += 1
        super()._advance(T_tcp, close_done, q)

    def step(self, t, q, **kw):
        q_prev = getattr(self, 'q_prev2', None)            # 須在基底 step 更新前取，否則當步差分恆為 0
        self.T_b_now = kw.get('T_bottle')
        q = np.asarray(q, float)
        if self.chain is not None and t >= self.t_next_scan - 1e-9 and np.isfinite(q).all():
            self.t_next_scan = t + 1.0 / SCAN_HZ
            t_s = time.perf_counter()
            r_ = self.scan_fn(t, q)
            if r_ is not None:                                 # None＝這幀掃描沒來（資料失效測試、感測中斷）
                self.chain.on_scan(t, r_, q[:3])
            self.t_ms['scan'].append((time.perf_counter() - t_s) * 1e3)
        out = super().step(t, q, **kw)
        if self.chain is not None:
            u_in = np.array(out['base_twist'], float)
            t_s = time.perf_counter()
            out['base_twist'] = self.chain.shield(t, u_in)     # 最終底盤命令：含停止路徑，無旁路
            self.t_ms['shield'].append((time.perf_counter() - t_s) * 1e3)
            dv = float(np.linalg.norm(out['base_twist'] - u_in))
            if dv > SHIELD_DV and self.judge.stop is None:     # 改寫後實際底盤與 MPC 計畫不再一致，不當原全身序列繼續：走受控停止
                self.judge._stop(self.t, 'shield_intervened', dv=dv, u_in=u_in.tolist(), u_out=out['base_twist'].tolist())
        if np.isfinite(q).all():
            in_zone = float(np.hypot(*(q[:2] - self.exit_c))) <= self.exit_r
            if in_zone and self.t_exit_in is None and self.st.phase == 'CARRY':
                self.t_exit_in = self.t
                self._ev('exit_enter', x=float(q[0]), y=float(q[1]))
            v_meas = float('inf') if q_prev is None else float(np.hypot(*(q[:2] - q_prev[:2]))) / C0.PHYS
            self.n_still = self.n_still + 1 if v_meas < 0.01 else 0   # 實測連續低速，不以參考速度歸零代替
            if self.done and in_zone and self.t_exit_in is not None and self.v_b == 0.0 and self.n_still >= N_STILL and self.judge.stop is None and not self.mission_done:
                self.mission_done = True
                self._ev('mission_done', x=float(q[0]), y=float(q[1]))
        out['done'] = self.mission_done
        return out

    def report(self):
        r = super().report()
        pc = lambda v: None if not v else dict(p50=round(float(np.percentile(v, 50)), 2), p95=round(float(np.percentile(v, 95)), 2), max=round(float(np.max(v)), 2), n=len(v))
        asm = [e['asm_ms'] for e in self.qp_log if e.get('asm_ms') is not None]
        gi = [e['gi_ms'] for e in self.qp_log if e.get('gi_ms')]
        r.update(timing=dict(solve=pc(self.solve_ms), assemble=pc(asm), gi=pc(gi), **{k: pc(v) for k, v in self.t_ms.items()}))
        cm = self.clear_min
        r.update(clear_bottle=dict(reject_m=CLEAR_REJECT, guard_m=CLEAR_GUARD, meas_min=None if cm['n'] == 0 else cm['meas'], pred_min=None if cm['n'] == 0 else cm['pred'], next_min=cm.get('next'), t_meas=cm['t_meas'], t_pred=cm['t_pred'], n=cm['n']), n_compact=None if self.compact is None else len(self.compact))
        if self.chain is not None:
            r.update(nav_chain=self.chain.report())
        r.update(body_bind=getattr(self, 'ob_body_bind', {}), body_rows_by=getattr(self, 'ob_body_rows', {}), body_rows_max=getattr(self, 'ob_rows', 0), body_planes=getattr(self, 'ob_plane_count', {}), body_cfg=None if not self.body_boxes else dict(d_body=self.d_body, slope=0.5, d_act=0.60, stride=2, grip_corners=[list(map(float, c_)) for c_ in self._obb_tcp()], boxes=[[np.asarray(c_).tolist(), np.asarray(z_).tolist()] for c_, z_ in self.body_boxes]), no_inflight=self.no_inflight, no_tcp_preview=self.no_tcp_preview, preview_range_mm={k: round(v * 1e3, 2) for k, v in self.preview.items()})
        r.update(exit_reached=self.t_exit_in is not None, t_exit_in=self.t_exit_in, mission_done=self.mission_done, blending=self.blending,
                 nav_source='truth_obstacle_to_cbf' if self.chain is None else 'lidar_tracker_to_cbf', shield=self.chain is not None, success_s0_judge=r['success'], success=bool(r['success'] and self.mission_done))
        return r


def link_aabb(name):
    """連桿點雲在該連桿座標的軸對齊包圍盒 8 角（幾何近似，取自點雲）。"""
    P = G.C.CL[name]
    lo, hi = P.min(axis=0), P.max(axis=0)
    return [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def save_compact(c, path):
    """存實跑 CBF 證據：實際命令、殘差、slack 與重解輸入；反事實重解由 bottle_s1_cbf_replay.py 在實跑後做。"""
    L = c.compact or []
    obs = np.array([json.dumps(e.get('obs') or []) for e in L])
    keys_arr = ('t', 'phase', 'qp', 'sp', 'u_prev', 'X_ref', 'xi_ref', 'T_seq', 'rel', 'w_p', 'bias', 'u', 'cbf_active', 'min_h', 'eps0', 'resid_ns', 'resid_s')
    np.savez_compressed(path, obs_json=obs, **{k: np.array([e[k] for e in L]) for k in keys_arr})


def exit_code(rep):
    """完整任務才回 0：抓持五項成立、實際進入出口區、底盤停下。"""
    if rep.get('stop') is None and rep.get('success') is True:
        return 0
    if rep.get('stop') is None:
        return 4                                               # 未中止也未完成（未抵出口、逾時）
    return 3 if (rep.get('stop_effect') or {}).get('ok') is True else 4
