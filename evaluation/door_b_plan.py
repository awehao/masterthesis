#!/usr/bin/env python3
"""主線 B 的線上接近規劃（由 door_obs_candidate.py 的離線搜尋整理成函式；任務監督節點在工作程序裡呼叫）。
輸入：當次把手估計 T_H（世界／地圖座標）、導航實際路徑（Smac /plan 點列）、任務朝向 yaw、當下手臂構型 q_now、抓持參數。
輸出 plan（dict）：
  path_door：路徑在門座標的 (a, sb)；door_line_sb：門前線離門距離；
  obs：觀察構型族 [(a, q, par)]（相機注視把手、可見、σ／限位／淨空）；
  a_w：接手點（門前線、沿門 −a 窗口可行者中，把手後方最近者）、q_pre：接手預抓位；
  a_t0、T_tr：觀察→預抓位過渡的起點與時長（盲段最短且全程可行者）；blind_s；
  goal_xy：導航遠端目標（a_w 之後，離接手點的弧長 ≥ 到達容差＋交權準備＋餘裕）；
  why：不可行時的原因（照實回報，不降級）。
所有檢查與離線候選同式：σ≥SIG_FLOOR、限位、淨空（規劃門檻 THR_PLAN）、可見（25 px 邊距）、立即停止的終點構型淨空、關節速率／加速度上限。"""
import math

import numpy as np

import door_fast_kin as DFK
import door_motm_grasp_ctrl as G
import door_wbopt as W
import door_clear_door_fast as _F
import shield_geom as SHG
import wrist_v0_capture as W0

W.clear_door = _F.clear_door_fast
K = G.K
S_PRE = float(G.PG.S_PRE)
LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float) + 0.05, np.asarray(G.XO.LITE6_SAFE.upper, float) - 0.05
Rg, T_HG, a_H = G.PG.baseline_reference()
DS_SEED = np.array([-0.354, -0.088, 1.435, -1.421, -0.358, -0.157])
OBS_P = [(rho, e, az) for rho in (0.35, 0.40, 0.45) for e in (0, 10, 20, 30) for az in range(-80, 81, 10)]
D_RNG, EDGE = (0.25, 1.2), 40                                        # 視覺管線的偵測距離與影像邊距（m8_door_reaim_online_r3 的 D_RNG／EDGE）
ENV_COVER = dict(range=(0.35, 0.50), oblique_max=13.0)               # 既有 Isaac 實際影格涵蓋的觀測包絡（results/motm_b/vis_error_by_range.json：0.35–0.5 m、斜視 6.2–12.8°，n＝101）
T_READY = 2.05


class BPlanner:
    def __init__(self, T_H, d, n, yaw, Kc, v_grasp=0.05, goal_tol=0.30, t_prep=0.1, margin=0.05):
        self.T_H, self.d, self.n, self.yaw = np.asarray(T_H, float), np.asarray(d, float), np.asarray(n, float), float(yaw)
        self.h_xy, self.H = self.T_H[:2, 3], self.T_H[:3, 3]
        self.a_n = self.T_H[:3, :3] @ np.asarray(a_H, float)
        self.Kc = np.asarray(Kc, float)
        self.ENDS = [self.H, self.H + self.T_H[:3, 0] * 0.06, self.H - self.T_H[:3, 0] * 0.06]
        self.V, self.goal_tol, self.t_prep, self.margin = float(v_grasp), float(goal_tol), float(t_prep), float(margin)
        self.SH = SHG.ShieldGeom()
        W.set_model(phi0=0.0, finger_q=G.XO.FINGER_OPEN)
        self.T_TC = None

    # ---- 幾何
    def base(self, a, sb):
        return np.r_[self.h_xy + sb * self.n - a * self.d, self.yaw]

    def to_door(self, xy):
        rel = np.asarray(xy, float) - self.h_xy
        return float(-(rel @ self.d)), float(rel @ self.n)

    def target(self, r):
        T = np.asarray(G.E.OT.object_target(self.T_H, T_HG, a_H, 0.0)['T_WE'], float).copy()
        T[:3, 3] += r * self.a_n
        return T

    def solve(self, b, Tt, seed):
        q, res, sm = DFK.ik_closure_fc(Tt, b, seed)
        if not (res < 1e-6 and np.isfinite(q).all()):
            return None, None
        return q, sm

    def visible(self, q9):
        Ti = np.linalg.inv(K.fk(q9, W0.OPT))
        for p in self.ENDS:
            pc = (Ti @ np.r_[p, 1.0])[:3]
            if pc[2] < 0.08:
                return False
            u, v = (self.Kc @ (pc / pc[2]))[:2]
            if not (25 <= u <= 615 and 25 <= v <= 455):
                return False
        return True

    def frame(self, q9, nrm=None):
        """視覺管線會產生有效估計的影格條件：把手中心與兩端在影像內（邊距 EDGE）、距離在 D_RNG。回 (距離, 斜視角°) 或 None。斜視角＝相機→把手方向與門面法向夾角。"""
        Tc = K.fk(q9, W0.OPT); Ti = np.linalg.inv(Tc); rng = None
        for j, p in enumerate(self.ENDS):
            pc = (Ti @ np.r_[p, 1.0])[:3]
            if pc[2] < 0.08:
                return None
            u, v = (self.Kc @ (pc / pc[2]))[:2]
            if not (EDGE <= u <= 640 - EDGE and EDGE <= v <= 480 - EDGE):
                return None
            if j == 0:
                rng = float(np.linalg.norm(pc))
        if not (D_RNG[0] <= rng <= D_RNG[1]):
            return None
        nrm = -np.r_[self.n, 0.0] if nrm is None else nrm               # 由機器人側看向門
        v = self.H - Tc[:3, 3]; v /= np.linalg.norm(v)
        return rng, math.degrees(math.acos(max(-1.0, min(1.0, float(v @ nrm)))))

    def covered(self, fr):
        return fr is not None and ENV_COVER['range'][0] <= fr[0] <= ENV_COVER['range'][1] and fr[1] <= ENV_COVER['oblique_max']

    def ok_cfg(self, b, q):
        lim = float(min((q - LO).min(), (HI - q).min()))
        cl = W.clear_door(np.r_[b, q, 0.0], self.T_H)
        return lim >= 0 and all(cl[k] >= W.THR_PLAN[k] for k in ('env', 'base', 'self')), lim, cl

    def sigma6(self, q9):
        J = K.jacobian(q9, 'link_tcp')[:, 3:9]
        return float(np.linalg.svd(J, compute_uv=False).min())

    def pregrasp(self, a, sb):
        """門旁分支預抓位（由門旁種子在 a＝−0.115、sb 0.45 的抓取位→後退 S_PRE→沿 a、再沿 sb 延拓）。"""
        sd, _ = self.solve(self.base(-0.115, 0.45), self.target(0.0), DS_SEED)
        if sd is None:
            return None
        sd, _ = self.solve(self.base(-0.115, 0.45), self.target(S_PRE), sd)
        for k in range(1, 41):
            sd, _ = self.solve(self.base(-0.115 + (a + 0.115) * k / 40, 0.45), self.target(S_PRE), sd)
            if sd is None:
                return None
        for k in range(1, 21):
            sd, _ = self.solve(self.base(a, 0.45 + (sb - 0.45) * k / 20), self.target(S_PRE), sd)
            if sd is None:
                return None
        return sd

    def look_at(self, b, rho, e_deg, az_deg):
        u = np.r_[b[:2] - self.H[:2], 0.0]; u /= np.linalg.norm(u)
        c, s = math.cos(math.radians(az_deg)), math.sin(math.radians(az_deg))
        u = np.array([c * u[0] - s * u[1], s * u[0] + c * u[1], 0.0])
        e = math.radians(e_deg)
        p = self.H + rho * (math.cos(e) * u + np.array([0, 0, math.sin(e)]))
        z = (self.H - p) / np.linalg.norm(self.H - p)
        x = np.cross(np.array([0, 0, 1.0]), z); x /= np.linalg.norm(x)
        Tc = np.eye(4); Tc[:3, :3] = np.c_[x, np.cross(z, x), z]; Tc[:3, 3] = p
        return Tc @ np.linalg.inv(self.T_TC)

    # ---- 窗口
    def window_ok(self, a_w, sb, q0, t_win=2.2):
        seed = q0.copy()
        for k in range(int(round(t_win / 0.1)) + 1):
            a_ = a_w - self.V * 0.1 * k
            r_ = S_PRE * max(0.0, 1.0 - k / 12)
            b = self.base(a_, sb)
            q, sm = self.solve(b, self.target(r_), seed)
            if q is None or np.abs(q - seed).max() > 0.3 or sm < W.SIG_FLOOR:
                return False
            okc, _, _ = self.ok_cfg(b, q)
            c, s_ = math.cos(self.yaw), math.sin(self.yaw)
            vw = -self.d * self.V
            m, _ = self.SH.margin(b, np.array([c * vw[0] + s_ * vw[1], -s_ * vw[0] + c * vw[1], 0.0]), 0.0)
            if not okc or m < 0:
                return False
            seed = q
        return True

    # ---- 主函式
    def plan(self, path_xy, q_now, a_w_cands=(-0.08, -0.10, -0.12), v_arm=1.0, a_arm=19.984, a_body=1.0):
        path = np.asarray(path_xy, float)
        AD = np.array([self.to_door(p) for p in path])
        if len(AD) < 3:
            return dict(why='path_short')
        sb_end = AD[-1, 1]
        line = np.abs(AD[:, 1] - sb_end) < 0.03
        sb = float(np.median(AD[line, 1]))
        out = dict(door_line_sb=round(sb, 4), path_door=AD.round(4).tolist())
        if not (AD[line, 0].max() >= 0.40 and AD[line, 0].min() <= -0.20):
            out['why'] = 'path_not_along_door_line_through_window'
            return out
        # 接手點
        a_w, q_pre = None, None
        for aw in a_w_cands:
            qp = self.pregrasp(aw, sb)
            if qp is not None and self.window_ok(aw, sb, qp):
                a_w, q_pre = aw, qp
                break
        if a_w is None:
            out['why'] = 'no_feasible_window'
            return out
        if self.T_TC is None:
            q9 = np.r_[self.base(a_w, sb), q_pre]
            self.T_TC = np.linalg.inv(K.fk(q9, 'link_tcp')) @ K.fk(q9, W0.OPT)
        # 觀察族
        fam, seed = [], None
        A_SEQ = np.round(np.arange(0.04, 0.601, 0.02), 3)                # 由近端往遠端延拓：近端先找正面（落在既有影格包絡內）的視角，再往遠端連續（遠端斜視可較大）
        for a in A_SEQ:
            b = self.base(float(a), sb)
            best = None
            for (rho, e, az) in OBS_P:
                tgt = self.look_at(b, rho, e, az)
                for sd in ([seed] if seed is not None else []) + [np.asarray(q_now, float), q_pre]:
                    q, sm = self.solve(b, tgt, sd)
                    if q is None or sm < W.SIG_FLOOR or (seed is not None and np.abs(q - seed).max() > 0.3):
                        continue
                    okc, _, _ = self.ok_cfg(b, q)
                    fr = self.frame(np.r_[b, q]) if okc else None
                    if fr is None:
                        continue
                    cost = (fr[1], 0.0 if seed is None else float(np.abs(q - seed).max()))   # 優先最小斜視（落在既有影格包絡），再看連續性
                    if best is None or cost < best[0]:
                        best = (cost, q, (rho, e, az), fr)
                    break
            if best is None:
                if fam:
                    break
                continue
            seed = best[1]
            fam.append(dict(a=float(a), q=best[1], par=best[2], range=round(best[3][0], 3), oblique=round(best[3][1], 1), covered=self.covered(best[3])))
        fam = fam[::-1]                                                  # 依行進順序（a 由大到小）
        if not fam:
            out['why'] = 'no_observation_family'
            return out
        out.update(a_w=a_w, q_pre=q_pre.tolist(), obs=[dict(a=f['a'], q=f['q'].tolist(), par=f['par'], range=f['range'], oblique=f['oblique'], covered=f['covered']) for f in fam])
        # 過渡
        best = None
        for f in fam:
            prof = self.speed_profile(f['a'], a_w)
            if prof is None:
                continue
            T, a_of_t = prof
            r = self._minjerk(f['q'], q_pre, T, f['a'], a_w, sb, v_arm, a_arm, a_body, a_of_t=a_of_t)
            if r is None:
                continue
            t_cov = r['t_cov'] if r['t_cov'] is not None else (0.0 if f['covered'] else None)
            if t_cov is None:
                continue                                              # 過渡起點與過渡中都沒有落在包絡內的有效影格 ⇒ 不採（不能外推）
            blind = (T - t_cov) + T_READY + 0.05
            if best is None or blind < best[0]:
                best = (blind, f, T, r, t_cov)
        if best is None:
            out['why'] = 'no_feasible_transition_with_covered_frame'
            return out
        f, T, r, t_cov = best[1], best[2], best[3], best[4]
        out.update(a_t0=f['a'], T_tr=round(T, 3), last_valid_t=r['t_valid'], last_covered_t=round(t_cov, 3), last_covered_frame=r['fr_cov'] or [f['range'], f['oblique']],
                   blind_s=round(best[0], 3), blind_def='最後一格落在既有影格包絡內的有效影格 → READY（抓持排程 2.05 s＋交接 1 拍；插入只在排程內）', v_tr=self.v_tr, a_dec=self.a_dec)
        # 導航遠端目標：沿路徑，接手點後弧長 ≥ 到達容差＋V·t_prep＋餘裕（取路徑點）
        need = self.goal_tol + self.V * self.t_prep + self.margin
        i_w = int(np.argmin(np.abs(AD[:, 0] - a_w) + 10 * (~line)))
        seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path[i_w:], axis=0), axis=1))]
        j = int(np.searchsorted(seg, need))
        if j >= len(seg):
            out['why'] = 'goal_beyond_path_end'
            return out
        out['goal_xy'] = path[i_w + j].round(4).tolist()
        out['why'] = None
        return out

    def stop_sweep(self, b, q, ub_body, ua, alpha, D=8, dt=0.01, a_body=(1.5, 1.0, 2.0), a_arm=19.984, every=2, max_steps=400):
        """完整停止前推（Codex 20261117_021349：含在途管線與中間路徑，不用終點近似）。停止決定當下：在途管線＝前 D 步的穩態命令（底盤本體速度 ub_body、手臂設定點速率 ua），
        之後進管線的命令以各軸加速度上限減速到零（底盤 a_body、手臂 a_arm）；底盤照管線輸出積分、手臂設定點照管線輸出積分、關節一階追設定點（α）。
        每 every 步核一次淨空（規劃門檻，門關閉）；回 (ok, 步數, 失敗描述)。"""
        al = np.asarray(alpha, float)
        bb = np.asarray(b, float).copy(); qq = np.asarray(q, float).copy(); sp = qq.copy()
        pipe = [np.r_[ub_body, ua] for _ in range(D)]
        u = np.r_[ub_body, ua].astype(float)
        da = np.r_[np.asarray(a_body, float), np.full(6, a_arm)] * dt
        for k in range(max_steps):
            u = u - np.clip(u, -da, da)
            pipe.append(u.copy()); ue = pipe.pop(0)
            c, s_ = math.cos(bb[2]), math.sin(bb[2])
            bb = bb + dt * np.array([c * ue[0] - s_ * ue[1], s_ * ue[0] + c * ue[1], ue[2]])
            sp = sp + dt * ue[3:]
            qq = qq + al * (sp - qq)
            if k % every == 0:
                okc, _, cl = self.ok_cfg(bb, qq)
                if not okc:
                    return False, k, dict(step=k, who=cl['who'], clear={kk: round(float(cl[kk]), 4) for kk in ('env', 'base', 'self')})
            if not np.any(u) and not any(np.any(x) for x in pipe) and float(np.abs(sp - qq).max()) < 1e-4:
                return True, k, None
        return False, max_steps, dict(why='not_settled')

    v_tr, a_dec = 0.10, 0.20

    def speed_profile(self, a0, a1):
        """過渡段底盤沿門前線（往 −a）：v_tr 等速，接手前以 a_dec 連續減速到抓持速度 V（接手點恰為 V）。回 (時長, a(t) 函式) 或 None。"""
        L = a0 - a1
        if L <= 0:
            return None
        d_dec = (self.v_tr ** 2 - self.V ** 2) / (2 * self.a_dec)
        if d_dec >= L:
            return None
        t1 = (L - d_dec) / self.v_tr
        t2 = (self.v_tr - self.V) / self.a_dec

        def a_of_t(t):
            if t <= t1:
                return a0 - self.v_tr * t
            tt = min(t - t1, t2)
            return a0 - (L - d_dec) - (self.v_tr * tt - 0.5 * self.a_dec * tt * tt)
        return t1 + t2, a_of_t

    def _minjerk(self, qa, qb, T, a0, a1, sb, v_arm, a_arm, a_body, n=60, a_of_t=None):
        """篩選用（不是停止證書）：關節空間最小急動＋底盤沿門前線依速度剖面；逐點 σ、淨空、限位、速率／加速度、穩態管線停止前推模型；記最後有效／包絡內影格。"""
        dq = qb - qa
        if float(np.abs(dq).max()) * 1.875 / T > v_arm or float(np.abs(dq).max()) * 5.7735 / T ** 2 > a_arm:
            return None
        t_valid, t_cov, fr_cov = None, None, None
        for k in range(n + 1):
            u = k / n
            t = u * T
            s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            sd = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T
            q = qa + s * dq
            a = a_of_t(t) if a_of_t is not None else a0 + (a1 - a0) * u
            b = self.base(a, sb)
            okc, _, _ = self.ok_cfg(b, q)
            if not okc or self.sigma6(np.r_[b, q]) < W.SIG_FLOOR:
                return None
            fr = self.frame(np.r_[b, q])
            if fr is not None:
                t_valid = t
                if self.covered(fr):
                    t_cov, fr_cov = t, [round(fr[0], 3), round(fr[1], 1)]
            v_b = (a_of_t(min(t + 1e-3, T)) - a_of_t(t)) / 1e-3 if a_of_t is not None else (a1 - a0) / T
            c, s_ = math.cos(self.yaw), math.sin(self.yaw)
            vw = -self.d * abs(v_b)
            ub = np.array([c * vw[0] + s_ * vw[1], -s_ * vw[0] + c * vw[1], 0.0])
            ok2, _, _ = self.stop_sweep(b, q, ub, sd * dq, self._alpha if hasattr(self, '_alpha') else np.full(6, 0.095))
            if not ok2:
                return None
        return dict(t_valid=None if t_valid is None else round(t_valid, 3), t_cov=t_cov, fr_cov=fr_cov)
