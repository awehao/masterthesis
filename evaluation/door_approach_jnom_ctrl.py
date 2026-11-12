#!/usr/bin/env python3
"""【jnom 對照版（2026-11-12；Codex 20261112_124807 診斷包）】door_approach_ctrl.py（r4 凍結，不改）的版本化複本。唯一差異：建構參數 joint_nominal（預設 False＝與 r4 逐行同行為）。
joint_nominal=True 時，底盤固定相位的求解把「重算出的關節軌跡」當線性化名目（gmpc_wb 既有的 arm['u_arm_nom']）：
  名目設定點＝一階設定點模型對關節參考的反算 s_ref＝q_ref＋τ·q̇_ref−bias/α；名目命令 u_k＝clip((s_ref(t_{k+1})−s_k)/dt, ±v_arm)，沒有新增可調增益。
  依據：m8house_da_approach_1 紀錄顯示關節內插參考經過近奇異構型（Jacobian 最小奇異值 0.009）；原版以「命令為零」為線性化點、只追 TCP，
  在該處所需關節速度 ~9.9 rad/s（上限 1.0）⇒ 偏離後飽和追趕。代價、限制、門檻、參考時間都不變；TCP 代價仍是唯一追蹤項。
以下為原說明。
住宅場景門把手「非接觸接近量測」控制器（Codex 20261112 r5 裁定；診斷配置：到位重瞄，底盤到位後保持、量手臂後段追蹤）。
不進插入、不閉爪、不開門。終點＝預抓位姿（沿接近軸離抓取位姿退 S_PRE＝0.03 m）。
執行鏈與瓶子主線同一套底層（不另寫簡化模型）：W-GMPC 全身求解（gmpc_wb.GMPC＋手臂設定點模型）、在途命令管線 D_STEPS 步、設定點積分與限位、
  偏置觀測器、停止＝命令逐控制拍以 A_BASE／A_ARM 有界減速（在途命令照管線走完）、停止預測＝bottle_w1_compactcarry21_ctrl.stop_traj（同一設定點模型逐物理步）。
相位（r3，依 Codex 20261112_112746）：TO_LOOK（底盤 base0→停位、手臂→重瞄姿態；全身求解）→ SETTLE（切到**底盤固定求解**：求解器內底盤速度上下界＝0；
  等在途底盤命令全部為零且實測底盤速度連續低於門檻才算到位）→ LOOK（只收到位後才拍的新量測）→ 重新規劃（實測底盤位姿、當下手臂）
  → TO_PRE（底盤固定求解；TCP 參考在世界座標）→ END（TCP 位置／姿態、底盤保持偏差、實測速度連續滿足登錄條件才完成；逾時 ⇒ 停止 end_not_reached）。
  底盤固定是求解器內的約束（第二份 GMPC 設定 u_min＝u_max＝0），不是求解後把命令歸零；固定下到不了就停，底盤不補。
每個控制拍：
  ・求解前：以真停止管線預測「現在停」（在途命令＋上一命令減速）整段軌跡，逐取樣核全碰撞對（手指對橫桿不豁免）；違反 ⇒ 受控停止。
  ・求解後、提交前：以「在途命令＋候選命令」預測「下一拍停／再晚一拍停」；違反 ⇒ 候選命令不提交、受控停止（candidate_stop_violation）。
  ・量測：逾時、早於到位、過期、未來時間戳、非有限值、非法位姿矩陣、重新規劃失敗／碰撞 ⇒ 拒收或受控停止。
  ・保持相位底盤偏離保持點超過門檻 ⇒ 受控停止（base_hold_violation）。
把手位姿只來自視覺估計（遠距估計→近距估計）；模擬真值不進控制。"""
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault('M8_SCENE_REG', '20261112_m8_house_scene.json')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import gmpc_wb as GW                          # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as S                     # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                  # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta                       # noqa: E402

import bottle_s0_ctrl as C0                                           # noqa: E402
import bottle_s0b_ctrl as B                                           # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21                           # noqa: E402
import door_clear_fast as FAST                                        # noqa: E402
import m8_door_reaim_online_r4 as R4                                  # noqa: E402
import offset_interval as OINT                                        # noqa: E402

R3, R2, R1, PG, DOC, K, IDX = R4.R3, R4.R2, R4.R1, R4.PG, R4.DOC, R4.K, R4.IDX
FINGER_Q = 0.0089                                                     # 本包手指固定張開量（m）；執行器起跑讀回兩指關節與連桿位姿核對，超出即拒跑（Codex 20261112_121455）


def CLEAR(q, T_WH):
    """全碰撞對淨空，手指用實際張開構型（finger_q＝0 時與 R3.cfg_clear_full(contact=False) 等值；door_clear_fast_check.py）。"""
    return FAST.clear_full(q, T_WH, finger_q=FINGER_Q)

PHYS, NPH, D_STEPS, N, DT = C0.PHYS, C0.NPH, C0.D_STEPS, B.N, B.DT
SETTLE_S, LOOK_TIMEOUT_S, OBS_MAX_AGE_S = 0.5, 3.0, 1.0
HOLD = ('SETTLE', 'LOOK', 'TO_PRE', 'END')
STILL = dict(base_mps=0.002, yaw_rps=0.002, n_ticks=4, timeout_s=2.0)                 # 到位停穩：實測（相鄰控制拍差分）連續 n 拍
HOLD_TOL = dict(base_m=0.005, yaw_rad=math.radians(0.5))                             # 保持相位底盤偏離保持點上限
END_TOL = dict(tcp_m=0.005, tcp_deg=1.0, arm_rps=0.02, n_ticks=10, wait_s=3.0)       # 終端到位：連續 n 拍；逾時 ⇒ end_not_reached
STOP_STILL = dict(arm_rps=1e-3, base_mps=0.002, yaw_rps=0.002, hold_s=0.3, timeout_s=3.0)
# 以上為工程設定（事前登錄，不依結果調整）
ARRIVE_TOL = dict(base_m=0.03, yaw_rad=math.radians(3.0), joint_rad=0.05)          # 到位判定（超過 ⇒ 受控停止 arrive_out_of_tol；不是精度宣稱）


def _mj(u):
    u = min(max(u, 0.0), 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5, (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4)


def _tb(b):
    c, s = math.cos(b[2]), math.sin(b[2])
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [b[0], b[1], 0.0]
    return T


def valid_pose(T):
    T = np.asarray(T, float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        return False
    R = T[:3, :3]
    return bool(np.abs(R.T @ R - np.eye(3)).max() < 1e-6 and np.linalg.det(R) > 0 and np.abs(T[3] - [0, 0, 0, 1]).max() < 1e-9)


def _rot_deg(Ta, Tb):
    return math.degrees(math.acos(float(np.clip((np.trace(Ta[:3, :3].T @ Tb[:3, :3]) - 1) / 2, -1, 1))))


class DoorApproachCtrl:
    def __init__(self, ident, base0, q_obs, b1, q_look, T_WH_far, target_fn, sweep_stride=2, joint_nominal=False):
        """target_fn(T_WH, s) → 世界座標 TCP 目標（s＝沿接近軸退讓量；0＝抓取）。"""
        self.cfg = C0.make_cfg(ident)
        self.joint_nominal, self.n_sat = bool(joint_nominal), {}
        self.cfg.base_fixed = False
        self.mpc = GW.GMPC(B.nav_cfg())
        hc = B.nav_cfg()
        hc.u_min, hc.u_max = np.zeros(3), np.zeros(3)                 # 底盤固定：求解器內速度界＝0
        self.mpc_hold = GW.GMPC(hc)
        self.s = np.array(q_obs, float)
        self.pipe = [np.zeros(9) for _ in range(D_STEPS)]
        self.u_prev, self.u_last = np.zeros(9), np.zeros(9)
        self.obs = OINT.IntervalOffsetObserver(self.cfg.arm_model.alpha, self.cfg.arm_model.phys_dt, 0.5, 0.05)
        self.bias0 = np.asarray(self.cfg.arm_model.bias, float).copy()
        self.q_anchor, self.s_seq = None, []
        self.lim_lo, self.lim_hi = np.asarray(LITE6_SAFE.lower, float) + 0.05, np.asarray(LITE6_SAFE.upper, float) - 0.05
        self.base0, self.b1, self.q_obs, self.q_look = np.array(base0, float), np.array(b1, float), np.array(q_obs, float), np.array(q_look, float)
        self.T_WH, self.T_src, self.target_fn = np.array(T_WH_far, float), 'far', target_fn
        self.t, self.n_step, self.phase, self.stop, self.done = 0.0, 0, 'TO_LOOK', None, False
        self.events, self.solve_ms, self.sweep_ms, self.cand_ms, self.tick_ms, self.replan = [], [], [], [], [], None
        self.stopped_t, self.q_last, self.stop_ok, self._still_steps = None, None, None, 0
        T1, _ = R2.bounded_T(self.base0, self.q_obs, self.b1, self.q_look)
        self.seg = dict(kind='rel', t0=0.0, T=float(T1), ba=self.base0.copy(), bb=self.b1.copy(), qa=self.q_obs.copy(), qb=self.q_look.copy())
        self.look, self.obs_in, self.obs_rej, self.end = None, None, {}, None
        self.q_tick, self.rate = None, dict(base=0.0, yaw=0.0, arm=0.0)
        self.hold0, self.hold_dev, self._n_still, self._n_end, self.t_settle = None, dict(base_m=0.0, yaw_rad=0.0), 0, 0, None
        self.sweep_stride = int(sweep_stride)
        z = dict(env=np.inf, base=np.inf, self=np.inf, bar_tool=np.inf)
        self.mins = dict(act=dict(z), stop=dict(z), cand=dict(z), n_sweep_cfg=0)
        self.track, self.fail_inject = [], None
        self._ev('plan', T_to_look=round(T1, 3), b1=[round(float(v), 4) for v in self.b1])

    def _ev(self, what, **kw):
        self.events.append(dict(t=round(float(self.t), 3), what=what, **kw))

    def _stop(self, why, **kw):
        if self.stop is None:
            self.stop = dict(t=round(float(self.t), 3), why=why, phase=self.phase, **kw)
            self._ev('stop', why=why, **kw)

    # ---- 參考
    def _ref(self, tau):
        """回 (底盤參考 (3,)、底盤世界速度 (3,)、手臂關節參考 (6,))。"""
        g = self.seg
        p, dp = _mj((tau - g['t0']) / g['T'])
        v = dp / g['T'] if g['t0'] <= tau <= g['t0'] + g['T'] else 0.0
        db = np.r_[g['bb'][:2] - g['ba'][:2], R1.wrap(g['bb'][2] - g['ba'][2])]
        return g['ba'] + p * db, v * db, g['qa'] + p * (g['qb'] - g['qa'])

    def _windows(self, t_eff):
        X, xi, Ts = [], [], []
        for k in range(N + 1):
            b, vb, qa = self._ref(t_eff + k * DT)
            c, s = math.cos(b[2]), math.sin(b[2])
            X.append(from_xytheta(float(b[0]), float(b[1]), float(b[2])))
            xi.append([c * vb[0] + s * vb[1], -s * vb[0] + c * vb[1], vb[2]])
            Trel = K.fk(np.r_[0.0, 0.0, 0.0, qa], 'link_tcp')
            Ts.append(Trel if self.seg['kind'] == 'rel' else _tb(self.seg['b_world']) @ Trel)
        return np.array(X), np.array(xi), np.array(Ts)

    # ---- 量測（到位後的新影格估計）
    def obs_intake(self, t_obs, T_WH, ok=True):
        self.obs_in = dict(t_obs=t_obs, t_recv=float(self.t), T_WH=T_WH, ok=bool(ok))

    def _rej(self, why):
        self.obs_rej[why] = self.obs_rej.get(why, 0) + 1

    def _try_replan(self, q):
        o = self.obs_in
        if o is None:
            return
        self.obs_in = None
        if not o['ok'] or o['T_WH'] is None:
            return self._rej('invalid')
        try:
            t_obs, T_near = float(o['t_obs']), np.array(o['T_WH'], float)
        except (TypeError, ValueError):
            return self._rej('malformed')
        if not math.isfinite(t_obs) or T_near.shape != (4, 4) or not np.isfinite(T_near).all():
            return self._rej('nonfinite')
        if not valid_pose(T_near):
            return self._rej('bad_pose_matrix')
        if t_obs > float(self.t) + 1e-9:
            return self._rej('future_stamp')
        if t_obs < self.look['t_arrive'] - 1e-9:
            return self._rej('before_arrival')
        age = float(self.t) - t_obs
        if age > OBS_MAX_AGE_S:
            return self._rej('stale')
        w0 = time.perf_counter()
        base = np.asarray(q[:3], float).copy()
        sol, info = PG.plan_at(base, np.asarray(q[3:9], float), T_near, self.target_fn(T_near, PG.S_PRE), self.target_fn(T_near, 0.0))
        ms = (time.perf_counter() - w0) * 1e3
        if sol is None:
            return self._stop('replan_failed', info={k: {kk: vv for kk, vv in d.items() if kk in ('why', 'joint_margin', 'env', 'env_who')} for k, d in info.items()}, replan_ms=round(ms, 1))
        q_pre = sol[0]
        cc = CLEAR(q_pre, T_near)
        if not cc['ok']:
            return self._stop('replan_collision', at={k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items()}, replan_ms=round(ms, 1))
        T2, _ = R2.bounded_T(base, np.asarray(q[3:9], float), base, q_pre[IDX])
        ms = (time.perf_counter() - w0) * 1e3
        self.T_WH, self.T_src = T_near, 'near'
        self.seg = dict(kind='world', t0=float(self.t) + D_STEPS * PHYS, T=float(T2), ba=base.copy(), bb=base.copy(), qa=np.asarray(q[3:9], float).copy(), qb=q_pre[IDX].copy(), b_world=base.copy())
        self.T_pre = np.asarray(self.target_fn(T_near, PG.S_PRE), float)
        self.replan = dict(t=round(float(self.t), 3), obs_age_s=round(age, 3), wall_ms=round(ms, 1), T_to_pre=round(float(T2), 3), base_at_replan=[round(float(v), 5) for v in base], q_pre=[round(float(v), 5) for v in q_pre[IDX]],
                           T_pre_world=self.T_pre.tolist(), pre_clear={k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items() if k != 'ok'})
        self.phase = 'TO_PRE'
        self._ev('replan', **{k: v for k, v in self.replan.items() if k not in ('T_pre_world', 'q_pre', 't')})

    # ---- 每控制拍：相位、量測、停止預測（都在求解與提交命令之前）
    def _advance(self, q):
        g = self.seg
        if self.phase == 'TO_LOOK' and self.t >= g['t0'] + g['T'] + SETTLE_S:
            if np.all(np.abs(self.u_prev[:3]) <= B.A_BASE * self.cfg.dt + 1e-12):          # 切到速度界＝0 時加速度約束仍可行
                self.phase, self.t_settle, self._n_still = 'SETTLE', float(self.t), 0
                self._ev('settle_begin', base_cmd=[round(float(v), 5) for v in self.u_prev[:3]])
            elif self.t >= g['t0'] + g['T'] + SETTLE_S + STILL['timeout_s']:
                return self._stop('base_not_settled', base_cmd=[float(v) for v in self.u_prev[:3]])
        if self.phase == 'SETTLE':
            quiet = not np.any(self.u_last[:3]) and not any(np.any(u[:3]) for u in self.pipe)
            still = self.rate['base'] <= STILL['base_mps'] and self.rate['yaw'] <= STILL['yaw_rps']
            self._n_still = self._n_still + 1 if (quiet and still) else 0
            if self._n_still >= STILL['n_ticks']:
                eb = float(np.linalg.norm(q[:2] - self.b1[:2]))
                ey = abs(R1.wrap(q[2] - self.b1[2]))
                ej = float(np.abs(q[3:9] - self.q_look).max())
                self.look = dict(t_arrive=float(self.t), base_err_m=round(eb, 5), yaw_err_rad=round(ey, 5), joint_err_rad=round(ej, 5), base=[round(float(v), 5) for v in q[:3]],
                                 base_speed=round(self.rate['base'], 6), yaw_rate=round(self.rate['yaw'], 6))
                self.hold0 = np.asarray(q[:3], float).copy()
                self._ev('arrived_look', **{k: v for k, v in self.look.items() if k != 't_arrive'})
                if eb > ARRIVE_TOL['base_m'] or ey > ARRIVE_TOL['yaw_rad'] or ej > ARRIVE_TOL['joint_rad']:
                    return self._stop('arrive_out_of_tol', base_err=eb, yaw_err=ey, joint_err=ej)
                self.phase = 'LOOK'
            elif self.t - self.t_settle > STILL['timeout_s']:
                return self._stop('base_not_settled', base_speed=self.rate['base'], yaw_rate=self.rate['yaw'], pipe_quiet=bool(quiet))
        if self.hold0 is not None:
            db, dy = float(np.linalg.norm(q[:2] - self.hold0[:2])), abs(R1.wrap(q[2] - self.hold0[2]))
            self.hold_dev = dict(base_m=max(self.hold_dev['base_m'], db), yaw_rad=max(self.hold_dev['yaw_rad'], dy))
            if db > HOLD_TOL['base_m'] or dy > HOLD_TOL['yaw_rad']:
                return self._stop('base_hold_violation', base_dev_m=db, yaw_dev_rad=dy)
        if self.phase == 'LOOK':
            self._try_replan(q)
            if self.stop is None and self.phase == 'LOOK' and self.t - self.look['t_arrive'] > LOOK_TIMEOUT_S:
                return self._stop('near_obs_timeout', rejected=dict(self.obs_rej))
        g = self.seg                                                # 重新規劃會換段：重新取（r1 假世界抓到：沿用舊段使 TO_PRE 立刻被判結束）
        if self.phase == 'TO_PRE' and self.t >= g['t0'] + g['T']:
            self.phase, self._n_end = 'END', 0
            self.end = dict(t=float(self.t))
            self._ev('end_ref_done')
        if self.phase == 'END':
            Tm = K.fk(q, 'link_tcp')
            ep, er = float(np.linalg.norm(Tm[:3, 3] - self.T_pre[:3, 3])), _rot_deg(self.T_pre, Tm)
            m = dict(tcp_m=round(ep, 5), tcp_deg=round(er, 4), arm_rps=round(self.rate['arm'], 5), base_mps=round(self.rate['base'], 6), yaw_rps=round(self.rate['yaw'], 6),
                     hold_base_m=round(float(np.linalg.norm(q[:2] - self.hold0[:2])), 5), hold_yaw_rad=round(abs(R1.wrap(q[2] - self.hold0[2])), 5))
            good = ep <= END_TOL['tcp_m'] and er <= END_TOL['tcp_deg'] and self.rate['arm'] <= END_TOL['arm_rps'] and self.rate['base'] <= STILL['base_mps'] and self.rate['yaw'] <= STILL['yaw_rps']
            self._n_end = self._n_end + 1 if good else 0
            self.end['last'] = m
            if self._n_end >= END_TOL['n_ticks']:
                self.done = True
                self.end['reached'] = dict(t=round(float(self.t), 3), **m)
                return self._stop('approach_done', **m)
            if self.t - self.end['t'] > END_TOL['wait_s']:
                return self._stop('end_not_reached', **m)
        if self.stop is None:
            self._stop_sweep(q)

    def _sweep(self, q, u, extras, key):
        """真停止管線預測（在途命令＋命令 u；extra 拍後開始減速）逐取樣核全碰撞對。回 None 或違反資訊。"""
        for e in extras:
            Q = C21.stop_traj(np.asarray(q, float), self.s, self.pipe, u, self.cfg, e)
            idx = list(range(0, len(Q), self.sweep_stride))
            if idx[-1] != len(Q) - 1:
                idx.append(len(Q) - 1)                              # 一定含終點
            for i in idx:
                if not np.isfinite(Q[i]).all():
                    return dict(extra=e, i=i, nonfinite=True)
                cc = CLEAR(Q[i], self.T_WH)
                self.mins['n_sweep_cfg'] += 1
                for k in self.mins[key]:
                    self.mins[key][k] = min(self.mins[key][k], cc[k])
                if not cc['ok']:
                    return dict(extra=e, i=i, at={k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items()})
        return None

    def _stop_sweep(self, q):
        w0 = time.perf_counter()
        bad = self._sweep(q, self.u_last, (0,), 'stop')
        self.sweep_ms.append((time.perf_counter() - w0) * 1e3)
        if bad is not None:
            self._stop('stop_sweep_violation', **bad)

    def _solve(self, q):
        w0 = time.perf_counter()
        cfg = self.cfg
        qp, sp = q.copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, cfg, 1)
        t_eff = self.t + D_STEPS * PHYS
        X_ref, xi_ref, T_seq = self._windows(t_eff)
        if self.fail_inject is not None and self.fail_inject(self.t):
            return None
        hold = self.phase in HOLD
        u_nom = None
        if self.joint_nominal and hold:
            al = np.asarray(cfg.arm_model.alpha, float)
            tau, off = PHYS / al, np.asarray(cfg.arm_model.bias, float) / al
            h = 1e-3

            def s_ref(tq):
                return self._ref(tq)[2] + tau * (self._ref(tq + h)[2] - self._ref(tq - h)[2]) / (2 * h) - off
            va, sk, u_nom = float(B.W_ARM['v_arm']), sp.copy(), []
            for k in range(N):
                u = np.clip((s_ref(t_eff + (k + 1) * DT) - sk) / DT, -va, va)
                u_nom.append(u)
                sk = sk + DT * u
            u_nom = np.array(u_nom)
        arm = dict(u_arm_nom=u_nom, mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=cfg.arm_model, T_seq=T_seq, w_p=C0.W_NEAR if self.phase in ('TO_PRE', 'END') else C0.W_FAR,
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=self.seg['kind'] == 'rel', qp_solver='gi', **B.W_ARM)
        r = (self.mpc_hold if hold else self.mpc).solve(X_now=from_xytheta(*qp[:3]), X_ref_win=X_ref, xi_ref_win=xi_ref, xi_prev=self.u_prev[:3], arm=arm, obstacles=None)
        self.solve_ms.append((time.perf_counter() - w0) * 1e3)
        ck = dict(r.arm_check or {})
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None:
            return None
        u = np.r_[np.asarray(r.u_opt, float), r.u_arm_seq[0]]
        if not np.isfinite(u).all():
            return None
        if np.abs(u[3:]).max() >= float(B.W_ARM['v_arm']) - 1e-3:
            self.n_sat[self.phase] = self.n_sat.get(self.phase, 0) + 1
        if hold and np.any(u[:3]):
            self._stop('hold_solver_base_nonzero', base_cmd=[float(v) for v in u[:3]])     # 求解器內約束應使其恆為 0；不在求解後歸零
            return None
        # 追蹤紀錄（參考在當拍；量測為當拍）
        b, _, qa = self._ref(self.t)
        Trel = K.fk(np.r_[0.0, 0.0, 0.0, qa], 'link_tcp')
        Tref = _tb(q[:3] if self.seg['kind'] == 'rel' else self.seg['b_world']) @ Trel
        Tm = K.fk(q, 'link_tcp')
        self.track.append([float(self.t), self.phase, float(np.linalg.norm(q[:2] - b[:2])), float(abs(R1.wrap(q[2] - b[2]))), float(np.linalg.norm(Tm[:3, 3] - Tref[:3, 3])), _rot_deg(Tref, Tm), float(np.abs(u[3:]).max()), float(np.hypot(u[0], u[1]))])
        return u

    def _decel(self):
        da = np.r_[B.A_BASE, np.full(6, C0.A_ARM)] * self.cfg.dt
        return self.u_last + np.clip(-self.u_last, -da, da)

    def step(self, t, q):
        """每物理步。回 dict(arm_target(6), base_twist(3)＝本步生效的底盤本體命令, phase, stopped, done)。"""
        self.t = float(t)
        q = np.asarray(q, float)
        tick = self.n_step % NPH == 0
        w_tick = time.perf_counter()
        if not np.isfinite(q).all():
            self._stop('state_nonfinite')
        if tick and self.stop is None:
            if self.q_tick is not None:
                self.rate = dict(base=float(np.linalg.norm(q[:2] - self.q_tick[:2])) / self.cfg.dt, yaw=abs(R1.wrap(q[2] - self.q_tick[2])) / self.cfg.dt, arm=float(np.abs(q[3:] - self.q_tick[3:]).max()) / self.cfg.dt)
            self.q_tick = q.copy()
            qa = q[3:]
            if self.q_anchor is not None and len(self.s_seq) >= 1:
                self.obs.update_setpoints(self.q_anchor, self.s_seq[:-1], qa)
                self.cfg.arm_model.bias = self.bias0 + np.asarray(self.cfg.arm_model.alpha, float) * np.asarray(self.obs.d, float)
            self.q_anchor, self.s_seq = qa.copy(), [self.s.copy()]
            cc = CLEAR(q, self.T_WH)
            for k in self.mins['act']:
                self.mins['act'][k] = min(self.mins['act'][k], cc[k])
            self._advance(q)                                      # 相位、量測、「現在停」預測：在求解之前；在此設停止 ⇒ 本拍不求解、不提交正常命令
        if self.stop is not None:
            if self.stopped_t is None:
                self.stopped_t = self.t
            if tick:
                self.u_last = self._decel()
        elif tick:
            u = self._solve(q)
            if u is None:
                self._stop('solver_fail')
            else:
                w0 = time.perf_counter()
                bad = self._sweep(q, u, (1, 2), 'cand')           # 候選命令生效後：下一拍停／再晚一拍停
                self.cand_ms.append((time.perf_counter() - w0) * 1e3)
                if bad is not None:
                    self._stop('candidate_stop_violation', **bad)
            if self.stop is not None:                             # 候選不提交：自上一命令開始減速（本拍求解前已核過這條「現在停」軌跡）
                self.stopped_t = self.t
                self.u_last = self._decel()
            else:
                self.u_last = u
            self.u_prev = self.u_last.copy()
        if tick:
            self.tick_ms.append((time.perf_counter() - w_tick) * 1e3)
        self.pipe.append(self.u_last.copy())
        ua = self.pipe.pop(0)
        nxt = self.s + PHYS * ua[3:]
        if np.any(nxt < self.lim_lo) or np.any(nxt > self.lim_hi):
            nxt = np.clip(nxt, self.lim_lo, self.lim_hi)
            self._stop('setpoint_limit')
        self.s = nxt
        self.s_seq.append(self.s.copy())
        if self.stopped_t is not None and self.q_last is not None and self.stop_ok is None:
            dq = float(np.abs(q[3:] - self.q_last[3:]).max()) / PHYS
            vb = float(np.hypot(*(q[:2] - self.q_last[:2]))) / PHYS
            wy = abs(R1.wrap(q[2] - self.q_last[2])) / PHYS
            empty = not np.any(self.u_last) and not any(np.any(u) for u in self.pipe)
            still = empty and dq < STOP_STILL['arm_rps'] and vb < STOP_STILL['base_mps'] and wy < STOP_STILL['yaw_rps']
            self._still_steps = self._still_steps + 1 if still else 0
            if self._still_steps * PHYS >= STOP_STILL['hold_s'] or self.t - self.stopped_t >= STOP_STILL['timeout_s']:
                self.stop_ok = dict(t=round(self.t, 3), since_stop_s=round(self.t - self.stopped_t, 3), arm_rate=dq, base_speed=vb, yaw_rate=wy, pipe_empty=bool(empty), held_s=round(self._still_steps * PHYS, 3),
                                    ok=bool(self._still_steps * PHYS >= STOP_STILL['hold_s']))
        self.q_last = q.copy()
        self.n_step += 1
        return dict(arm_target=self.s.copy(), base_twist=ua[:3].copy(), phase=self.phase, stopped=self.stop is not None, done=self.done)

    def report(self):
        tr = self.track

        def seg(ph, col):
            v = [r[col] for r in tr if r[1] == ph]
            return None if not v else dict(max=round(float(max(v)), 5), p95=round(float(np.percentile(v, 95)), 5), end=round(float(v[-1]), 5))

        def pct(a):
            a = np.array(a or [0.0])
            return dict(p50=round(float(np.percentile(a, 50)), 1), p95=round(float(np.percentile(a, 95)), 1), max=round(float(a.max()), 1))
        f4 = lambda d: {k: round(float(v), 4) for k, v in d.items()}   # noqa: E731
        return dict(stop=self.stop, done=self.done, phase=self.phase, stop_effect=self.stop_ok, events=self.events, look=self.look, replan=self.replan, end=self.end, obs_rejected=self.obs_rej, target_source=self.T_src,
                    hold_dev_max=dict(base_m=round(self.hold_dev['base_m'], 6), yaw_rad=round(self.hold_dev['yaw_rad'], 6)),
                    min_clear_actual=f4(self.mins['act']), min_clear_stop_now=f4(self.mins['stop']), min_clear_candidate=f4(self.mins['cand']), n_sweep_cfg=self.mins['n_sweep_cfg'],
                    tracking=dict(TO_LOOK=dict(base_m=seg('TO_LOOK', 2), yaw_rad=seg('TO_LOOK', 3), tcp_m=seg('TO_LOOK', 4), tcp_deg=seg('TO_LOOK', 5)), TO_PRE=dict(tcp_m=seg('TO_PRE', 4), tcp_deg=seg('TO_PRE', 5)),
                                  END=dict(tcp_m=seg('END', 4), tcp_deg=seg('END', 5)), arm_cmd_max=round(float(max((r[6] for r in tr), default=0.0)), 4), base_cmd_max=round(float(max((r[7] for r in tr), default=0.0)), 4),
                                  base_cmd_max_hold=round(float(max((r[7] for r in tr if r[1] in HOLD), default=0.0)), 6)),
                    d_hat=[round(float(x), 5) for x in self.obs.d], joint_nominal=self.joint_nominal, arm_saturation_s={k: round(v * self.cfg.dt, 2) for k, v in self.n_sat.items()},
                    timing=dict(solve_ms=pct(self.solve_ms), stop_now_sweep_ms=pct(self.sweep_ms), candidate_sweep_ms=pct(self.cand_ms), whole_tick_ms=pct(self.tick_ms), n_tick_over_period=int(sum(x > self.cfg.dt * 1e3 for x in self.tick_ms)), n_tick=len(self.tick_ms)),
                    params=dict(SETTLE_S=SETTLE_S, LOOK_TIMEOUT_S=LOOK_TIMEOUT_S, OBS_MAX_AGE_S=OBS_MAX_AGE_S, STILL=STILL, HOLD_TOL=HOLD_TOL, END_TOL=END_TOL, STOP_STILL=STOP_STILL, ARRIVE_TOL=ARRIVE_TOL, S_PRE=PG.S_PRE, FINGER_Q=FINGER_Q,
                                thresholds=dict(env=R1.ENV_M, base=R1.BASE_M, self=R1.SELF_M, bar_tool=R1.ENV_M)))
