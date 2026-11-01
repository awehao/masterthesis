#!/usr/bin/env python3
"""OBSERVE＋依量測選抓法（Codex 20261102_010259、010831；版本化子類，基線＝bottle_w1_segtarget_ctrl，不改基線與凍結檔）。
目標來源同基線：只有 known_type 不可變快照（量測 xy、支撐 z；先驗＝瓶半徑、抓持帶 0.10 m）。差別在「採用」：
  候選暫存、計畫通過才提交。合格候選（同基線的 ready／檢查 ok／新鮮／與快照一致／序號遞增）先暫存（hold），背景工作依它規劃（bottle_segtarget_plan）：
    全窗可行 ⇒ 移動抓取（同既有）；否則停車抓取（停車策略，明示；本搜尋找到的子窗內、容納 x／y／偏航停車誤差的點）；都不行 ⇒ segtarget_unreachable。
    再核整段動作（移動：漸變＋窗兩端與中點的插入／抬升；停車：回程＋行駛＋停車後關節直線靠近＋插入／抬升）的關節餘裕、自碰、量測支撐局部淨空、
    量測瓶身淨空，以及真值第一桌（拒絕式安全網：只拒絕、不修正估計）⇒ 不過 ⇒ segtarget_path_unsafe（真值網或量測淨空）／segtarget_motion_infeasible（IK／自碰／餘裕）。
  結果回來時重驗（綁定候選版本與規劃初態）：暫存後出現不一致拒收、或較新的合格快照偏離暫存 > 2 mm ⇒ segtarget_plan_candidate_changed；
    暫存起算 > hold_max ⇒ segtarget_plan_expired；停著規劃時實測底盤／手臂相對規劃初態變了 ⇒ segtarget_plan_state_changed。任何一項 ⇒ 不執行、受控停止。
OBSERVE：到既有漸變起點（x_blend）仍未提交 ⇒ 停車觀察（不依桌高判斷，只依「未提交」：沒有合格候選，或規劃仍在算）：
  1 停止點核對：以實測、在途命令與既有停止模型（_stop_rest），加參考減速距離 v²/(2·A_REF)，核停點在抓取窗前 ≥ x_guard、手臂停止構型對高度包絡淨空 ⇒ 否則 observe_stop_unsafe。
  2 停穩（實測底盤連續低速 N_STILL 步、手臂關節實測速度 < arm_still 連續 N_STILL 步）後錨定參考；有計畫在算 ⇒ 等它（有界）；否則背景線上算觀察視點（view_search，
    由實際停止位姿與目前手臂出發；共同任務區＋高度包絡，不讀桌高；工作量固定、逾時 view_timeout）⇒ 無解 ⇒ observe_view_unreachable。
  3 移動前核規劃初態沒變，手臂沿關節直線（min-jerk）到視點；到位且停穩後在視點最多等 look_wait 秒合格候選 ⇒ 沒有 ⇒ observe_no_target（不重設期限）。
  停穩逾時（settle_max）、行駛逾時（drive_max）、停在停車點前被擋（park_blocked）都受控停止。
  4 合格候選 ⇒ 暫存＋背景規劃（含從視點回新備妥的回程）⇒ 重驗 ⇒ 提交。提交前不改任務目標、不閉爪。
提交後：備妥姿態依計畫分支重選（停著提交時；接近途中提交的移動抓取沿用名目備妥）。手臂沿關節直線回備妥；
  移動 ⇒ 恢復原流程（漸變起點已過即在原地起漸變）；停車 ⇒ 底盤以參考速度剖面駛到 x_park 停穩 ⇒ 背景停車重核（實測位姿在包絡內、預抓與插入／抬升 IK 延續、
  靠近直線與插入／抬升逐點淨空；不過 ⇒ park_out_of_envelope／park_*）⇒ 手臂沿關節直線靠近到實測位姿的預抓 ⇒ 預抓／插入／閉爪／抬升時底盤速度界為 0（mpc_park，同 P 組作法）。
期限：停車提交時依計畫時程算一次期限（回程＋行駛＋重核＋靠近＋插入＋鎖定＋pad），設給基線與 V2 的凍結期限；之後不重設。
背景工作：mode='thread'（實跑，牆鐘並行，不阻塞控制迴圈）；mode='sim'（離線測試：啟動時同步算完，模擬時間 latency 後才交付，結果可重現）。逾時一律以模擬時間計。
"""
import dataclasses
import math
import threading
import time

import numpy as np

import bottle_ref as BRF
import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
import bottle_s1_geom as G
import bottle_segtarget_plan as SP
import bottle_seg3d_collect_design as CD
import bottle_seg3d_view_library as VL
import bottle_w1_segtarget_ctrl as SG
import bottle_w1_shrow_ctrl as SR
import wrist_v0_capture as W0

C3, C2 = SG.C3, SG.C2
OBSREG = dict(view_timeout=45.0, look_wait=10.0, plan_timeout=60.0, park_timeout=20.0, hold_max=30.0, cand_dev=0.002, state_dpos=0.005, state_dyaw=0.01, state_dq=0.02,
              arm_still=0.02, x_guard=0.05, expire_pad=10.0, park_tol=0.003, settle_max=8.0, drive_max=30.0)
STOP_STATES = ('STOPPING', 'VIEW', 'TO_VIEW', 'LOOK', 'PLAN', 'WAIT_PLAN', 'RETURN', 'PARK_CHECK', 'APPROACH')
MANIP = ('PREGRASP', 'INSERT', 'CLOSE', 'LIFT')                 # 停車抓取：預抓起底盤速度界為 0（離線 full 案例：PREGRASP 用導航 MPC 時底盤被拉退 5 cm）


class BgJob:
    def __init__(self, name, fn, args, t0, timeout, mode='thread', latency=1.0):
        self.name, self.t0, self.timeout, self.res, self.err, self.wall_s = name, float(t0), float(timeout), None, None, None
        self.w0, self.t_ready, self.th = time.perf_counter(), None, None
        if mode == 'thread':
            self.th = threading.Thread(target=self._run, args=(fn, args), daemon=True)
            self.th.start()
        else:
            self._run(fn, args)
            self.t_ready = self.t0 + float(latency)

    def _run(self, fn, args):
        try:
            self.res = fn(*args)
        except Exception as e:                                   # noqa: BLE001 — 背景工作的任何例外都當失敗回報，不讓執行緒默默死掉
            self.err = repr(e)
        self.wall_s = time.perf_counter() - self.w0

    def done(self, t):
        return t >= self.t_ready - 1e-9 if self.th is None else not self.th.is_alive()


def job_plan(cand, x_pre, blend_len, win, truth_boxes, base_obs, q_from, ready, plan=None, q5_sign=None):
    w0 = time.perf_counter()
    cx, cy, zs = float(cand['xy'][0]), float(cand['xy'][1]), float(cand['zs'])
    p = plan if plan is not None else SP.plan_grasp(cx, cy, zs, win=win, q5_sign=q5_sign)
    m = None
    if p['kind'] == 'moving' or (p['kind'] == 'parked' and base_obs is not None):
        m = SP.motion_check(p, cx, cy, zs, truth_boxes, x_pre, blend_len, base_obs=base_obs, q_from=q_from, ready=ready)
    return dict(plan=p, motion=m, wall_s=time.perf_counter() - w0, base0=None if base_obs is None else np.array(base_obs, float),
                q0=None if q_from is None else np.array(q_from, float), seq=cand['seq'])


def _fmt(w):
    if not w:
        return None
    out = {}
    for k, v in w.items():
        out[k] = _fmt(v) if isinstance(v, dict) else (None if not np.isfinite(v) else round(float(v), 4))
    return out


def _jT(dq):
    return max(1.0, math.ceil(1.875 * float(dq) / B.W_ARM['v_arm'] / 0.05) * 0.05)


class W1PlaceController(SG.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_OBS'

    def __init__(self, *a, obs=None, bg_mode='thread', bg_latency=1.0, **kw):
        super().__init__(*a, **kw)
        self.ob = dict(OBSREG, **(obs or {}))
        self.bg_mode, self.bg_latency = bg_mode, float(bg_latency)
        self.os, self.mode, self.job, self.hold, self.pending_plan = None, None, None, None, None
        self.n_incons, self.newer = 0, []
        self.q_meas, self.plan_park = None, None
        self.arm_path, self.x_park, self.parked, self.plan_used, self.motion_used, self.view, self.park_chk = None, None, False, None, None, None, None
        self.base_obs, self.q_obs, self.t_look0, self.t_state = None, None, None, None
        self.n_arm_still, self.expire_s_eff = 0, float(self.seg['expire_s'])
        self.x_blend0 = float(self.x_blend)
        self.ob_log, self.jobs = [], []
        self.mpc_main = self.mpc
        self.mpc_gpark = SR.GMPCShRows(dataclasses.replace(self.nav, u_min=np.zeros(3), u_max=np.zeros(3)), self)
        q0 = np.r_[0.0, 0.0, 0.0, self.q_arm_ready]
        self.T_mount = W0.urdf_chain_T(CD.URDF, 'link_eef', 'camera_color_optical_frame')
        self.T_eef_tcp = np.linalg.inv(self.K.fk(q0, 'link_eef')) @ self.K.fk(q0, 'link_tcp')
        self.view_fn = SP.view_search                            # 測試可替換（視點不可達案例）

    # ---- 工具
    def _oset(self, st, **info):
        self.os, self.t_state = st, float(self.t)
        self.ob_log.append(dict(t=round(float(self.t), 3), state=st, **info))
        self._ev('observe_state', state=st, **info)

    def _stop(self, why, **info):
        self.ob_log.append(dict(t=round(float(self.t), 3), stop=why, **info))
        self.judge._stop(self.t, why, **info)

    def _truth_boxes(self):
        return [(c_, s_) for _, c_, s_ in G.table_boxes(dict(G.S1_TABLE, top_z=float(C2.TOP_Z)))]   # 真值第一桌（安全網；isaac 腳本依場景設 C2.TOP_Z）

    def _launch(self, name, fn, args, timeout):
        self.job = BgJob(name, fn, args, self.t, timeout, self.bg_mode, self.bg_latency)
        self.jobs.append(dict(name=name, t0=round(float(self.t), 3)))

    def _q5_sign(self):
        return None if self.q_meas is None else float(np.sign(self.q_meas[7]))

    def _settled(self, q):
        return self.v_b == 0.0 and self.n_still >= S1.N_STILL and self.n_arm_still >= S1.N_STILL

    def _arm_go(self, q, qb, T=None):
        qa = np.asarray(q[3:], float).copy()
        qb = np.asarray(qb, float)
        self.arm_path = dict(t0=float(self.t), T=_jT(np.abs(qb - qa).max()) if T is None else float(T), qa=qa, qb=qb.copy())
        return self.arm_path['T']

    def _arm_done(self):
        ap = self.arm_path
        return ap is not None and self.t >= ap['t0'] + ap['T'] - 1e-9

    def _state_same(self, q, base0, q0):
        if base0 is None:
            return True
        return (float(np.linalg.norm(q[:2] - base0[:2])) <= self.ob['state_dpos'] and abs(float(q[2] - base0[2])) <= self.ob['state_dyaw']
                and float(np.abs(q[3:] - q0).max()) <= self.ob['state_dq'])

    # ---- 候選：暫存、計畫通過才提交
    def candidate_intake(self, t_recv, c):
        R = self.seg
        self.seg_log.append((float(t_recv), {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in c.items()}))
        if self.judge.stop is not None or self.w_locked:
            return
        vals = [c.get(k) for k in ('age_s', 'snap_seq', 'snap_cx', 'snap_cy', 'snap_z_support')]
        if any(v is None or not np.isfinite(v) for v in vals):
            return self._srej('nonfinite')
        if c.get('ready', 0) < 0.5:
            return self._srej('not_ready')
        if int(c['snap_ik']) != 2 or int(c['snap_local']) != 2:
            return self._srej('check_not_ok')
        if c['age_s'] > R['max_age'] or c['age_s'] < 0:
            return self._srej('stale')
        if not (c.get('snap_consistent', 0) > 0.5 and c.get('snap_dev_mm', 1e9) <= R['dev_tol_mm']):
            self.n_incons += 1
            return self._srej('inconsistent_with_snapshot')
        seq = int(c['snap_seq'])
        if seq <= self.seg_seq:
            return self._srej('seq_not_new')
        xy, zs = np.array([c['snap_cx'], c['snap_cy']], float), float(c['snap_z_support'])
        self.seg_obs.append((self.t, seq, xy.tolist(), zs))
        self.seg_seq = seq
        if self.vfix is not None:                                # 已提交：同基線，只修 xy（限速、總偏差 > 5 mm 停）
            corr = float(np.linalg.norm(xy - self.vfix['xy']))
            if corr > R['max_corr'] or abs(zs - self.seg_z) > R['max_corr']:
                self.judge._stop(self.t, 'segtarget_inconsistent', corr=corr, dz=zs - self.seg_z, seq=seq)
                return
            self.w_goal = xy
            self.seg_adopt.append((self.t, seq, xy.tolist(), self.seg_z))
            return
        if self.hold is not None:                                # 規劃中：記下較新的合格快照，提交時核是否改變
            self.newer.append(dict(t=float(self.t), seq=seq, xy=xy, zs=zs))
            return
        approaching = self.os is None and self.mode is None and self.st.phase == 'CRUISE' and not self.blending
        if not (approaching or self.os == 'LOOK'):
            return self._srej('not_planning_state')
        self.hold = dict(t=float(self.t), seq=seq, xy=xy.copy(), zs=zs, n_incons=self.n_incons, at=self.os or 'approach')
        self.newer = []
        if approaching:                                          # 接近途中：底盤繼續走，名目備妥；停車結果要停下後再核回程與行駛
            self._launch('plan', job_plan, (self.hold, self.x_pre, self.v_c * self.t_blend, self.win, self._truth_boxes(), None, None,
                                            (self.q_arm_ready.copy(), np.array(self.task.T_ready_rel, float)), None, self._q5_sign()), self.ob['plan_timeout'])
        else:
            qn = np.asarray(self.q_meas, float)
            self._launch('plan', job_plan, (self.hold, self.x_pre, self.v_c * self.t_blend, self.win, self._truth_boxes(), qn[:3].copy(), qn[3:].copy(), None, None,
                                            self._q5_sign()), self.ob['plan_timeout'])
            self._oset('PLAN', seq=seq)
        self._ev('segtarget_hold', xy=[round(float(v), 5) for v in xy], z_support=round(zs, 5), seq=seq, at=self.hold['at'])

    def _revalidate(self, res, q):
        h = self.hold
        if self.n_incons > h['n_incons']:
            return 'segtarget_plan_candidate_changed', dict(n_incons=self.n_incons - h['n_incons'])
        for nw in self.newer:
            d = max(float(np.linalg.norm(nw['xy'] - h['xy'])), abs(nw['zs'] - h['zs']))
            if d > self.ob['cand_dev']:
                return 'segtarget_plan_candidate_changed', dict(seq=nw['seq'], dev_mm=round(d * 1e3, 3))
        if self.t - h['t'] > self.ob['hold_max']:
            return 'segtarget_plan_expired', dict(age=round(float(self.t - h['t']), 3), limit=self.ob['hold_max'])
        if not self._state_same(q, res['base0'], res['q0']):
            return 'segtarget_plan_state_changed', dict(dpos=float(np.linalg.norm(q[:2] - res['base0'][:2])), dq=float(np.abs(q[3:] - res['q0']).max()))
        return None, None

    # ---- 背景工作完成
    def _job_done(self, job, q):
        info = dict(name=job.name, t0=job.t0, t_done=round(float(self.t), 3), wall_s=None if job.wall_s is None else round(job.wall_s, 2), err=job.err)
        self.jobs[-1].update(info)
        if job.err is not None:
            return self._stop(f'{job.name}_error', error=job.err)
        if job.name == 'view':
            v = job.res
            self.view = {k: v.get(k) for k in ('ok', 'why', 'n_try', 'score', 'cam', 'pitch_deg', 'T_tr', 'dq_max', 'n_cases')}
            if not v.get('ok'):
                return self._stop('observe_view_unreachable', why_view=v.get('why'), n_try=v.get('n_try'))
            if not self._state_same(q, self.base_obs, self.q_obs):
                return self._stop('observe_state_changed', at='view')
            T = self._arm_go(q, v['q'], v['T_tr'])
            self._oset('TO_VIEW', T=T, cam=[round(float(x), 3) for x in v['cam']], pitch=round(v['pitch_deg'], 1), score=list(v['score']))
            return
        if job.name == 'park':
            pc = job.res
            self.park_chk = {k: pc.get(k) for k in ('ok', 'why', 'err', 'worst', 'T_tr', 'dq_max', 'seg')}
            if not pc['ok']:
                return self._stop(pc['why'], err=pc.get('err'), seg=pc.get('seg'))
            if not self._state_same(q, self.base_obs, self.q_obs):
                return self._stop('park_state_changed')
            T = self._arm_go(q, pc['q_pre'], pc['T_tr'])
            self._oset('APPROACH', T=T, err={k: round(v, 4) for k, v in pc['err'].items()})
            return
        res = job.res                                            # plan
        p, m = res['plan'], res['motion']
        self.plan_used = {k: p.get(k) for k in ('kind', 'x_park', 'x_mid', 'margin', 'err_env', 'whole_window', 'n_cands', 'cand_x', 'n_env_checked', 'n_ik', 't_s', 'why', 'same_side')}
        self.plan_used['q5_pre'] = None if p.get('q_pre') is None else round(float(p['q_pre'][4]), 3)
        self.motion_used = None if m is None else dict({k: m.get(k) for k in ('ok', 'why', 'seg', 'n_cfg', 'segs', 'ready_offset', 'tried', 'x_grasp')}, worst=_fmt(m.get('worst')))
        if p['kind'] == 'timeout':
            return self._stop('segtarget_plan_timeout', why_plan=p.get('why'))
        if p['kind'] == 'unreachable':
            self.task.T_grasp = np.full((4, 4), np.nan)
            return self._stop('segtarget_unreachable', whole_window=p.get('whole_window'), n_cands=p.get('n_cands'))
        if res['base0'] is None and (m is None or self.os is not None):
            # 接近途中的計畫：停車結果，或回來時底盤已在停 ⇒ 停穩後以實際停止狀態重核整段（回程＋行駛／漸變＋靠近），不直接用接近途中的核對
            self.pending_plan = p
            self.ob_log.append(dict(t=round(float(self.t), 3), note='plan_recheck_after_stop', kind=p['kind'], x_park=p.get('x_park')))
            if self.os == 'WAIT_PLAN':
                self._launch_from_stop()
            return
        if not m['ok']:
            unsafe = m['why'] in ('truth_net', 'local', 'bottle')
            return self._stop('segtarget_path_unsafe' if unsafe else 'segtarget_motion_infeasible', why_motion=m['why'], seg=m.get('seg'), worst=_fmt(m.get('worst')))
        why, info = self._revalidate(res, q)
        if why is not None:
            return self._stop(why, **info)
        if res['base0'] is None and (self.blending or q[0] >= self.x_blend0 or self.st.phase != 'CRUISE'):
            return self._stop('segtarget_plan_late')             # 防呆：接近途中提交只能在漸變前（正常流程到 x_blend 就會先停下）
        self._commit(p, m, res, q)

    def _commit(self, p, m, res, q):
        h = self.hold
        self.seg_z = h['zs']
        self.task.T_grasp = self._Tg(h['xy'])
        self.vfix = dict(t=float(self.t), xy=h['xy'].copy(), spread=0.0, n=1, margin=float(p['margin']), seq=h['seq'], z_support=h['zs'], kind=p['kind'])
        self.w_goal = h['xy'].copy()
        self.seg_adopt.append((self.t, h['seq'], h['xy'].tolist(), self.seg_z))
        self.mode = p['kind']
        if res['q0'] is not None:                                # 停著提交：備妥依計畫分支重選，手臂沿關節直線回去
            self.q_arm_ready = np.array(m['q_ready'], float)
            self.task.T_ready_rel = np.array(m['T_ready_rel'], float)
            T_ret = self._arm_go(q, self.q_arm_ready)
            self._oset('RETURN', T=T_ret, kind=p['kind'])
        else:
            T_ret = 0.0
        if p['kind'] == 'parked':
            self.x_park, self.plan_park = float(p['x_park']), p
            dist = abs(self.x_park - float(q[0]))
            E = T_ret + dist / self.v_c + 2.0 * self.v_c / B.A_REF + 1.0 + self.ob['park_timeout'] + 3.0 + 1.0 + self.task.t_insert + SG.SEGREG['lock_after_insert_s'] + self.ob['expire_pad']
            self.expire_s_eff = max(self.expire_s_eff, E)
            self.v2 = dict(self.v2, expire_s=max(self.v2['expire_s'], E))
        self._ev('segtarget_freeze', xy=[round(float(v), 5) for v in h['xy']], z_support=round(h['zs'], 5), seq=h['seq'], margin=float(p['margin']), kind=p['kind'],
                 x_park=self.x_park, ready_offset=m.get('ready_offset'), expire_s=round(self.expire_s_eff, 2), plan_wall_s=round(res['wall_s'], 2))
        self.hold = None

    # ---- 狀態機
    def _launch_from_stop(self):
        p, self.pending_plan = self.pending_plan, None
        self._launch('plan', job_plan, (self.hold, self.x_pre, self.v_c * self.t_blend, self.win, self._truth_boxes(), self.base_obs.copy(), self.q_obs.copy(), None, p),
                     self.ob['plan_timeout'])
        self._oset('PLAN', note='recheck_from_stop', kind=p['kind'])

    def _enter_stop(self, q, why):
        qp, q_rest, q_next = self._stop_rest(q)
        d_ref = self.v_b ** 2 / (2.0 * B.A_REF) + abs(self.v_b) * C0.D_STEPS * C0.PHYS     # 參考減速比停止模型的煞車慢：底盤停點另以參考減速距離估
        x_rest = max(float(q_rest[0]), float(qp[0]) + d_ref)
        cfg = np.r_[x_rest, q_rest[1:3], q_rest[3:]]
        w = VL.clear_env(cfg)
        ok = x_rest <= self.x_ws - self.ob['x_guard'] and VL.ok_clear(w)
        self.x_blend = 1.0e6                                     # 不讓父類在觀察期間起漸變（有限值：連續展臂的進度函式要求有限）
        self._oset('STOPPING', why=why, x=round(float(q[0]), 4), x_rest=round(x_rest, 4), env=round(w['env'], 4), self_=round(w['self'], 4), margin=round(w['margin'], 3),
                  rej=dict(self.seg_rej))
        if not ok:
            self._stop('observe_stop_unsafe', x_rest=x_rest, env=w['env'], self_=w['self'])

    def _obs_step(self, q):
        t, x, R = float(self.t), float(q[0]), self.ob
        if self.job is not None:
            if self.job.done(t):
                job, self.job = self.job, None
                self._job_done(job, q)
                if self.judge.stop is not None:
                    return
            elif t - self.job.t0 > self.job.timeout:
                return self._stop(f'{self.job.name}_timeout', wait=round(t - self.job.t0, 3), limit=self.job.timeout)
        st = self.os
        if st is None:
            if self.mode is None and self.st.phase == 'CRUISE' and not self.blending and x >= self.x_blend0:
                self._enter_stop(q, 'plan_pending' if (self.job is not None or self.pending_plan is not None) else 'no_target')
            return
        if st in ('STOPPING', 'DRIVE') and t - self.t_state > R['settle_max' if st == 'STOPPING' else 'drive_max']:
            return self._stop(f'observe_{st.lower()}_timeout', wait=round(t - self.t_state, 3))
        if st == 'STOPPING':
            if self._settled(q):
                self.s_ref = x                                   # 錨定：停住時參考落回實測（同 S0b 收尾錨定作法）
                self.base_obs, self.q_obs = np.array(q[:3], float), np.array(q[3:], float)
                if self.pending_plan is not None:
                    self._launch_from_stop()
                elif self.job is not None:
                    self._oset('WAIT_PLAN')
                else:
                    self._launch('view', self.view_fn, (self.base_obs.copy(), self.q_obs.copy(), self.T_mount, self.T_eef_tcp), R['view_timeout'])
                    self._oset('VIEW', base=[round(float(v), 4) for v in self.base_obs])
        elif st == 'WAIT_PLAN':
            if self.job is None:
                self._stop('segtarget_plan_lost')                # 防呆：計畫回來卻沒有提交、重核或停止
        elif st == 'TO_VIEW':
            if self._arm_done() and self._settled(q):              # 到位且停穩才開始收候選（規劃初態綁定實測）
                self.t_look0 = t
                self._oset('LOOK')
            elif self._arm_done() and t - self.arm_path['t0'] - self.arm_path['T'] > R['settle_max']:
                return self._stop('observe_to_view_timeout', wait=round(t - self.t_state, 3))
        elif st == 'LOOK':
            if t - self.t_look0 > R['look_wait']:
                self._stop('observe_no_target', wait=round(t - self.t_look0, 3), rej=dict(self.seg_rej))
        elif st == 'RETURN':
            if self._arm_done():
                if self.mode == 'moving':
                    self.arm_path, self.x_blend = None, self.x_blend0   # 恢復原流程：漸變起點已過 ⇒ 父類在原地起漸變
                    self._oset('RESUME')
                    self.os = None
                else:
                    self._oset('DRIVE', x_park=self.x_park)
        elif st == 'DRIVE':
            if x > self.x_park + SP.EX:
                return self._stop('park_overshoot', x=x, x_park=self.x_park)
            if self._settled(q) and self.x_park - x >= R['park_tol'] + 0.02:
                self.n_blocked = getattr(self, 'n_blocked', 0) + 1
                if self.n_blocked * C0.PHYS > 5.0:
                    return self._stop('park_blocked', x=x, x_park=self.x_park)
            if self.x_park - x < R['park_tol'] + 0.02 and self._settled(q):
                self.s_ref = x
                self.base_obs, self.q_obs = np.array(q[:3], float), np.array(q[3:], float)
                h = self.vfix
                self._launch('park', SP.park_check, (self.plan_park, float(self.w_goal[0]), float(self.w_goal[1]), self.seg_z, self._truth_boxes(), self.base_obs.copy(),
                                                     self.q_obs.copy()), R['park_timeout'])
                self._oset('PARK_CHECK', x=round(x, 4), y=round(float(q[1]), 4), yaw=round(float(q[2]), 4), seq=h['seq'])
        elif st == 'APPROACH':
            if self._arm_done():
                self.parked = True
                self.blending = True                             # 停車後手動起漸變：預抓目標即實測位姿下的 T_pre，父類由 PREGRASP 接手
                self.d_static = np.array(self.obs.d, float)
                self.st.t_cruise0, self.st.t_pre = t - 1.0, t
                self._ev('blend_start', x=x, y=float(q[1]), parked=True)
                self._oset('GRASP')
                self.os = None

    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None:
            self._obs_step(q)
        if self.judge.stop is None and self.vfix is not None and self.t_close_req is None and self.t - self.vfix['t'] > self.expire_s_eff:
            return self._stop('segtarget_expired', age=self.t - self.vfix['t'], limit=self.expire_s_eff)
        C3.W1PlaceController._advance_inner(self, T_tcp, close_done, q)   # 跳過基線的 no_fix（改由 OBSERVE 處理）；期限上方另核

    # ---- 參考與求解
    def _rel_target(self, tau):
        ap = self.arm_path
        if ap is not None:
            s = BRF.min_jerk(min(max((tau - ap['t0']) / ap['T'], 0.0), 1.0))
            return self.K.fk(np.r_[0.0, 0.0, 0.0, ap['qa'] + (ap['qb'] - ap['qa']) * s], 'link_tcp')
        return super()._rel_target(tau)

    def _v_target(self, T_tcp, x):
        if self.st.phase == 'CARRY':
            return super()._v_target(T_tcp, x)
        if self.os == 'DRIVE':
            d = self.x_park - x
            return 0.0 if d < self.ob['park_tol'] else min(self.v_c, math.sqrt(2.0 * B.A_REF * d))
        if self.os in STOP_STATES or self.parked:
            return max(self.v_b - 2.0 * B.A_REF * B.DT, 0.0)
        return super()._v_target(T_tcp, x)

    def _in_tube(self, q):
        if self.parked and self.st.phase == 'PREGRASP':
            return super()._in_tube(q) and self.v_b == 0.0 and self.n_still >= S1.N_STILL
        return super()._in_tube(q)

    def _solve(self, q):
        if self.parked and self.st.phase in MANIP:
            cny, self.carry_no_yaw = self.carry_no_yaw, False
            self.mpc = self.mpc_gpark
            try:
                return super()._solve(q)
            finally:
                self.carry_no_yaw = cny
        if self.mpc is self.mpc_gpark:
            self.mpc = self.mpc_main                             # 放置段的 mpc_park（bottle_place_ctrl）另有其物件，不動
        return super()._solve(q)

    def step(self, t, q, **kw):
        qa = np.asarray(q, float)
        p = getattr(self, 'q_prev2', None)
        if p is not None and np.isfinite(qa).all():
            v_arm = float(np.abs(qa[3:] - p[3:]).max()) / C0.PHYS
            self.n_arm_still = self.n_arm_still + 1 if v_arm < self.ob['arm_still'] else 0
        self.q_meas = qa.copy()
        out = super().step(t, q, **kw)
        if self.parked and self.st.phase == 'CARRY':
            self.parked = False                                  # 持瓶段恢復導航（父類 CARRY 速度與持瓶路徑）
        return out

    def report(self):
        r = super().report()
        r['observe'] = dict(state=self.os, mode=self.mode, x_park=self.x_park, log=self.ob_log, jobs=self.jobs, view=self.view, plan=self.plan_used, motion=self.motion_used,
                            park_check=None if self.park_chk is None else {k: v for k, v in self.park_chk.items() if k != 'worst'},
                            park_worst=None if self.park_chk is None else _fmt(self.park_chk.get('worst')),
                            hold=None if self.hold is None else dict(self.hold, xy=self.hold['xy'].tolist()), expire_s=self.expire_s_eff, reg=self.ob, bg_mode=self.bg_mode,
                            note='候選暫存、計畫通過才提交；停車抓取＝停車策略（明示）；真值第一桌只作拒絕式安全網；觀察視點線上計算（共同任務區＋高度包絡，不讀桌高）')
        return r
