#!/usr/bin/env python3
"""K3-B 受控釋放判準核心 r5（door_k3b_release4.py〔r4，凍結 k3b r3–r5〕的版本化複本；路由測試 k3b5 發現）。
r5 唯一差異：終點收斂窗的追蹤軌跡改為「窗內樣本＋窗外最新一筆」。
  r4 只留 t ≥ t_now − 0.5 s 的樣本，卻要求跨度 ≥ 0.5 s：只有恰好有一筆樣本落在 0.5 s 前時才成立；
  任務端樣本間隔 0.06–0.08 s 時跨度常為 0.44–0.46 s，RETREAT_DONE 間歇延後數秒甚至逾時（實跑 30_1 任務端亦見 0.08 s 間隔）。
  r5 保留窗外最新一筆，使跨度 ≥ 0.5 s 一旦有足夠歷史即成立；位移以此（略長於 0.5 s 的）窗計，為保守方向。門檻不變。
以下為 r4 說明：
K3-B 受控釋放判準核心 r4（door_k3b_release3.py〔r3，凍結 k3b r2〕的版本化複本；Codex 20261026_012939）。
r4 唯一差異：終點收斂改以固定 H(q_rel) 座標中實測 TCP **三維位置**核對——0.5 s 窗口內最大點對位移 ≤ 1 mm
（r3 只看 s_meas 投影，沿 x／z 的振盪會漏檢）；進入 RETREAT_DONE 前與保持期間皆適用；姿態偏移界不變、不新增門檻。以下為 r3 說明：
K3-B 受控釋放判準核心 r3（door_k3b_release2.py〔r2，凍結 k3b r1〕的版本化複本；Codex 20261026_010158）。
r3 相對 r2：門角帶改以「自 0 起連續成立」重算（v4e 驗格點間覆蓋）；5 N 權威 tick 重核證據年齡與解除資料新鮮度、拒收未來時間證據；
終點保持逐筆核對（再動、離區即失敗，資料中斷計時重起）；沿退離軸為追蹤落後 ≤ 5 mm（候選，依 v4e 以實測 s_meas 查表推導）。以下為 r2 說明：
K3-B 受控釋放判準核心 r2（純邏輯；規格 K3B_release_clear_spec_draft2.md；Codex 20261026_000706）。
door_k3b_release.py（r1）保留供追溯；r1 的三個缺陷由本檔修正，並依 D3 改為分層：

一、ExecReleaseMonitor（執行端，逐物理步）
  - 輸入：步號、模擬時間、兩指開度、兩指—橫桿逐步接觸力（ambiguous 計入）、接觸資料有效旗標、門角、門速（差分）。
  - 負責：開度、配對力、資料有效性與連續性、確認持續窗、期限；
    並提供 mode 3 准入（新鮮的解除確認）與 5 N 恢復權威的逐步依據（靜窗）。
  - r1 缺陷修正：
    (1) 同一步重送：t 相同 ⇒ 忽略、不累積；t 不同 ⇒ 時間衝突、失敗。步號倒退或時間不遞增 ⇒ 失敗。
    (2) **先查期限再轉移**：每步先核期限；期限已過，就算本步條件剛成立也判失敗。
  - 缺步（步號跳號）⇒ 該步無效，持續窗歸零；連續無效 > 0.2 s ⇒ 失敗。

二、TaskReleaseMonitor（任務端，約 20 Hz）
  - 輸入 sample 必須帶 src_step（來源物理步）、state_age、compute_s。
    結果年齡 state_age＋compute_s > 0.2 s 或 src_step 未前進 ⇒ 本輪不用、計為停更；連續停更 > 0.2 s ⇒ 失敗。
  - 負責：配對幾何（d、d_pred）、工具四軸偏移、門角帶（含 T_pred 預測）、一般幾何淨空（含預測）、
    底盤錨點、退出近區證據（C_CLEAR）、退離終點與收斂、完成保持。
  - 執行端狀態（exec_state、t_conf、exec_step）也由 sample 帶入，任務端不自行判定解除。
  - r1 缺陷 (3) 修正：STOW_READY／COMPLETE 之後監看持續，直到呼叫 end()；之後的超力、淨空不足仍判失敗。
  - 沿退離軸（把手 y）監看改為**追蹤落後界限** |s_cmd − s_meas| ≤ y_track_max_m＝5 mm（候選；對應 0.25 s 的落後預算）。
    規格 draft-2 §4 原寫 |y − y_ref(s_cmd)| ≤ 1 mm：小於一個控制週期的退離行程（0.02 m/s × 0.05 s＝1 mm），加上求解延遲必然超出。
    門角帶以實測 s_meas 查表（向下取格），沿軸落後不影響淨空依據；離線 ey ±1 mm 對應 s 的格點量化。此變更待 Codex 裁定。
  - 本包止於退離終點（Codex D2）：
    - RETREAT_DONE＝s_meas ≥ 0.095 且運動收斂（最近 0.5 s 工具位移 ≤ 1 mm）；
    - COMPLETE＝RETREAT_DONE 後**連續新鮮資料**保持 ≥ 3 s（r2）：
      - 每筆核 s_meas ≥ 0.095 與 0.5 s 內位移 ≤ 1 mm，再動或離開終點區即失敗；
      - 資料中斷（停更、過期）使保持計時重起，不靠計時完成；
    - STOW_READY 只是旗標，表示收臂前置成立，不是已收臂。
  - 退離期間門角下界為 −1°（帶下界）；收臂前置 [−3, +2.5]° 只作旗標。

兩者失敗都閂鎖；之後一律走既有失效路徑（規格 §5）。
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 'door_k3b_release5/r5'

REG = {
    'finger_open_m': 0.0089, 'open_frac': 0.95, 'f_small_n': 0.1, 'confirm_hold_s': 0.5, 't_conf_max_s': 1.5,
    'invalid_max_s': 0.2, 'f_design_max_n': 18.0, 'dt': 0.01,
    'near_s_m': 0.02, 'near_exit_max_s': 2.0, 'pair_exit_m': 0.005, 'restore_quiet_s': 0.5,
    'lower_deg': -1.0, 't_pred_s': 0.2, 'age_max_s': 0.2, 'stale_max_s': 0.2,
    'x_max_m': 0.003, 'y_track_max_m': 0.005, 'rot_max_deg': 0.5,
    's_end_m': 0.10, 's_done_m': 0.095, 'retreat_max_s': 15.0, 'converge_win_s': 0.5, 'converge_max_m': 0.001,
    'complete_hold_s': 3.0, 'stow_lo_deg': -3.0, 'stow_hi_deg': 2.5,
    'nonfinger_m': 0.005, 'self_m': 0.005, 'base_m': 0.020, 'base_dev_m': 0.020, 'base_dev_deg': 2.0,
    'wrist_max_n': 30.0, 'unexpected_max_n': 0.1,
}
A_H = np.array([0.0, -1.0, 0.0])
# r2 帶（v4c 原始資料以「自 0 起連續成立」重算＋running min；v4e 以 0.5 mm 細格驗證格點間覆蓋）。
# 規格 draft-2 的帶取自 v4c 摘要的「成立 Δ 之最大值」，未要求自 0 連續成立：門再轉越過橫桿後配對距離又變大，
# 致 s 0.02 處誤判上界 +5°；v4d 細格覆蓋在 s 0.020～0.0245、Δ +0.5° 測到退出後配對 2.15～4.97 mm（< 5 mm）而發現。
BAND = [(0.0, 0.1), (0.0025, 0.15), (0.0225, 0.3), (0.025, 0.5), (0.03, 0.75), (0.0325, 1.0), (0.0375, 1.25), (0.04, 1.5),
        (0.0475, 2.0), (0.06, 3.0), (0.08, 4.0), (0.09, 5.0)]


def _fin(*xs):
    for x in xs:
        if isinstance(x, bool) or not isinstance(x, (int, float, np.floating, np.integer)) or not math.isfinite(float(x)):
            return False
    return True


def _vec(v, n):
    try:
        a = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return a if len(a) == n and all(math.isfinite(x) for x in a) else None


def band_upper_deg(s):
    if not _fin(s):
        return None
    s = max(0.0, float(s))
    up = BAND[0][1]
    for s0, u in BAND:
        if s + 1e-12 >= s0:
            up = u
    return up


def z_max_m(s):
    return 0.004 if s < 0.01 else 0.006 if s < 0.02 else 0.007


def tool_offsets(E, E_rel, s_cmd):
    E, E_rel = np.asarray(E, float), np.asarray(E_rel, float)
    d = E[:3, 3] - (E_rel[:3, 3] + s_cmd * A_H)
    R = E[:3, :3] @ E_rel[:3, :3].T
    rot = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    return float((E[:3, 3] - E_rel[:3, 3]) @ A_H), float(d[0]), float(d[1]), float(d[2]), rot


class _Latch:
    def _fail(self, why, t, step, **kw):
        if self.state != 'FAILED':
            self.state, self.why = 'FAILED', why
            self.t['failed'] = t
            self.events.append(dict({'t': t, 'step': step, 'to': 'FAILED', 'why': why}, **kw))
        return self.state

    def _go(self, st, t, step, **kw):
        self.state = st
        self.t[st] = t
        self.events.append(dict({'t': t, 'step': step, 'to': st}, **kw))


# ---------------------------------------------------------------- 一、執行端
class ExecReleaseMonitor(_Latch):
    def __init__(self, reg=None):
        self.r = dict(REG, **(reg or {}))
        self.state, self.why = 'IDLE', None
        self.t, self.events = {}, []
        self.last = None                       # (step, t) 最後看到的步
        self.ok_t0 = self.invalid_t0 = self.quiet_t0 = None
        self.step_conf = None
        self.last_valid_step = None
        self.n = {'steps': 0, 'invalid': 0, 'dup': 0, 'missing': 0}

    def open(self, step, t):
        if self.state != 'IDLE':
            return self._fail('open_twice', t, step)
        if type(step) is not int or not _fin(t):
            return self._fail('open_args_invalid', t, step)
        self.t['open'], self.last = float(t), (step, float(t))
        self._go('A_OPENING', t, step)
        return self.state

    def step(self, step, t, finger_q, pair_force, contacts_valid, door_q=None):
        if self.state in ('IDLE', 'FAILED'):
            return self.state
        r = self.r
        if type(step) is not int or not _fin(t):
            return self._fail('step_args_invalid', t, step)
        ls, lt = self.last
        if step == ls:                                                       # 同一步重送
            if abs(float(t) - lt) <= 1e-9:
                self.n['dup'] += 1
                return self.state                                             # 不累積
            return self._fail('time_conflict_same_step', t, step, t_prev=lt)
        if step < ls or not float(t) > lt:
            return self._fail('step_or_time_backward', t, step, prev=[ls, lt])
        missing = step != ls + 1
        self.last = (step, float(t))
        self.n['steps'] += 1
        # 先查期限再轉移（r1 缺陷 2）
        if self.state == 'A_OPENING' and float(t) - self.t['open'] > r['t_conf_max_s'] + 1e-9:
            return self._fail('confirm_timeout', t, step)
        fq, pf = _vec(finger_q, 2), _vec(pair_force, 2)
        valid = (not missing) and fq is not None and pf is not None and contacts_valid is True
        if missing:
            self.n['missing'] += 1
        if not valid:
            self.n['invalid'] += 1
            self.ok_t0 = self.quiet_t0 = None
            self.invalid_t0 = self.invalid_t0 if self.invalid_t0 is not None else float(t)
            if float(t) - self.invalid_t0 > r['invalid_max_s'] + 1e-9:
                return self._fail('data_invalid', t, step)
            return self.state
        self.invalid_t0 = None
        self.last_valid_step = step
        opened = min(fq) >= r['open_frac'] * r['finger_open_m']
        small = max(pf) <= r['f_small_n']
        if self.state == 'A_OPENING':
            if max(pf) > r['f_design_max_n']:
                return self._fail('pair_force_design_bound', t, step)
            if opened and small:
                self.ok_t0 = self.ok_t0 if self.ok_t0 is not None else float(t)
                if float(t) - self.ok_t0 >= r['confirm_hold_s'] - 1e-9:
                    self.step_conf = step
                    self._go('B_CONFIRMED', t, step, door_q=door_q)
                    self.quiet_t0 = self.ok_t0
            else:
                self.ok_t0 = None
        else:
            if not small:
                return self._fail('pair_force_after_confirm', t, step)
            if not opened:
                return self._fail('finger_closed_after_confirm', t, step)
            self.quiet_t0 = self.quiet_t0 if self.quiet_t0 is not None else float(t)
        return self.state

    def confirmed_fresh(self, step):
        """mode 3 准入：已確認，且本監看已處理到 step−1（逐步資料未斷）。"""
        return self.state == 'B_CONFIRMED' and self.last_valid_step is not None and self.last_valid_step >= step - 1

    def quiet_s(self, t):
        return 0.0 if self.quiet_t0 is None or self.state != 'B_CONFIRMED' else float(t) - self.quiet_t0

    def report(self):
        return {'version': VERSION, 'state': self.state, 'why': self.why, 't': dict(self.t), 'step_conf': self.step_conf,
                'events': self.events[:200], 'n': dict(self.n), 'reg': dict(self.r)}


# ---------------------------------------------------------------- 二、任務端
class TaskReleaseMonitor(_Latch):
    KEYS = ('door_q', 'door_qd', 's_cmd', 'nonfinger_d', 'nonfinger_d_pred', 'self_d', 'self_d_pred', 'base_d', 'base_d_pred',
            'base_dev_m', 'base_dev_yaw', 'wrist_n', 'unexpected_n', 'state_age', 'compute_s')

    def __init__(self, reg=None):
        self.r = dict(REG, **(reg or {}))
        self.state, self.why = 'IDLE', None
        self.t, self.events = {}, []
        self.q_rel = self.E_rel = None
        self.last_src = None
        self.stale_t0 = None
        self.track = []                        # (t, s_meas, p3)：收斂判斷（r4：p3＝H(q_rel) 座標中實測 TCP 三維位置）
        self.hold_t0 = None                    # r2：終點保持計時起點（資料中斷即重起）
        self.flags = {'stow_ready': False}
        self.clear_evidence = None
        self.ended = False
        self.n = {'samples': 0, 'stale': 0, 'dup': 0}
        self.worst = {'band_margin_deg': math.inf, 'pair_d_min_m': math.inf, 'nonfinger_min_m': math.inf}

    def open(self, t, src_step, q_rel, E_rel):
        if self.state != 'IDLE':
            return self._fail('open_twice', t, src_step)
        E = np.asarray(E_rel, float) if E_rel is not None else None
        if not _fin(t, q_rel) or type(src_step) is not int or E is None or E.shape != (4, 4) or not np.all(np.isfinite(E)):
            return self._fail('open_args_invalid', t, src_step)
        self.q_rel, self.E_rel = float(q_rel), E
        self.t['open'] = float(t)
        self._go('ARMED', t, src_step, q_rel=q_rel)
        return self.state

    def end(self, t):
        """本段真正結束（呼叫端在受控收尾時呼叫）；之後不再監看。"""
        self.ended = True
        self.events.append({'t': t, 'to': 'END', 'state': self.state})

    def step(self, x):
        if self.state in ('IDLE', 'FAILED') or self.ended:
            return self.state
        r = self.r
        t = x.get('t') if isinstance(x, dict) else None
        sst = x.get('src_step') if isinstance(x, dict) else None
        if not _fin(t):
            return self.state
        self.n['samples'] += 1
        # 先查期限（以本輪時刻；不論資料是否可用）
        dl = self._deadlines(t, sst, x.get('exec_state') if isinstance(x, dict) else None)
        if dl:
            return self.state
        fresh = (isinstance(x, dict) and type(sst) is int and all(_fin(x.get(k)) for k in self.KEYS)
                 and x['state_age'] >= 0 and x['state_age'] + x['compute_s'] <= r['age_max_s']
                 and (self.last_src is None or sst > self.last_src))
        if isinstance(x, dict) and type(sst) is int and self.last_src is not None and sst == self.last_src:
            self.n['dup'] += 1                                                # 同一來源重送：不累積
        if fresh:
            for k in ('pair_d', 'pair_d_pred'):
                if _vec(x.get(k), 2) is None:
                    fresh = False
            try:
                E = np.asarray(x.get('E'), float)
                fresh = fresh and E.shape == (4, 4) and bool(np.all(np.isfinite(E)))
            except (TypeError, ValueError):
                fresh = False
        if not fresh:
            self.n['stale'] += 1
            if self.state == 'RETREAT_DONE':
                self.hold_t0 = None                                               # r2：資料中斷 ⇒ 保持計時重起，不靠計時完成
            self.stale_t0 = self.stale_t0 if self.stale_t0 is not None else float(t)
            if float(t) - self.stale_t0 > r['stale_max_s'] + 1e-9:
                return self._fail('task_data_stale', t, sst)
            return self.state
        self.stale_t0 = None
        self.last_src = sst
        # 一般保護
        for cond, why in ((x['wrist_n'] > r['wrist_max_n'], 'wrist_force'), (x['unexpected_n'] > r['unexpected_max_n'], 'unexpected_contact'),
                          (min(x['nonfinger_d'], x['nonfinger_d_pred']) < r['nonfinger_m'], 'nonfinger_clearance'),
                          (min(x['self_d'], x['self_d_pred']) < r['self_m'], 'self_clearance'),
                          (min(x['base_d'], x['base_d_pred']) < r['base_m'], 'base_clearance'),
                          (x['base_dev_m'] > r['base_dev_m'] or abs(math.degrees(x['base_dev_yaw'])) > r['base_dev_deg'], 'base_anchor')):
            if cond:
                return self._fail(why, t, sst)
        self.worst['nonfinger_min_m'] = min(self.worst['nonfinger_min_m'], x['nonfinger_d'])
        s_meas, dx, dy, dz, rot = tool_offsets(x['E'], self.E_rel, x['s_cmd'])
        if abs(dx) > r['x_max_m'] or abs(dy) > r['y_track_max_m'] or abs(dz) > z_max_m(max(s_meas, 0.0)) or rot > r['rot_max_deg']:
            return self._fail('tool_offset', t, sst, off=[dx, dy, dz, rot])
        dq = math.degrees(x['door_q'] - self.q_rel)
        dq_pred = dq + math.degrees(max(x['door_qd'], 0.0)) * r['t_pred_s']
        up = band_upper_deg(s_meas)
        if dq < r['lower_deg'] or dq_pred > up:
            return self._fail('door_band', t, sst, dq=dq, dq_pred=dq_pred, up=up, s=s_meas)
        self.worst['band_margin_deg'] = min(self.worst['band_margin_deg'], up - dq_pred)
        es = x.get('exec_state')
        pd = min(min(x['pair_d']), min(x['pair_d_pred']))
        if es in ('B_CONFIRMED',):
            self.worst['pair_d_min_m'] = min(self.worst['pair_d_min_m'], min(x['pair_d']))
        # 段別（任務端只看幾何；解除由執行端判定）
        if self.state == 'ARMED':
            if es == 'B_CONFIRMED':
                self._go('NEAR', t, sst, t_conf=x.get('t_conf'))
        if self.state == 'NEAR':
            if pd < 0.0:
                return self._fail('pair_penetration_near', t, sst, pair=pd)
            if s_meas >= r['near_s_m'] and pd >= r['pair_exit_m']:
                self._go('C_CLEAR', t, sst, s_meas=s_meas, pair=pd)
                self.clear_evidence = {'src_step': sst, 't': float(t), 's_meas': s_meas, 'pair_d_min': pd, 'band_ok': True,
                                       'dq': dq, 'dq_pred': dq_pred}
        elif self.state in ('C_CLEAR', 'RETREAT_DONE', 'COMPLETE'):
            if pd < r['pair_exit_m']:
                return self._fail('pair_clearance_after_exit', t, sst, pair=pd)
            self.clear_evidence = {'src_step': sst, 't': float(t), 's_meas': s_meas, 'pair_d_min': pd, 'band_ok': True,
                                   'dq': dq, 'dq_pred': dq_pred}
            self.track.append((float(t), s_meas, np.asarray(x['E'], float)[:3, 3].copy()))
            _cut = float(t) - r['converge_win_s']                                 # r5：窗內樣本＋窗外最新一筆
            _old = [p for p in self.track if p[0] <= _cut + 1e-9]
            self.track = _old[-1:] + [p for p in self.track if p[0] > _cut + 1e-9]
            span = (self.track[-1][0] - self.track[0][0]) if len(self.track) > 1 else 0.0
            P3 = np.array([p[2] for p in self.track])
            move = float(max(np.linalg.norm(P3[i] - P3[j]) for i in range(len(P3)) for j in range(i, len(P3))))   # r4：三維最大點對位移
            if self.state == 'C_CLEAR' and s_meas >= r['s_done_m'] and span >= r['converge_win_s'] - 1e-6 and move <= r['converge_max_m']:
                self._go('RETREAT_DONE', t, sst, s_meas=s_meas)
                self.hold_t0 = float(t)
            elif self.state == 'RETREAT_DONE':                                    # r2：保持期間逐筆核對收斂／保持條件
                if s_meas < r['s_done_m'] - 1e-9:
                    return self._fail('end_hold_left_zone', t, sst, s=s_meas)
                if move > r['converge_max_m'] + 1e-9:
                    return self._fail('end_hold_moved', t, sst, move=move)
                if self.hold_t0 is None:
                    self.hold_t0 = float(t)
                if float(t) - self.hold_t0 >= r['complete_hold_s'] - 1e-9:
                    self._go('COMPLETE', t, sst, hold_from=self.hold_t0)
            if self.state in ('RETREAT_DONE', 'COMPLETE'):
                self.flags['stow_ready'] = r['stow_lo_deg'] <= dq and dq_pred <= r['stow_hi_deg']
        return self.state

    def _deadlines(self, t, sst, es):
        r = self.r
        if self.state == 'NEAR' and float(t) - self.t['NEAR'] > r['near_exit_max_s'] + 1e-9:
            return self._fail('near_exit_timeout', t, sst)
        if self.state in ('ARMED', 'NEAR', 'C_CLEAR') and float(t) - self.t['open'] > r['t_conf_max_s'] + r['retreat_max_s'] + 1e-9:
            return self._fail('retreat_timeout', t, sst)
        if self.state == 'ARMED' and es == 'FAILED':
            return self._fail('exec_monitor_failed', t, sst)
        return None

    def report(self):
        return {'version': VERSION, 'state': self.state, 'why': self.why, 't': dict(self.t), 'q_rel': self.q_rel, 'flags': dict(self.flags),
                'events': self.events[:300], 'n': dict(self.n), 'worst': dict(self.worst), 'reg': dict(self.r), 'ended': self.ended}


# ---------------------------------------------------------------- 三、5 N 恢復權威（執行端）
class EffortRestoreAuthority:
    """請求 {run_id, req_seq, mode_seq, evidence{src_step, t, s_meas, pair_d_min, band_ok}} → 'pending'／'reject:<why>'。
    執行端每步 tick() 核對後在物理步提交；讀回由呼叫端回報 confirm(readback_ok)。中止優先。"""

    def __init__(self, run_id, reg=None, deadline_s=0.5, evidence_age_max_s=0.2):
        self.r = dict(REG, **(reg or {}))
        self.run_id, self.deadline_s, self.ev_age_max = str(run_id), float(deadline_s), float(evidence_age_max_s)
        self.last_req_seq = 0
        self.pending = None
        self.done = None
        self.aborted = None
        self.events = []

    def request(self, msg, step, t, mode_seq):
        def rej(why):
            self.events.append({'t': t, 'step': step, 'reject': why, 'req': msg})
            return f'reject:{why}'
        if self.aborted is not None:
            return rej('after_abort')
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id:
            return rej('wrong_run_or_malformed')
        rs = msg.get('req_seq')
        if type(rs) is not int or rs <= self.last_req_seq:
            return rej('req_seq_not_increasing')
        if self.done is not None or self.pending is not None:
            return rej('already_pending_or_done')
        if msg.get('mode_seq') != mode_seq or mode_seq != 3:
            return rej('mode_not_retreat')
        ev = msg.get('evidence') or {}
        if type(ev.get('src_step')) is not int or not _fin(ev.get('t'), ev.get('s_meas'), ev.get('pair_d_min')) or ev.get('band_ok') is not True:
            return rej('evidence_malformed')
        if float(ev['t']) > float(t) + 1e-6 or ev['src_step'] > step:                     # r2：未來時間或未來步一律拒收
            return rej('evidence_future')
        if float(t) - float(ev['t']) > self.ev_age_max + 1e-9:
            return rej('evidence_stale')
        if ev['s_meas'] < self.r['near_s_m'] or ev['pair_d_min'] < self.r['pair_exit_m']:
            return rej('evidence_not_clear')
        self.last_req_seq = rs
        self.pending = {'req': dict(msg), 'step': step, 't': float(t)}
        self.events.append({'t': t, 'step': step, 'pending': rs})
        return 'pending'

    def tick(self, step, t, exec_mon, mode_seq, fingers_ok):
        """回傳 'apply'（本步設 5 N）、None 或 'reject:<why>'。"""
        if self.pending is None or self.aborted is not None:
            return None
        p = self.pending
        if float(t) - p['t'] > self.deadline_s + 1e-9:
            self.pending = None
            self.events.append({'t': t, 'step': step, 'reject': 'deadline'})
            return 'reject:deadline'
        if mode_seq != 3 or exec_mon.state != 'B_CONFIRMED' or not fingers_ok:
            self.pending = None
            self.events.append({'t': t, 'step': step, 'reject': 'exec_state'})
            return 'reject:exec_state'
        ev = p['req'].get('evidence') or {}
        if float(t) - float(ev['t']) > self.ev_age_max + 1e-9:                          # r2：動作當下重核證據年齡（不只收請求時）
            self.pending = None
            self.events.append({'t': t, 'step': step, 'reject': 'evidence_stale_at_tick'})
            return 'reject:evidence_stale_at_tick'
        if not exec_mon.confirmed_fresh(step):                                            # r2：解除資料須逐步新鮮
            self.pending = None
            self.events.append({'t': t, 'step': step, 'reject': 'release_data_not_fresh'})
            return 'reject:release_data_not_fresh'
        if exec_mon.quiet_s(t) < self.r['restore_quiet_s'] - 1e-9:
            return None                                                       # 等靜窗（仍在期限內）
        self.done = {'req_seq': p['req']['req_seq'], 'switch_step': step, 't': float(t), 'readback_ok': None}
        self.pending = None
        self.events.append({'t': t, 'step': step, 'apply': self.done['req_seq']})
        return 'apply'

    def confirm(self, readback_ok, step, t):
        if self.done is not None and self.done['readback_ok'] is None:
            self.done['readback_ok'] = bool(readback_ok)
            self.events.append({'t': t, 'step': step, 'readback_ok': bool(readback_ok)})

    def abort(self, why, step, t):
        if self.aborted is None:
            self.aborted = {'why': why, 'step': step, 't': t}
            self.pending = None

    def ack(self):
        return {'run_id': self.run_id, 'last_req_seq': self.last_req_seq, 'pending': self.pending is not None,
                'done': self.done, 'aborted': self.aborted}
