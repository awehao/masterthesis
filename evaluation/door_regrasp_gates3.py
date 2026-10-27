#!/usr/bin/env python3
"""r3（Codex r53 §1；door_regrasp_gates2.py 保留）：**接受恢復資料之前先核既有無效期限**——
  r2 只在跳步／時間不一致時先核；逐步送來無效、再恢復時，_hold 直接清除 invalid_t0（反例：0.01 有效 → 0.02–0.21 無效 → 0.22 恢復，
  距最後有效 0.21 s > 0.2 卻未 FAILED）。r3：_hold 收到有效樣本時，若 invalid_t0 存在且 t − invalid_t0 > invalid_max_s ⇒ FAILED data_invalid（不清計時）。
  Gate3Release 同規則：上一步無效、本步有效時先核 t − last_valid_t。
以下為 r2 說明：
r2（Codex r52 §1–§4；door_regrasp_gates1.py 保留供追溯）：
  (1) 接觸紀錄綁定當步：link_class_force(rec, step, t) 要求 rec['step']＝當步、|rec['t'] − t| ≤ 1e-6；缺欄、錯步、未來或過期 ⇒ 無效。
  (2) 無效／缺步立即撤銷確認：CONFIRMED 遇無效 ⇒ 回 WAITING（記 revoked），恢復後須重新完成整個確認窗；confirmed_fresh(step) 嚴格型別（int、非 bool），
      只接受已處理的當步或下一步（last_valid_step ∈ {step−1, step}），不接受任意過去步號。HoldConfirm 同規則；確認後再因條件不成立失去仍 FAILED。
  (3) 時間缺口：相鄰紀錄須滿足 t − t_prev ＝ (step − step_prev)·dt（dt＝0.01，容差 1e-6）；不符或跳號 ⇒ 缺口，無效計時自**最後一筆有效樣本時刻**起算
      （先核缺口／期限再處理本步，不先清除無效計時）；缺口 > invalid_max_s 0.2 ⇒ FAILED data_invalid。
  (4) Gate3 改用 Gate3Release（ExecReleaseMonitor r5 子類，原模組不改）：同 (1)(3) 的步／時規則由呼叫端以 contacts_valid＝紀錄綁定結果帶入，
      另加准入窗——B_CONFIRMED 後自最後一次無效起連續有效 ≥ confirm_hold_s 0.5 才可准入；confirmed_fresh 同 (2) 的嚴格步號。
  (5) S_EXIT 只是進度條件，**不代替實測配對淨空**：切換為一般手指規則（請求 mode 4）前，任務端另核當時兩指對外側橫桿距離的預測值
      （v3 監看後端 finger_bar_out，扣 v_floor·T_pred 與 ds_err）≥ d_min＋warn（door_regrasp_policy2）。
  數值出處（不再總稱「只有 S_EXIT 是新登錄」）：
    既有登錄：invalid_max_s 0.2、converge 0.5 s／1 mm、f_small 0.1、unexpected 0.1、retreat_max 15 s、base_dev 0.020 m／2.0°（door_k3b_release5.REG）；
             grip_tol 2 mm、grip_settle 0.5 s、contact_abort 0.1 N、contact_min 0.5 N、attach_hold 2.0 s、drift 5 mm／2°（bench_6 task config）；
    候選（本檔提出，待實跑或另證）：S_EXIT 0.02 m（離線 v4 推導）、s_done 0.055 m（目標 0.06 的 −5 mm）、
             arm_stow_tol 0.05 rad（**借用 joint_margin 數字，非已有收臂完成容差依據**）、arm_win 0.001 rad（借用舊 HELD 手臂門檻量級）；
    推導：t_hold_max 4.5 s＝close_ramp 1.0＋grip_settle 0.5＋attach_hold 2.0＋1.0 餘裕（餘裕 1.0 為候選）。
以下為 r1 說明：
換抓模式權威的執行端准入（gate 3–6）與短保持確認（純邏輯，逐物理步；Codex r49–r50；規格 REGRASP_inner_contact_prep_draft2.md §3、§8）。
全部門檻取既有登錄值（bench_6 task config、door_k3b_release5.REG），本檔不新造放寬值；唯一新登錄是離開外側橫桿的退開量 S_EXIT_M（下述離線依據）。
共同規則：每步輸入須有效（步號＋1 遞增、時間遞增、v3 接觸紀錄 status ok 且 complete、數值有限）；無效 ⇒ 持續窗歸零（不跨缺口累積），
  連續無效 > invalid_max_s（0.2 s）⇒ FAILED（閂鎖）；confirmed_fresh(step) 只在 CONFIRMED 且已處理到 step−1 時為真（比照 ExecReleaseMonitor）。
接觸力一律取 door_contact_v3 逐點分類的**法向力大小**，依機器人部位（手指）分別加總；ambiguous 計入「非指定」。

Gate3（mode 3 RETREAT_WB 准入）＝ door_k3b_release5.ExecReleaseMonitor（不改）：張爪開度 ≥ 0.95×0.0089、兩指配對力 ≤ 0.1 N 連續 0.5 s、期限 1.5 s。
  配對力＝該指對 bar_out＋post_out0／1＋ambiguous 的法向力和（外側把手全部形狀；ambiguous 保守計入）。
Gate4（mode 4 STOW_MOVE 准入；退開完成且離開外側把手留有餘裕）：
  s_meas＝(p_TCP,0 − p_TCP)·z_0（z_0＝解除時實測工具 z，固定；p_TCP,0＝解除時 TCP）；
  s_meas ≥ S_DONE_M（0.055，目標 0.06）且 ≥ S_EXIT_M（0.02）；0.5 s 窗內 TCP 位移 ≤ 1 mm（converge_win_s／converge_max_m）；
  兩指對外側把手（bar_out、post_out*、ambiguous）力 ≤ 0.1 N；上列條件（含已滿 0.5 s 的收斂窗）再**連續成立 0.5 s** 才 CONFIRMED（到位後約 1 s）；
  期限 retreat_max_s 15 s（自 start 起）。
  S_EXIT_M 依據：離線 release_bench6_v4 沿工具 −z 退開，手指—外側橫桿距離 s 0.01→8.2 mm、0.02→13.2 mm、0.06→47.2 mm；
  切換到一般手指規則所需 ≈ 5＋1（warn）＋0.002×0.12 m ≈ 6.24 mm ⇒ 取 0.02 m（13 mm）。
Gate5（mode 5 APPROACH_INNER 准入；到達門洞入口並收臂）：
  底盤位置對 B1 偏差 ≤ base_dev_m 0.020、偏航 ≤ base_dev_deg 2.0°；手臂各軸對 STOW ≤ ARM_STOW_TOL 0.05 rad（＝joint_margin）；
  0.5 s 窗內底盤位移 ≤ 1 mm、手臂窗內位移 ≤ 0.001 rad；任何部位對門任何類別接觸力 ≤ 0.1 N（unexpected_max_n）；全部條件再連續成立 0.5 s。
Gate6（mode 6 CLOSE_INNER 准入）：TCP 對 H_in(q_meas)·T_HG（當時實測門角的內側把手名目抓取位姿）位置 ≤ grip_tol_m 2 mm、姿態 ≤ 2°，
  連續 grip_settle_s 0.5 s；期間兩指對任何類別接觸力 ≤ contact_abort_n 0.1 N（閉爪前意外碰桿保護；r49 不允許輕觸）、非手指部位無接觸。
HoldConfirm（mode 6 計時子狀態；執行端逐步）：閉爪提交後，兩指各自 bar_in 法向力 ≥ contact_min_n 0.5 N，且兩指對其他類別（bar_out、post*、panel、ambiguous）
  ≤ 0.1 N；以第一個成立步的實測 TCP 對內側把手相對位姿為基準 E0（之後不重設），漂移 ≤ drift_abort_m 5 mm／2°；連續 attach_hold_s 2.0 s ⇒ CONFIRMED；
  閉爪後 t_hold_max_s（close_ramp_s 1.0＋grip_settle_s 0.5＋attach_hold_s 2.0＋1.0 餘裕＝4.5 s）內未成立 ⇒ FAILED hold_timeout。只證明短保持。
"""
from __future__ import annotations

import math

import numpy as np

import door_k3b_release5 as KR5

VERSION = 'door_regrasp_gates3/r3'
REG = {
    'invalid_max_s': 0.2, 'converge_win_s': 0.5, 'converge_max_m': 0.001, 'quiet_s': 0.5, 'f_small_n': 0.1, 'unexpected_max_n': 0.1,
    's_done_m': 0.055, 's_exit_m': 0.02, 'retreat_max_s': 15.0, 'base_dev_m': 0.020, 'base_dev_deg': 2.0, 'arm_stow_tol': 0.05,
    'arm_win_rad': 0.001, 'grip_tol_m': 0.002, 'grip_rot_rad': math.radians(2.0), 'grip_settle_s': 0.5, 'contact_abort_n': 0.1,
    'contact_min_n': 0.5, 'attach_hold_s': 2.0, 'drift_m': 0.005, 'drift_rad': math.radians(2.0), 't_hold_max_s': 4.5,
    'dt': 0.01, 'time_tol': 1e-6,
}
FINGERS = ('uflite_finger1', 'uflite_finger2')
OUTER = ('bar_out', 'post_out0', 'post_out1', 'ambiguous')
ALL = ('bar_out', 'bar_in', 'post_out0', 'post_out1', 'post_in0', 'post_in1', 'panel', 'ambiguous')


def link_class_force(rec, step, t, tol=1e-6):
    """v3 紀錄 → {(部位, 類別): 法向力大小和}；紀錄不可用或**不屬於當步**（step／t 不符、缺欄）⇒ None。"""
    if not isinstance(rec, dict) or rec.get('status') != 'ok' or rec.get('complete') is not True:
        return None
    rs, rt = rec.get('step'), rec.get('t')
    if type(rs) is not int or rs != step or not isinstance(rt, (int, float)) or isinstance(rt, bool) or not math.isfinite(rt) \
            or abs(float(rt) - float(t)) > tol:
        return None
    out = {}
    for p in rec.get('points', []):
        if p.get('class') == 'invalid':
            return None
        f = p.get('f')
        if f is None or not math.isfinite(f):
            return None
        k = (p.get('robot_link'), p.get('class'))
        out[k] = out.get(k, 0.0) + float(f)
    return out


def finger_force(lcf, finger, classes):
    return sum(v for (l, c), v in lcf.items() if l == finger and c in classes)


def nonfinger_force(lcf):
    return sum(v for (l, c), v in lcf.items() if l not in FINGERS)


def rot_angle(Ra, Rb):
    return math.acos(max(-1.0, min(1.0, (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2)))


class _Gate:
    name = 'gate'

    def __init__(self, reg=None):
        self.r = dict(REG, **(reg or {}))
        self.state, self.why = 'IDLE', None
        self.t, self.events = {}, []
        self.last = None
        self.ok_t0 = self.invalid_t0 = None
        self.last_valid_step = self.last_valid_t = None
        self.step_conf = None
        self.n = {'steps': 0, 'invalid': 0, 'dup': 0, 'missing': 0, 'time_gap': 0, 'revoked': 0}

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

    def start(self, step, t, **ctx):
        if self.state != 'IDLE':
            return self._fail('start_twice', t, step)
        if type(step) is not int or not (isinstance(t, (int, float)) and math.isfinite(t)):
            return self._fail('start_args_invalid', t, step)
        self.t['start'], self.last = float(t), (step, float(t))
        self.last_valid_t = float(t)
        self.ctx = ctx
        self._go('WAITING', t, step)
        return self.state

    def _seq(self, step, t):
        """步號／時間規則（同 ExecReleaseMonitor）；回傳 (繼續?, 缺步?)。"""
        if self.state in ('IDLE', 'FAILED'):
            return False, False
        if type(step) is not int or not (isinstance(t, (int, float)) and math.isfinite(t)):
            self._fail('step_args_invalid', t, step)
            return False, False
        ls, lt = self.last
        if step == ls:
            if abs(float(t) - lt) <= 1e-9:
                self.n['dup'] += 1
                return False, False
            self._fail('time_conflict_same_step', t, step)
            return False, False
        if step < ls or not float(t) > lt:
            self._fail('step_or_time_backward', t, step)
            return False, False
        gap = abs((float(t) - lt) - (step - ls) * self.r['dt']) > self.r['time_tol'] or step != ls + 1
        self.last = (step, float(t))
        self.n['steps'] += 1
        if gap:
            self.n['time_gap' if step == ls + 1 else 'missing'] += 1
            self.invalid_t0 = self.invalid_t0 if self.invalid_t0 is not None else self.last_valid_t      # 自最後有效樣本起算
            if float(t) - self.invalid_t0 > self.r['invalid_max_s'] + 1e-9:                          # 先核缺口期限
                self._fail('data_invalid', t, step, gap_from=self.invalid_t0)
                return False, True
        return True, gap

    def _invalid(self, step, t):
        self.n['invalid'] += 1
        self.ok_t0 = None
        self._reset_window()
        if self.state == 'CONFIRMED':                                     # r2：無效即撤銷確認
            self.state = 'WAITING'
            self.n['revoked'] += 1
            self.events.append({'t': t, 'step': step, 'to': 'WAITING', 'why': 'revoked_invalid'})
            self._on_revoke(step, t)
        self.invalid_t0 = self.invalid_t0 if self.invalid_t0 is not None else (self.last_valid_t if self.last_valid_t is not None else float(t))
        if float(t) - self.invalid_t0 > self.r['invalid_max_s'] + 1e-9:
            return self._fail('data_invalid', t, step)
        return self.state

    def _reset_window(self):
        pass

    def _on_revoke(self, step, t):
        pass

    def _hold(self, ok, step, t):
        if self.invalid_t0 is not None and float(t) - self.invalid_t0 > self.r['invalid_max_s'] + 1e-9:   # r3：恢復前先核期限
            self._fail('data_invalid', t, step, gap_from=self.invalid_t0)
            return self.state
        self.invalid_t0 = None
        self.last_valid_step, self.last_valid_t = step, float(t)
        if self.state == 'CONFIRMED':
            if not ok:
                self._go('WAITING', t, step, why='lost')
                self.ok_t0 = None
            return self.state
        if ok:
            self.ok_t0 = self.ok_t0 if self.ok_t0 is not None else float(t)
            if float(t) - self.ok_t0 >= self.hold_s() - 1e-9:
                self.step_conf = step
                self._go('CONFIRMED', t, step)
        else:
            self.ok_t0 = None
        return self.state

    def hold_s(self):
        return self.r['quiet_s']

    def confirmed_fresh(self, step):
        if type(step) is not int or self.last_valid_step is None:
            return False
        return self.state == 'CONFIRMED' and self.last_valid_step in (step - 1, step)

    def report(self):
        return {'version': VERSION, 'gate': self.name, 'state': self.state, 'why': self.why, 't': dict(self.t), 'step_conf': self.step_conf,
                'events': self.events[:200], 'n': dict(self.n), 'reg': dict(self.r)}


class _Window:
    """時間窗內最大點對位移（窗外保留最新一筆，同 door_k3b_release5 r5 的保守做法）。"""

    def __init__(self, win):
        self.win, self.buf = float(win), []

    def reset(self):
        self.buf = []

    def add(self, t, p):
        self.buf.append((float(t), np.asarray(p, float)))
        while len(self.buf) >= 2 and self.buf[1][0] <= t - self.win + 1e-12:
            self.buf.pop(0)

    def span_ok(self, t):
        return bool(self.buf) and t - self.buf[0][0] >= self.win - 1e-9

    def max_disp(self):
        P = np.array([p for _, p in self.buf])
        if len(P) < 2:
            return 0.0
        return float(max(np.max(np.linalg.norm(P - P[i], axis=1)) for i in range(len(P))))


class Gate4Exit(_Gate):
    name = 'gate4_exit'

    def start(self, step, t, T_tcp0=None):
        T = np.asarray(T_tcp0, float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            return self._fail('tcp0_invalid', t, step)
        r = super().start(step, t)
        self.p0, self.z0 = T[:3, 3].copy(), T[:3, 2].copy()
        self.win = _Window(self.r['converge_win_s'])
        return r

    def _reset_window(self):
        self.win.reset()

    def step(self, step, t, T_tcp, rec):
        go, missing = self._seq(step, t)
        if not go:
            return self.state
        if self.state != 'CONFIRMED' and float(t) - self.t['start'] > self.r['retreat_max_s'] + 1e-9:
            return self._fail('retreat_timeout', t, step)
        T = np.asarray(T_tcp, float) if T_tcp is not None else None
        lcf = link_class_force(rec, step, t, self.r['time_tol'])
        if missing or lcf is None or T is None or T.shape != (4, 4) or not np.all(np.isfinite(T)):
            return self._invalid(step, t)                                    # 缺步計數在 _seq（r2）
        p = T[:3, 3]
        s = float((self.p0 - p) @ self.z0)
        self.win.add(t, p)
        quiet = all(finger_force(lcf, f, OUTER) <= self.r['f_small_n'] for f in FINGERS) and nonfinger_force(lcf) <= self.r['unexpected_max_n']
        conv = self.win.span_ok(t) and self.win.max_disp() <= self.r['converge_max_m']
        ok = s >= self.r['s_done_m'] and s >= self.r['s_exit_m'] and conv and quiet
        self.last_s = s
        return self._hold(ok, step, t)


class Gate5Arrive(_Gate):
    name = 'gate5_arrive'

    def start(self, step, t, base_goal=None, arm_stow=None):
        bg, st = np.asarray(base_goal, float), np.asarray(arm_stow, float)
        if bg.shape != (3,) or st.shape != (6,) or not np.all(np.isfinite(bg)) or not np.all(np.isfinite(st)):
            return self._fail('goal_invalid', t, step)
        r = super().start(step, t)
        self.bg, self.stow = bg, st
        self.wb, self.wa = _Window(self.r['converge_win_s']), _Window(self.r['converge_win_s'])
        return r

    def _reset_window(self):
        self.wb.reset()
        self.wa.reset()

    def step(self, step, t, base, arm_q, rec):
        go, missing = self._seq(step, t)
        if not go:
            return self.state
        lcf = link_class_force(rec, step, t, self.r['time_tol'])
        try:
            b, q = np.asarray(base, float).reshape(3), np.asarray(arm_q, float).reshape(6)
            okv = np.all(np.isfinite(b)) and np.all(np.isfinite(q))
        except (TypeError, ValueError):
            okv = False
        if missing or lcf is None or not okv:
            return self._invalid(step, t)
        self.wb.add(t, b[:2])
        self.wa.add(t, q)
        dev = float(np.linalg.norm(b[:2] - self.bg[:2]))
        dyaw = abs(math.remainder(float(b[2] - self.bg[2]), 2 * math.pi))
        arm_ok = float(np.max(np.abs(q - self.stow))) <= self.r['arm_stow_tol']
        conv = self.wb.span_ok(t) and self.wb.max_disp() <= self.r['converge_max_m'] and self.wa.max_disp() <= self.r['arm_win_rad']
        quiet = sum(lcf.values()) <= self.r['unexpected_max_n']
        ok = dev <= self.r['base_dev_m'] and dyaw <= math.radians(self.r['base_dev_deg']) and arm_ok and conv and quiet
        self.last_dev = (dev, dyaw)
        return self._hold(ok, step, t)


class Gate6Admit(_Gate):
    name = 'gate6_admit'

    def hold_s(self):
        return self.r['grip_settle_s']

    def step(self, step, t, T_tcp, T_grasp_nominal, rec):
        """T_grasp_nominal＝H_in(q_meas)·T_HG（呼叫端以當步實測門角算）。"""
        go, missing = self._seq(step, t)
        if not go:
            return self.state
        lcf = link_class_force(rec, step, t, self.r['time_tol'])
        try:
            T, G = np.asarray(T_tcp, float), np.asarray(T_grasp_nominal, float)
            okv = T.shape == (4, 4) and G.shape == (4, 4) and np.all(np.isfinite(T)) and np.all(np.isfinite(G))
        except (TypeError, ValueError):
            okv = False
        if missing or lcf is None or not okv:
            return self._invalid(step, t)
        ep = float(np.linalg.norm(T[:3, 3] - G[:3, 3]))
        er = rot_angle(T[:3, :3], G[:3, :3])
        no_touch = all(finger_force(lcf, f, ALL) <= self.r['contact_abort_n'] for f in FINGERS) and nonfinger_force(lcf) <= self.r['unexpected_max_n']
        ok = ep <= self.r['grip_tol_m'] and er <= self.r['grip_rot_rad'] and no_touch
        self.last_err = (ep, er)
        return self._hold(ok, step, t)


class HoldConfirm(_Gate):
    """mode 6 提交後的短保持確認（不是 gate；結果 CONFIRMED／FAILED 供任務端與判定器）。"""
    name = 'hold_confirm'

    def hold_s(self):
        return self.r['attach_hold_s']

    def _reset_window(self):
        pass

    def step(self, step, t, T_tcp, H_in, rec):
        go, missing = self._seq(step, t)
        if not go:
            return self.state
        if self.state != 'CONFIRMED' and float(t) - self.t['start'] > self.r['t_hold_max_s'] + 1e-9:
            return self._fail('hold_timeout', t, step)
        lcf = link_class_force(rec, step, t, self.r['time_tol'])
        try:
            T, Hm = np.asarray(T_tcp, float), np.asarray(H_in, float)
            okv = T.shape == (4, 4) and Hm.shape == (4, 4) and np.all(np.isfinite(T)) and np.all(np.isfinite(Hm))
        except (TypeError, ValueError):
            okv = False
        if missing or lcf is None or not okv:
            return self._invalid(step, t)
        E = np.linalg.inv(Hm) @ T
        held = all(finger_force(lcf, f, ('bar_in',)) >= self.r['contact_min_n'] for f in FINGERS)
        other = all(finger_force(lcf, f, tuple(c for c in ALL if c != 'bar_in')) <= self.r['unexpected_max_n'] for f in FINGERS) \
            and nonfinger_force(lcf) <= self.r['unexpected_max_n']
        if held and other and getattr(self, 'E0', None) is None:
            self.E0 = E.copy()
            self.events.append({'t': t, 'step': step, 'baseline': True})
        drift_ok = True
        if getattr(self, 'E0', None) is not None:
            dp = float(np.linalg.norm(Hm[:3, :3] @ (E[:3, 3] - self.E0[:3, 3])))
            dr = rot_angle(self.E0[:3, :3], E[:3, :3])
            drift_ok = dp <= self.r['drift_m'] and dr <= self.r['drift_rad']
            self.last_drift = (dp, dr)
        ok = held and other and drift_ok
        was = self.state
        st = self._hold(ok, step, t)
        if was == 'CONFIRMED' and st == 'WAITING':                          # 確認後因條件不成立而失去（非資料無效）
            return self._fail('hold_lost_after_confirm', t, step)
        return st


class Gate3Release(KR5.ExecReleaseMonitor):
    """ExecReleaseMonitor r5（不改）＋r2 規則。呼叫端以 contacts_valid＝紀錄綁定當步（link_class_force 非 None）帶入。
    - 時間缺口：相鄰步 t 增量須＝(Δstep)·dt（容差 1e-6）；不符 ⇒ 本步以 contacts_valid False 送入父類；
      另先核「自最後有效樣本起」的缺口長度，> invalid_max_s ⇒ FAILED data_invalid（父類的無效計時自當步起算，會漏算缺口本身）。
    - 准入：B_CONFIRMED 且父類 quiet_t0（確認窗起點；任何無效重置、恢復後重新起算）至今 ≥ confirm_hold_s，且最後有效步 ∈ {step−1, step}。"""

    def __init__(self, reg=None):
        super().__init__(reg)
        self.r.setdefault('dt', 0.01)
        self.r.setdefault('time_tol', 1e-6)
        self.last_valid_t = None
        self.n['time_gap'] = 0

    def open(self, step, t):
        r = super().open(step, t)
        self.last_valid_t = float(t) if r == 'A_OPENING' else None
        return r

    def step(self, step, t, finger_q, pair_force, contacts_valid, door_q=None):
        if self.state in ('IDLE', 'FAILED'):
            return self.state
        if self.last is not None and type(step) is int and isinstance(t, (int, float)) and not isinstance(t, bool) and step > self.last[0]:
            if abs((float(t) - self.last[1]) - (step - self.last[0]) * self.r['dt']) > self.r['time_tol'] or step != self.last[0] + 1:
                self.n['time_gap'] += 1
                contacts_valid = False
                if self.last_valid_t is not None and float(t) - self.last_valid_t > self.r['invalid_max_s'] + 1e-9:
                    self.last = (step, float(t))
                    return self._fail('data_invalid', t, step, gap_from=self.last_valid_t)
        if contacts_valid is True and self.last_valid_t is not None and self.last_valid_step is not None and self.last is not None \
                and self.last[0] != self.last_valid_step and float(t) - self.last_valid_t > self.r['invalid_max_s'] + 1e-9:
            self.last = (step, float(t))                                                  # r3：上一步無效、本步恢復 ⇒ 先核期限
            return self._fail('data_invalid', t, step, gap_from=self.last_valid_t)
        lv = self.last_valid_step
        st = super().step(step, t, finger_q, pair_force, contacts_valid, door_q)
        if self.last_valid_step == step and lv != step:
            self.last_valid_t = float(t)
        return st

    def confirmed_fresh(self, step):
        if type(step) is not int or self.last_valid_step is None or self.state != 'B_CONFIRMED' or self.quiet_t0 is None:
            return False
        return self.last_valid_step in (step - 1, step) and self.last_valid_t - self.quiet_t0 >= self.r['confirm_hold_s'] - 1e-9
