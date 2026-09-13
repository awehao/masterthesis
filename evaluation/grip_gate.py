"""摩擦夾持的「閉合放行閘」：對中驗收通過才准閉合，並自行產生閉合斜坡。

為什麼要獨立成模組
------------------
第一版把閘寫在模擬迴圈裡，有兩個缺陷，而且都不是跑一趟就看得出來的：

1. **只檢查單一方向的投影**（沿閉合軸的對中量）。任意姿態都可能讓那個投影
   偶然落在門檻內 —— 實測在 sim 0.840 s、手臂還在初始姿態時就放行了。
2. **一旦通過就永久鎖定放行**，之後離開允許狀態也不會失效。

抽出來之後，這些情境可以**不開模擬器**直接測。

放行後如何閉合
--------------
閘不只是把輸出擋在全開。**斜坡由本模組自己產生**，起點是「閘已放行**且**
命令要求閉合」的那一刻，從全開連續走完整個斜坡長度。
這樣即使放行延遲，也不會接上已經走到中途（甚至全閉）的外部命令而跳變。

限制
----
閉合一旦開始，對中條件就**凍結不再評估**：手指壓上把手後，對中量本來就會變，
繼續評估只會把正常的接觸變化判成失效。失效判定只在閉合開始前有效。
"""
from __future__ import annotations


class GripGate:
    """條件全部同時成立並連續保持，才放行閉合。"""

    def __init__(self, *, stage_phases=('engage',), offset_limit_mm=0.30,
                 pos_perp_max_mm=20.0, rot_max_deg=1.0, track_max_rad=0.005,
                 hold_s=0.5, timeout_s=8.0, ramp_s=3.0):
        self.cfg = dict(stage_phases=tuple(stage_phases),
                        offset_limit_mm=float(offset_limit_mm),
                        pos_perp_max_mm=float(pos_perp_max_mm),
                        rot_max_deg=float(rot_max_deg),
                        track_max_rad=float(track_max_rad),
                        hold_s=float(hold_s), timeout_s=float(timeout_s),
                        ramp_s=float(ramp_s))
        self.released = False
        self.release_t = None
        self.ok_since = None
        self.first_request_t = None
        self.close_start_t = None
        self.abort = None
        self.n_blocked = 0
        self.n_invalidated = 0
        self.last_fail = None
        self.trace = []          # (t, released, fail_reason)

    # ---------------------------------------------------------------- 條件
    def _check(self, phase, data_ok, pos_perp_mm, rot_deg, offset_mm, track_rad):
        c = self.cfg
        if phase not in c['stage_phases']:
            return f'相位 {phase} 不在閉合前對中保持階段 {c["stage_phases"]}'
        if not data_ok:
            return '位姿資料無效'
        if not (abs(pos_perp_mm) <= c['pos_perp_max_mm']):
            return (f'抓取位置偏離 {pos_perp_mm:.3f} mm > '
                    f'{c["pos_perp_max_mm"]}')
        if not (abs(rot_deg) <= c['rot_max_deg']):
            return f'工具姿態偏差 {rot_deg:.4f}° > {c["rot_max_deg"]}'
        if not (abs(offset_mm) <= c['offset_limit_mm']):
            return f'沿閉合軸對中 {offset_mm:+.4f} mm > {c["offset_limit_mm"]}'
        if track_rad is not None and not (abs(track_rad) <= c['track_max_rad']):
            return f'關節追蹤誤差 {track_rad:.6f} rad > {c["track_max_rad"]}'
        return None

    def update(self, t, *, phase, data_ok, pos_perp_mm, rot_deg, offset_mm,
               track_rad=None):
        """每個物理步呼叫一次。閉合開始後凍結評估（見模組說明）。"""
        if self.close_start_t is not None:
            return
        fail = self._check(phase, data_ok, pos_perp_mm, rot_deg, offset_mm,
                           track_rad)
        self.last_fail = fail
        if fail is None:
            if self.ok_since is None:
                self.ok_since = t
            elif not self.released and t - self.ok_since >= self.cfg['hold_s']:
                self.released = True
                self.release_t = t
        else:
            # 離開允許狀態 ⇒ 舊的放行結果失效
            if self.released:
                self.n_invalidated += 1
            self.released = False
            self.release_t = None
            self.ok_since = None
        self.trace.append((round(float(t), 4), bool(self.released), fail))

    # ------------------------------------------------------------ 手指命令
    def finger_command(self, t, requested, f_open, f_closed):
        """回傳**實際要套用**的手指命令。"""
        if requested >= f_open - 1e-12:          # 外部要求張開
            self.close_start_t = None            # 重新閉合時斜坡重新起算
            return f_open
        # 外部要求閉合
        if self.close_start_t is None:
            if not self.released:
                if self.first_request_t is None:
                    self.first_request_t = t
                self.n_blocked += 1
                if t - self.first_request_t > self.cfg['timeout_s']:
                    self.abort = 'grip_gate_not_met'
                return f_open
            self.close_start_t = t               # 放行且有要求 ⇒ 斜坡起點
        u = min(1.0, (t - self.close_start_t) / max(self.cfg['ramp_s'], 1e-9))
        return f_open + (f_closed - f_open) * u

    def summary(self):
        return {'config': self.cfg, 'released': self.released,
                'release_sim_t': self.release_t,
                'first_close_request_sim_t': self.first_request_t,
                'close_start_sim_t': self.close_start_t,
                'blocked_samples': self.n_blocked,
                'invalidated_times': self.n_invalidated,
                'abort': self.abort, 'last_fail': self.last_fail,
                'note': ('斜坡由閘自行產生，起點為「已放行且命令要求閉合」；'
                         '閉合開始後凍結條件評估')}


class GripHold:
    """閉合後的夾持驗收：**由實際資料**確認連續滿足，才允許拉動。

    與 GripGate 的分工：GripGate 管「能不能開始閉合」，本類別管「能不能開始拉動」。
    兩者都不接受「時間排得夠久」當成通過 —— 必須逐樣本檢查並連續累積。

    相對位姿一律用 T_gripper→drawer = T_world→gripper⁻¹ · T_world→drawer，
    **不用世界座標差代替**：世界座標差會把夾爪自身的移動算進去。
    """

    def __init__(self, *, contact_min_n=0.5, rel_pos_max_mm=0.20,
                 rel_rot_max_deg=1.0, hold_s=2.0):
        self.cfg = dict(contact_min_n=float(contact_min_n),
                        rel_pos_max_mm=float(rel_pos_max_mm),
                        rel_rot_max_deg=float(rel_rot_max_deg),
                        hold_s=float(hold_s))
        self.ref = None          # 保持起點的相對位姿 (p, R)
        self.ok_since = None
        self.satisfied = False
        self.satisfied_t = None
        # **首次通過事件永久保留**，不被後續失效抹掉。
        # 但「曾經通過」不等於「永久允許拉動」——放行仍需當下條件有效。
        self.first_satisfied_t = None
        self.frozen = False
        self.frozen_t = None
        self.last_fail = None
        self.n_eval = 0
        self.worst = {'contact_min_n': None, 'rel_pos_mm': 0.0, 'rel_rot_deg': 0.0}

    def freeze(self, t):
        """進入拉動後凍結：靜態保持判準不再適用，也不得覆寫驗收歷史。

        靜態門檻（0.20 mm）比拉動門檻（2.0 mm）嚴。理想無滑移時把手與夾爪
        一起移動、相對位姿應保持；實際會有接觸變形或小幅滑移，那由**拉動段
        自己的判準**衡量，不能拿靜態判準去判。
        """
        if not self.frozen:
            self.frozen = True
            self.frozen_t = t

    def update(self, t, *, f1_n, f2_n, rel_p, rel_R, ang_deg_fn):
        """閉合斜坡完成後每步呼叫。`rel_p`/`rel_R` 為夾爪座標系下的抽屜位姿。"""
        if self.frozen:
            return None, None, None
        self.n_eval += 1
        if self.ref is None:
            self.ref = (rel_p.copy(), rel_R.copy())
        dp = float(((rel_p - self.ref[0]) ** 2).sum() ** 0.5) * 1000.0
        dr = float(ang_deg_fn(rel_R, self.ref[1]))
        cmin = min(float(f1_n), float(f2_n))
        w = self.worst
        w['contact_min_n'] = cmin if w['contact_min_n'] is None else min(
            w['contact_min_n'], cmin)
        w['rel_pos_mm'] = max(w['rel_pos_mm'], dp)
        w['rel_rot_deg'] = max(w['rel_rot_deg'], dr)
        c = self.cfg
        fail = None
        if cmin < c['contact_min_n']:
            fail = f'接觸量最小 {cmin:.4f} N < {c["contact_min_n"]}'
        elif dp > c['rel_pos_max_mm']:
            fail = f'相對位移 {dp:.4f} mm > {c["rel_pos_max_mm"]}'
        elif dr > c['rel_rot_max_deg']:
            fail = f'相對轉動 {dr:.4f}° > {c["rel_rot_max_deg"]}'
        self.last_fail = fail
        if fail is None:
            if self.ok_since is None:
                self.ok_since = t
            elif not self.satisfied and t - self.ok_since >= c['hold_s']:
                self.satisfied = True
                self.satisfied_t = t
                if self.first_satisfied_t is None:
                    self.first_satisfied_t = t
        else:
            # 中斷 ⇒ 重新累積；已達成的結果也失效
            self.ok_since = None
            self.satisfied = False
            self.satisfied_t = None
        return dp, dr, cmin

    def held_s(self, t):
        return 0.0 if self.ok_since is None else t - self.ok_since

    def summary(self):
        return {'config': self.cfg, 'satisfied': self.satisfied,
                'satisfied_sim_t': self.satisfied_t,
                'first_satisfied_sim_t': self.first_satisfied_t,
                'frozen': self.frozen, 'frozen_sim_t': self.frozen_t,
                'samples': self.n_eval,
                'worst': self.worst, 'last_fail': self.last_fail,
                'note': ('相對位姿為 T_gripper→drawer，相對**保持起點**；'
                         '接觸量門檻是「接觸存在」判準，不稱法向夾持力')}


class PullGate:
    """拉動放行：**許可在命令套用前決定**，未放行則零筆拉動命令被套用。

    三個性質（由 test_grip_gate.py 離線驗證）：

    1. **零套用**：條件未成立時，一筆拉動設定點都不會送進 articulation；
       手臂凍結在拉動前的最後命令。
    2. **不消耗軌跡時間**：等待期間收到的拉動設定點先進緩衝，不被丟棄；
       放行後**從首點**依原配速起步，不跳接已走到中途的命令。
    3. **曾經通過 ≠ 永久放行**：放行要當下條件成立；歷史通過只作為紀錄。

    等待上限：超過即中止。長時間等待會讓後續 hold/release/retreat 與軌跡
    脫節，與其跑一段不同步的序列，不如停下來。
    """

    def __init__(self, *, rate_hz=50.0, max_wait_s=1.0):
        self.cfg = dict(rate_hz=float(rate_hz), max_wait_s=float(max_wait_s))
        self.buffer = []            # [(seq, q)]，按 seq 遞增
        self.released = False
        self.release_t = None
        self.first_block_t = None
        self.n_blocked = 0
        self.idx = 0
        self.abort = None
        self.block_reason = None

    def offer(self, seq, q):
        """收到一筆拉動設定點。未放行也**先緩衝**，不丟棄、不套用。"""
        if not self.buffer or seq > self.buffer[-1][0]:
            self.buffer.append((int(seq), tuple(float(v) for v in q)))

    def decide(self, t, cond_ok, reason=None):
        """**在套用命令前**呼叫。cond_ok 為當下條件（非歷史通過）。"""
        if self.released or not self.buffer:
            return
        if cond_ok:
            self.released = True
            self.release_t = t
        else:
            self.n_blocked += 1
            self.block_reason = reason
            if self.first_block_t is None:
                self.first_block_t = t
            elif t - self.first_block_t > self.cfg['max_wait_s']:
                self.abort = 'pull_gate_not_met'

    def command(self, t, hold_q):
        """回傳這一步要套用的手臂命令；None = 尚未進入拉動段。"""
        if not self.buffer:
            return None
        if not self.released:
            return hold_q                      # 凍結在拉動前的命令
        k = int(round((t - self.release_t) * self.cfg['rate_hz']))
        self.idx = max(0, min(k, len(self.buffer) - 1))
        return self.buffer[self.idx][1]

    def summary(self):
        return {'config': self.cfg, 'released': self.released,
                'release_sim_t': self.release_t, 'buffered': len(self.buffer),
                'applied_index': self.idx, 'blocked_samples': self.n_blocked,
                'first_block_sim_t': self.first_block_t,
                'block_reason': self.block_reason, 'abort': self.abort,
                'note': ('未放行時零筆拉動命令被套用；等待不消耗軌跡時間，'
                         '放行後自首點起步')}
