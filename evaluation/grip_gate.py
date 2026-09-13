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
