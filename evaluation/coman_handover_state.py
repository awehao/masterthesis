"""協同抽屜操作的交接量測與相位狀態機（**純邏輯，不含 ROS、不跑模擬**）。

為什麼另立一支：判準草案要求四件事**分開記錄**，不得以其中一項代表另一項 ——
命令新鮮（安全層看到的上游）、執行端自身未逾時、保持追蹤合格、允許正常釋放；
而**緊急解除連接不得被上述任何一項閘住**。把這段邏輯抽成可離線測試的模組，
是為了在接上模擬器之前就能用狀態序列驗證它。

量測口徑（與 wb_coman_drawer20_criteria 草案一致）
------------------------------------------------
  * 相對位姿由**呼叫端在同一物理步**提供：夾爪與抽屜的實際世界位姿相乘得出。
    **本模組不做 FK 重建**，也不接受以固定底盤位姿推算的值 —— 底盤會動。
  * 兩個基準分開：
      pre_attachment  對**設計抓取關係**（判斷有沒有對準）
      post_attachment 對**連接當下的關係**（判斷約束殘差是否增加）
  * 合併餘裕 M1 與插入深度 H6 由相對位姿即時算出，並檢查是否落在
    已核對範圍內；超出範圍即拒絕放行（不外推）。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import yaml

SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'results', 'specs',
                         'wb_coman_drawer20_criteria_v0_draft.yaml')


class SpecNotFrozen(RuntimeError):
    pass


REQUIRED = (
    # (路徑, 下界, 上界)；**逐項驗值、驗型別、驗範圍**，不相信 currently_null 清單
    (('H_handover', 'M1_enclosure_margin_m_min'), 0.0, 0.0039),
    (('H_handover', 'H6_insertion_depth_tool_z_dev_m_max'), 0.0, 0.0072),
    (('H_handover', 'H3_window_s'), 0.0, 60.0),
    (('P_pull', 'P1_final_opening_err_m_max'), 0.0, 0.010),
    (('P_pull', 'P1_hold_s'), 0.0, 60.0),
    (('P_pull', 'P6_retreat_clear_tool_z_m'), 0.0, 0.100),
    (('F_force', 'F1_force_abort_n'), 0.0, 1000.0),
    (('F_force', 'F2_force_abort_sustain_s'), 0.0, 10.0),
    (('profile', 'target_stroke_m'), 0.0, 0.220),
    (('geometry', 'gripper_open_inner_m'), 0.0, 0.100),
    (('geometry', 'handle_bar_diameter_m'), 0.0, 0.100),
    (('geometry', 'tcp_to_bar_center_along_tool_z_m'), 0.0, 0.100),
)
RANGE_KEYS = ('t_y_abs_max_m', 't_z_abs_max_m', 'tilt_deg_max', 't_x_abs_max_m')


def load_spec(path: str = SPEC_PATH, allow_draft: bool = False) -> dict:
    """載入判準並**逐項查核**。

    不以 `currently_null` 清單代替查核 —— 清單可能與實際內容不一致。
    每個必要門檻都要有值、型別為數、且落在有效範圍內。
    """
    d = yaml.safe_load(open(path, encoding='utf-8'))
    bad = []
    for keys, lo, hi in REQUIRED:
        node = d
        for k in keys:
            node = (node or {}).get(k) if isinstance(node, dict) else None
        name = '.'.join(keys)
        if node is None:
            bad.append(f'{name} 為 null／缺漏')
        elif isinstance(node, bool) or not isinstance(node, (int, float)):
            bad.append(f'{name} 型別為 {type(node).__name__}，不是數值')
        elif not (lo < float(node) <= hi):
            bad.append(f'{name} = {node} 不在有效範圍 ({lo}, {hi}]')
    rng = (d.get('H_handover') or {}).get('M1_validated_range')
    if not isinstance(rng, dict):
        bad.append('M1_validated_range 缺漏')
    else:
        for k in RANGE_KEYS:
            v = rng.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                bad.append(f'M1_validated_range.{k} 無效：{v!r}')
    if bad:
        raise SpecNotFrozen('判準查核未過：' + '；'.join(bad))
    if d.get('status') != 'frozen' and not allow_draft:
        raise SpecNotFrozen(
            f"判準未凍結（status={d.get('status')}）；拒絕正式執行。"
            '測試可傳 allow_draft=True。')
    return d


@dataclass
class Frame:
    """一個週期的輸入。時間一律為模擬時間。

    相對位姿必須由呼叫端**在同一物理步**讀取夾爪與抽屜的實際世界位姿後算出。
    本模組無法驗證資料來源，只能要求並記錄；`same_step_read` 由呼叫端聲明。
    """
    t: float
    cmd_age_safety_s: float
    cmd_age_endpoint_s: float
    rel_pos_tool: np.ndarray
    bar_axis_tool: np.ndarray
    opening_m: float
    f_norm_n: float
    attached: bool
    monitor_ok: bool = True
    rel_rot_tool: np.ndarray | None = None      # 完整相對旋轉（3×3），缺則記為未提供
    decouple_confirmed: bool = False            # 執行端回報「已解除連接」
    release_requested: bool = False             # 任務流程**明確請求**釋放
    gripper_pos_world: np.ndarray | None = None  # 夾爪世界位置（退出量判定必需）
    gripper_rot_world: np.ndarray | None = None  # 夾爪世界姿態（同上）
    same_step_read: bool = True                 # 呼叫端聲明位姿為同一物理步讀取


@dataclass
class Flags:
    cmd_fresh: bool = False
    endpoint_recv_ok: bool = False
    in_validated_range: bool = False
    handover_cond_now: bool = False     # 當下瞬時符合
    handover_pass: bool = False         # 連續符合滿 H3 才為真
    hold_tracking_pass: bool = False
    normal_release_allowed: bool = False        # **具備資格**，不等於現在要執行
    release_handshake: bool = False             # 資格 ＋ 明確請求同時成立
    emergency_decouple: bool = False
    continuity_break: bool = False
    retreat_measurable: bool = False
    retreat_signed_m: float = float('nan')      # 沿退出起始方向的有號退開量
    release_stability_pass: bool = False        # 與 DONE **無關**，獨立判定


PHASES = ('HANDOVER', 'ENGAGE', 'PULL', 'HOLD', 'RELEASE', 'RETREAT',
          'DONE', 'EMERGENCY')


class HandoverMachine:
    """相位狀態機。每個旗標各自記時間戳，互不代表。

    * 交接與保持都要求**連續**符合：取樣中斷（間隔 > max_gap_s）即重新計時。
    * 正常釋放同時要求：保持合格、仍在連接中、當下命令新鮮、執行端未逾時、相位為 HOLD。
    * 緊急解除要求**閂鎖**，直到執行端回報已解除為止，不因單次旗標消失而取消。
    * 退出完成由**實測退出量**決定，不以多跑一個週期代替。
    """

    def __init__(self, spec: dict, max_gap_s: float = 0.15):
        g = spec['geometry']; h = spec['H_handover']; p = spec['P_pull']
        self.gap_half = float(g['gripper_open_inner_m']) / 2.0
        self.r_bar = float(g['handle_bar_diameter_m']) / 2.0
        self.L = float(h['M1_terms']['L'])
        self.m_min = float(h['M1_enclosure_margin_m_min'])
        self.rng = h['M1_validated_range']
        self.h6 = float(h['H6_insertion_depth_tool_z_dev_m_max'])
        self.h3 = float(h['H3_window_s'])
        self.z_nom = -float(g['tcp_to_bar_center_along_tool_z_m'])
        self.target = float(spec['profile']['target_stroke_m'])
        self.p1 = float(p['P1_final_opening_err_m_max'])
        self.p1_hold_s = float(p['P1_hold_s'])
        self.retreat_clear = float(p['P6_retreat_clear_tool_z_m'])
        p4 = p['P4_release_stability']
        self.stab_window_s = float(p4['window_s'])
        self.stab_change_max = float(p4['opening_change_m_max'])
        self.stab_rate_max = float(p4['opening_rate_m_per_s_max'])
        self.f_abort = float(spec['F_force']['F1_force_abort_n'])
        self.f_sustain_s = float(spec['F_force']['F2_force_abort_sustain_s'])
        self.max_gap_s = float(max_gap_s)
        self.age_safety, self.age_endpoint = 0.25, 0.20
        self.phase = 'HANDOVER'
        self.stamps: dict[str, float] = {}
        self.attach_ref: dict | None = None
        self.rot_recorded = False
        self.same_step_declared = True
        self._t_prev: float | None = None
        self._hand_since: float | None = None
        self._hold_since: float | None = None
        self._force_since: float | None = None
        self._emergency_latched = False
        self._retreat_ref: dict | None = None
        self._stab: dict | None = None
        self.log: list[tuple[float, str, Flags]] = []

    # ---------------- 幾何量 ----------------
    def margin(self, rel_pos, bar_axis) -> float:
        return ((self.gap_half - self.r_bar) - abs(float(rel_pos[1]))
                - (self.L / 2.0) * abs(float(bar_axis[1])))

    def insertion_dev(self, rel_pos) -> float:
        return abs(float(rel_pos[2]) - self.z_nom)

    def tilt_deg(self, bar_axis) -> float:
        c = abs(float(np.clip(bar_axis[0] / np.linalg.norm(bar_axis), -1, 1)))
        return math.degrees(math.acos(c))

    def in_range(self, rel_pos, bar_axis) -> bool:
        r = self.rng
        return (abs(float(rel_pos[1])) <= r['t_y_abs_max_m']
                and self.insertion_dev(rel_pos) <= r['t_z_abs_max_m']
                and self.tilt_deg(bar_axis) <= r['tilt_deg_max']
                and abs(float(rel_pos[0])) <= r['t_x_abs_max_m'])

    # ---------------- 逐週期 ----------------
    def step(self, f: Frame) -> Flags:
        fl = Flags()
        gap = None if self._t_prev is None else f.t - self._t_prev
        fl.continuity_break = gap is not None and gap > self.max_gap_s
        if fl.continuity_break:                 # **缺測不算連續**
            self._hand_since = None
            self._hold_since = None
            self._force_since = None
        self._t_prev = f.t
        if not f.same_step_read:
            self.same_step_declared = False

        fl.cmd_fresh = f.cmd_age_safety_s <= self.age_safety
        fl.endpoint_recv_ok = f.cmd_age_endpoint_s <= self.age_endpoint
        fl.in_validated_range = self.in_range(f.rel_pos_tool, f.bar_axis_tool)

        # 緊急路徑：先判、不看任何閘，且**閂鎖到執行端確認已解除**
        over = f.f_norm_n > self.f_abort
        if over and self._force_since is None:
            self._force_since = f.t
        elif not over:
            self._force_since = None
        sustained = (over and self._force_since is not None
                     and f.t - self._force_since >= self.f_sustain_s)
        if sustained or not f.monitor_ok:
            self._emergency_latched = True
            self._mark('emergency_decouple', f.t)
        if self._emergency_latched:
            if f.decouple_confirmed:
                self._emergency_latched = False
                self._mark('decouple_confirmed', f.t)
                fl.emergency_decouple = False
            else:
                fl.emergency_decouple = True
            self.phase = 'EMERGENCY'
            self.log.append((f.t, self.phase, fl))
            return fl

        # 交接：瞬時條件 → 連續滿 H3 才放行
        fl.handover_cond_now = (fl.in_validated_range
                                and self.margin(f.rel_pos_tool,
                                                f.bar_axis_tool) >= self.m_min
                                and self.insertion_dev(f.rel_pos_tool) <= self.h6)
        if fl.handover_cond_now:
            if self._hand_since is None:
                self._hand_since = f.t
            fl.handover_pass = (f.t - self._hand_since) >= self.h3
        else:
            self._hand_since = None

        # 保持追蹤：連續符合開度容差滿 P1_hold_s（與命令新鮮無關）
        if abs(f.opening_m - self.target) <= self.p1:
            if self._hold_since is None:
                self._hold_since = f.t
            fl.hold_tracking_pass = (f.t - self._hold_since) >= self.p1_hold_s
        else:
            self._hold_since = None

        ok_now = fl.cmd_fresh and fl.endpoint_recv_ok
        # 正常釋放**資格**：保持合格 ＋ 仍在連接 ＋ 當下命令與執行端合格 ＋ 相位為 HOLD
        fl.normal_release_allowed = bool(fl.hold_tracking_pass and f.attached
                                         and ok_now and self.phase == 'HOLD')
        # **資格與執行分開**：還要任務流程在當下明確請求
        fl.release_handshake = bool(fl.normal_release_allowed and f.release_requested)

        # 退出量：相對**退出起點**、沿**退出起始座標系固定方向**的有號位移
        if self._retreat_ref is not None:
            if f.gripper_pos_world is None:
                fl.retreat_measurable = False
            else:
                d = np.asarray(f.gripper_pos_world, float) - self._retreat_ref['p0']
                fl.retreat_signed_m = float(d @ self._retreat_ref['u'])
                fl.retreat_measurable = True

        # 釋放後穩定性：**獨立判定**，DONE 不代表它通過
        if self._stab is not None:
            st = self._stab
            st['lo'] = min(st['lo'], f.opening_m)
            st['hi'] = max(st['hi'], f.opening_m)
            span = st['hi'] - st['lo']
            dur = f.t - st['t0']
            rate = span / dur if dur > 0 else float('inf')
            fl.release_stability_pass = bool(dur >= self.stab_window_s
                                             and span <= self.stab_change_max
                                             and rate <= self.stab_rate_max)

        if f.attached and self.attach_ref is None:
            self.attach_ref = {'pos': np.asarray(f.rel_pos_tool, float).copy(),
                               'rot': (None if f.rel_rot_tool is None
                                       else np.asarray(f.rel_rot_tool, float).copy())}
            self.rot_recorded = f.rel_rot_tool is not None
            self._mark('attached', f.t)

        if self.phase == 'HANDOVER' and fl.handover_pass and ok_now:
            self.phase = 'ENGAGE'; self._mark('handover_pass', f.t)
        elif self.phase == 'ENGAGE' and f.attached and ok_now:
            self.phase = 'PULL'
        elif self.phase == 'PULL' and fl.hold_tracking_pass and ok_now:
            self.phase = 'HOLD'; self._mark('hold_tracking_pass', f.t)
        elif self.phase == 'HOLD' and fl.release_handshake:
            self.phase = 'RELEASE'; self._mark('release_requested', f.t)
        elif self.phase == 'RELEASE' and f.decouple_confirmed:
            # 退出**要等執行端確認解除完成**才開始
            self.phase = 'RETREAT'
            self._mark('decouple_confirmed', f.t)
            self._stab = {'t0': f.t, 'lo': f.opening_m, 'hi': f.opening_m}
            if f.gripper_pos_world is not None and f.gripper_rot_world is not None:
                R0 = np.asarray(f.gripper_rot_world, float)
                self._retreat_ref = {
                    'p0': np.asarray(f.gripper_pos_world, float).copy(),
                    # 退出方向 = 退出起始時夾爪的 +z（把手位於工具 −z 側）
                    'u': R0 @ np.array([0.0, 0.0, 1.0])}
            else:
                self._retreat_ref = None
        elif self.phase == 'RETREAT':
            # 完成條件：**沿退出方向**的有號退開量達門檻；缺世界位姿即無法判定
            if fl.retreat_measurable and fl.retreat_signed_m >= self.retreat_clear:
                self.phase = 'DONE'; self._mark('retreat_complete', f.t)
        self.log.append((f.t, self.phase, fl))
        return fl

    def _mark(self, key: str, t: float) -> None:
        self.stamps.setdefault(key, t)

    # ---------------- 兩個基準 ----------------
    def datum_pre(self, rel_pos, design_rel) -> float:
        return float(np.linalg.norm(np.asarray(rel_pos) - np.asarray(design_rel)))

    def datum_post(self, rel_pos, rel_rot=None) -> dict:
        """連接後：對連接當下關係的位置與**旋轉**漂移。缺旋轉則回報 NaN。"""
        if self.attach_ref is None:
            return {'pos_m': float('nan'), 'rot_rad': float('nan')}
        out = {'pos_m': float(np.linalg.norm(
            np.asarray(rel_pos) - self.attach_ref['pos']))}
        R0 = self.attach_ref['rot']
        if R0 is None or rel_rot is None:
            out['rot_rad'] = float('nan')
        else:
            dR = np.asarray(rel_rot, float) @ R0.T
            out['rot_rad'] = float(math.acos(
                max(-1.0, min(1.0, (np.trace(dR) - 1.0) / 2.0))))
        return out


# ------------------------------------------------------------------ 離線測試
def _frame(t, **kw):
    base = dict(t=t, cmd_age_safety_s=0.05, cmd_age_endpoint_s=0.05,
                rel_pos_tool=np.array([0.0, 0.0005, -0.0147]),
                bar_axis_tool=np.array([1.0, 0.0, 0.0]),
                opening_m=0.0, f_norm_n=5.0, attached=False)
    base.update(kw)
    return Frame(**base)


def _run(M, t0, n, dt=0.05, **kw):
    t = t0
    fl = None
    for _ in range(n):
        fl = M.step(_frame(t, **kw)); t += dt
    return fl, t


def selftest() -> int:
    spec = load_spec(allow_draft=True)
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f'  {name:52s} {"ok" if cond else "**錯**"}')
        bad += not cond

    # --- 反例 1：交接必須連續滿 H3，第一筆合格不得放行 ---
    M = HandoverMachine(spec)
    fl = M.step(_frame(0.0))
    check('交接首筆合格不得放行（需連續 3 s）',
          fl.handover_cond_now and not fl.handover_pass and M.phase == 'HANDOVER')
    fl, t = _run(M, 0.05, 40)                      # 共 2.0 s
    check('交接連續 2.0 s 仍不得放行', not fl.handover_pass)
    fl, t = _run(M, t, 25)                         # 累計 > 3.0 s
    check('交接連續滿 3 s 才進入 ENGAGE',
          fl.handover_pass and M.phase == 'ENGAGE')

    # --- 反例 2：交接中途不合格要重新計時 ---
    M2 = HandoverMachine(spec)
    _run(M2, 0.0, 50)
    M2.step(_frame(2.5, rel_pos_tool=np.array([0.0, 0.0035, -0.0147])))  # 超範圍
    fl, t = _run(M2, 2.55, 50)                     # 之後再連續 2.5 s
    check('中斷後重新計時，未滿 3 s 不得放行',
          not fl.handover_pass and M2.phase == 'HANDOVER')

    # --- 反例 3：缺測不得算連續保持（只餵 t=0 與 t=3）---
    M3 = HandoverMachine(spec)
    M3.step(_frame(0.0, attached=True, opening_m=0.0200))
    fl = M3.step(_frame(3.0, attached=True, opening_m=0.0200))
    check('t=0 與 t=3 兩筆：偵測到取樣中斷', fl.continuity_break)
    check('缺測不得算保持合格', not fl.hold_tracking_pass)

    # --- 反例 4：保持合格但當下執行端逾時 → 不得允許正常釋放 ---
    M4 = HandoverMachine(spec)
    _run(M4, 0.0, 65)                              # 交接滿 3 s
    _run(M4, 3.25, 4, attached=True)               # 進 PULL
    t = 3.45
    fl = None
    for _ in range(80):                            # 跑到剛進入 HOLD 就停
        fl = M4.step(_frame(t, attached=True, opening_m=0.0200)); t += 0.05
        if M4.phase == 'HOLD':
            break
    check('保持連續 2 s 後進入 HOLD',
          fl.hold_tracking_pass and M4.phase == 'HOLD'
          and abs(M4.stamps['hold_tracking_pass'] - M4.stamps.get('attached', 0)
                  - 2.0) < 0.3)
    fl = M4.step(_frame(t, attached=True, opening_m=0.0200,
                        cmd_age_endpoint_s=0.22))
    check('保持合格但執行端逾時 → 不具釋放資格',
          fl.hold_tracking_pass and not fl.normal_release_allowed)
    check('不具資格時相位不得進入 RELEASE', M4.phase == 'HOLD')

    # --- 反例 5：釋放需「資格 ＋ 明確請求」同時成立 ---
    t += 0.05
    fl = M4.step(_frame(t, attached=True, opening_m=0.0200))
    check('有資格但**未請求** → 不得進入 RELEASE',
          fl.normal_release_allowed and not fl.release_handshake
          and M4.phase == 'HOLD')
    t += 0.05
    fl = M4.step(_frame(t, attached=True, opening_m=0.0200,
                        cmd_age_endpoint_s=0.22, release_requested=True))
    check('有請求但**無資格** → 不得進入 RELEASE',
          not fl.normal_release_allowed and not fl.release_handshake
          and M4.phase == 'HOLD')
    t += 0.05
    fl = M4.step(_frame(t, attached=True, opening_m=0.0200,
                        release_requested=True))
    check('資格與請求同時成立 → 進入 RELEASE',
          fl.release_handshake and M4.phase == 'RELEASE')

    # --- 反例 6：退出須等執行端確認解除完成才開始 ---
    t += 0.05
    M4.step(_frame(t, attached=False, opening_m=0.0200))
    check('僅 attached=False 不足以開始退出', M4.phase == 'RELEASE')
    t += 0.05
    P0 = np.zeros(3); R0 = np.eye(3)
    M4.step(_frame(t, attached=False, opening_m=0.0200, decouple_confirmed=True,
                   gripper_pos_world=P0, gripper_rot_world=R0))
    check('確認解除後才進入 RETREAT', M4.phase == 'RETREAT')

    # --- 反例 7：退出量必須是沿退出方向的有號位移 ---
    t += 0.05
    fl = M4.step(_frame(t, attached=False, opening_m=0.0200,
                        rel_pos_tool=np.array([0.020, 0.0005, -0.0147]),
                        gripper_pos_world=np.array([0.020, 0.0, 0.0]),
                        gripper_rot_world=R0))
    check('沿橫桿方向移動 20 mm、未沿退出方向退開 → 不得 DONE',
          M4.phase == 'RETREAT' and abs(fl.retreat_signed_m) < 1e-9)
    t += 0.05
    fl = M4.step(_frame(t, attached=False, opening_m=0.0200,
                        rel_pos_tool=np.array([0.0, 0.0005, -0.0240]),
                        gripper_pos_world=np.array([0.0, 0.0, 0.0093]),
                        gripper_rot_world=R0))
    check('僅退開 9.3 mm（< 23.3）→ 不得 DONE',
          M4.phase == 'RETREAT' and abs(fl.retreat_signed_m - 0.0093) < 1e-9)
    t += 0.05
    fl = M4.step(_frame(t, attached=False, opening_m=0.0200,
                        gripper_pos_world=np.array([0.0, 0.0, 0.0233]),
                        gripper_rot_world=R0))
    check('沿退出方向退開 23.3 mm → DONE',
          M4.phase == 'DONE' and 'retreat_complete' in M4.stamps)
    check('DONE 當下釋放後穩定性尚未通過（兩者獨立）',
          not fl.release_stability_pass)

    # 方向固定於起始座標系：之後轉動工具不改變判讀
    M8 = HandoverMachine(spec)
    M8.phase = 'RETREAT'
    M8._retreat_ref = {'p0': np.zeros(3), 'u': np.array([0.0, 0.0, 1.0])}
    c30, s30 = math.cos(math.radians(30)), math.sin(math.radians(30))
    Rz = np.array([[c30, -s30, 0.0], [s30, c30, 0.0], [0.0, 0.0, 1.0]])
    fl = M8.step(_frame(0.0, gripper_pos_world=np.array([0.0, 0.0, 0.025]),
                        gripper_rot_world=Rz))
    check('退出開始後轉動工具不改變退出量',
          abs(fl.retreat_signed_m - 0.025) < 1e-12)
    M9 = HandoverMachine(spec)
    M9.phase = 'RETREAT'
    M9._retreat_ref = {'p0': np.zeros(3), 'u': np.array([0.0, 0.0, 1.0])}
    fl = M9.step(_frame(0.0))
    check('缺世界位姿 → 無法判定退出，維持 RETREAT',
          not fl.retreat_measurable and M9.phase == 'RETREAT')

    # --- 釋放後穩定性獨立判定 ---
    M10 = HandoverMachine(spec)
    M10.phase = 'RELEASE'
    M10.step(_frame(0.0, opening_m=0.0200, decouple_confirmed=True,
                    gripper_pos_world=np.zeros(3), gripper_rot_world=np.eye(3)))
    tt, fl = 0.05, None
    for _ in range(220):                       # 11 s，開度不變
        fl = M10.step(_frame(tt, opening_m=0.0200,
                             gripper_pos_world=np.zeros(3),
                             gripper_rot_world=np.eye(3))); tt += 0.05
    check('開度穩定滿 10 s → 穩定性通過', fl.release_stability_pass)
    M11 = HandoverMachine(spec)
    M11.phase = 'RELEASE'
    M11.step(_frame(0.0, opening_m=0.0200, decouple_confirmed=True,
                    gripper_pos_world=np.zeros(3), gripper_rot_world=np.eye(3)))
    tt = 0.05
    for k in range(220):
        fl = M11.step(_frame(tt, opening_m=0.0200 + (0.0002 if k > 100 else 0.0),
                             gripper_pos_world=np.zeros(3),
                             gripper_rot_world=np.eye(3))); tt += 0.05
    check('釋放後開度變動 0.2 mm → 穩定性不通過', not fl.release_stability_pass)

    # --- 反例 6：緊急解除要閂鎖到執行端確認 ---
    M5 = HandoverMachine(spec)
    _run(M5, 0.0, 4, attached=True)
    fl, t = _run(M5, 0.20, 3, attached=True, f_norm_n=35.0)
    check('超力持續 → 緊急解除', fl.emergency_decouple and M5.phase == 'EMERGENCY')
    fl = M5.step(_frame(t, attached=True, f_norm_n=5.0))
    check('力回到正常仍維持解除要求（閂鎖）',
          fl.emergency_decouple and M5.phase == 'EMERGENCY')
    fl = M5.step(_frame(t + 0.05, attached=False, f_norm_n=5.0,
                        decouple_confirmed=True))
    check('執行端確認已解除後才撤除要求', not fl.emergency_decouple)

    # --- 反例 7：判準查核不得只看清單 ---
    import copy, tempfile, os as _os
    bogus = copy.deepcopy(spec)
    bogus['H_handover']['M1_enclosure_margin_m_min'] = None
    bogus['runner_gate']['currently_null'] = []
    bogus['status'] = 'frozen'
    fd, tmp = tempfile.mkstemp(suffix='.yaml'); _os.close(fd)
    yaml.safe_dump(bogus, open(tmp, 'w', encoding='utf-8'), allow_unicode=True)
    try:
        load_spec(tmp)
        check('M1 為 null 但標 frozen → 應拒絕', False)
    except SpecNotFrozen as e:
        check('M1 為 null 但標 frozen → 仍被拒絕', 'M1' in str(e))
    bogus['H_handover']['M1_enclosure_margin_m_min'] = 0.05     # 大於名目餘隙
    yaml.safe_dump(bogus, open(tmp, 'w', encoding='utf-8'), allow_unicode=True)
    try:
        load_spec(tmp)
        check('M1 超出有效範圍 → 應拒絕', False)
    except SpecNotFrozen as e:
        check('M1 超出有效範圍 → 被拒絕', '有效範圍' in str(e))
    _os.unlink(tmp)

    # --- 基準：位置與旋轉分開保存 ---
    M6 = HandoverMachine(spec)
    R0 = np.eye(3)
    M6.step(_frame(0.0, attached=True, rel_rot_tool=R0))
    c, s_ = math.cos(0.01), math.sin(0.01)
    R1 = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    d = M6.datum_post(np.array([0.0, 0.0008, -0.0147]), R1)
    check('連接後基準含旋轉漂移 0.01 rad', abs(d['rot_rad'] - 0.01) < 1e-9)
    M7 = HandoverMachine(spec)
    M7.step(_frame(0.0, attached=True))
    check('未提供旋轉時記為 NaN 且標記未記錄',
          math.isnan(M7.datum_post(np.zeros(3))['rot_rad']) and not M7.rot_recorded)

    print('離線狀態序列測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(selftest())
