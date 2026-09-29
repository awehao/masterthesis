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


def load_spec(path: str = SPEC_PATH, allow_draft: bool = False) -> dict:
    """載入判準。未凍結或仍有 null 門檻時**拒絕**，除非明確允許草案（僅供測試）。"""
    d = yaml.safe_load(open(path, encoding='utf-8'))
    nulls = d.get('runner_gate', {}).get('currently_null', []) or []
    frozen = d.get('status') == 'frozen'
    if not (frozen and not nulls) and not allow_draft:
        raise SpecNotFrozen(
            f"判準未凍結（status={d.get('status')}）或仍有 null 門檻 {nulls}；"
            '拒絕正式執行。測試可傳 allow_draft=True。')
    return d


@dataclass
class Frame:
    """一個週期的輸入。時間一律為模擬時間。"""
    t: float
    cmd_age_safety_s: float      # 安全層看到的上游命令年齡
    cmd_age_endpoint_s: float    # 執行端自身的命令年齡
    rel_pos_tool: np.ndarray     # 把手（橫桿中心）在**工具座標**的位置
    bar_axis_tool: np.ndarray    # 橫桿軸在工具座標的單位向量
    opening_m: float
    f_norm_n: float
    attached: bool
    monitor_ok: bool = True


@dataclass
class Flags:
    cmd_fresh: bool = False
    endpoint_recv_ok: bool = False
    in_validated_range: bool = False
    handover_pass: bool = False
    hold_tracking_pass: bool = False
    normal_release_allowed: bool = False
    emergency_decouple: bool = False


PHASES = ('HANDOVER', 'ENGAGE', 'PULL', 'HOLD', 'RELEASE', 'RETREAT',
          'DONE', 'EMERGENCY')


class HandoverMachine:
    """相位狀態機。每個旗標各自記時間戳，互不代表。"""

    def __init__(self, spec: dict):
        g = spec['geometry']
        h = spec['H_handover']
        p = spec['P_pull']
        self.gap_half = float(g['gripper_open_inner_m']) / 2.0
        self.r_bar = float(g['handle_bar_diameter_m']) / 2.0
        self.L = float(h['M1_terms']['L'])
        self.m_min = h['M1_enclosure_margin_m_min']
        self.rng = h['M1_validated_range']
        self.h6 = h['H6_insertion_depth_tool_z_dev_m_max']
        self.z_nom = -float(g['tcp_to_bar_center_along_tool_z_m'])
        self.target = float(spec['profile']['target_stroke_m'])
        self.p1 = p['P1_final_opening_err_m_max']
        self.p1_hold_s = float(p['P1_hold_s'])
        self.f_abort = float(spec['F_force']['F1_force_abort_n'])
        self.f_sustain_s = float(spec['F_force']['F2_force_abort_sustain_s'])
        self.age_safety = 0.25      # 安全層 max_cmd_age
        self.age_endpoint = 0.20    # 執行端 max_cmd_age_s（較嚴）
        self.phase = 'HANDOVER'
        self.stamps: dict[str, float] = {}
        self.attach_ref: np.ndarray | None = None
        self._hold_since: float | None = None
        self._force_since: float | None = None
        self.log: list[tuple[float, str, Flags]] = []

    # ---------------- 幾何量 ----------------
    def margin(self, rel_pos, bar_axis) -> float:
        """合併餘裕 m（近似式，僅在已核對範圍內使用）。"""
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
        fl.cmd_fresh = f.cmd_age_safety_s <= self.age_safety
        fl.endpoint_recv_ok = f.cmd_age_endpoint_s <= self.age_endpoint
        fl.in_validated_range = self.in_range(f.rel_pos_tool, f.bar_axis_tool)

        # **緊急路徑先判，且不看任何閘** ——
        # 超力或監看失效時，必須能立即「先解除耦合、再停止」。
        over = f.f_norm_n > self.f_abort
        self._force_since = (f.t if over and self._force_since is None
                             else (None if not over else self._force_since))
        force_sustained = (over and self._force_since is not None
                           and f.t - self._force_since >= self.f_sustain_s)
        if force_sustained or not f.monitor_ok:
            fl.emergency_decouple = True
            self._mark('emergency_decouple', f.t)
            self.phase = 'EMERGENCY'
            self.log.append((f.t, self.phase, fl))
            return fl

        # 交接：合併餘裕與插入深度都要過，且必須在已核對範圍內
        if self.m_min is not None and self.h6 is not None:
            fl.handover_pass = (fl.in_validated_range
                                and self.margin(f.rel_pos_tool,
                                                f.bar_axis_tool) >= self.m_min
                                and self.insertion_dev(f.rel_pos_tool) <= self.h6)

        # 保持追蹤：開度在容差內並持續足夠時間（與命令新鮮無關）
        if self.p1 is not None and abs(f.opening_m - self.target) <= self.p1:
            self._hold_since = f.t if self._hold_since is None else self._hold_since
            fl.hold_tracking_pass = (f.t - self._hold_since) >= self.p1_hold_s
        else:
            self._hold_since = None

        # 正常釋放：**只**由保持條件閘控，且必須仍在連接中
        fl.normal_release_allowed = bool(fl.hold_tracking_pass and f.attached)

        if f.attached and self.attach_ref is None:
            self.attach_ref = np.asarray(f.rel_pos_tool, float).copy()
            self._mark('attached', f.t)

        # 相位推進（命令不新鮮或執行端逾時即原地保持，不推進）
        ok = fl.cmd_fresh and fl.endpoint_recv_ok
        if self.phase == 'HANDOVER' and fl.handover_pass and ok:
            self.phase = 'ENGAGE'; self._mark('handover_pass', f.t)
        elif self.phase == 'ENGAGE' and f.attached and ok:
            self.phase = 'PULL'
        elif self.phase == 'PULL' and fl.hold_tracking_pass and ok:
            self.phase = 'HOLD'; self._mark('hold_tracking_pass', f.t)
        elif self.phase == 'HOLD' and fl.normal_release_allowed and ok:
            self.phase = 'RELEASE'; self._mark('normal_release', f.t)
        elif self.phase == 'RELEASE' and not f.attached:
            self.phase = 'RETREAT'
        elif self.phase == 'RETREAT':
            self.phase = 'DONE'
        self.log.append((f.t, self.phase, fl))
        return fl

    def _mark(self, key: str, t: float) -> None:
        self.stamps.setdefault(key, t)

    # ---------------- 兩個基準 ----------------
    def datum_pre(self, rel_pos, design_rel) -> float:
        """連接前：對**設計抓取關係**的偏差。"""
        return float(np.linalg.norm(np.asarray(rel_pos) - np.asarray(design_rel)))

    def datum_post(self, rel_pos) -> float:
        """連接後：對**連接當下關係**的漂移。未連接則為 NaN。"""
        if self.attach_ref is None:
            return float('nan')
        return float(np.linalg.norm(np.asarray(rel_pos) - self.attach_ref))


# ------------------------------------------------------------------ 離線測試
def _frame(t, **kw):
    base = dict(t=t, cmd_age_safety_s=0.05, cmd_age_endpoint_s=0.05,
                rel_pos_tool=np.array([0.0, 0.0005, -0.0147]),
                bar_axis_tool=np.array([1.0, 0.0, 0.0]),
                opening_m=0.0, f_norm_n=5.0, attached=False)
    base.update(kw)
    return Frame(**base)


def selftest() -> int:
    """離線狀態序列測試：不需要模擬器，也不需要凍結判準。"""
    spec = load_spec(allow_draft=True)
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f'  {name:46s} {"ok" if cond else "**錯**"}')
        bad += not cond

    # 1 正常序列：交接 → 連接 → 拉到 20 mm → 保持 2 s → 允許釋放
    M = HandoverMachine(spec)
    t = 0.0
    for _ in range(10):
        M.step(_frame(t)); t += 0.05
    check('交接通過後進入 ENGAGE', M.phase == 'ENGAGE')
    for _ in range(4):
        M.step(_frame(t, attached=True)); t += 0.05
    check('連接後進入 PULL', M.phase == 'PULL')
    fl = None
    for _ in range(60):                      # 3 s 維持在容差內
        fl = M.step(_frame(t, attached=True, opening_m=0.0200)); t += 0.05
    check('保持追蹤合格', fl.hold_tracking_pass)
    check('允許正常釋放', fl.normal_release_allowed)
    check('保持 2 s 之前不得允許釋放',
          M.stamps['hold_tracking_pass'] - M.stamps['attached'] >= 2.0)

    # 2 命令過期：旗標分開，且相位不推進
    M2 = HandoverMachine(spec)
    fl = M2.step(_frame(0.0, cmd_age_safety_s=0.30, cmd_age_endpoint_s=0.05))
    check('安全層過期 → cmd_fresh 為假', not fl.cmd_fresh)
    check('同一週期執行端仍在容忍內 → endpoint_recv_ok 為真', fl.endpoint_recv_ok)
    check('過期時不推進相位', M2.phase == 'HANDOVER')
    fl = M2.step(_frame(0.05, cmd_age_safety_s=0.05, cmd_age_endpoint_s=0.22))
    check('執行端逾時（0.22 > 0.20）獨立判定', not fl.endpoint_recv_ok and fl.cmd_fresh)

    # 3 保持未達成 → 不得釋放
    M3 = HandoverMachine(spec)
    t = 0.0
    for _ in range(40):
        fl = M3.step(_frame(t, attached=True, opening_m=0.0190)); t += 0.05
    check('開度差 1.0 mm > 容差 0.5 mm → 保持不合格', not fl.hold_tracking_pass)
    check('保持不合格 → 不允許正常釋放', not fl.normal_release_allowed)

    # 4 超力 → 緊急解除，且**不受保持閘擋**
    M4 = HandoverMachine(spec)
    t = 0.0
    for _ in range(4):
        M4.step(_frame(t, attached=True, opening_m=0.0)); t += 0.05
    for _ in range(3):                        # 連續超力 0.10 s ≥ 0.05 s
        fl = M4.step(_frame(t, attached=True, f_norm_n=35.0)); t += 0.05
    check('超力持續 → 緊急解除', fl.emergency_decouple and M4.phase == 'EMERGENCY')
    check('緊急解除時保持條件未達成（證明未被閘住）', not fl.hold_tracking_pass)

    # 5 監看失效 → 立即緊急解除
    M5 = HandoverMachine(spec)
    fl = M5.step(_frame(0.0, monitor_ok=False))
    check('監看失效 → 立即緊急解除', fl.emergency_decouple)

    # 6 超出已核對範圍 → 拒絕放行
    M6 = HandoverMachine(spec)
    fl = M6.step(_frame(0.0, rel_pos_tool=np.array([0.0, 0.0035, -0.0147])))
    check('|t_y| 3.5 mm 超出已核對範圍 → in_validated_range 為假',
          not fl.in_validated_range)
    check('超出範圍 → 交接不通過', not fl.handover_pass)
    fl = M6.step(_frame(0.05, rel_pos_tool=np.array([0.0, 0.0005, -0.0147 - 0.0025])))
    check('插入偏差 2.5 mm > H6 2.0 mm → 交接不通過', not fl.handover_pass)

    # 7 兩個基準分開
    M7 = HandoverMachine(spec)
    design = np.array([0.0, 0.0, -0.0147])
    pre = M7.datum_pre(np.array([0.0, 0.0014, -0.0147]), design)
    check('連接前基準：對設計關係 1.4 mm', abs(pre - 0.0014) < 1e-12)
    check('未連接時連接後基準為 NaN', math.isnan(M7.datum_post(design)))
    M7.step(_frame(0.0, attached=True))
    post = M7.datum_post(np.array([0.0, 0.0008, -0.0147]))
    check('連接後基準：對連接當下關係 0.3 mm', abs(post - 0.0003) < 1e-9)

    print('離線狀態序列測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(selftest())
