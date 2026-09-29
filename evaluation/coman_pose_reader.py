"""協同執行端的**同一物理步位姿讀取**與記錄（與模擬器解耦，可離線測試）。

為什麼另立一支
--------------
協同案例的底盤會動，**不得**沿用「固定底盤位姿 ＋ 關節角 FK」的重建。
本模組要求呼叫端在**同一個物理步**內，直接讀取夾爪與抽屜的**實際世界位姿**，
再由抽屜位姿與**固定把手變換**算出把手世界位姿，最後求夾爪—把手相對位姿。

硬性規則（違反即判為無效，不得放行）
------------------------------------
  1. 夾爪與抽屜的讀取必須回報**相同的 physics_step_id**；不同即判不同步。
  2. 任一值缺漏或非有限即判無效。
  3. **不得**以 FK 重建補值，**不得**沿用上一筆資料 ——
     本模組沒有任何回退路徑，無效時 `relative` 為 None。
  4. 把手位姿由**抽屜實際位姿 × 固定把手變換**求得；
     **不得**把抽屜原點當成把手。

讀取順序（同一物理步內，固定且記錄於每筆）
------------------------------------------
    world.step() 之後、寫入任何命令之前：
      1 physics_step_id   2 sim_time
      3 夾爪世界位姿       4 抽屜世界位姿
      5 把手世界位姿（由 4 與固定變換算出）
      6 相對位姿 G_T_H    7 兩套基準
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

READ_ORDER = ('physics_step_id', 'sim_time', 'gripper_world', 'drawer_world',
              'handle_world', 'relative', 'datums')

LOG_COLS = (
    'physics_step_id', 'sim_time',
    'grip_px', 'grip_py', 'grip_pz', 'grip_qw', 'grip_qx', 'grip_qy', 'grip_qz',
    'draw_px', 'draw_py', 'draw_pz', 'draw_qw', 'draw_qx', 'draw_qy', 'draw_qz',
    'hand_px', 'hand_py', 'hand_pz', 'hand_qw', 'hand_qx', 'hand_qy', 'hand_qz',
    'rel_tx', 'rel_ty', 'rel_tz',
    'rel_r00', 'rel_r01', 'rel_r02', 'rel_r10', 'rel_r11', 'rel_r12',
    'rel_r20', 'rel_r21', 'rel_r22',
    'bar_axis_tool_x', 'bar_axis_tool_y', 'bar_axis_tool_z',
    'datum_pre_pos_m', 'datum_post_pos_m', 'datum_post_rot_rad',
    'valid', 'invalid_reason', 'same_step_read',
)


class PoseSource(Protocol):
    """呼叫端提供；每次回傳 (physics_step_id, position(3), quat_wxyz(4))。"""

    def read_gripper(self) -> tuple[int, np.ndarray, np.ndarray]: ...
    def read_drawer(self) -> tuple[int, np.ndarray, np.ndarray]: ...
    def sim_time(self) -> float: ...
    def physics_step_id(self) -> int: ...


def quat_to_rot(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if not np.isfinite(n) or n == 0.0:
        return np.full((3, 3), np.nan)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(R) -> np.ndarray:
    t = float(np.trace(R))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = math.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def homog(p, R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


@dataclass(frozen=True)
class HandleTransform:
    """抽屜座標系下的**固定**把手變換（橫桿中心與軸向）。"""
    bar_center_local: np.ndarray
    bar_axis_local: np.ndarray

    @classmethod
    def from_spec(cls, spec: dict) -> 'HandleTransform':
        bar = spec['drawer']['handle']['bar']
        axis = {'x': [1.0, 0, 0], 'y': [0, 1.0, 0], 'z': [0, 0, 1.0]}[bar['axis']]
        return cls(np.array(bar['center'], float), np.array(axis, float))

    def world(self, drawer_pos, drawer_rot) -> np.ndarray:
        """把手世界位姿 = 抽屜實際位姿 × 固定變換。**不是**抽屜原點。"""
        R = np.asarray(drawer_rot, float)
        return homog(np.asarray(drawer_pos, float)
                     + R @ self.bar_center_local, R)


@dataclass
class PoseRecord:
    physics_step_id: int
    sim_time: float
    gripper: np.ndarray | None
    drawer: np.ndarray | None
    handle: np.ndarray | None
    relative: np.ndarray | None
    valid: bool
    invalid_reason: str
    same_step: bool = False          # **實際核對結果**，不是固定值
    read_order: tuple = field(default=READ_ORDER)


class SameStepPoseReader:
    """同一物理步讀取。**沒有回退路徑**：無效就是無效。"""

    def __init__(self, source: PoseSource, handle: HandleTransform,
                 design_rel: np.ndarray | None = None):
        self.src = source
        self.handle = handle
        self.design_rel = design_rel
        self.attach_ref: dict | None = None
        self.n_invalid = 0

    def read(self) -> PoseRecord:
        step = self.src.physics_step_id()
        t = self.src.sim_time()
        bad = ''
        try:
            gs, gp, gq = self.src.read_gripper()
            ds, dp, dq = self.src.read_drawer()
        except Exception as e:                    # 讀取失敗即無效，不補值
            self.n_invalid += 1
            # 讀取失敗**不得冒稱同步**
            return PoseRecord(step, t, None, None, None, None, False,
                              f'讀取例外：{type(e).__name__}', same_step=False)
        same_step = (gs == ds == step)
        if not same_step:
            bad = f'不同步：夾爪 step {gs}、抽屜 step {ds}、當前 step {step}'

        def _vec(v, n, name):
            nonlocal bad
            a = np.asarray(v, float).reshape(-1) if v is not None else None
            if a is None or a.shape != (n,):
                bad = bad or f'{name} 維度不符（應為 {n}）'
                return None
            if not np.isfinite(a).all():
                bad = bad or f'{name} 含非有限值'
                return None
            return a

        gpv = _vec(gp, 3, 'grip_pos'); gqv = _vec(gq, 4, 'grip_quat')
        dpv = _vec(dp, 3, 'draw_pos'); dqv = _vec(dq, 4, 'draw_quat')
        for q, name in ((gqv, 'grip_quat'), (dqv, 'draw_quat')):
            if q is not None and float(np.linalg.norm(q)) < 1e-8:
                bad = bad or f'{name} 模長近零，不是有效四元數'
        if not np.isfinite(t) or not isinstance(step, (int, np.integer)):
            bad = bad or '時間或步序無效'
        if bad:
            self.n_invalid += 1
            # **不以 FK 重建、不沿用上一筆**：直接回報無效
            return PoseRecord(step, t, None, None, None, None, False, bad,
                              same_step=same_step)

        Tg = homog(gpv, quat_to_rot(gqv))
        Td = homog(dpv, quat_to_rot(dqv))
        Th = self.handle.world(Td[:3, 3], Td[:3, :3])
        try:
            rel = np.linalg.inv(Tg) @ Th
        except np.linalg.LinAlgError:
            self.n_invalid += 1
            return PoseRecord(step, t, None, None, None, None, False,
                              '夾爪變換不可逆', same_step=same_step)
        # **算完再查一次**：四元數合法不保證變換有限
        if not all(np.isfinite(M).all() for M in (Tg, Td, Th, rel)):
            self.n_invalid += 1
            return PoseRecord(step, t, None, None, None, None, False,
                              '計算後的變換含非有限值', same_step=same_step)
        return PoseRecord(step, t, Tg, Td, Th, rel, True, '', same_step=True)

    # ---- 兩套基準 ----
    def mark_attached(self, rec: PoseRecord) -> None:
        if not rec.valid:
            raise RuntimeError('無效讀數不得用來設定連接基準')
        self.attach_ref = {'pos': rec.relative[:3, 3].copy(),
                           'rot': rec.relative[:3, :3].copy()}

    def datums(self, rec: PoseRecord) -> dict:
        out = {'pre_pos_m': float('nan'), 'post_pos_m': float('nan'),
               'post_rot_rad': float('nan')}
        if not rec.valid:
            return out
        if self.design_rel is not None:
            out['pre_pos_m'] = float(np.linalg.norm(
                rec.relative[:3, 3] - np.asarray(self.design_rel, float)))
        if self.attach_ref is not None:
            out['post_pos_m'] = float(np.linalg.norm(
                rec.relative[:3, 3] - self.attach_ref['pos']))
            dR = rec.relative[:3, :3] @ self.attach_ref['rot'].T
            out['post_rot_rad'] = float(math.acos(
                max(-1.0, min(1.0, (np.trace(dR) - 1.0) / 2.0))))
        return out

    def row(self, rec: PoseRecord) -> list:
        """依 LOG_COLS 攤平成一列；無效時幾何欄位一律 NaN，不補值。"""
        nan3, nan4, nan9 = [float('nan')] * 3, [float('nan')] * 4, [float('nan')] * 9
        d = self.datums(rec)
        if rec.valid:
            g, dr, h = rec.gripper, rec.drawer, rec.handle
            pose = lambda T: list(T[:3, 3]) + list(rot_to_quat(T[:3, :3]))
            rel_t = list(rec.relative[:3, 3])
            rel_R = list(rec.relative[:3, :3].reshape(-1))
            axis_tool = list(rec.relative[:3, :3] @ self.handle.bar_axis_local)
        else:
            pose = None
            rel_t, rel_R, axis_tool = nan3, nan9, nan3
        return ([rec.physics_step_id, rec.sim_time]
                + (pose(rec.gripper) if rec.valid else nan3 + nan4)
                + (pose(rec.drawer) if rec.valid else nan3 + nan4)
                + (pose(rec.handle) if rec.valid else nan3 + nan4)
                + rel_t + rel_R + axis_tool
                + [d['pre_pos_m'], d['post_pos_m'], d['post_rot_rad']]
                + [bool(rec.valid), rec.invalid_reason, bool(rec.same_step)])


# ------------------------------------------------------------------ 離線測試
class _FakeSource:
    """測試用位姿來源；可注入步序不一致、NaN、例外。"""

    def __init__(self):
        self.step = 0
        self.t = 0.0
        self.g = (np.zeros(3), np.array([1.0, 0, 0, 0]))
        self.d = (np.zeros(3), np.array([1.0, 0, 0, 0]))
        self.g_step_offset = 0
        self.raise_on_drawer = False

    def physics_step_id(self): return self.step
    def sim_time(self): return self.t
    def read_gripper(self): return (self.step + self.g_step_offset,) + self.g

    def read_drawer(self):
        if self.raise_on_drawer:
            raise RuntimeError('讀取失敗')
        return (self.step,) + self.d


def selftest() -> int:
    import os
    import yaml
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f'  {name:50s} {"ok" if cond else "**錯**"}')
        bad += not cond

    spec = yaml.safe_load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'src', 'my_omnibot_description', 'config', 'drawer_unit.yaml'),
        encoding='utf-8'))
    H = HandleTransform.from_spec(spec)
    check('把手固定變換讀自資產：中心 [0,-0.285,0.55]、軸 x',
          np.allclose(H.bar_center_local, [0.0, -0.285, 0.550])
          and np.allclose(H.bar_axis_local, [1.0, 0.0, 0.0]))

    src = _FakeSource()
    rdr = SameStepPoseReader(src, H, design_rel=np.array([0.0, 0.0, -0.0147]))

    # 1 把手 ≠ 抽屜原點；且隨抽屜姿態旋轉
    src.d = (np.array([10.5, 9.0, 0.0]), np.array([1.0, 0, 0, 0]))
    src.g = (np.array([10.5, 8.0, 0.55]), np.array([1.0, 0, 0, 0]))
    rec = rdr.read()
    check('讀數有效', rec.valid)
    check('把手世界位置 = 抽屜位姿 × 固定變換（非抽屜原點）',
          np.allclose(rec.handle[:3, 3], [10.5, 9.0 - 0.285, 0.550])
          and not np.allclose(rec.handle[:3, 3], rec.drawer[:3, 3]))
    c, s_ = math.cos(math.radians(30)), math.sin(math.radians(30))
    src.d = (np.array([10.5, 9.0, 0.0]),
             np.array([math.cos(math.radians(15)), 0, 0, math.sin(math.radians(15))]))
    rec2 = rdr.read()
    exp = np.array([10.5, 9.0, 0.0]) + np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]]) \
        @ np.array([0.0, -0.285, 0.550])
    check('抽屜轉 30° 時把手偏移一併旋轉', np.allclose(rec2.handle[:3, 3], exp))

    # 2 相對位姿 = 夾爪^-1 × 把手
    src.d = (np.array([10.5, 9.0, 0.0]), np.array([1.0, 0, 0, 0]))
    rec = rdr.read()
    check('相對位置 = 把手世界 − 夾爪世界（夾爪為單位姿態）',
          np.allclose(rec.relative[:3, 3], [0.0, 0.715, 0.0]))

    # 3 不同步、NaN、例外 → 無效，且**不補值、不沿用**
    src.g_step_offset = 1
    rec_bad = rdr.read()
    check('夾爪與抽屜步序不一致 → 無效', not rec_bad.valid and '不同步' in rec_bad.invalid_reason)
    check('無效時 relative 為 None（未沿用上一筆）', rec_bad.relative is None)
    src.g_step_offset = 0
    src.g = (np.array([np.nan, 8.0, 0.55]), np.array([1.0, 0, 0, 0]))
    rec_nan = rdr.read()
    check('位置含 NaN → 無效', not rec_nan.valid and '非有限' in rec_nan.invalid_reason)
    src.raise_on_drawer = True
    rec_exc = rdr.read()
    check('讀取例外 → 無效（不以 FK 補值）',
          not rec_exc.valid and '例外' in rec_exc.invalid_reason)
    src.raise_on_drawer = False
    src.g = (np.array([10.5, 8.0, 0.55]), np.array([1.0, 0, 0, 0]))
    rec_ok = rdr.read()
    check('無效之後的有效讀數重新計算，未帶入舊值',
          rec_ok.valid and np.allclose(rec_ok.relative[:3, 3], [0.0, 0.715, 0.0]))
    check('無效筆數已計數', rdr.n_invalid == 3)

    # 3b 步序不一致時**不得**記成同步；零四元數不得判為有效
    src.g_step_offset = 1
    rec_ns = rdr.read()
    row_ns = rdr.row(rec_ns)
    check('不同步時 same_step_read 記為 False',
          rec_ns.same_step is False and row_ns[LOG_COLS.index('same_step_read')] is False)
    src.g_step_offset = 0
    src.g = (np.array([10.5, 8.0, 0.55]), np.array([0.0, 0.0, 0.0, 0.0]))
    rec_zq = rdr.read()
    check('零四元數 → 無效（不得標為有效）',
          not rec_zq.valid and '四元數' in rec_zq.invalid_reason
          and rec_zq.relative is None)
    src.g = (np.array([10.5, 8.0, 0.55]), np.array([1.0, 0, 0]))
    rec_dim = rdr.read()
    check('四元數維度不符 → 無效', not rec_dim.valid and '維度' in rec_dim.invalid_reason)
    src.g = (np.array([10.5, 8.0]), np.array([1.0, 0, 0, 0]))
    rec_dim2 = rdr.read()
    check('位置維度不符 → 無效', not rec_dim2.valid and '維度' in rec_dim2.invalid_reason)
    src.raise_on_drawer = True
    rec_ex2 = rdr.read()
    check('讀取例外時 same_step_read 記為 False', rec_ex2.same_step is False)
    src.raise_on_drawer = False
    src.g = (np.array([10.5, 8.0, 0.55]), np.array([1.0, 0, 0, 0]))
    rec_ok2 = rdr.read()
    check('有效讀數的 same_step_read 為 True',
          rec_ok2.same_step is True
          and rdr.row(rec_ok2)[LOG_COLS.index('same_step_read')] is True)

    # 4 兩套基準
    rdr.mark_attached(rec_ok)
    src.d = (np.array([10.5, 9.0 - 0.001, 0.0]), np.array([1.0, 0, 0, 0]))
    rec3 = rdr.read()
    d = rdr.datums(rec3)
    check('連接後基準：位置漂移 1.0 mm', abs(d['post_pos_m'] - 0.001) < 1e-12)
    try:
        rdr.mark_attached(rec_bad)
        check('無效讀數不得設定連接基準', False)
    except RuntimeError:
        check('無效讀數不得設定連接基準', True)

    # 5 與既有 FK 路徑交叉比對（固定底座 20 mm 趟；僅驗算式，不代表執行端來源）
    import json
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'src', 'ammr_wholebody_mpc'))
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    run = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'evaluation', 'runs', 'drawer_220102_offset20',
                       'sim', 'drawer_run.json')
    if os.path.exists(run):
        K = WholeBodyKinematics.from_urdf_file(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'models',
            'omni_bot_wholebody_expanded.urdf'))
        dd = json.load(open(run))
        ix = {k: j for j, k in enumerate(dd['log_cols'])}
        ph = [str(r[ix['phase']]) for r in dd['log']]
        k = max(i for i, p in enumerate(ph) if p == 'engage')
        row = dd['log'][k]
        park, unit = dd['park'], dd['pose']
        q = {'base_x': park[0], 'base_y': park[1], 'base_theta': park[2]}
        for j in range(6):
            q[f'joint{j+1}'] = row[ix[f'joint{j+1}']]
        Tg = K.fk(np.array([q.get(n, 0.0) for n in K.dof_names]), 'link_tcp')
        src.g = (Tg[:3, 3], rot_to_quat(Tg[:3, :3]))
        src.d = (np.array([unit[0], unit[1] - row[ix['opening']], 0.0]),
                 np.array([1.0, 0, 0, 0]))
        rec4 = rdr.read()
        prev = np.array([0.054, 1.414, -14.527]) / 1000.0    # 先前 FK 路徑的值
        diff = float(np.linalg.norm(rec4.relative[:3, 3] - prev))
        check(f'與既有 FK 路徑交叉比對：差 {1000*diff:.4f} mm', diff < 5e-6)
    else:
        check('交叉比對趟次存在', False)

    print('位姿讀取離線測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(selftest())
