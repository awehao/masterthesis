"""把**既有趟次的實際資料**餵進 v1 狀態機的離線試跑（不開模擬器）。

用途：在接上協同執行端之前，先看 v1 判準在真實幾何下會給出什麼。

**口徑**：相對位姿由「固定底盤位姿 ＋ 關節角」FK 重建，
是**模型重建值**，不是同一物理步的實測世界位姿；底盤在該趟是固定的。
因此本檔只能當**判準可滿足性的預覽**，不是協同案例的驗證。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa
from coman_handover_state import Frame, HandoverMachine, load_spec       # noqa
from coman_pose_reader import HandleTransform, homog                     # noqa


def main(run='drawer_220102_offset20') -> int:
    spec = load_spec()                      # v1，已凍結
    M = HandoverMachine(spec)
    import yaml
    dspec = yaml.safe_load(open(os.path.join(
        WS, 'src/my_omnibot_description/config/drawer_unit.yaml'), encoding='utf-8'))
    H = HandleTransform.from_spec(dspec)
    K = WholeBodyKinematics.from_urdf_file(
        os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    d = json.load(open(os.path.join(HERE, 'runs', run, 'sim', 'drawer_run.json')))
    ix = {k: j for j, k in enumerate(d['log_cols'])}
    park, unit = d['park'], d['pose']
    rows = d['log']
    ph = [str(r[ix['phase']]) for r in rows]
    out = []
    for r, p in zip(rows, ph):
        q = {'base_x': park[0], 'base_y': park[1], 'base_theta': park[2]}
        for j in range(6):
            q[f'joint{j+1}'] = r[ix[f'joint{j+1}']]
        Tg = K.fk(np.array([q.get(n, 0.0) for n in K.dof_names]), 'link_tcp')
        Td = homog(np.array([unit[0], unit[1] - r[ix['opening']], 0.0]), np.eye(3))
        Th = H.world(Td[:3, 3], Td[:3, :3])
        rel = np.linalg.inv(Tg) @ Th
        f = Frame(t=float(r[ix['t']]), cmd_age_safety_s=0.0, cmd_age_endpoint_s=0.0,
                  rel_pos_tool=rel[:3, 3],
                  bar_axis_tool=rel[:3, :3] @ H.bar_axis_local,
                  opening_m=float(r[ix['opening']]), f_norm_n=float(r[ix['f_norm']]),
                  attached=p in ('postengage', 'pull', 'hold', 'release'),
                  rel_rot_tool=rel[:3, :3],
                  gripper_pos_world=Tg[:3, 3], gripper_rot_world=Tg[:3, :3])
        fl = M.step(f)
        out.append((f.t, p, M.margin(f.rel_pos_tool, f.bar_axis_tool),
                    M.insertion_dev(f.rel_pos_tool), M.tilt_deg(f.bar_axis_tool),
                    fl.in_validated_range, fl.handover_cond_now, fl.handover_pass,
                    fl.hold_tracking_pass, M.phase))
    A = np.array([[o[2], o[3], o[4]] for o in out], float)
    print(f'趟次 {run}：{len(out)} 筆（**FK 模型重建**，非同步實測）')
    print(f'  合併餘裕 m：min {1000*A[:,0].min():.3f}、max {1000*A[:,0].max():.3f} mm'
          f'（門檻 ≥ {1000*M.m_min:.1f} mm）')
    print(f'  插入深度偏差：min {1000*A[:,1].min():.3f}、max {1000*A[:,1].max():.3f} mm'
          f'（門檻 ≤ {1000*M.h6:.1f} mm）')
    print(f'  桿軸傾角：max {A[:,2].max():.3f}°（已核對範圍 ≤ '
          f'{M.rng["tilt_deg_max"]}°）')
    for p in ('engage', 'postengage', 'pull', 'hold', 'release'):
        sel = [o for o in out if o[1] == p]
        if not sel:
            continue
        n_in = sum(1 for o in sel if o[5])
        n_cond = sum(1 for o in sel if o[6])
        print(f'  {p:11s} n={len(sel):4d} 在核對範圍 {n_in:4d}'
              f'  交接條件成立 {n_cond:4d}  m 中位 '
              f'{1000*np.median([o[2] for o in sel]):7.3f} mm')
    print(f'  狀態機最終相位 {M.phase}；交接曾通過 '
          f'{"是" if any(o[7] for o in out) else "否"}；'
          f'保持曾合格 {"是" if any(o[8] for o in out) else "否"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
