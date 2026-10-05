#!/usr/bin/env python3
"""DL0 第一個子包：腳本合成的腕部相機**觀測路徑**（不是控制或接觸軌跡）→ 逐步姿態檔，供 run_wrist_v0.sh --wrist-replay 只渲染擷取。

每條序列先設計相機 3D 路徑（距把手 d、觀看方向、俯仰、偏離與滾轉）與底盤路徑（相機後方 0.25 m），
再逐格（5 Hz）以最小平方 IK 解底盤偏航＋六關節（關節在安全限位內縮 0.05 rad），使相機光學座標達到目標位姿；
相鄰格之間以線性內插產生 10 ms 物理步。抽屜關閉、手指張開、無接觸。
擷取前以 FK 投影預估每格類別（完整入鏡／裁切／無把手）——**不渲染、不看影像**。

    python3 evaluation/dl0_paths.py            # 產生三條序列＋登錄檔 results/vision/DL0_paths_registration.yaml
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
sys.path.insert(0, HERE)
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                    # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from wrist_v0_capture import R_to_wxyz, wxyz_to_R                     # noqa: E402

OPT = 'camera_color_optical_frame'
H_C = np.array([0.0, 1.165, 0.55])         # 抽屜關閉時把手中心（truth.json M_world_handle_prim）
AX = np.array([1.0, 0.0, 0.0])
LEN, W, HH = 0.200, 640, 480
KM = np.array([[465.6028747558594, 0, 320.0], [0, 465.60284423828125, 240.0], [0, 0, 1]])
DT_FRAME, DT_PHYS = 0.2, 0.01
Q_FINGER_OPEN = [0.0089, 0.0089]           # 實錄接近段的手指開度
OUT = os.path.join(HERE, 'results', 'vision', 'dl0_paths')
Kin = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
LO = LITE6_SAFE.lower + 0.05
HI = LITE6_SAFE.upper - 0.05


def look_R(z, x_hint):
    """光學座標（ROS：x 右、y 下、z 前）：z＝觀看方向；x 盡量對齊 x_hint（投影到垂直 z 的平面）。"""
    z = z / np.linalg.norm(z)
    x = x_hint - (x_hint @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def rotz_about(R, ang):
    """繞光軸 z 轉 ang（滾轉）。"""
    c, s = math.cos(ang), math.sin(ang)
    return R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def solve(base_xyth, p_t, R_t, seed):
    """底盤位姿固定（偏航＝朝向把手的方位），只解六關節；以前一格的解為初值並加極小連續性正則，避免跳到別的 IK 解支。
    回傳 (j1..j6, pos_err_m, rot_err_deg)。"""
    def pose_res(v):
        q = np.r_[base_xyth, v]
        T = Kin.fk(q, OPT)
        dp = (T[:3, 3] - p_t) / 0.01
        Rr = R_t.T @ T[:3, :3]
        ang = math.acos(max(-1.0, min(1.0, (np.trace(Rr) - 1) / 2)))
        w = (np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]]) / (2 * math.sin(ang))
             * ang if ang > 1e-9 else np.zeros(3))
        return np.r_[dp, np.degrees(w)]

    def res(v):
        return np.r_[pose_res(v), 1e-3 * (v - seed)]
    r = least_squares(res, np.clip(seed, LO + 1e-6, HI - 1e-6), bounds=(LO, HI), xtol=1e-12, ftol=1e-12, max_nfev=600)
    e = pose_res(r.x)
    return r.x, float(np.linalg.norm(e[:3]) * 0.01), float(np.linalg.norm(e[3:]))


def predict(q):
    """FK 投影預估：(dist_m, 類別, 兩端是否入鏡, 中心是否入鏡)。"""
    T = Kin.fk(q, OPT)
    R, t = T[:3, :3], T[:3, 3]

    def proj(p):
        pc = R.T @ (p - t)
        return None if pc[2] <= 0.05 else (KM @ pc)[:2] / pc[2]
    inside = (lambda uv: uv is not None and 0 <= uv[0] < W and 0 <= uv[1] < HH)
    ends = [inside(proj(H_C + s * AX * LEN / 2)) for s in (-1, 1)]
    cen = inside(proj(H_C))
    samples = [inside(proj(H_C + s * AX * LEN / 2)) for s in np.linspace(-1, 1, 21)]
    d = float(np.linalg.norm(H_C - t))
    cat = ('no_handle' if not any(samples) else 'full' if all(ends) else 'truncated')
    return d, cat, ends, cen


def make_seq_once(name, n_frames, cam_fn, x_hint_fn, roll_fn, aim_fn, standoff, write=True):
    """cam_fn(i)→相機位置；aim_fn(i)→觀看目標點；x_hint_fn(i)→影像 x 方向提示；roll_fn(i)→額外滾轉。"""
    seed = np.array([-0.031, 0.916, 1.487, -0.358, -1.03, -1.392])   # 實錄 d1s4b_M 約 32 s（ALIGN 附近）的手臂姿態
    frames, bad = [], []
    yaw_prev = None
    for i in range(n_frames):
        p = cam_fn(i)
        a = aim_fn(i)
        R = rotz_about(look_R(a - p, x_hint_fn(i)), roll_fn(i))
        h_xy = (H_C - p)[:2]
        u_xy = h_xy / max(np.linalg.norm(h_xy), 1e-9)                # 底盤朝向把手（與相機觀看目標無關，保持路徑平順）
        base_xy = p[:2] - standoff * u_xy
        yaw = math.atan2(u_xy[1], u_xy[0])
        if yaw_prev is not None:
            yaw = yaw_prev + ((yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi)
        yaw_prev = yaw
        base = np.r_[base_xy, yaw]
        if i == 0:                                                     # 第一格多個初值，取殘差最小者
            cands = [solve(base, p, R, s0) for s0 in (seed, np.array([0.0, 0.5, 1.0, 0.0, -1.0, 0.0]),
                                                       np.array([0.0, 0.2, 0.6, 0.0, -1.2, 0.0]))]
            x, pe, re = min(cands, key=lambda c: c[1] / 0.002 + c[2] / 0.5)
        else:
            x, pe, re = solve(base, p, R, seed)
        seed = x
        q = np.r_[base, x]
        d, cat, ends, cen = predict(q)
        frames.append({'i': i, 't': round(i * DT_FRAME, 3), 'q': q.tolist(), 'pos_err_mm': round(pe * 1e3, 3),
                       'rot_err_deg': round(re, 3), 'dist_m': round(d, 3), 'pred': cat})
        if pe > 0.002 or re > 0.5:
            bad.append(i)
    # 逐物理步內插（底盤偏航取最短角差）
    steps = []
    k = 0
    for a_, b_ in zip(frames[:-1], frames[1:]):
        qa, qb = np.array(a_['q']), np.array(b_['q'])
        dq = qb - qa
        dq[2] = (dq[2] + math.pi) % (2 * math.pi) - math.pi
        n = int(round(DT_FRAME / DT_PHYS))
        for j in range(n):
            qq = qa + dq * (j / n)
            steps.append([k, round(10.0 + k * DT_PHYS, 4), [float(qq[0]), float(qq[1]), float(qq[2])],
                          [float(v) for v in qq[3:]], 0.0, Q_FINGER_OPEN])
            k += 1
    if not write:
        return frames, bad
    rec = {'schema': 'dl0_synthetic_observation_path/1', 'name': name, 'standoff_m': standoff,
           'note': '腳本合成的腕部相機觀測路徑（IK 解出的底盤＋關節姿態）；不是實際控制或接觸軌跡；抽屜關閉、手指張開',
           'steps_cols': ['step', 'sim_t', 'base_xyth', 'q_arm_meas', 'opening_m', 'q_finger'], 'steps': steps}
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'{name}.json')
    json.dump(rec, open(p, 'w'))
    cats = {}
    for f in frames:
        cats[f['pred']] = cats.get(f['pred'], 0) + 1
    return {'file': os.path.relpath(p, os.path.join(HERE, '..')), 'sha256': hashlib.sha256(open(p, 'rb').read()).hexdigest(),
            'base_standoff_m（相機後方，沿朝向把手方向）': standoff,
            'n_frames_5hz': len(frames), 'n_phys_steps': len(steps), 'sim_t_window': [steps[0][1], steps[-1][1]],
            'ik_bad_frames': bad, 'ik_pos_err_mm_max': max(f['pos_err_mm'] for f in frames),
            'ik_rot_err_deg_max': max(f['rot_err_deg'] for f in frames),
            'max_joint_step_rad_per_frame': round(max(float(np.abs(np.array(b['q'][3:]) - np.array(a['q'][3:])).max())
                                                      for a, b in zip(frames[:-1], frames[1:])), 3),
            'max_base_yaw_step_rad_per_frame': round(max(abs((b['q'][2] - a['q'][2] + math.pi) % (2 * math.pi) - math.pi)
                                                         for a, b in zip(frames[:-1], frames[1:])), 3),
            'dist_m_range': [min(f['dist_m'] for f in frames), max(f['dist_m'] for f in frames)],
            'predicted_categories（FK 投影，非渲染）': cats}, frames


def make_seq(name, n_frames, cam_fn, x_hint_fn, roll_fn, aim_fn):
    """同一序列內前距固定；在候選前距中選 IK 位置誤差最小者（再寫檔）。"""
    best = None
    for so in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        fr, bad = make_seq_once(name, n_frames, cam_fn, x_hint_fn, roll_fn, aim_fn, so, write=False)
        score = (len(bad), max(f['pos_err_mm'] for f in fr))
        if best is None or score < best[0]:
            best = (score, so)
    return make_seq_once(name, n_frames, cam_fn, x_hint_fn, roll_fn, aim_fn, best[1])


def smooth(a, b, s):
    s = min(1.0, max(0.0, s))
    s = s * s * (3 - 2 * s)
    return a + (b - a) * s


def main():
    reg = {'schema': 'dl0_paths_registration/1', 'date': '2026-10-05',
           'rule': 'Codex DL0 draft-2 審查：1 條開發補充＋2 條封存測試觀測序列；只渲染；腳本合成姿態照實標示；相機設定不變',
           'camera': 'URDF 掛載、640×480、hfov 69°、clipping 0.05–10 m、5 Hz（run_wrist_v0.sh 預設）',
           'handle': 'drawer_unit_bar26，抽屜關閉；把手中心 (0, 1.165, 0.55)、軸 x', 'sequences': {}}
    up = np.array([0, 0, 1.0])

    # ---- dev_near：正面，0.40→0.20 m（8 s）兩端入鏡，再 0.20→0.13 m（2 s）裁切過渡；橫桿沿影像寬邊；小幅左右偏移
    n = 51

    def d_dev(i):
        return smooth(0.40, 0.20, i / 40) if i <= 40 else smooth(0.20, 0.13, (i - 40) / 10)
    reg['sequences']['dev_near'] = {'用途': '開發補充（標註核對、訓練）', 'design': '正面；距離 0.40→0.20 m 兩端入鏡，0.20→0.13 m 裁切過渡；相機略高 3 cm；左右偏移 ±4 cm 正弦'}
    info, _ = make_seq('dev_near', n,
                       cam_fn=lambda i: H_C + np.array([0.04 * math.sin(i / 8), -d_dev(i), 0.03]),
                       aim_fn=lambda i: H_C, x_hint_fn=lambda i: AX, roll_fn=lambda i: 0.0)
    reg['sequences']['dev_near'].update(info)

    # ---- test_lateral：側向（方位 35°，由 −x 側進場）2.0→0.2 m；前 2 s 相機朝外（無把手），1 s 轉向把手
    n = 86
    u_lat = np.array([math.cos(math.radians(35)), math.sin(math.radians(35)), 0.0])

    def cam_lat(i):
        d = (2.0 if i < 15 else smooth(2.0, 0.2, (i - 15) / 60) if i <= 75 else smooth(0.2, 0.12, (i - 75) / 10))
        return H_C - d * u_lat + np.array([0, 0, 0.05])

    def aim_lat(i):
        away = cam_lat(i) + 1.5 * np.array([-0.6, 0.8, -0.1])
        return smooth(away, H_C, (i - 10) / 5)
    reg['sequences']['test_lateral'] = {'用途': '封存測試（模型與門檻凍結前不看）', 'design': '側向接近：觀看方位 35°（由 −x 側），2.0→0.2 m，再 2 s 0.2→0.12 m 裁切過渡；前 2 s 朝外、1 s 轉向；相機高於把手 5 cm'}
    info, _ = make_seq('test_lateral', n, cam_fn=cam_lat, aim_fn=aim_lat, x_hint_fn=lambda i: AX, roll_fn=lambda i: 0.0)
    reg['sequences']['test_lateral'].update(info)

    # ---- test_view：正面接近但改觀測朝向——相機高於把手、俯視；目標點偏離把手 0.1·d；滾轉 30°；前 2 s 朝外
    n = 86

    def d_view(i):
        return 2.0 if i < 15 else smooth(2.0, 0.2, (i - 15) / 60) if i <= 75 else smooth(0.2, 0.12, (i - 75) / 10)

    def cam_view(i):
        d = d_view(i)
        h = min(0.30, 0.25 * d + 0.08)
        return H_C + np.array([0.0, -math.sqrt(max(d * d - h * h, 1e-6)), h])

    def aim_view(i):
        target = H_C + np.array([0.1 * d_view(i), 0.0, 0.0])
        away = cam_view(i) + 1.5 * np.array([0.7, 0.7, -0.1])
        return smooth(away, target, (i - 10) / 5)
    reg['sequences']['test_view'] = {'用途': '封存測試（模型與門檻凍結前不看）', 'design': '正面接近、改觀測朝向：相機高於把手（俯視，最高 0.30 m）、目標偏離把手 0.1·d、滾轉 30°；2.0→0.2 m，再 2 s 0.2→0.12 m 裁切過渡；前 2 s 朝外、1 s 轉向'}
    info, _ = make_seq('test_view', n, cam_fn=cam_view, aim_fn=aim_view, x_hint_fn=lambda i: AX,
                       roll_fn=lambda i: math.radians(30))
    reg['sequences']['test_view'].update(info)

    reg['capture_commands'] = {k: (f"ROS_DOMAIN_ID=95 RUN_ID=dl0_{k} RUNNER=evaluation/run_wrist_v0.sh "
                                   f"WRIST_EXTRA='--wrist-replay $WS/{v['file']} --wrist-replay-t0 {v['sim_t_window'][0]} "
                                   f"--wrist-replay-t1 {v['sim_t_window'][1]}' bash evaluation/run_guarded.sh")
                               for k, v in reg['sequences'].items()}
    reg['test_sealing'] = ('test_lateral／test_view 擷取後只做技術完整性檢查（影格數、時間匹配、檔案雜湊）；不跑標籤、不跑辨識、'
                           '不看影像做取捨；模型與門檻凍結前不查看；不得依辨識結果更換路徑')
    import yaml
    p = os.path.join(HERE, 'results', 'vision', 'DL0_paths_registration.yaml')
    yaml.safe_dump(reg, open(p, 'w'), allow_unicode=True, sort_keys=False, width=200)
    for k, v in reg['sequences'].items():
        print(k, {kk: v[kk] for kk in ('n_frames_5hz', 'ik_bad_frames', 'ik_pos_err_mm_max', 'ik_rot_err_deg_max',
                                       'max_joint_step_rad_per_frame', 'max_base_yaw_step_rad_per_frame',
                                       'dist_m_range', 'predicted_categories（FK 投影，非渲染）')})
    print('寫出', p)


if __name__ == '__main__':
    main()
