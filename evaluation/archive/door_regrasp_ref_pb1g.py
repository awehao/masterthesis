#!/usr/bin/env python3
"""參考包 pb1g（mode 4 重設計 R1–R3；Codex 20261029_154350 核可接線準備）：pb1f 只換 mode 4，其餘段、登錄欄位原樣。
  段 '4'  ＝ R1（door_regrasp_m4r_v3：收臂＋轉向＋依朝向前進；末速沿 R2 起點切線 V_HAND）；
  段 '41' ＝ R3（自導航交回點起、末端 B1＋STOW 靜止，接既有段 5）；
  每列加 'base_v'（世界座標解析底盤速度，與取樣同一條軌跡；任務端前饋用它，不以差分近似）；'tcp'＝FK(base＋arm, link_tcp)。
  'm4r'：R2 折線（導航第二段任務，每 0.05 m 取點）、交導航／交回位姿、V_HAND、bridge_r3（R3 起算銜接範圍：導航交回的實際位姿相對 R3 起點）。
  時間：R1／R3 列為 20 Hz 取樣＋精確終止時刻（最後一間隔 < 0.05 s）。
    python3 evaluation/door_regrasp_ref_pb1g.py   # 寫 results/door/door_regrasp_ref_pb1g.json（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb1f.json')
M4R = os.path.join(HERE, 'results', 'door', 'door_regrasp_m4r_v3.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb1g.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
BRIDGE_R3 = {'base_m': 0.05, 'base_deg': 5.0, 'arm_rad': 0.05, 'door_deg': 0.7}


def sha(p):
    import hashlib
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def seg_of(rows, K, qf, qd, stage):
    X = np.array([r['x'] for r in rows])
    return {'t': [float(r['t']) for r in rows],
            'tcp': [np.asarray(K.fk(list(map(float, x)), 'link_tcp'), float).reshape(-1).tolist() for x in X],
            'base': X[:, :3].tolist(), 'arm': X[:, 3:9].tolist(), 'base_v': [list(map(float, r['v'][:3])) for r in rows],
            'qf': [qf] * len(rows), 'q_door_deg': [qd] * len(rows), 'src_index': list(range(len(rows))), 'stage': [stage] * len(rows)}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p = json.load(open(SRC))
    m = json.load(open(M4R))
    K = WholeBodyKinematics.from_urdf_file(URDF)
    qf = float(p['segments']['4']['qf'][-1])
    qd = float(p['q_ref_deg'])
    old4 = p['segments']['4']
    p['segments']['4'] = seg_of(m['R1']['rows'], K, qf, qd, 'R1')
    p['segments']['41'] = seg_of(m['R3']['rows'], K, qf, qd, 'R3')
    s3e, s4s = p['segments']['3'], p['segments']['4']
    j3 = {'base': float(np.max(np.abs(np.asarray(s3e['base'][-1]) - np.asarray(s4s['base'][0])))),
          'arm': float(np.max(np.abs(np.asarray(s3e['arm'][-1]) - np.asarray(s4s['arm'][0]))))}
    s41, s5 = p['segments']['41'], p['segments']['5']
    _db = np.asarray(s41['base'][-1]) - np.asarray(s5['base'][0])
    _db[2] = math.atan2(math.sin(_db[2]), math.cos(_db[2]))                       # 偏航 270° ≡ −90°
    j5 = {'base': float(np.max(np.abs(_db))),
          'arm': float(np.max(np.abs(np.asarray(s41['arm'][-1]) - np.asarray(s5['arm'][0]))))}
    P = np.array(m['path']['xy'])
    ih, ir = m['handover_nav']['i'], m['handover_wb']['i']
    seg = P[ih:ir + 1]
    keep = [0]
    for i in range(1, len(seg)):
        if np.linalg.norm(seg[i] - seg[keep[-1]]) >= 0.05 - 1e-9:
            keep.append(i)
    if keep[-1] != len(seg) - 1:
        keep.append(len(seg) - 1)
    p['m4r'] = {'source': {'path': os.path.relpath(M4R, WS), 'sha256': sha(M4R)}, 'V_HAND': m['params']['V_HAND'],
                'R2': {'vertices': seg[keep].tolist(), 'goal': [float(seg[-1][0]), float(seg[-1][1]), float(m['handover_wb']['yaw'])]},
                'handover_nav': {'xy': m['handover_nav']['xy'], 'yaw': m['handover_nav']['yaw']},
                'handover_wb': {'xy': m['handover_wb']['xy'], 'yaw': m['handover_wb']['yaw']},
                'bridge_r3': BRIDGE_R3, 'join_3_to_R1_max_abs': j3, 'join_R3_to_5_max_abs': j5,
                'old_mode4': {'n': len(old4['t']), 'T_s': float(old4['t'][-1])},
                'not_claimed': '名目時間不含閉迴路、交棒與 5 Hz 振盪；R1 仍近原地轉 −152.8°'}
    p['schema'] = str(p.get('schema')) + '+pb1g'
    p['derived_from'] = {'path': os.path.relpath(SRC, WS), 'sha256': sha(SRC)}
    json.dump(p, open(OUT, 'w'), ensure_ascii=False)
    print(f"寫 {OUT}；段4 {s4s['t'][-1]:.2f} s、段41 {s41['t'][-1]:.2f} s；接點 3→R1 {j3}；R3→5 {j5}；R2 折線 {len(keep)} 點")


if __name__ == '__main__':
    main()
