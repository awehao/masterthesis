"""近接配對的實際間距量測（離線，用既有趟次資料）。

配對：夾爪殼 ↔ 抽屜面板、手指 ↔ 把手支柱。

**口徑**：本檔量的是**既有固定底座趟次**在該配對上的歷史最小間距，
屬**可行性參考**，**不是**安全門檻的推導依據。
幾何近似、量測誤差與所需餘裕另外交代（見輸出末段）。
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa

MESH_DIR = os.path.expanduser('~/masterthesis/install/xarm_description/share/'
                              'xarm_description/meshes/gripper/lite/visual')


def load_stl(name):
    with open(os.path.join(MESH_DIR, name), 'rb') as f:
        f.read(80)
        n = struct.unpack('<I', f.read(4))[0]
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    return np.frombuffer(raw[:, 12:48].tobytes(),
                         dtype='<f4').reshape(n, 3, 3).astype(float)


def sample_tri(T, spacing):
    pts = []
    for A, B, C in T:
        n = max(int(np.ceil(max(np.linalg.norm(B - A),
                                np.linalg.norm(C - A)) / spacing)), 1)
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
        m = (i + j) <= n
        u, v = (i[m] / n)[:, None], (j[m] / n)[:, None]
        pts.append(A[None] + u * (B - A)[None] + v * (C - A)[None])
    return np.unique(np.round(np.concatenate(pts), 6), axis=0)


def box_dist(P, center, size):
    """點到軸對齊方塊表面的距離（外部為正，內部為 0）。"""
    d = np.abs(P - center[None]) - (size / 2.0)[None]
    return np.linalg.norm(np.maximum(d, 0.0), axis=1)


def main(run='drawer_220102_offset20', every=10, spacing=0.002) -> int:
    spec = yaml.safe_load(open(os.path.join(
        WS, 'src/my_omnibot_description/config/drawer_unit.yaml'), encoding='utf-8'))
    d = json.load(open(os.path.join(HERE, 'runs', run, 'sim', 'drawer_run.json')))
    ix = {k: j for j, k in enumerate(d['log_cols'])}
    park, unit = d['park'], d['pose']
    K = WholeBodyKinematics.from_urdf_file(
        os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    shell = sample_tri(load_stl('shell.stl'), spacing)
    f1 = sample_tri(load_stl('finger1.stl'), spacing)
    f2 = sample_tri(load_stl('finger2.stl'), spacing)
    print(f'取樣點：殼 {len(shell)}、指1 {len(f1)}、指2 {len(f2)}'
          f'（格距 ≤ {1000*spacing:.1f} mm）')

    panel = [np.array(x, float) for x in spec['drawer']['body'][0][1:]]
    posts = {p[0]: [np.array(p[1], float), np.array(p[2], float)]
             for p in spec['drawer']['handle']['posts']}
    rows = d['log']
    ph = [str(r[ix['phase']]) for r in rows]
    res = {}
    for k in range(0, len(rows), every):
        r = rows[k]
        q = {'base_x': park[0], 'base_y': park[1], 'base_theta': park[2]}
        for j in range(6):
            q[f'joint{j+1}'] = r[ix[f'joint{j+1}']]
        qv = np.array([q.get(n, 0.0) for n in K.dof_names])
        Tsh = K.fk(qv, 'uflite_gripper_link')
        T1 = K.fk(qv, 'uflite_finger1')
        T2 = K.fk(qv, 'uflite_finger2')
        op = float(r[ix['opening']])
        off = np.array([unit[0], unit[1] - op, 0.0])
        Psh = (Tsh[:3, :3] @ shell.T).T + Tsh[:3, 3]
        P1 = (T1[:3, :3] @ f1.T).T + T1[:3, 3]
        P2 = (T2[:3, :3] @ f2.T).T + T2[:3, 3]
        pairs = {
            'shell↔front_panel': box_dist(Psh, off + panel[0], panel[1]).min(),
            'finger1↔post_l': box_dist(P1, off + posts['post_l'][0],
                                       posts['post_l'][1]).min(),
            'finger2↔post_r': box_dist(P2, off + posts['post_r'][0],
                                       posts['post_r'][1]).min(),
            # **關鍵配對**：資產註解的 27.8 mm 指的是**指尖**對面板，不是殼
            'finger1↔front_panel': box_dist(P1, off + panel[0], panel[1]).min(),
            'finger2↔front_panel': box_dist(P2, off + panel[0], panel[1]).min(),
        }
        p = ph[k]
        for name, v in pairs.items():
            res.setdefault((name, p), []).append(v)

    print(f'\n趟次 {run}（**固定底座、FK 重建**；歷史值＝可行性參考，非門檻依據）')
    order = ['approach', 'engage', 'postengage', 'pull', 'hold', 'release',
             'retreat']
    for name in ('shell↔front_panel', 'finger1↔front_panel',
                 'finger2↔front_panel', 'finger1↔post_l', 'finger2↔post_r'):
        print(f'  {name}')
        for p in order:
            v = res.get((name, p))
            if not v:
                continue
            print(f'    {p:11s} n={len(v):3d}  最小 {1000*min(v):7.3f} mm'
                  f'  中位 {1000*np.median(v):7.3f} mm')
    print('\n近似與誤差（填門檻前必須一併交代）：')
    print(f'  * 抽屜部件以**軸對齊方塊**近似（面板、支柱），未含倒角與細節')
    print(f'  * 機器人側以視覺網格取樣，格距 ≤ {1000*spacing:.1f} mm ⇒ '
          f'距離可能高估至多約 {1000*spacing:.1f} mm')
    print(f'  * 相對位姿為 FK 重建（底盤固定假設），非同一物理步實測')
    print(f'  * 每 {every} 筆取樣一次，非逐筆；瞬態最小值可能落在未取樣處')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
