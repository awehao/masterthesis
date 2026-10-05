#!/usr/bin/env python3
"""GC2：夾爪殼＋腕部相機／支架的最小靜態檢查（規格 results/vision/GC2_gripper_shell_spec.md draft-2）。

沿用 GC1 凍結函式（geometry_contract.py）與狀態語意，不改 GC1。只對網格凸包下結論：
凸包分離 ⇒ 指定姿態下網格也分離；凸包交疊只代表保守核對拒絕，不證明原網格或實際模擬一定碰撞。
殼與相機的模擬碰撞近似未讀回。結果分存 fingers（GC1）／shell／camera_assembly，三者全部通過才整體通過。
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml

import geometry_contract as GC
import object_target_geometry as OT

HERE = GC.HERE
REPO = GC.REPO
CONTRACT2_DEFAULT = os.path.join(HERE, 'geometry_contract_gc2_shell.yaml')
GC1_FREEZE = 'evaluation/results/vision/freeze_gc1_offline.sha256'
LOCKED2 = {
    'shell_stl': 'install/xarm_description/share/xarm_description/meshes/gripper/lite/visual/shell.stl',
    'camera_stl': 'install/xarm_description/share/xarm_description/meshes/camera/realsense/collision/d435_with_cam_stand.stl',
    'gc1_freeze': GC1_FREEZE,
}
# 元件 → (連桿, 鎖定網格鍵)
PARTS = {'shell': ('uflite_gripper_link', 'shell_stl'), 'camera_assembly': ('link_eef', 'camera_stl')}


class Reject(Exception):
    def __init__(self, why):
        super().__init__(why)
        self.why = why


def write_contract2(path=CONTRACT2_DEFAULT):
    if os.path.exists(path):
        raise FileExistsError(path)
    c = {'schema': 'gc2_contract/1', 'contract_id': 'gc2_shell_camera_v1',
         'scope': '夾爪殼＋腕部相機／支架（網格凸包）對抽屜櫃的靜態核對；不含手臂、軌跡、模擬',
         'files': {k: {'path': p, 'sha256': GC._sha(os.path.join(REPO, p))} for k, p in LOCKED2.items()}}
    with open(path, 'w') as f:
        yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
    return c


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                                     # URDF：固定軸 roll→pitch→yaw


def _origin_T(o):
    T = np.eye(4)
    if o is None:
        return T
    T[:3, :3] = _rpy(*[float(v) for v in (o.get('rpy') or '0 0 0').split()])
    T[:3, 3] = [float(v) for v in (o.get('xyz') or '0 0 0').split()]
    return T


def _urdf_parts(urdf, repo):
    """各元件：連桿在夾爪連桿座標的位姿 T_gl,link（由固定關節導出）、collision origin、scale、網格路徑。"""
    root = ET.parse(urdf).getroot()
    J = {j.get('name'): j for j in root.findall('joint')}
    L = {l.get('name'): l for l in root.findall('link')}
    gf = J['gripper_fix']
    if gf.get('type') != 'fixed' or gf.find('parent').get('link') != 'link_eef' or gf.find('child').get('link') != 'uflite_gripper_link':
        raise Reject('gripper_fix_unexpected')
    T_eef_gl = _origin_T(gf.find('origin'))
    T_gl_link = {'uflite_gripper_link': np.eye(4), 'link_eef': np.linalg.inv(T_eef_gl)}
    out = {}
    for part, (link, key) in PARTS.items():
        cols = L[link].findall('collision')
        if len(cols) != 1:
            raise Reject(f'{part}_collision_count_{len(cols)}')
        col = cols[0]
        mesh = col.find('geometry/mesh')
        if mesh is None:
            raise Reject(f'{part}_collision_not_mesh')
        fn = mesh.get('filename')
        path = fn[len('file://'):] if fn.startswith('file://') else fn
        if os.path.abspath(path) != os.path.abspath(os.path.join(REPO, LOCKED2[key])):
            raise Reject(f'{part}_mesh_path_not_locked')
        scale = np.array([float(v) for v in (mesh.get('scale') or '1 1 1').split()])
        out[part] = {'link': link, 'T_gl_link': T_gl_link[link], 'T_link_col': _origin_T(col.find('origin')),
                     'scale': scale, 'key': key}
    return out


def load_contract2(path=CONTRACT2_DEFAULT, repo=REPO, gc1_path=GC.CONTRACT_DEFAULT):
    """GC2 契約＋GC1 契約；GC1 凍結清單本身與清單內每個檔案逐項重算。"""
    try:
        try:
            c = yaml.safe_load(open(path))
        except (OSError, yaml.YAMLError):
            raise Reject('contract2_unreadable')
        if not isinstance(c, dict) or c.get('schema') != 'gc2_contract/1' or not isinstance(c.get('contract_id'), str):
            raise Reject('contract2_schema')
        if not isinstance(c.get('files'), dict) or set(c['files']) != set(LOCKED2):
            raise Reject('contract2_incomplete:files')
        bad = []
        for k, ent in c['files'].items():
            if not isinstance(ent, dict) or ent.get('path') != LOCKED2[k] or not isinstance(ent.get('sha256'), str):
                raise Reject(f'contract2_incomplete:files:{k}')
            p = os.path.join(repo, ent['path'])
            if not os.path.isfile(p) or GC._sha(p) != ent['sha256']:
                bad.append(k)
        if bad:
            raise Reject('contract2_version_mismatch:' + ','.join(sorted(bad)))
        # GC1 凍結清單內每個檔案
        n = 0
        for line in open(os.path.join(repo, GC1_FREEZE)):
            if not line.strip():
                continue
            h, rel = line.rstrip('\n').split('  ', 1)
            p = os.path.join(repo, rel)
            if not os.path.isfile(p) or GC._sha(p) != h:
                raise Reject(f'gc1_freeze_entry_mismatch:{rel}')
            n += 1
        C1 = GC.load_contract(gc1_path, repo=repo)
        if not C1['ok']:
            raise Reject('gc1:' + C1['why'])
        parts = _urdf_parts(os.path.join(repo, C1['contract']['files']['urdf']['path']), repo)
        for part, P in parts.items():
            V = GC._stl_vertices(os.path.join(repo, LOCKED2[P['key']])) * P['scale']
            T = P['T_gl_link'] @ P['T_link_col']               # 網格座標 → 夾爪連桿座標
            Vg = V @ T[:3, :3].T + T[:3, 3]
            P['V_mesh_gl'] = Vg
            P['shape_gl'] = GC._shape(Vg)                       # 整網格凸包（⊇ 網格）
        return {'ok': True, 'contract': c, 'gc1': C1, 'parts': parts, 'gc1_freeze_entries': n}
    except Reject as r:
        return {'ok': False, 'why': r.why}
    except GC.Reject as r:
        return {'ok': False, 'why': r.why}


def part_shapes(C2, T_WE):
    """T_W,gl = T_WE · T_tcp,gl（GC1 同一 URDF 鏈；T_EG = I）。"""
    g = C2['gc1']['grip']
    T_W_gl = T_WE @ np.linalg.inv(OT.trans(g['joint_tcp']['xyz']))
    return {part: GC.xf_shape(T_W_gl, P['shape_gl']) for part, P in C2['parts'].items()}


def check2(C2, T_WO, q, T_HG, a_H, s, frame='asset_prim'):
    """回傳 dict：ok、why、fingers（GC1 輸出）、shell、camera_assembly。"""
    out = {'ok': False, 'scope': 'gc2_shell_camera_convex_hull_static'}
    try:
        if not isinstance(C2, dict) or C2.get('ok') is not True or not all(k in C2 for k in ('gc1', 'parts', 'contract')):
            raise Reject('contract2_not_loaded')
        f = GC.check(C2['gc1'], T_WO, q, T_HG, a_H, s, frame=frame)
        out['fingers'] = f
        if 'targets' not in f:
            raise Reject('fingers:' + str(f.get('why')))         # 輸入不合法，未產生目標
        bad = [] if f['ok'] else ['fingers:' + f['why']]
        T_WO = OT.rigid(T_WO, 'T_WO')
        O = GC.object_shapes(C2['gc1'], T_WO, float(q))
        for part in C2['parts']:
            out[part] = {}
        for tag in ('s0', 's'):
            if tag == 's' and f['targets']['s_m'] == 0.0:
                continue
            S = part_shapes(C2, f['targets'][tag])
            for part, A in S.items():
                pp = {f'{part}|{on}': GC.classify(A, sh) for on, sh in O.items()}
                out[part][tag] = pp
                for k, v in pp.items():
                    if v['status'] == 'penetrating':
                        bad.append(f'penetration_at_target:{tag}:{k}')
                    elif v['status'] == 'indeterminate':
                        bad.append(f'indeterminate_at_target:{tag}:{k}')
        if bad:
            raise Reject(';'.join(bad))
        out['ok'] = True
        out['claim'] = ('在指定目標下，夾爪殼及腕部相機／支架通過凸包靜態核對；不代表整機碰撞安全、軌跡可達或物理夾持成功。')
        return out
    except (Reject, GC.Reject, OT.Reject) as r:
        out['why'] = r.why
        return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--write-contract', action='store_true')
    a = ap.parse_args()
    if a.write_contract:
        write_contract2()
        print('寫入', CONTRACT2_DEFAULT)
