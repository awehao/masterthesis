#!/usr/bin/env python3
"""GC1：手指—抽屜靜態局部幾何契約（bar26＋split 手指；規格 results/vision/GC1_geometry_contract_spec.md draft-3）。

純離線幾何：不接節點、不跑模擬、不改資產、不新訂門檻。通過時的正式說法：
bar26＋split 手指在指定開爪目標下，通過局部靜態幾何契約核對；不代表整機碰撞安全、軌跡可達或物理夾持成功。

座標：`O`＝櫃體座標（資產本地框），`q`＝抽屜開度。櫃體方塊 T_WO·T_OS；抽屜本體、支柱、橫桿 T_WO·Trans(q·u_O)·T_OS。
把手座標由契約重建 T_WH = T_WO·Trans(q·u_O)·Trans(c_bar)（DL1 資產 prim 約定）。
手指：同一版 finger_collision.split_hulls 作用於契約鎖定的 STL；連桿變換由契約鎖定的 URDF 讀出。

距離與交疊（凸模型的數值核對，不稱數學上精確）：
* 交疊：兩凸多面體半空間式的 Chebyshev LP（HiGHS）求交集內切半徑 t*。
* 距離：凸 QP（SLSQP）最近點，**不憑 success 放行**——以最近點差向量做分離平面證書：
  下界 = min_A n·a − max_B n·b，上界 = ‖p_A − p_B‖；間隙 > CERT_TOL ⇒ indeterminate。
* 有限圓柱以內接／外切 N 邊稜柱夾擠。
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np
import yaml

import object_target_geometry as OT
from finger_collision import inner_face_y, split_hulls  # noqa: F401（inner_face_y 供測試對照）

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
CONTRACT_DEFAULT = os.path.join(HERE, 'geometry_contract_bar26_split.yaml')

TOL = 1e-6           # 交疊／接觸判定（公尺）
CERT_TOL = 1e-7      # 距離證書上下界間隙
N_PRISM = 256        # 圓柱夾擠稜柱邊數（半徑差 r(1/cos(π/N) − 1) ≈ 1 µm＠r = 13 mm）
AABB_SKIP_M = 0.05   # AABB 間距超過此值：只回報下界（AABB 距離是合法下界）
BISECT_ITERS = 40

STATUSES = ('separated', 'contact', 'penetrating', 'indeterminate')
# contact＝容差內接觸或微小交疊（距離上界 ≤ TOL 且交集內切半徑 t* ≤ TOL），不是嚴格零穿透；t* 不等於穿透深度。
LOADED_KEYS = ('contract', 'spec', 'grip', 'fingers', 'finger_shapes', 'band', 'q_open', 'margin_min',
               'bar_r', 'bar_L', 'bar_c', 'u_O', 'q_lim')


class Reject(Exception):
    def __init__(self, why):
        super().__init__(why)
        self.why = why


# ------------------------------------------------------------------ 契約
LOCKED = {
    'asset': 'src/my_omnibot_description/config/drawer_unit_bar26.yaml',
    'urdf': 'evaluation/models/omni_bot_wholebody_expanded.urdf',
    'finger1_stl': 'install/xarm_description/share/xarm_description/meshes/gripper/lite/visual/finger1.stl',
    'finger2_stl': 'install/xarm_description/share/xarm_description/meshes/gripper/lite/visual/finger2.stl',
    'finger_collision': 'evaluation/finger_collision.py',
    'drawer_asset': 'evaluation/drawer_asset.py',
    'object_target_geometry': 'evaluation/object_target_geometry.py',
}
NOT_READ = [
    'drawer_unit_bar26.yaml grasp_surface.tcp_offset_along_tool_z（0.0147，10 mm 桿／凸包手指舊值）',
    'drawer_unit_bar26.yaml grasp_surface.finger_gap_open_m（0.0178，同上）',
    "drawer_task_node.py --grasp-depth-m 預設 0.01226 與 L462、L469 的「12.26 mm」來源字串（實錄以 task.json args.grasp_depth_m 為準）",
]


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_contract(path=CONTRACT_DEFAULT):
    """建立契約檔（已存在則拒絕，不覆寫）。"""
    if os.path.exists(path):
        raise FileExistsError(path)
    c = {
        'schema': 'gc1_contract/1',
        'contract_id': 'bar26_split_v1',
        'scope': 'bar26＋split 手指；手指—抽屜靜態局部幾何；不含夾爪殼、手臂、軌跡、閉爪力學',
        'handle_frame': 'asset_prim',
        'z_split_m': 0.008,
        'files': {k: {'path': p, 'sha256': _sha(os.path.join(REPO, p))} for k, p in LOCKED.items()},
        'adopted_asset_fields': ['handle.bar', 'handle.posts', 'cabinet', 'drawer.body', 'drawer.joint',
                                 'grasp_surface.finger_joint_open', 'grasp_surface.enclosure_margin_min_m'],
        'not_read': NOT_READ,
    }
    with open(path, 'w') as f:
        yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
    return c


def _stl_vertices(path):
    d = open(path, 'rb').read()
    n = struct.unpack('<I', d[80:84])[0]
    a = np.frombuffer(d[84:84 + n * 50], dtype=np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))
    return np.unique(a['v'].reshape(-1, 3).astype(float), axis=0)


def _stl_triangles(path):
    d = open(path, 'rb').read()
    n = struct.unpack('<I', d[80:84])[0]
    a = np.frombuffer(d[84:84 + n * 50], dtype=np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))
    return a['v'].reshape(-1, 3, 3).astype(float)


def _clip_poly_z(poly, zlo, zhi):
    """Sutherland–Hodgman：把多邊形（頂點列）裁到 zlo ≤ z ≤ zhi；回傳裁後頂點列（可能空）。"""
    def clip(pts, inside, inter):
        out = []
        for k in range(len(pts)):
            a, b = pts[k], pts[(k + 1) % len(pts)]
            ia, ib = inside(a), inside(b)
            if ia:
                out.append(a)
            if ia != ib:
                out.append(inter(a, b))
        return out

    def at(zc):
        return lambda a, b: a + (zc - a[2]) / (b[2] - a[2]) * (b - a)
    pts = [np.asarray(v, float) for v in poly]
    if zlo is not None:
        pts = clip(pts, lambda v: v[2] >= zlo, at(zlo)) if pts else []
    if zhi is not None:
        pts = clip(pts, lambda v: v[2] <= zhi, at(zhi)) if pts else []
    return pts


def clipped_regions(tris, z0, z1):
    """把網格三角形裁成三段（z ≤ z0、z0 ≤ z ≤ z1、z ≥ z1），回傳各段裁後頂點（含邊界交點）。"""
    reg = {'below': [], 'band': [], 'above': []}
    for t in tris:
        for name, lo, hi in (('below', None, z0), ('band', z0, z1), ('above', z1, None)):
            P = _clip_poly_z(t, lo, hi)
            if len(P):
                reg[name].extend(P)
    return {k: (np.array(v) if v else np.zeros((0, 3))) for k, v in reg.items()}


def contained(S, P, tol=1e-9):
    """點集 P 是否全在凸形狀 S 內（半空間殘差 ≤ tol）；回傳 (bool, 最大殘差)。"""
    if len(P) == 0:
        return True, 0.0
    r = float((P @ S['N'].T + S['d']).max())
    return r <= tol, r


def _xyz(e):
    return np.array([float(v) for v in (e.get('xyz') or '0 0 0').split()])


def _urdf_gripper(path):
    root = ET.parse(path).getroot()
    J = {j.get('name'): j for j in root.findall('joint')}
    out = {}
    for name in ('joint_tcp', 'finger_joint1', 'finger_joint2'):
        j = J[name]
        o = j.find('origin')
        if o is not None and any(abs(float(v)) > 0 for v in (o.get('rpy') or '0 0 0').split()):
            raise Reject(f'{name}_rpy_nonzero')       # 本契約只支援零 rpy（現況）
        out[name] = {'parent': j.find('parent').get('link'), 'xyz': _xyz(o) if o is not None else np.zeros(3)}
        if name != 'joint_tcp':
            out[name]['axis'] = _xyz(j.find('axis'))
            lim = j.find('limit')
            out[name]['upper'] = float(lim.get('upper'))
            out[name]['lower'] = float(lim.get('lower'))
    if out['finger_joint1']['parent'] != 'uflite_gripper_link' or out['joint_tcp']['parent'] != 'uflite_gripper_link':
        raise Reject('gripper_chain_unexpected')
    return out


def load_contract(path=CONTRACT_DEFAULT, repo=REPO):
    """讀契約並重算雜湊；任一不符 ⇒ ok=False（contract_version_mismatch）。"""
    try:
        try:
            c = yaml.safe_load(open(path))
        except (OSError, yaml.YAMLError):
            raise Reject('contract_unreadable')
        if not isinstance(c, dict) or c.get('schema') != 'gc1_contract/1':
            raise Reject('contract_schema')
        for key, typ in (('contract_id', str), ('handle_frame', str), ('z_split_m', float), ('files', dict)):
            if not isinstance(c.get(key), typ):
                raise Reject(f'contract_incomplete:{key}')
        from finger_collision import Z_SPLIT_M
        if c['z_split_m'] != Z_SPLIT_M:
            raise Reject('contract_z_split_not_locked_model')
        if set(c['files']) != set(LOCKED):
            raise Reject('contract_incomplete:files:' + ','.join(sorted(set(LOCKED) ^ set(c['files']))))
        bad = []
        for k, ent in c['files'].items():
            if (not isinstance(ent, dict) or ent.get('path') != LOCKED[k] or not isinstance(ent.get('sha256'), str)
                    or len(ent['sha256']) != 64):
                raise Reject(f'contract_incomplete:files:{k}')
            p = os.path.join(repo, ent['path'])
            if not os.path.isfile(p) or _sha(p) != ent['sha256']:
                bad.append(k)
        if bad:
            raise Reject('contract_version_mismatch:' + ','.join(sorted(bad)))
        F = {k: os.path.join(repo, ent['path']) for k, ent in c['files'].items()}
        spec = yaml.safe_load(open(F['asset']))
        grip = _urdf_gripper(F['urdf'])
        fingers = {}
        for i, key in ((1, 'finger1_stl'), (2, 'finger2_stl')):
            parts = split_hulls(_stl_vertices(F[key]), float(c['z_split_m']))
            fingers[i] = {'base': parts['base'][0], 'blade': parts['blade'][0]}
        finger_sh = {i: {p: _shape(fingers[i][p]) for p in ('base', 'blade')} for i in (1, 2)}
        # gap_guard：split 兩凸塊（根部頂 z0、指片底 z1）沒有完整包含網格——分界帶 [z0, z1] 的跨界三角形不屬任一塊，
        # 且長三角形在 z0／z1 截面上的部分落在兩凸塊外（手指外側倒角處）。把所有三角形裁成三段
        # （z ≤ z0、[z0, z1]、z ≥ z1；原頂點＋邊界交點），各段取凸包 ⇒ 三段聯集包含整個網格。
        # 離線保守補查，不是模擬所用碰撞形狀；載入時核對每段完整包含其裁切幾何。
        band = {}
        for i, key in ((1, 'finger1_stl'), (2, 'finger2_stl')):
            z0 = float(fingers[i]['base'][:, 2].max())
            z1 = float(fingers[i]['blade'][:, 2].min())
            reg = clipped_regions(_stl_triangles(F[key]), z0, z1)
            G = {k: _shape(reg[k]) for k in ('below', 'band', 'above')}
            for k in G:
                ok_k, _ = contained(G[k], reg[k])
                if not ok_k:
                    raise Reject(f'gap_guard_not_covering:finger{i}:{k}')
            band[i] = {'shapes': G, 'z_mm': [round(z0 * 1e3, 3), round(z1 * 1e3, 3)],
                       'split_excess_m': {'below_vs_base': contained(finger_sh[i]['base'], reg['below'])[1],
                                          'above_vs_blade': contained(finger_sh[i]['blade'], reg['above'])[1]},
                       'n_points': {k: int(len(v)) for k, v in reg.items()}}
        bar = spec['drawer']['handle']['bar']
        if bar.get('shape') != 'cylinder' or bar.get('axis') != 'x':
            raise Reject('bar_shape_unsupported')
        gs = spec['grasp_surface']
        q_open = float(gs['finger_joint_open'])
        if abs(q_open - grip['finger_joint1']['upper']) > 1e-12:
            raise Reject('finger_open_not_urdf_upper')
        return {'ok': True, 'contract': c, 'spec': spec, 'grip': grip, 'fingers': fingers, 'finger_shapes': finger_sh, 'band': band,
                'q_open': q_open, 'margin_min': float(gs['enclosure_margin_min_m']),
                'bar_r': float(bar['radius']), 'bar_L': float(bar['length']),
                'bar_c': np.array(bar['center'], float),
                'u_O': np.array(spec['drawer']['joint']['axis'], float),
                'q_lim': (float(spec['drawer']['joint']['lower']), float(spec['drawer']['joint']['upper']))}
    except Reject as r:
        return {'ok': False, 'why': r.why}


# ------------------------------------------------------------------ 形狀（頂點＋半空間 n·x + d ≤ 0，n 單位向量）
def _shape(V):
    from scipy.spatial import ConvexHull
    h = ConvexHull(V)
    return {'V': np.asarray(V, float)[h.vertices], 'N': h.equations[:, :3].copy(), 'd': h.equations[:, 3].copy()}


def xf_shape(T, S):
    """剛體變換凸形狀：頂點 R v + t；半空間 n' = R n、d' = d − n'·t。"""
    R, t = T[:3, :3], T[:3, 3]
    N = S['N'] @ R.T
    return {'V': S['V'] @ R.T + t, 'N': N, 'd': S['d'] - N @ t}


def _box_vertices(c, s):
    h = np.asarray(s, float) / 2.0
    return np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def _prism_x(c, r, L, n=N_PRISM, outer=False):
    """沿 x 的有限圓柱的內接（outer=False）或外切（outer=True）n 邊稜柱頂點。"""
    R = r / math.cos(math.pi / n) if outer else r
    a = 2 * math.pi * (np.arange(n) + 0.5) / n
    V = []
    for x in (-L / 2, L / 2):
        V.append(np.column_stack([np.full(n, c[0] + x), c[1] + np.cos(a) * R, c[2] + np.sin(a) * R]))
    return np.vstack(V)


def _local_object_shapes(C):
    """資產本地（O 座標、開度 0）的凸形狀；櫃體與抽屜分組。只算一次。"""
    if '_local' in C:
        return C['_local']
    spec = C['spec']
    cab = {f'cabinet/{n}': _shape(_box_vertices(c, s)) for n, c, s in spec['cabinet']}
    drw = {f'drawer/{n}': _shape(_box_vertices(c, s)) for n, c, s in spec['drawer']['body']}
    drw.update({f'drawer/{n}': _shape(_box_vertices(c, s)) for n, c, s in spec['drawer']['handle']['posts']})
    bar = {'inner': _shape(_prism_x(C['bar_c'], C['bar_r'], C['bar_L'])),
           'outer': _shape(_prism_x(C['bar_c'], C['bar_r'], C['bar_L'], outer=True))}
    C['_local'] = (cab, drw, bar)
    return C['_local']


def object_shapes(C, T_WO, q):
    """回傳 {name: ('poly', S) 或 ('cyl', {'inner': S, 'outer': S})}；櫃體 T_WO·T_OS，抽屜 T_WO·Trans(q·u_O)·T_OS。"""
    cab, drw, bar = _local_object_shapes(C)
    T_D = T_WO @ OT.trans(q * C['u_O'])
    out = {k: ('poly', xf_shape(T_WO, S)) for k, S in cab.items()}
    out.update({k: ('poly', xf_shape(T_D, S)) for k, S in drw.items()})
    out['drawer/handle_bar'] = ('cyl', {k: xf_shape(T_D, S) for k, S in bar.items()})
    return out


def finger_link_T(C, T_WE, q_f):
    """回傳 {1: T_W,f1, 2: T_W,f2}；T_EG = I 時 E＝link_tcp。"""
    g = C['grip']
    T_W_gl = T_WE @ np.linalg.inv(OT.trans(g['joint_tcp']['xyz']))
    return {i: T_W_gl @ OT.trans(g[f'finger_joint{i}']['xyz'] + q_f * g[f'finger_joint{i}']['axis']) for i in (1, 2)}


def finger_shapes(C, T_WE, q_f):
    Tf = finger_link_T(C, T_WE, q_f)
    return {f'finger{i}/{p}': xf_shape(Tf[i], C['finger_shapes'][i][p]) for i in (1, 2) for p in ('base', 'blade')}


# ------------------------------------------------------------------ 凸幾何核對
def chebyshev_overlap(A, B):
    """交集內切半徑 t*（> 0 表示有體積交疊；< 0 表示不相交）。LP 失敗回 None。"""
    from scipy.optimize import linprog
    N = np.vstack([A['N'], B['N']])
    d = np.concatenate([A['d'], B['d']])
    A_ub = np.hstack([N, np.ones((len(N), 1))])
    res = linprog(c=[0, 0, 0, -1.0], A_ub=A_ub, b_ub=-d, bounds=[(None, None)] * 3 + [(None, 1.0)], method='highs')
    if res.status != 0 or res.fun is None or not np.isfinite(res.fun):
        return None
    return float(-res.fun)


RESID_TOL = 1e-9


def certified_distance(A, B):
    """凸 QP：min ‖x − y‖²，x ∈ A、y ∈ B（半空間約束，6 變數，SLSQP）。不憑 success：

    * 約束殘差 max(N x + d) ≤ 1e-9（兩邊）——否則回 None；上界 hi = ‖x − y‖（近可行點）。
    * 下界 lo = min_B n·b − max_A n·a，n = (y − x)/‖y − x‖（任何 n 的分離寬度都是距離下界）。
    回傳 (lo, hi, resid) 或 None。"""
    from scipy.optimize import minimize
    x0 = np.concatenate([A['V'].mean(0), B['V'].mean(0)])

    def f(z):
        r = z[:3] - z[3:]
        return float(r @ r)

    def g(z):
        r = z[:3] - z[3:]
        return np.concatenate([2 * r, -2 * r])
    cons = [{'type': 'ineq', 'fun': lambda z: -(A['N'] @ z[:3] + A['d']), 'jac': lambda z: np.hstack([-A['N'], np.zeros((len(A['N']), 3))])},
            {'type': 'ineq', 'fun': lambda z: -(B['N'] @ z[3:] + B['d']), 'jac': lambda z: np.hstack([np.zeros((len(B['N']), 3)), -B['N']])}]
    res = minimize(f, x0, jac=g, method='SLSQP', constraints=cons, options={'ftol': 1e-18, 'maxiter': 1000})
    z = res.x
    if not np.isfinite(z).all():
        return None
    x, y = z[:3], z[3:]
    resid = float(max((A['N'] @ x + A['d']).max(), (B['N'] @ y + B['d']).max()))
    if resid > RESID_TOL:
        return None
    hi = float(np.linalg.norm(y - x))
    if hi < 1e-12:
        return 0.0, 0.0, resid
    n = (y - x) / hi
    lo = float((B['V'] @ n).min() - (A['V'] @ n).max())
    return max(lo, 0.0), hi, resid


def _aabb_gap(VA, VB):
    g = np.maximum(0.0, np.maximum(VA.min(0) - VB.max(0), VB.min(0) - VA.max(0)))
    return float(np.linalg.norm(g))


def classify_poly(A, B):
    """凸多面體配對：status、距離區間 [d_lo, d_hi]、Chebyshev t*。"""
    gap = _aabb_gap(A['V'], B['V'])
    if gap > AABB_SKIP_M:
        return {'status': 'separated', 'd_lo': gap, 'd_hi': None, 'note': 'aabb_lower_bound'}
    t = chebyshev_overlap(A, B)
    if t is None:
        return {'status': 'indeterminate', 'why': 'lp_failed'}
    if t > TOL:
        return {'status': 'penetrating', 't': t, 'd_lo': 0.0, 'd_hi': 0.0}
    dd = certified_distance(A, B)
    if dd is None:
        return {'status': 'indeterminate', 'why': 'qp_infeasible_or_nonfinite', 't': t}
    lo, hi, resid = dd
    r = {'d_lo': lo, 'd_hi': hi, 't': t, 'resid': resid}
    if hi - lo > CERT_TOL:
        return {'status': 'indeterminate', 'why': 'certificate_gap', **r}
    if lo > TOL:
        return {'status': 'separated', **r}
    if hi <= TOL:
        return {'status': 'contact', **r}
    return {'status': 'indeterminate', 'why': 'between_tolerances', **r}


def classify_cyl(A, cyl):
    """凸多面體對有限圓柱：外切稜柱給距離下界、內接稜柱給交疊判定與距離上界。"""
    o = classify_poly(A, cyl['outer'])
    if o['status'] == 'indeterminate':
        return o
    if o['status'] == 'separated':
        r = dict(o)
        if o.get('note') != 'aabb_lower_bound':
            i = classify_poly(A, cyl['inner'])
            if i['status'] == 'indeterminate':
                return i
            r['d_hi'] = i.get('d_hi')
        r['note'] = 'outer_prism_lower_bound' + ('+aabb' if o.get('note') == 'aabb_lower_bound' else '')
        return r
    i = classify_poly(A, cyl['inner'])
    if i['status'] in ('penetrating', 'indeterminate'):
        return i
    if i.get('d_hi') is not None and i['d_hi'] <= TOL:
        return {'status': 'contact', 'd_lo': 0.0, 'd_hi': i['d_hi'], 't': i.get('t')}
    return {'status': 'indeterminate', 'why': 'cylinder_bracket_open', 'd_lo': 0.0, 'd_hi': i.get('d_hi')}


def classify(A, shape):
    kind, S = shape
    return classify_poly(A, S) if kind == 'poly' else classify_cyl(A, S)


# ------------------------------------------------------------------ 抓取相容（只在 s = 0）
def _bar_in_finger_frame(C, T_WH, T_f):
    """回傳橫桿軸在手指連桿座標的 (點, 方向)。"""
    Tinv = np.linalg.inv(T_f)
    p = Tinv[:3, :3] @ T_WH[:3, 3] + Tinv[:3, 3]
    d = Tinv[:3, :3] @ T_WH[:3, 0]
    return p, d / np.linalg.norm(d)


def contact_q(C, T_WE, T_WH, T_D, i):
    """手指 i 由全開閉合到首次碰到橫桿（內接／外切稜柱各求一次）的手指位置。

    回傳 (q_lo, q_hi)：真圓柱的接觸位置落在兩者之間；全開已交疊 ⇒ 'open_overlap'；閉到 0 仍未碰 ⇒ 'never'。"""
    res = []
    _, _, bar = _local_object_shapes(C)
    for which in ('outer', 'inner'):
        S_bar = xf_shape(T_D, bar[which])

        def touching(qf):
            S = xf_shape(finger_link_T(C, T_WE, qf)[i], C['finger_shapes'][i]['blade'])
            if _aabb_gap(S['V'], S_bar['V']) > 0:
                return False
            t = chebyshev_overlap(S, S_bar)
            if t is None:
                raise Reject('lp_failed_in_contact_search')
            return t >= 0.0
        if touching(C['q_open']):
            return 'open_overlap'
        if not touching(0.0):
            return 'never'
        lo, hi = 0.0, C['q_open']               # touching(lo) 真、touching(hi) 假
        for _ in range(BISECT_ITERS):
            m = 0.5 * (lo + hi)
            if touching(m):
                lo = m
            else:
                hi = m
        res.append(0.5 * (lo + hi))
    return min(res), max(res)                     # 外切先碰（q 較大）


def grasp_compat(C, T_WE0, T_WH, T_D):
    """回傳 (報告, 不相容原因清單)；不在第一個原因就停，全部列出。"""
    rep, bad = {}, []
    Tf = finger_link_T(C, T_WE0, C['q_open'])
    # 深度：桿軸上最接近夾爪 z 軸的點，在手指座標的 z；須落在指片 z 範圍，且桿最近點高於根部塊
    zs = []
    for i in (1, 2):
        p, d = _bar_in_finger_frame(C, T_WH, Tf[i])
        # 最近點：桿軸與手指連桿 z 軸（x = y = 0）的公垂線足點
        w = np.array([0.0, 0.0, 1.0])
        A = np.array([[d @ d, -d @ w], [d @ w, -w @ w]])
        if abs(np.linalg.det(A)) < 1e-12:
            raise Reject('bar_axis_parallel_to_approach')
        b = np.array([-(p @ d), -(p @ w)])
        tt = np.linalg.solve(A, b)
        zs.append(float((p + tt[0] * d)[2]))
    z_c = float(np.mean(zs))
    bl = C['fingers'][1]['blade'][:, 2]
    base_top = float(max(C['fingers'][i]['base'][:, 2].max() for i in (1, 2)))
    rep['bar_center_z_finger_mm'] = round(z_c * 1e3, 4)
    rep['blade_z_range_mm'] = [round(float(bl.min()) * 1e3, 3), round(float(bl.max()) * 1e3, 3)]
    rep['root_gap_along_approach_mm'] = round((z_c - C['bar_r'] - base_top) * 1e3, 4)
    tilt = math.degrees(math.acos(min(1.0, abs(float(T_WH[:3, 0] @ (T_WE0[:3, :3] @ [1.0, 0, 0]))))))
    rep['bar_tilt_vs_gripper_x_deg'] = tilt
    if not (bl.min() <= z_c <= bl.max()):
        bad.append('depth_incompatible:center_outside_blade')
    if z_c - C['bar_r'] <= base_top:
        bad.append('depth_incompatible:reaches_root_block')
    # 開口：逐手指求首次接觸的手指位置；全開每側餘裕 = q_open − q_c
    for i in (1, 2):
        qc = contact_q(C, T_WE0, T_WH, T_D, i)
        if qc in ('open_overlap', 'never'):
            bad.append(f'opening_incompatible:finger{i}:{qc}')
            continue
        rep[f'finger{i}_contact_q_mm'] = [round(qc[0] * 1e3, 4), round(qc[1] * 1e3, 4)]
        margin = C['q_open'] - qc[1]
        rep[f'finger{i}_open_margin_mm'] = round(margin * 1e3, 4)
        if margin < C['margin_min']:
            bad.append(f'opening_incompatible:finger{i}:margin')
    return rep, bad


# ------------------------------------------------------------------ 主核對
def collisions(C, T_WE, T_WO, q, q_f):
    F = finger_shapes(C, T_WE, q_f)
    O = object_shapes(C, T_WO, q)
    pairs = {}
    for fn, VF in F.items():
        for on, sh in O.items():
            pairs[f'{fn}|{on}'] = classify(VF, sh)
    return pairs


def check(C, T_WO, q, T_HG, a_H, s, T_EG=None, frame='asset_prim', T_WH_external=None):
    """回傳 dict：ok、why、targets、grasp_compat、pairs、geometry_contract。"""
    out = {'contract_id': None, 'geometry_contract': 'unchecked'}
    try:
        if not isinstance(C, dict) or C.get('ok') is not True:
            raise Reject('contract_not_loaded')
        for key in LOADED_KEYS:
            if key not in C:
                raise Reject(f'contract_incomplete:{key}')
        if not isinstance(C['contract'], dict) or not isinstance(C['contract'].get('contract_id'), str):
            raise Reject('contract_incomplete:contract')
        out['contract_id'] = C['contract']['contract_id']
        if not isinstance(frame, str) or frame != C['contract']['handle_frame']:
            raise Reject('handle_frame_mismatch')
        if T_EG is not None:
            T_EG = OT.rigid(T_EG, 'T_EG', OT.T_MAX_M)
            if float(np.max(np.abs(T_EG - np.eye(4)))) > 1e-12:
                raise Reject('T_EG_not_identity_unsupported')   # 手指鏈以 E = link_tcp 建立
        T_WO = OT.rigid(T_WO, 'T_WO')
        q = OT.scalar(q, 'q', *C['q_lim'])
        T_D = T_WO @ OT.trans(q * C['u_O'])
        T_WH = T_D @ OT.trans(C['bar_c'])
        if T_WH_external is not None:
            Te = OT.rigid(T_WH_external, 'T_WH_external')
            if (OT.geodesic(Te[:3, :3], T_WH[:3, :3]) > 1e-9 or float(np.linalg.norm(Te[:3, 3] - T_WH[:3, 3])) > 1e-9):
                raise Reject('offline_input_inconsistent')
        r0 = OT.object_target(T_WH, T_HG, a_H, 0.0)
        rs = OT.object_target(T_WH, T_HG, a_H, s)
        if not r0['ok']:
            raise Reject('target:' + r0['why'])
        if not rs['ok']:
            raise Reject('target:' + rs['why'])
        out['targets'] = {'s0': r0['T_WE'], 's': rs['T_WE'], 's_m': float(s)}
        out['grasp_compat'], bad = grasp_compat(C, r0['T_WE'], T_WH, T_D)
        out['pairs'] = {}          # split_model：模擬所用的兩凸塊
        out['gap_guard'] = {}      # 分界帶保守凸包（離線補查；穿透或無法判定同樣拒絕）
        for tag, T in (('s0', r0['T_WE']), ('s', rs['T_WE'])):
            if tag == 's' and float(s) == 0.0:
                continue
            pp = collisions(C, T, T_WO, q, C['q_open'])
            out['pairs'][tag] = pp
            Tf = finger_link_T(C, T, C['q_open'])
            O = object_shapes(C, T_WO, q)
            gg = {f'finger{i}/guard_{k}|{on}': classify(xf_shape(Tf[i], G), sh)
                  for i in (1, 2) for k, G in C['band'][i]['shapes'].items() for on, sh in O.items()}
            out['gap_guard'][tag] = gg
            for src, d in (('', pp), ('gap_guard:', gg)):
                for k, v in d.items():
                    if v['status'] in ('penetrating', 'indeterminate'):
                        bad.append(f"{'penetration' if v['status'] == 'penetrating' else 'indeterminate'}_at_target:{src}{tag}:{k}")
        if bad:
            raise Reject(';'.join(bad))
        out['ok'] = True
        out['geometry_contract'] = 'checked_bar26_split_static'
        out['claim'] = 'bar26＋split 手指在指定開爪目標下，通過局部靜態幾何契約核對；不代表整機碰撞安全、軌跡可達或物理夾持成功。'
        return out
    except Reject as r:
        out['ok'] = False
        out['why'] = r.why
        return out
    except OT.Reject as r:
        out['ok'] = False
        out['why'] = r.why
        return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--write-contract', action='store_true')
    a = ap.parse_args()
    if a.write_contract:
        write_contract()
        print('寫入', CONTRACT_DEFAULT)
