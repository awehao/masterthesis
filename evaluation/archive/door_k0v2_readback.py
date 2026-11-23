#!/usr/bin/env python3
"""【v2】door_k0_readback.py（K1 r2 凍結，不改）的複本：改用 door_asset_v2／door_local_v2，並加四項：線性阻尼讀回、門所有碰撞形狀的
contactOffset／restOffset、兩側牆端面位置、**開門掃掠最小距離**（0 → 上限每 0.25°，門扇／立柱／橫桿對兩側牆的最小距離 ≥ 2·contactOffset＋登錄餘裕）。
原：K0 讀回核對：在記憶體建 USD 舞台（Isaac venv 的 pxr；**不啟動模擬器、不跑物理**），讀回屬性並核對：
  1. 讀回值＝規格（質量、質心、慣量、角阻尼、限位、關節摩擦、零驅動、把手尺寸與位置）；
  2. 關節語意：用讀回的 localRot0／localRot1 計算「關節角 q」造成的門扇世界旋轉，須等於 door_geom 的 Rz(−q)（q > 0 朝機器人側拉開）；
  3. 把手世界位姿（q = 0）＝door_geom.handle_pose；
  4. 反例：placement yaw = 37° 時上述 2、3 在世界座標仍成立（轉換有套 yaw），且若故意不翻轉關節框架，語意檢查會失敗。

    bash evaluation/run_usd_offline.sh evaluation/door_k0v2_readback.py   # 輸出 results/door/door_k0_readback.json（已存在則拒絕覆寫）；任一失敗 rc 1
"""
import copy
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_asset_v2 as DA                                # noqa: E402
import door_geom as G                                  # noqa: E402

SPEC_P = os.path.join(WS, 'src/my_omnibot_description/config/door_local_v2.yaml')
OUT = os.path.join(HERE, 'results', 'door', 'door_k0v2_readback.json')


def qmat(wxyz):
    w, x, y, z = wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def leaf_world_R(rb, q):
    """關節角 q（rad）時門扇剛體的世界旋轉：R_W1 = R_0 · Rz_joint(q) · R_1ᵀ（body0 為世界）。"""
    return qmat(rb['joint_local_rot0_wxyz']) @ G.Rz(q) @ qmat(rb['joint_local_rot1_wxyz']).T


def _box_surface(c, R, hs, n=24):
    g = np.linspace(-1, 1, n)
    U, V = np.meshgrid(g, g)
    pts = []
    for ax in range(3):
        a_, b_ = [k for k in range(3) if k != ax]
        for sgn in (-1, 1):
            P = np.zeros((U.size, 3))
            P[:, ax], P[:, a_], P[:, b_] = sgn, U.ravel(), V.ravel()
            pts.append(P * hs)
    return np.vstack(pts) @ R.T + c


def sweep_clearance(d, step_deg=0.25):
    """開門 0 → 上限：門扇、立柱、橫桿（以方盒近似）對兩側牆（含鉸鏈側退讓）的最小距離。兩向取樣（門→牆、牆→門）取小。"""
    g = float(d['wall']['hinge_side_recess'])
    W, T, Ht = d['frame_clear_width'], d['wall']['thickness'], d['wall']['height']
    x0, x1 = d['wall']['extent_x']
    walls = {'wall_hinge_side': (np.array([(x0 - g) / 2, T / 2, Ht / 2]), np.eye(3), np.array([(-x0 - g) / 2, T / 2, Ht / 2])),
             'wall_latch_side': (np.array([(W + x1) / 2, T / 2, Ht / 2]), np.eye(3), np.array([(x1 - W) / 2, T / 2, Ht / 2]))}
    best = (np.inf, None, None)
    Hd = d['handle']
    for qd in np.arange(0.0, d['leaf']['limit_q_deg'][1] + 1e-9, step_deg):
        q = np.radians(qd)
        parts = [s for s in G.shapes(d, q, Hd['height']) if s[0] in ('leaf', 'post0', 'post1')]
        Tb = G.handle_pose(d, q, Hd['height'])
        r, L = Hd['bar']['radius'], Hd['bar']['length']
        parts.append(('bar', Tb[:3, 3], Tb[:3, :3], np.array([L / 2, r, r])))
        for pn, c, R, hs in parts:
            Pp = _box_surface(c, R, hs)
            for wn, (cw, Rw, hw) in walls.items():
                dd = min(float(G.box_dist(Pp, cw, Rw, hw).min()), float(G.box_dist(_box_surface(cw, Rw, hw), c, R, hs).min()))
                if dd < best[0]:
                    best = (dd, f'{pn}|{wn}', float(qd))
    return {'min_m': best[0], 'pair': best[1], 'at_q_deg': best[2], 'step_deg': step_deg,
            'note': '方盒表面取樣（每面 24×24）兩向取小；橫桿以外接方盒近似（保守）；牆的遠端面取樣稀疏，只影響遠離門的距離'}


def build(d, flip=True):
    DA.register_physx_plugins()
    from pxr import Usd, UsdGeom
    st = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    rb = DA.build_usd(st, d)
    if not flip:                                       # 反例：不翻轉關節框架
        from pxr import Gf, UsdPhysics
        j = UsdPhysics.RevoluteJoint(st.GetPrimAtPath('/World/door/hinge'))
        yaw = math.radians(d['placement']['pose']['yaw_deg'])
        j.GetLocalRot0Attr().Set(Gf.Quatf(math.cos(yaw / 2), Gf.Vec3f(0, 0, math.sin(yaw / 2))))
        j.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
        rb = DA.readback(st, '/World/door')
    return rb


def checks(d, rb):
    out = {}
    m, com, I = DA.leaf_inertia(d)
    ph, Hd = d['physics'], d['handle']
    near = lambda a, b, tol=1e-5: bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol))   # noqa: E731
    out['rigid_body'] = rb['is_rigid_body']
    out['mass'] = near(rb['mass_kg'], m)
    out['center_of_mass'] = near(rb['center_of_mass'], com)
    out['inertia'] = near(rb['diagonal_inertia'], I, 1e-4)
    out['angular_damping'] = near(rb['angular_damping'], ph['angular_damping'])
    out['joint_friction'] = near(rb['joint_friction'], ph['joint_friction'])
    out['no_drive'] = rb['drive_stiffness'] == 0.0 and rb['drive_damping'] == 0.0 and rb['drive_max_force'] == 0.0
    out['limits'] = near([rb['limit_lower_deg'], rb['limit_upper_deg']], d['leaf']['limit_q_deg'])
    out['body0_world'] = rb['joint_body0'] == [] and rb['joint_body1'] == ['/World/door/leaf']
    out['revolute_z'] = rb['joint_type'] == 'PhysicsRevoluteJoint' and rb['joint_axis'] == 'Z'
    out['handle_bar'] = (rb['handle_bar']['axis'] == 'X' and near(rb['handle_bar']['radius'], Hd['bar']['radius'])
                         and near(rb['handle_bar']['length'], Hd['bar']['length'])
                         and near(rb['handle_bar']['center_leaf_local'], [Hd['from_hinge'], -Hd['standoff'], Hd['height']]))
    p = d['placement']['pose']
    yaw = math.radians(p['yaw_deg'])
    Rr = G.Rz(yaw)
    trw = np.array([p['x'], p['y'], 0.0])
    out['anchor_at_hinge'] = near(rb['joint_local_pos0'], trw) and near(rb['joint_local_pos1'], [0, 0, 0])
    sem = []
    for qd in (0.0, 10.0, 45.0, 90.0):
        q = math.radians(qd)
        R_leaf = leaf_world_R(rb, q)
        sem.append(near(R_leaf, Rr @ G.Rz(-q), 1e-5))
        T_loc = G.handle_pose(d, q, Hd['height'])
        bar_w = trw + R_leaf @ np.asarray(rb['handle_bar']['center_leaf_local'])
        sem.append(near(bar_w, trw + Rr @ T_loc[:3, 3], 1e-5))
    out['joint_semantics_q_positive_pulls'] = all(sem)
    hm = rb.get('handle_material') or {}
    out['handle_material'] = (hm.get('bound_at') == '/World/door/leaf/handle_bar'
                               and near(hm.get('static_friction'), ph['handle_material']['static_friction'])
                               and near(hm.get('dynamic_friction'), ph['handle_material']['dynamic_friction']))
    out['linear_damping'] = near(rb.get('linear_damping'), ph['linear_damping'])
    co = d['collision']
    offs = rb.get('collision_offsets') or {}
    out['collision_offsets'] = (len(offs) == 6 and all(near(v, [co['contact_offset'], co['rest_offset']]) for v in offs.values()))
    ends = rb.get('wall_end_faces_x') or {}
    out['wall_end_faces'] = (near(ends.get('wall_hinge_side'), -d['wall']['hinge_side_recess'])
                             and near(ends.get('wall_latch_side'), d['frame_clear_width']))
    sw = sweep_clearance(d)
    out['sweep_clearance'] = sw['min_m'] >= 2 * co['contact_offset'] + co['sweep_clear_margin']
    out['collision_prims'] = sorted(x.split('/')[-1] for x in rb['collision_prims']) == ['handle_bar', 'leaf_box', 'post0', 'post1',
                                                                                         'wall_hinge_side', 'wall_latch_side']
    return out


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d = DA.load(SPEC_P)
    rb = build(d)
    c = checks(d, rb)
    d37 = copy.deepcopy(d)
    d37['placement']['pose'] = {'x': 1.3, 'y': -0.7, 'yaw_deg': 37.0}
    c37 = checks(d37, build(d37))
    cneg = checks(d, build(d, flip=False))
    res0 = sweep_clearance(d)
    res = {'schema': 'door_k0v2_readback/1', 'sweep': res0, 'spec': os.path.relpath(SPEC_P, WS),
           'spec_sha256': __import__('hashlib').sha256(open(SPEC_P, 'rb').read()).hexdigest(),
           'scope': '只建 USD 舞台並讀回屬性（不啟動模擬器、不跑物理）；物理行為（被動性、阻尼、限位生效）留 K1 短測',
           'readback': rb, 'checks': c, 'checks_yaw37': c37,
           'counterexample_no_flip': {'joint_semantics_q_positive_pulls': cneg['joint_semantics_q_positive_pulls'],
                                      'expected': False}}
    ok = all(c.values()) and all(c37.values()) and cneg['joint_semantics_q_positive_pulls'] is False
    res['pass'] = ok
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({'checks': c, 'yaw37_all': all(c37.values()), 'no_flip_semantics': cneg['joint_semantics_q_positive_pulls'], 'pass': ok},
                     ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
