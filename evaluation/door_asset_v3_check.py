#!/usr/bin/env python3
"""雙面把手門資產 v3 的離線建構＋讀回＋幾何一致性核對（Codex r45 §4；不啟動 Kit、不演化物理）。
在記憶體舞台以 door_asset_v3.build_usd 建門，讀回後核：
  1. 內側橫桿／立柱與 door_geom_v3（離線幾何，handle_pose_inner、shapes 的 post_in*）同位（門角 0 的門扇局部＝門局部座標）；外側橫桿／立柱與 door_geom／door_geom_v2 同位；
  2. 內外把手對門扇中面（v＝thickness/2）鏡像；同型（半徑、長度、立柱尺寸）；
  3. v2 原有讀回值不變（質量、質心、慣量、角／線阻尼、鉸鏈框架與限位、零驅動、牆端面、外側把手材質）——與 door_local_v2 建出的舞台逐欄比對；
  4. 兩支橫桿實際生效的 physics 材質＝規格（μs 1.0、μd 0.8、restitution 0）；所有碰撞形狀 contactOffset／restOffset＝規格；
  5. 各碰撞 prim 路徑互異且可分別辨識（外側橫桿、內側橫桿、四根立柱、門扇、兩面牆）；
  6. 門角 0–110° 每 1°：內側橫桿＋立柱對兩面牆的最小距離（門扇局部→門局部旋轉後以方盒距離計），確認新增幾何不與牆重疊（≥ 2·contactOffset＋sweep_clear_margin）。
另匯出 .usda 供檢視。

    bash evaluation/run_usd_offline.sh evaluation/door_asset_v3_check.py   # 寫 results/door/door_asset_v3_check.json、door_asset_v3_offline.usda（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_asset_v2 as A2                             # noqa: E402
import door_asset_v3 as A3                             # noqa: E402
import door_geom_v3 as G3                              # noqa: E402

CFG = os.path.join(WS, 'src', 'my_omnibot_description', 'config')
OUT = os.path.join(HERE, 'results', 'door', 'door_asset_v3_check.json')
USDA = os.path.join(HERE, 'results', 'door', 'door_asset_v3_offline.usda')
TOL = 1e-6


def close(a, b, tol=TOL):
    return bool(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))) <= tol)


def main():
    for p in (OUT, USDA):
        if os.path.exists(p):
            raise SystemExit(f'{p} 已存在，不覆寫')
    A3.register_physx_plugins()
    from pxr import Usd
    d3 = A3.load(os.path.join(CFG, 'door_local_v3.yaml'))
    d2 = A2.load(os.path.join(CFG, 'door_local_v2.yaml'))
    s3, s2 = Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()
    rb3 = A3.build_usd(s3, d3)
    rb2 = A2.build_usd(s2, d2)
    chk = {}
    t, H = float(d3['leaf']['thickness']), float(d3['handle']['height'])
    # 1. 與離線幾何同位（placement 為原點、yaw 0 ⇒ 門角 0 的門扇局部＝門局部）
    assert d3['placement']['pose'] == {'x': 0.0, 'y': 0.0, 'yaw_deg': 0.0}
    Ti = G3.handle_pose_inner(d3, 0.0)
    chk['inner_bar_center_vs_geom'] = close(rb3['handle_bar_inner']['center_leaf_local'], Ti[:3, 3])
    chk['inner_bar_axis_is_u'] = rb3['handle_bar_inner']['axis'] == 'X' and close(np.abs(Ti[:3, 0]), [1, 0, 0])
    To = G3.handle_pose(d3, 0.0, H)
    chk['outer_bar_center_vs_geom'] = close(rb3['handle_bar']['center_leaf_local'], To[:3, 3])
    sh = {name: (c, hs) for name, c, R, hs in G3.shapes(d3, 0.0, H)}
    chk['inner_posts_vs_geom'] = all(close(rb3['posts'][f'post_in{k}']['center_leaf_local'], sh[f'post_in{k}'][0]) and
                                     close(rb3['posts'][f'post_in{k}']['half_size'], sh[f'post_in{k}'][1]) for k in (0, 1))
    chk['outer_posts_vs_geom'] = all(close(rb3['posts'][f'post{k}']['center_leaf_local'], sh[f'post{k}'][0]) and
                                     close(rb3['posts'][f'post{k}']['half_size'], sh[f'post{k}'][1]) for k in (0, 1) if f'post{k}' in sh)
    chk['outer_posts_names_in_geom'] = [n for n in sh if n.startswith('post')]
    # 2. 鏡像、同型
    mid = t / 2
    co, ci = np.array(rb3['handle_bar']['center_leaf_local']), np.array(rb3['handle_bar_inner']['center_leaf_local'])
    chk['bars_mirror_about_leaf_midplane'] = close([co[0], 2 * mid - co[1], co[2]], ci)
    chk['bars_same_type'] = (rb3['handle_bar']['radius'] == rb3['handle_bar_inner']['radius'] and rb3['handle_bar']['length'] == rb3['handle_bar_inner']['length'])
    chk['posts_mirror'] = all(close([rb3['posts'][f'post{k}']['center_leaf_local'][0], 2 * mid - rb3['posts'][f'post{k}']['center_leaf_local'][1],
                                     rb3['posts'][f'post{k}']['center_leaf_local'][2]], rb3['posts'][f'post_in{k}']['center_leaf_local'])
                              and close(rb3['posts'][f'post{k}']['half_size'], rb3['posts'][f'post_in{k}']['half_size']) for k in (0, 1))
    # 3. v2 原讀回不變
    keys_same = ['mass_kg', 'center_of_mass', 'diagonal_inertia', 'angular_damping', 'linear_damping', 'joint_type', 'joint_axis', 'joint_body0', 'joint_body1',
                 'joint_local_pos0', 'joint_local_rot0_wxyz', 'joint_local_pos1', 'joint_local_rot1_wxyz', 'limit_lower_deg', 'limit_upper_deg',
                 'joint_friction', 'drive_stiffness', 'drive_damping', 'drive_max_force', 'handle_bar', 'handle_material', 'wall_end_faces_x']
    diff = [k for k in keys_same if rb3[k] != rb2[k]]
    chk['v2_fields_unchanged'] = not diff
    chk['v2_fields_changed'] = diff
    new_prims = sorted(set(rb3['collision_prims']) - set(rb2['collision_prims']))
    chk['added_collision_prims'] = new_prims
    chk['only_inner_added'] = new_prims == ['/World/door/leaf/handle_bar_inner', '/World/door/leaf/post_in0', '/World/door/leaf/post_in1']
    chk['v2_collision_prims_kept'] = set(rb2['collision_prims']) <= set(rb3['collision_prims'])
    # 4. 材質與偏移
    m = d3['physics']['handle_material']
    chk['bar_materials_match_spec'] = all(abs(v.get('static_friction', math.nan) - m['static_friction']) < 1e-6           # USD 以 float32 存（0.8 讀回 0.80000001）
                                          and abs(v.get('dynamic_friction', math.nan) - m['dynamic_friction']) < 1e-6
                                          and v.get('restitution') == 0.0 and v.get('bound_at') == p
                                          for p, v in ((f"/World/door/leaf/{b}", rb3['handle_material_by_bar'][b]) for b in ('handle_bar', 'handle_bar_inner')))
    co_, ro_ = float(d3['collision']['contact_offset']), float(d3['collision']['rest_offset'])
    chk['collision_offsets_match_spec'] = all(abs(v[0] - co_) < 1e-9 and abs(v[1] - ro_) < 1e-9 for v in rb3['collision_offsets'].values())
    # 5. 可分別辨識
    names = [p.split('/')[-1] for p in rb3['collision_prims']]
    chk['collision_prim_names_unique'] = len(names) == len(set(names))
    chk['collision_prims'] = rb3['collision_prims']
    # 6. 內側把手對牆掃掠
    walls = [(n, c, R, hs) for n, c, R, hs in G3.shapes(d3, 0.0, H) if n.startswith('wall')]
    need = 2 * co_ + float(d3['collision']['sweep_clear_margin'])
    worst = (math.inf, None, None)
    rng = np.random.default_rng(0)
    for qd in range(0, 111):
        q = math.radians(qd)
        Rl = np.array([[math.cos(q), math.sin(q), 0], [-math.sin(q), math.cos(q), 0], [0, 0, 1]])    # 門扇局部→門局部（繞 z 轉 −q）
        pts = []
        cb = rb3['handle_bar_inner']['center_leaf_local']
        r_, L_ = rb3['handle_bar_inner']['radius'], rb3['handle_bar_inner']['length']
        for s_ in np.linspace(-L_ / 2, L_ / 2, 21):
            for a_ in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                pts.append([cb[0] + s_, cb[1] + r_ * math.cos(a_), cb[2] + r_ * math.sin(a_)])
        for k in (0, 1):
            c_, hs_ = rb3['posts'][f'post_in{k}']['center_leaf_local'], rb3['posts'][f'post_in{k}']['half_size']
            pts += (np.asarray(c_) + (rng.uniform(-1, 1, (200, 3)) * hs_)).tolist()
            pts += [[c_[0] + sx * hs_[0], c_[1] + sy * hs_[1], c_[2] + sz_ * hs_[2]] for sx in (-1, 1) for sy in (-1, 1) for sz_ in (-1, 1)]
        P = np.asarray(pts) @ Rl.T
        for n, c, R, hs in G3.shapes(d3, q, H):
            if not n.startswith('wall'):
                continue
            dmin = float(np.min(G3.box_dist(P, c, R, hs)))
            if dmin < worst[0]:
                worst = (dmin, qd, n)
    chk['inner_handle_vs_walls_min_m'] = {'d': worst[0], 'q_deg': worst[1], 'wall': worst[2], 'need': need, 'ok': worst[0] >= need,
                                          'method': '橫桿表面 21×16 點＋立柱角點與內部取樣，門角 0–110° 每 1°；離散取樣未見越界（取樣最小值，非連續保證）'}
    chk['walls_in_geom'] = [w[0] for w in walls]
    ok_keys = ['inner_bar_center_vs_geom', 'inner_bar_axis_is_u', 'outer_bar_center_vs_geom', 'inner_posts_vs_geom', 'outer_posts_vs_geom',
               'bars_mirror_about_leaf_midplane', 'bars_same_type', 'posts_mirror', 'v2_fields_unchanged', 'only_inner_added', 'v2_collision_prims_kept',
               'bar_materials_match_spec', 'collision_offsets_match_spec', 'collision_prim_names_unique']
    res = {'schema': 'door_asset_v3_check/1', 'spec': 'door_local_v3.yaml', 'asset': 'evaluation/door_asset_v3.py', 'all_ok': all(chk[k] for k in ok_keys)
           and chk['inner_handle_vs_walls_min_m']['ok'], 'checks': chk, 'readback_v3': rb3,
           'not_done': ['物理演化', '接觸行為', '質量／慣量未含把手（同 v2，揭露）']}
    s3.GetRootLayer().Export(USDA)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({'all_ok': res['all_ok'], **{k: chk[k] for k in ok_keys}, 'inner_vs_walls': chk['inner_handle_vs_walls_min_m'],
                      'v2_changed': diff, 'added': new_prims}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
