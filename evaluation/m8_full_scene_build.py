#!/usr/bin/env python3
"""第八次完整任務場景組建（Howard 11/11 17:39「希望實驗場景整個都先做好，要有完整牆壁」；reviews/20261111_howard_complete_mission_scene.md）。
不跑模擬；只產生／登錄同一份場景定義，供 Isaac 執行端、靜態地圖、掃描、安全核對共用。產物：
  1 粗馬克筆代理資產 src/my_omnibot_description/config/marker_proxy_m8.yaml：圓柱直徑 26 mm、長 140 mm、質量 20 g，平躺於抽屜底板前段（抽屜座標）。
    直徑依 drawer_unit_bar26 已登錄的夾爪指片範圍（閉 22.4、開 40.2 mm）：接觸於 finger_joint 1.8 mm、全開兩側各 7.1 mm 對準餘裕（同 26 mm 把手桿的推導）。
    代理物，不是一般細筆；未經實物驗證；夾持可行性不在本包宣稱。
  2 低速行人障礙檔 results/mission/mission_obstacles_m8_seed1_slow.json：未知箱體、行人路線、停留、相位全部沿用 seed1 r2（不重抽），
    只把行人速度明示改為 0.20 m/s（Howard 11/10「行人走太快了」的低速開發方向；舊 0.4–0.8 m/s 不無聲恢復）。
  3 靜態地圖 src/ammr_bringup/maps/mission_room_m8_fullwalls.pgm／.yaml：由 mission_room_m8_fullwalls.sdf 以既有產生器的同一規則重產
    （只含已知物：SDF 牆與家具＋門資產牆＋抽屜櫃外框；不含未知箱體、行人、門扇），並與現用 mission_room.pgm 逐格比對。
  4 場景登錄 experiments/20261111_m8_full_scene.json：世界、門、抽屜、代理物、障礙、起點、地圖、相機的檔案與雜湊。
    python3 evaluation/m8_full_scene_build.py
"""
import hashlib
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_bringup', 'scripts'))
import generate_mission_map as GM                       # noqa: E402

WORLD = os.path.join(WS, 'src/ammr_bringup/worlds/mission_room_m8_fullwalls.sdf')
DOOR = os.path.join(WS, 'src/my_omnibot_description/config/door_local_v3.yaml')
DRAWER = os.path.join(WS, 'src/my_omnibot_description/config/drawer_unit_bar26.yaml')
MARKER = os.path.join(WS, 'src/my_omnibot_description/config/marker_proxy_m8.yaml')
OBS_SRC = os.path.join(HERE, 'results/mission/mission_obstacles_seed1.json')
OBS_OUT = os.path.join(HERE, 'results/mission/mission_obstacles_m8_seed1_slow.json')
POSES = os.path.join(HERE, 'results/mission/mission_room_poses.json')
MAP_OLD = os.path.join(WS, 'src/ammr_bringup/maps/mission_room')
MAP_NEW = os.path.join(WS, 'src/ammr_bringup/maps/mission_room_m8_fullwalls')
REG = os.path.join(HERE, 'experiments/20261111_m8_full_scene.json')
DRAWER_POSE = (2.4, 5.05, 0.0)
WALKER_SPEED = 0.20


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def write_marker():
    D = yaml.safe_load(open(DRAWER))
    bot = [b for b in D['drawer']['body'] if b[0] == 'bottom'][0]
    fr = [b for b in D['drawer']['body'] if b[0] == 'front_panel'][0]
    z_top = bot[1][2] + bot[2][2] / 2
    y_front_in = fr[1][1] + fr[2][1] / 2
    r, L = 0.013, 0.140
    doc = dict(schema='object_proxy/1', name='marker_proxy_m8',
               note='粗馬克筆代理（Howard 11/10 定案：接受粗馬克筆代理，不換指形）。不是一般細筆；未經實物驗證；本檔只登錄尺寸、質量、初始配置，不宣稱夾持可行。',
               shape='cylinder', axis='x', radius=r, length=L, mass_kg=0.020,
               material=dict(static_friction=0.6, dynamic_friction=0.5, restitution=0.0, note='同瓶代理的材料量級；未辨識'),
               gripper_basis=dict(source='drawer_unit_bar26.yaml 檔頭（指片內面閉 22.4 mm、開 40.2 mm，官方網格）', contact_finger_joint_m=round((2 * r * 1000 - 22.4) / 2 / 1000, 4),
                                  open_margin_each_side_m=round((40.2 - 2 * r * 1000) / 2 / 1000, 4)),
               placement=dict(frame='drawer（抽屜剛體本地座標；隨抽屜滑動）', center=[0.05, round(y_front_in + 0.015 + r + 0.050, 4), round(z_top + r + 0.001, 4)],
                              yaw_deg=0.0, rule='平躺沿抽屜 x 軸，置於底板前段：抽屜拉出全行程 0.22 m 時整支在櫃口外（從上方可見可取）；離前板內面 0.065 m、離側板 ≥ 0.12 m'),
               drawer_interior=dict(bottom_top_z=round(z_top, 4), front_inner_y=round(y_front_in, 4)))
    yaml.safe_dump(doc, open(MARKER, 'w'), allow_unicode=True, sort_keys=False)
    return doc


def write_obstacles():
    d = json.load(open(OBS_SRC))
    d2 = dict(d)
    d2['derived_from'] = os.path.relpath(OBS_SRC, HERE)
    d2['map'] = os.path.relpath(WORLD, WS)
    d2['m8_change'] = f'行人速度明示改為 {WALKER_SPEED} m/s（低速開發方向）；箱體、路線、停留、相位沿用 seed1 r2，不重抽、不依機器人失敗重新對時'
    d2['walkers'] = [dict(w, speed_mps=WALKER_SPEED, speed_mps_seed1=w['speed_mps']) for w in d['walkers']]
    json.dump(d2, open(OBS_OUT, 'w'), ensure_ascii=False, indent=1)
    return d2


def write_map():
    GM.SDF, GM.OUT = WORLD, MAP_NEW
    if os.path.exists(MAP_NEW + '.pgm'):
        os.remove(MAP_NEW + '.pgm')
    if os.path.exists(MAP_NEW + '.yaml'):
        os.remove(MAP_NEW + '.yaml')
    GM.main()
    y = yaml.safe_load(open(MAP_NEW + '.yaml'))
    y['image'] = os.path.basename(MAP_NEW) + '.pgm'
    yaml.safe_dump(y, open(MAP_NEW + '.yaml', 'w'), sort_keys=False)

    def load(p):
        b = open(p + '.pgm', 'rb').read()
        parts = b.split(b'\n', 3)
        w, h = map(int, parts[1].split())
        return np.frombuffer(parts[3], np.uint8).reshape(h, w)
    a, b = load(MAP_OLD), load(MAP_NEW)
    return dict(shape=list(b.shape), identical_to_mission_room=bool(a.shape == b.shape and (a == b).all()), n_diff=int((a != b).sum()) if a.shape == b.shape else None,
                occupied=int((b < 128).sum()))


def main():
    mk = write_marker()
    ob = write_obstacles()
    mp = write_map()
    poses = json.load(open(POSES))['poses']
    reg = dict(name='m8_full_scene_r1', created='2026-11-11（Howard reviews/20261111_howard_complete_mission_scene.md）',
               frame='門局部座標（door_local_v3）：鉸鏈在原點、隔牆 y∈[0, 0.12]、A 房 y<0、B 房 y>0.12；外牆內面 x∈[−4.0, 5.0]、y∈[−5.45, 5.45]',
               world=dict(path=os.path.relpath(WORLD, WS), sha256=sha(WORLD), note='外牆 2.4 m（同隔牆、門資產牆）；平面足跡同 mission_room.sdf'),
               door=dict(asset=os.path.relpath(DOOR, WS), sha256=sha(DOOR), builder='evaluation/door_asset_v3.py', note='可動門扇＋把手（物理機構）；地圖門洞留空'),
               drawer=dict(asset=os.path.relpath(DRAWER, WS), sha256=sha(DRAWER), pose=list(DRAWER_POSE), builder='evaluation/drawer_asset.py', note='可動抽屜＋把手（滑動關節，無驅動）'),
               marker=dict(asset=os.path.relpath(MARKER, WS), sha256=sha(MARKER), in_drawer=True, radius=mk['radius'], length=mk['length'], mass_kg=mk['mass_kg'], center_drawer_frame=mk['placement']['center']),
               obstacles=dict(path=os.path.relpath(OBS_OUT, WS), sha256=sha(OBS_OUT), n_static_unknown=len(ob['static_unknown']), n_walkers=len(ob['walkers']), walker_speed_mps=WALKER_SPEED,
                              note='未知箱體不進靜態地圖；真值只給場景與評估'),
               start=dict(pose=poses['S'], note='A 房起點 S'),
               static_map=dict(path=os.path.relpath(MAP_NEW, WS) + '.yaml', sha256=sha(MAP_NEW + '.pgm'), **mp),
               not_included='天花板（不加）；完整任務實跑、夾持可行性、在線操作能力（場景就緒不代表任務完成）')
    json.dump(reg, open(REG, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(dict(marker=dict(radius=mk['radius'], length=mk['length'], center=mk['placement']['center'], basis=mk['gripper_basis']),
                          walkers=[(w['name'], w['speed_mps'], w['speed_mps_seed1']) for w in ob['walkers']], map=mp), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
