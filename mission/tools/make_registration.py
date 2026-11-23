#!/usr/bin/env python3
import json, subprocess, os, sys
"""用法：python3 make_reg_next.py N STUB '本趟說明' 'CONFIG 追加' '控制追加；以；分隔'"""
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'evaluation'))
N, stub, status_note, cfg_add = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
ctrl_add = [x for x in (sys.argv[5].split('；') if len(sys.argv) > 5 else []) if x]
r = json.load(open('experiments/20261119_m8_house_d2_unified31_registration.json'))
r['name'] = f'm8_house_d2_unified_r{N}_desktop_house_r2_chain'; r['run_id'] = f'm8house_d2_unified_{N}'
r['status'] = f'Claude 自行核可單趟（Codex 用量用完）：{status_note}；替身 {stub} 全程通過'
r['config'] = r['config'] + '；' + cfg_add
r['control'] = r['control'] + ctrl_add
r['not_claims'] = [x for x in r['not_claims'] if '行人改為會讓路' not in x] + ['行人速度 ×0.75（D2_WALKER_SPEED_SCALE，Howard 11/19「行人可以慢一點點」）；不讓路、不暫停；不開 DETOUR']
r['previous'] = r['previous'] + '；unified_29 作廢、unified_30（行人撞停著的機器人）、unified_31（移動中夾取對齊偏離路線）失敗保留'
r['machine'] = '桌機 Ryzen 7 9800X3D＋RTX 5070（11/19 晚換機）；耗時不與筆電趟直接比'
json.dump(r, open(f'experiments/20261119_m8_house_d2_unified{N}_registration.json', 'w'), ensure_ascii=False, indent=1)
files = [l.split()[1] for l in open('experiments/20261119_m8_house_d2_unified31_freeze.sha256') if l.strip()]
files = [f for f in files if 'unified31_registration' not in f]
files += [f'evaluation/experiments/20261119_m8_house_d2_unified{N}_registration.json', 'evaluation/drawer_motm.py', 'evaluation/door_close_ctrl.py', 'evaluation/door_leaf_detect.py', 'evaluation/handle_detect.py', 'evaluation/rear_bin.py', 'evaluation/yield_side.py', 'evaluation/drawer_reach.py', 'evaluation/door_pick_ctrl.py', 'evaluation/door_pass_ctrl.py', 'evaluation/door_drawer_ctrl.py', 'evaluation/motm_b/b_d2_node.py', 'evaluation/walkers.py', 'evaluation/door_leaf_scan_mask.py', 'src/my_omnibot_description/launch/omni_bot_dynamic.launch.py', 'install/my_omnibot_description/share/my_omnibot_description/launch/omni_bot_dynamic.launch.py', 'evaluation/motm_dry/world_d2_node.py', 'evaluation/door_ahead_exec.py', 'evaluation/run_d2_isaac.sh', 'evaluation/drawer_scene.py', 'evaluation/isaac_m8_house_d2.py', 'evaluation/motm_merge.py', 'evaluation/cyl_reach_map.py', 'evaluation/experiments/20261121_m8_house_r2w_scene.json', 'evaluation/results/mission/mission_obstacles_m8_house_r2w.json', 'evaluation/experiments/20261121_m8_house_r2w2_scene.json', 'evaluation/results/mission/mission_obstacles_m8_house_r2w2.json', 'evaluation/yield_side.py', 'evaluation/experiments/20261121_m8_house_r2w3_scene.json', 'evaluation/results/mission/mission_obstacles_m8_house_r2w3.json', 'evaluation/door_motm_grasp_ctrl.py']
files = sorted(set(files))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
out = subprocess.run(['sha256sum'] + files, capture_output=True, text=True, check=True).stdout
open(f'evaluation/experiments/20261119_m8_house_d2_unified{N}_freeze.sha256', 'w').write(out)
print(len(files), 'files frozen')
