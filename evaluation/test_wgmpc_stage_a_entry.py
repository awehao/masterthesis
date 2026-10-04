"""階段 A 啟動核對：每一條中止條件都要有反例。"""
import os, sys
import numpy as np
sys.path.insert(0, 'evaluation')
sys.path.insert(0, 'src/ammr_wholebody_mpc')
import drawer_asset as DA
import wgmpc_stage_a_entry as E
from wgmpc_stage_a_entry import EntryError

OK = []
def chk(c, m):
    OK.append(bool(c)); print(f'{"ok  " if c else "FAIL"}  {m}')

def fails(fn, why, want=''):
    try:
        fn(); chk(False, f'{why} 應中止但沒有')
    except EntryError as e:
        hit = (want in str(e)) if want else True
        chk(hit, f'{why} 正確中止：{str(e)[:46]}…')

SPEC = DA.load('src/my_omnibot_description/config/drawer_unit.yaml')
POSE = (0.0, 1.45)
OBS, AP = E.expected_drawer_params(SPEC, POSE, 0.0)
DP = {'scene_truth': True, 'geometry': 'links',
      'obstacles': OBS, 'scene_truth_approved': AP}

# ---------- A. 目標：由當步位姿算出，六維，記來源與時間 ----------
p, R, meta = E.compute_pre_contact_pause(SPEC, POSE, 0.0, 0.120,
                                         stamp_sim_t=12.34, source='測試')
chk(np.allclose(p, [0.0, 1.0597, 0.55], atol=1e-4),
    f'A1 退讓 0.120 的 TCP = {np.round(p, 4).tolist()}')
chk(R.shape == (3, 3) and abs(np.linalg.det(R) - 1.0) < 1e-12,
    f'A2 姿態是合法旋轉矩陣（det = {np.linalg.det(R):.12f}）')
chk(meta['stamp_sim_t'] == 12.34 and meta['target_source'] == '測試'
    and 'wall_utc' in meta,
    'A3 記下目標來源、模擬時刻與牆鐘時間')
chk(meta['drawer_opening_m'] == 0.0 and meta['approach_backoff_m'] == 0.120,
    'A4 記下算目標所用的抽屜開度與退讓')
chk(meta['asset_schema'] == 'drawer_unit/1',
    f'A5 記下資產 schema（{meta["asset_schema"]}）⇒ 事後能確認用的是哪版幾何')

# 當步位姿變了，目標要跟著變（把手隨抽屜移動）
p2, _, _ = E.compute_pre_contact_pause(SPEC, POSE, 0.05, 0.120)
chk(abs(float(p2[1] - p[1]) + 0.05) < 1e-12,
    f'A6 開度 +0.05 ⇒ 目標沿 −y 跟著移 0.05（實得 {float(p2[1]-p[1]):+.4f}）')
fails(lambda: E.compute_pre_contact_pause(SPEC, POSE, 0.0, -0.1), 'A7 退讓為負')
fails(lambda: E.compute_pre_contact_pause(SPEC, POSE, 0.0, float('nan')),
      'A8 退讓非有限')

E.check_target(p, R, SPEC, POSE, 0.0, 0.120)
chk(True, 'A9 核對實際使用的目標 == 當步位姿推得的目標')
fails(lambda: E.check_target(p + np.array([0, 0.002, 0]), R, SPEC, POSE,
                             0.0, 0.120), 'A10 目標位置差 2 mm', '目標位置')
fails(lambda: E.check_target(p, np.eye(3), SPEC, POSE, 0.0, 0.120),
      'A11 目標姿態不是 R_DES', '目標姿態')

# ---------- B. 距離節點 ----------
n = E.check_dist_params(DP, SPEC, POSE, 0.0)
chk(n == 14, f'B1 具名幾何 14 個且名單與身分一致（得 {n}）')
fails(lambda: E.check_dist_params(dict(DP, scene_truth=False), SPEC, POSE, 0.0),
      'B2 scene_truth 不是真', 'scene_truth')
fails(lambda: E.check_dist_params(dict(DP, scene_truth='False'), SPEC, POSE, 0.0),
      'B3 scene_truth 字串 "False"', 'scene_truth')
fails(lambda: E.check_dist_params(dict(DP, scene_truth='maybe'), SPEC, POSE, 0.0),
      'B4 布林讀回無法判讀 ⇒ 不當成假', '無法判讀')
fails(lambda: E.check_dist_params(dict(DP, geometry='points'), SPEC, POSE, 0.0),
      'B5 geometry 不是 links', 'links')
fails(lambda: E.check_dist_params(dict(DP, obstacles=OBS[:-1]), SPEC, POSE, 0.0),
      'B6 少一個具名幾何（把手橫桿）', 'obstacles')
fails(lambda: E.check_dist_params(
          dict(DP, obstacles=OBS + ['decoy::box:1,1,1:9,9,0.5:']),
          SPEC, POSE, 0.0), 'B7 多一個設定裡不該有的障礙物', 'obstacles')
fails(lambda: E.check_dist_params(dict(DP, scene_truth_approved=AP[:-1]),
                                  SPEC, POSE, 0.0),
      'B8 核准清單少一項', 'scene_truth_approved')
# 幾何位置被改 ⇒ 名單數量一樣但內容不符
bad = list(OBS); bad[0] = bad[0].replace('-0.29,1.45', '-0.29,1.50')
fails(lambda: E.check_dist_params(dict(DP, obstacles=bad), SPEC, POSE, 0.0),
      'B9 某個幾何的位置被改（數量相同、內容不符）', 'obstacles')
# 抽屜被宣告為靜態 ⇒ 身分不符
stat = [x.replace(':drawer_unit:', '::') for x in OBS]
fails(lambda: E.check_dist_params(dict(DP, obstacles=stat), SPEC, POSE, 0.0),
      'B10 抽屜與把手被宣告為靜態（它有被動滑動自由度）')

# ---------- C. 安全層 ----------
chk(E.check_safety_params({'freespace_confirmed': False}),
    'C1 freespace_confirmed=false ⇒ 通過')
fails(lambda: E.check_safety_params({'freespace_confirmed': True}),
      'C2 freespace_confirmed=true', 'freespace_confirmed')
fails(lambda: E.check_safety_params({'freespace_confirmed': 'True'}),
      'C3 freespace_confirmed 字串 "True"', 'freespace_confirmed')
for k, v in (('d0', 0.03), ('eps', 0.01), ('tau', 0.05), ('alpha', 5.0)):
    fails(lambda k=k, v=v: E.check_safety_params(
              {'freespace_confirmed': False, k: v}),
          f'C4 安全層 {k} 被放寬為 {v}', '不是原值')
chk(E.check_safety_params({'freespace_confirmed': False, 'd0': 0.05,
                           'eps': 0.03, 'tau': 0.15, 'alpha': 2.0}),
    'C5 四個保護參數都是原值 ⇒ 通過')

# ---------- D. joint_margin 與模式 ----------
chk(E.check_margin_consistency(0.05, 0.05) == 0.05, 'D1 兩端都是 0.05 ⇒ 通過')
fails(lambda: E.check_margin_consistency(0.05, 0.0),
      'D2 執行端為 0（求解器 0.05）', 'joint_margin')
fails(lambda: E.check_margin_consistency(0.0, 0.05), 'D3 求解器為 0')
fails(lambda: E.check_margin_consistency(0.03, 0.03),
      'D4 兩端一致但不是約定值 0.05')
chk(E.check_mode('solver_drawer') == 'solver_drawer', 'D5 模式正確 ⇒ 通過')
for m in ('sync', 'solver_freespace', 'base', 'arm'):
    fails(lambda m=m: E.check_mode(m), f'D6 模式 {m!r}')

# ---------- E. 與產生器輸出一致（幾何只有一份來源） ----------
import subprocess, re
out = subprocess.run([sys.executable, 'evaluation/gen_drawer_obstacle_params.py',
                      '--ros-args'], capture_output=True, text=True).stdout
gen_obs = [x.strip('"') for x in
           re.search(r'-p obstacles:=\[(.*?)\]', out, re.S).group(1).split('","')]
gen_ap = [x.strip('"') for x in
          re.search(r'-p scene_truth_approved:=\[(.*?)\]', out, re.S)
          .group(1).split('","')]
chk(sorted(gen_obs) == sorted(OBS),
    f'E1 產生器的 obstacles 與啟動核對期望的相同（{len(gen_obs)} 項）')
chk(sorted(gen_ap) == sorted(AP),
    f'E2 產生器的 scene_truth_approved 與期望的相同（{len(gen_ap)} 項）')

print(f'\n{sum(OK)}/{len(OK)} 通過')
sys.exit(0 if all(OK) else 1)
