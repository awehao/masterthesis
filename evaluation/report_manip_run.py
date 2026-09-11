"""手臂短測試的分層報告。

四件事分開講，任何一項都不由另一項推論：

  命令交付   播放端發布幾則 / Isaac 收到幾則 / Isaac 實際套用幾次
  關節追蹤   只在「有收到命令」時才有意義；沒收到就是「未測到」
  TCP 誤差   對照**箱體錨定**的世界目標，不是底盤相對位姿 ——
             底盤停偏了，TCP 相對箱體就會偏，這一項才是操作的判準
  底盤保持   實際位移與朝向漂移

碰撞證據另列範圍，不以正的點雲距離宣稱全程安全。
"""
import argparse, json, math, os
import numpy as np, yaml

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--sent', default='')
ap.add_argument('--cases', default='src/my_omnibot_description/config/manipulation_cases.yaml')
ap.add_argument('--path-check', default='')
a = ap.parse_args()

d = json.load(open(a.run))
C = yaml.safe_load(open(a.cases))['cases'][d['case']]
tol = C['tolerance']
sent = json.load(open(a.sent)) if a.sent and os.path.exists(a.sent) else None

print('=' * 66)
print(f'手臂短測試 {d["case"]}   stop_reason={d["stop_reason"]}  '
      f'sim {d["sim_time"]:.2f} s  取樣 {d["samples"]}')
print('=' * 66)

# ---- 1. 命令交付 -----------------------------------------------------------
dl = d.get('delivery', {})
pub = sent['published_msgs'] if sent else None
print('\n[1] 命令交付')
print(f'    播放端發布      {pub if pub is not None else "（無 traj_sent.json）"}')
print(f'    Isaac 收到      {dl.get("received_msgs")}')
print(f'    Isaac 實際套用  {dl.get("applied_actions")}')
if dl.get('first_recv_sim_t') is not None:
    print(f'    首則收到於 sim  {dl["first_recv_sim_t"]:.3f} s')
delivered = bool(dl.get('received_msgs'))
if not delivered:
    print('    → **命令未送達**。以下「關節追蹤」與「TCP 誤差」皆為'
          '**未測到**，不代表控制器有問題，也不代表沒問題。')
elif pub and dl['received_msgs'] < pub * 0.9:
    print(f'    → 收到 {dl["received_msgs"]}/{pub}，**有遺失**')

# ---- 2. 關節追蹤 -----------------------------------------------------------
print('\n[2] 關節追蹤')
if not delivered:
    print('    未測到（命令未送達）')
else:
    log = d['log']
    errs = [r['track_err'] for r in log if r['cmd'] != r['q']]
    qf = np.array(d['arm_final']); qg = np.array(d['arm_goal'])
    print(f'    最終關節誤差（對軌跡終點）逐關節 '
          f'{[round(v*1000,3) for v in (qf-qg)]} mrad')
    print(f'    最終關節誤差 max {d["arm_final_err_max"]*1000:.3f} mrad')
    if errs:
        e = np.array(errs)
        print(f'    執行中追蹤誤差 p50 {np.median(e)*1000:.2f} / '
              f'p95 {np.percentile(e,95)*1000:.2f} / max {e.max()*1000:.2f} mrad')

# ---- 3. TCP 對箱體錨定的世界目標 -------------------------------------------
print('\n[3] TCP 對箱體錨定的世界目標')
obj = C['object']; pg = C['pregrasp']
fc = obj['face_center']; face = obj['approach_face']
nx, ny = {'+x': (1, 0), '-x': (-1, 0), '+y': (0, 1), '-y': (0, -1)}[face]
stand = pg['standoff_from_face_m']
tgt = np.array([fc[0] + nx * stand, fc[1] + ny * stand, pg['tcp_xyz'][2]])
tcp = np.array(d['tcp_final_world'])
err = tcp - tgt
print(f'    箱面中心 ({fc[0]:.3f}, {fc[1]:.3f})  接近面 {face}  站距 {stand:.3f} m')
print(f'    箱體錨定目標  ({tgt[0]:.4f}, {tgt[1]:.4f}, {tgt[2]:.4f})')
print(f'    實際 TCP      ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f})')
print(f'    誤差          ({err[0]:+.4f}, {err[1]:+.4f}, {err[2]:+.4f})  '
      f'距離 {np.linalg.norm(err):.4f} m   容差 {tol["tcp_pos_m"]:.4f} m')
ok3 = np.linalg.norm(err) <= tol['tcp_pos_m']
print(f'    → {"合格" if ok3 else "**不合格**"}'
      + ('' if delivered else '（命令未送達，此為起始姿態的位置，非執行結果）'))
# 實際離箱面多遠（沿法線），這是操作上真正關心的
along = (tcp[0] - fc[0]) * nx + (tcp[1] - fc[1]) * ny
print(f'    TCP 沿法線離箱面 {along:.4f} m（設定站距 {stand:.3f} m）')

# ---- 4. 底盤保持 -----------------------------------------------------------
print('\n[4] 底盤保持')
bf = d['base_final']; pt = d['parking_target']
print(f'    目標 ({pt["x"]:.3f}, {pt["y"]:.3f}) yaw {pt["yaw_deg"]:.2f}°')
print(f'    實際 ({bf["x"]:.4f}, {bf["y"]:.4f}) yaw {bf["yaw_deg"]:.3f}°')
print(f'    位移 {bf["drift_m"]*1000:.3f} mm   yaw 差 '
      f'{abs(bf["yaw_deg"]-pt["yaw_deg"]):.3f}°')
print(f'    /cmd_vel 非零 {d["base_cmd_nonzero"]} 則（應為 0，guard 持零）')

# ---- 碰撞證據範圍 ----------------------------------------------------------
print('\n[碰撞證據範圍]')
print('    沿途檢查是**離線**、對 227 個離散取樣點做的點雲最近鄰距離，')
print('    取樣間隔 20 ms，且用的是 URDF <collision> 的取樣點雲（每 link ≤900 點）。')
print('    **這不是連續碰撞證明，也不是模擬器內的接觸量測**；')
print('    本次執行沒有讀取 Isaac 的接觸事件，因此不宣稱全程安全。')
print('=' * 66)
