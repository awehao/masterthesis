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
cb = d.get('callbacks', {})
# schema 3：軌跡點與保持命令分開，不再有單一的 published_msgs
pub = (sent.get('n_traj', 0) + sent.get('n_hold', 0)) if sent else None
third = None
_tp = os.path.join(os.path.dirname(os.path.abspath(a.run)), 'third_party_counts.json')
if os.path.exists(_tp):
    third = json.load(open(_tp))['stats']

print('\n[1] 命令交付（四個位置分開記，計數皆在回呼／發布入口）')
print(f'    {"位置":<34}{"數量":>8}  備註')
g = (third or {}).get('/wheel_guard/status', {})
cv = (third or {}).get('/cmd_vel', {})
print(f'    {"guard 發布（/wheel_guard/status）":<30}{g.get("count","—"):>8}  '
      f'跨度 {g.get("span_s","—")} s  約 {g.get("hz","—")} Hz  [**牆鐘**]')
print(f'    {"第三方收到 /cmd_vel":<32}{cv.get("count","—"):>8}  '
      f'匹配發布者 {cv.get("matched_publishers","—")}  約 {cv.get("hz","—")} Hz  [**牆鐘**]')
b = cb.get('base', {})
print(f'    {"Isaac /cmd_vel 回呼入口":<31}{b.get("entered","—"):>8}  '
      f'其中非零 {b.get("nonzero","—")}  首 {b.get("first_sim_t")}  '
      f'末 {b.get("last_sim_t")}  [**模擬時間**]')
print('    ！第三方那兩列是牆鐘、Isaac 回呼這列是模擬時間 —— '
      '**兩者的頻率不可相除**，不能據以推論接收比例。')
print(f'    {"手臂播放端發布":<34}{pub if pub is not None else "—":>8}  '
      + (f'軌跡 {sent.get("n_traj")} + 保持 {sent.get("n_hold")}；'
         f'匹配訂閱 前 {sent.get("matched_subs_before")} / '
         f'後 {sent.get("matched_subs_after")}'
         if sent else '（無 traj_sent.json）'))
m = cb.get('arm', {})
print(f'    {"Isaac 手臂回呼入口":<32}{m.get("entered","—"):>8}  '
      f'拒絕(長度) {m.get("rejected_bad_len","—")}  首 sim {m.get("first_sim_t")}  '
      f'末 sim {m.get("last_sim_t")}')
print(f'    {"Isaac 內容合格並記為命令":<30}{dl.get("received_msgs"):>8}')
print(f'    {"Isaac 實際套用關節目標":<31}{dl.get("applied_actions"):>8}')
if sent:
    print(f'    型別 {sent.get("msg_type")}')
delivered = bool(dl.get('received_msgs'))
arm_in = m.get('entered') or 0
base_in = b.get('entered') or 0
print('\n    判讀（只縮小範圍，不定案成因）：')
if arm_in == 0 and base_in == 0:
    print('    → 兩個受測 topic 都沒有進入 Isaac 的回呼。'
          '優先查共同的通訊設定與 executor；')
    print('      **尚不能區分「訊息沒抵達」與「抵達但回呼沒被執行」**，'
          '也不能就此推廣到所有外部輸入。')
elif base_in > 0 and arm_in == 0:
    print('    → 底盤有收到、手臂沒有。優先查手臂 topic 的發布、匹配、'
          '型別／QoS 與回呼路徑。')
elif arm_in > 0 and (dl.get('applied_actions') or 0) == 0:
    print('    → 手臂回呼有進、但未套用關節目標。往訊息驗證與套用層查'
          f'（長度拒絕 {m.get("rejected_bad_len")} 則）。')
elif not delivered:
    print('    → **命令未送達**。')
elif pub and dl['received_msgs'] < pub * 0.9:
    print(f'    → 收到 {dl["received_msgs"]}/{pub}，**有遺失**')
else:
    print('    → 命令交付成立。')
if not delivered:
    print('    以下「關節追蹤」與「TCP 誤差」皆為**未測到**，'
          '不代表控制器有問題，也不代表沒問題。')

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
