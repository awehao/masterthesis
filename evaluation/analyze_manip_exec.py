"""軌跡執行一致性分析：分段、對齊、對**實際關節路徑**重做幾何檢查。

四件事分開：

  交付與套用   逐序號追：發布 → 回呼收到 → 物理步套用。
               「收到」不等於「套用」—— 兩個物理步之間收到多則時，
               較早的設定點會被覆寫而從未套用。
  時間一致性   軌跡以模擬時間排程，比較預定執行時刻與實際進回呼／被套用的時刻。
  追蹤誤差     只在**運動段**計算，且以 applied_seq 對齊參考值，
               不拿當下的 node.cmd 充當「該物理步套用的設定點」。
  幾何         對**實際量到的關節路徑**重做離散最近鄰檢查，
               而不是沿用對參考軌跡做的那一份。仍是離散取樣，非連續證明。
"""
import argparse, json, math, os, sys
import numpy as np, yaml
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
sys.path.insert(0, HERE)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
from verify_self_collision import link_clouds, DESIGNED_CONTACT

ARM = [f'joint{i}' for i in range(1, 7)]
ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--sent', required=True)
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--wb-urdf', default=os.path.join(
    WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
ap.add_argument('--geom-stride', type=int, default=10, help='幾何檢查的取樣間隔（物理步）')
ap.add_argument('--warn', type=float, default=0.020)
ap.add_argument('--hard', type=float, default=0.005)
a = ap.parse_args()

d = json.load(open(a.run)); sent = json.load(open(a.sent))
C = yaml.safe_load(open(a.cases))['cases'][d['case']]
log = d['log']; st = d['seq_trace']
ref = {m['seq']: m for m in sent['msgs']}
Qref = np.array(sent['traj'])
n_traj = sent['n_traj']

print('=' * 70)
print(f'軌跡執行一致性  {d["case"]}   stop={d["stop_reason"]}  sim {d["sim_time"]:.2f} s')
print(f'  軌跡起點 sim {sent["t_start_sim"]:.3f} s；排程時鐘 = {sent["clock"]}')
print('=' * 70)

# ---- 1. 交付與套用（軌跡點 / 保持 分開）--------------------------------------
rx = {r[0]: r for r in st['received']}
applied = set(st['applied_seqs'])
traj_seqs = {m['seq'] for m in sent['msgs'] if m['kind'] == 0}
hold_seqs = {m['seq'] for m in sent['msgs'] if m['kind'] == 1}
print('\n[1] 交付與套用（軌跡點與終點保持分開）')
print(f'    {"":16}{"發布":>8}{"回呼收到":>10}{"曾被套用":>10}{"收到未套用":>12}')
for lab, S in (('軌跡設定點', traj_seqs), ('終點保持', hold_seqs)):
    got = S & set(rx); app = S & applied
    print(f'    {lab:<14}{len(S):>8}{len(got):>10}{len(app):>10}{len(got-app):>12}')
print(f'    物理步套用次數合計 {d["delivery"]["applied_actions"]}'
      f'（含重複使用同一設定點，非相異軌跡點數）')

# ---- 2. 時間一致性 ----------------------------------------------------------
print('\n[2] 時間一致性（皆為模擬時間）')
lat = [(rx[s][3] - (sent['t_start_sim'] + ref[s]['t_sched']))
       for s in sorted(traj_seqs & set(rx))]
if lat:
    lat = np.array(lat)
    print(f'    軌跡點「進回呼」相對預定時刻：中位 {np.median(lat)*1000:+.1f} ms  '
          f'p95 {np.percentile(lat,95)*1000:+.1f} ms  max {lat.max()*1000:+.1f} ms')
pub_late = [m['late_s'] for m in sent['msgs']
            if m['kind'] == 0 and m['late_s'] is not None]
if pub_late:
    pl = np.array(pub_late)
    print(f'    軌跡點「發布」相對預定時刻：中位 {np.median(pl)*1000:+.1f} ms  '
          f'max {pl.max()*1000:+.1f} ms')
# 軌跡類與保持類必須分開算跨度。把兩者合在一起會把「到位後長時間保持」
# 算進運動時間 —— 本檔先前就是這樣得出「拉長 9.6 倍」的錯誤結論。
ex_t = [r for r in log if r.get('applied_seq') in traj_seqs]
ex_h = [r for r in log if r.get('applied_seq') in hold_seqs]
if ex_t:
    t0_, t1_ = ex_t[0]['t'], ex_t[-1]['t']
    print(f'    **軌跡類**設定點的套用區間 {t0_:.3f} – {t1_:.3f} s '
          f'（跨度 **{t1_-t0_:.3f} s**，標稱 {sent["T_traj"]:.2f} s）'
          f'  seq {ex_t[0]["applied_seq"]} → {ex_t[-1]["applied_seq"]}')
if ex_h:
    h0, h1 = ex_h[0]['t'], ex_h[-1]['t']
    print(f'    保持類的套用區間     {h0:.3f} – {h1:.3f} s '
          f'（跨度 {h1-h0:.3f} s，含到模擬時間上限為止的保持）')
    print(f'    （兩者合計 {ex_h[-1]["t"]-ex_t[0]["t"]:.3f} s —— '
          f'**這個數字不是運動時間，不可用來說軌跡被拉長**）')

# ---- 3. 追蹤誤差：只在運動段，且以 applied_seq 對齊 --------------------------
print('\n[3] 追蹤誤差（兩種參考分開算，回答不同問題）')
T_START = sent['t_start_sim']; dt_ref = sent['dt']; T_traj = sent['T_traj']

def q_ref_at(t):
    """預定時間軌跡 q_ref(t)：起點前夾住 q0，終點後夾住 q1。"""
    u = (t - T_START) / dt_ref
    if u <= 0:
        return Qref[0]
    if u >= len(Qref) - 1:
        return Qref[-1]
    i0 = int(math.floor(u)); f = u - i0
    return Qref[i0] * (1 - f) + Qref[i0 + 1] * f

rows = [r for r in log if r.get('applied_seq') in traj_seqs]
print(f'    (a) 相對**該步實際套用的設定點**（檢查位置驅動的追蹤）')
if not rows:
    print('        運動段沒有樣本')
else:
    ea = np.array([np.max(np.abs(np.array(r['q']) - np.array(r['cmd'])))
                   for r in rows])
    print(f'        樣本 {len(ea)} 物理步，相異序號 '
          f'{len({r["applied_seq"] for r in rows})}')
    print(f'        p50 {np.median(ea)*1000:.2f}  p95 {np.percentile(ea,95)*1000:.2f}'
          f'  max {ea.max()*1000:.2f} mrad')

print(f'    (b) 相對**預定時間軌跡 q_ref(t)**（檢查整體延遲、跳點與執行偏差）')
win = [r for r in log if T_START <= r['t'] <= T_START + T_traj]
if not win:
    print('        軌跡時間窗內沒有樣本')
else:
    eb = np.array([np.max(np.abs(np.array(r['q']) - q_ref_at(r['t']))) for r in win])
    print(f'        樣本 {len(eb)} 物理步（時間窗 {T_START:.3f} – '
          f'{T_START+T_traj:.3f} s）')
    print(f'        p50 {np.median(eb)*1000:.2f}  p95 {np.percentile(eb,95)*1000:.2f}'
          f'  max {eb.max()*1000:.2f} mrad')
    k = int(np.argmax(eb))
    print(f'        最大於 sim {win[k]["t"]:.3f} s（軌跡內 '
          f'{win[k]["t"]-T_START:.3f} s），該步 applied_seq={win[k].get("applied_seq")}')

hold = [r for r in log if r.get('applied_seq') in hold_seqs]
if hold:
    eh = np.array([np.max(np.abs(np.array(r['q']) - Qref[-1])) for r in hold])
    print(f'    (c) 保持段對終點（{len(eh)} 步）：p50 {np.median(eh)*1000:.2f}'
          f'  max {eh.max()*1000:.2f} mrad')

# ---- 4. 對實際關節路徑重做幾何檢查 ------------------------------------------
print(f'\n[4] 幾何：對**實際量到的關節路徑**重做離散檢查（每 {a.geom_stride} 個物理步取樣）')
xml = open(a.wb_urdf).read()
K = WholeBodyKinematics.from_urdf_string(xml)
clouds = link_clouds(xml)
adj, rigid = set(), {}
def find(x):
    while rigid.get(x, x) != x:
        x = rigid[x]
    return x
for j in K.joints.values():
    adj.add(frozenset((j.parent, j.child)))
    if j.jtype not in ('revolute', 'prismatic', 'continuous'):
        p_, c_ = find(j.parent), find(j.child)
        if p_ != c_:
            rigid[p_] = c_
names = [n for n in clouds if n in K.parent_of or n == 'base_link']
pairs = [(x, y) for i, x in enumerate(names) for y in names[i+1:]
         if frozenset((x, y)) not in adj and frozenset((x, y)) not in DESIGNED_CONTACT
         and find(x) != find(y)]
px, py = C['parking']['x'], C['parking']['y']
th = math.radians(C['parking']['yaw_deg'])

import re
ws = re.sub(r'<!--.*?-->', '', open(os.path.join(
    WS, 'src/ammr_bringup/worlds/bigarena.sdf')).read(), flags=re.S)
boxes = []
for m in re.finditer(r'<model name="(known_obs_\d+)">(.*?)</model>', ws, re.S):
    p = [float(v) for v in re.search(r'<pose>([^<]+)</pose>', m.group(2)).group(1).split()]
    sz = [float(v) for v in re.search(r'<box>\s*<size>([^<]+)</size>',
                                      m.group(2)).group(1).split()]
    boxes.append((m.group(1), np.array(p[:3]), np.array(sz)))

def box_dist(P, c, sz):
    dd = np.abs(P - c) - sz / 2.0
    return (np.linalg.norm(np.maximum(dd, 0.0), axis=1)
            + np.minimum(dd.max(axis=1), 0.0))

sel = [r for r in log if r.get('applied_seq') is not None][::a.geom_stride]
ws_self, ws_env, at_s, at_e = 9.9, 9.9, None, None
for r in sel:
    q = np.zeros(9); q[0], q[1], q[2] = px, py, th
    q[3:9] = r['q']
    world = {}
    for nmk, P in clouds.items():
        if nmk not in K.parent_of and nmk != 'base_link':
            continue
        Tm = K.fk(q, nmk)
        world[nmk] = (Tm[:3, :3] @ P.T).T + Tm[:3, 3]
    for x, y in pairs:
        if x in world and y in world:
            dv = cKDTree(world[x]).query(world[y], k=1)[0].min()
            if dv < ws_self:
                ws_self, at_s = dv, (r['t'], (x, y))
    for nmk, P in world.items():
        if nmk == 'base_link' or nmk.startswith(('rim', 'roller')):
            continue
        for bn, c, sz in boxes:
            dv = box_dist(P, c, sz).min()
            if dv < ws_env:
                ws_env, at_e = dv, (r['t'], (nmk, bn))
print(f'    取樣 {len(sel)} 個實際姿態')
print(f'    實際路徑最小自碰餘裕  {ws_self*1000:8.2f} mm  於 sim {at_s[0]:.2f} s  {at_s[1]}')
print(f'    實際路徑最小環境餘裕  {ws_env*1000:8.2f} mm  於 sim {at_e[0]:.2f} s  {at_e[1]}')
print(f'    （參考軌跡那份的對應值：自碰 8.71 mm、環境 242.49 mm）')
print(f'    門檻：{a.hard*1000:.0f} mm 規劃器拒絕 / {a.warn*1000:.0f} mm 舒適線；'
      f'**離散取樣，非連續無碰撞證明**')

# ---- 4b. 到位時間：四個時刻分列，並對照事前定義的容差 --------------------
print('\n[4b] 到位時間（四個時刻分列，對照**事前定義**的容差）')
obj0 = C['object']; pg0 = C['pregrasp']
fc0 = obj0['face_center']
nx0, ny0 = {'+x': (1, 0), '-x': (-1, 0),
            '+y': (0, 1), '-y': (0, -1)}[obj0['approach_face']]
tgt0 = np.array([fc0[0] + nx0 * pg0['standoff_from_face_m'],
                 fc0[1] + ny0 * pg0['standoff_from_face_m'], pg0['tcp_xyz'][2]])
tol_p = C['tolerance']['tcp_pos_m']
hold_s = C['tolerance'].get('arrival_hold_s', 1.0)
tol_t = C['tolerance'].get('arrival_time_s')
te = [(r['t'], float(np.linalg.norm(np.array(r['tcp']) - tgt0))) for r in log]
seg_start = seg_confirm = None
i = 0
while i < len(te):
    if te[i][1] > tol_p:
        i += 1; continue
    j2 = i
    while j2 < len(te) and te[j2][1] <= tol_p:
        if te[j2][0] - te[i][0] >= hold_s:
            seg_start, seg_confirm = te[i][0], te[j2][0]
            break
        j2 += 1
    if seg_start is not None:
        break
    i = j2 + 1
print(f'    ① 預定軌跡起點              sim {T_START:.3f} s')
first_app = rows[0]['t'] if rows else None
print(f'    ② 首次實際套用軌跡設定點    sim '
      + (f'{first_app:.3f} s（相對 ① +{first_app-T_START:.3f} s，起步延遲）'
         if first_app else '—'))
if seg_start is None:
    print(f'    ③ 誤差 ≤{tol_p*1000:.1f} mm 的持續區段起點  —— 未出現')
    print(f'    ④ 持續 {hold_s:.1f} s 的確認時刻            —— 未出現')
else:
    print(f'    ③ 誤差 ≤{tol_p*1000:.1f} mm 的持續區段起點  sim {seg_start:.3f} s'
          f'（事後辨識）')
    print(f'    ④ 持續 {hold_s:.1f} s 的確認時刻            sim {seg_confirm:.3f} s')
    d_sched = seg_start - T_START
    print(f'\n    到位時間 = ③ − ① = **{d_sched:.3f} s**；軌跡標稱 {T_traj:.3f} s'
          f'  → 慢 **{d_sched-T_traj:+.3f} s**')
    if first_app:
        print(f'    （若改由 ② 起算為 {seg_start-first_app:.3f} s —— '
              f'這會略去起步延遲，**不可用來宣稱準時**）')
    if tol_t is not None:
        ok_t = abs(d_sched - T_traj) <= tol_t
        print(f'    對照事前定義的容差 ±{tol_t:.2f} s：'
              f'**{"合格" if ok_t else "不合格"}**')
    else:
        print('    **案例未定義到位時間容差，因此不判定合格與否**')

# ---- 5. TCP 位置與姿態 ------------------------------------------------------
print('\n[5] TCP 位置與**工具軸方向**（對箱體錨定目標）')
obj = C['object']; pg = C['pregrasp']
fc = obj['face_center']
nx, ny = {'+x': (1, 0), '-x': (-1, 0), '+y': (0, 1), '-y': (0, -1)}[obj['approach_face']]
tgt = np.array([fc[0] + nx * pg['standoff_from_face_m'],
                fc[1] + ny * pg['standoff_from_face_m'], pg['tcp_xyz'][2]])
tcp = np.array(d['tcp_final_world'])
print(f'    位置目標 ({tgt[0]:.4f}, {tgt[1]:.4f}, {tgt[2]:.4f})')
print(f'    實際     ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f})   '
      f'誤差 {np.linalg.norm(tcp-tgt):.4f} m  容差 {C["tolerance"]["tcp_pos_m"]:.4f}')

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

R = quat_to_R(d['tcp_final_quat_wxyz'])
z_act = R[:, 2]
fwd = np.array(pg['tcp_forward_axis'], float)          # 底盤座標的工具指向
c, s_ = math.cos(th), math.sin(th)
z_tgt = np.array([c*fwd[0] - s_*fwd[1], s_*fwd[0] + c*fwd[1], fwd[2]])
ang = math.degrees(math.acos(float(np.clip(np.dot(z_act, z_tgt), -1, 1))))
print(f'    工具軸目標（世界） ({z_tgt[0]:+.4f}, {z_tgt[1]:+.4f}, {z_tgt[2]:+.4f})')
print(f'    工具軸實際（世界） ({z_act[0]:+.4f}, {z_act[1]:+.4f}, {z_act[2]:+.4f})')
print(f'    夾角 **{ang:.3f}°**   容差 {C["tolerance"]["tcp_rot_deg"]:.1f}°  '
      f'{"合格" if ang <= C["tolerance"]["tcp_rot_deg"] else "**不合格**"}')
print('    注意：這是**工具軸方向**，不是完整三維姿態 —— '
      '繞工具軸的旋轉未受此判準約束。')
print('=' * 70)
