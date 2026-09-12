"""產生並驗證抽屜案例的時間參數化軌跡（含接近、拉開、釋放、退出）。

階段順序（**釋放在退出之前**）：
    reach     關節空間：test_start → 接近起點
    approach  笛卡兒直線：接近起點 → 夾持位（抽屜關閉）
    engage    原地停留，起點發出 engage 事件
    pull      笛卡兒直線：夾持位 → 全開位，抽屜同步被帶開
    hold      原地停留（開度必須維持在容差內）
    release   原地停留，起點發出 release 事件
    retreat   笛卡兒直線：退開；抽屜留在目標開度
    settle    原地停留

固定連接若還在，「退出」只會繼續把抽屜拉出來，所以 release 一定排在 retreat 前面。

時間參數化沿用**已凍結的** arm_traj.trapezoid，不另寫剖面：把 ∞-範數關節弧長
當成單一自由度餵給它，取回的就是同一條 s(t)。這樣速度／加速度上限與預抓取
軌跡是同一套實作，不會因為各寫一份而分岔。

產出後**對實際取樣點**驗證（不是對設計時的均勻網格）：
    每關節速度、加速度的實際最大值
    每個取樣點的限位餘裕、自碰、對櫃體／抽屜的餘裕
"""
import argparse, csv, hashlib, json, math, os, sys, time
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from arm_traj import trapezoid                                     # noqa: E402
import drawer_asset as DA                                          # noqa: E402
import drawer_kin as DK                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--case', default='drawer_open_a_fixed')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--spec', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--poses', default=os.path.join(
    WS, 'src/my_omnibot_description/config/arm_initial_pose.yaml'))
ap.add_argument('--urdf', default=os.path.join(
    WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--cart-step-m', type=float, default=0.002, help='笛卡兒段的 IK 取樣間距')
ap.add_argument('--engage-settle-s', type=float, default=2.0,
                help='連接前的沉降時間；事件在這段的**最後**才發')
ap.add_argument('--release-settle-s', type=float, default=1.0)
ap.add_argument('--settle-s', type=float, default=1.0)
ap.add_argument('--check-stride', type=int, default=5,
                help='幾何檢查的取樣間隔（1 = 每點都查）')
ap.add_argument('--hard', type=float, default=0.005, help='規劃器拒絕門檻 m')
ap.add_argument('--limit-margin', type=float, default=0.050, help='限位餘裕門檻 rad')
a = ap.parse_args()

CASES = yaml.safe_load(open(a.cases))
CASE = CASES['cases'][a.case]
SPEC = DA.load(a.spec)
POSES = yaml.safe_load(open(a.poses))
ARM = [f'joint{i}' for i in range(1, 7)]
Q_START = np.array([float(POSES[CASE['pregrasp']['start_config']][j]) for j in ARM])

DR = CASE['drawer']
TARGET = float(DR['target_opening_m'])
BACK_A = float(DR['approach_backoff_m'])
BACK_R = float(DR['retreat_backoff_m'])
VMAX = float(DR['vel_max_rps']); AMAX = float(DR['acc_max_rps2'])
HZ = float(DR['rate_hz'])
HOLD_S = float(CASE['tolerance']['opening_hold_s'])
PARK = (CASE['parking']['x'], CASE['parking']['y'],
        math.radians(CASE['parking']['yaw_deg']))
POSE = (CASE['object']['pose'][0], CASE['object']['pose'][1])
GRASP_MODEL = CASE['grasp_model']
F_OPEN = float(SPEC['grasp_surface']['finger_joint_open'])
F_CLOSED = float(CASE.get('grip', {}).get('finger_cmd_closed', 0.0))

os.makedirs(a.out, exist_ok=True)
K = DK.DrawerKin(a.urdf, SPEC, POSE, PARK)

tcp0 = DA.grasp_tcp_world(SPEC, POSE, 0.0)          # 夾持點（抽屜關閉）
X0 = float(tcp0[1] - PARK[1])                        # 底盤座標的 x
print(f'案例 {a.case}（抓取模型 {GRASP_MODEL}）')
print(f'  夾持 TCP 底盤 x：{X0:.4f}（關閉） → {X0-TARGET:.4f}（開 {TARGET:.3f} m）'
      f' → {X0-TARGET-BACK_R:.4f}（退出後）')
print(f'  接近起點 x = {X0-BACK_A:.4f}；速度上限 {VMAX} rad/s、加速度 {AMAX} rad/s²、'
      f'{HZ:.0f} Hz')


def tcp_at(x_base, q_d):
    """底盤座標 x 對應的世界 TCP（y 沿工具 z，即世界 +y）。"""
    p = DA.grasp_tcp_world(SPEC, POSE, q_d).copy()
    p[1] = PARK[1] + x_base
    return p


def cart_path(x_a, x_b, qd_a, qd_b, seed):
    """笛卡兒直線上的關節路徑（尚未配時）。開度與 x 同步線性變化。"""
    n = max(int(round(abs(x_b - x_a) / a.cart_step_m)) + 1, 2)
    xs = np.linspace(x_a, x_b, n)
    qds = np.linspace(qd_a, qd_b, n)
    Q, sw = [], 0
    prev = seed
    for x, qd in zip(xs, qds):
        r, s = K.ik_at(tcp_at(x, qd), prev, a.limit_margin)
        if s and prev is not None:
            sw += 1
        prev = r.q[K.idx].copy()
        Q.append(prev.copy())
        if not r.ok or r.pos_err > 1e-3:
            print(f'  **IK 未收斂** x={x:.4f} 殘差 {r.pos_err*1000:.3f} mm')
    return np.array(Q), qds, sw


def _measure(Q, hz):
    """實際送出的設定點串所要求的速度與加速度（逐步差分，不是中央差分）。

    量的是「相鄰設定點之間機器人必須達到的速率」，那才是位置介面真正的需求。
    """
    if len(Q) < 3:
        return np.zeros(6), np.zeros(6)
    V = np.diff(Q, axis=0) * hz
    A = np.diff(V, axis=0) * hz
    return np.abs(V).max(axis=0), np.abs(A).max(axis=0)


def _fit(build, vmax, amax, hz, tag):
    """收斂式時間縮放，直到**實際量到的**速度與加速度落在上限內。

    ∞-範數弧長參數化只保證「分段線性意義下」不超限；沿 s 重取樣之後，單一關節
    的瞬時速率仍可能溢出百分之一二。門檻（0.35 / 0.7）不因此放寬 —— 改成把規劃
    用的上限往下縮，讓產生出來的軌跡真的滿足原門檻。回傳 (Q, T, scale)。
    """
    scale = 1.0
    for _ in range(12):
        Q, T = build(vmax / scale, amax / (scale * scale))
        v, ac = _measure(Q, hz)
        rv = v.max() / vmax if vmax > 0 else 0.0
        ra = ac.max() / amax if amax > 0 else 0.0
        if rv <= 1.0 and ra <= 1.0:
            return Q, T * scale, scale
        scale *= max(rv, math.sqrt(max(ra, 0.0))) * 1.02
    print(f'  **{tag}：時間縮放未收斂**')
    return Q, T * scale, scale


def time_param(Q_path, vmax, amax, hz, tag='cart'):
    """用 ∞-範數關節弧長當單一自由度，餵給已凍結的 trapezoid 取回 s(t)。"""
    if len(Q_path) < 2:
        return Q_path.copy(), 0.0, 1.0
    seg = np.abs(np.diff(Q_path, axis=0)).max(axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(u[-1])
    if L < 1e-12:
        return Q_path[:1].copy(), 0.0, 1.0

    def build(vm, am):
        Sq, T = trapezoid(np.zeros(6), np.array([L, 0, 0, 0, 0, 0]), vm, am, hz)
        s = Sq[:, 0]
        return np.stack([np.interp(s, u, Q_path[:, j]) for j in range(6)],
                        axis=1), float(T)
    return _fit(build, vmax, amax, hz, tag)


def dwell(q, secs, hz):
    n = max(int(round(secs * hz)), 1)
    return np.tile(np.asarray(q, float), (n, 1))


# ------------------------------------------------------------------ 建立各段
rows = []          # (phase, event, q(6), finger, opening)
seed = None

# reach：關節空間。從 all-zeros 出發，joint3 往負向 0.05 rad 內就會相交，
# 所以這一段的方向必須檢查，不能只看兩端。
r0, _ = K.ik_at(tcp_at(X0 - BACK_A, 0.0), None, a.limit_margin)
q_appr = r0.q[K.idx].copy()
Qr, Tr, sc_r = _fit(lambda vm, am: (lambda R: (R[0], R[1]))(
    trapezoid(Q_START, q_appr, vm, am, HZ)), VMAX, AMAX, HZ, 'reach')
for q in Qr:
    rows.append(('reach', '', q, F_OPEN, 0.0))
seed = q_appr

Qa, qda, sw_a = cart_path(X0 - BACK_A, X0, 0.0, 0.0, seed)
Qa_t, Ta, sc_a = time_param(Qa, VMAX, AMAX, HZ, 'approach')
for q in Qa_t:
    rows.append(('approach', '', q, F_OPEN, 0.0))
seed = Qa_t[-1].copy()

# engage：原地停留。fixed 版手指**維持全開** —— 讓固定關節單獨承擔，
# 再去夾住橫桿會產生接觸力，那就不是「理想固定連接」了。
# engage 事件排在沉降**之後**：第一版排在接近段結束後 20 ms 就發，手臂還有
# 2.96 mm 的動態落後沒收斂，固定關節把這個殘差鎖成預壓，位置驅動接著把抽屜
# 頂在關閉硬限位上，量到 −147 N 沿抽屜軸的持續力。先讓命令收斂再連接。
f_hold = F_OPEN if GRASP_MODEL == 'fixed_attachment' else F_CLOSED
_eng = dwell(seed, a.engage_settle_s, HZ)
for i, q in enumerate(_eng):
    last = (i == len(_eng) - 1)
    rows.append(('engage', 'engage' if last else '', q,
                 F_OPEN if GRASP_MODEL == 'fixed_attachment' else
                 (f_hold if last else F_OPEN), 0.0))

Qp, qdp, sw_p = cart_path(X0, X0 - TARGET, 0.0, TARGET, seed)
Qp_t, Tp, sc_p = time_param(Qp, VMAX, AMAX, HZ, 'pull')
# 開度由 TCP 的 x 反推，與軌跡同步（不是獨立給的）
for q in Qp_t:
    x = float(K.tcp_world(q)[1, 3] - PARK[1])
    rows.append(('pull', '', q, f_hold, max(0.0, X0 - x)))
seed = Qp_t[-1].copy()

for q in dwell(seed, HOLD_S, HZ):
    rows.append(('hold', '', q, f_hold, TARGET))
for i, q in enumerate(dwell(seed, a.release_settle_s, HZ)):
    rows.append(('release', 'release' if i == 0 else '', q, F_OPEN, TARGET))

Qt, qdt, sw_t = cart_path(X0 - TARGET, X0 - TARGET - BACK_R, TARGET, TARGET, seed)
Qt_t, Tt, sc_t = time_param(Qt, VMAX, AMAX, HZ, 'retreat')
for q in Qt_t:
    rows.append(('retreat', '', q, F_OPEN, TARGET))
seed = Qt_t[-1].copy()
for q in dwell(seed, a.settle_s, HZ):
    rows.append(('settle', '', q, F_OPEN, TARGET))

DTS = 1.0 / HZ
Q = np.array([r[2] for r in rows])
FING = np.array([r[3] for r in rows])
OPEN = np.array([r[4] for r in rows])
TS = np.arange(len(rows)) * DTS
print(f'\n  {len(rows)} 個設定點，總時長 {TS[-1]:.3f} s')
ph_span = {}
for i, r in enumerate(rows):
    ph_span.setdefault(r[0], [i, i])[1] = i
for p, (i0, i1) in ph_span.items():
    print(f'    {p:9s} {i1-i0+1:5d} 點  {TS[i0]:7.3f} – {TS[i1]:7.3f} s')

# ---------------------------------------------------- 驗證：速度與加速度
vmax_got, amax_got = _measure(Q, HZ)
print(f'  規劃用的時間縮放 reach {sc_r:.4f} / approach {sc_a:.4f} / '
      f'pull {sc_p:.4f} / retreat {sc_t:.4f}（1.0 = 未縮放）')
print(f'\n  實際最大關節速度   {np.round(vmax_got,4).tolist()}  上限 {VMAX}')
print(f'  實際最大關節加速度 {np.round(amax_got,4).tolist()}  上限 {AMAX}')
fail = []
if vmax_got.max() > VMAX + 1e-6:
    fail.append(f'速度超限 {vmax_got.max():.4f} > {VMAX}')
if amax_got.max() > AMAX + 1e-6:
    fail.append(f'加速度超限 {amax_got.max():.4f} > {AMAX}')
# 相鄰設定點的跳動（連接前後不得有位置跳變）
step = np.abs(np.diff(Q, axis=0)).max(axis=1)
i_st = int(np.argmax(step))
print(f'  相鄰設定點最大跳動 {step.max():.5f} rad（第 {i_st} 點，'
      f'{rows[i_st][0]} → {rows[i_st+1][0]}）')
for p, (i0, i1) in ph_span.items():
    if i0 > 0 and step[i0 - 1] > VMAX * DTS + 1e-6:
        fail.append(f'{p} 起點有位置跳變 {step[i0-1]:.5f} rad')

# ---------------------------------------------------- 驗證：取樣點幾何
idxs = sorted(set(list(range(0, len(rows), max(a.check_stride, 1)))
                  + [i for i0, i1 in ph_span.values() for i in (i0, i1)]
                  + [len(rows) - 1]))
print(f'\n  幾何檢查 {len(idxs)} / {len(rows)} 點（stride {a.check_stride}，含所有段界）')
t0 = time.time()
worst = {'lm': (9e9, None), 'self': (9e9, None), 'env': (9e9, None), 'shell': (9e9, None)}
for c, i in enumerate(idxs):
    qa = Q[i]
    m, jb = K.margin(qa)
    e = K.evaluate(qa, OPEN[i], FING[i])
    if m < worst['lm'][0]: worst['lm'] = (m, f'{rows[i][0]}@{TS[i]:.2f}s joint{jb}')
    if e['self'] < worst['self'][0]: worst['self'] = (e['self'], f'{rows[i][0]} {e["self_pair"]}')
    if e['env'] < worst['env'][0]: worst['env'] = (e['env'], f'{rows[i][0]} {e["env_pair"]}')
    if e['shell_bar'] < worst['shell'][0]: worst['shell'] = (e['shell_bar'], rows[i][0])
    if c % 25 == 0:
        print(f'    {c}/{len(idxs)} …', flush=True)
print(f'  （{time.time()-t0:.1f} s）')
print(f'  限位餘裕最小      {worst["lm"][0]:.4f} rad   {worst["lm"][1]}')
print(f'  自碰餘裕最小      {worst["self"][0]*1000:.2f} mm   {worst["self"][1]}')
print(f'  對櫃體／抽屜最小  {worst["env"][0]*1000:.2f} mm   {worst["env"][1]}')
print(f'  夾爪殼對橫桿最小  {worst["shell"][0]*1000:.2f} mm   {worst["shell"][1]}')
j3 = Q[:12, 2]
print(f'  joint3 起步前 12 點 {np.round(j3,5).tolist()}')
print(f'    （安全下限 −0.0611，起點 0.0 距中止線 {0.0611-a.limit_margin:.4f} rad；'
      f'{"往正向離開，方向正確" if j3[1] >= j3[0] - 1e-12 else "**往負向，方向錯誤**"}）')
if j3.min() < -1e-9:
    fail.append('joint3 起步往負向')
encl = DK.enclosure_margin(SPEC)
print(f'  手指全開包覆餘裕  {encl*1000:.2f} mm（解析值）')
if worst['lm'][0] < a.limit_margin: fail.append('限位餘裕不足')
if worst['self'][0] < a.hard: fail.append('自碰低於門檻')
if worst['env'][0] < a.hard: fail.append('環境低於門檻')
if worst['shell'][0] < a.hard: fail.append('夾爪殼碰到橫桿')

# ---------------------------------------------------------------- 寫出
csv_p = os.path.join(a.out, 'traj.csv')
with open(csv_p, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['seq', 't', 'phase', 'event'] + ARM + ['finger', 'expected_opening'])
    for i, r in enumerate(rows):
        w.writerow([i, f'{TS[i]:.4f}', r[0], r[1]]
                   + [f'{v:.9f}' for v in r[2]] + [f'{r[3]:.6f}', f'{r[4]:.6f}'])
meta = {
    'schema': 'drawer_traj/1', 'case': a.case, 'grasp_model': GRASP_MODEL,
    'spec_sha256_16': hashlib.sha256(open(a.spec, 'rb').read()).hexdigest()[:16],
    'cases_sha256_16': hashlib.sha256(open(a.cases, 'rb').read()).hexdigest()[:16],
    'urdf_sha256_16': hashlib.sha256(open(a.urdf, 'rb').read()).hexdigest()[:16],
    'rate_hz': HZ, 'n_points': len(rows), 'duration_s': float(TS[-1]),
    'x_closed': X0, 'x_open': X0 - TARGET, 'x_retreat': X0 - TARGET - BACK_R,
    'target_opening_m': TARGET,
    'phases': {p: {'i0': i0, 'i1': i1, 't0': float(TS[i0]), 't1': float(TS[i1])}
               for p, (i0, i1) in ph_span.items()},
    'limits': {'vel_max_rps': VMAX, 'acc_max_rps2': AMAX},
    'time_scale': {'reach': sc_r, 'approach': sc_a, 'pull': sc_p, 'retreat': sc_t},
    'measured': {'vel_max_per_joint': vmax_got.tolist(),
                 'acc_max_per_joint': amax_got.tolist(),
                 'max_step_rad': float(step.max())},
    'geometry': {'checked_points': len(idxs), 'stride': a.check_stride,
                 'limit_margin_min_rad': worst['lm'][0],
                 'limit_margin_at': worst['lm'][1],
                 'self_min_m': worst['self'][0], 'self_at': worst['self'][1],
                 'env_min_m': worst['env'][0], 'env_at': worst['env'][1],
                 'shell_bar_min_m': worst['shell'][0],
                 'enclosure_margin_m': encl},
    'ik_branch_switches': {'approach': sw_a, 'pull': sw_p, 'retreat': sw_t},
    'fail': fail,
    'pass': not fail,
    'note': ('幾何為離散取樣，不是連續碰撞證明；stride 見上。'
             '底盤假設精確停在停放位姿。'),
}
json.dump(meta, open(os.path.join(a.out, 'traj_meta.json'), 'w'),
          ensure_ascii=False, indent=2)
print(f'\n判定：{"通過" if not fail else "**未通過**"}')
for f_ in fail:
    print(f'   ✗ {f_}')
print(f'輸出：{csv_p}')
sys.exit(0 if not fail else 1)
