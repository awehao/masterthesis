#!/usr/bin/env python3
"""茶几圓柱體隨機放：離線可達性地圖（Claude 自行裁定 20261117_2315 第一步；不動控制、不跑 Isaac）。
桌面（茶几中心 (1.35, 2.3)、0.6×1.1 m、高 0.45 m）上離邊 ≥0.05 m 的 5 cm 網格，每個圓柱位置：
  候選底盤位姿＝西／南／北三側（東側到沙發只剩約 0.45 m，不列），底盤中心離桌邊 d∈D_SIDE、沿邊橫移 lat∈LAT、朝向＝正對桌邊；
  抓取 TCP＝瓶子線側向水平抓取定義（TCP z＝水平接近方向＝底盤到圓柱、TCP y＝水平開合、TCP 在圓柱中段 z＝0.55 m），預抓＝沿接近軸後退 0.06 m；
  可達＝抓取與預抓兩個位姿 IK 殘差 ≤1e-4、在 LITE6_SAFE 內（留 0.05 rad）、σ_min ≥ 0.02；
  淨空（粗核）：底盤圓（半徑 0.30 m）不碰家具箱、手臂各連桿原點在桌面投影內時高度 ≥ 0.50 m（抓取構型）。
輸出：每格最佳側、最佳候選、可達與否；整體可達比例；PNG 圖＋JSON。
    python3 evaluation/cyl_reach_map.py OUT_PREFIX"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_motm_grasp_ctrl as G                                     # noqa: E402
import door_fast_kin as DFK                                           # noqa: E402

K, XO = G.K, G.XO
LIM = XO.LITE6_SAFE
OUT = sys.argv[1] if (__name__ == '__main__' and len(sys.argv) > 1) else None
scene = json.load(open(os.path.join(HERE, 'experiments', os.environ.get('M8_SCENE_REG', '20261112_m8_house_scene.json'))))
FURN = {f['name']: f for f in scene['furniture']}
TAB = FURN['coffee_table']
TX, TY, _ = TAB['c']; SX, SY, SZ = TAB['s']
TOP = SZ
X0, X1, Y0, Y1 = TX - SX / 2, TX + SX / 2, TY - SY / 2, TY + SY / 2
H_GRASP, PRE = TOP + 0.10, 0.06
RB = 0.30
D_SIDE = tuple(float(x) for x in os.environ.get('CYL_D_SIDE', '0.43,0.46,0.48,0.51,0.53').split(','))   # 底盤中心離桌邊；淨空另以控制器同一份 clear_door_fast＋門檻核
LAT = tuple(np.round(np.arange(-0.30, 0.301, 0.05), 3))
SEEDS = [np.array([-0.0128, -0.0519, 0.086, 0.0116, -1.6249, -0.0018]), np.array([-0.354, -0.088, 1.435, -1.421, -0.358, -0.157]),
         np.array([0.0, 0.3, 1.2, 0.0, -0.9, 0.0]), np.array([0.0, 0.6, 1.6, 0.0, -1.0, 0.0])]
LINKS = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link_eef', 'link_tcp']
BOXES = [(np.array(o['c']), np.array(o['s'])) for o in scene['walls'] + scene['furniture']]


def grasp_T(c_xy, a, flip=False):
    sg = -1.0 if flip else 1.0                                        # 兩指對稱：開合軸正反兩向是同一種抓法（瓶子線 grasp_pose 的 flip）
    R = np.zeros((3, 3)); R[:, 2] = [a[0], a[1], 0.0]; R[:, 1] = [-sg * a[1], sg * a[0], 0.0]; R[:, 0] = np.cross(R[:, 1], R[:, 2])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [c_xy[0], c_xy[1], H_GRASP]
    return T


def base_clear(b, margin):
    """底盤圓（半徑 RB）離登錄方盒 ≥ margin（與控制器／世界的底盤淨空門檻 0.10 一致時用 0.12）。"""
    for cc, s in BOXES:
        dx = max(abs(b[0] - cc[0]) - s[0] / 2, 0.0); dy = max(abs(b[1] - cc[1]) - s[1] / 2, 0.0)
        if math.hypot(dx, dy) < RB + margin:
            return False
    return True


def base_free(b):
    for c, s in BOXES:
        dx = max(abs(b[0] - c[0]) - s[0] / 2, 0.0); dy = max(abs(b[1] - c[1]) - s[1] / 2, 0.0)
        if math.hypot(dx, dy) < RB:
            return False
    return True


SCAN_PTS = None                                                       # 目前雷射看到、但不在場景登錄（地圖）裡的點（里程座標 N×2）；None＝只看地圖
SCAN_CLEAR = float(os.environ.get('CYL_SCAN_CLEAR', '0.42'))
APPROACH_CLEAR = float(os.environ['CYL_APPROACH_CLEAR']) if os.environ.get('CYL_APPROACH_CLEAR') else (0.15 if os.environ.get('PICK_PATH') == '1' else None)   # 選站位：目前位置直線到站位的底盤淨空
APPROACH_PEN = float(os.environ.get('CYL_APPROACH_PEN', '2.0'))
PATH_BASE_MARGIN = float(os.environ.get('PATH_BASE_MARGIN', '0.12'))   # 經過段底盤離家具（控制器門檻 0.10＋餘裕）
SCAN_CLEAR_EXT = float(os.environ.get('CYL_SCAN_CLEAR_EXT', '0.36'))   # 經過段（機器人已在附近）：Shield 半徑 0.30＋d0 0.05＋0.01（選位用較保守的 SCAN_CLEAR；s113d 目前位置就 <0.42 ⇒ 誤判整段不可行）
SCAN_TAIL = float(os.environ.get('CYL_SCAN_TAIL', '0.0'))           # 閉爪窗之後沿線再核的長度（抬起後轉交導航）         # 底盤中心離未知點至少（底盤半徑 0.30＋0.15）：Isaac 第三十二趟路線起點離 unknown_l5 角 0.2 m ⇒ Shield 擋住對齊


def set_scan_pts(P):
    """只留地圖解釋不了的點（離所有登錄方盒 >0.08 m）；回保留數。"""
    global SCAN_PTS
    if P is None or len(P) == 0:
        SCAN_PTS = None
        return 0
    keep = []
    for p in np.asarray(P, float):
        ok = True
        for c, s in BOXES:
            dx = max(abs(p[0] - c[0]) - s[0] / 2, 0.0); dy = max(abs(p[1] - c[1]) - s[1] / 2, 0.0)
            if math.hypot(dx, dy) < 0.08:
                ok = False; break
        if ok:
            keep.append(p)
    SCAN_PTS = np.array(keep) if keep else None
    return 0 if SCAN_PTS is None else len(SCAN_PTS)


def _poly_at(P, cum, s):
    """折線 P（累積弧長 cum）上弧長 s 的位置與單位切向（超出兩端沿端點切向外插）。"""
    s = float(s); i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(P) - 2))
    seg = P[i + 1] - P[i]; L = max(float(cum[i + 1] - cum[i]), 1e-9)
    return P[i] + seg * ((s - cum[i]) / L), seg / L


def path_ext(T_grasp, p_now, th_now, dirv, b0_sel, v_now, extra_dirs=None, path_pts=None, ds=0.05, s_max=1.2, t_need=1.6, v_min=0.04, v_cap=float(os.environ.get('PATH_V_CAP', '0.15'))):
    """依導航路徑的移動中抓取（Howard 11/20：不要預設路線）：自目前位置沿目前行進方向（沒在往抓取位前進 ⇒ 朝抓取位）
    每 ds 取樣底盤位（目前偏航、面桌偏航兩種），核底盤淨空（登錄方盒＋雷射未知點）、抓取 IK、全身淨空；撞到就不再往前取樣。
    取第一段連續可行區 [s_a, s_b]：閉爪點 s_g＝s_a＋min(0.15, L/3)，經過速度＝(s_b−s_g)/t_need（閉爪＋確認需時），夾在 [v_min, max(v_now, 0.08), v_cap]。
    回 motm dict（無對齊、自目前投影起算）或 None（沿此方向抓不到 ⇒ 呼叫端用停車抓）。"""
    p = np.asarray(p_now, float)[:2]; b0 = np.asarray(b0_sel, float)
    to_b = b0[:2] - p
    dm = np.asarray(dirv, float)[:2]; nm = float(np.linalg.norm(dm))
    dirs = []
    if nm > 1e-3:
        dirs.append(dm / nm)                                           # 實際行進方向
    if float(np.linalg.norm(to_b)) > 1e-3:
        dirs.append(to_b / float(np.linalg.norm(to_b)))
    for xd in (extra_dirs or []):                                       # 導航路徑最後一段切向（實際會走的方向）
        xd = np.asarray(xd, float)[:2]
        if float(np.linalg.norm(xd)) > 1e-3:
            dirs.append(xd / float(np.linalg.norm(xd)))               # 朝抓取位（s119d：路徑還在轉彎，瞬時行進方向側向偏 0.59 m 永遠進不了可達區）
    best = None
    ths = [float(th_now)]
    if abs((float(b0[2]) - float(th_now) + math.pi) % (2 * math.pi) - math.pi) > 0.15:
        ths.append(float(b0[2]))
    c0 = float(np.min(np.hypot(SCAN_PTS[:, 0] - p[0], SCAN_PTS[:, 1] - p[1]))) if SCAN_PTS is not None else 9.0
    clr = min(SCAN_CLEAR_EXT, max(0.33, c0 - 0.01))                    # 不比現在更靠近未知障礙（下限 0.33：Shield 半徑＋3 cm）
    polys = [np.array([p, p + s_max * d_]) for d_ in dirs]
    if path_pts is not None and len(path_pts) >= 2:                      # 導航路徑折線（目前位置起）＋末段切向再延 0.6 m（Howard 11/20：配合導航實際走出的路徑）
        Pp = np.vstack([p, np.asarray(path_pts, float)[:, :2]])
        tg = Pp[-1] - Pp[-2]; tg = tg / max(float(np.linalg.norm(tg)), 1e-9)
        polys.append(np.vstack([Pp, Pp[-1] + 0.6 * tg]))
    for poly, th in [(P_, th_) for P_ in polys for th_ in ths]:
        cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
        feas = []
        for s_ in np.arange(0.0, min(float(cum[-1]), s_max + 0.6) + 1e-9, ds):
            pp_, d = _poly_at(poly, cum, s_)
            b = np.array([pp_[0], pp_[1], th])
            if not base_clear(b, PATH_BASE_MARGIN) or not scan_free(b, clr):   # Isaac 39：只用 RB 會允許底盤貼桌（實測淨空 0.0997 中止）
                break
            g = ik(np.asarray(T_grasp, float), b)
            feas.append((float(s_), g is not None and clear_ok(b, g[0])[0], g))
        i0 = next((i for i, f in enumerate(feas) if f[1]), None)
        if i0 is None:
            continue
        i1 = i0
        while i1 + 1 < len(feas) and feas[i1 + 1][1]:
            i1 += 1
        s_a, s_b = feas[i0][0], feas[i1][0]
        L = s_b - s_a
        s_g = s_a + min(0.15, L / 3.0)
        dyaw = abs((th - float(th_now) + math.pi) % (2 * math.pi) - math.pi)
        pg_, dg_ = _poly_at(poly, cum, s_g)
        dev = (1.0 - float(np.dot(dg_, dm / nm))) if nm > 1e-3 else 0.0    # 閉爪處切向偏離實際行進方向（0＝同向）
        sc = s_a - 0.5 * L + 0.3 * dyaw + 0.5 * dev
        if best is None or sc < best[0]:
            ig = min(range(i0, i1 + 1), key=lambda i: abs(feas[i][0] - s_g))
            best = (sc, th, s_a, s_b, s_g, feas[ig][2], dg_.copy(), pg_.copy(), poly.copy())
    if best is None:
        return None
    _, th, s_a, s_b, s_g, g, d, pg, poly = best
    v = float(np.clip((s_b - s_g) / t_need, v_min, min(v_cap, max(float(v_now), 0.08))))
    b0l = [float(pg[0]), float(pg[1]), float(th)]
    return dict(b0=b0l, e=[float(d[0]), float(d[1])], v_g=v, s0=-0.02, t_arm=1.0, q_grasp=[float(x) for x in g[0]], no_align=True, path=True,
                poly=[[float(x), float(y)] for x, y in poly], s_feas=[round(s_a, 3), round(s_b, 3)], s_grasp=round(s_g, 3), s_end=round(s_b - s_g, 3))


def seg_free(p0, p1, margin=0.10, ds=0.05):
    """底盤圓（半徑 RB＋margin）沿直線 p0→p1 不碰登錄方盒、也離雷射未知點 ≥SCAN_CLEAR（全向底盤原地轉不改變圓形佔據）。"""
    p0, p1 = np.asarray(p0, float)[:2], np.asarray(p1, float)[:2]
    n = max(2, int(math.ceil(float(np.linalg.norm(p1 - p0)) / ds)) + 1)
    for u in np.linspace(0.0, 1.0, n):
        p = p0 + u * (p1 - p0)
        for cc, s in BOXES:
            dx = max(abs(p[0] - cc[0]) - s[0] / 2, 0.0); dy = max(abs(p[1] - cc[1]) - s[1] / 2, 0.0)
            if math.hypot(dx, dy) < RB + margin:
                return False
        if not scan_free(p):
            return False
    return True


def line_scan_ok(b0, e, s_start, s1):
    """經過線（選線與接近中重核同一式）：[s_start, s1] 離雷射未知點 ≥SCAN_CLEAR；抓後續走 (s1, s1＋LINE_TAIL] ≥SCAN_CLEAR_EXT。"""
    b0 = np.asarray(b0, float)[:2]; e = np.asarray(e, float)[:2]
    if not all(scan_free(b0 + s_ * e) for s_ in np.arange(s_start, s1 + 1e-9, 0.1)):
        return False
    tail = LINE_TAIL if LINE_MERGE else SCAN_TAIL
    return all(scan_free(b0 + s_ * e, SCAN_CLEAR_EXT) for s_ in np.arange(s1, s1 + tail + 1e-9, 0.1))


def scan_free(b, clear=None):
    if SCAN_PTS is None:
        return True
    return float(np.min(np.hypot(SCAN_PTS[:, 0] - b[0], SCAN_PTS[:, 1] - b[1]))) >= (SCAN_CLEAR if clear is None else clear)


def ik(T, b):
    best = None
    for sd in SEEDS:
        q, r, sm = DFK.ik_closure_fc(T, b, sd)
        q = (np.asarray(q, float) + math.pi) % (2 * math.pi) - math.pi   # 角度正規化到 (−π, π]
        if r <= 1e-4 and np.all(q >= LIM.lower + 0.05) and np.all(q <= LIM.upper - 0.05) and sm >= 0.02:
            if best is None or sm > best[2]:
                best = (q, r, sm)
    return best


import door_clear_door_fast as CF                                     # noqa: E402
import door_wbopt as WB                                               # noqa: E402
BASE_MARGIN = float(os.environ.get('CYL_BASE_MARGIN', '0.03'))      # 底盤淨空比執行門檻多留（第十三趟只多 1 mm ⇒ 返回一開始候選尾段被拒）
_T_WH0 = np.eye(4); _T_WH0[:3, 3] = [0.8, -0.04, 0.95]                # 門把（只影響門相關項；這裡門角取開門後 85°）


def clear_ok(b, q):
    """控制器同一份淨空（clear_door_fast；env、base、self 用執行時門檻 WB.THR；門角 85°）。"""
    c = CF.clear_door_fast(np.r_[b, q, math.radians(85.0)], _T_WH0)
    return all(float(c[k]) >= WB.THR[k] + (BASE_MARGIN if k == 'base' else 0.0) for k in ('env', 'base', 'self')), c


def arm_over_table_ok(b, q):
    x9 = np.r_[b, q]
    for n in LINKS:
        p = K.fk(x9, n)[:3, 3]
        if X0 - 0.02 <= p[0] <= X1 + 0.02 and Y0 - 0.02 <= p[1] <= Y1 + 0.02 and p[2] < TOP + 0.05:
            return False
    return True


def candidates(c):
    out = []
    for side in ('west', 'south', 'north', 'east'):                     # r2：東側到沙發 1.35 m 可站（r1 只剩 0.45 m，由 base_free／淨空自然排除）
        for d in D_SIDE:
            for lat in LAT:
                if side == 'west':
                    b = np.array([X0 - d, c[1] + lat, 0.0])
                elif side == 'south':
                    b = np.array([c[0] + lat, Y0 - d, math.pi / 2])
                elif side == 'north':
                    b = np.array([c[0] + lat, Y1 + d, -math.pi / 2])
                else:
                    b = np.array([X1 + d, c[1] + lat, math.pi])
                out.append((side, d, lat, b))
    return out


ROUTE = os.environ.get('PICK_PATH') == '1' and os.environ.get('PICK_ROUTE', '1') == '1'
ROUTE_RES, ROUTE_FURN, ROUTE_SCAN, ROUTE_PAD = 0.05, 0.14, 0.42, 2.5    # 格點、底盤離登錄方盒（中心 ≥0.44）、離雷射未知點、範圍


def route_field(b_now, c_xy):
    """Isaac 39–46：雷射只掃到桌腳 ⇒ Smac 從茶几角與未知箱 unknown_l5 之間 0.73 m 的縫鑽（底盤中心離兩邊都要 ≥0.40 ⇒ 過不去）。
    以登錄方盒（牆、家具，含桌面）＋雷射未知點建格點可行域（中心離方盒 ≥RB＋ROUTE_FURN、離未知點 ≥ROUTE_SCAN），自目前位置做 8 鄰 Dijkstra。
    回 dict(x0, y0, res, W, H, free, dist, start) 或 None。"""
    import heapq
    p0 = np.asarray(b_now, float)[:2]; c = np.asarray(c_xy, float)[:2]
    lo = np.minimum(p0, c) - ROUTE_PAD; hi = np.maximum(p0, c) + ROUTE_PAD
    W = int(math.ceil((hi[0] - lo[0]) / ROUTE_RES)) + 1; H = int(math.ceil((hi[1] - lo[1]) / ROUTE_RES)) + 1
    xs = lo[0] + np.arange(W) * ROUTE_RES; ys = lo[1] + np.arange(H) * ROUTE_RES
    X, Y = np.meshgrid(xs, ys)                                          # [H, W]
    free = np.ones((H, W), bool)
    for cc, sz in BOXES:
        dx = np.maximum(np.abs(X - cc[0]) - sz[0] / 2, 0.0); dy = np.maximum(np.abs(Y - cc[1]) - sz[1] / 2, 0.0)
        free &= np.hypot(dx, dy) >= RB + ROUTE_FURN
    if SCAN_PTS is not None and len(SCAN_PTS):
        for P_ in np.array_split(np.asarray(SCAN_PTS, float), max(1, len(SCAN_PTS) // 64)):
            d2 = (X[..., None] - P_[:, 0]) ** 2 + (Y[..., None] - P_[:, 1]) ** 2
            free &= d2.min(axis=2) >= ROUTE_SCAN ** 2
    i0, j0 = int(round((p0[1] - lo[1]) / ROUTE_RES)), int(round((p0[0] - lo[0]) / ROUTE_RES))
    near0 = np.hypot(X - p0[0], Y - p0[1]) <= 0.20                      # 起點本身可能已在邊距內（剛繞過東西）⇒ 起點附近放行
    free |= near0
    dist = np.full((H, W), np.inf); dist[i0, j0] = 0.0
    pq = [(0.0, i0, j0)]
    nb = [(-1, -1, 1.4142), (-1, 0, 1.0), (-1, 1, 1.4142), (0, -1, 1.0), (0, 1, 1.0), (1, -1, 1.4142), (1, 0, 1.0), (1, 1, 1.4142)]
    while pq:
        d_, i, j = heapq.heappop(pq)
        if d_ > dist[i, j]:
            continue
        for di, dj, w in nb:
            a, b = i + di, j + dj
            if 0 <= a < H and 0 <= b < W and free[a, b]:
                nd = d_ + w * ROUTE_RES
                if nd < dist[a, b]:
                    dist[a, b] = nd; heapq.heappush(pq, (nd, a, b))
    return dict(x0=float(lo[0]), y0=float(lo[1]), res=ROUTE_RES, W=W, H=H, free=free, dist=dist, start=(i0, j0))


def _rf_ij(F, p):
    return int(round((p[1] - F['y0']) / F['res'])), int(round((p[0] - F['x0']) / F['res']))


def route_len(F, p):
    i, j = _rf_ij(F, p)
    if not (0 <= i < F['H'] and 0 <= j < F['W']):
        return float('inf')
    return float(F['dist'][i, j])


def route_via(F, p_goal):
    """沿 Dijkstra 場自站位回溯到起點，再拉直（格點直線全在可行域內就跳過中間點）；回途經點串列（不含起點與站位，map 座標）。"""
    i, j = _rf_ij(F, p_goal)
    if not (0 <= i < F['H'] and 0 <= j < F['W']) or not np.isfinite(F['dist'][i, j]):
        return None
    path = [(i, j)]
    while (i, j) != F['start']:
        best = None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if (di or dj) and 0 <= a < F['H'] and 0 <= b < F['W'] and (best is None or F['dist'][a, b] < F['dist'][best]):
                    best = (a, b)
        if best is None or F['dist'][best] >= F['dist'][i, j]:
            break
        i, j = best; path.append(best)
    path = path[::-1]

    def vis(p, q):
        n = max(abs(q[0] - p[0]), abs(q[1] - p[1])) * 2 + 1
        for u in np.linspace(0.0, 1.0, n):
            a, b = int(round(p[0] + u * (q[0] - p[0]))), int(round(p[1] + u * (q[1] - p[1])))
            if not F['free'][a, b]:
                return False
        return True
    out, k = [], 0
    while k < len(path) - 1:
        m = len(path) - 1
        while m > k + 1 and not vis(path[k], path[m]):
            m -= 1
        out.append(path[m]); k = m
    pts = [np.array([F['x0'] + b * F['res'], F['y0'] + a * F['res']]) for a, b in out[:-1]]
    g_ = np.asarray(p_goal, float)[:2]; res_ = []
    for p in pts:                                                     # 相距 <0.25 m 的連續點只留後一個；站位 0.5 m 內的點不留（最後一段直接去站位）
        if float(np.hypot(*(p - g_))) < 0.5:
            continue
        if res_ and float(np.hypot(*(p - res_[-1]))) < 0.25:
            res_[-1] = p
        else:
            res_.append(p)
    return [[round(float(x), 3) for x in p] for p in res_]


def select(c_xy, b_now, w_dist=1.0, w_sig=0.02):
    """線上抓取底盤位選擇（監督用）：圓柱估計 c_xy（map）、目前底盤 b_now（map）。回最佳候選 dict 或 None（不可達）。
    評分＝底盤直線距離×w_dist＋w_sig／σ_min（同可達性地圖的 IK、限位、σ、家具與桌面淨空條件）。"""
    c = np.asarray(c_xy, float); best = None
    F = route_field(b_now, c) if ROUTE else None                      # 依可走路線（Howard 11/20 原則 1：操作配合實際走得到的路徑）
    for side, d, lat, b in candidates(c):
        if not base_free(b) or not scan_free(b):
            continue
        L_ = route_len(F, b) if F is not None else None
        if L_ is not None and not np.isfinite(L_):
            continue                                                  # 走不到的站位不選
        v = c - b[:2]; dist = float(np.linalg.norm(v))
        if dist < 0.25 or dist > 0.80:
            continue
        a = v / dist
        for fl in (False, True):
            g = ik(grasp_T(c, a, fl), b)
            if g is None or not arm_over_table_ok(b, g[0]) or not clear_ok(b, g[0])[0]:
                continue
            Tp = grasp_T(c, a, fl); Tp[:3, 3] -= PRE * np.r_[a, 0.0]
            gp = ik(Tp, b)
            if gp is None or not clear_ok(b, gp[0])[0]:
                continue
            cost = w_dist * (L_ if L_ is not None else float(np.hypot(*(b[:2] - np.asarray(b_now[:2], float))))) + w_sig / max(float(g[2]), 1e-6)
            if L_ is None and APPROACH_CLEAR is not None and not seg_free(np.asarray(b_now, float), b, margin=APPROACH_CLEAR):
                cost += APPROACH_PEN                                     # Isaac 40：直線到站位要繞桌角 ⇒ 導航貼角、底盤淨空觸 0.10 停 ⇒ 偏好可直接接近的站位
            if best is None or cost < best['cost']:
                best = dict(side=side, d=float(d), lat=float(lat), base=b.tolist(), a=a.tolist(), flip=bool(fl), sigma_min=float(g[2]), cost=cost, q=g[0].tolist(), T_grasp=grasp_T(c, a, fl).tolist(), route_len=L_)
            break
    if best is not None and F is not None:
        best['via'] = route_via(F, np.asarray(best['base'], float)[:2])
    return best


MOTM_S0 = float(os.environ.get('MOTM_S0', '-0.05'))                  # 經過路線核對範圍起點（m，沿桌邊；0＝抓取底盤位）
MOTM_T_WIN = float(os.environ.get('MOTM_T_WIN', '1.9'))              # TCP 世界固定的時間窗（閉爪開始容許 0.5＋閉爪 0.6＋確認 0.3＋證據寬限 0.5 s）
MOTM_DS = 0.05
MOTM_T_ARM = float(os.environ.get('MOTM_T_ARM', '4.0'))              # 手臂自收臂展開到抓取線的最短時間（s73：只剩 2.4 s ⇒ 自身淨空 4.0 mm）
LINE_MERGE = os.environ.get('PICK_LINE', '0') == '1' or os.environ.get('PICK_FLOW', '0') == '1'   # PICK_FLOW（11/21 晚）只借用選位（路線、窗口、車頭候選），底盤照導航                 # 沿桌邊經過線＋邊走邊併線（不原地對齊）；路線起點經格點路線途經點
LINE_DMIN = float(os.environ.get('PICK_LINE_DMIN', '0.46'))          # 經過線底盤中心離桌邊下限
LINE_TAIL = float(os.environ.get('PICK_LINE_TAIL', '0.15'))          # 窗口後沿線續走（抬升 1.2 s＋確認 0.3 s 約 0.15 m，之後交導航）要淨空的長度；0.35 時 l5 全見即無線可選
LINE_W_YAW = float(os.environ.get('PICK_LINE_W_YAW', '0.25'))
LINE_W_TRAV = float(os.environ.get('PICK_LINE_W_TRAV', '0.8'))
LINE_W_REACH = float(os.environ.get('PICK_LINE_W_REACH', '4.0'))     # 舒適代價（Howard 11/24「手臂伸那麼遠不是好事、要順暢舒服」）：手臂基座（車心前 ARM_X）到圓柱水平距離超過 REACH_OK 的每 m 折算
REACH_OK = float(os.environ.get('PICK_REACH_OK', '0.38'))
ARM_X = 0.11                                                          # arm_base_joint xyz 0.11 0 0.28       # 車頭與經過方向夾角代價（m/rad）
HEAD_OFFS = tuple(float(x) for x in os.environ.get('PICK_LINE_HEAD_OFFS', '0,0.3927,0.7854').split(','))   # 車頭自面桌往行進方向轉的候選（rad）        # 偏航代價（m/rad）


def _path_ik(T, b0, e, ss, q0):
    """沿經過路線逐點 IK（以前一點的解當種子，保持同一分支）；任一點不可解、淨空不過或關節跳 >0.3 rad ⇒ None。回 (最小 σ, 最小 env, 最小 base, 各點解)。"""
    qs, sig, env, bc, prev = [], 9.0, 9.0, 9.0, np.asarray(q0, float)
    for s_ in ss:
        b = np.array([b0[0] + s_ * e[0], b0[1] + s_ * e[1], b0[2]])
        if not base_free(b):
            return None
        q, r, sm = DFK.ik_closure_fc(T, b, prev)
        q = (np.asarray(q, float) + math.pi) % (2 * math.pi) - math.pi
        if r > 1e-4 or np.any(q < LIM.lower + 0.05) or np.any(q > LIM.upper - 0.05) or sm < 0.02 or float(np.abs(q - prev).max()) > 0.3:
            return None
        ok, c = clear_ok(b, q)
        if not ok or not arm_over_table_ok(b, q):
            return None
        sig, env, bc = min(sig, float(sm)), min(env, float(c['env'])), min(bc, float(c['base']))
        qs.append(q); prev = q
    return sig, env, bc, qs


SEL_BNB = os.environ.get('PICK_SEL_BNB', '1') == '1'                # 11/24 Howard「為什麼過頭再急轉」：逐一 IK 9.4 s（RTF 0.8 下佔 8.9 s 模擬，結果出來只剩 0.74 m）⇒ 先算與手臂解無關的可行性＋代價下界、依下界排序，下界 ≥ 目前最佳即停（結果與逐一搜尋相同）


def _motm_lb(c, b, e, th, b_now, goal_next, F, Fg, s_start, s1, w_dist):
    """select_motm 中與手臂解（flip、IK、σ）無關的部分：不可行回 None；否則回代價下界（σ 懲罰與併線偏航兩項 ≥0 不計）。各式與 select_motm 內相同。"""
    b_start = np.array([b[0] + s_start * e[0], b[1] + s_start * e[1], th])
    if not base_free(b_start):
        return None
    b_end = b[:2] + s1 * e
    if not line_scan_ok(b, e, s_start, s1):
        return None
    L_ = None
    if LINE_MERGE:
        if not all(base_clear(np.r_[b[:2] + s_ * e, th], PATH_BASE_MARGIN) for s_ in np.arange(s_start, s1 + LINE_TAIL + 1e-9, 0.05)):
            return None
        if F is not None:
            L_ = route_len(F, b_start)
            if not np.isfinite(L_):
                return None
    lb = w_dist * (L_ if L_ is not None else float(np.hypot(*(b_start[:2] - np.asarray(b_now[:2], float)))))
    if goal_next is not None and Fg is not None:
        Lg_ = route_len(Fg, b_end)
        if not np.isfinite(Lg_):
            return None
        lb += w_dist * Lg_
    elif goal_next is not None:
        lb += w_dist * float(np.hypot(*(b_end - np.asarray(goal_next[:2], float))))
    if LINE_MERGE:
        lb += LINE_W_TRAV * abs((th - math.atan2(e[1], e[0]) + math.pi) % (2 * math.pi) - math.pi)
    reach_ = float(np.hypot(c[0] - b[0] - ARM_X * math.cos(th), c[1] - b[1] - ARM_X * math.sin(th)))
    lb += LINE_W_REACH * max(0.0, reach_ - REACH_OK)
    return lb


def select_motm(c_xy, b_now, v_g, goal_next=None, w_dist=1.0, w_sig=0.02, w_dir=0.5):
    """移動中夾取（Howard 11/19「茶几那個要移動中夾取 我要做整體速度最佳」）選位：底盤朝向面桌、沿桌邊以 v_g 橫移經過抓取底盤位 b0。
    同一個世界固定抓取位姿 T_grasp（接近軸＝b0→圓柱），沿路線 s∈[MOTM_S0, v_g·MOTM_T_WIN] 每 5 cm 都要 IK（同分支連續）＋控制器同一份淨空＋不壓桌面；
    s＝0 另核預抓（後退 PRE）。方向 e 兩向都試；評分＝到路線起點距離＋σ 懲罰＋（有下一目標時）e 背離下一目標的懲罰。回 dict 或 None。"""
    c = np.asarray(c_xy, float); best = None
    s1 = float(v_g) * MOTM_T_WIN
    ss = np.round(np.arange(MOTM_S0, s1 + 1e-9, MOTM_DS), 4)
    F = route_field(b_now, c) if (ROUTE and LINE_MERGE) else None    # 併線版
    Fg = route_field(np.asarray(goal_next, float), c) if (ROUTE and LINE_MERGE and goal_next is not None) else None   # 抓完後到下一站（s135d：往西的經過線盡頭被 unknown_l5 擋，直線估計誤選 ⇒ 導航繞回東邊）（Howard 11/21「茶几要真的移動中抓」）：路線起點要走得到（同 select 的格點路線）、途經點交原導航
    jobs = []
    for side, d, lat, b in candidates(c):
        if not base_free(b):
            continue
        if LINE_MERGE and d < LINE_DMIN:
            continue                                                  # 經過段底盤 CBF 邊距 0.42（家具）⇒ 離桌邊太近會一路貼著 CBF 走
        v = c - b[:2]; dist = float(np.linalg.norm(v))
        if dist < 0.25 or dist > 0.80:
            continue
        a = v / dist
        th_f, b_f = float(b[2]), b
        for sg, kh in [(s_, k_) for s_ in (1.0, -1.0) for k_ in (HEAD_OFFS if LINE_MERGE else (0.0,))]:
            e = sg * np.array([-math.sin(th_f), math.cos(th_f)])      # 沿桌邊（垂直於面桌方向）
            sgn_ = 1.0 if ((math.atan2(e[1], e[0]) - th_f + math.pi) % (2 * math.pi) - math.pi) > 0 else -1.0
            th = th_f + sgn_ * kh; b = np.array([b_f[0], b_f[1], th])   # Howard 11/21「抓完倒退走很不自然」：車頭可往行進方向轉 kh（面桌與行進方向之間，斜著往前開經過）
            lb = 0.0
            if SEL_BNB:
                lb = _motm_lb(c, b, e, th, b_now, goal_next, F, Fg, MOTM_S0 - 0.40 - float(v_g) * MOTM_T_ARM, s1, w_dist)
                if lb is None:
                    continue                                          # 與手臂解無關的不可行（原流程在 IK 之後同樣 continue，兩個 flip 都一樣）
            jobs.append((lb, len(jobs), side, d, lat, a, b, e, th))
    jobs.sort(key=lambda j_: (j_[0], j_[1]))                          # 關閉時 lb 全 0 ⇒ 原順序、不剪枝
    for lb, ix, side, d, lat, a, b, e, th in jobs:
        if best is not None and lb >= best['cost']:
            break                                                     # 其餘下界都不小於目前最佳（σ 項 >0 ⇒ 實際代價嚴格大於下界）
        for fl in (False, True):
            Tg = grasp_T(c, a, fl)
            g = ik(Tg, b)
            if g is None:
                continue
            Tp = Tg.copy(); Tp[:3, 3] -= PRE * np.r_[a, 0.0]
            gp = ik(Tp, b)
            if gp is None or not clear_ok(b, gp[0])[0]:
                continue
            fwd = _path_ik(Tg, b, e, [x for x in ss if x >= 0.0], g[0])
            bwd = _path_ik(Tg, b, e, [x for x in ss[::-1] if x <= 0.0], g[0])
            if fwd is None or bwd is None:
                continue
            s_start = MOTM_S0 - 0.40 - float(v_g) * MOTM_T_ARM        # 路線起點（導航目標）：進線加速 0.40 m＋手臂展開時間內的行程
            b_start = np.array([b[0] + s_start * e[0], b[1] + s_start * e[1], th])
            if not base_free(b_start):
                continue
            b_end = b[:2] + s1 * e
            if not line_scan_ok(b, e, s_start, s1):                # 路線起點→經過→終點後都離未知障礙點夠遠（地圖沒有、雷射看得到；接近中重核同一式）
                continue
            via_, L_ = None, None
            if LINE_MERGE:
                if not all(base_clear(np.r_[b[:2] + s_ * e, th], PATH_BASE_MARGIN) for s_ in np.arange(s_start, s1 + LINE_TAIL + 1e-9, 0.05)):
                    continue                                      # 抓到後底盤沿線續走 LINE_TAIL（抬升＋確認時間）再交回導航
                if F is not None:
                    L_ = route_len(F, b_start)
                    if not np.isfinite(L_):
                        continue
            cost = w_dist * (L_ if L_ is not None else float(np.hypot(*(b_start[:2] - np.asarray(b_now[:2], float))))) + w_sig / max(min(fwd[0], bwd[0]), 1e-6)
            if goal_next is not None and Fg is not None:           # 併線版：終點到下一站用格點路線長（走不到 ⇒ 不選）
                Lg_ = route_len(Fg, b_end)
                if not np.isfinite(Lg_):
                    continue
                cost += w_dist * Lg_
            elif goal_next is not None:                              # 整體路程（s75：方向偏好權重太小 ⇒ 選了背離抽屜的方向）：到起點＋終點到下一目標（直線近似）
                cost += w_dist * float(np.hypot(*(b_end - np.asarray(goal_next[:2], float))))
            if LINE_MERGE:                                         # 併線中要轉的偏航（到路線起點的來向≈導航行進方向）：每 rad 折算 LINE_W_YAW m
                if F is not None:
                    via_ = route_via(F, b_start[:2])
                p_in = np.asarray((via_[-1] if via_ else b_now[:2]), float)[:2]
                if float(np.hypot(*(b_start[:2] - p_in))) > 1e-3:
                    h_in = math.atan2(b_start[1] - p_in[1], b_start[0] - p_in[0])
                    cost += LINE_W_YAW * abs((th - h_in + math.pi) % (2 * math.pi) - math.pi)
                cost += LINE_W_TRAV * abs((th - math.atan2(e[1], e[0]) + math.pi) % (2 * math.pi) - math.pi)   # 車頭與行進方向夾角（抓完沿線往前開，不倒退、不橫走）
            reach_ = float(np.hypot(c[0] - b[0] - ARM_X * math.cos(th), c[1] - b[1] - ARM_X * math.sin(th)))
            cost += LINE_W_REACH * max(0.0, reach_ - REACH_OK)
            if best is None or cost < best['cost'] or (cost == best['cost'] and ix < best_ix):   # 同分取原順序在前者（與逐一搜尋一致）
                best_ix = ix
                best = dict(side=side, d=float(d), lat=float(lat), base=b.tolist(), a=a.tolist(), e=e.tolist(), flip=bool(fl), v_g=float(v_g), s_start=float(s_start),
                            base_start=b_start.tolist(), sigma_min=float(min(fwd[0], bwd[0])), env_min=float(min(fwd[1], bwd[1])), base_clear_min=float(min(fwd[2], bwd[2])),
                            s_range=[float(ss[0]), float(ss[-1])], t_arm=MOTM_T_ARM, cost=cost, reach=reach_, q=g[0].tolist(), T_grasp=Tg.tolist(), via=via_, route_len=L_, b_end=b_end.tolist())
            break
    if best is not None and Fg is not None:                             # 抓完後的途經點（終點→下一站；route_via 自 Fg 起點＝下一站回溯 ⇒ 反轉）
        v_ = route_via(Fg, np.asarray(best['b_end'], float)[:2])
        best['via_next'] = None if v_ is None else v_[::-1]
    return best


def main():
    xs = np.round(np.arange(X0 + 0.05, X1 - 0.05 + 1e-9, 0.05), 3)
    ys = np.round(np.arange(Y0 + 0.05, Y1 - 0.05 + 1e-9, 0.05), 3)
    cells = []
    for cx in xs:
        for cy in ys:
            c = np.array([cx, cy]); best = None; n_try = 0
            for side, d, lat, b in candidates(c):
                if not base_free(b):
                    continue
                v = c - b[:2]; dist = float(np.linalg.norm(v))
                if dist < 0.25 or dist > 0.80:
                    continue
                a = v / dist
                n_try += 1
                g, fl = None, False
                for fl in (False, True):
                    g = ik(grasp_T(c, a, fl), b)
                    if g is not None and arm_over_table_ok(b, g[0]) and clear_ok(b, g[0])[0]:
                        break
                    g = None
                if g is None:
                    continue
                Tp = grasp_T(c, a, fl); Tp[:3, 3] -= PRE * np.r_[a, 0.0]
                p = ik(Tp, b)
                if p is None or not clear_ok(b, p[0])[0]:
                    continue
                score = float(g[2])
                if best is None or score > best['sigma_min']:
                    best = dict(side=side, flip=bool(fl), d=float(d), lat=float(lat), base=[round(float(x), 4) for x in b], sigma_min=round(score, 4), q=[round(float(x), 4) for x in g[0]])
            cells.append(dict(c=[float(cx), float(cy)], reachable=best is not None, best=best, n_try=n_try))
            print('%.2f %.2f %s %s' % (cx, cy, 'OK ' if best else '-- ', '' if best is None else '%s d=%.2f lat=%.2f σ=%.3f' % (best['side'], best['d'], best['lat'], best['sigma_min'])), flush=True)
    n_ok = sum(1 for c_ in cells if c_['reachable'])
    summ = dict(table=dict(center=[TX, TY], size=[SX, SY], top=TOP), grid=dict(xs=xs.tolist(), ys=ys.tolist(), step=0.05, margin=0.05),
                grasp=dict(h=H_GRASP, pre=PRE, def_='TCP z＝水平接近方向、TCP y＝水平開合（瓶子線 grasp_pose 定義推廣）'),
                base=dict(sides=['west', 'south', 'north'], d=list(D_SIDE), lat=list(LAT), rb=RB, note='東側到沙發約 0.45 m，不列'),
                reach_rule='抓取＋預抓 IK 殘差 ≤1e-4、LITE6_SAFE 內留 0.05 rad、σ_min ≥ 0.02；底盤圓不碰家具；手臂連桿原點在桌面投影內高度 ≥ 桌面＋0.05；抓取與預抓構型過控制器同一份 clear_door_fast（env／base／self 執行門檻 %s）' % {k: WB.THR[k] for k in ('env', 'base', 'self')},
                n_cells=len(cells), n_reachable=n_ok, frac_reachable=round(n_ok / max(len(cells), 1), 4), cells=cells)
    json.dump(summ, open(OUT + '.json', 'w'), ensure_ascii=False, indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.add_patch(plt.Rectangle((X0, Y0), SX, SY, fill=False, lw=2))
    for c, s in BOXES:
        if abs(c[0] - TX) < 2.5 and abs(c[1] - TY) < 2.5:
            ax.add_patch(plt.Rectangle((c[0] - s[0] / 2, c[1] - s[1] / 2), s[0], s[1], color='0.85', zorder=0))
    col = {'west': 'tab:blue', 'south': 'tab:orange', 'north': 'tab:green', 'east': 'tab:purple'}
    for c_ in cells:
        if c_['reachable']:
            ax.plot(*c_['c'], 'o', color=col[c_['best']['side']], ms=6)
        else:
            ax.plot(*c_['c'], 'x', color='tab:red', ms=6)
    ax.set_xlim(X0 - 0.8, X1 + 0.6); ax.set_ylim(Y0 - 0.8, Y1 + 0.8); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title('reachable %d/%d (o=best side: blue W, orange S, green N; x=unreachable)' % (n_ok, len(cells)), fontsize=9)
    fig.savefig(OUT + '.png', dpi=110, bbox_inches='tight')
    print('reachable', n_ok, '/', len(cells))


if __name__ == '__main__':
    main()
