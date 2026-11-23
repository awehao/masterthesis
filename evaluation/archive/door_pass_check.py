#!/usr/bin/env python3
"""開門後繞門扇與穿門（X3／X4）的離線地圖幾何核對（設計 r1 §3；r2 依 Codex 20261113_010828：收臂後全身包絡、偏航、轉向與停止掃掠、原基準導航參數）。
地圖＝住宅靜態地圖（門洞本來就是空，門扇不入靜態圖）＋開門後門扇（85° 實測＋0.2° 漂移上界，厚 0.04、長 0.95、兩側把手突出 0.04＋0.013 桿半徑＋立柱）＋登錄的未知靜態障礙（5 個；世界有、地圖沒有）。
1 cm 格；距離轉換；A* 找 起點（退出後底盤）→ P1 → P2 的路徑，成本＝長度＋低淨空懲罰；回沿路最小淨空（底盤中心到障礙），與底盤半徑 0.30、膨脹 0.45 比較。
不含行人（動態，由導航即時處理）。
    python3 evaluation/door_pass_check.py EXIT_JSON OUT.json"""
import heapq, json, math, os, sys
import numpy as np
import yaml
from PIL import Image
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); WS = os.path.dirname(HERE)
ex = json.load(open(sys.argv[1]))
reg = json.load(open(os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json')))
m = yaml.safe_load(open(os.path.join(WS, 'src/ammr_bringup/maps/mission_house_m8.yaml')))
im = np.array(Image.open(os.path.join(WS, 'src/ammr_bringup/maps', m['image'].split('/')[-1])))
R0, ORG = float(m['resolution']), m['origin']
RES = 0.01
X0, X1, Y0, Y1 = -3.2, 3.2, -4.4, 2.2                                  # 臥室＋門洞＋客廳前段
nx, ny = int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))
xs = X0 + (np.arange(nx) + 0.5) * RES; ys = Y0 + (np.arange(ny) + 0.5) * RES
XX, YY = np.meshgrid(xs, ys, indexing='ij')
ii = ((XX - ORG[0]) / R0).astype(int); jj = ((YY - ORG[1]) / R0).astype(int)
inb = (ii >= 0) & (ii < im.shape[1]) & (jj >= 0) & (jj < im.shape[0])
occ = np.ones_like(XX, bool)
occ[inb] = im[im.shape[0] - 1 - jj[inb], ii[inb]] < 128
static_occ = occ.copy()


def add_box(cx, cy, lx, ly, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = XX - cx, YY - cy
    u, v = c * dx + s * dy, -s * dx + c * dy
    occ[(np.abs(u) <= lx / 2) & (np.abs(v) <= ly / 2)] = True


phi = math.radians(ex['state']['phi_deg'])
L_, T_ = 0.95, 0.04
for ph in (phi, phi + math.radians(0.2)):                            # 門扇（實測與漂移上界兩個位置都佔）
    th = -ph                                                         # door_q＝−門扇偏航；關門時沿 +x、厚度在 y∈[0,0.04]（牆內側）——開門後繞原點轉
    u = np.array([math.cos(th), math.sin(th)]); n = np.array([-math.sin(th), math.cos(th)])
    ctr = 0.5 * L_ * u + 0.5 * T_ * n
    add_box(ctr[0], ctr[1], L_, T_, th)
    for side in (-1.0, 1.0):                                         # 兩側把手（橫桿＋立柱）包絡：沿 u 0.70–0.90、突出 0.04＋0.013＋0.012
        off = (0.0 if side < 0 else T_) + side * (0.04 + 0.013 + 0.012) / 2
        hc = 0.80 * u + off * n
        add_box(hc[0], hc[1], 0.20, 0.04 + 0.013 + 0.012 + 0.002, th)
ob = json.load(open(os.path.join(WS, reg['obstacles']['path'])))
for o in ob['static_unknown']:
    add_box(o['x'], o['y'], o['size'][0], o['size'][1], math.radians(o['yaw_deg']))
dist = ndimage.distance_transform_edt(~occ) * RES
# 原基準導航（規格/原導航基準_來源對照_r1.md）：代價圖機器人半徑 0.33、膨脹 0.70；GMPC CBF 靜態餘裕 0.38（中心到掃描點）；輪面速度上限 0.2775
R_BASE, R_COST, INFL, CBF_STATIC = 0.30, 0.33, 0.70, 0.38
V_MAX, A_BRAKE = 0.2775, 1.5
D_STOP = V_MAX ** 2 / (2 * A_BRAKE)
BODY_THR = 0.02                                                      # 全身包絡對障礙門檻（同 base 淨空門檻）
MARGIN = CBF_STATIC - R_BASE                                         # A* 中心淨空＝CBF 靜態餘裕
sys.path.insert(0, HERE)
import door_open_exec_ctrl as XO                                     # noqa: E402
import door_insert_ctrl as DI                                        # noqa: E402
Q_STOW9 = np.r_[0.0, 0.0, 0.0, ex['X2']['q_stow']]
_arm = np.vstack([((XO.K.fk(Q_STOW9, n)[:3, :3] @ DI.DOC.CL[n].T).T + XO.K.fk(Q_STOW9, n)[:3, 3])[:, :2] for n in DI.DOC.ARM_LINKS])
_ang = np.linspace(0, 2 * math.pi, 72, endpoint=False)
FOOT = np.vstack([np.c_[R_BASE * np.cos(_ang), R_BASE * np.sin(_ang)], _arm])   # 本體座標 2D 包絡點（車體圓＋收臂手臂投影）


def body_clear(x, y, yaw):
    c, s_ = math.cos(yaw), math.sin(yaw)
    P = np.c_[x + c * FOOT[:, 0] - s_ * FOOT[:, 1], y + s_ * FOOT[:, 0] + c * FOOT[:, 1]]
    ix = np.clip(((P[:, 0] - X0) / RES).astype(int), 0, nx - 1); iy = np.clip(((P[:, 1] - Y0) / RES).astype(int), 0, ny - 1)
    return float(dist[ix, iy].min())


def idx(x, y):
    return int((x - X0) / RES), int((y - Y0) / RES)


def astar(a, b, rmin):
    sa, sb = idx(*a), idx(*b)
    free = dist >= rmin
    if not (free[sa] and free[sb]):
        return None, dict(start_clear=float(dist[sa]), goal_clear=float(dist[sb]))
    pen = np.where(dist < INFL, (INFL - dist) * 40.0, 0.0)
    g = {sa: 0.0}; prev = {}; pq = [(0.0, sa)]
    nb = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    while pq:
        f, u = heapq.heappop(pq)
        if u == sb:
            break
        for dx, dy, w in nb:
            v = (u[0] + dx, u[1] + dy)
            if not (0 <= v[0] < nx and 0 <= v[1] < ny) or not free[v]:
                continue
            ng = g[u] + w * (1.0 + pen[v])
            if ng < g.get(v, 1e18):
                g[v] = ng; prev[v] = u
                heapq.heappush(pq, (ng + math.hypot(v[0] - sb[0], v[1] - sb[1]), v))
    if sb not in prev and sb != sa:
        return None, {}
    path = [sb]
    while path[-1] != sa:
        path.append(prev[path[-1]])
    return path[::-1], {}


b_exit = ex['X1b']['base_end'][:2]
P1, P2 = (0.55, -1.55), (0.55, 1.0)
yaw_exit = float(ex['X1b']['base_end'][2])
out = dict(start=b_exit, start_yaw_deg=round(math.degrees(yaw_exit), 2), P1=P1, P2=P2, nav=dict(R_BASE=R_BASE, R_COST=R_COST, INFL=INFL, CBF_STATIC=CBF_STATIC, V_MAX=V_MAX, A_BRAKE=A_BRAKE, D_STOP=round(D_STOP, 4)),
           footprint=dict(max_radius=round(float(np.hypot(FOOT[:, 0], FOOT[:, 1]).max()), 3), note='車體圓 0.30＋收臂手臂碰撞點投影（夾爪向本體 +x 突出到 0.36）；障礙一律當全高'), door_deg=ex['state']['phi_deg'])
segs = []
for a, b, nm in ((b_exit, P1, 'X3 繞門扇'), (P1, P2, 'X4 穿門')):
    path, info = astar(a, b, CBF_STATIC)
    if path is None:
        segs.append(dict(seg=nm, found=False, **info)); continue
    cl = np.array([dist[p] for p in path])
    pts = np.array([(X0 + (p[0] + 0.5) * RES, Y0 + (p[1] + 0.5) * RES) for p in path])
    k = int(np.argmin(cl))
    tang = np.unwrap(np.arctan2(np.gradient(ndimage.uniform_filter1d(pts[:, 1], 15)), np.gradient(ndimage.uniform_filter1d(pts[:, 0], 15))))
    bc_tan = np.array([body_clear(p[0], p[1], th_) for p, th_ in zip(pts, tang)])
    bc_90 = np.array([body_clear(p[0], p[1], math.pi / 2) for p in pts])
    segs.append(dict(seg=nm, found=True, length_m=round(float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))), 3), min_center_clear_m=round(float(cl.min()), 3), at=pts[k].round(3).tolist(),
                     frac_within_infl=round(float(np.mean(cl < INFL)), 3), center_clear_minus_Rcost=round(float(cl.min() - R_COST), 3),
                     body_clear_tangent_yaw=dict(min=round(float(bc_tan.min()), 3), at=pts[int(np.argmin(bc_tan))].round(3).tolist(), ok_with_stop=bool(bc_tan.min() >= BODY_THR + D_STOP)),
                     body_clear_fixed_90=dict(min=round(float(bc_90.min()), 3), at=pts[int(np.argmin(bc_90))].round(3).tolist(), ok_with_stop=bool(bc_90.min() >= BODY_THR + D_STOP)),
                     path_xy=pts[::10].round(3).tolist()))
out['segments'] = segs
# 原地轉向掃掠：退出點（起始偏航 → 90°、→ X3 起始切線）與 P1（→ 90°）
def turn_sweep(x, y, y0, y1):
    ys_ = np.linspace(y0, y1, max(2, int(abs(y1 - y0) / math.radians(2)) + 1))
    v = [body_clear(x, y, yy) for yy in ys_]
    return dict(min=round(float(min(v)), 3), from_deg=round(math.degrees(y0), 1), to_deg=round(math.degrees(y1), 1), ok=bool(min(v) >= BODY_THR))
out['turn_sweeps'] = dict(exit_to_90=turn_sweep(b_exit[0], b_exit[1], yaw_exit, math.pi / 2), at_P1_to_90=turn_sweep(P1[0], P1[1], -math.pi / 2, math.pi / 2),
                          exit_full_circle=turn_sweep(b_exit[0], b_exit[1], 0.0, 2 * math.pi))
# 門洞橫截面：y＝0.06（牆中線）沿 x 的淨空，與 y＝−0.3（門洞前）
for yy in (0.06, -0.3):
    j = int((yy - Y0) / RES)
    row = dist[:, j]
    xs_free = xs[row >= R_BASE]
    sel = xs_free[(xs_free > -0.5) & (xs_free < 1.5)]
    out[f'doorway_y{yy:+.2f}'] = dict(center_x_range_for_R0p30=[round(float(sel.min()), 3), round(float(sel.max()), 3)] if len(sel) else None,
                                      max_clear_m=round(float(row[(xs > -0.2) & (xs < 1.2)].max()), 3))
out['note'] = '離線幾何／運動學存在性；門扇取實測與 +0.2° 兩位置；未知靜態障礙用登錄真值加入——只用於幾何存在性，不得當成導航成果（導航能否規劃、避障、通過留給原全開 ROS 鏈驗證）；行人未含'
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
for s_ in segs:
    print({k: v for k, v in s_.items() if k != 'path_xy'})
print(out['footprint'], out['turn_sweeps'])
print({k: v for k, v in out.items() if k.startswith('doorway')})
