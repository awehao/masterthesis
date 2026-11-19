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


def base_free(b):
    for c, s in BOXES:
        dx = max(abs(b[0] - c[0]) - s[0] / 2, 0.0); dy = max(abs(b[1] - c[1]) - s[1] / 2, 0.0)
        if math.hypot(dx, dy) < RB:
            return False
    return True


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


def select(c_xy, b_now, w_dist=1.0, w_sig=0.02):
    """線上抓取底盤位選擇（監督用）：圓柱估計 c_xy（map）、目前底盤 b_now（map）。回最佳候選 dict 或 None（不可達）。
    評分＝底盤直線距離×w_dist＋w_sig／σ_min（同可達性地圖的 IK、限位、σ、家具與桌面淨空條件）。"""
    c = np.asarray(c_xy, float); best = None
    for side, d, lat, b in candidates(c):
        if not base_free(b):
            continue
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
            cost = w_dist * float(np.hypot(*(b[:2] - np.asarray(b_now[:2], float)))) + w_sig / max(float(g[2]), 1e-6)
            if best is None or cost < best['cost']:
                best = dict(side=side, d=float(d), lat=float(lat), base=b.tolist(), a=a.tolist(), flip=bool(fl), sigma_min=float(g[2]), cost=cost, q=g[0].tolist(), T_grasp=grasp_T(c, a, fl).tolist())
            break
    return best


MOTM_S0 = float(os.environ.get('MOTM_S0', '-0.05'))                  # 經過路線核對範圍起點（m，沿桌邊；0＝抓取底盤位）
MOTM_T_WIN = float(os.environ.get('MOTM_T_WIN', '1.9'))              # TCP 世界固定的時間窗（閉爪開始容許 0.5＋閉爪 0.6＋確認 0.3＋證據寬限 0.5 s）
MOTM_DS = 0.05
MOTM_T_ARM = float(os.environ.get('MOTM_T_ARM', '4.0'))              # 手臂自收臂展開到抓取線的最短時間（s73：只剩 2.4 s ⇒ 自身淨空 4.0 mm）


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


def select_motm(c_xy, b_now, v_g, goal_next=None, w_dist=1.0, w_sig=0.02, w_dir=0.5):
    """移動中夾取（Howard 11/19「茶几那個要移動中夾取 我要做整體速度最佳」）選位：底盤朝向面桌、沿桌邊以 v_g 橫移經過抓取底盤位 b0。
    同一個世界固定抓取位姿 T_grasp（接近軸＝b0→圓柱），沿路線 s∈[MOTM_S0, v_g·MOTM_T_WIN] 每 5 cm 都要 IK（同分支連續）＋控制器同一份淨空＋不壓桌面；
    s＝0 另核預抓（後退 PRE）。方向 e 兩向都試；評分＝到路線起點距離＋σ 懲罰＋（有下一目標時）e 背離下一目標的懲罰。回 dict 或 None。"""
    c = np.asarray(c_xy, float); best = None
    s1 = float(v_g) * MOTM_T_WIN
    ss = np.round(np.arange(MOTM_S0, s1 + 1e-9, MOTM_DS), 4)
    for side, d, lat, b in candidates(c):
        if not base_free(b):
            continue
        v = c - b[:2]; dist = float(np.linalg.norm(v))
        if dist < 0.25 or dist > 0.80:
            continue
        a = v / dist
        th = float(b[2])
        for sg in (1.0, -1.0):
            e = sg * np.array([-math.sin(th), math.cos(th)])          # 沿桌邊（垂直於面桌方向）
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
                cost = w_dist * float(np.hypot(*(b_start[:2] - np.asarray(b_now[:2], float)))) + w_sig / max(min(fwd[0], bwd[0]), 1e-6)
                if goal_next is not None:
                    gn = np.asarray(goal_next[:2], float) - b[:2]
                    cost += w_dir * (1.0 - float(np.dot(gn / max(float(np.linalg.norm(gn)), 1e-9), e)))
                if best is None or cost < best['cost']:
                    best = dict(side=side, d=float(d), lat=float(lat), base=b.tolist(), a=a.tolist(), e=e.tolist(), flip=bool(fl), v_g=float(v_g), s_start=float(s_start),
                                base_start=b_start.tolist(), sigma_min=float(min(fwd[0], bwd[0])), env_min=float(min(fwd[1], bwd[1])), base_clear_min=float(min(fwd[2], bwd[2])),
                                s_range=[float(ss[0]), float(ss[-1])], t_arm=MOTM_T_ARM, cost=cost, q=g[0].tolist(), T_grasp=Tg.tolist())
                break
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
