#!/usr/bin/env python3
"""動態障礙物會車選邊（s91d／s92d：行人 walker_r1 沿 y＝9.0 正面迎來，W-GMPC 的 CBF 線性化把底盤推向已在的一側＝牆側，
橫向最多 0.47 m＜需要的 0.55 m ⇒ 被行人沿路推著倒退 7 s 後接觸）。

做法（只改 W-GMPC 的導航參考視窗；不改 CBF、Shield、原導航參數；行人不讓路、不暫停）：
  1. 觸發：非靜態障礙物，速度 ≥ V_MIN；以底盤目前速度與障礙物等速預測最近接近時刻 t*∈(0, T_LOOK)、
     最近距離 d* ＜ R_NEED＝底盤半徑＋障礙物半徑＋PAD。
  2. 選邊：障礙物運動方向的法向 n，候選側向位置 s'＝±R_NEED（相對障礙物軌跡線），
     在地圖距離場上檢查「底盤平移到該側」的目標點與路徑淨空 ≥ 底盤半徑＋WALL_PAD；可行者取側移量小者。
  3. 鎖邊：選定後保持到障礙物已通過（t*≤0 且相對距離增加）或離開 T_LOOK；避免左右切換。
  4. 套用：參考視窗各點沿 n 平移，使其相對軌跡線的側向座標至少達 s'（同號）；平移量以 RATE（m/s）逐拍爬升。
與原導航 committed detour（detour.py，預設關、Howard 判定效果差：偏移率 0.08 m/s、不看地圖選邊）不同處：依地圖淨空選邊、偏移率 0.25 m/s。
座標＝里程（替身／Isaac 里程與地圖原點一致；地圖只用來看牆，誤差由 WALL_PAD 吸收）。"""
import math
import os

import numpy as np

ENABLE = os.environ.get('D2_YIELD_SIDE', '0') == '1'
V_MIN = float(os.environ.get('YS_V_MIN', '0.08'))
T_LOOK = float(os.environ.get('YS_T_LOOK', '5.0'))
PAD = float(os.environ.get('YS_PAD', '0.15'))
WALL_PAD = float(os.environ.get('YS_WALL_PAD', '0.10'))
RATE = float(os.environ.get('YS_RATE', '0.25'))
R_BASE = 0.30
R_MIN = float(os.environ.get('YS_R_MIN', '0.25'))                     # 行人半徑下限（追蹤器回報的半徑偏小，s94d need 只有 0.45＜真實 0.55）

_MAP = None


def _load_map():
    global _MAP
    if _MAP is not None:
        return _MAP
    name = os.environ.get('M8_MAP')
    _MAP = False
    if not name:
        return _MAP
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(here, '..', 'install', 'ammr_bringup', 'share', 'ammr_bringup', 'maps'), os.path.join(here, '..', 'src', 'ammr_bringup', 'maps')):
        yp = os.path.join(d, name + '.yaml')
        if os.path.exists(yp):
            import yaml
            from PIL import Image
            from scipy.ndimage import distance_transform_edt
            y = yaml.safe_load(open(yp))
            im = np.array(Image.open(os.path.join(d, y['image'])))
            free = im >= 100                                          # 佔據＝暗（同 occupied_thresh 0.65 的保守判讀）
            res = float(y['resolution'])
            _MAP = dict(dist=distance_transform_edt(free) * res, res=res, ox=float(y['origin'][0]), oy=float(y['origin'][1]), H=im.shape[0], W=im.shape[1])
            break
    return _MAP


def clearance(x, y):
    m = _load_map()
    if not m:
        return float('inf')
    c = int((x - m['ox']) / m['res']); r = m['H'] - 1 - int((y - m['oy']) / m['res'])
    if not (0 <= r < m['H'] and 0 <= c < m['W']):
        return 0.0
    return float(m['dist'][r, c])


def _path_clear(p0, p1, need, n=8):
    return all(clearance(*(p0 + u * (p1 - p0))) >= need for u in np.linspace(0.0, 1.0, n))


def _groups(obs):
    """把追蹤器給 CBF 的可見表面點（每點帶同一軌跡速度、半徑≈0）依距離＜0.5 m 且速度差＜0.15 m/s 併成一個行人。"""
    pts = [o for o in (obs or []) if not o.get('static') and math.hypot(float(o.get('vx', 0.0)), float(o.get('vy', 0.0))) >= V_MIN]
    lab = list(range(len(pts)))
    def f(i):
        while lab[i] != i:
            lab[i] = lab[lab[i]]; i = lab[i]
        return i
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            a, b = pts[i], pts[j]
            if math.hypot(float(a['x']) - float(b['x']), float(a['y']) - float(b['y'])) < 0.5 and \
               math.hypot(float(a.get('vx', 0.0)) - float(b.get('vx', 0.0)), float(a.get('vy', 0.0)) - float(b.get('vy', 0.0))) < 0.15:
                lab[f(i)] = f(j)
    g = {}
    for i, o in enumerate(pts):
        g.setdefault(f(i), []).append(o)
    out = []
    for ps in g.values():
        P_ = np.array([[float(o['x']), float(o['y'])] for o in ps]); V_ = np.array([[float(o.get('vx', 0.0)), float(o.get('vy', 0.0))] for o in ps])
        r_ = max(float(o.get('radius', 0.0)) for o in ps)
        out.append(dict(P=P_, v=V_.mean(axis=0), r=r_))
    return out


def _interval(gp, n):
    """行人在法向 n 上的保守側向區間：可見表面點範圍＋背面未見部分（寬度至少 2·R_MIN）。"""
    s = gp['P'] @ n
    lo, hi = float(s.min()) - gp['r'], float(s.max()) + gp['r']
    return min(lo, hi - 2.0 * R_MIN), max(hi, lo + 2.0 * R_MIN)


class YieldSide:
    def __init__(self):
        self.lock = None                                              # dict(name_key, n, s_tgt)
        self.off = 0.0                                                # 目前已套用的側移量（m，沿 lock['n']）
        self.log = []

    def _conflict(self, q, v_b, obs, any_ds=False):
        """回 (群, t*, d*, need, n, (lo, hi)) 或 None。以群的保守區間中心當圓心、半寬當半徑做最近接近預測。"""
        best = None
        for gp in _groups(obs):
            vo = gp['v']; sp = float(np.linalg.norm(vo))
            if sp < V_MIN:
                continue
            u = vo / sp; n = np.array([-u[1], u[0]])
            lo, hi = _interval(gp, n)
            al = float((gp['P'] @ u).mean())
            c = al * u + 0.5 * (lo + hi) * n
            rad = 0.5 * (hi - lo)
            p = c - q[:2]; vr = vo - v_b; vv = float(vr @ vr)
            if vv < 1e-6:
                continue
            ts = -float(p @ vr) / vv
            if not (0.0 < ts < T_LOOK):
                continue
            ds = float(np.linalg.norm(p + vr * ts))
            need = R_BASE + rad + PAD
            if (ds < need or (any_ds and ds < need + 0.6)) and (best is None or ts < best[1]):
                best = (gp, ts, ds, need, n, (lo, hi))
        return best

    def apply(self, t, q, v_b, X, obs, dt):
        """X：參考視窗（3×3 SE2 串列，就地修改）。側向座標＝點與 n 的內積（絕對）；目標＝保守區間外 R_BASE＋PAD。"""
        if not ENABLE:
            return None
        q = np.asarray(q, float); v_b = np.asarray(v_b, float)[:2]
        c = self._conflict(q, v_b, obs, any_ds=self.lock is not None)
        if c is not None and self.lock is None:
            gp, ts, ds, need, n, (lo, hi) = c
            s_now = float(q[:2] @ n)
            cands = []
            for sg in (1.0, -1.0):
                s_t = (hi + R_BASE + PAD) if sg > 0 else (lo - R_BASE - PAD)
                tgt = q[:2] + (s_t - s_now) * n
                cl = clearance(*tgt)
                if cl >= R_BASE + WALL_PAD and _path_clear(q[:2], tgt, R_BASE + WALL_PAD * 0.5):
                    cands.append((abs(s_t - s_now) - 0.2 * min(cl, 1.0), sg, s_t, cl))
            if cands:
                cands.sort()
                _, sg, s_t, cl = cands[0]
                self.lock = dict(n=n, sg=sg, s_t=s_t, t0=t)
                self.off = 0.0
                self.log.append(dict(t=round(t, 2), what='yield_lock', side=sg, s_now=round(s_now, 3), s_tgt=round(s_t, 3), lo=round(lo, 3), hi=round(hi, 3), ts=round(ts, 2), ds=round(ds, 3), n_cand=len(cands), clear=round(cl, 2)))
            else:
                self.log.append(dict(t=round(t, 2), what='yield_no_side', s_now=round(s_now, 3), lo=round(lo, 3), hi=round(hi, 3), ts=round(ts, 2), ds=round(ds, 3)))
        if self.lock is not None:
            L = self.lock
            if c is None:                                            # 已無衝突（通過或離開預測窗）⇒ 平移量以同速率收回
                self.off = max(0.0, self.off - RATE * dt)
                if self.off <= 1e-4:
                    self.log.append(dict(t=round(t, 2), what='yield_release'))
                    self.lock = None
                    return dict(off=0.0)
            else:
                gp, ts, ds, need, n, (lo, hi) = c
                if float(n @ L['n']) < 0.0:                          # 法向翻轉（速度方向估計反向）⇒ 換到同一參考
                    n, lo, hi = -n, -hi, -lo
                L['s_t'] = (hi + R_BASE + PAD) if L['sg'] > 0 else (lo - R_BASE - PAD)
                L['n'] = n
                self.off = min(self.off + RATE * dt, 1.5)
            n, sg = L['n'], L['sg']
            shift = 0.0
            for Xk in X:                                              # 各參考點側向座標至少達 s_t（鎖定側），平移量上限＝爬升量 off
                s_k = float(np.array([Xk[0, 2], Xk[1, 2]]) @ n)
                d_k = min(max(0.0, sg * (L['s_t'] - s_k)), self.off)
                Xk[0, 2] += sg * d_k * n[0]
                Xk[1, 2] += sg * d_k * n[1]
                shift = max(shift, d_k)
            return dict(off=round(shift, 3), side=sg)
        return None
        q = np.asarray(q, float); v_b = np.asarray(v_b, float)[:2]
        c = self._conflict(q, v_b, obs, any_ds=self.lock is not None)
        if c is not None and self.lock is None:
            o, ts, ds, need, vo = c
            u = vo / float(np.linalg.norm(vo)); n = np.array([-u[1], u[0]])
            o_xy = np.array([float(o['x']), float(o['y'])])
            s_now = float((q[:2] - o_xy) @ n)
            cands = []
            for sg in (1.0, -1.0):
                s_t = sg * need
                tgt = q[:2] + (s_t - s_now) * n
                if clearance(*tgt) >= R_BASE + WALL_PAD and _path_clear(q[:2], tgt, R_BASE + WALL_PAD * 0.5):
                    cands.append((abs(s_t - s_now), sg, s_t))
            if cands:
                cands.sort()
                _, sg, s_t = cands[0]
                self.lock = dict(n=n, u=u, o_xy=o_xy, s_tgt=s_t, sg=sg, t0=t, need=need)
                self.off = 0.0
                self.log.append(dict(t=round(t, 2), what='yield_lock', side=sg, s_now=round(s_now, 3), s_tgt=round(s_t, 3), ts=round(ts, 2), ds=round(ds, 3), n_cand=len(cands)))
            else:
                self.log.append(dict(t=round(t, 2), what='yield_no_side', s_now=round(s_now, 3), ts=round(ts, 2), ds=round(ds, 3)))
        if self.lock is not None:
            L = self.lock
            if c is None:                                            # 已無衝突（通過或離開預測窗）⇒ 平移量以同速率收回
                self.off = max(0.0, self.off - RATE * dt)
                if self.off <= 1e-4:
                    self.log.append(dict(t=round(t, 2), what='yield_release'))
                    self.lock = None
                    return dict(off=0.0)
            else:
                o, ts, ds, need, vo = c
                L['o_xy'] = np.array([float(o['x']), float(o['y'])])
                self.off = min(self.off + RATE * dt, 1.5)
            n, sg = L['n'], L['sg']
            shift = 0.0
            for Xk in X:                                              # 各參考點相對行人軌跡線的側向座標至少達 need（鎖定側），平移量上限＝爬升量 off
                s_k = float((np.array([Xk[0, 2], Xk[1, 2]]) - L['o_xy']) @ n)
                d_k = min(max(0.0, L['need'] - sg * s_k), self.off)
                Xk[0, 2] += sg * d_k * n[0]
                Xk[1, 2] += sg * d_k * n[1]
                shift = max(shift, d_k)
            return dict(off=round(shift, 3), side=sg)
        return None
