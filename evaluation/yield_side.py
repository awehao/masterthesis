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
PAD = float(os.environ.get('YS_PAD', '0.25'))
WALL_PAD = float(os.environ.get('YS_WALL_PAD', '0.14'))           # 目標點與側移路徑：底盤中心離地圖障礙 ≥0.30＋0.14（控制器底盤淨空門檻 0.10＋餘裕；Isaac 37 回程側移經床頭櫃 0.0954 被停）
RATE = float(os.environ.get('YS_RATE', '0.25'))
R_BASE = 0.30
ALONG_MAX_S = float(os.environ.get('YS_ALONG_MAX_S', '10.0'))         # 同向並行讓行的安全時限（s）
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
    pts = [o for o in (obs or []) if not o.get('static')]           # s100d：新出現的軌跡速度由 0 慢慢收斂，不先用速度門檻濾掉
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


def _interval(gp, n, q_xy=None):
    """行人在法向 n 上的側向區間：中心＝可見表面點平均往遠離機器人方向推一個半徑（表面點偏向感測器），半寬＝R_MIN
    （s98d：直接用點的範圍＋背面補寬，雜點讓區間寬到 1.0 m、中心偏 0.23 m ⇒ 選到要穿越行人路線的一側）。"""
    m = gp['P'].mean(axis=0)
    if q_xy is not None:
        d = m - np.asarray(q_xy, float)[:2]; nd = float(np.linalg.norm(d))
        if nd > 1e-6:
            m = m + R_MIN * d / nd
    sc = float(m @ n)
    return sc - R_MIN, sc + R_MIN


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
            if sp >= V_MIN:
                u = vo / sp
            else:                                                     # 速度尚未收斂（剛出遮蔽）或靜止的未知物：以相對運動方向當行進線，仍要讓開
                vr_ = vo - v_b; nr_ = float(np.linalg.norm(vr_))
                if nr_ < V_MIN or float(np.min(np.linalg.norm(gp['P'] - q[:2], axis=1))) > 2.0:
                    continue
                u = vr_ / nr_
            n = np.array([-u[1], u[0]])
            lo, hi = _interval(gp, n, q[:2])
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

    def _alongside(self, q, v_b, obs):
        """同向並行：行人在動、機器人朝同方向走、目前距離 < 安全距＋0.2 ⇒ 回 (u, n, 中心, 距離, need)。"""
        best = None
        for gp in _groups(obs):
            vo = gp['v']; sp = float(np.linalg.norm(vo))
            if sp < V_MIN or float(np.dot(v_b, vo / sp)) < 0.05:
                continue
            u = vo / sp; n = np.array([-u[1], u[0]])
            lo, hi = _interval(gp, n, q[:2])
            c = float((gp['P'] @ u).mean()) * u + 0.5 * (lo + hi) * n
            need = R_BASE + 0.5 * (hi - lo) + PAD
            dist = float(np.linalg.norm(c - q[:2]))
            lat = abs(float((q[:2] - c) @ n)); alg = abs(float((q[:2] - c) @ u))
            if lat < need and alg < need + 0.5 and (best is None or dist < best[3]):   # s115d：只在真的落在行人帶內才算（側向 0.9 m 外也鎖 ⇒ 卡 130 s）
                best = (u, n, c, dist, need)
        return best

    def _apply_along(self, t, q, X, al):
        """讓行人先走：沿行人方向不再前進（參考沿 u 的進度釘在目前位置），側向離開行人帶到目前所在側；行人領先 > need＋0.3 才放行（之後從它後面過）。"""
        if al is not None:
            u, n, c, dist, need = al
            if getattr(self, 'along', None) is None:
                sg = 1.0 if float((q[:2] - c) @ n) >= 0.0 else -1.0
                self.along = dict(u=u, n=n, sg=sg, need=need)
                self.log.append(dict(t=round(t, 2), what='yield_along', dist=round(dist, 3), side=sg))
            self.along.update(c=c)
        L = self.along
        L.setdefault('t0', t)
        lat_ = abs(float((q[:2] - L['c']) @ L['n']))
        if lat_ >= L['need'] or float((L['c'] - q[:2]) @ L['u']) > L['need'] + 0.8 or t - L['t0'] > ALONG_MAX_S:   # 已離開行人帶／行人領先夠遠（從後方過）／安全時限 ⇒ 放行
            self.log.append(dict(t=round(t, 2), what='yield_along_release'))
            self.along = None
            return dict(off=0.0)
        u, n, sg, c = L['u'], L['n'], L['sg'], L['c']
        s_c = float(c @ n)
        for Xk in X:
            p = np.array([Xk[0, 2], Xk[1, 2]])
            a_ = float((p - q[:2]) @ u)
            if a_ > 0.0:
                p = p - a_ * u                                        # 不沿行人方向前進
            s_k = float(p @ n)
            if sg * (s_k - s_c) < L['need']:
                p = p + (sg * L['need'] - (s_k - s_c)) * n            # 側向至少離行人帶 need（目前所在側）
            Xk[0, 2], Xk[1, 2] = float(p[0]), float(p[1])
        return dict(off=0.0, along=True)

    def apply(self, t, q, v_b, X, obs, dt):
        """X：參考視窗（3×3 SE2 串列，就地修改）。側向座標＝點與 n 的內積（絕對）；目標＝保守區間外 R_BASE＋PAD。"""
        if not ENABLE:
            return None
        q = np.asarray(q, float); v_b = np.asarray(v_b, float)[:2]
        al = self._alongside(q, v_b, obs)
        if al is not None or getattr(self, 'along', None) is not None:   # 同向並行（s113d：相對速度≈0 ⇒ 最近接近預測不觸發，被行人帶著走 30 s）
            return self._apply_along(t, q, X, al)
        c = self._conflict(q, v_b, obs, any_ds=self.lock is not None)
        if c is not None and self.lock is None:
            gp, ts, ds, need, n, (lo, hi) = c
            s_now = float(q[:2] @ n)
            cands = []
            for sg in (1.0, -1.0):
                s_t = (hi + R_BASE + PAD) if sg > 0 else (lo - R_BASE - PAD)
                tgt = q[:2] + (s_t - s_now) * n
                cl = clearance(*tgt)
                if cl >= R_BASE + WALL_PAD and _path_clear(q[:2], tgt, R_BASE + WALL_PAD):
                    same = (s_now - 0.5 * (lo + hi)) * sg > 0.0              # 機器人目前已在該側（不必穿越行人路線）
                    cands.append((0 if same else 1, abs(s_t - s_now), sg, s_t, cl))
            if cands:
                cands.sort()
                _, _, sg, s_t, cl = cands[0]
                self.lock = dict(n=n, sg=sg, s_t=s_t, t0=t)
                self.off = 0.0
                self.log.append(dict(t=round(t, 2), what='yield_lock', side=sg, s_now=round(s_now, 3), s_tgt=round(s_t, 3), lo=round(lo, 3), hi=round(hi, 3), ts=round(ts, 2), ds=round(ds, 3), n_cand=len(cands), clear=round(cl, 2)))
            else:
                self.log.append(dict(t=round(t, 2), what='yield_no_side', s_now=round(s_now, 3), lo=round(lo, 3), hi=round(hi, 3), ts=round(ts, 2), ds=round(ds, 3)))
                if abs(s_now - 0.5 * (lo + hi)) >= 0.5 * (hi - lo) + R_BASE + 0.05:   # 兩側都沒淨空、但機器人還不在行人帶上 ⇒ 原地等行人通過（參考視窗釘在目前位置）
                    self.lock = dict(n=n, sg=0.0, s_t=s_now, t0=t, hold=np.asarray(q[:3], float).copy())
                    self.log.append(dict(t=round(t, 2), what='yield_hold', s_now=round(s_now, 3)))
        if self.lock is not None and self.lock.get('hold') is not None:
            if c is None:
                self.log.append(dict(t=round(t, 2), what='yield_release'))
                self.lock = None
                return dict(off=0.0)
            h_ = self.lock['hold']
            for Xk in X:                                              # 等待：位置釘住、朝向照參考
                Xk[0, 2], Xk[1, 2] = float(h_[0]), float(h_[1])
            return dict(off=0.0, side=0.0, hold=True)
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
                if abs(float(n @ L['n'])) < math.cos(math.radians(30.0)):   # s101d：行進線估計改變（速度收斂、相對運動→行人速度）⇒ 依新行進線重新選邊
                    self.log.append(dict(t=round(t, 2), what='yield_relock'))
                    self.lock = None
                    return self.apply(t, q, v_b, X, obs, dt)
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
