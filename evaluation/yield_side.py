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
import time as _time

import numpy as np

ENABLE = os.environ.get('D2_YIELD_SIDE', '0') == '1'
V_MIN = float(os.environ.get('YS_V_MIN', '0.08'))
T_LOOK = float(os.environ.get('YS_T_LOOK', '5.0'))
PAD = float(os.environ.get('YS_PAD', '0.25'))
WALL_PAD = float(os.environ.get('YS_WALL_PAD', '0.14'))           # 目標點與側移路徑：底盤中心離地圖障礙 ≥0.30＋0.14（控制器底盤淨空門檻 0.10＋餘裕；Isaac 37 回程側移經床頭櫃 0.0954 被停）
RATE = float(os.environ.get('YS_RATE', '0.25'))
R_BASE = 0.30
ALONG_MAX_S = float(os.environ.get('YS_ALONG_MAX_S', '6.0'))         # 同向並行讓行的安全時限（s）
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


NO_BACK = os.environ.get('YS_NO_BACK', '0') == '1'                  # s154d：行人從側邊直衝時往後讓正是唯一閃避方向 ⇒ 預設關
ALONG_PAD = float(os.environ.get('YS_ALONG_PAD', '0.10'))
ALONG_COS = math.cos(math.radians(float(__import__('os').environ.get('YS_ALONG_DEG', '30'))))


CROSS = os.environ.get('YS_CROSS', '1') == '1'                       # Isaac 84：橫越行人（walker_l2 向西）⇒ 側移選邊選到行人去的方向、進入行人帶並行被追撞 ⇒ 橫越改「路徑上停在行人帶外等它過」
CROSS_COS = math.cos(math.radians(float(os.environ.get('YS_CROSS_DEG', '45'))))   # 行人方向與參考走向夾角 > 45° 才算橫越
CROSS_MARGIN_S = float(os.environ.get('YS_CROSS_MARGIN_S', '1.0'))  # 先過／等待判定的時間餘裕
CROSS_V = float(os.environ.get('YS_CROSS_V', '0.22'))                # 機器人通過行人帶的名目速度（導航名目）
CROSS_LOOK = float(os.environ.get('YS_CROSS_LOOK', '8.0'))
CROSS_BEHIND_PAD = float(os.environ.get('YS_CROSS_BEHIND_PAD', '0.10'))   # 行人已過交點後從它後面過的餘裕（m）
CROSS_STOP_PAD = float(os.environ.get('YS_CROSS_STOP_PAD', '0.20'))   # 等待停止線＝need＋此值（s168d：0.05 時 CBF 仍推）
PIN_AHEAD = os.environ.get('YS_PIN_AHEAD', '1') == '1'
V_WIN = float(os.environ.get('YS_V_WIN', '1.0'))                    # 行人速度＝群中心位移的時間窗（s）
V_TAU = float(os.environ.get('YS_V_TAU', '0.6'))                    # 行人群速度平滑時間常數（s）
CROSS_SLOW = os.environ.get('YS_CROSS_SLOW', '1') == '1'             # 橫越：可算出行人通過時刻 ⇒ 減速使抵達停止線時剛好清空，不先衝到線再停
CROSS_REL_S = float(os.environ.get('YS_CROSS_REL_S', '0.5'))         # 橫越等待：衝突消失多久放行（原 1.0）
AHEAD_SKIP = os.environ.get('YS_AHEAD_SKIP', '1') == '1'
PRED = os.environ.get('YS_PRED', '1') == '1'                         # Howard 11/23「停下來就沒意義、不是能預測障礙物之後位置」：以等速預測模擬候選動作（速度×側移）8 s，選損失時間最少且全程安全者；全不安全才停
PRED_T = float(os.environ.get('YS_PRED_T', '8.0'))
PRED_DT = float(os.environ.get('YS_PRED_DT', '0.25'))
PRED_R = float(os.environ.get('YS_PRED_R', '6.0'))                    # 只看這距離內、移動中的行人
PRED_VMAX = float(os.environ.get('YS_PRED_VMAX', '0.26'))            # 加速上限（輪速硬體上限 95%，同導航名目上限）
PRED_SWITCH = float(os.environ.get('YS_PRED_SWITCH', '1.0'))         # 換選擇的代價（s），防來回跳
PRED_BEHIND = [float(x) for x in os.environ.get('YS_PRED_BEHIND', '0.6,1.0').split(',') if x]   # 往行人背後偏的量（m）
PRED_V_FAST = float(os.environ.get('YS_PRED_V_FAST', '0.18'))
PRED_HYST = float(os.environ.get('YS_PRED_HYST', '0.10'))
PRED_LAT_W = float(os.environ.get('YS_PRED_LAT_W', '1.0'))           # 側移代價（s／m）
CROSS_MAX_S = float(os.environ.get('YS_CROSS_MAX_S', '15.0'))
PRED_BUDGET_MS = float(os.environ.get('YS_PRED_BUDGET_MS', '15'))   # 預測選動作每拍時間預算（ms；0＝不限）
PRED_MIN_AGE = float(os.environ.get('YS_PRED_MIN_AGE', '0.8'))       # 預測選動作只看追蹤滿此秒數的群（＝0.8×V_WIN，速度改由群中心位移估）
GHOST_CLR = float(os.environ.get('YS_GHOST_CLR', '0.10'))            # 移動群的點離地圖佔據格（家具、牆）中位距離小於此 ⇒ 視為靜物假移動（0＝關）
if GHOST_CLR > 0.0:                                                  # 匯入時先載地圖（第一次載入約 40 ms，不要落在控制拍裡）
    try:
        _load_map()
    except Exception:                                                # noqa: BLE001
        pass


class YieldSide:
    def __init__(self):
        self.lock = None                                              # dict(name_key, n, s_tgt)
        self.off = 0.0                                                # 目前已套用的側移量（m，沿 lock['n']）
        self.log = []
        self._trk, self._G, self._t_G, self.ref_dt = [], [], None, 0.1
        self.pchoice, self.p_off = None, 0.0

    def _smooth(self, obs, t):
        """行人群速度跨拍平滑（s172d：追蹤速度 0.14–0.57 m/s 跳動，真值 0.15 ⇒ 等待／側移／放行 10 s 內切換二十餘次）：
        以上一拍各群「位置＋速度×dt」最近者（＜0.6 m）配對，速度一階濾波（時間常數 V_TAU）；新群用原速度。只在本層用，不改追蹤器與 CBF。"""
        if self._t_G == t:
            return
        G = _groups(obs); dt_ = 0.0 if self._t_G is None else max(0.0, float(t) - self._t_G)
        if GHOST_CLR > 0.0 and _load_map():                           # Isaac 124（無行人）仍觸發 yield_pred 兩次：落在地圖家具／牆上的「移動群」是追蹤器把靜物可見面變化當成移動（桌腳等），不是行人 ⇒ 本層略過（不改追蹤器與 CBF）
            keep_ = []
            for gp in G:
                c_ = float(np.median([clearance(float(p_[0]), float(p_[1])) for p_ in gp['P']]))
                if c_ < GHOST_CLR:
                    if float(t) - getattr(self, '_t_ghost_log', -9.0) >= 1.0:
                        self._t_ghost_log = float(t)
                        m_g = gp['P'].mean(axis=0)
                        self.log.append(dict(t=round(float(t), 2), what='ys_map_ghost', m=[round(float(m_g[0]), 2), round(float(m_g[1]), 2)], v=[round(float(x), 3) for x in gp['v']], n=int(len(gp['P'])), clr=round(c_, 3)))
                    continue
                keep_.append(gp)
            G = keep_
        a_ = 1.0 - math.exp(-dt_ / V_TAU) if V_TAU > 0 and dt_ > 0 else 1.0
        new = []
        for gp in G:
            m_ = gp['P'].mean(axis=0); best, bd = None, 0.6
            for tr in self._trk:
                d_ = float(np.linalg.norm(tr['m'] + tr['v'] * dt_ - m_))
                if d_ < bd:
                    best, bd = tr, d_
            h_ = ([] if best is None else [x for x in best['h'] if float(t) - x[0] <= V_WIN + 0.5]) + [(float(t), m_)]
            v_m = gp['v']
            if float(t) - h_[0][0] >= 0.8 * V_WIN:                     # s178d：移動中感測時追蹤速度 0.525（真值 0.19）⇒ 改用群中心在 V_WIN 秒內的位移（世界座標，與機器人運動無關）
                v_m = (m_ - h_[0][1]) / (float(t) - h_[0][0])
            v_ = v_m if best is None else best['v'] + a_ * (v_m - best['v'])
            gp['v_raw'] = gp['v']; gp['v'] = v_; gp['age'] = float(t) - h_[0][0]
            new.append(dict(m=m_, v=v_, h=h_))
        self._trk, self._G, self._t_G = new, G, float(t)

    def _conflict(self, q, v_b, obs, any_ds=False):
        """回 (群, t*, d*, need, n, (lo, hi)) 或 None。以群的保守區間中心當圓心、半寬當半徑做最近接近預測。"""
        best = None
        for gp in self._G:
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
            if AHEAD_SKIP and sp >= V_MIN:                            # 離線 isaac85：橫越等待放行（從後方過、間距 ≥need_b）後，最近接近預測（need 0.8）又判衝突 ⇒ 側移鎖定、跟在行人後面被帶走；行人在前方且機器人沿其方向追不上 ⇒ 間距只增不減，不算衝突
                m_ = gp['P'].mean(axis=0); d_ = m_ - q[:2]; nd_ = float(np.linalg.norm(d_))
                m_ = m_ + (R_MIN * d_ / nd_ if nd_ > 1e-6 else 0.0)
                if float((m_ - q[:2]) @ u) >= R_BASE + R_MIN + CROSS_BEHIND_PAD and float(v_b @ u) <= sp:
                    continue
            al = float((gp['P'] @ u).mean())
            c = al * u + 0.5 * (lo + hi) * n
            rad = 0.5 * (hi - lo)
            p = c - q[:2]; vr = vo - v_b; vv = float(vr @ vr)
            if vv < 1e-6:
                continue
            ts = -float(p @ vr) / vv
            if not (0.0 < ts < T_LOOK + (2.0 if any_ds else 0.0)):   # s178d：鎖定中 ts 在 4.5–5.0 s 邊界上 ⇒ 每 0.1 s 鎖／放；鎖定後預測窗延長 2 s（遲滯）
                continue
            ds = float(np.linalg.norm(p + vr * ts))
            need = R_BASE + rad + PAD
            if (ds < need or (any_ds and ds < need + 0.6)) and (best is None or ts < best[1]):
                best = (gp, ts, ds, need, n, (lo, hi))
        return best

    def _alongside(self, q, v_b, obs):
        """同向並行：行人在動、機器人朝同方向走、目前距離 < 安全距＋0.2 ⇒ 回 (u, n, 中心, 距離, need)。"""
        best = None
        for gp in self._G:
            vo = gp['v']; sp = float(np.linalg.norm(vo))
            nb_ = float(np.linalg.norm(v_b))
            if sp < V_MIN or nb_ < 0.05 or float(np.dot(v_b / nb_, vo / sp)) < ALONG_COS:   # s147d：橫越的行人（夾角 45°）也被當同向並行 ⇒ 進度被釘、卡 10 s ⇒ 只在夾角 ≤30° 時算並行
                continue
            u = vo / sp; n = np.array([-u[1], u[0]])
            lo, hi = _interval(gp, n, q[:2])
            c = float((gp['P'] @ u).mean()) * u + 0.5 * (lo + hi) * n
            need = R_BASE + 0.5 * (hi - lo) + ALONG_PAD                # s147d：側向 0.8 m 平行經過也被當成在行人帶內 ⇒ 進度被釘 8 s；只在真的同一條帶（餘裕 0.10）才算
            dist = float(np.linalg.norm(c - q[:2]))
            lat = abs(float((q[:2] - c) @ n)); alg = abs(float((q[:2] - c) @ u))
            if lat < need and alg < need + 0.5 and (best is None or dist < best[3]):   # s115d：只在真的落在行人帶內才算（側向 0.9 m 外也鎖 ⇒ 卡 130 s）
                best = (u, n, c, dist, need)
        return best

    def _crossing(self, q, e, obs, sticky=False):
        """橫越衝突：回 dict(n, c_lat, need, sg, sig_in, a, sp) 或 None。只用追蹤速度 ≥ V_MIN 的群（不用相對運動代替）。
        機器人在行人帶外、參考走向 e 朝帶內；以 CROSS_V 直線走過帶的期間，沿行人方向的間距（線性）兩端都夠大（從後面過 need_b）或都夠負（先過 need＋餘裕）才不算衝突
        （Isaac 85：斜向路徑與行人同向時，用中心線交點判會一直等交點被追上，改看實際間距）。"""
        best = None
        for gp in self._G:
            vo = gp['v']; sp = float(np.linalg.norm(vo))
            if sp < V_MIN:
                continue
            u = vo / sp
            if abs(float(u @ e)) > CROSS_COS:
                continue
            n = np.array([-u[1], u[0]])
            lo, hi = _interval(gp, n, q[:2])
            c_lat = 0.5 * (lo + hi); need = R_BASE + 0.5 * (hi - lo) + PAD
            s_q = float(q[:2] @ n) - c_lat
            en = float(e @ n)
            if abs(s_q) < (need - 0.25) or en * s_q >= 0.0 or abs(en) < 0.2:   # 已在帶內／不朝帶走。初判也用 need−0.25（s175d：帶緣側距 0.8、行人並排 ⇒ 初判 need−0.05 不接手、衝突預測每 0.2 s 開關 ⇒ 沿行人方向被帶 3 m）
                continue
            sig_in = (abs(s_q) - need) / abs(en); sig_out = (abs(s_q) + need) / abs(en)
            tr0, tr1 = max(sig_in, 0.0) / CROSS_V, sig_out / CROSS_V
            look_, mg_ = (CROSS_LOOK + 3.0, CROSS_MARGIN_S + 0.5) if sticky else (CROSS_LOOK, CROSS_MARGIN_S)   # 等待中放寬（s167d：預測窗邊界上 0.6–2.8 s 開關閃爍）
            if tr0 > look_:
                continue
            m_ = gp['P'].mean(axis=0); d_ = m_ - q[:2]; nd_ = float(np.linalg.norm(d_))
            c_al = float((m_ + (R_MIN * d_ / nd_ if nd_ > 1e-6 else 0.0)) @ u)   # 可見表面點偏向機器人 ⇒ 中心往遠離機器人推一個半徑（同 _interval）
            x_q, eu = float(q[:2] @ u), float(e @ u)
            g0 = (c_al + sp * tr0) - (x_q + eu * CROSS_V * tr0)       # 機器人在帶內期間（等速直線）沿行人方向的間距：正＝行人在前（已過）
            g1 = (c_al + sp * tr1) - (x_q + eu * CROSS_V * tr1)
            need_b = need - PAD + CROSS_BEHIND_PAD
            if min(g0, g1) >= need_b or max(g0, g1) <= -(need + sp * mg_):   # 從行人後面過／先過（間距線性 ⇒ 看兩端）
                continue
            if best is None or tr0 < best['tr0']:
                best = dict(en=en, n=n, c_lat=c_lat, need=need, sg=1.0 if s_q > 0 else -1.0, sig_in=sig_in, a=g0, g1=g1, sp=sp, tw0=tr0, tr0=tr0, tr1=tr1, u=u)
        return best

    def _apply_cross(self, t, q, X, cr):
        """停在行人帶外（自己的參考路徑上）：參考點側向進入帶（＋0.05）的部分改成路徑上最後一個帶外點；朝向照參考。"""
        if cr is not None:
            if getattr(self, 'cross', None) is None:
                self.cross = dict(t0=t)
                self.log.append(dict(t=round(t, 2), what='yield_cross_hold', g0=round(cr['a'], 3), g1=round(cr['g1'], 3), tr0=round(cr['tr0'], 2), tr1=round(cr['tr1'], 2), sp=round(cr['sp'], 3), sig_in=round(cr['sig_in'], 3), need=round(cr['need'], 3)))
            self.cross.update(cr=cr, t_seen=t)
        C = self.cross
        if t - C['t_seen'] > CROSS_REL_S or t - C['t0'] > CROSS_MAX_S:
            self.log.append(dict(t=round(t, 2), what='yield_cross_release', held_s=round(t - C['t0'], 2)))
            self.cross = None
            return None
        cr = C['cr']; n, c_lat, need, sg = cr['n'], cr['c_lat'], cr['need'], cr['sg']
        stop_ = need + CROSS_STOP_PAD                                   # s168d：停止線 need＋0.05 太貼行人帶 ⇒ 行人經過時 CBF（0.38）把機器人沿行人方向推走；多留一點
        if CROSS_SLOW and C.get('pin') is None:                       # 減速代替停車：行人還要前進 Δ＝need_b−min(g0,g1) 才清空 ⇒ 延後 τ＝Δ/sp；抵達停止線的時刻排在清空時刻
            en_ = max(abs(float(cr.get('en', 1.0))), 0.2)
            d_stop = max(0.0, float(cr['sig_in']) - CROSS_STOP_PAD / en_)  # 沿路徑到停止線的距離
            dl_ = max(0.0, (need - PAD + CROSS_BEHIND_PAD) - min(float(cr['a']), float(cr['g1'])))
            T_ = float(cr['tr0']) + dl_ / max(float(cr['sp']), 0.05)
            v_app = d_stop / max(T_, 1e-3)
            P_ = [np.array([float(Xk[0, 2]), float(Xk[1, 2])]) for Xk in X]
            S_ = np.r_[0.0, np.cumsum([float(np.linalg.norm(P_[k + 1] - P_[k])) for k in range(len(P_) - 1)])]
            v_nav = float(S_[-1]) / max((len(P_) - 1) * float(self.ref_dt), 1e-3)
            rho = min(1.0, v_app / v_nav) if v_nav > 0.02 else 1.0
            if rho < 0.98:
                for k, Xk in enumerate(X):
                    sk = rho * float(S_[k]); j = int(np.searchsorted(S_, sk, side='right')) - 1; j = min(max(j, 0), len(P_) - 2)
                    w = (sk - S_[j]) / max(S_[j + 1] - S_[j], 1e-9); p_ = P_[j] + min(max(w, 0.0), 1.0) * (P_[j + 1] - P_[j])
                    Xk[0, 2], Xk[1, 2] = float(p_[0]), float(p_[1])
                if not C.get('slow_logged'):
                    C['slow_logged'] = True
                    self.log.append(dict(t=round(t, 2), what='yield_cross_slow', v_app=round(v_app, 3), v_nav=round(v_nav, 3), T=round(T_, 2), d_stop=round(d_stop, 2)))
        last = np.array([float(q[0]), float(q[1])])
        pin = C.get('pin')                                            # s168d：已比停止線近時「停在目前位置」每拍重設 ⇒ 漂移被逐拍接受（棘輪）⇒ 停點只設一次
        if pin is None and sg * (float(last @ n) - c_lat) < stop_:
            pin = C['pin'] = last.copy()
        for Xk in X:
            p = np.array([Xk[0, 2], Xk[1, 2]])
            if pin is None and sg * (float(p @ n) - c_lat) >= stop_:
                last = p
                continue
            if pin is None:
                pin = C['pin'] = last.copy()
            Xk[0, 2], Xk[1, 2] = float(pin[0]), float(pin[1])
        return dict(off=0.0, cross_hold=True)

    def _apply_along(self, t, q, X, al):
        """讓行人先走：沿行人方向不再前進（參考沿 u 的進度釘在目前位置），側向離開行人帶到目前所在側；行人領先 > need＋0.3 才放行（之後從它後面過）。"""
        if al is not None:
            u, n, c, dist, need = al
            if getattr(self, 'along', None) is None:
                lat0_ = float((q[:2] - c) @ n); dr_ = getattr(self, '_d_ref', None)
                if abs(lat0_) < 0.15 and dr_ is not None and float(np.linalg.norm(dr_)) > 0.10:
                    sg = 1.0 if float(dr_ @ n) >= 0.0 else -1.0           # s151d：已在行人帶中央 ⇒ 往機器人要去的那側讓出
                else:
                    sg = 1.0 if lat0_ >= 0.0 else -1.0
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
        behind_ = float((c - q[:2]) @ u) < 0.0                          # 行人在機器人後方（追上來）⇒ 不釘進度（s151d：釘住＝站在行人前面等它撞），只側向讓出
        for Xk in X:
            p = np.array([Xk[0, 2], Xk[1, 2]])
            a_ = float((p - q[:2]) @ u)
            if a_ > 0.0 and not behind_:
                p = p - a_ * u                                        # 不沿行人方向前進
            s_k = float(p @ n)
            if sg * (s_k - s_c) < L['need']:
                p = p + (sg * L['need'] - (s_k - s_c)) * n            # 側向至少離行人帶 need（目前所在側）
            Xk[0, 2], Xk[1, 2] = float(p[0]), float(p[1])
        return dict(off=0.0, along=True)

    def _pred_choose(self, t, q, X, obs):
        """預測選動作：候選＝速度比 ρ × 側移（無／左右到行人帶外／再外 0.3 m）。機器人沿參考走向 e 等速直線（ρ·v_nav），側移以 RATE 爬升到目標側向座標（同 lock 公式）；
        行人等速預測。有效＝PRED_T 內離每位行人中心 ≥ need；側移點離地圖障礙 ≥R_BASE＋WALL_PAD、離靜態未知點 ≥R_BASE＋0.12（直線近似本身已入障礙的時刻不判）。
        代價＝時間損失（PRED_T − 沿 e 前進量／v_nav）＋側移量×PRED_LAT_W＋換選擇代價。全無效 ⇒ 取最小距離最大者。回 dict 或 None（名目即可）。"""
        P_ = [np.array([float(Xk[0, 2]), float(Xk[1, 2])]) for Xk in X]
        L_ = float(sum(np.linalg.norm(P_[k + 1] - P_[k]) for k in range(len(P_) - 1)))
        v_nav = L_ / max((len(P_) - 1) * float(self.ref_dt), 1e-3)
        if v_nav < 0.05:
            return None
        e = (P_[-1] - P_[0]) / max(float(np.linalg.norm(P_[-1] - P_[0])), 1e-6)
        G = []
        for gp in self._G:
            vo = gp['v']; sp = float(np.linalg.norm(vo))
            m_ = gp['P'].mean(axis=0); d_ = m_ - q[:2]; nd_ = float(np.linalg.norm(d_))
            if sp < V_MIN or nd_ > PRED_R or float(gp.get('age', 9.0)) < PRED_MIN_AGE:   # 新群速度仍是追蹤器原始值（Isaac 126：未知箱 unknown_l1 剛出現 0.44 m/s）⇒ 追到 PRED_MIN_AGE 改用群中心位移速度後才參與預測
                continue
            m_ = m_ + (R_MIN * d_ / nd_ if nd_ > 1e-6 else 0.0)
            G.append((m_, vo, R_BASE + R_MIN + PAD))
        if not G:
            return None
        stat = np.array([[float(o['x']), float(o['y'])] for o in (obs or []) if o.get('static')]).reshape(-1, 2)
        taus = np.arange(PRED_DT, PRED_T + 1e-9, PRED_DT)
        W = [(m_ + np.outer(taus, vo), nd) for m_, vo, nd in G]
        # 側移方向：以最近衝突行人的行進法向（同 lock）
        dmin_nom, k_ = 9.0, 0
        base_nom = q[:2] + np.outer(taus, e) * v_nav
        for i, (Wp, nd) in enumerate(W):
            dd = float(np.min(np.linalg.norm(base_nom - Wp, axis=1)) - nd)
            if dd < dmin_nom:
                dmin_nom, k_ = dd, i
        if dmin_nom >= 0.0 and getattr(self, 'pchoice', None) is None:
            return None
        vo = G[k_][1]; spk = float(np.linalg.norm(vo)); u = vo / spk; n = np.array([-u[1], u[0]])
        sc = float(G[k_][0] @ n); need_l = R_BASE + PAD + R_MIN
        manip_ = bool(getattr(self, 'manip', False))
        rmax = min(PRED_VMAX / v_nav, 1.2) if (v_nav >= PRED_V_FAST and not manip_) else 1.0   # 加速只在一般行駛（v_nav ≥ 0.18）且不在抓取中
        rhos = sorted(set([round(rmax, 3), 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]), reverse=True)
        lats = [(0.0, None)] if manip_ else [(0.0, None)] + [(sg, sc + sg * (need_l + ex)) for sg in (1.0, -1.0) for ex in (0.0, 0.3)] + [(2.0, float(q[:2] @ u) - dl) for dl in PRED_BEHIND]   # sg=2：往行人來的方向（−u）偏 dl（絕對座標線：沿 u 座標 ≤ 目前−dl），從它背後繞過
        prev = getattr(self, 'pchoice', None); off0 = self.p_off
        best = None
        t_b0_ = _time.perf_counter()
        for rho in rhos:
            if PRED_BUDGET_MS > 0 and best is not None and (_time.perf_counter() - t_b0_) * 1e3 > PRED_BUDGET_MS:
                self.n_pred_budget = getattr(self, 'n_pred_budget', 0) + 1   # 11/25 第 155 趟：未知箱假移動群觸發預測，每拍 40–60 ms、一拍 121 ms 超 120 ms 期限 ⇒ 預測限時，超時取目前最佳（ρ 由大到小，已含名目速度）
                break
            base = q[:2] + np.outer(taus, e) * v_nav * rho
            for sg, st in lats:
                P = base.copy()
                if sg == 2.0:
                    sh = np.minimum(np.maximum(0.0, P @ u - st), off0 + RATE * taus); P = P - np.outer(sh, u)
                elif sg != 0.0:
                    sb = P @ n; sh = sg * np.minimum(np.maximum(0.0, sg * (st - sb)), off0 + RATE * taus)
                    P = P + np.outer(sh, n)
                if sg != 0.0:
                    ok_map = True
                    for j in range(1, len(taus), 2):
                        if abs(sh[j]) > 0.05 and clearance(*base[j]) >= R_BASE + WALL_PAD and clearance(*P[j]) < R_BASE + WALL_PAD:
                            ok_map = False; break
                        if abs(sh[j]) > 0.05 and len(stat) and float(np.min(np.linalg.norm(stat - P[j], axis=1))) < R_BASE + 0.12:
                            ok_map = False; break
                    if not ok_map:
                        continue
                    lat_m = float(np.max(np.abs(sh)))
                else:
                    lat_m = 0.0
                Pa = np.vstack([q[:2], P]); sg_ = np.linalg.norm(np.diff(Pa, axis=0), axis=1); arc_i = np.cumsum(sg_)   # 合成速度上限 PRED_VMAX（前進＋側移不可超過輪速上限）：沿理想路徑以受限弧長重新取時
                arc_a = np.minimum(arc_i, PRED_VMAX * taus)
                for j in range(1, len(arc_a)):
                    arc_a[j] = min(arc_a[j], arc_a[j - 1] + PRED_VMAX * PRED_DT)
                if float(np.max(arc_i - arc_a)) > 1e-4:
                    cum = np.r_[0.0, arc_i]; idx = np.clip(np.searchsorted(cum, arc_a, side='right') - 1, 0, len(cum) - 2)
                    w_ = (arc_a - cum[idx]) / np.maximum(cum[idx + 1] - cum[idx], 1e-9)
                    P = Pa[idx] + w_[:, None] * (Pa[idx + 1] - Pa[idx])
                dm = min(float(np.min(np.linalg.norm(P - Wp, axis=1)) - nd) for Wp, nd in W)
                prog = float((P[-1] - q[:2]) @ e)
                cost = (PRED_T - prog / v_nav) + PRED_LAT_W * lat_m + (PRED_SWITCH if prev is not None and (abs(prev['rho'] - rho) > 1e-6 or prev['sg'] != sg) else 0.0)
                same_ = prev is not None and abs(prev['rho'] - rho) < 1e-6 and prev['sg'] == sg
                ok_ = dm >= (0.0 if (same_ or prev is None) else PRED_HYST)       # 換成別的選擇要多 PRED_HYST 安全餘裕（離線 isaac84：搶先過走到一半翻成等待 ⇒ 撞）
                key = (0 if ok_ else 1, cost if ok_ else -dm)
                if best is None or key < best[0]:
                    best = (key, dict(rho=rho, sg=sg, st=st, n=n, u=u, dm=round(dm, 3), cost=round(cost, 2), v_nav=round(v_nav, 3), gm=[round(float(x), 2) for x in G[k_][0]], gv=[round(float(x), 3) for x in G[k_][1]]))
        return None if best is None else best[1]

    def _apply_pred(self, t, q, X, obs, dt):
        c = self._pred_choose(t, q, X, obs)
        prev = getattr(self, 'pchoice', None)
        if c is None or (c['rho'] >= 0.999 and c['sg'] == 0.0):
            if prev is not None and self.p_off <= 1e-4:
                self.log.append(dict(t=round(t, 2), what='yield_pred_release'))
                self.pchoice = None
            if prev is None:
                return None
            c = dict(rho=1.0, sg=prev['sg'], st=prev['st'], n=prev['n'], u=prev['u'], dm=None, cost=None, v_nav=None, fade=True)
            self.p_off = max(0.0, self.p_off - RATE * dt)
        else:
            if prev is None or abs(prev['rho'] - c['rho']) > 1e-6 or prev['sg'] != c['sg']:
                self.log.append(dict(t=round(t, 2), what='yield_pred', rho=c['rho'], side=c['sg'], dm=c['dm'], cost=c['cost'], v_nav=c['v_nav'], gm=c.get('gm'), gv=c.get('gv')))   # gm／gv：觸發的行人群中心與速度（Isaac 124 無行人仍觸發，查來源）
            self.pchoice = c
            self.p_off = min(self.p_off + RATE * dt, 1.5) if c['sg'] != 0.0 else max(0.0, self.p_off - RATE * dt)
        rho = c['rho']
        P_ = [np.array([float(Xk[0, 2]), float(Xk[1, 2])]) for Xk in X]
        S_ = np.r_[0.0, np.cumsum([float(np.linalg.norm(P_[k + 1] - P_[k])) for k in range(len(P_) - 1)])]
        if abs(rho - 1.0) > 1e-3:                                     # 時間縮放（ρ＞1 時沿最後一段外推）
            d_end = (P_[-1] - P_[-2]) / max(S_[-1] - S_[-2], 1e-9) if len(P_) > 1 else np.zeros(2)
            for k, Xk in enumerate(X):
                sk = rho * float(S_[k])
                if sk >= S_[-1]:
                    p_ = P_[-1] + (sk - S_[-1]) * d_end
                else:
                    j = min(max(int(np.searchsorted(S_, sk, side='right')) - 1, 0), len(P_) - 2)
                    w = (sk - S_[j]) / max(S_[j + 1] - S_[j], 1e-9); p_ = P_[j] + w * (P_[j + 1] - P_[j])
                Xk[0, 2], Xk[1, 2] = float(p_[0]), float(p_[1])
        shift = 0.0
        if c['sg'] == 2.0 and self.p_off > 0.0:                      # 往行人背後偏：整個視窗沿 −u 平移 min(dl, 爬升量)
            u_ = c['u']
            for Xk in X:
                d_k = min(max(0.0, float(np.array([Xk[0, 2], Xk[1, 2]]) @ u_) - c['st']), self.p_off)
                Xk[0, 2] -= d_k * u_[0]; Xk[1, 2] -= d_k * u_[1]
                shift = max(shift, d_k)
        elif c['sg'] != 0.0 and self.p_off > 0.0:
            n, sg, st = c['n'], c['sg'], c['st']
            for Xk in X:
                s_k = float(np.array([Xk[0, 2], Xk[1, 2]]) @ n)
                d_k = min(max(0.0, sg * (st - s_k)), self.p_off)
                Xk[0, 2] += sg * d_k * n[0]; Xk[1, 2] += sg * d_k * n[1]
                shift = max(shift, d_k)
        return dict(off=round(shift, 3), pred=True, rho=rho, side=c['sg'])

    def apply(self, t, q, v_b, X, obs, dt):
        """X：參考視窗（3×3 SE2 串列，就地修改）。側向座標＝點與 n 的內積（絕對）；目標＝保守區間外 R_BASE＋PAD。"""
        if not ENABLE:
            return None
        q = np.asarray(q, float); v_b = np.asarray(v_b, float)[:2]
        self._smooth(obs, t)
        d_ref = np.array([float(X[-1][0, 2] - X[0][0, 2]), float(X[-1][1, 2] - X[0][1, 2])])   # s147d：以「要去的方向」（參考視窗走向）判並行，不用實際速度（被側移推向行人方向後又被當同向、被帶走 3 m）
        v_al = d_ref / float(np.linalg.norm(d_ref)) * max(float(np.linalg.norm(v_b)), 0.06) if float(np.linalg.norm(d_ref)) > 0.10 else np.zeros(2)
        self._d_ref = d_ref
        if PRED:
            return self._apply_pred(t, q, X, obs, dt)
        if CROSS and self.lock is None and getattr(self, 'along', None) is None and float(np.linalg.norm(d_ref)) > 0.10:   # 橫越：路徑上等（優先於側移選邊；側移中／並行中不切換）
            cr = self._crossing(q, d_ref / float(np.linalg.norm(d_ref)), obs, sticky=getattr(self, 'cross', None) is not None)
            if cr is not None or getattr(self, 'cross', None) is not None:
                r_ = self._apply_cross(t, q, X, cr)
                if r_ is not None:
                    return r_
        al = self._alongside(q, v_al, obs)
        if self.lock is None and (al is not None or getattr(self, 'along', None) is not None):   # s153d：橫越選邊鎖定中又進並行（選邊互相矛盾）⇒ 鎖定時不進並行   # 同向並行（s113d：相對速度≈0 ⇒ 最近接近預測不觸發，被行人帶著走 30 s）
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
                self.lock = dict(n=n, sg=sg, s_t=s_t, t0=t, lo=lo, hi=hi)
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
                    self.lock, self._pin_logged = None, False
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
                L['n'], L['lo'], L['hi'] = n, lo, hi
                self.off = min(self.off + RATE * dt, 1.5)
            n, sg = L['n'], L['sg']
            s_now_ = float(q[:2] @ n)
            s_t_ = L['s_t']
            if NO_BACK and sg * (s_t_ - s_now_) > 0.0 and sg * (s_now_ - 0.5 * (L.get('lo', s_now_) + L.get('hi', s_now_))) > 0.0:
                s_t_ = s_now_                                         # Howard 11/21「倒退閃避」：已在選定側但比安全距離近 ⇒ 停在目前側向位置等行人過，不往回退
            u_pin = None
            if PIN_AHEAD and c is not None:                           # Isaac 84／85：側移鎖定時機器人在行人前方、參考又沿行人方向 ⇒ 被行人從後面追著走 3 m 撞上家具 ⇒ 在帶外時不沿行人方向前進（站在帶外讓它過）
                vg_ = c[0]['v']; spg_ = float(np.linalg.norm(vg_))
                if spg_ >= V_MIN:
                    ug_ = vg_ / spg_
                    m_ = c[0]['P'].mean(axis=0); d_ = m_ - q[:2]; nd_ = float(np.linalg.norm(d_))
                    m_ = m_ + (R_MIN * d_ / nd_ if nd_ > 1e-6 else 0.0)
                    lat_ = abs(s_now_ - 0.5 * (L.get('lo', s_now_) + L.get('hi', s_now_)))
                    if float((q[:2] - m_) @ ug_) > -0.3 and lat_ >= R_BASE + 0.5 * (L.get('hi', s_now_) - L.get('lo', s_now_)) + 0.05:
                        u_pin = ug_
            if u_pin is not None and not getattr(self, '_pin_logged', False):
                self._pin_logged = True
                self.log.append(dict(t=round(t, 2), what='yield_pin_ahead'))
            shift = 0.0
            for Xk in X:                                              # 各參考點側向座標至少達 s_t（鎖定側），平移量上限＝爬升量 off
                if u_pin is not None:
                    p_ = np.array([Xk[0, 2], Xk[1, 2]]); a_ = float((p_ - q[:2]) @ u_pin)
                    if a_ > 0.0:
                        Xk[0, 2], Xk[1, 2] = float(p_[0] - a_ * u_pin[0]), float(p_[1] - a_ * u_pin[1])
                s_k = float(np.array([Xk[0, 2], Xk[1, 2]]) @ n)
                d_k = min(max(0.0, sg * (s_t_ - s_k)), self.off)
                Xk[0, 2] += sg * d_k * n[0]
                Xk[1, 2] += sg * d_k * n[1]
                shift = max(shift, d_k)
            return dict(off=round(shift, 3), side=sg)
        return None
