#!/usr/bin/env python3
"""在線視向候選（Codex 20261102_092601；版本化子類，基線＝bottle_w1_online5_ctrl，GMPC、抓取速度、新鮮度、處置期限、安全規則全部相同）。
問題：V_HIGH 是固定的底盤座標關節構型，光軸偏左約 47°；R-B1 接近路徑在區域 y＋0.33 時瓶幾乎在車頭正前方，固定視角把瓶推出畫面右緣（實錄 8.2–11.33 s）。
做法：只在序列 U／H 段給 q1 加一個偏移 g（關節空間加法，序列本身的 U／H／R min-jerk 與換段條件不變）：
  目標 g*：用實際相機 FK（link_eef × 固定外參 T_MOUNT，與 Isaac 手腕相機同一外參）算「已知區域觀看點」LOOK 在影像的水平座標；
    觀看點在中央 ±BAND_PX 內 ⇒ g*＝0（不介入）；超出 ⇒ 以 q1 牛頓法解到最近的帶邊（最小修正），|g*| ≤ G_MAX。
    手臂構型取序列在 t＋LEAD 的參考（U 段過渡時光軸變化快，前瞻補追蹤器落後；LEAD＝V_G／A_G＝加速到限速所需時間），底盤取本拍實測位姿。
    觀看點在相機後方或 q1 對水平座標的靈敏度過低 ⇒ 沿用上一拍目標（記 hold）。
  追蹤：g 以速度 V_G、加速度 A_G 限制追 g*（離散、不越過目標），參考在位置與速度上連續；不每拍重啟插值、不跳回固定 V_HIGH。
  R 段起 g*＝0：g 在限制內回到 0；R→DONE 另要求 g、ġ 都已歸零（否則照基線規則等候，超過 WAIT_MAX 受控停止）。U→H 的到位判定比對 V_HIGH＋g。
  時域預覽：每拍以目前目標前推追蹤器 N＋10 拍，_rel_target(τ) 在這條預覽上取值（超出端點保持）。
已知區域只用區域知識，不讀真值瓶位或生成桌高：LOOK＝本批設計域中心（瓶 xy (0, 0.42)＋dx∈[−0.08, 0.08]、dy∈[−0.025, 0.04] 的中心；
  z＝桌高域 [0.55, 0.65] 中點＋瓶高一半）。AREA 只供離線覆蓋核對，控制器不用它。
BAND_PX：畫面半寬 320 px 減去容許區＋支撐環（0.08＋0.09 m）在 H 段約 1.3 m 處的半寬約 61 px，再留約 60 px ⇒ 200 px。
不改 q2–q6，不改漸變後的任何參考；gaze_on＝False 時與 online5 逐拍相同。
"""
import math

import numpy as np

import bottle_ref as BR
import bottle_s0b_ctrl as B
import bottle_w1_online5_ctrl as ON5

C3, C2 = ON5.C3, ON5.C2
LOOK = np.array([0.0, 0.4275, 0.70])
AREA = dict(x=(-0.08, 0.08), y=(0.395, 0.46), z1=(0.55, 0.65))
V_G, A_G = 0.6, 1.5
LEAD = V_G / A_G
BAND_PX, G_MAX = 200.0, 0.9
Z_MIN, DXN_MIN = 0.2, 0.2                  # 觀看點至少在相機前 0.2 m；|d(x/z)/dq1| 下限（1/rad）
W_PX, HFOV = 640, math.radians(69.0)
FX = (W_PX / 2) / math.tan(HFOV / 2)
E1 = np.r_[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def _tf(xyz, r=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _rpy(*r), xyz
    return T


T_MOUNT = _tf((0.06746, -0.0175, 0.0237), (math.pi, -math.pi / 2, 0)) @ _tf((0, 0.015, 0)) @ _tf((0, 0, 0), (-math.pi / 2, 0, -math.pi / 2))   # link_eef→camera_color_optical_frame（URDF）


def look_cam(K, qb, qa, d, L=LOOK):
    """觀看點在相機光學座標（x 右、y 下、z 前）；qb＝底盤 (x, y, yaw)、qa＝手臂 6 軸、d＝q1 偏移。"""
    Tc = K.fk(np.r_[qb, np.asarray(qa, float) + d * E1], 'link_eef') @ T_MOUNT
    R, p = Tc[:3, :3], Tc[:3, 3]
    return R.T @ (np.asarray(L, float) - p)


def gaze_target(K, qb, qa, d0, L=LOOK):
    """回傳 (目標 q1 偏移或 None, 原因, 解後水平像素殘差)。"""
    p0 = look_cam(K, qb, qa, 0.0, L)
    if p0[2] < Z_MIN:
        return None, 'behind', None
    e0, b = p0[0] / p0[2], BAND_PX / FX
    if abs(e0) <= b:
        return 0.0, 'in_band', None
    u = math.copysign(b, e0)
    d = float(d0)
    for _ in range(4):
        p = look_cam(K, qb, qa, d, L)
        if p[2] < Z_MIN:
            return None, 'behind', None
        e = p[0] / p[2] - u
        p2 = look_cam(K, qb, qa, d + 1e-4, L)
        de = (p2[0] / p2[2] - u - e) / 1e-4
        if abs(de) < DXN_MIN:
            return None, 'ill', None
        d = float(np.clip(d - e / de, -G_MAX, G_MAX))
    p = look_cam(K, qb, qa, d, L)
    return d, 'solved', float(FX * (p[0] / p[2] - u))


def track(g, v, tgt, dt, vmax=V_G, amax=A_G):
    """速度、加速度限制的離散追蹤；回傳新的 (g, v)。本拍速度取「走完本拍後仍能以 amax 逐拍煞停於目標」的上限：
    v²／(2A)＋1.5·v·dt ≤ |e|（離散煞停距離），並 ≤ |e|／dt、≤ vmax，所以不越過固定目標。"""
    e = tgt - g
    vb = amax * (-1.5 * dt + math.sqrt(2.25 * dt * dt + 2.0 * abs(e) / amax))
    vd = math.copysign(min(vmax, vb, abs(e) / dt), e) if e != 0.0 else 0.0
    v2 = v + float(np.clip(vd - v, -amax * dt, amax * dt))
    g2 = g + v2 * dt
    if e != 0.0 and abs(tgt - g2) <= 1e-12 * max(1.0, abs(tgt)):   # 只吸收捨入：真正越過（目標移動時加速度限制所致）照實保留，不拉回（拉回＝位置跳動）
        g2 = tgt
    if g2 == tgt and abs(v2) <= amax * dt * 1e-9:
        v2 = 0.0
    return g2, v2


def preview(g, v, tgt, n, dt):
    out = [g]
    for _ in range(n):
        g, v = track(g, v, tgt, dt)
        out.append(g)
    return np.array(out)


class W1PlaceController(ON5.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN_ONLINE5_CENTERCONV_GAZE'

    def __init__(self, *a, gaze_on=True, **kw):
        super().__init__(*a, **kw)
        self.gaze_on = bool(gaze_on)
        self.gz = dict(g=0.0, v=0.0, tgt=0.0, t0=0.0, P=np.zeros(1), why='off')
        self.gz_log, self.gz_n = [], dict(in_band=0, solved=0, hold_behind=0, hold_ill=0, zero=0)
        self.gz_max = dict(g=0.0, v=0.0, a=0.0, res_px=0.0)
        self.gz_ev = dict(t_first_offset=None, g_at_R=None, t_zero_after_R=None)

    # ---- 序列參考（同 segtarget_vhigh._rel_target 的關節值，未加偏移）
    def _qseq(self, tau):
        sq = self.vh_seq
        if sq['ph'] in ('U', 'R'):
            s = BR.min_jerk(min(max((tau - sq['t0']) / sq['T'], 0.0), 1.0))
            return sq['qa'] + (sq['qb'] - sq['qa']) * s
        return sq['qb']

    def _gz_at(self, tau):
        P = self.gz['P']
        k = (tau - self.gz['t0']) / B.DT
        if k <= 0.0:
            return float(P[0])
        if k >= len(P) - 1:
            return float(P[-1])
        i = int(k)
        return float(P[i] + (P[i + 1] - P[i]) * (k - i))

    def _at(self, q, qc):
        if self.vh_seq['ph'] == 'R' and (self.gz['g'] != 0.0 or self.gz['v'] != 0.0):
            return False                                      # 回到備妥前 g 必須已歸零
        return super()._at(q, np.asarray(qc, float) + self.gz['g'] * E1)

    def _vstep(self, q):
        super()._vstep(q)
        if not self.gaze_on:
            return
        ph, t = self.vh_seq['ph'], float(self.t)
        gz = self.gz
        if ph in ('U', 'H'):
            tgt, why, res = gaze_target(self.K, np.asarray(q[:3], float), self._qseq(t + LEAD), gz['tgt'])
            if tgt is None:
                self.gz_n['hold_' + why] += 1
                tgt = gz['tgt']
            else:
                self.gz_n[why] += 1
                if res is not None:
                    self.gz_max['res_px'] = max(self.gz_max['res_px'], abs(res))
        else:
            tgt, why = 0.0, 'zero'
            self.gz_n['zero'] += 1
            if ph == 'R' and self.gz_ev['g_at_R'] is None:
                self.gz_ev['g_at_R'] = round(gz['g'], 5)
        g0, v0 = gz['g'], gz['v']
        g1, v1 = track(g0, v0, tgt, B.DT)
        if ph not in ('U', 'H') and g1 == 0.0 and v1 == 0.0 and self.gz_ev['t_zero_after_R'] is None and self.gz_ev['g_at_R'] is not None:
            self.gz_ev['t_zero_after_R'] = round(t, 3)
        if g1 != 0.0 and self.gz_ev['t_first_offset'] is None:
            self.gz_ev['t_first_offset'] = round(t, 3)
        gz.update(g=g1, v=v1, tgt=float(tgt), t0=t, why=why)
        gz['P'] = preview(g1, v1, float(tgt), B.N + 10, B.DT)
        self.gz_max['g'] = max(self.gz_max['g'], abs(g1))
        self.gz_max['v'] = max(self.gz_max['v'], abs(v1))
        self.gz_max['a'] = max(self.gz_max['a'], abs(v1 - v0) / B.DT)
        if len(self.gz_log) < 4000:
            self.gz_log.append((round(t, 3), ph, round(float(q[0]), 4), round(float(q[1]), 4), round(g1, 5), round(v1, 5), round(float(tgt), 5), why))

    def _rel_target(self, tau):
        if not self.gaze_on:
            return super()._rel_target(tau)
        return self.K.fk(np.r_[0.0, 0.0, 0.0, self._qseq(tau) + self._gz_at(tau) * E1], 'link_tcp')

    def report(self):
        r = super().report()
        r['gaze'] = dict(on=self.gaze_on, look=LOOK.tolist(), band_px=BAND_PX, v_g=V_G, a_g=A_G, lead=LEAD, g_max=G_MAX, z_min=Z_MIN, dxn_min=DXN_MIN, fx=FX,
                         T_mount=T_MOUNT.round(6).tolist(), counts=self.gz_n, max=self.gz_max, ev=self.gz_ev, state=dict(g=self.gz['g'], v=self.gz['v'], tgt=self.gz['tgt']),
                         log=self.gz_log, note='q1 偏移只在 U／H 段；R 段起歸零；觀看點＝本批設計域中心（區域知識），不讀真值或生成桌高')
        return r
