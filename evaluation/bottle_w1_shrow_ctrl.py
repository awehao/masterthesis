#!/usr/bin/env python3
"""第一拍命令與 Shield 一致候選（版本化子類，2026-11-01；Codex 20261101_130612 選 (a) 限定版）。以分段抬升＋進 CARRY 重規劃（bottle_w1_liftc_ctrl）為底，凍結 G 不動。
動機（bottle_w1_dyn2lc_g1）：CARRY_NAV 中預測 CBF 讓底盤往 +y 讓路（vy +0.19 m/s），朝第一桌腳；GMPC 的靜態 CBF 允許的接近速度比 Shield 屏障大，
Shield 改寫 9.8 mm/s ⇒ 「介入即停」。本候選只讓規劃端提前尊重執行端已存在的限制，Shield 本身、參數、介入即停、原 CBF 都不改：
  每次 GMPC 求解，把最新一幀原始掃描以 Shield 自己的掃描處理（ScanSafetyShield._scan_cb：量程、鄰點一致、最多點數、自身回波排除、lidar→base 平移 0）
  轉成底盤座標點，對每點把 Shield 屏障 a_i·u ≤ α(d_i − d_stop(v))，v＝max(0, a_i·u)、d_stop＝d0＋v·τ_eff＋v²/(2a_brake)＋eps，
  解成逐點允許接近速度上界（二次式正根；c_i＝α(d_i−d0−eps) < 0 時上界＝c_i，即須退離）⇒ 精確等價的線性列 a_i·u ≤ b_i，加上 Shield 速度框；
  a_i 的旋轉分量對圓形外框恆為 0。只限第一拍（gmpc_wb u0_rows）。
  τ_eff＝tau＋掃描年齡上界：（求解時刻 − 掃描時刻）＋命令期限 D_STEPS·PHYS（Shield 須在 8 物理步內回覆，否則 cmd_timeout；故評估時年齡不超過此值）。
  不會改變的列（在 Shield 速度框內不可能貼到：b_i ≥ |n_x|·vx_max＋|n_y|·vy_max）不加，屬精確剪枝。
回傳命令（煞停、加速度夾、輪級後處理之後）對列違反 > 1e-6 ⇒ 標記 wb_fallback:u0_rows_violated，走既有受控停止（該筆命令不送出）。
不能宣稱此後 Shield 永不介入：命令在途期間若有新掃描先到 Shield、或移動物移動，Shield 用的點與本列不同。掃描失效仍走原 scan_stale 停止。
"""
import numpy as np

from ammr_wholebody_mpc import gmpc_wb as GW

import bottle_nav_chain as NC
import bottle_s0_ctrl as C0
import bottle_w1_liftc_ctrl as LC
from ammr_wholebody_mpc.scan_safety_shield import ScanSafetyShield

TOL = 1e-6
AGE_PAD = C0.D_STEPS * C0.PHYS


def make_shield():
    return NC._node(ScanSafetyShield, NC.SHIELD)


def scan_points(sh, ranges):
    """同 NC.on_scan 的訊息格式餵 Shield 自己的 _scan_cb（lidar→base 平移 0、同向），回傳其底盤座標點。"""
    m = NC.RS.LaserScan()
    m.header.frame_id = 'lidar_link'
    m.angle_min, m.angle_max, m.angle_increment = NC.SCAN_MIN, NC.SCAN_MAX, NC.SCAN_INC
    m.range_min, m.range_max = NC.RANGE_MIN, NC.RANGE_MAX
    m.ranges = [float(v) for v in ranges]
    sh.tf_buffer.tf = NC.RS.transform(0.0, 0.0, 0.0)
    sh._scan_cb(m)
    return np.array(sh._pts, float).reshape(-1, 2)


def shield_rows(sh, pts, age, prune=True):
    """Shield 屏障的逐點精確線性化：回傳 (A (m×3), b (m), info)。"""
    u_max = np.asarray(sh.u_max, float)
    A_box = np.vstack([np.eye(3), -np.eye(3)])
    b_box = np.r_[u_max, u_max]
    if not len(pts):
        return A_box, b_box, dict(n_pts=0, n_rows=0, d_min=None)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    n = pts / np.maximum(rad[:, None], 1e-9)
    d = rad - sh.R
    al, ab = float(sh.alpha), float(sh.a_brake)
    tau = float(sh.tau) + max(0.0, float(age))
    c = al * (d - sh.d0 - sh.eps)
    qa, qb = al / (2.0 * ab), 1.0 + al * tau
    vbar = (-qb + np.sqrt(qb * qb + 4.0 * qa * np.maximum(c, 0.0))) / (2.0 * qa)
    b = np.where(c >= 0.0, vbar, c)
    A = np.column_stack([n, np.zeros(len(n))])
    keep = (b < np.abs(n[:, 0]) * u_max[0] + np.abs(n[:, 1]) * u_max[1]) if prune else np.ones(len(b), bool)
    return np.vstack([A[keep], A_box]), np.r_[b[keep], b_box], dict(n_pts=int(len(pts)), n_rows=int(keep.sum()), d_min=float(d.min()), tau_eff=tau)


class GMPCShRows(GW.GMPC):
    def __init__(self, cfg, ctrl):
        super().__init__(cfg)
        self.ctrl = ctrl

    def solve(self, *a, **kw):
        if 'u0_rows' in kw:                                        # gmpc_wb 內部 hold／耦合遞迴：照傳
            return super().solve(*a, **kw)
        rows = self.ctrl._u0_rows()
        r = super().solve(*a, u0_rows=None if rows is None else rows[:2], **kw)
        self.ctrl._u0_post(rows, r)
        return r


class W1PlaceController(LC.W1PlaceController):
    group = 'G_SHROW'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.sh_rows = make_shield()
        self.scan_last, self.u0_log, self.u0_viol = None, [], None
        self.u0_n, self.u0_bind, self.u0_rows_max = 0, 0, 0
        fn = self.scan_fn
        if fn is not None:
            def cap(t, q, _fn=fn):
                r_ = _fn(t, q)
                if r_ is not None:
                    self.scan_last = (float(t), np.array(r_, float))
                return r_
            self.scan_fn = cap
        wrap = {}                                                  # 原本同一物件（mpc／mpc_nav／mpc_nav0）包裝後仍同一個：切換語義不變
        for nm in ('mpc', 'mpc_nav', 'mpc_nav0', 'mpc_park'):
            g = getattr(self, nm, None)
            if isinstance(g, GW.GMPC) and not isinstance(g, GMPCShRows):
                wrap.setdefault(id(g), GMPCShRows(g.cfg, self))
                setattr(self, nm, wrap[id(g)])

    def _u0_rows(self):
        if self.scan_last is None:
            return None
        t_s, rr = self.scan_last
        pts = scan_points(self.sh_rows, rr)
        age = (self.t - t_s) + AGE_PAD
        A, b, info = shield_rows(self.sh_rows, pts, age)
        return A, b, dict(info, t_scan=t_s, age_bound=age)

    def _u0_post(self, rows, r):
        if rows is None:
            return
        A, b, info = rows
        u = np.asarray(r.u_opt, float)[:3]
        s = b - A @ u
        viol = float(r.u0_rows_viol)
        n_act = int((s[:info['n_rows']] <= 1e-6).sum())
        self.u0_n += 1
        self.u0_bind += int(n_act > 0)
        self.u0_rows_max = max(self.u0_rows_max, info['n_rows'])
        self.u0_log.append((round(float(self.t), 3), info['n_rows'], n_act, round(float(s[:info['n_rows']].min()), 6) if info['n_rows'] else None, round(viol, 9),
                            None if info['d_min'] is None else round(info['d_min'], 4), round(info['age_bound'], 3)))
        if viol > TOL:
            if self.u0_viol is None:
                self.u0_viol = dict(t=float(self.t), viol=viol, u=u.tolist(), status=str(r.status), accept=str(r.accept_action), n_rows=info['n_rows'])
            ck = dict(r.arm_check or {})
            ck.update(fallback='u0_rows_violated', failed=True)        # 走既有 wb_fallback 受控停止，這筆命令不送出
            r.arm_check = ck

    def report(self):
        r = super().report()
        act = [x for x in self.u0_log if x[2] > 0]
        r['paired_strategy'] = dict(r['paired_strategy'], name='G_W1_liftc_shield_rows', u0_rows=dict(n_solve=self.u0_n, n_bind=self.u0_bind, rows_max=self.u0_rows_max,
                                    violation=self.u0_viol, bind_first=act[:5], bind_last=act[-5:], log_tail=self.u0_log[-40:], age_pad=AGE_PAD, tol=TOL,
                                    note='第一拍接受列＝Shield 屏障逐點精確線性化（原始掃描、Shield 自身掃描處理）；不改 Shield'))
        return r
