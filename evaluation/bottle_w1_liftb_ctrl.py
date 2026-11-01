#!/usr/bin/env python3
"""抓持後分段抬升候選（版本化子類，2026-11-01；Codex 20261101_123440）。凍結 G（bottle_w1_ctrl）不動。
問題：瓶已抓牢後，LIFT 仍要求 TCP 維持原世界位置（世界座標抬升目標），避障推動底盤時手臂必須補償水平位移，耗掉抬升能力
（bottle_w1_dyn2b2_g1：底盤每 0.05 s 把 TCP 推離 0.5–6.3 mm、手臂補 1.7–3.3 mm、關節 4 速度飽和、抬升停滯）。
假說：瓶已離桌後，運送不需要鎖世界水平位置。
本候選：
  LIFT_A（不變）：世界固定抬升；抓持證據未成立或瓶未可靠離桌前，不切換目標語義。
  切換條件（離桌可靠）：LIFT 相位、抓持證據已成立、桌面受力 |f| ≤ 0.02 N（同離桌判定的 table_force_tol_n）、瓶底間隙 ≥ G_SWITCH，連續一個控制週期。
    G_SWITCH＝滑移判準 slip_m 5.0 mm（超過即停止）＋既有趟次持瓶段 TCP 最大下沉 0.51 mm（w1／ho／dyn／p 共 10 趟實測）
            ＋瓶淨空拒絕線（CLEAR_REJECT＋CLEAR_GUARD＝5.5 mm）＝11.0 mm：運送與停止期間瓶底仍高於桌面且留拒絕線餘裕。
  LIFT_B（運送式抬升）：與 CARRY 同一種底盤座標任務（tcp_rel_base）——水平位置與姿態承接切換當下的實際 TCP 相對底盤位姿，
    垂直沿 min-jerk（時長＝task.t_lift）由切換時高度升到原抬升高度（世界 z＝原 T_lift 高度）；設定點與在途命令照常承接，目標不跳。
  CARRY 進入：同原判定（抬升時長已過、TCP 在容差內），只是比對的抬升目標改為「目前實測底盤 × 底盤座標終點」；進入 CARRY 時 T_carry_rel＝該終點。
  不變：窗口與越窗停止、最終離桌 ≥40 mm 連續 1 s、抓持、滑移、禁止接觸、淨空、CBF、Shield、放置判準。這是物體附著狀態改變後的任務參考切換，不是控制器交棒。
"""
import dataclasses

import numpy as np

import bottle_ref as BR
import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
import bottle_w1_ctrl as W1

TCP_DROP_MAX = 0.00051
G_SWITCH = 0.005 + TCP_DROP_MAX + S1.CLEAR_REJECT + S1.CLEAR_GUARD
TABLE_F_TOL = 0.02


@dataclasses.dataclass
class BottleTaskB(BR.BottleTask):
    T_lift_ov: np.ndarray | None = None

    @property
    def T_lift(self):
        return self.T_lift_ov if self.T_lift_ov is not None else BR.lift_pose(self.T_grasp, self.lift_m)


def base_mat(b):
    return np.linalg.inv(BR.base_inv(b))


class W1PlaceController(W1.W1PlaceController):
    group = 'G_LIFTB'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        t = self.task
        self.task = BottleTaskB(**{f.name: getattr(t, f.name) for f in dataclasses.fields(BR.BottleTask)})
        self.lb, self.n_sw = None, 0
        self.lb_log = []

    def _liftb_on(self):
        return self.lb is not None and self.st.phase == 'LIFT'

    def step(self, t, q, **kw):
        q = np.asarray(q, float)
        if self.lb is None and self.st.phase == 'LIFT' and self.judge.t_grasp is not None and self.judge.stop is None:
            Tb, ft = kw.get('T_bottle'), kw.get('f_table')
            ok = (Tb is not None and np.isfinite(Tb).all() and ft is not None and np.isfinite(float(ft)) and abs(float(ft)) <= TABLE_F_TOL
                  and self.judge.bottle_bottom_gap(Tb) >= G_SWITCH)
            self.n_sw = self.n_sw + 1 if ok else 0
            if self.n_sw >= C0.NPH:
                T_tcp = self.K.fk(q, 'link_tcp')
                T_rel = BR.base_inv(q[:3]) @ T_tcp
                z_fin = float(BR.lift_pose(self.task.T_grasp, self.task.lift_m)[2, 3])
                T_fin = T_rel.copy()
                T_fin[2, 3] = z_fin
                self.lb = dict(t=float(t), T_rel=T_rel, z0=float(T_rel[2, 3]), z_fin=z_fin, T_fin=T_fin, gap=float(self.judge.bottle_bottom_gap(Tb)),
                               lift_elapsed=float(t - self.st.t_phase), x=float(q[0]), y=float(q[1]))
                self._ev('liftb_switch', gap=round(self.lb['gap'], 4), z0=round(self.lb['z0'], 4), z_fin=round(z_fin, 4), lift_elapsed=round(self.lb['lift_elapsed'], 3))
        if self._liftb_on():
            self.task.T_lift_ov = base_mat(q[:3]) @ self.lb['T_fin']           # CARRY 判定與承接都用底盤座標終點
        out = super().step(t, q, **kw)
        if self._liftb_on() and self.n_step % C0.NPH == 0:
            e = BR.pose_err(self.K.fk(q, 'link_tcp'), self.task.T_lift)
            self.lb_log.append((round(float(t), 3), round(e[0], 5), round(e[1], 5)))
        return out

    def _rel_task(self):
        return self._liftb_on() or super()._rel_task()

    def _T_seq(self, q, t_eff):
        if not self._liftb_on():
            return super()._T_seq(q, t_eff)
        L = self.lb
        T = []
        for k in range(B.N + 1):
            s = BR.min_jerk((t_eff + k * B.DT - L['t']) / max(self.task.t_lift, 1e-6))
            Tk = L['T_rel'].copy()
            Tk[2, 3] = L['z0'] + s * (L['z_fin'] - L['z0'])
            T.append(Tk)
        T = np.array(T)
        return np.stack([T[0]] * (B.N + 1)) if self.no_tcp_preview else T

    def report(self):
        r = super().report()
        r['paired_strategy'] = dict(name='G_W1_lift_transport', g_switch=G_SWITCH, switch=None if self.lb is None else {k: v for k, v in self.lb.items() if k not in ('T_rel', 'T_fin')},
                                    lb_err_log=self.lb_log[-60:],
                                    scope='凍結 G；抓持且離桌可靠（間隙 ≥ G_SWITCH、桌面無受力、連續一控制週期）後 LIFT 改為底盤座標運送式抬升；窗口與判準不變')
        return r
