#!/usr/bin/env python3
"""W1 同任務的反應式單步全身 QP 移植基線 M（版本化，2026-11-01；reviews/20261101_howard_continue_physical_experiments_motm.md §C）。
來源：bottle_s1_groups.S1MotM（motm_b1_port＋wg4b_b1_core 單步 QP），只換「朝瓶漸變 → 預抓 → 插入 → 閉爪 → 抬升」的控制；
導航、視覺（底盤凍結＋手腕修正、閉爪閘、期限）、物理／延遲、命令限制、Shield、停止與淨空核對、持瓶規劃與第二桌放置全部沿用 W1（MRO：S1MotM → W1PlaceController → … → S1Controller）。
與 G 不等價處（照實揭露，不是只差「是否 MPC」）：
  - CBF 只有當下一步、沒有 slack，不可行即走停止路徑（G：全時域預測 CBF＋逐步 slack）。
  - 無時域 TCP 參考與設定點模型預測，任務速度＝參考前饋＋kp·誤差（G：時域 TCP 參考＋設定點模型＋延遲補償的耦合 QP）。
  - 姿態代價：朝 W1 備妥姿態的阻尼項（mu_post）；原 MotM 的可操作度最大化、關節限位避讓原式、PBS 切換門檻未重現。
  - 手腕修正的施加目標每拍更新，但 M 只看當拍目標，不吃 W1 _T_seq 的逐拍預測。
可調參數（事前宣告，至多 2 個）：kp（任務增益）、mu_post（姿態阻尼）。舊 S1 M 凍結值 kp 4.0、mu_post 0.01（bottle_s1_m1：finger_bottle_preclose 失敗，保留）。
不稱 MotM 原方法重現；稱「受文獻啟發的反應式單步全身 QP 移植基線」。
"""
import bottle_s1_groups as GR
import bottle_w1_ctrl as W1

KP0, MU0 = 4.0, 0.01


class W1PlaceController(GR.S1MotM, W1.W1PlaceController):
    group = 'M'

    def __init__(self, *a, kp=KP0, mu_post=MU0, **kw):
        super().__init__(*a, kp=kp, **kw)
        self.p1.mu_post = float(mu_post)

    def report(self):
        r = super().report()
        r['paired_strategy'] = dict(name='M_W1_reactive_one_step', kp=float(self.p1.kp), mu_post=float(self.p1.mu_post), n_m=self.n_m,
                                    scope='同 W1 任務；只換朝瓶漸變到抬升的控制為單步全身 QP（當下一步 CBF、無 slack）；不是 MotM 原方法重現')
        return r
