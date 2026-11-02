#!/usr/bin/env python3
"""在線 MotM 第四版：抓取段進度參考速度上限 0.10 → 0.12 m/s（Codex 20261102_082423 唯一抓取提速候選；版本化子類，基線＝bottle_w1_online3_ctrl，其餘全部相同）。
只改「漸變開始後、運送段之前」（漸變、預抓、插入、閉爪、抬升）啟發式進度調速的上限：heuristic_speed(..., V_C_GRASP)。
不動：v_c（0.10）推導的幾何排程（x_blend＝x_pre − v_c·t_blend、x_unfold、t_pre）、漸變前慢速接近段（x_slow→x_blend 仍 v_c）、
參考速度斜率限制（_slew）、GMPC、硬體速度／加速度上限、插入與閉爪時長、證據持續、窗口、全部閘與停止、滑移警告。
0.12 是上限，不強制維持：調速仍依剩餘窗口與任務所需時間只減速不加速。
"""
import bottle_s0b_ctrl as B
import bottle_w1_online3_ctrl as ON3

C3, C2 = ON3.C3, ON3.C2
V_C_GRASP = 0.12


class W1PlaceController(ON3.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN_ONLINE4_VC012'

    def _v_target(self, T_tcp, x):
        if self.blending and self.st.phase != 'CARRY':
            return B.heuristic_speed(self.task, self.st, self.t, T_tcp, x, self.x_we, V_C_GRASP)
        return super()._v_target(T_tcp, x)

    def report(self):
        r = super().report()
        r['online']['grasp_speed'] = dict(v_c_geometry=self.v_c, v_c_grasp_cap=V_C_GRASP, note='只改漸變後到運送前的進度調速上限；幾何排程仍由 v_c 推導')
        return r
