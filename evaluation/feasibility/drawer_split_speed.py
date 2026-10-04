"""兩邊同時出力能快多少：把 200 mm 行程拆給底盤與手臂。

TCP 速度 = J_base·u_base + J_arm·u_arm ⇒ 兩邊同時動，速度是**相加**的。
底盤是慢的那一邊（0.035255 m/s，由低速介面界限推得），所以把一部分行程
交給手臂就能直接縮短歷時。代價是手臂不再維持定姿態（見
drawer_speed_effort_tradeoff.py 的取捨曲線）。

**本檔末段那些「底盤解到 N mm/s」只是算術推演，不是建議。** 2026-10-03 的
裁定是暫不放寬速度框：低速界限是軟體保守值，**不是因為輪級功能漏做** ——
wb_cmd_chain_e2 的順序是 wheel_ok（低速界限，越界閂鎖）→ limit9（輪速與輪
加速度限制，以 λ 修改完整九維增量）→ _apply，兩層都在。要提速是另立較高速
配置的獨立工作，還要核對 λ 修改後命令的避碰與夾持效果。

**修正紀錄**：先前把單趟歷時下界寫成 5.7 s，那是假設底盤獨自走完 200 mm。
守住「底盤沿軸 ≥0.10 m」的閘門後，下界是 2.84 s。
"""
import sys
import numpy as np
sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics as WK
from ammr_wholebody_mpc.wgmpc_core import WGMPCConfig

K = WK.from_urdf_file('evaluation/models/omni_bot_wholebody_expanded.urdf')
R = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
cfg = WGMPCConfig()
V_BASE, V_ARM = cfg.v_base_lin, cfg.v_arm
BX, BYAW, BY0 = -0.136412, 1.297349, 0.56
Q_HOLD = np.array([-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813])
N_DIR = np.array([0.0, -1.0, 0.0])        # 開啟方向：世界 −y
# **座標提醒**：底盤三分量是**本體座標**（adapter 已轉換）。世界 −y 在停位
# yaw 1.297349 下換到本體是 (−0.9628, −0.2701)，幾乎是本體 −x。輪矩陣的
# 可行上限要用換算後的方向算：0.2882 m/s，不是本體 +y 的 0.2775。

q = np.concatenate([[BX, BY0, BYAW], Q_HOLD])
J = K.jacobian(q, 'link_tcp')[:3]
# 手臂單獨能在 −y 方向產生多快的 TCP 速度（各軸獨立取 ±V_ARM）
row = N_DIR @ J[:, 3:]
v_arm_max = float(np.abs(row).sum() * V_ARM)
row_b = N_DIR @ J[:, :3]
print(f'求解器上限：底盤線速度 {V_BASE:.6f} m/s、手臂關節 {V_ARM:.4f} rad/s')
print('夾持姿態下：')
print(f'  手臂單獨在 −y 的最大 TCP 速度 {v_arm_max:.4f} m/s'
      f'（= {v_arm_max/V_BASE:.1f} 倍底盤上限）')
print(f'  各軸對 −y 的貢獻 (m/s per rad/s): {np.round(row, 4)}')
print(f'  底盤三維對 −y 的貢獻: {np.round(row_b, 4)}')
print('  **注意**：這是該姿態的瞬時線性化上界，不等於整段都維持得住 ——'
      ' 手臂走出去後姿態會變，貢獻跟著變。')

print('\n=== 把 200 mm 拆給兩邊：歷時與代價 ===')
print(f'{"底盤分擔":>8}{"底盤走":>8}{"手臂走":>8}{"底盤需時":>9}'
      f'{"手臂需時":>9}{"整段歷時":>9}{"對比全交底盤":>12}')
T_ALL_BASE = 0.200 / V_BASE
for frac in (1.00, 0.75, 0.50, 0.30, 0.10, 0.05):
    db, da = 0.200*frac, 0.200*(1-frac)
    tb = db / V_BASE
    ta = da / v_arm_max if v_arm_max > 0 else float('inf')
    t = max(tb, ta)
    print(f'{frac*100:>7.0f}%{db*1e3:>7.0f}mm{da*1e3:>7.0f}mm'
          f'{tb:>8.2f}s{ta:>8.2f}s{t:>8.2f}s{T_ALL_BASE/t:>11.1f}倍')
print(f'\n底盤位移閘門 0.10 m ⇒ 底盤至少分擔 50% ⇒ '
      f'歷時下界 {0.10/V_BASE:.2f} s（不是 5.7 s）')

print('\n=== 兩條加速途徑的算術推演（**不是建議**，本輪裁定暫不放寬） ===')
print(f'{"做法":<34}{"歷時":>8}{"手臂行程":>10}{"相對現況":>10}')
print(f'{"現況：全交底盤 35 mm/s":<34}{T_ALL_BASE:>7.2f}s'
      f'{0.0001:>9.4f}r{1.0:>9.1f}倍')
print(f'{"拆一半給手臂（守住 0.10 m 閘門）":<34}{0.10/V_BASE:>7.2f}s'
      f'{1.4562:>9.4f}r{T_ALL_BASE/(0.10/V_BASE):>9.1f}倍')
for v in (0.10, 0.20, 0.30):
    t = 0.200/v
    print(f'{f"底盤解到 {v*1e3:.0f} mm/s、手臂仍不動":<34}{t:>7.2f}s'
          f'{0.0001:>9.4f}r{T_ALL_BASE/t:>9.1f}倍')
print(f'{"底盤 300 mm/s ＋ 再拆一半給手臂":<34}{0.10/0.30:>7.2f}s'
      f'{1.4562:>9.4f}r{T_ALL_BASE/(0.10/0.30):>9.1f}倍')
print('\n導航段 2 m 的影響（底盤單獨，RTF 0.579）')
for v in (V_BASE, 0.10, 0.20, 0.30):
    print(f'  {v*1e3:>5.0f} mm/s ⇒ {2.0/v:>6.1f} s 模擬、'
          f'{2.0/v/0.579:>6.1f} s 牆鐘')
