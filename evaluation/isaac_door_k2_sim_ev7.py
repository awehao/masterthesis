#!/usr/bin/env python3
# EV7 版（K3-B 受控釋放執行端；Codex 20261026_000706；設計 K3B_impl_design_draft1.md，規格 K3B_release_clear_spec_draft2.md）：ev6r3 的複本，**只做插入**。
#   --w2-release（需 --w2-mode-run-id 與 --pause-finger-max-effort > 0；預設關 ⇒ 與 ev6r3 同行為）：
#   (1) 權威改 door_w2_mode5.ModeAuthority5：模式 0→1→2→3；mode 3 准入＝執行端 ExecReleaseMonitor.confirmed_fresh（請求時與提交當下）。
#   (2) mode 2 提交步＝t_open：同一步依權威模式切換手指目標為張開（finger_target）；mode ≥ 1 時手指命令由權威模式決定，任務端夾爪命令只作參考；
#       K2 閂鎖後照既有凍結（不張爪、不閉爪）。
#   (3) 逐物理步 ExecReleaseMonitor（door_k3b_release2）：本步兩指開度、兩指—橫桿接觸（bar＋ambiguous）、接觸資料有效、門角；
#       失敗 ⇒ 已知中止（release_monitor:<why>）⇒ 權威中止＋既有 K2 閂鎖路徑。
#   (4) 5 N 恢復權威（door_k3b_release2.EffortRestoreAuthority）：/door/finger_effort_req（序號、mode 3、任務端 C_CLEAR 證據）→ 執行端以逐步資料
#       再核（mode 3、確認後靜窗 ≥ 0.5 s、開度 ≥ 95 %、期限 0.5 s）→ 物理步 door_k3b_finger_restore.restore（兩指 20→5 N、讀回、非手指不變）→
#       /door/finger_effort_ack；讀回失敗走既有中止；中止優先。
#   (5) 權威狀態另帶 release（狀態、t_conf、步）與 effort（ack）；room_run.json w2 另記 release、effort。
#   注意：本步手指開度取物理步前讀值（與 room 紀錄列相同），接觸力取本步物理步後讀值。
# EV6 r3 版（Codex 20261025_223632；規格 W2c_run_pack_spec.md §9）：r2 全部＋room_run.json w2.config（w2 run_id、run_id、run_id_num 與權威參數；**只加紀錄**）。
# EV6 r2 版（Codex 20261025_220903；規格 W2c_run_pack_spec.md）：r1 全部＋(5) 權威 r4（door_w2_mode4：提交後以最後有效套用證據計時，0.25 s 無有效套用 ⇒ 中止）；
#   (6) 既有中止同步進權威：每物理步先彙整 K2 閂鎖（含入口違規、幾何／任務端 /door/k2_abort、走廊、手指讀回）、PARK／K3-M 違規為 known_abort，
#       經 door_w2_exec2.W2Exec2.step_begin **先於**請求與 select 同步 auth.abort；本步 select 之後才發現者以 sync_after 照實記在本步；
#   (7) 逐步列分列選定筆的來源時間 sel_stamp 與實際接收時刻 sel_t_recv。以下為 r1 說明：
# EV6 版（W2 保持／待交接的執行端模式權威；Codex 20261025_214144；規格 W2b_exec_wiring_spec.md）：isaac_door_k2_sim_ev5.py 的複本，**只做插入**：
#   --w2-mode-run-id（預設空 ⇒ 與 EV5 同行為；需 --cmd-env、--run-id）：
#   (1) 唯一入口改為 v3 封裝 /wb_vel_cmd_env3（cmd_envelope_v3，含 mode_seq）；v2 /wb_vel_cmd_env 收到只計數、不進鏈。
#       回呼只解碼並排入有界佇列（另存接收牆鐘，不覆寫來源 stamp_sim_t）；每物理步在主迴圈依序交 door_w2_exec.W2Exec.on_env3。
#   (2) 模式請求：任務端 /door/w2_mode_req_task → ModeAuthority3.on_request；登記為 pending 後才轉送求解端 /door/w2_mode_req（latched）。
#   (3) 每物理步 select（先核請求期限、以原始 stamp 判新鮮）→ EnvIntake 進 E2 鏈（鏈接收時刻＝來源 stamp，K2 過濾照舊）→ ex.step
#       → applied_source／限幅後九軸 → confirm_step；權威中止、入口違規 ⇒ 既有 K2 閂鎖中止（有界停止、不回切、不張爪）。
#   (4) 每 2 物理步發布權威狀態 /door/w2_mode_state；room_run.json 另記 w2（權威統計、逐步列、權威確認列、接收紀錄）。
#   測試模式 RELEASE_PREP_TEST 與 HELD 命令相同（不張爪、不退離、不增加角度）；速度、加速度、輪級、關節限制與 92 °C 熱中止不變。
# EV5 版（W1e：暫停期提高手指驅動輸出上限；Codex 20261025_023608，規格 W1e_pause_finger_effort_spec.md）：isaac_door_k2_sim_ev4.py 的複本，**只做插入**：
#   --pause-finger-max-effort N（預設 0 ⇒ 與 EV4 同行為，只多記一筆啟動讀回）：執行端暫停切換那一步（world.step 之前）把兩指 finger_joint1／2 的
#   有效驅動輸出上限（ArticulationController.set_max_efforts）設為 N 並讀回，之後每 1 s 讀回核對；閉合目標、kp／kd、摩擦、手臂／底盤控制與判準不變。
#   紀錄 room_run finger_effort：啟動讀回、旗標值、暫停切換時的 old／set／readback／ok 與定期讀回（door_k3m_finger_effort）。需 --park-gate-mobile。
#   r2（Codex 20261025_024339 必修①③）：啟動讀回缺失／非法／不是兩指各 --pause-finger-original-effort（5）⇒ 啟用時拒跑（38）；
#   切換設定／讀回失敗、非有限、不符或非手指上限被改 ⇒ 保存原因、走既有 K2 中止閂鎖（/door/k3m_abort），暫停 ack 改回覆拒絕；
#   定期讀回失敗或被改回 ⇒ 同樣中止。不自動重試、不加力、不放爪；原停止與保護門檻不變。切換前後記完整 DOF 上限向量與名稱。
# EV4 版（K3-M 受控暫停接入；規格 K3M_pause_integration_spec_draft1.md）：isaac_door_k2_sim_ev3.py 的複本。變更：
#   (1) --cmd-env：求解端命令只經 adapter 封裝 /wb_vel_cmd_env（進鏈時以 E2 鏈 recv_seq 對照封裝 source_seq）；
#       /wb_vel_cmd 照舊保留給展開節點（UNFOLD；求解端在封裝模式不走這條）。r2：第一筆封裝進鏈＝求解端接管，之後舊路徑命令**不進鏈**並走既有閂鎖中止；
#       封裝進鏈前核對本趟 run_id、段別（adapter）、種類（normal）、derived 與序號（同序同值冪等、同序異值或倒序 ⇒ 違規中止）；
#   (2) 每個實際物理步發布 /door/applied_src：[step, sim_t, source_seq（非正常套用／非全身控制 ⇒ −1）, 鏈模式（0 正常、1 逾時、2 停止、3 失效）,
#       底盤本體 3, 手臂 6]，並寫入 room steps 新欄 applied_src＝[step, sim_t, source_seq, 鏈模式]；
#   (3) 受控暫停握手的執行端權威（door_k3m_pause_mode.ExecutorPauseAuthority；/door/pause_req → /door/pause_ack_exec）；
#       走廊與包絡監看在暫停期間照常（不改）。
# EV3 版（K3-M）：isaac_door_k2_sim_ev.py 的複本，**只做插入**：--park-gate-mobile（預設關 ⇒ 與 EV 版逐字同行為）。開啟時（需 --park-hold）：
# HOLD 前照 PARK_HOLD；收到任務端 /door/base_mode_req 後由本端（模式權威，door_k3m_mode.ExecutorModeAuthority）在明確物理步切換：
# 關閉 HoldMonitor（保存至該步判定）、啟動逐物理步的走廊監看（door_k3m_corridor，內側帶）與粗略活動上限，回覆 /door/base_mode_ack；
# 越界 ⇒ 本端立即 K2 中止閂鎖（鏈內命令過期、全身命令濾掉、夾爪凍結）並發布 /door/k2_abort 與 /park/violation。
# EV 版（取證）：isaac_door_k2_sim.py 的複本，**只做插入**：--evidence-out PATH（預設空 ⇒ 與原版逐字同行為、不讀任何新 API）。
# 給定時逐物理步把手指／夾爪殼等剛體對門扇的逐點接觸（法向力、點、法向、分離）、切向摩擦、把手 prim 位姿、
# joint6 反作用旋量（6 維，非馬達輸出）、DOF 方向量測關節力（非馬達命令）、手臂設定點寫入 JSONL。不改任何控制或判定。
"""【K2】局部通行門的固定停位對準與夾持執行端（2026-10-21）。isaac_drawer_room_sim.py（凍結，不改）的複本。

規格：evaluation/results/specs/K2_door_grasp_spec.md（draft-2＋draft-3）。與原檔的差異：
  * 場景：door_local.sdf（只有地面）＋ door_asset_v2 建門（door_local_v2.yaml）；不建抽屜。
  * 機器人生成在候選停位（--start-pose），手臂起始 K2_READY（--stow-q；停車閘門以它為參考）。
  * 發布 /door/state、/door/handle_pose、/door/sync_state（29 欄：t, step, q, H(16), x, y, yaw, q_arm(6), q̇）、
    /door/contacts（每 50 Hz 一筆，內容為該窗口內逐物理步的最大值：手指—橫桿每指、非預期接觸最大值與來源、牆接觸、腕力）。
  * **非預期接觸量測**：門扇剛體 ↔ 機器人各剛體的接觸配對，依每個接觸點到橫桿軸的距離分成「橫桿」與「門板／立柱」；
    只有 finger1／finger2—橫桿是允許接觸（是否在允許相位由任務節點判斷）；機器人各剛體 ↔ 兩側牆另以每剛體一個視圖量測。
  * **腕力**：robot.get_measured_joint_forces() 的 joint6 列（同 isaac_drawer_sim.py 做法），逐物理步記錄與發布。
  * 移除：抽屜、重播渲染、腕部相機擷取、地圖輸出（K2 不用）。
其餘（雙來源執行層、命令鏈、交棒、PARK_HOLD 閘門與保持監看、夾爪、逐物理步證據、受控停止）照原檔。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from physics_contact_hold import PhysicsContactHold

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--world', default=os.path.join(
    WS, 'src/ammr_bringup/worlds/door_local.sdf'))
ap.add_argument('--door-asset', default=os.path.join(
    WS, 'src/my_omnibot_description/config/door_local_v2.yaml'))
ap.add_argument('--urdf', default=os.path.join(
    WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--start-pose', default='0.6,-0.5332,1.570796',
                help='K2 候選停位（前距 0.50、側向 −0.20；door_k2_offline_v2.base_of(0,0,0)）')
# ---- V0 腕部 RGB-D 擷取（預設關閉；開啟時只擺位＋算圖，不跑控制、不建既有 ROS 節點）----
ap.add_argument('--wrist-v0', action='store_true',
                help='V0：腕部相機擷取模式（evaluation/results/vision/V0_spec.yaml）')
ap.add_argument('--wrist-res', default='640x480')
ap.add_argument('--wrist-hfov', type=float, default=69.0)
ap.add_argument('--wrist-frames', type=int, default=20)
ap.add_argument('--wrist-warmup', type=int, default=60)
ap.add_argument('--wrist-max-steps', type=int, default=2000)
ap.add_argument('--wrist-hz', type=float, default=5.0,
                help='腕部相機取樣頻率（模擬時間）；存檔與發布都依此頻率')
ap.add_argument('--wrist-replay', default=None,
                help='重播既有實錄（room_run.json）的一段運動來擷取動態腕部影像（不跑控制）')
ap.add_argument('--wrist-replay-t0', type=float, default=0.0)
ap.add_argument('--wrist-live', action='store_true',
                help='D1 S4：與控制同跑的腕部深度擷取＋發布（D1_S4_online_plan.md；預設關，關時不進入）')
ap.add_argument('--wrist-replay-t1', type=float, default=1e9)
ap.add_argument('--wrist-base', default='-0.136412,0.560,1.297349',
                help='擺位底盤 x,y,yaw（預設＝基準停位）')
ap.add_argument('--wrist-q', default='-0.383712,0.301253,0.428917,-0.661155,-1.469862,-1.492557',
                help='擺位手臂六軸（預設＝基準夾持姿態的 TCP 沿 −y 退 0.20 m，離線 IK；'
                     '真值橫桿中心投影約 (375,354)、光軸距離 0.275 m）')
ap.add_argument('--stow-q', default='0,0,0.035,0,-1.5707963,0',
                help='K2_READY（K2 規格 draft-3：其餘同 STOW、j3 = 0.035）；起始姿態與停車閘門的參考')
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--sim-limit', type=float, default=240.0)
ap.add_argument('--evidence-out', default='',
                help='（EV 版）逐物理步接觸承載取證 JSONL 路徑；空字串＝不取證（預設，與原版同行為）')
ap.add_argument('--scan-rate', type=float, default=10.0)
ap.add_argument('--state-rate', type=float, default=50.0)
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--arm-rate-max', type=float, default=1.0)
ap.add_argument('--base-lin-max', type=float, default=0.05,
                help='**低速介面界限**（m/s）。只套用在**全身**那條路徑；'
                     '導航不經命令鏈，故不受它限制')
ap.add_argument('--base-ang-max', type=float, default=0.20)
ap.add_argument('--max-cmd-age-s', type=float, default=0.2)
ap.add_argument('--joint-margin', type=float, default=0.05)
ap.add_argument('--cmd-timeout', type=float, default=1.0,
                help='導航命令逾時（模擬秒）：超過就歸零，**失效處置**')
ap.add_argument('--out', default=os.path.join(WS, 'evaluation/runs/room_sim'))
ap.add_argument('--no-frictionless', action='store_true',
                help='診斷用：不綁零摩擦材質。**不綁會偏航** —— '
                     '底盤被直接設速度，支撐球與地面的摩擦抵抗橫向運動就會'
                     '產生淨力矩（實測 0.25 m/s 驅動 5 s 偏 21.1°）')
ap.add_argument('--v-box-lin', type=float, default=0.035255,
                help='切換當步要核的底盤速度框（逐軸）')
ap.add_argument('--v-box-ang', type=float, default=0.199900)
ap.add_argument('--v-min-lin', type=float, default=0.010,
                help='**滾動交棒下界**：切換當步底盤至少要這麼快。0 = 不要求')
# **PARK_FIXED（固定底盤操作）**：預設關 ⇒ 既有行為不變。開啟時執行端做靜止閘門（轉給全身前）與
# 保持監看（實際轉給全身 → 實際交還導航），並取消滾動交棒下界（MOTM 保持原值）。見 park_fixed.py。
ap.add_argument('--park-fixed', action='store_true')
ap.add_argument('--park-hold', action='store_true',
                help='PARK_HOLD（v2）：閘門／監看同 PARK_FIXED，保持期的套用命令改核停車伺服上界')
ap.add_argument('--park-gate-mobile', action='store_true',                         # EV3（K3-M）
                help='K3-M：PARK_HOLD 閘門照舊；模式握手後改走廊監看（需 --park-hold）')
ap.add_argument('--run-id', default='')                                           # EV3：握手用本趟識別
ap.add_argument('--corridor-table', default='')                                   # EV3：走廊表 JSON（door_k3m_corridor_table.json）
ap.add_argument('--corridor-inner-mm', type=float, default=20.0)                  # EV3：內側中止帶（候選）
ap.add_argument('--corridor-inner-deg', type=float, default=0.8)
ap.add_argument('--envelope', default='-0.15,0.02,-0.05,0.10,6.0',                # EV3：粗略活動上限 前進lo,hi,側向lo,hi,|偏航|°
                help='K3-M 粗略活動上限（相對錨點座標）')
ap.add_argument('--park-pose', default='0.6,-0.5332,1.570796',
                help='PARK_FIXED 共同停位 x,y,yaw（入場容差的參考）')
ap.add_argument('--park-stow-tol', type=float, default=0.02,
                help='PARK_FIXED 閘門：手臂各軸距收攏姿態的上限（rad）')
# **底盤命令變化率上限**（逐軸）。既有的兩層都攔不住交棒當步的跳變：
# 低速介面界限只看命令大小，輪級 λ 的 α_max=125 rad/s² 允許 6.25 m/s²。
# 實測交棒當步跳 48.76 mm/s＝4.88 m/s²，λ 沒攔。這裡**套用既有值**
# a_base_lin 0.50 / a_base_ang 2.00（wgmpc_core.py 標為「開發值，
# 無已核准來源」，標籤照搬）。設 0 或負數 = 關閉。
ap.add_argument('--base-accel-lin', type=float, default=0.50,
                help='底盤線速度命令的逐軸變化率上限 [m/s²]；<=0 關閉')
ap.add_argument('--base-accel-ang', type=float, default=2.00,
                help='底盤偏航率命令的變化率上限 [rad/s²]；<=0 關閉')
# **導航直寫路徑的逐軸變化率上限。** 那條路徑不經命令鏈，鏈上三層都沒套到。
# 實測單一物理步逐軸跳 75.000 mm/s＝ax_max(1.5)×dt_cfg(0.05)，而 gmpc 自己
# 量到的實際週期是 0.1800 s ⇒ 那是 7.5 m/s² 的脈衝。這裡套用**導航自己的**
# 既有值（gmpc_node.py 宣告的 ax/ay/az_max），按真實物理步長執行。<=0 關閉。
ap.add_argument('--nav-accel-x', type=float, default=1.5)
ap.add_argument('--nav-accel-y', type=float, default=1.0)
ap.add_argument('--nav-accel-w', type=float, default=2.0)
# ---- 第三人視角錄影（預設關閉，開著會大幅拖慢 RTF）----------------------
# 視角沿用 isaac_drawer_room_snapshot.py 的 `06_route`：同時看得到起點、
# 路線與抽屜櫃。規格要求影片**同時看到機器人與抽屜**，相位與開度由**同一趟**
# 的紀錄標註，不另外配音軌。
ap.add_argument('--cam', default='false')
# ---- 夾爪（沿用 isaac_drawer_sim.py 的規格）----
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--finger-open', type=float, default=0.0089,
                help='手指張開位置（URDF 上限 0.0089 m）')
ap.add_argument('--finger-closed', type=float, default=0.0,
                help='手指閉合目標（URDF 下限 0）')
ap.add_argument('--finger-collision', default='hull', choices=['hull', 'split'],
                help='hull = 匯入時的單一凸包（既有行為）；split = 根部／指片兩個'
                     '凸塊，保留真實的指片開口（finger_collision.py）')
ap.add_argument('--bar-class-tol', type=float, default=0.006,
                help='接觸分類容差（door_k2_io.classify_contacts）：橫桿自由段且徑向 ≤ 半徑＋此值 ⇒ bar；支柱 u 範圍或超出橫桿端 ⇒ ambiguous')
ap.add_argument('--park-dev-pos-max', type=float, default=0.020,
                help='K2 登錄的保持期相對錨點位置上限（m；K2 規格 draft-2 §F）——取代 ParkCfg 預設 1 mm')
ap.add_argument('--park-dev-yaw-max-deg', type=float, default=2.0,
                help='K2 登錄的保持期相對錨點偏航上限（度）——取代 ParkCfg 預設 0.5°')
ap.add_argument('--contact-min-n', type=float, default=0.5,
                help='逐物理步連續接觸計時的每指門檻（N）')
ap.add_argument('--pub-tf', default='false',
                help='發 odom→base_footprint 的 TF（Phase 1 感知鏈需要）')
ap.add_argument('--map-out', default=None,
                help='在光達高度逐格做重疊查詢，輸出 pgm/yaml 靜態地圖後結束')
ap.add_argument('--map-res', type=float, default=0.05)
ap.add_argument('--map-x0', type=float, default=-4.30)
ap.add_argument('--map-x1', type=float, default=4.30)
ap.add_argument('--map-y0', type=float, default=-5.30)
ap.add_argument('--map-y1', type=float, default=2.00)
ap.add_argument('--replay-t0', type=float, default=0.0,
                help='重播只渲染這段模擬時間（起）')
ap.add_argument('--replay-t1', type=float, default=1e9,
                help='重播只渲染這段模擬時間（訖）')
ap.add_argument('--replay', default=None,
                help='重播已封存的 room_run.json，只渲染不跑物理')
ap.add_argument('--cam-hz', type=float, default=10.0)
ap.add_argument('--cam-res', default='1280x720')
ap.add_argument('--cam-eye', default='-1.4,-7.4,4.2')
ap.add_argument('--cam-at', default='-1.2,-1.0,0.3')
ap.add_argument('--cam-foc', type=float, default=20.0)
ap.add_argument('--grip-mu-static', type=float, default=1.0,
                help='夾持面（兩指與把手橫桿）的**靜摩擦**。'
                     '**執行前選定並記錄**，不靠 PhysX 預設')
ap.add_argument('--grip-mu-dynamic', type=float, default=0.8,
                help='夾持面的動摩擦')
ap.add_argument('--allow-friction-mismatch', action='store_true',
                help='摩擦核對未通過仍繼續（**診斷用**；結果必須標明）')
ap.add_argument('--cmd-env', action='store_true',                                  # EV4：封裝入口（擇一）＋套用來源回報
                help='（EV4）只收 adapter 封裝 /wb_vel_cmd_env（不訂 /wb_vel_cmd）；逐物理步發布 /door/applied_src')
ap.add_argument('--pause-finger-max-effort', type=float, default=0.0,             # EV5
                help='（EV5）暫停切換那一步起兩指有效驅動輸出上限（N）；0＝不改（URDF effort 5）')
ap.add_argument('--pause-finger-original-effort', type=float, default=5.0,        # EV5 r2
                help='（EV5）啟動讀回應有的兩指原上限（URDF effort）；啟用時不符即拒跑')
ap.add_argument('--w2-mode-run-id', default='',                                    # EV6
                help='（EV6）W2 模式權威 run_id；空＝關閉（EV5 行為）。需 --cmd-env、--run-id')
ap.add_argument('--w2-cand-timeout-s', type=float, default=1.0)                    # EV6：候選期限（設計值，不改）
ap.add_argument('--w2-post-commit-timeout-s', type=float, default=0.25)            # EV6
ap.add_argument('--w2-release', action='store_true',                               # EV7
                help='（EV7）K3-B 受控釋放：模式 2／3、逐步解除監看、5 N 恢復權威（需 --w2-mode-run-id 與 --pause-finger-max-effort）')
ap.add_argument('--headless', default='true')
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import (FixedCuboid,              # noqa: E402
                                       FixedCylinder, GroundPlane)
from isaacsim.core.prims import SingleArticulation               # noqa: E402
from isaacsim.core.prims import RigidPrim as _RigidPrim         # noqa: E402
from isaacsim.core.utils.types import ArticulationAction         # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics                  # noqa: E402

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
import door_asset_v2 as DA                                       # noqa: E402
import door_geom_v2 as DG                                        # noqa: E402
import door_k2_io as KIO                                         # noqa: E402
import door_k3_evidence as KEV                                   # noqa: E402  EV 版：取證擷取（純邏輯）
import time as _ev_time                                          # noqa: E402  EV 版：取證耗時
import door_k3m_mode as K3MMODE                                  # noqa: E402  EV3：模式握手（純邏輯）
import door_k3m_pause_mode as K3MP                               # noqa: E402  EV4：暫停握手（純邏輯）
import wgmpc_cmd_envelope as ENV                                 # noqa: E402  EV4：命令追蹤封裝
import door_k3m_applied_src as K3MSRC                            # noqa: E402  EV4：封裝進鏈與套用來源（純函式）
import door_k3m_corridor as K3MCOR                               # noqa: E402  EV3：走廊（純函式）
import door_k3m_finger_effort as FEFF                            # noqa: E402  EV5：暫停期手指驅動輸出上限（純邏輯）
import cmd_envelope_v3 as E3                                     # noqa: E402  EV6：v3 封裝（mode_seq）
import door_w2_mode4 as W2M4                                     # noqa: E402  EV6 r2：模式權威 r4
import door_w2_exec2 as W2X2                                     # noqa: E402  EV6 r2：執行端接線 r2（與假世界共用）
import door_w2_mode5 as W2M5                                     # noqa: E402  EV7：多段模式權威
import door_k3b_release2 as K3BR                                 # noqa: E402  EV7：解除監看與 5 N 恢復權威
import door_k3b_finger_restore as K3BF                           # noqa: E402  EV7：手指上限恢復
import collections as _w2_col                                    # noqa: E402  EV6
from isaac_common import (bind_friction, bind_frictionless,     # noqa: E402
                          collision_prims, import_urdf,
                          physics_parts, walk)
from control_authority import (AUTH_GLIDE, AUTH_NAV,            # noqa: E402
                               AUTH_WHOLEBODY)
from dual_source_executor import DualSourceExecutor              # noqa: E402
from park_fixed import HoldMonitor, ParkCfg, StaticGate            # noqa: E402
from wb_cmd_chain_e2 import CmdChainE2                           # noqa: E402
from wb_wheel_limit import WheelLimitConfig                      # noqa: E402

import rclpy                                                     # noqa: E402
from rclpy.node import Node                                      # noqa: E402
from rclpy.qos import (DurabilityPolicy, QoSProfile)             # noqa: E402
from geometry_msgs.msg import TransformStamped                   # noqa: E402
from tf2_ros import TransformBroadcaster                         # noqa: E402
from geometry_msgs.msg import Twist                              # noqa: E402
from nav_msgs.msg import Odometry                                # noqa: E402
from rosgraph_msgs.msg import Clock                              # noqa: E402
from sensor_msgs.msg import JointState, LaserScan                # noqa: E402
from std_msgs.msg import Float64, Float64MultiArray, String               # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]
SCAN_N, SCAN_MIN, SCAN_MAX = 360, -3.14159, 3.14159
SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
RANGE_MIN, RANGE_MAX = 0.12, 10.0
ROBOT = '/World/omni_bot'
DOOR_ROOT = '/World/door'
WALLS = ('/World/door/frame/wall_hinge_side', '/World/door/frame/wall_latch_side')
# /coman/applied_cmd 的 18 欄契約（與 E2 相同，下游共用同一份解析）
AP_COLS = ['physics_step_id', 'sim_t', 'bvx_body', 'bvy_body', 'wz',
           'qd1', 'qd2', 'qd3', 'qd4', 'qd5', 'qd6',
           'exec_mode_code', 'cmd_age_s', 'n_recv', 'n_rejected',
           'api_applied', 'src_recv_seq', 'src_recv_sim_t']


def read_world(path):
    raw = re.sub(r'<!--.*?-->', '', open(path, encoding='utf-8').read(),
                 flags=re.S)
    w = ET.fromstring(raw).find('world')
    out = []
    for m in w.findall('model'):
        pose = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
        pose += [0.0] * (6 - len(pose))
        g = m.find('.//collision/geometry')
        kind, dims = None, None
        if g is not None and len(g):
            e = list(g)[0]
            kind = e.tag
            if kind == 'box':
                dims = [float(v) for v in e.findtext('size').split()]
            elif kind == 'cylinder':
                dims = [float(e.findtext('radius')),
                        float(e.findtext('length'))]
        mat = m.find('.//visual/material/diffuse')
        rgb = ([float(v) for v in mat.text.split()][:3] if mat is not None
               else [0.8, 0.8, 0.8])
        out.append(dict(name=m.get('name'), pose=pose, kind=kind, dims=dims,
                        rgb=rgb))
    return out


def yaw_of(q):
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def ray_range(query, origin, direction, max_range, excluded):
    best = [float('inf'), None]

    def report(hit):
        coll = str(getattr(hit, 'collision', '') or '')
        body = str(getattr(hit, 'rigidBody', '') or '')
        d = float(getattr(hit, 'distance', 0.0))
        if any(coll == e or (body and e.startswith(body + '/'))
               for e in excluded):
            return True
        if 0.0 <= d < best[0]:
            best[0], best[1] = d, (coll or body)
        return True

    query.raycast_all(origin, direction, max_range, report)
    return best[0]


class Bridge(Node):
    def __init__(self):
        super().__init__('door_k2_sim')
        # **關節順序要可被查核。** arm_vel_adapter 會向消費節點查這個參數，
        # 核對它與安全鏈用的順序一致 —— 順序不同的話每個關節都拿到別人的
        # 速度，而下游殘差看起來仍然健康，等到手臂動到不該去的地方才發現。
        self.declare_parameter('joints', list(ARM))
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan_raw', 10)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.applied_pub = self.create_publisher(
            Float64MultiArray, '/coman/applied_cmd', 10)
        self.sp_pub = self.create_publisher(
            Float64MultiArray, '/coman/arm_setpoint', 10)
        self.drawer_pub = self.create_publisher(
            Float64MultiArray, '/door/state', 10)
        # /door/contacts：JSON，窗口內逐物理步最大值（見 contacts_window）
        self.contacts_pub = self.create_publisher(String, '/door/contacts', 10)
        self.ho_pub = self.create_publisher(String, '/handover/state', 10)
        # **設定點介面契約**。求解節點的握手閘門把它當 ARM 的必要條件
        #（`_meta_ok is not True` 就不會 ARMED），所以要發。
        # 內容與 isaac_wholebody_sim_e2.py 同一份契約 —— 欄位、關節順序、
        # 取樣時刻與物理步長都要一致，否則閘門會判「介面契約未通過」。
        _lat1 = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sp_meta_pub = self.create_publisher(
            String, '/coman/arm_setpoint_meta', _lat1)
        self._sp_meta_sent = False
        # [sim_t, 開度] ＋ 列優先 4x4 把手世界位姿（USD 真值）
        # **TF（預設關閉）**：Phase 1 的感知鏈要用 TF 把掃描點換到全域座標。
        # 不借 odom_tf_broadcaster —— 它吃 /odom_raw 再重發 /odom，會和本節點
        # 的 /odom 變成兩個發布者。本節點本來就握有真值位姿，自己發最乾淨。
        self.tf_pub = (TransformBroadcaster(self)
                       if a.pub_tf.lower() == 'true' else None)
        self.handle_pub = self.create_publisher(Float64MultiArray,
                                                '/door/handle_pose', 10)
        self.sync_pub = self.create_publisher(Float64MultiArray,
                                              '/door/sync_state', 10)
        self.create_subscription(Twist, '/cmd_vel', self._counted(self._cmd_vel), 10)
        # **減速交接段**：第三個控制權擁有者，與導航共用直寫路徑。
        self.create_subscription(Twist, '/glide_vel', self._counted(self._glide_vel), 10)
        # 夾爪：目標手指位置（m）。沒收到就維持張開。
        self.grip_cmd = None
        self.create_subscription(Float64, '/gripper/cmd', self._counted(self._grip_cmd), 10)
        # 前五欄相容舊格式；追加 physics_step, continuous_s, threshold_n。
        self.grip_pub = self.create_publisher(Float64MultiArray,
                                              '/gripper/state', 10)
        self.wb_env_latest = None                                                   # EV4
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd',
                                 self._counted(self._wb_cmd), 10)                  # 展開節點的路徑（求解端在封裝模式不走這條）
        if a.cmd_env:                                                              # EV4：求解端命令只經封裝入口
            self.create_subscription(Float64MultiArray, ENV.TOPIC[ENV.ST_ADAPTER], self._counted(self._wb_env), 10)
            self.applied_src_pub = self.create_publisher(Float64MultiArray, '/door/applied_src', 100)
        if a.w2_mode_run_id:                                                       # EV6：v3 唯一入口、請求轉送、權威狀態
            self.w2_inbox = _w2_col.deque()
            self.w2_inbox_cap = 512
            self.w2_n_overflow = self.w2_n_bad3 = self.w2_n_v2_ignored = 0
            self.w2_req_q = []
            self.create_subscription(Float64MultiArray, E3.TOPIC3[ENV.ST_ADAPTER], self._counted(self._wb_env3), 50)
            _lat_w2 = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, '/door/w2_mode_req_task', self._counted(self._w2_req), _lat_w2)
            self.w2_req_pub = self.create_publisher(String, '/door/w2_mode_req', _lat_w2)
            self.w2_state_pub = self.create_publisher(String, '/door/w2_mode_state', 50)
            self.w2_eff_q = []                                                         # EV7：5 N 恢復請求
            self.create_subscription(String, '/door/finger_effort_req', self._counted(self._w2_eff_req), _lat_w2)
            self.w2_eff_ack_pub = self.create_publisher(String, '/door/finger_effort_ack', _lat_w2)
        self.create_subscription(String, '/handover/request', self._counted(self._ho), 10)
        # **受控停止**：收到就跳出主迴圈並封存。沒有這條，運行器在任務結束後
        # cleanup 會在封存寫出前把模擬器殺掉 —— 逐步紀錄就全沒了。
        self.create_subscription(String, '/room/stop', self._counted(self._stop), 10)
        # **K2 中止閂鎖**（latched）：收到後不再把全身命令送進命令鏈、夾爪凍結（door_k2_io.K2AbortLatch）
        self.k2_abort_msg = None
        self.create_subscription(String, '/door/k2_abort', self._counted(self._k2_abort),
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.k3m_req_msg = None                                                     # EV3：模式請求（原始）
        self.k3m_pause_req_msg = None                                               # EV4：暫停請求（原始）
        if a.park_gate_mobile:
            _lat3 = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, '/door/base_mode_req', self._counted(self._k3m_req), _lat3)
            self.k3m_ack_pub = self.create_publisher(String, '/door/base_mode_ack', _lat3)
            self.k3m_abort_pub = self.create_publisher(String, '/door/k2_abort', _lat3)
            self.create_subscription(String, '/door/pause_req', self._counted(self._k3m_pause_req), _lat3)   # EV4
            self.k3m_pause_ack_pub = self.create_publisher(String, '/door/pause_ack_exec', _lat3)
        self.stop_req = None
        self.n_cb = 0
        self.n_drain_max = 0
        self.nav_cmd = None
        self.nav_cmd_t = None
        self.n_nav_cmd = 0
        self.glide_cmd = None
        self.glide_cmd_t = None
        self.n_glide_cmd = 0
        self.ho_req = None
        self.step_id = 0
        self.last_sim_t = 0.0      # 主迴圈每步更新；回呼用它當接收時間

    def _counted(self, f):
        """包一層計數，主迴圈才能判斷佇列是否已清空。"""
        def g(m):
            self.n_cb += 1
            f(m)
        return g

    def stamp(self, t):
        from builtin_interfaces.msg import Time
        s = Time()
        s.sec = int(t)
        s.nanosec = int(round((t - int(t)) * 1e9))
        return s

    def _grip_cmd(self, m):
        v = float(m.data)
        if math.isfinite(v):
            self.grip_cmd = v

    def _glide_vel(self, m: Twist):
        self.glide_cmd = (float(m.linear.x), float(m.linear.y),
                          float(m.angular.z))
        self.glide_cmd_t = self.last_sim_t
        self.n_glide_cmd += 1

    def _cmd_vel(self, m: Twist):
        self.nav_cmd = (float(m.linear.x), float(m.linear.y),
                        float(m.angular.z))
        # **接收時間只在這裡更新。** 先前在主迴圈裡每個物理步都刷新，
        # 只要曾收到過命令就永遠算新鮮 —— 導航停止發布後舊命令會一直被沿用，
        # 既定的停止處置永遠不會觸發。
        self.nav_cmd_t = self.last_sim_t
        self.n_nav_cmd += 1

    def _wb_cmd(self, m: Float64MultiArray):
        # 轉給命令鏈；鏈自己有逾時、限位與輪級處理
        if len(m.data) >= 9:
            self.wb_latest = (tuple(float(x) for x in m.data[:9]),)

    def _stop(self, m: String):
        self.stop_req = str(m.data)

    def _stop(self, m: String):
        self.stop_req = str(m.data)

    def _k2_abort(self, m: String):
        self.k2_abort_msg = str(m.data)

    def _k3m_req(self, m: String):                                                  # EV3
        try:
            self.k3m_req_msg = json.loads(m.data)
        except Exception:
            self.k3m_req_msg = {'_unparsable': str(m.data)[:200]}

    def _k3m_pause_req(self, m: String):                                            # EV4
        try:
            self.k3m_pause_req_msg = json.loads(m.data)
        except Exception:
            self.k3m_pause_req_msg = {'_unparsable': str(m.data)[:200]}

    def _wb_env(self, m: Float64MultiArray):                                        # EV4：封裝（身分＋九維值）
        if a.w2_mode_run_id:                                                       # EV6：v2 不進鏈（唯一入口＝v3），只計數
            self.w2_n_v2_ignored += 1
            return
        try:
            self.wb_env_latest = ENV.decode(m.data)
        except ValueError:
            self.wb_env_bad = getattr(self, 'wb_env_bad', 0) + 1

    def _wb_env3(self, m: Float64MultiArray):                                       # EV6：只解碼排隊；接收牆鐘另存
        try:
            e = E3.decode3(m.data)
        except ValueError:
            self.w2_n_bad3 += 1
            return
        if len(self.w2_inbox) >= self.w2_inbox_cap:
            self.w2_n_overflow += 1
            return
        self.w2_inbox.append((e, _ev_time.monotonic()))

    def _w2_eff_req(self, m: String):                                               # EV7
        try:
            self.w2_eff_q.append(json.loads(m.data))
        except Exception:
            self.w2_eff_q.append({'_unparsable': str(m.data)[:200]})

    def _w2_req(self, m: String):                                                   # EV6
        try:
            self.w2_req_q.append(json.loads(m.data))
        except Exception:
            self.w2_req_q.append({'_unparsable': str(m.data)[:200]})

    def _ho(self, m: String):
        try:
            self.ho_req = json.loads(m.data)
        except Exception as e:
            self.get_logger().error(f'交棒請求無法解析：{e}')
            self.ho_req = None

    def publish_clock(self, t):
        c = Clock()
        c.clock = self.stamp(t)
        self.clock_pub.publish(c)


def main() -> int:
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)

    # ---- 房間 ----
    n_box = n_cyl = 0
    for m in read_world(a.world):
        if m['kind'] in (None, 'plane'):
            continue
        x, y, z = m['pose'][0], m['pose'][1], m['pose'][2]
        path = f"/World/{m['name']}"
        col = np.array(m['rgb'])
        if m['kind'] == 'box':
            FixedCuboid(prim_path=path, name=m['name'],
                        position=np.array([x, y, z]),
                        scale=np.array(m['dims']), color=col)
            n_box += 1
        else:
            r, l = m['dims']
            FixedCylinder(prim_path=path, name=m['name'],
                          position=np.array([x, y, z]), radius=r, height=l,
                          color=col)
            n_cyl += 1
    print(f'[room] 房間：{n_box} 方塊、{n_cyl} 圓柱', flush=True)

    # ---- 門：door_asset_v2（door_local_v2.yaml；幾何單一來源）----
    with open(a.door_asset, 'rb') as _f:
        door_asset_sha256 = hashlib.sha256(_f.read()).hexdigest()
    dspec = DA.load(a.door_asset)
    dauth = DA.build_usd(stage, dspec, root=DOOR_ROOT)
    for k in ('drive_stiffness', 'drive_damping', 'drive_max_force'):
        if abs(float(dauth[k])) > 1e-12:
            print(f'[door] **{k} 讀回 {dauth[k]}，不為零** ⇒ 中止', flush=True)
            return 14
    print(f'[door] 門建於 {DOOR_ROOT}；限位 {dauth["limit_lower_deg"]}–{dauth["limit_upper_deg"]}°；'
          f'牆端面 {dauth["wall_end_faces_x"]}；碰撞偏移 {sorted(set(map(tuple, dauth["collision_offsets"].values())))}', flush=True)

    # ---- 機器人 ----
    rx, ry, ryaw = (float(v) for v in a.start_pose.split(','))
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(rx, ry, 0.0))
    xf.AddRotateZOp().Set(math.degrees(ryaw))
    finger_col_rep = {'mode': a.finger_collision}
    if a.finger_collision == 'split':
        from finger_collision import apply_split
        finger_col_rep.update(apply_split(stage, ROBOT))
        for _ln, _r in finger_col_rep['fingers'].items():
            print(f'[room] 手指碰撞拆塊 {_ln}：停用 {_r["disabled"]}；'
                  + '；'.join(f'{k} y {v["y_range_mm"]} z {v["z_range_mm"]} mm'
                             for k, v in _r['parts'].items()), flush=True)

    # ---- 零摩擦材質 ----
    # URDF 的 <gazebo><mu1> importer 不讀，產生的 USD 沒有材質，PhysX 退回
    # 預設（約 0.5）。**接觸雙方都要綁**：預設混合規則取平均，只歸零機器人側
    # 仍留一半地面摩擦。不綁的實測後果：0.25 m/s 驅動 5 s 偏航 21.1°。
    fric = None
    if not a.no_frictionless:
        sph = collision_prims(stage, ROBOT, lambda p: 'support_' in p)
        gr = collision_prims(stage, '/World/ground')
        nb = bind_frictionless(stage, sph + gr)
        fric = {'support_spheres': len(sph), 'ground': len(gr), 'bound': nb}
        print(f'[room] 零摩擦材質：支撐球 {len(sph)} + 地面 {len(gr)} '
              f'→ 綁定 {nb}', flush=True)
        if nb == 0:
            print('[room] **一個都沒綁到** ⇒ 底盤會偏航，中止', flush=True)
            return 19
    # ---- 夾持面的摩擦：**明確指定**，不靠 PhysX 預設 ----
    # 兩指與把手橫桿承擔整個夾持。不綁的話它們退回 PhysX 預設（約 0.5）——
    # 那個值不在任何規格裡，也從來沒有被選過。接觸操作的摩擦要像抽屜的
    # 質量與阻尼一樣，執行前選定並記錄。
    grip = [pr for pr in collision_prims(stage, ROBOT)
            if 'finger' in str(pr.GetPath()).lower()]
    grip += [pr for pr in collision_prims(stage, DOOR_ROOT)
             if 'handle_bar' in str(pr.GetPath()).lower()]
    n_grip = bind_friction(stage, grip, a.grip_mu_static, a.grip_mu_dynamic)
    print(f'[room] 夾持面摩擦 μs={a.grip_mu_static} μd={a.grip_mu_dynamic}'
          f' → 綁定 {n_grip}／{len(grip)}', flush=True)
    if n_grip == 0:
        print('[room] **夾持面一個都沒綁到** ⇒ 夾持會靠未指定的預設值，中止',
              flush=True)
        return 21

    # **逐面核實際材質與摩擦值。** 只看綁定數會漏掉未綁的面；而且夾持面
    # （手指、把手橫桿）**不得**被歸零 —— 歸零就夾不住。
    from friction_audit import audit as _fric_audit, summarise as _fric_sum
    fr_ok, fr_rep = _fric_audit(stage, walk, robot_root=ROBOT,
                                drawer_root=DOOR_ROOT)
    print('[room] 摩擦核對：', flush=True)
    for _l in _fric_sum(fr_rep).split('\n'):
        print('   ' + _l, flush=True)
    if not fr_ok and not a.allow_friction_mismatch:
        print('[room] **摩擦核對未通過** ⇒ 中止（要跳過請用 '
              '--allow-friction-mismatch，並在報告標明）', flush=True)
        json.dump(fr_rep, open(os.path.join(a.out, 'friction_audit.json'), 'w'),
                  ensure_ascii=False, indent=1, default=str)
        return 20
    json.dump(fr_rep, open(os.path.join(a.out, 'friction_audit.json'), 'w'),
              ensure_ascii=False, indent=1, default=str)

    _k = UsdLux.DistantLight.Define(stage, '/World/key')
    _k.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(_k).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(600.0)

    bodies, arts = physics_parts(stage, ROBOT)
    if not arts:
        print('[room] **找不到 articulation root** ⇒ 中止', flush=True)
        return 15
    art_root = arts[0]
    print(f'[room] articulation root {art_root}（剛體 {len(bodies)}）',
          flush=True)

    # ---- 夾爪：手指對抽屜的接觸力視圖（**必須在 world.reset() 之前建**）----
    # 做法照搬 isaac_drawer_sim.py（grip_090257_pull20full 用它拉開 20 mm）：
    # 每指一個剛體、只過濾抽屜本體，讀 get_contact_force_matrix。
    # 夾持成立的判準也沿用那一套：每指 ≥ 0.5 N、保持 2 s。
    finger_bodies = sorted(b for b in bodies
                           if str(b).rstrip('/').split('/')[-1]
                           in ('uflite_finger1', 'uflite_finger2'))
    finger_v = None
    if len(finger_bodies) == 2:
        # **用單一字串的字元類別，不用路徑清單。** 清單形式下初始化報
        # 「Provided patterns for sensor and filters did not match any rigid
        # contact entries」，接觸力全程讀成 NaN（motm_cal20_073038）。
        # isaac_drawer_sim.py 也是這樣寫，並註明不要用 (A|B) 的全路徑交替。
        finger_v = _RigidPrim(prim_paths_expr=str(finger_bodies[0])[:-1] + '[12]',
                              name='finger_v', track_contact_forces=True,
                              max_contact_count=128,
                              prepare_contact_sensors=True,
                              contact_filter_prim_paths_expr=[
                                  dauth['leaf_prim']])
        print(f'[door] 手指接觸視圖：{finger_bodies} ↔ {dauth["leaf_prim"]}（含橫桿與門板；分類見門扇配對）',
              flush=True)
    else:
        print(f'[room] **找不到兩隻手指剛體**（{finger_bodies}）⇒ 夾爪不可用',
              flush=True)

    # ---- 門扇 ↔ 機器人各剛體的接觸配對（非預期接觸量測的來源）----
    # 感測端是門扇剛體（門板、立柱、橫桿都屬於它），每個機器人剛體各為一個過濾對象。
    # 讀不到 ⇒ 該步記 None（任務節點視同監看失效），**不當成沒有接觸**。
    pair_names = [str(b).rstrip('/').split('/')[-1] for b in bodies]
    leaf_pair_v = _RigidPrim(prim_paths_expr=dauth['leaf_prim'], name='leaf_pairs',
                             track_contact_forces=True, max_contact_count=1024,
                             prepare_contact_sensors=True,
                             contact_filter_prim_paths_expr=[str(b) for b in bodies])
    print(f'[door] 門扇接觸配對：{len(bodies)} 個機器人剛體', flush=True)
    # ---- 機器人各剛體 ↔ 兩側牆（每剛體一個視圖；牆是靜態碰撞體，不能當感測端）----
    wall_views = {}
    for b in bodies:
        _nm = str(b).rstrip('/').split('/')[-1]
        wall_views[_nm] = _RigidPrim(prim_paths_expr=str(b), name=f'wall_{_nm}', track_contact_forces=True,
                                     max_contact_count=64, prepare_contact_sensors=True,
                                     contact_filter_prim_paths_expr=list(WALLS))
    print(f'[door] 牆接觸視圖：{len(wall_views)} 個剛體 ↔ {WALLS}', flush=True)
    leaf_v = _RigidPrim(prim_paths_expr=dauth['leaf_prim'], name='leaf_state')

    world.reset()
    robot = SingleArticulation(prim_path=art_root, name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = {n: names.index(n) for n in ARM if n in names}
    missing = [n for n in ARM if n not in idx]
    if missing:
        print(f'[room] **URDF 缺關節 {missing}** ⇒ 中止', flush=True)
        return 16

    # 收攏姿態：導航期間由位置驅動維持
    stow = [float(v) for v in a.stow_q.split(',')]
    if len(stow) != len(ARM):
        print(f'[room] **--stow-q 需要 {len(ARM)} 個值** ⇒ 中止', flush=True)
        return 17
    q0 = robot.get_joint_positions()
    for k, j in enumerate(ARM):
        q0[idx[j]] = stow[k]
    kp = np.zeros(robot.num_dof, dtype=np.float32)
    kd = np.zeros(robot.num_dof, dtype=np.float32)
    for j in ARM:
        kp[idx[j]] = a.kp
        kd[idx[j]] = a.kd
    # **手指要有驅動增益**。URDF 匯入時記「Stiffness and damping not
    # available ... actuator will be created without gain parameters」，
    # 先前這裡只給手臂增益，其他關節全是 0 ⇒ 手指是軟的，下目標也不會動。
    # 值沿用 isaac_drawer_sim.py 的 kp 1e4 / kd 1e3。
    FJ = [j for j in ('finger_joint1', 'finger_joint2') if j in names]
    fidx = {j: names.index(j) for j in FJ}
    for j in FJ:
        kp[fidx[j]] = a.finger_kp
        kd[fidx[j]] = a.finger_kd
        q0[fidx[j]] = a.finger_open
    print(f'[room] 手指關節 {FJ}；增益 kp {a.finger_kp} kd {a.finger_kd}；'
          f'初始張開 {a.finger_open}', flush=True)
    if finger_v is not None:
        finger_v.initialize()

    def finger_contact_n():
        """每指對抽屜的接觸力大小（N）。讀不到就回 (nan, nan)，**不當成 0**。"""
        if finger_v is None:
            return float('nan'), float('nan')
        try:
            M = np.asarray(finger_v.get_contact_force_matrix(dt=a.physics_dt))
            # (手指, 過濾對象, 3) ⇒ 每指對抽屜合力的大小
            f = np.linalg.norm(M.reshape(M.shape[0], -1, 3).sum(axis=1), axis=1)
            return float(f[0]), float(f[1])
        except Exception as _e:
            if not _fc_warned[0]:
                _fc_warned[0] = True
                print(f'[room] **手指接觸力讀不到**：{_e!r}（之後回 NaN，'
                      f'不當成 0）', flush=True)
            return float('nan'), float('nan')
    _fc_warned = [False]
    leaf_pair_v.initialize()
    leaf_v.initialize()
    for _v in wall_views.values():
        _v.initialize()

    def _np(x):
        return np.asarray(x.cpu().numpy() if hasattr(x, 'cpu') else x, dtype=float)

    Hd = dspec['handle']
    _bar_r = float(Hd['bar']['radius'])
    _bar_half = float(Hd['bar']['length']) / 2.0

    def leaf_contacts(T_WH):
        """門扇 ↔ 機器人各剛體：door_k2_io.classify_contacts（bar／ambiguous／panel）。讀不到回 None。"""
        try:
            fn, pts, nrm, dist, cnt, start = (_np(v) for v in leaf_pair_v.get_contact_force_data(dt=a.physics_dt))
            cnt, start = cnt.reshape(-1), start.reshape(-1)
            out = {}
            for jb in range(len(cnt)):
                k, s0 = int(cnt[jb]), int(start[jb])
                if k == 0:
                    continue
                out[pair_names[jb]] = KIO.classify_contacts(pts[s0:s0 + k].reshape(-1, 3), np.abs(fn[s0:s0 + k, 0]), T_WH, _bar_r, _bar_half,
                                                            [float(x) for x in Hd['posts']['along_u']], float(Hd['posts']['size'][0]) / 2.0,
                                                            a.bar_class_tol)
            return out
        except Exception as _e:
            if not _lc_warned[0]:
                _lc_warned[0] = True
                print(f'[door] **門扇接觸配對讀不到**：{_e!r}（之後記 None）', flush=True)
            return None
    _lc_warned = [False]

    def wall_contacts():
        """{剛體名: 對兩側牆的接觸力大小}；任一讀不到回 None。"""
        try:
            out = {}
            for nm, v in wall_views.items():
                M = _np(v.get_contact_force_matrix(dt=a.physics_dt)).reshape(-1, 3)
                out[nm] = float(np.linalg.norm(M.sum(axis=0)))
            return out
        except Exception as _e:
            if not _wc_warned[0]:
                _wc_warned[0] = True
                print(f'[door] **牆接觸讀不到**：{_e!r}（之後記 None）', flush=True)
            return None
    _wc_warned = [False]

    # joint6 反作用力列：**只用 articulation metadata**（同 isaac_drawer_sim.py；固定關節會改變索引）。讀不到 ⇒ 拒跑，不退回 DOF 順序。
    try:
        _md = robot._articulation_view._metadata
        _j6_row = int(_md.joint_indices['joint6']) + 1
        _jnames = list(_md.joint_names)
    except Exception as _e:
        print(f'[door] **讀不到 articulation metadata（{_e!r}）⇒ 無法確認腕力列號，拒跑**', flush=True)
        return 27
    print(f'[door] joint6 反作用力列號 {_j6_row}（metadata.joint_indices；關節 {_jnames}）', flush=True)

    def wrist_force():
        try:
            F = np.asarray(robot.get_measured_joint_forces(), float)
            v = F[_j6_row][:3]
            return float(np.linalg.norm(v)) if np.isfinite(v).all() else None
        except Exception as _e:
            if not _wf_warned[0]:
                _wf_warned[0] = True
                print(f'[door] **腕力讀不到**：{_e!r}（之後記 None）', flush=True)
            return None
    _wf_warned = [False]

    # ---- **EV 版取證**（--evidence-out；未給 ⇒ 不建檔、不讀任何新 API）----
    # 開檔／寫出／flush／關檔全由 KEV.EvidenceWriter 隔離故障；任何失敗只記入摘要，不中斷模擬、不阻擋 room_run.json 封存。
    _ev_w = KEV.EvidenceWriter(a.evidence_out) if a.evidence_out else None
    _ev_f = _ev_w if (_ev_w is not None and _ev_w.active) else None
    if _ev_w is not None and _ev_f is None:
        print(f'[door] **取證開檔失敗**：{_ev_w.status["errors"]}（取證停用，摘要標為不完整）', flush=True)
    _ev_dofs, _why = (KEV.safe(lambda: list(robot.dof_names)) if _ev_f is not None else (None, None))
    if _why:
        _ev_w.count_err('dof_names')

    def _ev_write(step_id, t, T_WH, arm_sp):
        rec_ev = {'step': int(step_id), 't': KEV._r(t, 6), 'T_WH': [KEV._r(x, 9) for x in np.asarray(T_WH, float).reshape(-1)],
                  'arm_sp': None if arm_sp is None else [KEV._r(x, 9) for x in arm_sp]}
        cd, why = KEV.safe(lambda: KEV.extract_contacts(*(leaf_pair_v.get_contact_force_data(dt=a.physics_dt)), pair_names))
        rec_ev['contacts'], rec_ev['contacts_err'] = cd, why
        fd, why = KEV.safe(lambda: KEV.extract_friction(*(leaf_pair_v.get_friction_data(dt=a.physics_dt)), pair_names))
        rec_ev['friction'], rec_ev['friction_err'] = fd, why
        wr, why = KEV.safe(lambda: KEV.wrench_row(robot.get_measured_joint_forces(), _j6_row))
        rec_ev['j6_reaction_wrench'], rec_ev['j6_reaction_wrench_err'] = wr, why
        ef, why = KEV.safe(lambda: KEV.efforts_by_name(robot.get_measured_joint_efforts(), _ev_dofs, KEV.ARM_DOF + KEV.FINGER_DOF))
        rec_ev['dof_measured_efforts'], rec_ev['dof_measured_efforts_err'] = ef, why
        for k in ('contacts_err', 'friction_err', 'j6_reaction_wrench_err', 'dof_measured_efforts_err'):
            if rec_ev[k]:
                _ev_w.count_err(k)
        _ev_w.write(rec_ev)

    def door_q_qd():
        """門角 q（門扇繞 +z 轉 −q；placement yaw 須為 0）與 q̇＝−ω_z。"""
        p, o = leaf_v.get_world_poses()
        w, x, y, z = (float(v) for v in o[0])
        r10, r00 = 2 * (x * y + z * w), 1 - 2 * (y * y + z * z)
        q = -math.atan2(r10, r00)
        om = _np(leaf_v.get_angular_velocities()).reshape(-1)
        return q, -float(om[2])
    if abs(float(dspec['placement']['pose']['yaw_deg'])) > 1e-12:
        print('[door] **placement yaw 須為 0**（門角讀法假設）⇒ 中止', flush=True)
        return 26
    robot.set_joint_positions(q0)
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=q0))
    print(f'[room] 手臂收攏於 {[round(v, 6) for v in stow]}', flush=True)
    _fe_idx = [fidx[j] for j in FJ]                                                   # EV5：啟動讀回（不改值）
    fin_eff_init = FEFF.read_init(robot.get_articulation_controller(), _fe_idx, FJ, names, a.pause_finger_original_effort)
    fin_eff = FEFF.PauseFingerEffort(a.pause_finger_max_effort, _fe_idx, FJ)
    print(f'[room] 手指驅動輸出上限讀回 {fin_eff_init["readback"]}（{FJ}）；暫停期設定 {a.pause_finger_max_effort or "不改"}', flush=True)
    if fin_eff.enabled and (not a.park_gate_mobile or len(_fe_idx) != 2):
        print('[room] **--pause-finger-max-effort 需要 --park-gate-mobile 與兩隻手指關節** ⇒ 中止', flush=True)
        return 37
    if fin_eff.enabled and not fin_eff_init['ok']:
        print(f'[room] **手指驅動上限啟動讀回不合格**（{fin_eff_init["why"]}）⇒ 拒跑', flush=True)
        return 38

    from pxr import Gf as _Gf
    from pxr import UsdGeom as _UG
    xfc = _UG.XformCache()
    hprims = [pr for pr in collision_prims(stage, DOOR_ROOT)
              if 'handle_bar' in str(pr.GetPath()).lower()]
    hprim = hprims[0] if hprims else None
    if hprim is None:
        print('[door] **找不到把手 prim** ⇒ 中止', flush=True)
        return 22
    print(f'[door] 把手 prim {hprim.GetPath()}', flush=True)
    q_door0, _ = door_q_qd()
    print(f'[door] 初始門角 {math.degrees(q_door0):.4f}°', flush=True)

    # ---- 第三人視角相機（錄影用）----
    cam = None
    cam_dir = os.path.join(a.out, 'frames')
    cam_csv = None
    n_frame = 0
    if a.cam.lower() == 'true':
        import imageio.v2 as _imageio
        from isaacsim.sensors.camera import Camera as _Camera
        _W, _H = (int(v) for v in a.cam_res.split('x'))
        os.makedirs(cam_dir, exist_ok=True)
        cam = _Camera(prim_path='/World/rec_cam', resolution=(_W, _H))
        cam.initialize()
        _cg = _UG.Camera(stage.GetPrimAtPath('/World/rec_cam'))
        _cg.GetClippingRangeAttr().Set(_Gf.Vec2f(0.05, 200.0))
        _cg.GetFocalLengthAttr().Set(float(a.cam_foc))
        _eye = [float(v) for v in a.cam_eye.split(',')]
        _at = [float(v) for v in a.cam_at.split(',')]
        _cx = _UG.Xformable(stage.GetPrimAtPath('/World/rec_cam'))
        _cx.ClearXformOpOrder()
        _cx.AddTransformOp().Set(_Gf.Matrix4d().SetLookAt(
            _Gf.Vec3d(*_eye), _Gf.Vec3d(*_at), _Gf.Vec3d(0, 0, 1)).GetInverse())
        cam_csv = open(os.path.join(a.out, 'frames.csv'), 'w')
        cam_csv.write('frame,sim_t,step,owner,kind,door_q_deg,x,y,yaw\n')
        print(f'[room] **錄影開啟** {_W}x{_H} @ {a.cam_hz} Hz → {cam_dir}',
              flush=True)

    # ---- lidar ----
    lidar_prim = None
    for pr in walk(stage, ROBOT):
        if pr.GetName() == 'lidar_link':
            lidar_prim = pr
            break
    if lidar_prim is None:
        print('[room] **找不到 lidar_link** ⇒ 中止', flush=True)
        return 18
    excluded = []
    for pr in walk(stage, ROBOT):
        if not pr.HasAPI(UsdPhysics.CollisionAPI):
            continue
        p_ = str(pr.GetPath())
        if p_.endswith('/base_link/cylinder') or '/lidar_link/' in p_:
            excluded.append(p_)
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()

    # ---- 命令鏈 ＋ 雙來源執行層 ----
    def low_speed_bound(vx, vy, wz):
        lin = math.hypot(vx, vy)
        if lin > a.base_lin_max:
            return (False, f'線速度 {lin:.4f} > {a.base_lin_max} m/s')
        if abs(wz) > a.base_ang_max:
            return (False, f'角速度 {abs(wz):.4f} > {a.base_ang_max} rad/s')
        return (True, None)

    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    K = WholeBodyKinematics.from_urdf_file(os.path.join(
        WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
    lim = np.array(K.joint_limits())
    chain = CmdChainE2(max_cmd_age_s=a.max_cmd_age_s,
                       arm_rate_max=a.arm_rate_max, wheel_ok=low_speed_bound,
                       joint_lower=tuple(float(x) for x in lim[0, 3:]),
                       joint_upper=tuple(float(x) for x in lim[1, 3:]),
                       joint_margin=a.joint_margin, mode='solver_drawer',
                       wheel_cfg=WheelLimitConfig(
                           arm_rate_max=a.arm_rate_max,
                           dt_max=max(5.0 * a.physics_dt, 0.05)),
                       base_accel_max_lin=(a.base_accel_lin
                                           if a.base_accel_lin > 0 else None),
                       base_accel_max_ang=(a.base_accel_ang
                                           if a.base_accel_ang > 0 else None))
    # ---- PARK_FIXED：靜止閘門與保持監看（純邏輯在 park_fixed.py）----
    if a.park_fixed and a.park_hold:
        print('[room] **--park-fixed 與 --park-hold 只能擇一** ⇒ 中止', flush=True)
        return 3
    park_on = bool(a.park_fixed or a.park_hold)
    _pk = [float(v) for v in a.park_pose.split(',')]
    park_cfg = ParkCfg(park_x=_pk[0], park_y=_pk[1], park_yaw=_pk[2],
                       hold_s=max(0.5, float(a.max_cmd_age_s)), hold_mode=bool(a.park_hold),
                       dev_pos_max=float(a.park_dev_pos_max), dev_yaw_max=math.radians(float(a.park_dev_yaw_max_deg)))
    print(f'[door] 停車保持條件（K2 登錄）：相對錨點 ≤ {park_cfg.dev_pos_max} m、≤ {a.park_dev_yaw_max_deg}°；'
          f'伺服上限 {park_cfg.hold_v_max} m/s、{park_cfg.hold_w_max} rad/s', flush=True)
    park_gate = StaticGate(park_cfg) if park_on else None       # 入場靜止閘門：沿用 ParkCfg 既有嚴格靜止條件
    # K2 保持期監看：只核 K2 登錄的相對錨點 20 mm／2° 與伺服命令上界；**實測速率只另報、不中止**（K2 規格 draft-2 §F）
    import dataclasses as _dc
    hold_cfg = _dc.replace(park_cfg, v_lin_max=math.inf, w_max=math.inf)
    park_hold = None
    park_done = False
    park_state = {'violation': None, 'transfer_without_gate': False, 'external': None}
    park_prev_cmd = (float('nan'),) * 3      # 上一物理步實際寫入的底盤命令（閘門用）
    park_last_sample = None                  # 上一物理步 (step, t, x, y, yaw)（保持監看承接用）
    # 規格：PARK_FIXED **明確取消滾動交棒下界**（停住反而不能接手）；MOTM 保持原值。其他預核照做。
    v_min_lin_eff = 0.0 if park_on else a.v_min_lin

    def _park_guard(to, sid, st_):
        """提交切換當步：停車閘門已通過、停車條件仍成立、閘門資料是前一步（新鮮）、未違規。"""
        if park_state['violation'] is not None:
            return False, 'PARK_FIXED 已違規閂鎖'
        if not park_gate.passed:
            return False, '停車靜止閘門尚未通過'
        if not park_gate.ok_now:
            return False, f'停車條件當前不成立：{park_gate.why_now}'
        # **本步**的停車量測（ex.step 之前已以本步位姿更新閘門）；不以前一步的 ok_now 充當當步核對
        if park_gate.last is None or park_gate.last['step'] != int(sid):
            return False, (f'停車閘門沒有本步量測（最後一筆 '
                           f'{None if park_gate.last is None else park_gate.last["step"]}，本步 {sid}）')
        return True, None

    ex = DualSourceExecutor(chain, initial=AUTH_NAV,
                            stow_setpoint=tuple(stow),
                            max_seed_age_s=max(5.0 * a.physics_dt, 0.05),
                            v_box_lin=a.v_box_lin, v_box_ang=a.v_box_ang,
                            v_min_lin=v_min_lin_eff,
                            switch_guard=(_park_guard if park_on else None),
                            nav_accel_max=(None if a.nav_accel_x <= 0 else
                                           (a.nav_accel_x, a.nav_accel_y,
                                            a.nav_accel_w)))

    # ---- 重播渲染模式：只渲染，不跑物理也不開 ROS ----------------------
    # 邊跑邊渲染會和控制搶 CPU：實測 RTF 由 ~1.0 掉到 0.51，控制週期變少，
    # 機器人在**模擬時間**裡也跟著變慢，軌跡就不是原來那條了
    #（nav_video_025746：164 s 模擬時間只走到 d = 0.69 m）。
    # 所以物理與控制照原樣跑，渲染另外一趟重播已封存的逐步位姿。
    # 場景由**同一份程式**建起，不另寫一支重播器。
    # 重播渲染與腕部擷取模式：K2 不支援（已移除）
    rclpy.init()
    node = Bridge()
    node.wb_latest = None
    wlive = None
    _wlive_done = None
    if park_on:
        node.park_gate_pub = node.create_publisher(String, '/park/gate', 10)
        node.park_vio_pub = node.create_publisher(String, '/park/violation', 10)
        node.park_ext_vio = None

        def _ext_vio(m):
            try:
                node.park_ext_vio = json.loads(m.data)
            except Exception:
                node.park_ext_vio = {'why': m.data}
        node.create_subscription(String, '/park/violation', _ext_vio, 10)
        print(f'[room] **{"PARK_HOLD" if a.park_hold else "PARK_FIXED"}**：停位 {_pk}；閘門保持 {park_cfg.hold_s:.3f} s；'
              f'滾動交棒下界取消（v_min_lin {a.v_min_lin} → 0）', flush=True)
    def _new_win():
        return {'n': 0, 'invalid': 0, 'fb': [0.0, 0.0], 'fb_min': [float('inf'), float('inf')], 'unexp': 0.0,
                'unexp_who': None, 'wall': 0.0, 'wrist': 0.0}
    node.contacts_win = _new_win()
    k2_latch = KIO.K2AbortLatch()
    k3m = {'active': False, 'switch': None, 'violation': None, 'log': [], 'events': []}           # EV3
    env_in = K3MSRC.EnvIntake(run_id_num=ENV.run_id_num(a.run_id) if a.run_id else None, stage_id=ENV.ST_ADAPTER, kinds=(ENV.K_NORMAL,))   # EV4 r2
    env_stats = {'n_applied_src_reports': 0}
    w2x = None                                                                      # EV6
    w2_xm = w2_effa = None                                                          # EV7
    if a.w2_mode_run_id:
        if not a.cmd_env or not a.run_id:
            print('[door] **--w2-mode-run-id 需要 --cmd-env 與 --run-id** ⇒ 中止', flush=True)
            sys.exit(41)
        w2_auth = W2M4.ModeAuthority4(a.w2_mode_run_id, run_id_num=ENV.run_id_num(a.run_id), stage_id=ENV.ST_ADAPTER,
                                      cand_timeout_s=a.w2_cand_timeout_s, cmd_max_age_s=a.max_cmd_age_s,
                                      post_commit_timeout_s=a.w2_post_commit_timeout_s, dt=a.physics_dt,
                                      a_lin=a.base_accel_lin, a_ang=a.base_accel_ang)
        w2x = W2X2.W2Exec2(w2_auth, env_in, filt=lambda u: k2_latch.filter_wb(u))
        if a.w2_release:                                                            # EV7：權威 r5＋逐步解除監看＋5 N 恢復權威
            if not fin_eff.enabled:
                print('[door] **--w2-release 需要 --pause-finger-max-effort > 0** ⇒ 中止', flush=True)
                sys.exit(42)
            w2_xm = K3BR.ExecReleaseMonitor(reg={'dt': a.physics_dt})
            w2_auth = W2M5.ModeAuthority5(a.w2_mode_run_id, run_id_num=ENV.run_id_num(a.run_id), stage_id=ENV.ST_ADAPTER,
                                          cand_timeout_s=a.w2_cand_timeout_s, cmd_max_age_s=a.max_cmd_age_s,
                                          post_commit_timeout_s=a.w2_post_commit_timeout_s, dt=a.physics_dt,
                                          a_lin=a.base_accel_lin, a_ang=a.base_accel_ang, arm_rate_max=a.arm_rate_max,
                                          gate3=lambda _s7: w2_xm.confirmed_fresh(_s7))
            w2x = W2X2.W2Exec2(w2_auth, env_in, filt=lambda u: k2_latch.filter_wb(u))
            w2_effa = K3BR.EffortRestoreAuthority(a.w2_mode_run_id)
            w2_auth.extra_state = lambda: {'release': {'state': w2_xm.state, 'why': w2_xm.why, 't_open': w2_xm.t.get('open'),
                                                       't_conf': w2_xm.t.get('B_CONFIRMED'), 'step_conf': w2_xm.step_conf,
                                                       'last_valid_step': w2_xm.last_valid_step},
                                           'effort': w2_effa.ack()}
        w2_rt = {'abort_handled': False}

    def _w2_known_abort():                                                          # EV6 r2：既有中止彙整（無 ⇒ None）
        if k2_latch.latched is not None:
            return f"k2_latch:{k2_latch.latched.get('why')}"
        if park_on and park_state.get('violation') is not None:
            return f"park_violation:{(park_state['violation'] or {}).get('why')}"
        if k3m.get('violation') is not None:
            return f"k3m_violation:{(k3m['violation'] or {}).get('why')}"
        if w2_xm is not None and w2_xm.state == 'FAILED':                               # EV7：解除監看失敗＝已知中止
            return f"release_monitor:{w2_xm.why}"
        return None
    if a.park_gate_mobile:
        if not a.park_hold or not a.run_id or not a.corridor_table:
            print('[door] **--park-gate-mobile 需要 --park-hold、--run-id、--corridor-table** ⇒ 中止', flush=True)
            sys.exit(21)
        k3m_auth = K3MMODE.ExecutorModeAuthority(a.run_id)
        k3m_pauth = K3MP.ExecutorPauseAuthority(a.run_id)                           # EV4
        k3m_cor = K3MCOR.Corridor.from_json(json.load(open(a.corridor_table))['table'])
        k3m_env = [float(v) for v in a.envelope.split(',')]
        k3m['config'] = {'run_id': a.run_id, 'corridor_table': a.corridor_table,
                         'corridor_table_sha256': hashlib.sha256(open(a.corridor_table, 'rb').read()).hexdigest(),
                         'inner_mm': a.corridor_inner_mm, 'inner_deg': a.corridor_inner_deg, 'envelope': k3m_env,
                         'corridor_table_json': k3m_cor.to_json()}
    print('[door] 進入主迴圈', flush=True)

    se = max(1, int(round((1.0 / a.scan_rate) / a.physics_dt)))
    ste = max(1, int(round((1.0 / a.state_rate) / a.physics_dt)))
    next_scan = next_state = next_cam = next_grip = 0.0
    step_id = 0
    contact_hold = PhysicsContactHold(physics_dt=a.physics_dt,
                                      contact_min_n=a.contact_min_n)
    prev_pose = None
    rec = {'steps': [], 'events': [],
           'steps_cols': ['step', 'sim_t', 'owner(0=nav,1=wb,2=glide)', 'kind',
                          'base_cmd_body', 'meas_vb_body', 'applied_arm',
                          'base_xyth', 'q_arm_meas', 'door_q_rad', 'door_qdot_rad_s',
                          'q_finger', 'finger_contact_n', 'contact_held_s',
                          'leaf_contacts', 'wall_contacts', 'wrist_force_n']
                         + (['park_mode'] if park_on else [])
                         + (['applied_src'] if a.cmd_env else []),                         # EV4
           'leaf_contacts': {'source': 'every_physics_step', 'row': '{robot_body: {bar: N, panel: N}}（法向力大小和）',
                             'classification': f'接觸點到橫桿軸線段距離 ≤ 半徑＋{a.bar_class_tol} m ⇒ bar', 'none_means': '讀不到（不是零）'},
           'wall_contacts': {'source': 'every_physics_step', 'row': '{robot_body: N}', 'none_means': '讀不到（不是零）'},
           'wrist_force': {'source': 'robot.get_measured_joint_forces()[joint6 row][:3] 之大小', 'none_means': '讀不到'},
           'contact_hold': {'source': 'every_physics_step',
                            'physics_dt': a.physics_dt,
                            'contact_min_n': a.contact_min_n},
           'owner_codes': {'0': AUTH_NAV, '1': AUTH_WHOLEBODY,
                           '2': AUTH_GLIDE}}

    def _w1e_abort(_why5, _step5, _t5):                                             # EV5 r2：W1e 讀回異常 ⇒ 既有 K2 中止閂鎖
        rec['events'].append({'sim_t': _t5, 'step': _step5, 'w1e_finger_effort_abort': _why5})
        if k2_latch.latched is None:
            _gnow5 = (a.finger_open if node.grip_cmd is None else min(max(float(node.grip_cmd), a.finger_closed), a.finger_open))
            k2_latch.on_msg(f'w1e_finger_effort：{_why5}', _step5, _t5, _gnow5)
            k2_latch.expire_chain(chain)
            ex.cancel_handover(f'w1e_finger_effort：{_why5}')
            node.k3m_abort_pub.publish(String(data=json.dumps({'why': f'w1e_finger_effort：{_why5}', 't': _t5,
                                                               'source': 'executor_w1e'}, default=float)))
        print(f'[door] **W1e 手指驅動上限異常 ⇒ 中止閂鎖**（{_why5}）@ step {_step5}', flush=True)

    while rclpy.ok():
        t = float(world.current_time)
        if t >= a.sim_limit:
            print(f'[room] 達模擬上限 {a.sim_limit} s', flush=True)
            break
        if node.stop_req is not None:
            print(f'[room] **受控停止**：{node.stop_req}', flush=True)
            rec['stop_reason'] = node.stop_req
            break
        # **每個物理步把待處理的回呼全部處理完。** spin_once 一次只處理一筆；
        # 數個持續發布的話題（/cmd_vel、/wb_vel_cmd、/gripper/cmd …）在每步
        # 一筆的處理速率下會在佇列裡積壓 —— 實測全身命令由發布到生效 p50
        # 150–210 ms、最大 390 ms（求解端延遲補償假設 80 ms），這是求解節點
        # 上線後手臂上下擺盪的根因（motm_182255、grip_bar26_161806）。
        # 沒有積壓時只多一次呼叫；全身命令仍只取最新一筆進鏈。
        _n_this = 0
        for _ in range(64):
            _before = node.n_cb
            rclpy.spin_once(node, timeout_sec=0.0)
            if node.n_cb == _before:
                break
            _n_this += 1
        node.n_drain_max = max(node.n_drain_max, _n_this)
        node.publish_clock(t)

        # K2 中止閂鎖：收到即閂鎖（之後全身命令不進鏈、夾爪凍結；命令鏈逾時歸零 ⇒ 底盤零速、手臂設定點保持）
        if node.k2_abort_msg is not None and k2_latch.latched is None:
            _gnow = (a.finger_open if node.grip_cmd is None else min(max(float(node.grip_cmd), a.finger_closed), a.finger_open))
            k2_latch.on_msg(node.k2_abort_msg, step_id, t, _gnow)
            # 閂鎖當步：鏈內既有命令立即過期 ⇒ 鏈既有逾時停止路徑（手臂速率歸零、底盤經輪級限制減速）；取消待提交交棒
            _expired = k2_latch.expire_chain(chain)
            ex.cancel_handover('K2 中止閂鎖')
            rec['events'].append({'sim_t': t, 'step': step_id, 'k2_abort_latched': k2_latch.latched, 'grip_frozen': _gnow,
                                  'chain_cmd_expired': _expired, 'handover_cancelled': True})
            print(f'[door] **K2 中止閂鎖** @ step {step_id}：{node.k2_abort_msg}；夾爪凍結 {_gnow}', flush=True)
        # 全身命令進鏈（閂鎖後丟棄）
        def _ev4_violation(_why4):                                                # EV4 r2：入口違規 ⇒ 既有 K2 閂鎖中止
            env_stats.setdefault('violations', []).append({'sim_t': t, 'step': step_id, 'why': _why4})
            if k2_latch.latched is None:
                _gnow4 = (a.finger_open if node.grip_cmd is None else min(max(float(node.grip_cmd), a.finger_closed), a.finger_open))
                k2_latch.on_msg(f'cmd_entry：{_why4}', step_id, t, _gnow4)
                k2_latch.expire_chain(chain)
                ex.cancel_handover(f'cmd_entry：{_why4}')
                if a.park_gate_mobile:
                    node.k3m_abort_pub.publish(String(data=json.dumps({'why': f'cmd_entry：{_why4}', 't': t, 'source': 'executor_ev4'},
                                                                      default=float)))
                rec['events'].append({'sim_t': t, 'step': step_id, 'k2_abort_latched_by_ev4': _why4})
                print(f'[door] **EV4 命令入口違規**（閂鎖中止）@ step {step_id}：{_why4}', flush=True)
        if node.wb_latest is not None and a.cmd_env and not env_in.plain_allowed():   # EV4 r2：接管後舊路徑命令不得進鏈
            env_stats['n_plain_after_env'] = env_stats.get('n_plain_after_env', 0) + 1
            node.wb_latest = None
            _ev4_violation('plain_cmd_after_takeover')
        if node.wb_latest is not None:
            _wbc = k2_latch.filter_wb(list(node.wb_latest[0]))
            if _wbc is not None:
                chain.receive(_wbc, t)
            node.wb_latest = None
        if a.cmd_env and node.wb_env_latest is not None:                           # EV4：封裝入口；以 recv_seq 對照 source_seq
            _e4, node.wb_env_latest = node.wb_env_latest, None
            _st4 = env_in.receive(_e4, chain, t, filt=k2_latch.filter_wb)
            if _st4.startswith('violation:'):
                _ev4_violation(_st4.split(':', 1)[1])

        _w2sel = None                                                               # EV6 r2：已知中止 → v3 入口 → 請求 → 進鏈
        if w2x is not None:
            _ib6 = []
            while node.w2_inbox:
                _ib6.append(node.w2_inbox.popleft())
            _rq6, node.w2_req_q = list(node.w2_req_q), []
            _w2sel, _relay6, _nv6 = w2x.step_begin(step_id, t, chain, known_abort=_w2_known_abort(), inbox=_ib6, requests=_rq6)
            if w2_xm is not None and (_w2sel.get('event') or {}).get('kind') == 'commit' and w2_auth.mode_seq == 2:
                w2_xm.open(step_id, t)                                                  # EV7：t_open＝mode 2 提交步
            for _r6 in _relay6:                                                    # 權威登記（pending）後才轉送求解端
                node.w2_req_pub.publish(String(data=json.dumps(_r6, default=float)))
            if _nv6:
                _ev4_violation(f"w2_entry:{_nv6[0]['why']}")
            if isinstance(_w2sel.get('fed'), str) and _w2sel['fed'].startswith('violation:'):
                _ev4_violation(_w2sel['fed'].split(':', 1)[1])

        # 交棒請求
        if node.ho_req is not None:
            r = node.ho_req
            node.ho_req = None
            if k2_latch.latched is not None and r.get('action') != 'cancel':
                rec['events'].append({'sim_t': t, 'step': step_id, 'k2': 'handover_refused_after_abort', 'request': r})
                print(f'[door] K2 中止已閂鎖 ⇒ 拒絕換手請求 {r}', flush=True)
            elif park_on and park_state['violation'] is not None and r.get('action') != 'cancel':
                # **違規閂鎖後拒絕所有後續正常換手請求**（不只轉給全身）
                rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'handover_refused_after_violation',
                                      'request': r})
                print(f'[room] PARK 違規已閂鎖 ⇒ 拒絕換手請求 {r}', flush=True)
            elif r.get('action') == 'cancel':
                ex.cancel_handover(r.get('why', ''))
            else:
                ex.request_handover(r.get('to', AUTH_WHOLEBODY),
                                    int(r.get('at_step', step_id + 5)), t,
                                    window_steps=int(
                                        r.get('window_steps', 0)))

        # 導航命令逾時 ⇒ 歸零（失效處置）。
        # **接收時間只在回呼裡更新**，不在這裡刷新 —— 否則導航停止發布後
        # 舊命令會一直被算成新鮮，逾時處置永遠不會觸發。
        node.last_sim_t = t
        nav = node.nav_cmd
        nav_timed_out = False
        if nav is not None and node.nav_cmd_t is not None \
                and t - node.nav_cmd_t > a.cmd_timeout:
            nav = (0.0, 0.0, 0.0)
            nav_timed_out = True
            if not getattr(node, '_to_logged', False):
                node._to_logged = True
                print(f'[room] **導航命令逾時** '
                      f'{t - node.nav_cmd_t:.2f}s > {a.cmd_timeout}s ⇒ '
                      f'歸零（失效處置，不宣稱滿足加速度限制）', flush=True)
        elif nav is not None:
            node._to_logged = False
        # 減速段命令逾時 ⇒ 同樣歸零。它與導航共用直寫路徑，所以失效處置
        # 必須一致，否則兩個來源的逾時語意不同，紀錄就對不起來。
        gld = node.glide_cmd
        glide_timed_out = False
        if gld is not None and node.glide_cmd_t is not None \
                and t - node.glide_cmd_t > a.cmd_timeout:
            gld = (0.0, 0.0, 0.0)
            glide_timed_out = True
            if not getattr(node, '_gto_logged', False):
                node._gto_logged = True
                print(f'[room] **減速段命令逾時** '
                      f'{t - node.glide_cmd_t:.2f}s > {a.cmd_timeout}s ⇒ '
                      f'歸零（失效處置）', flush=True)
        elif gld is not None:
            node._gto_logged = False

        q_meas = robot.get_joint_positions()
        q_arm = [float(q_meas[idx[j]]) for j in ARM]

        # **切換當步的實測速度**要在這裡算好交給執行層 —— 它是唯一在這一步
        # 握有真實狀態的地方。任務節點的量測是數十步之前的。
        _p0, _q0 = robot.get_world_pose()
        _yw0 = yaw_of(_q0)
        _mvb_now = None
        if prev_pose is not None:
            _ddx = float(_p0[0]) - prev_pose[0]
            _ddy = float(_p0[1]) - prev_pose[1]
            _cc, _ss = math.cos(_yw0), math.sin(_yw0)
            _mvb_now = [(_ddx * _cc + _ddy * _ss) / a.physics_dt,
                        (-_ddx * _ss + _ddy * _cc) / a.physics_dt,
                        (_yw0 - prev_pose[2]) / a.physics_dt]
        if park_on:
            # 外部違規（減速段逾時、全身／任務節點）⇒ 同一條閂鎖：取消待提交切換、拒絕後續換手
            if node.park_ext_vio is not None and park_state['violation'] is None:
                _v = {'step': step_id, 't': t, 'phase': 'external',
                      'why': f'external：{node.park_ext_vio.get("why")}', 'source': node.park_ext_vio}
                park_state['violation'] = _v
                park_state['external'] = node.park_ext_vio
                ex.cancel_handover(f'PARK_FIXED 違規：{_v["why"]}')
                rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'violation', **_v})
                print(f'[room] **PARK_FIXED 外部違規**（閂鎖中止）@ step {step_id}：{_v["why"]}', flush=True)
            # 閘門以**本步**實測位姿與上一步實際寫入的底盤命令更新（守門在 ex.step 內核本步）
            if park_hold is None and not park_done and ex.owner != AUTH_WHOLEBODY:
                _stowed = max(abs(float(q_arm[k]) - float(stow[k]))
                              for k in range(len(stow))) <= a.park_stow_tol
                park_gate.update(step_id, t, float(_p0[0]), float(_p0[1]), float(_yw0),
                                 park_prev_cmd, _stowed)
        res = ex.step(step_id, t, a.physics_dt, nav_cmd=nav, glide_cmd=gld,
                      q_arm_measured=q_arm, meas_vb=_mvb_now)
        _asrc = None                                                                # EV4：本步實際套用命令的來源序號
        if a.cmd_env:
            _src4, _mc4 = K3MSRC.applied_source(chain, env_in.map, res.owner == AUTH_WHOLEBODY)
            _asrc = [int(step_id), round(float(t), 6), int(_src4), int(_mc4)]
            _am4 = Float64MultiArray()
            _am4.data = ([float(step_id), float(t), float(_src4), float(_mc4)] + [float(x) for x in res.base_cmd]
                         + [float(x) for x in res.u_applied[3:]])
            node.applied_src_pub.publish(_am4)
            env_stats['n_applied_src_reports'] += 1

        if w2x is not None:                                                         # EV6：鏈的實際來源與限幅後九軸 → 確認
            w2x.post_step(step_id, t, chain, res.owner == AUTH_WHOLEBODY,
                          [float(x) for x in res.base_cmd] + [float(x) for x in res.u_applied[3:]], _w2sel)
            w2x.sync_after(step_id, t, known_abort=_w2_known_abort())                # 本步 select 之後才發現的既有中止
            if w2_effa is not None:                                                     # EV7：5 N 恢復權威（中止優先）
                if w2_auth.aborted is not None or k2_latch.latched is not None:
                    w2_effa.abort('w2_or_k2_abort', step_id, t)
                while node.w2_eff_q:
                    _r7 = node.w2_eff_q.pop(0)
                    _k7 = w2_effa.request(_r7, step_id, t, w2_auth.mode_seq)
                    node.w2_eff_ack_pub.publish(String(data=json.dumps(dict(w2_effa.ack(), result=_k7, step=step_id, t=t), default=float)))
                _fq7b = [float(q_meas[fidx[j]]) for j in FJ] if FJ else [0.0, 0.0]
                _o7 = w2_effa.tick(step_id, t, w2_xm, w2_auth.mode_seq, min(_fq7b) >= 0.95 * a.finger_open)
                if _o7 == 'apply':
                    _rr7 = K3BF.restore(fin_eff, robot.get_articulation_controller(), a.pause_finger_original_effort, step_id, t)
                    w2_effa.confirm(_rr7['ok'], step_id, t)
                    if not _rr7['ok']:
                        _ev4_violation(f"w2_effort_restore:{_rr7['why']}")
                if _o7 is not None:
                    node.w2_eff_ack_pub.publish(String(data=json.dumps(dict(w2_effa.ack(), result=_o7, step=step_id, t=t), default=float)))
            if w2_auth.aborted is not None and not w2_rt['abort_handled']:
                w2_rt['abort_handled'] = True
                if not str(w2_auth.aborted['why']).startswith('external:'):           # 外部中止已有自己的閂鎖，不重記入口違規
                    _ev4_violation(f"w2_mode:{w2_auth.aborted['why']}")
            if step_id % 2 == 0:
                node.w2_state_pub.publish(String(data=json.dumps(w2_auth.state_msg(step_id, t), default=float)))
        if fin_eff.record is not None:                                               # EV5：切換後每 1 s 讀回核對（不改值）
            _pe5 = fin_eff.periodic(robot.get_articulation_controller(), step_id, t)
            if _pe5 is not None and _pe5['ok'] is not True:
                _w1e_abort(f'periodic：{_pe5["why"]}', step_id, t)
        # ---- 真正寫進 API ----
        p_now, qn = robot.get_world_pose()
        yw = yaw_of(qn)
        vx, vy, wz = res.base_cmd
        wx = vx * math.cos(yw) - vy * math.sin(yw)
        wy = vx * math.sin(yw) + vy * math.cos(yw)
        robot.set_linear_velocity(np.array([wx, wy, 0.0], dtype=np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, wz], dtype=np.float32))
        _fcmd = k2_latch.grip(a.finger_open if node.grip_cmd is None
                              else min(max(float(node.grip_cmd), a.finger_closed),
                                       a.finger_open))
        if w2_xm is not None and k2_latch.latched is None and w2_auth.mode_seq >= 1:   # EV7：手指目標由權威模式決定（mode 2 提交步同步張開）
            _fcmd = a.finger_open if w2_auth.finger_target() == 'open' else a.finger_closed
        if res.arm_setpoint is not None:
            tgt = robot.get_joint_positions()
            for k, j in enumerate(ARM):
                tgt[idx[j]] = res.arm_setpoint[k]
            # **兩隻手指都下目標**（URDF 的 mimic 不保證 PhysX 會強制）
            for j in FJ:
                tgt[fidx[j]] = _fcmd
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=tgt))

        # ---- PARK_FIXED：本步的保持監看（同一物理步的真值位姿與實際寫入的底盤命令）----
        _pmode = None
        if park_on:
            _cmd3 = [float(x) for x in res.base_cmd]
            _x, _y = float(p_now[0]), float(p_now[1])
            if park_hold is None and not park_done and res.owner == AUTH_WHOLEBODY:
                # 實際切換給全身的第一步 ⇒ 開始保持監看；錨點＝閘門錨點（之後不更新）；
                # **承接切換前一物理步的位姿**，切換當步的速度才核得到
                if park_gate.anchor is None:
                    park_state['transfer_without_gate'] = True
                park_hold = HoldMonitor(hold_cfg, park_gate.anchor or (_x, _y, float(yw)),
                                        step_id, t, prev=park_last_sample)
                rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'hold_start',
                                      'anchor': park_hold.anchor,
                                      'gate_pass_step': park_gate.pass_step,
                                      'prev_sample': park_last_sample})
            if park_hold is not None and park_hold.end is None:
                if res.owner == AUTH_WHOLEBODY:
                    _v = park_hold.update(step_id, t, _x, _y, float(yw), _cmd3, phase=res.kind)
                elif res.owner == AUTH_NAV:
                    # **交還導航當步**：核最後一個全身控制區間（位姿 k−1→k），不核導航首筆命令；
                    # 交還時手臂須已收攏才算正常結束
                    _v = park_hold.update(step_id, t, _x, _y, float(yw), None, phase='handback')
                    _stowed_hb = max(abs(float(q_arm[k]) - float(stow[k]))
                                     for k in range(len(stow))) <= a.park_stow_tol
                    park_hold.close(step_id, t, stowed=_stowed_hb)
                    park_done = True
                    rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'hold_end',
                                          'verdict': park_hold.verdict(),
                                          'handback_stowed': _stowed_hb})
                else:
                    _v = {'step': step_id, 't': t, 'phase': res.kind,
                          'why': f'保持期內控制者變成 {res.owner}（不是交還導航）'}
                if (_v is not None or park_state['transfer_without_gate']) \
                        and park_state['violation'] is None:
                    _v = _v or {'step': step_id, 't': t, 'phase': res.kind,
                                'why': '未通過停車閘門即切換給全身'}
                    park_state['violation'] = _v
                    # **閂鎖本趟中止**：取消待提交切換、通知所有命令來源停止；不靜默改寫命令
                    ex.cancel_handover(f'PARK_FIXED 違規：{_v["why"]}')
                    node.park_vio_pub.publish(String(data=json.dumps(_v)))
                    rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'violation', **_v})
                    print(f'[room] **PARK_FIXED 違規**（閂鎖中止）@ step {step_id}：{_v["why"]}',
                          flush=True)
            if a.park_gate_mobile:                                                  # EV3（K3-M）：模式權威與逐步走廊監看
                if node.k3m_req_msg is not None:
                    _rq, node.k3m_req_msg = node.k3m_req_msg, None
                    _abt = k2_latch.latched is not None or park_state['violation'] is not None
                    if not k3m['active'] and (park_hold is None or park_hold.end is not None):
                        _kind, _ack = 'reject', {'run_id': a.run_id, 'seq': _rq.get('seq') if isinstance(_rq, dict) else None,
                                                 'why': 'hold_monitor_not_active', 'step': step_id}
                    else:
                        _kind, _ack = k3m_auth.on_request(_rq, step_id, t, aborted=_abt, owner_wholebody=(res.owner == AUTH_WHOLEBODY))
                    if _kind == 'switch':
                        park_hold.close(step_id, t)
                        park_done = True
                        k3m['active'], k3m['switch'] = True, dict(_ack, hold_verdict=park_hold.verdict())
                        rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'hold_end_k3m_switch', 'ack': _ack,
                                              'verdict': park_hold.verdict()})
                        print(f'[door] **K3-M 模式切換**（權威）@ step {step_id}：HoldMonitor 結束、走廊監看啟動', flush=True)
                    k3m['events'].append({'step': step_id, 't': t, 'kind': _kind, 'msg': _ack})
                    node.k3m_ack_pub.publish(String(data=json.dumps(_ack, default=float)))
                if node.k3m_pause_req_msg is not None:                              # EV4：暫停握手（執行端權威；中止優先）
                    _rqp, node.k3m_pause_req_msg = node.k3m_pause_req_msg, None
                    _abtp = k2_latch.latched is not None or park_state['violation'] is not None
                    _kp, _ackp = k3m_pauth.on_request(_rqp, step_id, t, aborted=_abtp, base_mode_free=bool(k3m['active']))
                    if _kp == 'switch':
                        k3m['pause_switch'] = dict(_ackp)
                        print(f'[door] **K3-M 暫停切換**（權威）@ step {step_id}', flush=True)
                        if fin_eff.enabled:                                          # EV5：同一步、world.step 之前
                            _fer = fin_eff.on_pause_switch(robot.get_articulation_controller(), step_id, t)
                            print(f'[door] **W1e 手指驅動輸出上限** {_fer["old"]} → {_fer["set"]}，讀回 {_fer["readback"]}'
                                  f'（{"生效" if _fer["ok"] else "**讀回不符**"}）@ step {step_id}', flush=True)
                            if _fer['ok'] is not True:                               # EV5 r2：不回覆正常暫停 ack
                                _w1e_abort(f'switch：{_fer["why"]}', step_id, t)
                                k3m['pause_switch_w1e_failed'] = dict(_ackp, why=_fer['why'])
                                _kp, _ackp = 'reject', {'run_id': _ackp.get('run_id'), 'seq': _ackp.get('seq'),
                                                        'why': f'w1e_finger_effort：{_fer["why"]}', 'step': step_id}
                    k3m['events'].append({'step': step_id, 't': t, 'kind': 'pause_' + _kp, 'msg': _ackp})
                    node.k3m_pause_ack_pub.publish(String(data=json.dumps(_ackp, default=float)))
                if k3m['active'] and k3m['violation'] is None:
                    _qd3, _ = door_q_qd()
                    _ok3, _e3, _why3 = k3m_cor.check((_x, _y, float(yw)), math.degrees(_qd3), a.corridor_inner_mm * 1e-3,
                                                     math.radians(a.corridor_inner_deg))
                    _anc = park_hold.anchor
                    _ca, _sa = math.cos(_anc[2]), math.sin(_anc[2])
                    _fw, _lt = _ca * (_x - _anc[0]) + _sa * (_y - _anc[1]), -_sa * (_x - _anc[0]) + _ca * (_y - _anc[1])
                    _dy3 = math.degrees(math.atan2(math.sin(float(yw) - _anc[2]), math.cos(float(yw) - _anc[2])))
                    if _ok3 and not (k3m_env[0] <= _fw <= k3m_env[1] and k3m_env[2] <= _lt <= k3m_env[3] and abs(_dy3) <= k3m_env[4]):
                        _ok3, _why3 = False, 'envelope'
                    k3m['log'].append([step_id, round(t, 4), None if _e3 is None else [round(v, 6) for v in _e3],
                                       round(math.degrees(_qd3), 5), [round(_fw, 6), round(_lt, 6), round(_dy3, 5)]])
                    if not _ok3:
                        _v3 = {'step': step_id, 't': t, 'phase': 'k3m', 'why': f'K3-M 執行端監看：{_why3}', 'dev': _e3,
                               'anchor_frame': [_fw, _lt, _dy3]}
                        k3m['violation'] = _v3
                        park_state['violation'] = _v3
                        if k2_latch.latched is None:
                            _gnow3 = (a.finger_open if node.grip_cmd is None else
                                      min(max(float(node.grip_cmd), a.finger_closed), a.finger_open))
                            k2_latch.on_msg(_v3['why'], step_id, t, _gnow3)
                            k2_latch.expire_chain(chain)
                        ex.cancel_handover(_v3['why'])
                        node.k3m_abort_pub.publish(String(data=json.dumps({'why': _v3['why'], 't': t, 'source': 'executor_k3m'}, default=float)))
                        node.park_vio_pub.publish(String(data=json.dumps(_v3, default=float)))
                        rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'violation', **_v3})
                        print(f'[door] **K3-M 執行端越界**（閂鎖中止）@ step {step_id}：{_why3} {_e3}', flush=True)
            park_prev_cmd = tuple(_cmd3)
            park_last_sample = (step_id, t, _x, _y, float(yw))
            _pmode = ('hold' if (park_hold is not None and park_hold.end is None)
                      else 'post' if park_done else 'gate')
            if step_id % 2 == 0:
                _gs = park_gate.state()
                node.park_gate_pub.publish(String(data=json.dumps({
                    'step': step_id, 'sim_t': t, 'passed': _gs['passed'], 'ok_now': _gs['ok_now'],
                    'why_now': _gs['why_now'], 'pass_step': _gs['pass_step'], 'anchor': _gs['anchor'],
                    'mode': _pmode, 'violation': park_state['violation'] is not None})))

        # ---- 套用回報（18 欄契約）----
        _mode = 0 if res.owner == AUTH_WHOLEBODY else 4
        am = Float64MultiArray()
        am.data = ([float(step_id), float(t)]
                   + [float(x) for x in res.u_applied]
                   + [float(_mode), float('nan'),
                      float(getattr(chain, 'n_recv', 0)),
                      float(getattr(chain, 'n_rejected', 0)),
                      1.0 if res.kind in ('nav_applied', 'glide_applied',
                                          'wholebody_applied')
                      else 0.0, -1.0, float('nan')])
        node.applied_pub.publish(am)
        if res.arm_setpoint is not None:
            if not node._sp_meta_sent:
                node._sp_meta_sent = True
                node.sp_meta_pub.publish(String(data=json.dumps({
                    'cols': ['physics_step_id', 'sim_t', 'ready']
                            + [f'sp_{j}' for j in ARM]
                            + ['exec_mode_code', 'api_applied'],
                    'joint_order': list(ARM),
                    'units': 'rad（prismatic 不在此列；手臂六軸皆 revolute）',
                    'sampling_instant': '**本步寫入後**（apply_action 之後）',
                    'physics_dt_s': float(a.physics_dt),
                    'physics_dt_nominal_s': float(a.physics_dt),
                    'pairing': '與同一輪 /joint_states 與 /odom 共用同一個'
                               ' sim_t（三者的 stamp 都由本步的 t 導出）'
                               ' ⇒ 以**共同時間戳**配對；'
                               'JointState 沒有 physics_step_id 欄位，'
                               '不得只靠本說明視為已核對',
                    'recursion': 'act_{i+1} = act_i + α·(sp_i − act_i) + b'
                                 '　⇒ sp_i **先作用於下一物理段**',
                    'G_not_reported': '合成增益 G 由模型端以 α 與 dt 計算；'
                                      '取樣時刻決定用哪一種合成，'
                                      '故契約只需核對取樣時刻與物理步長',
                    'ready_false_means': '設定點**尚未建立**'
                                         '（執行端在第一筆有效命令時由實測'
                                         '關節位置建立）⇒ 求解器**不得**'
                                         '假設 s = q，須走初始化握手',
                    'integration_rate': '設定點每物理步積分一次'
                                        f'（physics_dt，約 '
                                        f'{1.0/max(a.physics_dt,1e-9):.0f} Hz）'
                                        '；20 Hz 只是上游控制頻率',
                    'does_not_change': '本話題為純新增回報，'
                                       '不改既有命令訊息、不改控制行為',
                }, ensure_ascii=False)))
            sm = Float64MultiArray()
            # 末欄是 `api_applied`，**不是常數**：握手閘門把它當 ARM 的必要
            # 條件（`if not s.api_applied: return self._abnormal(...)`）。
            # 原本硬編 0.0 ⇒ 閘門永遠停在 init，求解節點整趟只送零命令
            #（實跑 nav_align_solve_021536：ever_armed = False、published = 0、
            # 近 600 萬筆初始化命令）。這裡回報**本步設定點是否真的寫進 API**，
            # 與 isaac_wholebody_sim_e2.py 同一個語意。
            sm.data = ([float(step_id), float(t), 1.0]
                       + [float(x) for x in res.arm_setpoint]
                       + [float(_mode), 1.0])
            node.sp_pub.publish(sm)

        # ---- **逐物理步證據**：套用命令、實測速度、誰在控制 ----
        # 這一份由模擬器自己記 —— 它每步都有這些值。靠 ROS 訂閱抓會因為
        # 佇列深度與處理速率而只取到樣本（實測 174／3615 筆），
        # 判斷不了「有沒有非預期零命令或跳變」。
        _mvb = None
        if prev_pose is not None:
            _dx = float(p_now[0]) - prev_pose[0]
            _dy = float(p_now[1]) - prev_pose[1]
            _c, _s = math.cos(yw), math.sin(yw)
            _mvb = [(_dx * _c + _dy * _s) / a.physics_dt,
                    (-_dx * _s + _dy * _c) / a.physics_dt,
                    (yw - prev_pose[2]) / a.physics_dt]
        prev_pose = (float(p_now[0]), float(p_now[1]), yw)
        _qd_rec, _qdot_rec = door_q_qd()
        xfc.Clear()
        _hm_s = xfc.GetLocalToWorldTransform(hprim)
        _HM_s = np.array([[float(_hm_s[r][c]) for r in range(4)] for c in range(4)], float)
        _lc = leaf_contacts(_HM_s)
        _wc = wall_contacts()
        _wf = wrist_force()
        if _ev_f is not None:                              # EV 版：同一物理步、同一把手位姿
            _ev_t0 = _ev_time.perf_counter()
            try:
                _ev_write(step_id, t, _HM_s, getattr(res, 'arm_setpoint', None))
            except Exception as _e:                        # 取證不得中斷模擬
                _ev_w.count_err('record')
            _ev_w.add_time(_ev_time.perf_counter() - _ev_t0)
        # 窗口內逐物理步最大值（50 Hz 發布時送出並清空）；讀不到記 invalid，不當成零
        _ok_fb = _lc is not None
        _fb1 = (_lc or {}).get('uflite_finger1', {}).get('bar', 0.0) if _ok_fb else float('nan')
        _fb2 = (_lc or {}).get('uflite_finger2', {}).get('bar', 0.0) if _ok_fb else float('nan')
        _unexp, _unexp_who = 0.0, None
        if _lc is not None:
            for _b, _v in _lc.items():
                _x = _v['panel'] + _v['ambiguous'] + (0.0 if _b in ('uflite_finger1', 'uflite_finger2') else _v['bar'])
                if _x > _unexp:
                    _unexp, _unexp_who = _x, _b
        _wmax = max(_wc.values()) if _wc else (float('nan') if _wc is None else 0.0)
        cw = node.contacts_win
        cw['n'] += 1
        cw['invalid'] += int(_lc is None or _wc is None or _wf is None)
        if _ok_fb:
            cw['fb'] = [max(cw['fb'][0], _fb1), max(cw['fb'][1], _fb2)]
            cw['fb_min'] = [min(cw['fb_min'][0], _fb1), min(cw['fb_min'][1], _fb2)]
        if _unexp > cw['unexp']:
            cw['unexp'], cw['unexp_who'] = _unexp, _unexp_who
        if _wc is not None and _wmax > cw['wall']:
            cw['wall'] = _wmax
        if _wf is not None and _wf > cw['wrist']:
            cw['wrist'] = _wf
        # 與本列 q 同一個物理步。不得在 50 Hz 發布分支內計時。
        # **夾持證據只用確認的橫桿接觸**（不是整個門扇剛體合力；Codex 20261021）
        _contact = (_fb1, _fb2) if FJ else (float('nan'), float('nan'))
        _contact_held = contact_hold.update(
            step_id, t, *_contact,
            closing=(node.grip_cmd is not None
                     and node.grip_cmd < a.finger_open - 1e-12))
        rec['steps'].append([
            step_id, round(t, 4),
            {AUTH_NAV: 0, AUTH_WHOLEBODY: 1}.get(res.owner, 2),
            res.kind, [round(float(x), 6) for x in res.base_cmd],
            None if _mvb is None else [round(x, 6) for x in _mvb],
            [round(float(x), 6) for x in res.u_applied[3:]],
            [round(float(p_now[0]), 6), round(float(p_now[1]), 6),
             round(float(yw), 6)],
            # **重播渲染要的兩個量**：手臂實測關節角與抽屜開度。
            # 邊跑邊渲染會和控制搶 CPU —— 實測 RTF 由 ~1.0 掉到 0.51，
            # 控制週期變少，機器人在**模擬時間**裡也跟著變慢，軌跡就不是
            # 原來那條了。所以物理照原樣跑，渲染另外一趟重播。
            [round(float(x), 6) for x in q_arm],
            round(float(_qd_rec), 8), round(float(_qdot_rec), 8),
            # 手指位置（重播要擺手指）與每指接觸力（判斷夾持與滑脫）。
            # 接觸力只在夾爪被下過命令後才讀，導航期間記 None —— 讀不到
            # 或沒讀**不寫成 0**，淨力為零不等於沒有接觸。
            ([round(float(q_meas[fidx[j]]), 6) for j in FJ] if FJ else None),
            ([round(x, 4) for x in _contact]
             if (FJ and node.grip_cmd is not None) else None),
            float(_contact_held),
            None if _lc is None else {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in _lc.items()},
            None if _wc is None else {k: round(v, 4) for k, v in _wc.items() if v > 0.0},
            None if _wf is None else round(_wf, 4),
            *([_pmode] if park_on else []),
            *([_asrc] if a.cmd_env else [])])                                       # EV4

        if w2_xm is not None and w2_xm.state not in ('IDLE', 'FAILED'):                 # EV7：逐物理步解除監看（本步接觸）
            _fq7 = [float(q_meas[fidx[j]]) for j in FJ] if FJ else None
            _pf7 = (None if _lc is None else [float(_lc.get(_n7, {}).get('bar', 0.0)) + float(_lc.get(_n7, {}).get('ambiguous', 0.0))
                                              for _n7 in ('uflite_finger1', 'uflite_finger2')])
            w2_xm.step(step_id, t, _fq7, _pf7, _lc is not None, door_q=float(_qd_rec))
        # 發布已完成逐步驗收的同時刻資料；不在 world.step 後重讀並混用時間。
        if FJ and t >= next_grip - 1e-9:
            next_grip = max(next_grip + ste * a.physics_dt, t)
            gm = Float64MultiArray()
            gm.data = [t, float(q_meas[fidx[FJ[0]]]),
                       float(q_meas[fidx[FJ[-1]]]), *_contact,
                       float(step_id), float(_contact_held), a.contact_min_n]
            node.grip_pub.publish(gm)

        world.step(render=False)
        # ---- 錄影：到時間才渲染一次（渲染很貴，不是每個物理步都做）----
        if cam is not None and t >= next_cam - 1e-9:
            next_cam = max(next_cam + 1.0 / a.cam_hz, t)
            world.render()
            _qnow, _ = door_q_qd()
            _img = cam.get_rgba()
            if _img is not None and _img.size:
                n_frame += 1
                _imageio.imwrite(
                    os.path.join(cam_dir, f'f_{n_frame:05d}.png'),
                    _img[:, :, :3].astype(np.uint8))
                _ow = {AUTH_NAV: 'nav', AUTH_WHOLEBODY: 'wholebody'}.get(
                    res.owner, 'glide')
                cam_csv.write(f'{n_frame},{t:.4f},{step_id},{_ow},'
                              f'{res.kind},{math.degrees(_qnow):.4f},'
                              f'{float(p_now[0]):.5f},{float(p_now[1]):.5f},'
                              f'{float(yw):.5f}\n')
                cam_csv.flush()
        if w2x is not None:                                                         # EV6 r2：本步稍後（手指讀回、走廊、幾何）才發現的中止記在本步
            w2x.sync_after(step_id, t, known_abort=_w2_known_abort())
        step_id += 1
        node.step_id = step_id
        t_after = float(world.current_time)

        # ---- 感測與狀態 ----
        if t_after >= next_state - 1e-9:
            next_state = max(next_state + ste * a.physics_dt, t_after)
            qm = robot.get_joint_positions()
            js = JointState()
            js.header.stamp = node.stamp(t_after)
            js.name = list(ARM)
            js.position = [float(qm[idx[j]]) for j in ARM]
            node.js_pub.publish(js)
            p2, q2 = robot.get_world_pose()
            od = Odometry()
            od.header.stamp = node.stamp(t_after)
            od.header.frame_id = 'odom'
            od.child_frame_id = 'base_footprint'
            od.pose.pose.position.x = float(p2[0])
            od.pose.pose.position.y = float(p2[1])
            od.pose.pose.orientation.w = float(q2[0])
            od.pose.pose.orientation.x = float(q2[1])
            od.pose.pose.orientation.y = float(q2[2])
            od.pose.pose.orientation.z = float(q2[3])
            node.odom_pub.publish(od)
            if node.tf_pub is not None:
                tfm = TransformStamped()
                tfm.header.stamp = od.header.stamp
                tfm.header.frame_id = 'odom'
                tfm.child_frame_id = 'base_footprint'
                tfm.transform.translation.x = float(p2[0])
                tfm.transform.translation.y = float(p2[1])
                tfm.transform.rotation.w = float(q2[0])
                tfm.transform.rotation.x = float(q2[1])
                tfm.transform.rotation.y = float(q2[2])
                tfm.transform.rotation.z = float(q2[3])
                node.tf_pub.sendTransform(tfm)
            qd_now, qdot_now = door_q_qd()
            xfc.Clear()
            dm = Float64MultiArray()
            dm.data = [float(t_after), float(qd_now), float(qdot_now)]
            node.drawer_pub.publish(dm)
            _hm = xfc.GetLocalToWorldTransform(hprim)
            _hM = np.array([[float(_hm[r][c]) for r in range(4)]
                            for c in range(4)], float)
            hp = Float64MultiArray()
            hp.data = ([float(t_after), float(qd_now)] + [float(x) for x in _hM.reshape(-1)])
            node.handle_pub.publish(hp)
            _yaw2 = math.atan2(2 * (float(q2[0]) * float(q2[3])
                                    + float(q2[1]) * float(q2[2])),
                               1 - 2 * (float(q2[2]) ** 2
                                        + float(q2[3]) ** 2))
            ss = Float64MultiArray()
            # [sim_t, step, q_door, H(16), x, y, yaw, q_arm(6), qdot_door]，共 29 欄
            ss.data = ([float(t_after), float(step_id), float(qd_now)]
                       + [float(x) for x in _hM.reshape(-1)]
                       + [float(p2[0]), float(p2[1]), _yaw2]
                       + [float(qm[idx[j]]) for j in ARM] + [float(qdot_now)])
            node.sync_pub.publish(ss)
            cw = node.contacts_win
            node.contacts_pub.publish(String(data=json.dumps({
                'sim_t': t_after, 'step': step_id, 'n_steps': cw['n'], 'n_invalid': cw['invalid'],
                'finger_bar_max_n': cw['fb'], 'finger_bar_min_n': cw['fb_min'],
                'unexpected_max_n': cw['unexp'], 'unexpected_who': cw['unexp_who'],
                'wall_max_n': cw['wall'], 'wrist_max_n': cw['wrist']}, default=float)))
            node.contacts_win = _new_win()
            hs = String()
            # `pending` 要發出去 —— 任務節點必須能分辨「還在窗口裡等」
            # 與「已被取消」。沒有這個欄位，它只能靠步數猜，就會在執行端
            # 正常等待時誤判為失敗而重試。
            hs.data = json.dumps({'owner': ex.owner, 'step': step_id,
                                  'sim_t': t_after,
                                  'chain_seeded': ex.wb_seeded,
                                  **ex.handover_pending_info(),
                                  'nav_applied': ex.nav_applied_report()},
                                 default=float)
            node.ho_pub.publish(hs)

        if t_after >= next_scan - 1e-9:
            next_scan = max(next_scan + se * a.physics_dt, t_after)
            xfc.Clear()
            o5 = xfc.GetLocalToWorldTransform(
                lidar_prim).ExtractTranslation()
            _, q3 = robot.get_world_pose()
            y5 = yaw_of(q3)
            sc = LaserScan()
            sc.header.stamp = node.stamp(t_after)
            sc.header.frame_id = 'lidar_link'
            sc.angle_min, sc.angle_max = SCAN_MIN, SCAN_MAX
            sc.angle_increment = SCAN_INC
            sc.range_min, sc.range_max = RANGE_MIN, RANGE_MAX
            org = [float(o5[0]), float(o5[1]), float(o5[2])]
            sc.ranges = [ray_range(query, org,
                                   [math.cos(y5 + SCAN_MIN + i * SCAN_INC),
                                    math.sin(y5 + SCAN_MIN + i * SCAN_INC),
                                    0.0], RANGE_MAX, excluded)
                         for i in range(SCAN_N)]
            node.scan_pub.publish(sc)

    # ---- 封存 ----
    rec['summary'] = ex.summary()
    rec['chain'] = chain.summary()
    rec['ros_drain'] = {'max_callbacks_in_one_step': node.n_drain_max,
                        'n_callbacks': node.n_cb}
    rec['config'] = {'finger_collision': finger_col_rep,
                     'door_asset': a.door_asset,
                     'door_asset_sha256': door_asset_sha256,
                     'world': a.world,
                     'start_pose': a.start_pose, 'stow_q': stow,
                     'physics_dt': a.physics_dt,
                     'low_speed_bound': {'lin': a.base_lin_max,
                                         'ang': a.base_ang_max},
                     'low_speed_bound_applies_to': '**只有全身那條路徑**；'
                                                   '導航不經命令鏈',
                     'door': dauth, 'q_door0_rad': q_door0, 'frictionless': fric, 'bar_class_tol': a.bar_class_tol,
                     'wrist_row': {'joint6_row': _j6_row, 'source': 'metadata.joint_indices', 'joint_names': _jnames},
                     'park_cfg': park_cfg.__dict__, 'park_hold_cfg': {k: (str(v) if isinstance(v, float) and math.isinf(v) else v) for k, v in hold_cfg.__dict__.items()},
                     'park_split': '入場閘門＝park_cfg（嚴格靜止）；保持監看＝hold_cfg（20 mm／2°＋伺服上界；實測速率只另報）',
                     'k2_abort_latch': k2_latch.latched,
                     'grip_evidence_source': '確認的橫桿接觸力（classify_contacts bar），非整個門扇剛體合力',
                     'nav_cmd_timeout_s': a.cmd_timeout,
                     'switch_speed_check': {'v_box_lin': a.v_box_lin,
                                            'v_box_ang': a.v_box_ang,
                                            'v_min_lin': a.v_min_lin},
                     'nav_rate_cap': ex.summary()['nav_rate_cap'],
                     'base_rate_cap': {
                         'a_lin': a.base_accel_lin,
                         'a_ang': a.base_accel_ang,
                         'source': 'wgmpc_core.py 的 a_base_lin/a_base_ang，'
                                   '原註標為「**開發值**，無已核准來源」；'
                                   '被執行層採用不改變這個標籤',
                         'per_axis': True,
                         'combined_implication_mps': round(
                             (2 ** 0.5) * max(a.base_accel_lin, 0.0)
                             * a.physics_dt, 6),
                         'n_capped': chain.n_base_rate_capped,
                         'max_cut_mps': round(chain.base_rate_cap_max, 6)},
                     'friction_audit': {'ok': fr_ok,
                                        'n_faces': fr_rep['n_faces'],
                                        'violations': fr_rep['violations']}}
    if cam_csv is not None:
        cam_csv.close()
        print(f'[room] 錄影收尾：共 {n_frame} 幀 → {cam_dir}', flush=True)
    # 相機資料寫出失敗不得阻止 room_run.json 封存
    if _ev_w is not None:                                  # EV 版：取證檔收尾與摘要（開檔失敗也封存錯誤摘要，標為不完整）
        rec['evidence'] = dict(_ev_w.close(), semantics={'contacts': 'get_contact_force_data：逐點法向力、點、法向、分離（法向正負未核，原樣）',
                                                    'friction': 'get_friction_data：逐錨點切向力向量與位置',
                                                    'j6_reaction_wrench': 'get_measured_joint_forces joint6 列：入射關節反作用力＋力矩（子關節座標），非馬達輸出',
                                                    'dof_measured_efforts': 'get_measured_joint_efforts：解算器 DOF 方向關節力，非馬達命令'})
    if a.park_gate_mobile:                                                          # EV3：模式與走廊紀錄
        rec['k3m'] = {k: k3m[k] for k in ('config', 'active', 'switch', 'violation', 'events', 'log')}
        rec['k3m']['rejections'] = k3m_auth.rejections
        rec['k3m']['pause_switch'] = k3m.get('pause_switch')                        # EV4
        rec['k3m']['pause_rejections'] = k3m_pauth.rejections
    rec['finger_effort'] = {'init': fin_eff_init, 'arg_pause_max_effort': a.pause_finger_max_effort,   # EV5
                            'arg_original_effort': a.pause_finger_original_effort,
                            'pause': fin_eff.record, 'pause_switch_failed': k3m.get('pause_switch_w1e_failed'), 'tol': FEFF.TOL}
    if a.cmd_env:                                                                   # EV4：封裝與套用來源統計（含 n_plain_after_env）
        rec['cmd_env'] = dict(env_stats, n_env_recv=env_in.n_recv, n_env_into_chain=env_in.n_into_chain, n_env_dup=env_in.n_dup,
                              takeover_t=env_in.takeover_t, intake_violations=env_in.violations[:50], n_env_bad=getattr(node, 'wb_env_bad', 0), applied_src_cols=['step', 'sim_t', 'source_seq', 'chain_mode'],
                              chain_mode={'0': 'normal', '1': 'timeout', '2': 'stop_unverified', '3': 'fail'},
                              note='source_seq＝−1：本步不是正常套用的全身命令（逾時減速、停止、非全身控制或對照表外）')
    if w2x is not None:                                                             # EV6
        rec['w2'] = {'config': {'w2_mode_run_id': a.w2_mode_run_id, 'run_id': a.run_id, 'run_id_num': w2_auth.run_id_num,
                                'authority_run_id': w2_auth.run_id, 'cand_timeout_s': w2_auth.cand_timeout_s,
                                'cmd_max_age_s': w2_auth.cmd_max_age_s, 'post_commit_timeout_s': w2_auth.post_commit_timeout_s,
                                'applied_timeout_s': w2_auth.applied_timeout_s, 'dt': w2_auth.dt, 'hold_v': w2_auth.hold_v,
                                'hold_w': w2_auth.hold_w, 'arm_tol': w2_auth.arm_tol},
                     'auth': w2_auth.stats(), 'rows': w2x.rows, 'auth_rows': w2_auth.applied_rows, 'recv_log': w2x.recv_log[:20000],
                     'intake_acts': w2x.intake_acts, 'feed': w2x.feed, 'runtime': dict(w2_rt, requests=w2x.requests, relayed=w2x.relayed,
                                                                                     external_abort=w2x.external_abort),
                     'n_v2_ignored': node.w2_n_v2_ignored, 'n_bad_v3': node.w2_n_bad3, 'n_inbox_overflow': node.w2_n_overflow,
                     'cols': {'recv_log': ['source_seq', 'mode_seq', 'stamp_sim_t', 't_recv_sim', 't_recv_wall_mono', 'act'],
                              'rows_time': {'sel_stamp': '選定筆來源 stamp_sim_t（鏈接收時刻）', 'sel_t_recv': '該筆實際進入執行端的模擬時刻'}},
                     'semantics': '鏈接收時刻＝來源 stamp_sim_t；src／mode_seq／chain_mode／u 由鏈讀出（applied_source），非指定'}
    if w2x is not None and w2_xm is not None:                                       # EV7
        rec['w2']['release'] = w2_xm.report()
        rec['w2']['effort'] = {'ack': w2_effa.ack(), 'events': w2_effa.events[:100], 'restore': (fin_eff.record or {}).get('restore')}
    out = os.path.join(a.out, 'room_run.json')
    json.dump(rec, open(out, 'w'), ensure_ascii=False, indent=1, default=str)
    if park_on:
        if park_hold is not None and park_hold.end is None:
            pass                          # 未交還導航 ⇒ HoldMonitor 判定 INSUFFICIENT（不是 PASS）
        json.dump({'spec': ('park_hold_v2_spec.md' if a.park_hold else 'park_fixed_vs_motm_spec.md'),
                   'mode': ('PARK_HOLD' if a.park_hold else 'PARK_FIXED'), 'park_pose': _pk,
                   'cfg': park_cfg.__dict__, 'v_min_lin_set': a.v_min_lin,
                   'v_min_lin_effective': v_min_lin_eff,
                   'gate': park_gate.state(), 'gate_last': park_gate.last,
                   'hold': None if park_hold is None else park_hold.report(),
                   'violation': park_state['violation'],
                   'transfer_without_gate': park_state['transfer_without_gate'],
                   'source': '同一物理步的真值底盤位姿（robot.get_world_pose）與實際寫入 API 的底盤命令（res.base_cmd）'},
                  open(os.path.join(a.out, 'park_gate.json'), 'w'), ensure_ascii=False, indent=1,
                  default=str)
    print(f'[room] 封存 {out}', flush=True)
    print(f'[room] 控制者 {ex.owner}；步數 {step_id}；'
          f'導航命令 {node.n_nav_cmd} 筆', flush=True)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sim_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
