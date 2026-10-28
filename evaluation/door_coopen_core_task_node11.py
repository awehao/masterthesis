#!/usr/bin/env python3
# node11（2026-10-28；Codex 切法 B）：node10 的版本化複本，只加 --release-at-mode3（策略改 door_regrasp_policy8b.RegraspPolicy8B）。
# 核心段任務節點 v10（Codex r63／r64）：v8（台架開門線）與 k3b5（W2／K3-B 線）以 w1b2 為祖先三方合併；衝突 3 處（檔頭、策略建構、紀錄欄位）。
#   策略：--regrasp 時 BG.bench_grasp2(IG2.init_grasp2(AG2.gated2(door_regrasp_policy6.RegraspPolicy6)))——開門前綴的台架初態、INIT_CHECK、
#   接近閘控與 v2 監看照 v8；PAUSE → HELD（W2 mode 1 測試保持）→ 換抓 mode 2–6（policy6：v3 監看器獨立、mode 5 預抓取時鐘、張爪前銜接預檢、
#   登錄核驗）；未給 --regrasp ⇒ 與 v8 同行為；給了 W2／K3-B 旗標卻沒給 --regrasp ⇒ 拒絕（k3b5 的釋放策略分支不採）。
#   RG10：v3 接觸窗口摘要 /door/contact_v3_summary → pol.on_c3；換抓相位（RG*）以策略輸出 coord 取代 W1b phase_coord（後者在 RG 相位為關閉）；
#   task.json 另記 regrasp（策略 rg_report）。
# ---- 以下為 v8 線說明 ----
# 核心段任務節點 v8（door_coopen_core_task_node7.py 的版本化修正；Codex r21 §2；v7 保留）：只改三處（F1–F3），其餘全同 v7：
#   (F1) 起始相位模組換為 door_coopen_bench_grasp2（只在確認凍結時跳過監看；物理前進即監看；無釋放證明 0.2 s 內中止）；
#   (F2) 台架契約換為 door_bench_start2；/bench/state 經 StateFilter 驗本趟 run_id、t0 不變、次序不回退，未通過不採用（記 rejects）；
#   (F3) task.json bench 另記 state_filter（最終狀態、t0、rejects）。
# 核心段任務節點 v7（door_coopen_core_task_node6.py 的版本化複本；Codex r19／r20：台架初態啟動）：只改五處（E1–E5），其餘全同 v6：
#   (E1) 匯入 door_coopen_bench_grasp、door_bench_start；登錄檔須有 bench（BenchConfig 核：六軸有限、s_init＝q_nominal − d_init、|d|≤0.05）⇒ 否則拒跑；
#   (E2) 策略改為 BG.bench_grasp(IG2.init_grasp2(AG2.gated2(PP2.K3MP2Policy)))：起始 BENCH_START（不經 WAIT_HANDOVER／UNFOLD），執行端回報已釋放且模擬時間前進 ⇒ INIT_CHECK；
#   (E3) 訂閱 /bench/state（latched）；每輪 m.bench_state、m.bench_t0；(E4) 停車錨點：/park/gate 的 anchor 缺而 bench 為真時改用 bench_anchor（底盤漂移監看照算）；
#   (E5) task.json 另記 bench（登錄值、模組 sha256、bench_stats、最後收到的 /bench/state）。
# 核心段任務節點 v6（door_coopen_core_task_node5.py 的版本化複本；Codex r16 §2：INIT_CHECK 連續確認 r2）：只改三處（D1–D3），其餘全同 v5：
#   (D1) 初態模組換為 door_coopen_init_grasp2（位置 < 2 mm 且姿態 ≤ 0.02 rad 連續有效 0.5 s；重複／倒退無效；空白 > 0.2 s 重置）；
#   (D2) 策略建構改為 IG2.init_grasp2(AG2.gated2(PP2.K3MP2Policy))；grasp_init.module／module_sha256 記 r2 模組；
#   (D3) trace 每列另記 init（act['init']：valid、why、err_p_m、err_r_rad、good、win_t0；非 INIT_CHECK 輪為 None），供判定器核對閉爪前確認窗。
# 核心段任務節點 v5（door_coopen_core_task_node3.py 的版本化複本；Howard 2026-10-27 確認、Codex r15 補充裁定：指定抓取初態的隔離驗證）：
#   只改五處（B1–B5），其餘全同 v3（**不含 v4 的接近段成本候選**）：
#   (B1) 匯入 door_coopen_init_grasp；(B2) 登錄檔須有 grasp_init（qa_grasp 六值有限、init_check_timeout_s 正）⇒ cfg.timeouts['INIT_CHECK']；缺或不合 ⇒ 拒跑；
#   (B3) 策略改為 IG.init_grasp(AG2.gated2(PP2.K3MP2Policy))：UNFOLD 完成 ⇒ INIT_CHECK（不經 ALIGN／APPROACH）⇒ 位置誤差 < 2 mm 連續 0.5 s ⇒ ENGAGE（K2 原閉爪與接觸證據）；
#   (B4) task.json 另記 grasp_init（登錄值、模組 sha256）；(B5) save() 另記 init_stats。
# 核心段任務節點 v3（door_coopen_core_task_node2.py 的版本化複本；Codex 20261027_004139：閘控 r2）：只把閘控模組換成 door_coopen_approach_gate2（gated2），
#   coopen_gate.module_sha256 記 r2 模組；其餘（含登錄鍵檢查、trace gate、approach_gate_stats）全同 v2。
# 核心段任務節點 v2（door_coopen_core_task_node.py 的版本化複本；Codex 20261027_003024：接近進度閘控）：只改五處，其餘全同 v1：
#   (G1) 匯入 door_coopen_approach_gate；(G2) 策略改為 AG.gated(K3MP2Policy)，閘控設定取自登錄檔 approach_gate（鍵須與 ApproachGateConfig 一致，否則拒跑）；
#   (G3) task.json coopen 另記 approach_gate 設定與模組 sha256；(G4) trace 每列另記 gate（paused、誤差分量、align_ok）；(G5) save() 另記 approach_gate_stats。
#   閘控只改接近參考推進（ALIGN_HOLD 連續對準確認、APPROACH 偏離凍結 s）；控制照常發布、不新增命令來源、既有中止優先。
# 核心段任務節點（door_k3m_task_node_w1b2.py 的版本化複本；規格 results/specs/COOPEN_core_spec_draft2.md，Codex 20261026_235425）：
#   只改三處，其餘（任務邏輯、相位、話題、門檻、紀錄器、引導相位開關）全同 w1b2：
#   (C1) 抓取基準：grasp_basis.args 有 T_HG_full 時取完整 4×4（繞緣夾法），並以 args.a_H 為接近方向傳入策略（K2Policy a_H）；
#        沒有這兩欄時行為同原版（T_HG＝[R_grasp | (0, depth, 0)]、a_H＝(0, −1, 0)）。
#   (C2) 門角參考最大速率：取登錄檔 task_param_extension.r_max_dps（缺省＝K3AConfig 預設 1.0）。
#   (C3) task.json 另記 coopen（基準來源、T_HG_full、a_H、門速、登錄檔雜湊）。
#   --w1b-params 參數名沿用（傳入 coopen_core_params_48_r4.json；鍵與 w1b 登錄同結構）。
# ---- 以下為 k3b5 線說明 ----
# K3-B 任務節點 r5（door_k3m_task_node_k3b4.py 的版本化複本，只插入）：--w2-unload 時策略改 door_k3m_policy_k3b5.K3MReleasePolicy5
#   （核心 door_k3b_release5：終點收斂窗保留窗外最新一筆）。其餘同 k3b4。
# K3-B 任務節點 r4（door_k3m_task_node_k3b3.py 的版本化複本，只插入；Codex 20261026_025647）：--w2-unload（需 --w2-release）時
#   策略改 door_k3m_policy_k3b4.K3MReleasePolicy4(unload=True)：mode 1 後依卸載准入請求 mode 2、mode 3 另等求解端 release_settled、
#   不送 5 N 請求改讀回核對。未給 --w2-unload ⇒ 與 k3b3 同行為。
# K3-B 任務節點 r3（door_k3m_task_node_k3b2.py 的版本化複本，只插入；Codex 20261026_012939）：--w2-release 時策略改 door_k3m_policy_k3b3.K3MReleasePolicy3
#   （core r4：終點收斂以實測 TCP 三維位置核對）。其餘同 k3b2。
# K3-B 任務節點 r2（door_k3m_task_node_k3b.py 的版本化複本，只插入；Codex 20261026_010158）：--w2-release 時策略改 door_k3m_policy_k3b2.K3MReleasePolicy2
#   （core r3：新帶、終點保持逐筆核對；樣本 s_cmd＝上一輪已發布目標；錨點證據）。其餘同 k3b。
# K3-B 任務節點（door_k3m_task_node_w2.py 的版本化複本，只插入；Codex 20261026_000706；規格 K3B_release_clear_spec_draft2.md）：
#   --w2-release（需 --w2-mode-run-id；預設關 ⇒ 與 w2 同行為）：策略換為 door_k3m_policy_k3b.K3MReleasePolicy——測試模式保持後請求 mode 2、
#   RELEASE 相位（解除監看、mode 3 請求、退離目標、5 N 恢復請求、退離終點保持後 DONE；不收臂）。
#   另發 /door/finger_effort_req（latched）、訂 /door/finger_effort_ack；trace 每列另記 k3b；task.json 另記 k3b。
# W2 任務節點（door_k3m_task_node_w1b2.py 的版本化複本，只插入；Codex 20261025_214144；設計 W2_hold_handover_design_draft2）：
#   --w2-mode-run-id（預設空 ⇒ 與 w1b2 同行為）：策略換為 door_k3m_policy_w2.K3MW2Policy——求解端 HELD（觀察通過）後不結束、
#   held_wait 滿 --w2-req-delay-s 送一次模式請求到 /door/w2_mode_req_task（latched；執行端權威登記後才轉送求解端）；
#   只以執行端權威狀態 /door/w2_mode_state 判提交（applied_confirmed）；測試模式保持 --w2-test-hold-s 後 DONE；
#   請求被拒、確認不符、提交逾時、權威中止、狀態停更、HELD 有界超時 ⇒ 中止（有界失敗、不回切）。task.json 另記 w2；trace 每列另記 w2。
#   測試模式不張爪、不退離、不增加角度；監看、門檻、pause_total_s 不變。
# W1b2 任務節點（door_k3m_task_node_w1b.py 的版本化複本；Codex 20261025_034022／20261025_144446：任務端最小紀錄修正）：
#   刪改兩處：(D1) 移除迴圈內「每 100 筆在同一迴圈同步重寫整份 task.json」（w1f_60_1 第 1000 筆後 0.30 s 更新間隙的候選原因，未確認根因）；
#   (D2) 收尾順序改為先送 /wgmpc/stop 與 /room/stop，再有期限（2 s）關閉紀錄器、寫一次完整 task.json（控制停止不等紀錄）。
#   紀錄改用 bounded_trace_writer.BoundedTraceWriter（r2）：每列深拷貝後 put_nowait 進有界佇列（容量 4096），背景執行緒序列化、append 到
#   <out 去 .json>_trace.jsonl、定期 flush、覆寫心跳檔、收尾 flush／close；佇列滿 ⇒ 丟棄計數；寫入／flush／close 失敗 ⇒ incomplete。
#   trace 每列另記 wall_mono（本輪迴圈起點，回呼處理之前的單調時間）、loop_seq、trace_seq；LoopTimer 記每輪（起點→入列）耗時。
#   task.json 另記 trace_log、loop_timing、shutdown（收尾順序與時間）。任務邏輯、策略、話題、門檻全部不變。
# W1b 任務節點（door_k3m_task_node_p2c.py 的版本化複本；Codex 20261024_231908 B 節，規格 W1b_wide_open_30_spec.md）：
#   --w1b-params（results/door/w1b_params_<θ>.json）為唯一參數來源：θ_goal、OPEN 逾時、走廊表（核對 sha256，不符 ⇒ 拒跑）、引導設定與協同話題。
#   引導（door_wide_guidance.CorridorRef，與走廊監看同一參考位姿）：OPEN／SETTLE 且底盤模式已切 free 時每輪發布 w_vref＝1；
#   其他相位（含 PAUSE、ABORTED、DONE）每輪發布 w_vref＝0；計算失敗 ⇒ 發布 0 並記錄。不新增獨立底盤命令來源。
#   trace 每列另記 guide；task.json 另記 w1b（參數、雜湊、引導統計）。其餘同 p2c。
# K3-M 暫停版 p2c 任務節點（door_k3m_task_node_p2b.py 的版本化複本；2026-10-24 Codex 無法使用，經 Howard 授權由 Claude 裁定，
#   見 results/specs/K3M_pause_p2c_handover_precondition.md）：只加**交棒前置條件**——
#   p2b_1／p2b_2 都因展開端暖機命令晚於交棒窗口（命令鏈 n_recv = 0）被執行端取消、WAIT_HANDOVER 逾時。
#   p2c 訂閱 /wb_vel_cmd（只計數，不轉送、不改命令），交棒請求另需 wb_cmd_ready：累計 ≥ WB_READY_N 筆、最新一筆牆鐘年齡 ≤ WB_READY_AGE_S。
#   未就緒就不送請求（WAIT_HANDOVER 30 s 逾時照舊中止）；task.json 另記 handover_precondition。其餘同 p2b。
# K3-M 暫停版 p2b 任務節點（door_k3m_task_node_p2.py 的版本化複本；2026-10-24 Codex 無法使用，經 Howard 授權由 Claude 裁定，
#   見 results/specs/K3M_pause_p2b_delivery_fix.md）：只修 SETTLE 完成訊號的**資料送達**——
#   p2 的窗口掛在深度 1 的 /door/sync_state 訂閱上，兩輪之間只留最新一筆（實收約 12.5 Hz，door_k3m_pause_p2_1）。
#   p2b 另開一個**只餵窗口**的 /door/sync_state 訂閱（深度 64）；原深度 1 的訂閱、SyncTracker 與量測路徑不變；窗口參數不變。
#   trace 每列另記 settle_rx（兩輪之間窗口訂閱實收筆數）；task.json 另記 settle_delivery。
# K3-M 暫停版 p2 任務節點（door_k3m_task_node_p.py 的版本化複本；Codex 20261024_101421 (a)）：
#   策略換為 door_k3m_policy_p2.K3MP2Policy（只換 SETTLE 完成判斷的門速來源）；/door/sync_state 每筆另餵 door_k3m_settle_sig.SettleRateWindow
#   （50 Hz 同步流的 0.1 s 因果後向門角差分），每輪 m.settle_sig＝窗口值（最新樣本必須是本輪步號）；m.qdot_door 不改。
#   trace 每列另記 settle_sig（窗口兩端步號／門角／時間、rate、why）與 settle（各子條件、連續時間）；task.json 另記 k3m_settle。
# K3-M 暫停版任務節點（door_k3m_task_node.py 的版本化複本；規格 K3M_pause_integration_spec_draft1.md）：
#   策略換為 door_k3m_policy_p.K3MPPolicy（完整 SETTLE ⇒ PAUSE、暫停期間監看照舊、求解端 DONE 才 DONE、FAIL ⇒ 中止）；
#   另發 /door/pause_req（latched；策略 pause_outbox，已中止不發），訂閱 /door/pause_ack_exec（執行端，權威）、/door/pause_ack_solver（求解端）、
#   /wgmpc/pause_status（求解端監督狀態）；每輪把目前已發布目標序號交給策略（凍結目標 seq）；task.json 另記 k3m_pause。
#   實驗保護（同長窗版）：/door/lw_guard（獨立觀察器觸發）⇒ 立即結束並記錄；觀察器心跳就緒前不推進，就緒後心跳 > 1.0 s ⇒ 記 lw_observer_unhealthy、有界結束。其餘同原檔。
"""K3-M 任務節點（door_k3a_task_node.py 的版本化複本；K3-A 檔不改；相位機換成 door_k3m_policy：OPEN 起底盤可動協同開門）。
另發 /door/base_mode_req（latched；策略 outbox），訂閱 /door/base_mode_ack（執行端，權威）、/wgmpc/base_mode_ack（求解器）、/park/violation（執行端越界）；
Meas 另帶 state_step、wall、base_pose、qf（底盤＋手臂）、q_finger（兩指平均）、executor_violation；ack 的 t_recv＝收到時最新狀態的模擬時間。
trace 每列另記 k3m_mode、k3m_corr（走廊偏差）、k3m_geom（決策、各類距離與預測、計算耗時）；task.json 另存 k3m（請求、ack、到達時刻、結果）。
原說明：K3-A 任務節點（door_k2_task_node.py 的版本化複本；K2 檔不改）：把量測組成 Meas、執行 door_k3a_policy 動作、寫紀錄。
決策全部在 door_k3a_policy（已離線測試）。K3-A＝夾持 → 開到 8° → SETTLE → 保留抓持受控停止；不張爪、不退離。
話題沿用 K2（執行端與展開端訂閱 /door/k2_abort；不改凍結檔）。

輸入：/door/sync_state（29 欄）、/door/contacts（JSON）、/gripper/state（8 欄）、/handover/state、/wholebody/state、
      /wgmpc/status、/coman/applied_fail、/park/gate。
輸出：/drawer/tcp_target（4×4 列優先，求解端 --target-topic）、/gripper/cmd、/drawer/solver_start、/wgmpc/stop、
      /handover/request（停車閘門通過後送一次）、/door/k2_phase、/room/stop（結束時要求執行端受控停止並封存）。
斜坡速率以**模擬時間**差計算（RTF < 1 時不按牆鐘推進）。牆鐘上限：全趟 600 s、中止後觀察 60 s。

    python3 evaluation/door_k3m_task_node.py --out <run_dir>/task.json --run-id <RUN_ID>
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402
import door_k2_io as KIO                                                  # noqa: E402
import door_k3m_policy as P                                               # noqa: E402
import door_k3m_policy_p as PP                                            # noqa: E402  暫停版策略
import door_k3m_policy_p2 as PP2                                          # noqa: E402  p2：SETTLE 完成訊號
import door_k3m_policy_w2 as PW2                                          # noqa: E402  W2：HELD_WAIT／模式請求
import door_k3m_settle_sig as SSIG                                        # noqa: E402
import door_wide_guidance as WG                                          # noqa: E402  W1b

WB_READY_N = 20                            # p2c：展開端 20 Hz ⇒ 約 1 s 的連續暖機輸出
WB_READY_AGE_S = 0.2
import door_k3ab_lw_guard as G                                            # noqa: E402  觀察器心跳監看
import door_k3m_corridor as K3MCOR                                        # noqa: E402
import door_k3m_geom_monitor as K3MGEO                                    # noqa: E402
import bounded_trace_writer as BTW                                        # noqa: E402  W1b2：有界非阻塞紀錄
import door_target_k3 as T3                                              # noqa: E402
import door_coopen_approach_gate as AG                                    # noqa: E402  G1（設定類別）
import door_coopen_approach_gate2 as AG2                                  # noqa: E402  v3：閘控 r2
import door_coopen_init_grasp2 as IG2                                     # noqa: E402  B1／D1：指定抓取初態 r2
import door_coopen_bench_grasp2 as BG                                     # noqa: E402  E1／F1：台架初態起始相位 r2
import door_bench_start2 as BENCH                                         # noqa: E402  E1／F2：台架契約 r2
import door_regrasp_policy8 as RP8                                        # noqa: E402  RG10：換抓＋關門策略 r8（mode 1–8）
import door_regrasp_policy8b as RP8B                                      # noqa: E402  node11：切法 B 策略
import door_k3m_geom_monitor_v3 as GM3                                    # noqa: E402  RG10：v3 監看後端（雙把手）
import door_regrasp_ref_registry as REGY                                  # noqa: E402  RG10：參考包登錄核驗
from physics_contact_hold import contact_established                     # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]


class Node2(Node):
    def __init__(self):
        super().__init__('door_k3m_task')
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tgt_pub = self.create_publisher(Float64MultiArray, '/drawer/tcp_target', 10)
        self.grip_pub = self.create_publisher(Float64, '/gripper/cmd', 10)
        self.start_pub = self.create_publisher(Bool, '/drawer/solver_start', lat)
        self.stop_pub = self.create_publisher(String, '/wgmpc/stop', 10)
        self.ho_pub = self.create_publisher(String, '/handover/request', 10)
        self.phase_pub = self.create_publisher(String, '/door/k2_phase', 10)
        self.room_stop_pub = self.create_publisher(String, '/room/stop', 10)
        # **K2 中止閂鎖**（latched）：展開端與執行端都訂閱；送出後不收回
        self.abort_pub = self.create_publisher(String, '/door/k2_abort', lat)
        self.sync_tr = KIO.SyncTracker()
        self.settle_win = SSIG.SettleRateWindow()                      # p2
        self.ct_agg = KIO.ContactsAggregator()
        self.sync = None          # (t, step, q, H, pose(3), q_arm(6), qdot)
        self.contacts = None      # dict＋接收時的模擬時間
        self.gs = None
        self.hs = None
        self.wb = None
        self.ws_t = None          # 最後一筆 /wgmpc/status 的模擬時間
        self.n_ws = 0
        self.fail = None
        self.gate = None
        self.create_subscription(Float64MultiArray, '/door/sync_state', self._sync, 1)
        self.win_rx = 0                                                    # p2b：窗口訂閱實收（累計）
        self.win_bad = 0
        self.create_subscription(Float64MultiArray, '/door/sync_state', self._sync_win, 64)   # p2b：只餵 SETTLE 窗口
        self.wb_cmd_n, self.wb_cmd_wall, self.wb_cmd_t_first = 0, None, None                   # p2c：展開端命令（只計數）
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd', self._wb_seen, 10)
        self.create_subscription(String, '/door/contacts', self._ct, 10)
        self.bench = None                                                                     # E3
        self.bench_filter = None                                                              # F2：main() 依 run_id 建立
        self.create_subscription(String, BENCH.STATE_TOPIC, self._bench_st, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                                                                      durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pol = None
        self.exec_violation = None
        self.mode_msgs = []
        self.mode_req_pub = self.create_publisher(String, '/door/base_mode_req', lat)
        self.create_subscription(String, '/door/base_mode_ack', self._mode_exec, lat)
        self.create_subscription(String, '/wgmpc/base_mode_ack', self._mode_solver, lat)
        self.create_subscription(String, '/park/violation', self._pviol, 10)
        self.create_subscription(Float64MultiArray, '/gripper/state', self._gs, 1)
        self.create_subscription(String, '/handover/state', self._hs, 1)
        self.create_subscription(String, '/wholebody/state', self._wb, 10)
        self.create_subscription(Float64MultiArray, '/wgmpc/status', self._ws, 1)
        self.create_subscription(String, '/coman/applied_fail', self._af, 10)
        self.create_subscription(String, '/park/gate', self._gate, 1)
        self.pause_msgs = []                                                      # 暫停版
        self.pause_req_pub = self.create_publisher(String, '/door/pause_req', lat)
        self.coord_pub = None                                              # W1b：main() 依登錄檔建立
        self.create_subscription(String, '/door/pause_ack_exec', self._pause_exec, lat)
        self.create_subscription(String, '/door/pause_ack_solver', self._pause_solver, lat)
        self.create_subscription(String, '/wgmpc/pause_status', self._pause_status, lat)
        self.lw_guard = self.lw_guard_wall = None                                 # 實驗保護（獨立觀察器）
        self.hbw = G.HeartbeatWatch()
        self.create_subscription(String, '/door/lw_observer/heartbeat', self._lw_hb, 10)
        self.create_subscription(String, '/door/lw_guard', self._lw_guard, lat)

    def _sync(self, m):
        d = [float(x) for x in m.data]
        if len(d) != 29:
            self.sync_tr.anomalies.append(('bad_length', time.monotonic()))
            return
        if self.sync_tr.on_msg(d[1], d[0], time.monotonic()):       # 只接受步號嚴格遞增的新狀態
            self.sync = (d[0], d[1], d[2], np.asarray(d[3:19]).reshape(4, 4), d[19:22], np.asarray(d[22:28]), d[28])

    def _wb_seen(self, m):
        if len(m.data) < 9:
            return
        self.wb_cmd_n += 1
        self.wb_cmd_wall = time.monotonic()
        if self.wb_cmd_t_first is None:
            self.wb_cmd_t_first = self._t_now()

    def wb_cmd_ready(self, wnow):
        """p2c 交棒前置條件：展開端已持續送出命令（≥ WB_READY_N 筆）且仍在送（最新一筆 ≤ WB_READY_AGE_S）。"""
        return self.wb_cmd_n >= WB_READY_N and self.wb_cmd_wall is not None and wnow - self.wb_cmd_wall <= WB_READY_AGE_S

    def _sync_win(self, m):
        d = m.data
        if len(d) != 29:
            self.win_bad += 1
            return
        self.win_rx += 1
        self.settle_win.on_sync(float(d[0]), int(d[1]), float(d[2]), time.monotonic())   # 重送／倒退／非有限由窗口拒收並計數

    def _ct(self, m):
        try:
            d = json.loads(m.data)
        except Exception:
            d = {}
        self.ct_agg.on_msg(d, time.monotonic())                     # 未消費的窗口全部合併（短脈衝不被覆蓋）

    def _t_now(self):
        return float('nan') if self.sync is None else float(self.sync[0])

    def _mode_exec(self, m):
        try:
            d = json.loads(m.data)
        except Exception:
            d = None
        self.mode_msgs.append({'src': 'executor', 't_recv': self._t_now(), 'wall': time.monotonic(), 'msg': d})
        if self.pol is not None:
            self.pol.on_executor_ack(d, self._t_now(), time.monotonic())

    def _mode_solver(self, m):
        try:
            d = json.loads(m.data)
        except Exception:
            d = None
        self.mode_msgs.append({'src': 'solver', 't_recv': self._t_now(), 'wall': time.monotonic(), 'msg': d})
        if self.pol is not None:
            self.pol.on_solver_ack(d, self._t_now(), time.monotonic())

    def _pviol(self, m):
        self._pviol0(m)

    def _lw_hb(self, m):
        try:
            d = json.loads(m.data)
        except Exception:
            d = None
        self.hbw.on_beat(d, time.monotonic())

    def _lw_guard(self, m):
        if self.lw_guard is None:
            self.lw_guard, self.lw_guard_wall = str(m.data), time.monotonic()

    def _pause_msg(self, m, src):                                                 # 暫停版：ack 與狀態
        try:
            d = json.loads(m.data)
        except Exception:
            d = None
        self.pause_msgs.append({'src': src, 't_recv': self._t_now(), 'wall': time.monotonic(), 'msg': d})
        if len(self.pause_msgs) > 2000:
            self.pause_msgs = self.pause_msgs[:200] + self.pause_msgs[-1000:]
        return d

    def _pause_exec(self, m):
        d = self._pause_msg(m, 'executor')
        if self.pol is not None:
            self.pol.on_pause_executor(d, self._t_now(), time.monotonic())

    def _pause_solver(self, m):
        d = self._pause_msg(m, 'solver')
        if self.pol is not None:
            self.pol.on_pause_solver(d, self._t_now(), time.monotonic())

    def _pause_status(self, m):
        try:
            d = json.loads(m.data)
        except Exception:
            d = None
        self.pause_last_status = d
        if self.pol is not None:
            self.pol.on_pause_status(d)

    def _pviol0(self, m):
        try:
            self.exec_violation = json.loads(m.data)
        except Exception:
            self.exec_violation = {'raw': str(m.data)[:200]}

    def _gs(self, m):
        d = list(m.data)
        self.gs = tuple(float(x) for x in d) if len(d) == 8 else None

    def _hs(self, m):
        try:
            self.hs = json.loads(m.data)
        except Exception:
            pass

    def _wb(self, m):
        try:
            self.wb = json.loads(m.data)
        except Exception:
            pass

    def _ws(self, m):
        self.n_ws += 1
        self.ws_t = time.monotonic()                                # 獨立牆鐘

    def _af(self, m):
        self.fail = str(m.data)

    def _bench_st(self, m):                                                                   # E3／F2
        try:
            msg = json.loads(m.data)
        except Exception:
            msg = None
        if self.bench_filter is not None and self.bench_filter.on_msg(msg):
            self.bench = {'state': self.bench_filter.state, 't0': self.bench_filter.t0}

    def _gate(self, m):
        try:
            self.gate = json.loads(m.data)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--tcp', default='link_tcp')
    ap.add_argument('--grasp-basis', default=os.path.join(HERE, 'runs', 'phf_01_M', 'task.json'))
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--corridor-table', default=os.path.join(HERE, 'results', 'door', 'door_k3m_corridor_table.json'))
    ap.add_argument('--w1b-params', required=True)
    ap.add_argument('--geom-config', default=os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json'))
    ap.add_argument('--wall-total-s', type=float, default=600.0)
    ap.add_argument('--wall-post-abort-s', type=float, default=60.0)
    ap.add_argument('--lw-ready-timeout-s', type=float, default=60.0)         # 等觀察器就緒（牆鐘）
    ap.add_argument('--lw-heartbeat-max-age-s', type=float, default=1.0)      # 觀察器心跳最大年齡（牆鐘）
    ap.add_argument('--w2-mode-run-id', default='')                            # W2：空 ⇒ 關閉（w1b2 行為）
    ap.add_argument('--w2-req-delay-s', type=float, default=1.0)
    ap.add_argument('--w2-commit-timeout-s', type=float, default=2.0)
    ap.add_argument('--w2-test-hold-s', type=float, default=3.0)
    ap.add_argument('--w2-state-stale-s', type=float, default=1.5)
    ap.add_argument('--w2-release', action='store_true')                         # K3-B：需 --w2-mode-run-id
    ap.add_argument('--w2-unload', action='store_true')                          # K3-B r4：需 --w2-release
    ap.add_argument('--release-at-mode3', action='store_true')                 # node11：切法 B（mode 2 夾持靜止確認、mode 3 張爪＋退離）
    ap.add_argument('--regrasp', action='store_true')                            # RG10：換抓 mode 2–6（需 --w2-mode-run-id、--regrasp-ref）
    ap.add_argument('--regrasp-ref', default='')                                 # RG10：已登錄參考包
    ap.add_argument('--regrasp-geom-self-mode', default='full', choices=('full', 'sphere'))   # RG10：v3 監看自碰模式（離線核對用 full）
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f'[k3a] {a.out} 已存在 ⇒ 拒絕覆寫', flush=True)
        return 2
    gb = json.load(open(a.grasp_basis))['args']
    T_HG = np.eye(4)
    T_HG[:3, :3] = np.asarray(gb['R_grasp'], float)
    T_HG[:3, 3] = [0.0, float(gb['grasp_depth_m']), 0.0]
    a_H = (0.0, -1.0, 0.0)
    if 'T_HG_full' in gb:                                                  # C1：繞緣夾法完整抓取基準與接近方向
        T_HG = np.asarray(gb['T_HG_full'], float)
        a_H = tuple(float(v) for v in gb['a_H'])
        if T_HG.shape != (4, 4) or not np.all(np.isfinite(T_HG)) or abs(np.linalg.det(T_HG[:3, :3]) - 1.0) > 1e-6 or abs(np.linalg.norm(a_H) - 1.0) > 1e-6:
            raise SystemExit('核心段：T_HG_full／a_H 不合法 ⇒ 拒跑')
    w1b = json.load(open(a.w1b_params))                                  # W1b：唯一參數來源
    _tp = w1b['task_param_extension']
    _ps = w1b['protection_scope_change']
    _tbl = os.path.join(WS, _ps['corridor_table'])
    if os.path.realpath(a.corridor_table) != os.path.realpath(_tbl) or hashlib.sha256(open(a.corridor_table, 'rb').read()).hexdigest() != _ps['corridor_table_sha256']:
        raise SystemExit('W1b：--corridor-table 與登錄檔的走廊表或 sha256 不符 ⇒ 拒跑')
    cfg = PP.K3MPConfig(theta_goal_deg=float(_tp['theta_goal_deg']), over_goal_deg=float(_tp['over_goal_deg']),
                        r_max_dps=float(_tp.get('r_max_dps', PP.K3MPConfig.r_max_dps)))   # C2
    cfg.timeouts['OPEN'] = float(_tp['open_timeout_s'])
    _gi = w1b.get('grasp_init')                                            # B2
    if not (isinstance(_gi, dict) and isinstance(_gi.get('qa_grasp'), list) and len(_gi['qa_grasp']) == 6 and all(math.isfinite(float(v)) for v in _gi['qa_grasp'])
            and isinstance(_gi.get('init_check_timeout_s'), (int, float)) and math.isfinite(float(_gi['init_check_timeout_s'])) and float(_gi['init_check_timeout_s']) > 0):
        raise SystemExit('核心段 v5：登錄檔 grasp_init 缺或不合法 ⇒ 拒跑')
    cfg.timeouts['INIT_CHECK'] = float(_gi['init_check_timeout_s'])
    try:                                                                   # E1
        BENCH.BenchConfig(w1b.get('bench'), 0.05)
    except (ValueError, TypeError) as e:
        raise SystemExit(f'核心段 v7：登錄檔 bench 不合法（{e}）⇒ 拒跑')
    w1b_gcfg = WG.GuidanceCfg(**w1b['control_cost_change']['guidance'])   # r2：不與下方幾何設定 _gc 同名（Codex 20261024_234630 第 1 項）
    corridor = K3MCOR.Corridor.from_json(json.load(open(a.corridor_table))['table'])
    _gc = json.load(open(a.geom_config))
    gcfg = K3MGEO.GeomMonitorConfig(v_floor=dict(_gc['v_floor_registered']), ds_err=dict(_gc['downsampling']['ds_err_registered']))
    _agp = w1b.get('approach_gate')                                        # G2
    if not isinstance(_agp, dict) or set(_agp) != set(AG.ApproachGateConfig.__dataclass_fields__):
        raise SystemExit('核心段 v2：登錄檔 approach_gate 缺或鍵不符 ⇒ 拒跑')
    if (a.w2_mode_run_id or a.w2_release or a.w2_unload) and not a.regrasp:   # RG10：v10 的 W2 只用於換抓
        raise SystemExit('任務節點 v10：--w2-mode-run-id／--w2-release／--w2-unload 只在 --regrasp 下使用 ⇒ 拒跑')
    if a.regrasp:                                                          # RG10：換抓策略（台架開門前綴的包裝鏈不變）
        if not a.w2_mode_run_id or not a.regrasp_ref:
            raise SystemExit('--regrasp 需 --w2-mode-run-id 與 --regrasp-ref ⇒ 拒跑')
        if a.w2_release or a.w2_unload:
            raise SystemExit('--regrasp 與 --w2-release／--w2-unload 互斥 ⇒ 拒跑')
        rg_pkg = json.load(open(a.regrasp_ref))
        _okr, _whyr = REGY.validate(rg_pkg)
        if not _okr:
            raise SystemExit(f'換抓參考包登錄核驗不過：{_whyr} ⇒ 拒跑')
        rg_fn = GM3.backend_v3(self_mode=a.regrasp_geom_self_mode)
        pol = BG.bench_grasp2(IG2.init_grasp2(AG2.gated2(RP8B.RegraspPolicy8B if a.release_at_mode3 else RP8.RegraspPolicy8)))(
            cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()), a_H=a_H,
            w2_run_id=a.w2_mode_run_id, req_delay_s=a.w2_req_delay_s, commit_timeout_s=a.w2_commit_timeout_s,
            test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s,
            ref_pkg=rg_pkg, kin=WholeBodyKinematics.from_urdf_file(a.urdf), geom_v3_fn=rg_fn)
    else:
        pol = BG.bench_grasp2(IG2.init_grasp2(AG2.gated2(PP2.K3MP2Policy)))(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()), a_H=a_H)   # C1／G2
    pol.gate_init(AG.ApproachGateConfig(**{k: float(v) for k, v in _agp.items()}))
    K = WholeBodyKinematics.from_urdf_file(a.urdf)
    rep = {'schema': 'door_k3m_task_p2/1', 'args': vars(a), 'config': dataclasses.asdict(cfg), 'geom_config': dataclasses.asdict(gcfg),
           'corridor_sha256': hashlib.sha256(open(a.corridor_table, 'rb').read()).hexdigest(),
           'geom_config_sha256': hashlib.sha256(open(a.geom_config, 'rb').read()).hexdigest(),
           'grasp_basis': {'path': os.path.relpath(a.grasp_basis, WS), 'sha256': hashlib.sha256(open(a.grasp_basis, 'rb').read()).hexdigest(),
                           'T_HG': T_HG.tolist()},
           'policy_sha256': hashlib.sha256(open(P.__file__, 'rb').read()).hexdigest(),
           'target_sha256': hashlib.sha256(open(T3.__file__, 'rb').read()).hexdigest(),
           'parent_policy_sha256': hashlib.sha256(open(P.P.__file__, 'rb').read()).hexdigest(),
           'mode_sha256': hashlib.sha256(open(P.MODE.__file__, 'rb').read()).hexdigest(),
           'corridor_module_sha256': hashlib.sha256(open(K3MCOR.__file__, 'rb').read()).hexdigest(),
           'geom_module_sha256': hashlib.sha256(open(K3MGEO.__file__, 'rb').read()).hexdigest(), 'trace': [], 'events': []}
    rep['handover_precondition'] = {'n_required': WB_READY_N, 'max_age_s': WB_READY_AGE_S}
    rep['policy_p2_sha256'] = hashlib.sha256(open(PP2.__file__, 'rb').read()).hexdigest()
    rep['bench'] = {'config': w1b['bench'], 'module_sha256': {'bench_grasp': hashlib.sha256(open(BG.__file__, 'rb').read()).hexdigest(),
                                                              'bench_start': hashlib.sha256(open(BENCH.__file__, 'rb').read()).hexdigest()}}   # E5
    rep['grasp_init'] = {'config': dict(_gi), 'module_sha256': hashlib.sha256(open(IG2.__file__, 'rb').read()).hexdigest(), 'module': 'door_coopen_init_grasp2'}   # B4／D2
    rep['coopen_gate'] = {'config': vars(pol.gcfg), 'module_sha256': hashlib.sha256(open(AG2.__file__, 'rb').read()).hexdigest(), 'module': 'door_coopen_approach_gate2'}   # G3／v3
    rep['coopen'] = {'grasp_basis_has_full': 'T_HG_full' in gb, 'T_HG': T_HG.tolist(), 'a_H': list(a_H), 'r_max_dps': cfg.r_max_dps,
                     'params_path': os.path.relpath(a.w1b_params, WS), 'params_sha256': hashlib.sha256(open(a.w1b_params, 'rb').read()).hexdigest()}   # C3
    if a.w2_mode_run_id:                                                   # W2
        rep['policy_w2_sha256'] = hashlib.sha256(open(PW2.__file__, 'rb').read()).hexdigest()
    rep['w1b'] = {'params_path': os.path.relpath(a.w1b_params, WS), 'params_sha256': hashlib.sha256(open(a.w1b_params, 'rb').read()).hexdigest(),
                  'guidance_module_sha256': hashlib.sha256(open(WG.__file__, 'rb').read()).hexdigest(), 'guidance_cfg': vars(w1b_gcfg),
                  'theta_goal_deg': cfg.theta_goal_deg, 'open_timeout_s': cfg.timeouts['OPEN'], 'n_on': 0, 'n_off': 0, 'n_err': 0, 'errors': []}
    _gref = WG.CorridorRef(corridor)
    rep['settle_sig_sha256'] = hashlib.sha256(open(SSIG.__file__, 'rb').read()).hexdigest()
    _sw = SSIG.SettleRateWindow()                                          # 與節點內窗口同預設值
    rep['settle_sig_config'] = {'window_s': _sw.window_s, 'gap_max_s': _sw.gap_max_s, 'stale_wall_s': _sw.stale_wall_s}

    def save():
        rep['policy_events'] = pol.events
        rep['approach_gate_stats'] = dict(pol.gate_stats)                     # G5
        rep['init_stats'] = dict(pol.init_stats)                              # B5
        rep['bench']['stats'] = dict(pol.bench_stats)                         # E5
        rep['bench']['last_state'] = nd.bench
        rep['bench']['state_filter'] = (None if nd.bench_filter is None else
                                        {'state': nd.bench_filter.state, 't0': nd.bench_filter.t0, 'rejects': dict(nd.bench_filter.rejects)})   # F3
        rep['final_phase'], rep['abort'], rep['max_abs_q_door_rad'] = pol.phase, pol.abort, pol.max_abs_q
        rep.setdefault('n_target_published', 0)
        rep['open_entry'], rep['max_grasp_drift'] = pol.open_entry, {'pos_m': pol.max_drift[0], 'rot_rad': pol.max_drift[1]}
        W = pol.waiter
        rep['k3m'] = {'mode_state': pol.mode_state, 'request': W.request, 'exec_ack': W.exec_ack, 'solver_ack': W.solver_ack,
                      't_exec_recv': W.t_exec, 't_solver_recv': W.t_solver, 'wall_req': W.wall_req, 'wall_exec_recv': W.wall_exec,
                      'wall_solver_recv': W.wall_solver, 't_recv_semantics': '收到時最新同步狀態的來源模擬時間（非獨立量測）',
                      'result': W.result, 'n_geom_warn': pol.n_geom_warn,
                      'mode_msgs': nd.mode_msgs, 'executor_violation': nd.exec_violation}
        PW = pol.pwaiter                                                          # 暫停版
        rep['k3m_pause'] = {'pause_state': pol.pause_state, 'request': PW.request, 'exec_ack': PW.exec_ack, 'solver_ack': PW.solver_ack,
                            't_exec_recv': PW.t_exec, 't_solver_recv': PW.t_solver, 'wall_req': PW.wall_req,
                            'wall_exec_recv': PW.wall_exec, 'wall_solver_recv': PW.wall_solver, 'result': PW.result,
                            'pause_t0': pol.pause_t0, 'outcome': pol.pause_outcome,
                            'last_status': getattr(nd, 'pause_last_status', None), 'msgs': nd.pause_msgs[-300:]}
        rep['handover_precondition'].update(n_wb_cmd=nd.wb_cmd_n, wb_cmd_t_first=nd.wb_cmd_t_first)
        rep['settle_delivery'] = {'variant': 'p2b_dedicated_subscription', 'depth': 64, 'n_rx': nd.win_rx, 'n_bad_length': nd.win_bad}
        rep['k3m_settle'] = {'resets': pol.settle_resets[-500:], 'n_resets': len(pol.settle_resets),
                             'sig_anomalies': dict(nd.settle_win.anomalies)}
        tmp = a.out + '.tmp'
        json.dump(rep, open(tmp, 'w'), ensure_ascii=False, indent=1, default=float)
        os.replace(tmp, a.out)

    _tw_base = a.out[:-5] if a.out.endswith('.json') else a.out                 # W1b2
    _tw2 = BTW.BoundedTraceWriter(_tw_base + '_trace.jsonl', capacity=4096, heartbeat_path=_tw_base + '_trace_hb.json').start()
    _lt2 = BTW.LoopTimer(over_s=0.1)
    _loop_seq2 = 0
    rclpy.init()
    nd = Node2()
    nd.bench_filter = BENCH.StateFilter(a.run_id)                           # F2
    nd.pol = pol
    if a.w2_mode_run_id:                                                   # W2：請求（給執行端權威）與權威狀態
        _lat_w2 = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        nd.w2_req_pub = nd.create_publisher(String, '/door/w2_mode_req_task', _lat_w2)
        nd.w2_bad_state = 0

        def _w2_state_cb(m_):
            try:
                pol.on_w2_state(json.loads(m_.data), nd._t_now())
            except Exception:                                              # noqa: BLE001
                nd.w2_bad_state += 1
        nd.create_subscription(String, '/door/w2_mode_state', _w2_state_cb, 50)
        if a.regrasp:                                                      # RG10：執行端 v3 接觸窗口摘要
            nd.c3_bad = 0
            nd.c3_n = 0

            def _c3_cb(m_):
                try:
                    pol.on_c3(json.loads(m_.data), nd._t_now())
                    nd.c3_n += 1
                except Exception:                                          # noqa: BLE001
                    nd.c3_bad += 1
            nd.create_subscription(String, '/door/contact_v3_summary', _c3_cb, 50)
        if a.w2_release:                                                   # K3-B：5 N 恢復請求與 ack
            nd.eff_req_pub = nd.create_publisher(String, '/door/finger_effort_req', _lat_w2)
            nd.eff_bad_ack = 0

            def _eff_ack_cb(m_):
                try:
                    pol.on_effort_ack(json.loads(m_.data), nd._t_now())
                except Exception:                                          # noqa: BLE001
                    nd.eff_bad_ack += 1
            nd.create_subscription(String, '/door/finger_effort_ack', _eff_ack_cb, _lat_w2)
    w0 = time.monotonic()
    t_prev = None
    ho_sent = False
    w_abort = None
    dt_w = 1.0 / a.rate
    last_phase = None
    abort_sent = False
    sync_once = False
    last_ct = None
    n_pub = 0                              # 已發布目標的序號（判定器對帳用）
    rx_prev = 0                            # p2b
    ho_wait_t0 = None                      # p2c
    g_qref_prev = None                     # W1b
    try:
        while rclpy.ok():
            _w2_loop0 = time.monotonic()                                       # W1b2 r2：本輪迴圈起點（回呼處理之前）
            _loop_seq2 += 1
            for _ in range(64):
                rclpy.spin_once(nd, timeout_sec=0.0)
            wnow = time.monotonic()
            if nd.lw_guard is not None:                                       # 實驗保護 ⇒ 提前結束（不是成功停住）
                rep['lw_guard'] = {'msg': nd.lw_guard, 'wall_recv': nd.lw_guard_wall}
                break
            nd.hbw.max_age_s = a.lw_heartbeat_max_age_s
            if not nd.hbw.ready():
                if wnow - w0 > a.lw_ready_timeout_s:
                    rep['lw_observer_not_ready'] = {'wall_waited_s': wnow - w0}
                    break
                time.sleep(dt_w)
                continue
            if not nd.hbw.healthy(wnow):
                rep['lw_observer_unhealthy'] = {'wall': wnow, 'last_beat_wall': nd.hbw.last_wall, 'last_seq': nd.hbw.last_seq,
                                                'age_s': wnow - nd.hbw.last_wall, 'phase': pol.phase}
                break
            if wnow - w0 > a.wall_total_s and pol.phase not in ('ABORTED', 'DONE'):
                rep['wall_timeout'] = True
                pol._abort('wall_total_timeout', t_prev if t_prev is not None else float('nan'))
                w_abort = w_abort or wnow
            if pol.phase == 'ABORTED' and not abort_sent:
                nd.abort_pub.publish(String(data=json.dumps({'why': pol.abort, 't': pol.t_abort}, default=float)))
                nd.stop_pub.publish(String(data=f'door_k3a 中止：{pol.abort}'))
                abort_sent = True
                w_abort = w_abort or wnow
            if nd.sync is None:
                # 從未收到同步狀態：只能等牆鐘上限；中止後的觀察亦受牆鐘上限
                if w_abort is not None and wnow - w_abort > min(a.wall_post_abort_s, 5.0):
                    rep['ended_without_sync'] = True
                    break
                time.sleep(dt_w)
                continue
            sync_once = True
            t, step, q, H, pose, q_arm, qdot = nd.sync
            _ctn = nd.ct_agg.consume()
            if _ctn is not None:
                last_ct = _ctn                     # 新窗口（已合併所有未消費窗口）
            ct = last_ct                           # 無新窗口 ⇒ 沿用上一份；是否過期由獨立牆鐘年齡判斷（contacts_age）
            gate = nd.gate or {}
            anchor = gate.get('anchor')
            if anchor is None and gate.get('bench') is True:                 # E4
                anchor = gate.get('bench_anchor')
            if anchor is not None:
                bd = math.hypot(pose[0] - anchor[0], pose[1] - anchor[1])
                byaw = math.atan2(math.sin(pose[2] - anchor[2]), math.cos(pose[2] - anchor[2]))
            else:
                bd = byaw = None
            T_tcp = K.fk(np.array(list(pose) + list(q_arm), float), a.tcp)
            gs = nd.gs
            ev = contact_established(gs, now=t, close_started=pol.close_t0, contact_min_n=cfg.contact_min_n,
                                     hold_s=cfg.attach_hold_s) if (gs is not None and pol.close_t0 is not None) else False
            st_age = nd.sync_tr.age(wnow)
            if not nd.sync_tr.ok():
                st_age = float('inf')                                 # 倒退／非有限 ⇒ 監看失效
            m = P.Meas(t=t, owner=(nd.hs or {}).get('owner'),
                       unfold_done=(nd.wb or {}).get('phase') in ('UNFOLD_DONE', 'RELEASED'),
                       solver_up=nd.n_ws > 0, solver_failed=nd.fail,
                       solver_status_age=(None if nd.ws_t is None else wnow - nd.ws_t), state_age=st_age,
                       contacts_age=nd.ct_agg.age(wnow),
                       q_door=q, qdot_door=qdot, H=H, T_tcp=T_tcp,
                       finger_bar_min=None if ct is None else tuple(ct['finger_bar_min_n']),
                       finger_bar_max=None if ct is None else tuple(ct['finger_bar_max_n']),
                       unexpected_max=None if ct is None else ct['unexpected_max_n'], unexpected_who=None if ct is None else ct['unexpected_who'],
                       wall_max=None if ct is None else ct['wall_max_n'], wrist_max=None if ct is None else ct['wrist_max_n'],
                       contacts_invalid=None if ct is None else ct['n_invalid'], contact_evidence_ok=bool(ev),
                       finger_q=None if gs is None else (gs[1], gs[2]), base_drift_m=bd, base_drift_yaw=byaw)
            m.state_step, m.wall = int(step), wnow                   # K3-M：模式請求、走廊與幾何監看所需
            m.base_pose = tuple(float(x) for x in pose)
            m.qf = np.array(list(pose) + list(q_arm), float)
            m.q_finger = None if gs is None else 0.5 * (float(gs[1]) + float(gs[2]))
            m.executor_violation = nd.exec_violation
            m.settle_sig = nd.settle_win.value(int(step), wnow)      # p2：SETTLE 完成訊號（只在 SETTLE 使用）
            settle_rx, rx_prev = nd.win_rx - rx_prev, nd.win_rx       # p2b：兩輪之間窗口訂閱實收筆數
            dt = 0.0 if t_prev is None else max(0.0, t - t_prev)
            t_prev = t
            pol.pub_seq = n_pub                                # 暫停版：凍結目標序號＝目前已發布序號
            m.bench_state, m.bench_t0 = (nd.bench or {}).get('state'), (nd.bench or {}).get('t0')   # E3
            act = pol.step(m, dt)
            # W1b：引導只在 OPEN／SETTLE 且底盤已切 free（door_wide_guidance.phase_coord）
            if nd.coord_pub is None:
                nd.coord_pub = nd.create_publisher(Float64MultiArray, w1b['control_cost_change']['coord_topic'], 1)
            _qr = pol.q_ref if pol.q_ref is not None else q
            _rate = 0.0 if (g_qref_prev is None or dt <= 0) else (_qr - g_qref_prev) / dt
            _coord, _ginfo, _gerr = WG.phase_coord(_gref, w1b_gcfg, pol.phase, pol.mode_state, m.base_pose, q, _rate)
            rep['w1b']['n_on' if _ginfo is not None else 'n_off'] += 1
            if _gerr is not None:
                rep['w1b']['n_err'] += 1
                if len(rep['w1b']['errors']) < 20:
                    rep['w1b']['errors'].append({'t': t, 'why': _gerr})
            g_qref_prev = pol.q_ref
            if a.regrasp and str(pol.phase).startswith('RG') and act.get('coord') is not None:   # RG10：換抓段由策略給底盤引導
                _coord = list(act['coord'])
                rep.setdefault('regrasp_coord', {'n': 0})['n'] += 1
            nd.coord_pub.publish(Float64MultiArray(data=[float(x) for x in _coord]))
            for _rq in pol.take_outbox():                      # 已中止 ⇒ take_outbox 清除、不發布
                nd.mode_req_pub.publish(String(data=json.dumps(_rq, default=float)))
                rep['events'].append({'t': t, 'k3m_mode_request_published': _rq})
            for _rq in pol.take_pause_outbox():                # 暫停版：已中止 ⇒ 清除、不發布
                nd.pause_req_pub.publish(String(data=json.dumps(_rq, default=float)))
                rep['events'].append({'t': t, 'k3m_pause_request_published': _rq})
            if a.w2_mode_run_id:                                               # W2：模式請求（已中止 ⇒ 清除不發）
                for _rq in pol.take_w2_outbox():
                    nd.w2_req_pub.publish(String(data=json.dumps(_rq, default=float)))
                    rep['events'].append({'t': t, 'w2_mode_request_published': _rq})
            if a.w2_release:                                                   # K3-B：5 N 恢復請求（已中止 ⇒ 清除不發）
                for _rq in pol.take_eff_outbox():
                    nd.eff_req_pub.publish(String(data=json.dumps(_rq, default=float)))
                    rep['events'].append({'t': t, 'k3b_effort_request_published': _rq})
            _g = pol.geom_last
            if act['request_handover'] and not ho_sent and gate.get('passed') and gate.get('ok_now') and ho_wait_t0 is None:
                ho_wait_t0 = t                                                   # p2c：閘門通過、開始等展開端命令
            if act['request_handover'] and not ho_sent and gate.get('passed') and gate.get('ok_now') and nd.wb_cmd_ready(wnow):
                rep['handover_precondition'].update(ready_t=t, wait_s=t - ho_wait_t0, n_wb_cmd_at_request=nd.wb_cmd_n,
                                                    wb_cmd_age_wall=wnow - nd.wb_cmd_wall)
                nd.ho_pub.publish(String(data=json.dumps({'to': 'wholebody', 'at_step': int(step) + 5, 'window_steps': 50})))
                ho_sent = True
                rep['events'].append({'t': t, 'handover_requested_at_step': int(step) + 5, 'gate': gate})
            if pol.phase == 'ABORTED' and not abort_sent:
                nd.abort_pub.publish(String(data=json.dumps({'why': pol.abort, 't': pol.t_abort}, default=float)))
                abort_sent = True
                w_abort = w_abort or wnow
            pub_seq = None
            if act['target'] is not None and pol.phase != 'ABORTED':
                nd.tgt_pub.publish(Float64MultiArray(data=[float(x) for x in np.asarray(act['target']).reshape(-1)]))
                n_pub += 1
                pub_seq = n_pub
                rep['n_target_published'] = n_pub
            nd.grip_pub.publish(Float64(data=float(act['grip'])))
            if act['solver_start'] and pol.phase != 'ABORTED':
                nd.start_pub.publish(Bool(data=True))
            if act['stop_solver']:
                nd.stop_pub.publish(String(data=f'door_k3a {pol.phase}：{pol.abort or "done"}'))
            if pol.phase != last_phase:
                nd.phase_pub.publish(String(data=pol.phase))
                print(f'[k3a] 相位 → {pol.phase} @ sim {t:.2f}' + (f'（中止：{pol.abort}）' if pol.abort else ''), flush=True)
                last_phase = pol.phase
            if pol.H_frozen is not None and 'H_frozen' not in rep:
                rep['H_frozen'] = pol.H_frozen.tolist()
                rep['H_frozen_t'] = t
            if pol.G_T_H is not None and 'G_T_H' not in rep:
                rep['G_T_H'] = pol.G_T_H.tolist()
                rep['G_T_H_t'] = t
            rep['trace'].append({'t': t, 'step': step, 'phase': pol.phase, 'q_door': q, 'qdot_door': qdot, 's': pol.s, 'grip': act['grip'],
                                 'target': (None if act['target'] is None or pol.phase == 'ABORTED' else np.asarray(act['target']).reshape(-1).tolist()),
                                 'err_p': (None if pol.last_target is None else float(np.linalg.norm(T_tcp[:3, 3] - pol.last_target[:3, 3]))),
                                 'pub_seq': pub_seq, 'q_ref': pol.q_ref, 'ref': (None if pol.ref_info is None else {k: pol.ref_info.get(k) for k in ('q_ref_prev', 'rate', 'lo', 'hi', 'init')}),
                                 'base_drift_m': bd, 'base_drift_yaw': byaw, 'contacts': ct, 'evidence': bool(ev),
                                 'state_age_wall': st_age, 'contacts_age_wall': nd.ct_agg.age(wnow),
                                 'state_step': int(step), 'k3m_mode': pol.mode_state,
                                 'guide': {'w': _coord[1], 'v_body': _coord[2:5], 'err': None if _ginfo is None else list(_ginfo['err'])},
                                 'init': act.get('init'),                                       # D3
                                 'settle_sig': m.settle_sig, 'settle_rx': settle_rx, 'settle': None if pol.settle_last is None or pol.phase not in ('SETTLE', 'PAUSE') else
                                 {k: v for k, v in pol.settle_last.items() if k != 'sig'},
                                 'k3m_corr': None if pol.corr_last is None else [float(v) for v in pol.corr_last],
                                 'k3m_geom': None if _g is None else {'dec': _g[0], 'why': _g[1].get('why'), 'compute_s': _g[1].get('compute_s'),
                                                                     'cls': {k: [v['d'], v['d_pred']] for k, v in (_g[1].get('classes') or {}).items()}},
                                 'gate': act.get('gate')})                                                   # G4
            rep['trace'][-1].update(wall_mono=_w2_loop0, loop_seq=_loop_seq2, trace_seq=len(rep['trace']))   # W1b2
            if a.w2_mode_run_id:                                               # W2
                rep['trace'][-1]['w2'] = pol.w2_state
            if a.w2_release:                                                   # K3-B
                rep['trace'][-1]['k3b'] = {'state': pol.k3b.state, 's_cmd': pol.rel['s_cmd'], 'rel': pol.rel['state']}
            _tw2.put(rep['trace'][-1])                                          # W1b2：有界非阻塞逐列紀錄（取代每 100 筆同步 save）
            _lt2.record(_w2_loop0, time.monotonic(), _loop_seq2)                # W1b2 r2：本輪起點（回呼處理前）至入列；結束輪也計入
            if act['finish'] or (w_abort is not None and wnow - w_abort > a.wall_post_abort_s):
                if w_abort is not None and not act['finish']:
                    rep['post_abort_wall_cap_hit'] = True
                break
            time.sleep(dt_w)
    finally:
        rep['wall_s'] = round(time.monotonic() - w0, 2)
        rep['lw_heartbeat'] = {'ready_wall': nd.hbw.ready_wall, 'last_seq': nd.hbw.last_seq, 'last_wall': nd.hbw.last_wall, 'bad': nd.hbw.bad[:20]}
        rep['sync_anomalies'] = {'n_dup': nd.sync_tr.n_dup, 'n_back': nd.sync_tr.n_back, 'list': nd.sync_tr.anomalies[:20]}
        rep['received_any_sync'] = sync_once
        _w2_sd = {'order': ['stop_and_room_stop_published', 'trace_writer_closed', 'task_json_saved', 'spin_flush']}   # W1b2 r2：收尾順序
        try:
            nd.stop_pub.publish(String(data='door_k3a 結束'))
            nd.room_stop_pub.publish(String(data=f'door_k3a 結束：{pol.phase} {pol.abort or ""}'))
            _w2_sd['stop_published_wall'] = time.monotonic()
        except Exception as _e2:                                                # noqa: BLE001
            _w2_sd['stop_publish_error'] = f'{type(_e2).__name__}: {_e2}'
        rep['trace_log'] = _tw2.close(deadline_s=2.0)                           # W1b2 r2：呼叫端只做有期限等待（不做 I/O）
        rep['trace_log']['jsonl_semantics'] = '每行 {enq_seq, enq_wall_mono, row}；row＝該筆 trace（含 wall_mono／loop_seq／trace_seq）'
        rep['loop_timing'] = dict(_lt2.stats(), semantics='每筆 trace 一輪：本輪起點（回呼處理前）至入列完成', n_loops_total=_loop_seq2)
        _w2_sd['writer_closed_wall'] = time.monotonic()
        if a.w2_mode_run_id:                                                   # W2
            rep['w2'] = dict(pol.w2_report(), n_bad_state=getattr(nd, 'w2_bad_state', None))
        if a.regrasp:                                                          # RG10
            rep['regrasp'] = dict(pol.rg_report(), n_c3=getattr(nd, 'c3_n', None), n_c3_bad=getattr(nd, 'c3_bad', None),
                                  ref=os.path.relpath(a.regrasp_ref, WS), ref_sha256=hashlib.sha256(open(a.regrasp_ref, 'rb').read()).hexdigest())
        if a.w2_release:                                                       # K3-B
            rep['k3b'] = dict(pol.k3b_report(), n_bad_ack=getattr(nd, 'eff_bad_ack', None))
        rep['shutdown'] = _w2_sd
        save()
        try:
            for _ in range(10):
                rclpy.spin_once(nd, timeout_sec=0.05)
        except Exception:
            pass
        print(f'[k3a] 結束：{pol.phase}' + (f'（中止 {pol.abort}）' if pol.abort else ''), flush=True)
        nd.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if pol.phase == 'DONE' else 1


if __name__ == '__main__':
    sys.exit(main())
