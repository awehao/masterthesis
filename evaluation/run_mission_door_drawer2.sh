#!/usr/bin/env bash
# r3＝run_mission_door_drawer2（2026-10-30；mission_dd_1／2 抽屜段零命令真因＝我漏傳 fingers 致 R2 包絡 fail-closed）：r2 的複本，只把執行端換 ev9r8u（door_drawer_phase2）。
#   離門短段、/gmpc/diag、退出碼照 r2。用法：RUN_ID=mission_dd_3 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_mission_door_drawer2.sh bash evaluation/run_guarded_v3.sh
# run_mission_door_drawer（2026-10-30；門→抽屜同趟串接 S1，設計 results/specs/door_drawer_chain_s1_r1.md；Codex m5a）：run_door_m4r14.sh 的複本，只改：
#   世界 mission_room.sdf（門座標；隔牆與門資產牆不重疊）＋門＋抽屜櫃（2.4,5.05）；地圖 mission_room.yaml；執行端 ev9r8t --with-drawer --subtask-handoff；
#   任務節點 node10o --subtask-mode；模擬上限 1400 s。門段 DONE＋執行端「門段交接」後：門段節點（solver／wholebody／nav1／lw_observer／gmpc）受控結束、
#   核對門段發布者全數消失，才起抽屜段：GMPC 以抽屜定版參數重啟（heading 開；機器人靜止）、cmd_relay（取代 nav1 選擇器 CRUISE 轉送）、
#   drawer_glide／drawer_wholebody／drawer_task／arm_vel_adapter（consumer /door_k2_sim）／wgmpc_wg2／drawer_mission（MOTM；常數平移到門座標；起點＝/odom 實際位姿；收臂 STOW_F）。
# r2（2026-10-30；mission_dd_1 抽屜段導航命令全零、起點距門扇 0.514 m < 動態 CBF 邊距 0.69；Codex m5c 核可 A＋B）：
#   抽屜段 GMPC 起動即錄 /gmpc/diag（gmpc_diag_d.log）；relay 後先跑離門短段 nav_escape_plan_node（實際位姿沿 +y 0.4 m、GMPC＋CBF 照舊、期限 40 s 模擬時間），
#   成功才以離門後實測位姿起抽屜段節點，失敗即 /room/stop；退出碼依抽屜任務 DONE、編排 rc 0、模擬器正常結束判定（不再無條件 0）。
# 用法：RUN_ID=mission_dd_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_mission_door_drawer.sh bash evaluation/run_guarded_v3.sh
# m4r14（2026-10-30；door_m4r_15 回報門速不可靠＋逾時路徑 TypeError）：run_door_m4r13.sh 的複本，只改：執行端 ev9r8r → ev9r8s、任務節點 node10m → node10n（close_settle r2）。
# 用法：RUN_ID=door_m4r_16 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r14.sh bash evaluation/run_guarded_v3.sh
# m4r13（2026-10-30；door_m4r_14 /clock 超前容許過緊誤中止）：run_door_m4r12.sh 的複本，只把任務節點 node10l → node10m（AHEAD 0.2 s）。
# 用法：RUN_ID=door_m4r_15 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r13.sh bash evaluation/run_guarded_v3.sh
# m4r12（2026-10-30；door_m4r_13 RG8 放手時門仍滑動 ⇒ rg_door_moved；Codex m4r 選 A）：run_door_m4r11.sh 的複本，只改：執行端 ev9r8q → ev9r8r、任務節點 node10k → node10l
#   （拉關終點停穩 |qdot| ≤ 0.002 rad/s 連續 0.5 s 才放手；兩端都核）。
# 用法：RUN_ID=door_m4r_14 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r12.sh bash evaluation/run_guarded_v3.sh
# m4r11（2026-10-30；door_m4r_11 RG7 主機停頓 ⇒ rg_monitor_stale；Codex m4p 核可）：run_door_m4r10.sh 的複本，只改：任務節點 node10j → node10k
#   （RG* 相位監看年齡改模擬時間基準＋1.0 s 牆鐘看門狗；只限模擬）；標頭參考包字樣改由 $REF 實際值印出（原「ref pb1g」為沿用文字）。
# 用法：RUN_ID=door_m4r_12 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r11.sh bash evaluation/run_guarded_v3.sh
# m4r10（2026-10-30；door_m4r_10 第一段導航啟動競態）：run_door_m4r9.sh 的複本，只改：nav1 r9（訂閱者增加時重發計畫）；
#   平滑器生命週期 active 檢查（同 run_drawer_mission.sh：讀狀態手動 configure／activate，仍非 active 拒跑 86）。
# 用法：RUN_ID=door_m4r_11 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r10.sh bash evaluation/run_guarded_v3.sh
# m4r9（2026-10-30；Howard 門開大一點；Codex 20261030_001949 核可 60° 整合版一趟）：run_door_m4r8.sh 的複本，只改：
#   開門登錄 pullback_params_60_r1（θ 60°、包絡／觀察器 64°／逾時取自 w1f_params_60_r2）；參考包 pb60h（60° 換抓＋R1–R3 v6、0.9×時間化）；
#   模擬時間上限 560 → 800 s（執行端 --sim-limit、求解端 --duration-sim-s；60° 開門與拉關較長）。其餘同 m4r8。
# 用法：RUN_ID=door_m4r_9 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r9.sh bash evaluation/run_guarded_v3.sh
# m4r8（2026-10-30；door_m4r_7 R3 base_box；Codex 20261030_000022 修正版 A）：run_door_m4r6.sh 的複本，只把執行端換成 ev9r8q --base-box-snap。
#   （run_door_m4r7.sh／ADM24 的求解端裁切經 Codex 判定不對症，未使用。）
# 用法：RUN_ID=door_m4r_8 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r8.sh bash evaluation/run_guarded_v3.sh
# m4r6（2026-10-29；door_m4r_6 R2 固定點停住＝第二段折線未加密＋GMPC vertex 投影）：run_door_m4r5.sh 的複本，只把 nav1 換成 r8（第二段折線每段加密 ≤ 0.05 m）。
#   以 run_guarded_v3.sh 執行。其餘與 m4r5 相同。
# 用法：RUN_ID=door_m4r_7 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r6.sh bash evaluation/run_guarded_v3.sh
# m4r5（2026-10-29；door_m4r_4 第二段 GMPC 輸出凍結；Codex 20261029_230142 核可 A＋B）：run_door_m4r4.sh 的複本，只改：
#   A nav1 r7（第一段交棒後發一次空計畫，GMPC 閒置、ξ_prev 歸零）；B GMPC wheel_fit_mode:=project（既有輸出整形，速度／加速度／輪速／CBF／護盾不變）；
#   /gmpc/diag 以 ros2 topic echo --csv 存 gmpc_diag.log（CSV；含最終回傳命令 CBF 殘差欄）；第二段快照改由 nav1 print 行（非日誌行）第 2 次 CRUISE 觸發。
# 用法：RUN_ID=door_m4r_5 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r5.sh bash evaluation/run_guarded.sh
# m4r4（2026-10-29；door_m4r_3 後發現：各趟導航的 ros2 run 子程序未被收尾，累積 88 個孤兒在 domain 97，干擾導航）：run_door_m4r3.sh 的複本，只改程序管理：
#   (1) 每個 spawn 以 setsid 開新 session（PGID＝登錄 PID）；(2) cleanup 對整個程序群組 TERM → 3 s → KILL；
#   (3) 起跑前掃描衝突程序（同 ROS_DOMAIN_ID 或無 domain 設定，且帶 --ros-args 或屬已知節點：模擬、求解、任務、轉接、nav1、展開、觀察器、GMPC），有即拒跑（exit 99），不自動清除；
#   (4) Codex 20261029_183843：procs.tsv 記每個登錄程序的 PID／PGID；收尾只對 PGID＝登錄 PID 的群組動作，之後逐群組核殘留；
#       關鍵話題發布者（/cmd_vel、/cmd_vel_pre_mux、/cmd_vel_pre_shield、/cmd_vel_nav、/scan、/tf、/tf_static、/plan）身分與數量於起跑後與第二段導航開始時各存一次（topic_pubs_*.txt）。
#   控制、保護、參數與 m4r3 完全相同。
# 用法：RUN_ID=door_m4r_4 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r4.sh bash evaluation/run_guarded.sh
# m4r3（2026-10-29；door_m4r_2 R3 銜接只剩橫向 0.109 m）：run_door_m4r2.sh 的複本，只把 nav1 換成 r6（第二段 GLIDE 橫向上限 0.010→0.03 m/s、
#   HANDOVER_READY 另需橫向 ≤ 0.03 m）。其餘配置、保護、溫度守護同 door_m4r_2。
# 用法：RUN_ID=door_m4r_3 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r3.sh bash evaluation/run_guarded.sh
# m4r2（2026-10-29；Codex 20261029_174949 採 A／175625 同意 A′，核可 door_m4r_2）：run_door_m4r.sh 的複本，只換參考包 pb1h（導航第二段折線＝保留段→圓弧 r0.15→直線進場 0.506 m；
#   R3 含 0.40 m 前置直線）與任務端 node10j（policy2：前置區間投影接點、Δ0 自接手衰減、R2_HO 預覽 R3 目標）。其餘配置、保護、溫度守護同 door_m4r_1。
# 用法：RUN_ID=door_m4r_2 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r2.sh bash evaluation/run_guarded.sh
# mode 4 重設計 R1–R3 版（2026-10-29；Codex 20261029_145418／145654／151139／154350 核可接線準備；**未核可實跑**）：run_door_nav1_door_roll.sh r2 的複本，改為
#   換抓 mode 4＝R1（W-GMPC 收臂＋轉向＋依朝向前進）→ 交給導航（包絡准入、導航先備妥）→ R2（GMPC＋CBF＋護盾，朝前繞過門扇）→ GLIDE 交回全身（新任期）→
#   R3（W-GMPC 接近＋連續轉入操作朝向）→ Gate5 → 既有 mode 5 起。參考包 pb1g；執行端 ev9r8p --tenure-loop；求解端 ADM23 --tenure-loop；
#   adapter k3m4 --cmd-env-v4；任務端 node10i --m4r；nav1 r5（第二段任務）。
#   導航包絡一致化（Codex 145654）：護盾 robot_radius 0.30→0.39、self_return_radius 0.30（自體過濾不隨保護半徑擴大）；
#   GMPC static_cbf_safe_margin 0.38→0.47、cbf_safe_margin 0.60→0.69（保留原額外間距＋0.09）。第一段導航同樣套用（D1 對關門淨空 0.505 m，護盾低速停止距離約 0.49 m）。
#   與 roll r2 的差異：**不在交棒後停掉導航計算**（R2 需要）。
# 用法：RUN_ID=door_m4r_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_m4r.sh bash evaluation/run_guarded.sh
# 以下為 run_door_nav1_door_roll.sh 說明。
# 滾動版（2026-10-29；Codex 20261029_090630 選 B，核可一次功能趟）：run_door_nav1_door.sh r2 的複本，改為
#   朝前導航（折線 S→P0→D1，最後一段沿操作朝向）→ GLIDE 平順減速至入口速度框 → 底盤仍移動時滾動交棒 → 展開節點 --motm 移動中展臂 →
#   W-GMPC 底盤自由＋approach coord 協同接近（ALIGN／APPROACH）→ ENGAGE 前減速、seq 1 hold（錨點＝實測）→ 閉爪 → 既有後退開門（seq 2 free）→ 換抓…
#   執行端 ev9r8o --rolling-handover --gate6-log；求解端 ADM22 --rolling-approach；任務端 node10h --rolling；nav1 r4 --rolling；展開節點 --motm（取代 --park-hold）。
#   approach_vref 參數三處一致：v_cap 0.030、a_ref 0.002、k_yaw 1.0、w_cap 0.10、v_creep 0、overshoot 0（抽屜為 0.004／0.030：門端要在 hold 前停下，揭露）。
#   保護：接近段＝執行端接近走廊（線段偏離 ≤50 mm、偏航 ≤5°）＋任務端線段偏離代入 base_drift（20 mm／2°）＋幾何／接觸／資料年齡；hold 後同既有 PARK_HOLD；OPEN 起同既有。
# r2（Codex 20261029_123819，自 door_nav1_roll_3 起）：交棒確認後（nav1 記 HANDED_OVER＝其觀察到執行端 /handover/state owner＝wholebody）由跑法以**任務安排停止**結束
#   本趟登錄的 gmpc、tracker、smoother、smlife（TERM 本程序及其直接子程序；不用廣域 pkill、不用 SIGSTOP）；保留 scan relay、TF、地圖、護盾、nav1（零輸出）、觀察器。
#   導航 CBF 設定不關閉（導航有權威時才作用；全身接手期間由操作保護負責）。NAV2 前須重新啟動導航計算（本趟不含 NAV2）。
# 用法：RUN_ID=door_nav1_roll_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_nav1_door_roll.sh bash evaluation/run_guarded.sh
# 以下為 run_door_nav1_door.sh 說明。
# NAV1＋門段同趟（2026-10-29；Codex 20261029_081047／081705 核可單跑一次新 RUN_ID）：起點 S → NAV1 → W1 停位 D1 精停 → PARK 閘門＋權威交棒 →
#   全身展開（K2_READY→Q_PRE）→ W1 正面實際接近、閉爪取證 → 既有門段（往後拉開 48°→換抓→穿門拉關至約 2°→放手退出）。
# 與 extf_leader_sc5 的差異（其餘旗標、SETTLE／HoldConfirm 語義、主從夾爪、ref pb1f、ADM21 全同）：
#   - 執行端 ev9r8n --nav-start --pub-tf true --start-pose S（取代 --bench-start／--bench-params／--bench-setpoint；--stow-q＝K2_READY）；gc.freeze 於主迴圈前一次。
#   - 任務端 node10g --nav-start（非台架換抓鏈，WAIT_HANDOVER 起；--wait-handover-timeout-s 240）；不帶 --bench-wait-contacts。
#   - 求解端 ADM21 去 --bench-start／--bench-params，改 --wait-for-drawer-handover（等展開節點握手，同 w1f2）；轉接器去 --bench-ready-run-id。
#   - 新增展開節點 door_k2_wholebody_node（同 w1f2）。
#   - 新增導航（照抄 run_drawer_phase1_nav.sh 定版參數）：map_server（maps/door_scene_v3.yaml，由 generate_door_scene_map.py 自門資產幾何生成）、
#     scan_relay（/scan_raw→/scan）、scan_obstacle_tracker、gmpc_node（GMPC＋horizon CBF α 0.5／動態 0.60／靜態 0.38）→ /cmd_vel_nav →
#     velocity_smoother → /cmd_vel_pre_mux → **door_nav1_mission_node 選擇器**（CRUISE 轉送｜PARK_ALIGN P 伺服 ≤0.03 m/s、≤0.1 rad/s｜ZERO）→
#     /cmd_vel_pre_shield → scan_safety_shield（掃描 barrier）→ /cmd_vel。
# 揭露：/odom 真值（無 AMCL）；直線計畫（非 nav2 planner）；精停段＝P 伺服＋掃描 barrier（不是 GMPC）；本場景無移動／未知障礙（本輪不加）；門外 A 房無外牆。
# r2（Codex 20261029_084216；自 door_nav1_door_5 起）：Howard「移動時車頭朝前、操作前才對位」⇒ gmpc heading_enable＝true（參考航向＝路徑前視弦方向，
#   不要求終點航向；定版其他參數不變），PARK_ALIGN 內才轉至操作朝向 90°；PARK_ALIGN 位置容差 5 → 1 mm（--tol-pos 0.001；限速、期限、閘門不變）。
# 用法：RUN_ID=door_nav1_door_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_nav1_door.sh bash evaluation/run_guarded.sh
OFFSET_MOVING=1
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
: "${RUN_ID:?RUN_ID 未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
# shellcheck disable=SC1091
if [ -f "$WS/install/setup.bash" ]; then set +u; source "$WS/install/setup.bash"; set -u; fi
START_S="${START_S:--0.6,-2.8,1.570796}"
MAP="$WS/src/ammr_bringup/maps/mission_room.yaml"
WORLD="$WS/src/ammr_bringup/worlds/mission_room.sdf"
D_PARK_XY="2.263588,4.16"; D_PARK_YAW="1.297349"; D_PARK="$D_PARK_XY,$D_PARK_YAW"; D_STOW="0,0,0.15,0,-1.10,0"
URDF="$WS/evaluation/models/omni_bot_manip.urdf"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
PRM=evaluation/results/door/pullback_params_60_r1.json   # pb1 r3：r2＋暫停期手指上限 5 N（pb1_3 張爪中止後，Codex 選 (a)）
GB=evaluation/runs/phf_01_M/task.json
PIDS=(); NAMES=()
source "$WS/evaluation/k2_proc.sh"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
spawn(){ local n="$1"; shift; setsid "$@" >>"$DIR/$n.log" 2>&1 & PIDS+=("$!"); NAMES+=("$n"); sleep 0.2; local g; g=$(ps -o pgid= -p "${PIDS[-1]}" 2>/dev/null | tr -d " "); printf "%s\t%s\t%s\n" "$n" "${PIDS[-1]}" "${g:-NA}" >> "$DIR/procs.tsv"; say "  起 $n PID=${PIDS[-1]} PGID=${g:-NA}"; }
cleanup(){ say "cleanup（只針對本趟 PID；TERM → 3 s → KILL）"
  for i in "${!PIDS[@]}"; do _g=$(awk -v p="${PIDS[$i]}" '$2==p{print $3}' "$DIR/procs.tsv"); if [ "$_g" = "${PIDS[$i]}" ]; then kill -TERM -- "-${PIDS[$i]}" 2>/dev/null || true; else kill -TERM "${PIDS[$i]}" 2>/dev/null || true; fi; done
  sleep 3
  for i in "${!PIDS[@]}"; do _g=$(awk -v p="${PIDS[$i]}" '$2==p{print $3}' "$DIR/procs.tsv"); if [ "$_g" = "${PIDS[$i]}" ]; then kill -KILL -- "-${PIDS[$i]}" 2>/dev/null || true; else kill -KILL "${PIDS[$i]}" 2>/dev/null || true; fi; done
  sleep 1; : > "$DIR/cleanup_residual.tsv"
  for i in "${!PIDS[@]}"; do _r=$(pgrep -g "${PIDS[$i]}" 2>/dev/null | tr '\n' ' '); printf "%s\t%s\t%s\n" "${NAMES[$i]}" "${PIDS[$i]}" "${_r:-none}" >> "$DIR/cleanup_residual.tsv"; done
  say "cleanup 後各群組殘留：$(awk '$3!="none"' "$DIR/cleanup_residual.tsv" | wc -l) 組有殘留（cleanup_residual.tsv）"; }
trap cleanup EXIT
trap 'say "收到 TERM"; exit 143' TERM
trap 'say "收到 INT"; exit 130' INT

REF="${PB1_REF:-evaluation/results/door/door_regrasp_ref_pb60h.json}"   # 加速變體可用 PB1_REF 指定 *_wp.json
DOOR3=src/my_omnibot_description/config/door_local_v3.yaml
sha256sum evaluation/isaac_door_k2_sim_ev9r8u.py evaluation/door_drawer_phase2.py evaluation/isaac_door_k2_sim_ev9r8t.py evaluation/door_coopen_core_task_node10o.py evaluation/door_drawer_phase.py evaluation/cmd_relay_node.py evaluation/drawer_glide_node.py evaluation/drawer_wholebody_node.py evaluation/drawer_task_node.py evaluation/drawer_mission_node.py evaluation/arm_vel_adapter.py evaluation/wgmpc_wg2_node.py src/ammr_bringup/worlds/mission_room.sdf src/ammr_bringup/maps/mission_room.yaml src/ammr_bringup/maps/mission_room.pgm src/my_omnibot_description/config/drawer_unit.yaml evaluation/isaac_door_k2_sim_ev9r8s.py evaluation/door_coopen_core_task_node10n.py evaluation/door_regrasp_close_settle2.py evaluation/isaac_door_k2_sim_ev9r8r.py evaluation/door_coopen_core_task_node10m.py evaluation/door_coopen_core_task_node10l.py evaluation/door_regrasp_close_settle.py evaluation/isaac_door_k2_sim_ev9r8q.py evaluation/door_k3m_applied_src_tb.py evaluation/isaac_door_k2_sim_ev9r8p.py evaluation/dl3_wgmpc_wg2_adm23_node.py evaluation/door_coopen_core_task_node10i.py evaluation/door_nav1_mission_node_r5.py evaluation/door_nav1_mission_node_r6.py evaluation/door_nav1_mission_node_r7.py evaluation/door_nav1_mission_node_r8.py evaluation/door_nav1_mission_node_r9.py evaluation/arm_vel_adapter_k3m4.py evaluation/dual_source_executor_r2.py evaluation/wb_cmd_chain_e2t.py evaluation/door_k3m_applied_src_t.py evaluation/cmd_envelope_v4.py evaluation/door_solver_tenure.py evaluation/door_nav_envelope_monitor.py evaluation/door_regrasp_m4r_policy.py evaluation/door_regrasp_m4r_policy2.py evaluation/door_coopen_core_task_node10j.py evaluation/door_coopen_core_task_node10k.py evaluation/door_w2_tenure_auth.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/scan_safety_shield.py evaluation/isaac_door_k2_sim_ev9r8o.py evaluation/dl3_wgmpc_wg2_adm22_node.py evaluation/door_coopen_core_task_node10h.py evaluation/door_rolling_task.py evaluation/door_k3m_mode2.py evaluation/motm_coord.py evaluation/isaac_door_k2_sim_ev9r8n.py evaluation/door_coopen_core_task_node10g.py evaluation/door_nav1_mission_node.py src/ammr_bringup/scripts/generate_door_scene_map.py src/ammr_bringup/maps/door_scene_v3.yaml src/ammr_bringup/maps/door_scene_v3.pgm src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_node.py evaluation/door_k2_wholebody_node.py evaluation/isaac_door_k2_sim_ev9r8m.py evaluation/isaac_door_k2_sim_ev9r8j.py evaluation/door_coopen_core_task_node10f.py evaluation/door_coopen_core_task_node10e.py evaluation/door_coopen_bench_grasp3.py evaluation/door_coopen_bench_grasp2.py evaluation/door_coopen_core_task_node10d.py evaluation/door_settle_cont.py evaluation/door_coopen_core_task_node10c.py evaluation/dl3_wgmpc_wg2_adm21_node.py evaluation/arm_vel_adapter_k3m3.py \
  evaluation/door_regrasp_*.py evaluation/door_finger_loss_*.py evaluation/door_mimic_follow.py evaluation/door_k3m_observer.py "$PRM" "$GB" "$REF" "$DOOR3" > "$DIR/versions.sha256" || { say "**版本紀錄失敗** ⇒ 拒跑"; exit 95; }
say "版本紀錄 $DIR/versions.sha256（$(wc -l < "$DIR/versions.sha256") 檔；git $(git rev-parse --short HEAD)）"
say "python3＝$(command -v python3)；VIRTUAL_ENV=${VIRTUAL_ENV:-（無）}"
python3 -c "import osqp" 2>/dev/null || { say "**python3 環境錯誤（無法匯入 osqp；可能誤啟用虛擬環境）** ⇒ 拒跑"; exit 96; }
[ -s "$PRM" ] || { say "**核心段登錄檔不存在：$PRM** ⇒ 拒跑"; exit 97; }
[ -s "$GB" ] || { say "**抓取基準不存在：$GB** ⇒ 拒跑"; exit 97; }
[ -s "$MAP" ] || { say "**找不到門場景地圖 $MAP** ⇒ 拒跑"; exit 80; }
ORPH=$(python3 - "$ROS_DOMAIN_ID" <<'PYEOF'
import os, sys
dom = sys.argv[1]; me = os.getpid(); n = []
for p in os.listdir('/proc'):
    if not p.isdigit() or int(p) == me:
        continue
    try:
        env = open(f'/proc/{p}/environ', 'rb').read().split(b'\0')
        cmd = open(f'/proc/{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='ignore')
    except OSError:
        continue
    dm = [e for e in env if e.startswith(b'ROS_DOMAIN_ID=')]
    same = (not dm) or dm[0] == f'ROS_DOMAIN_ID={dom}'.encode()
    known = any(k in cmd for k in ('isaac_door_k2_sim', 'dl3_wgmpc', 'door_coopen_core_task', 'arm_vel_adapter', 'door_nav1_mission',
                                   'door_k2_wholebody', 'door_k3m_observer', 'gmpc_node'))
    if same and ('--ros-args' in cmd or known) and 'run_door_' not in cmd and 'run_guarded' not in cmd:
        n.append(f'{p}:{cmd[:80]}')
print(len(n)); [print(x) for x in n[:10]]
PYEOF
)
[ "$(echo "$ORPH" | head -1)" = 0 ] || { say "**domain $ROS_DOMAIN_ID 已有 $(echo "$ORPH" | head -1) 個 ROS 程序（可能是前趟孤兒）⇒ 拒跑**"; echo "$ORPH" | tail -n +2 | tee -a "$DIR/run.log"; exit 99; }
say "起跑前核對：domain $ROS_DOMAIN_ID 未發現衝突程序（帶 --ros-args 或已知節點名；只代表此掃描範圍）"
SHIELD_INST=$(python3 -c "import ammr_wholebody_mpc.scan_safety_shield as m;print(m.__file__)" 2>/dev/null)
grep -q "self_return_radius" "$SHIELD_INST" 2>/dev/null || { say "**已安裝的護盾不含 self_return_radius（$SHIELD_INST）⇒ 須先 colcon build（不用 --symlink-install）** ⇒ 拒跑"; exit 98; }
python3 -c "import json;p=json.load(open('$REF'));assert 'm4r' in p and '41' in p['segments'] and 'r3_pre' in p['m4r']" 2>/dev/null || { say "**參考包不是 pb1h（缺 m4r.r3_pre／段 41）：$REF** ⇒ 拒跑"; exit 97; }
[ "$(python3 -c "import json;print(json.load(open('$PRM'))['solver_change'].get('offset_update_in_open') is False)")" = True ] || { say "**登錄 r2 須明確 offset_update_in_open＝False（W1 語義）** ⇒ 拒跑"; exit 97; }
GB_SHA=$(python3 -c "import json;print(json.load(open('$PRM'))['grasp']['basis_sha256'])")
[ "$(sha256sum "$GB" | cut -d' ' -f1)" = "$GB_SHA" ] || { say "**抓取基準 sha256 與登錄不符** ⇒ 拒跑"; exit 97; }
W1B_THETA=$(python3 -c "import json;print(json.load(open('$PRM'))['theta_goal_deg'])")
STAND=$(python3 -c "import json;print(json.load(open('$PRM'))['start']['stand_cli'])")
K2_READY=$(python3 -c "import json;print(json.load(open('$PRM'))['start']['k2_ready_cli'])")
Q_PRE=$(python3 -c "import json;print(json.load(open('$PRM'))['start']['q_pre_cli'])")
# door_nav1_door_3：登錄 start.q_pre_cli＝null（台架流程不經展開）⇒ 展開節點 --q-unfold 收到 'None' 崩潰。改用 W1 線實跑過的 Q_PRE
# （run_door_wide_w1f2.sh；door_wide_w1f_60_3 UNFOLD→ALIGN→APPROACH→ENGAGE 實跑；pullback 登錄衍生自 w1f_params_60_r2）；登錄有值時以登錄為準。
Q_PRE_W1='-0.9471,0.0758,1.6024,-1.539,-0.9479,-0.0544'
[ "$Q_PRE" = None ] && Q_PRE="$Q_PRE_W1"
Q_PRE_SRC=registry; [ "$Q_PRE" = "$Q_PRE_W1" ] && Q_PRE_SRC='run_door_wide_w1f2.sh（door_wide_w1f_60_3 實跑）'
python3 -c "import math;v=[float(x) for x in '$Q_PRE'.split(',')];assert len(v)==6 and all(math.isfinite(x) for x in v)" 2>/dev/null || { say "**Q_PRE 不合法（須六個有限數）：$Q_PRE** ⇒ 拒跑"; exit 97; }
python3 -c "import json;json.dump({'q_pre':[float(x) for x in '$Q_PRE'.split(',')],'source':'$Q_PRE_SRC','registry_q_pre_cli':None},open('$DIR/q_pre.json','w'),ensure_ascii=False)"
say "Q_PRE（展開目標）＝$Q_PRE；來源 $Q_PRE_SRC（記入 q_pre.json）"
Q_INIT=$(python3 -c "import json;print(json.load(open('$PRM'))['bench']['q_nominal_cli'])")   # bench：名目初態（關節生成）
S_INIT=$(python3 -c "import json;print(json.load(open('$PRM'))['bench']['s_init_cli'])")   # bench：寫入致動器初始設定點（q_nominal − d_init）
W1B_TABLE=$(python3 -c "import json;print(json.load(open('$PRM'))['protection_scope_change']['corridor_table'])")
W1B_ENV=$(python3 -c "import json;print(json.load(open('$PRM'))['protection_scope_change']['envelope']['cli'])")
W1B_OBS=$(python3 -c "import json;print(json.load(open('$PRM'))['protection_scope_change']['observer_door_max_deg'])")
W1B_COORD=$(python3 -c "import json;print(json.load(open('$PRM'))['control_cost_change']['coord_topic'])")
W1E_FEFF=$(python3 -c "import json;print(json.load(open('$PRM'))['executor_change']['pause_finger_max_effort'])")
W1E_FEFF0=$(python3 -c "import json;print(json.load(open('$PRM'))['executor_change']['original_effort'])")
say "W1e 執行端變更：暫停期手指驅動輸出上限 ${W1E_FEFF0} → ${W1E_FEFF} N（ev5 r2；啟動讀回不符拒跑、切換／定期讀回異常走中止閂鎖）"
say "W1b 登錄：$PRM（sha256 $(sha256sum "$PRM" | cut -c1-16)）θ=${W1B_THETA}° 走廊表 $W1B_TABLE 包絡 $W1B_ENV 觀察器上限 ${W1B_OBS}° 協同 $W1B_COORD"
say "=== NAV1＋門段＋mode 4 R1–R3 同趟（ev9r8p＋node10i＋ADM23＋k3m4；ref $(basename "$REF")；導航 GMPC＋CBF→平滑器→選擇器〔巡航／PARK_ALIGN〕→掃描 barrier 護盾）RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID；起點 $START_S → 停位 $STAND；交棒後 K2_READY→Q_PRE→W1 正面實際接近閉爪→開門 ${W1B_THETA}°→換抓→穿門拉關→放手退出 ==="
say "揭露：/odom 真值無 AMCL；直線計畫；巡航 heading 開（路徑前視弦）；精停段＝P 伺服＋掃描 barrier（非 GMPC，含轉至 90°，容差 1 mm／1°）；無移動／未知障礙；A 房無外牆"

spawn sim "$ISAAC_PY" -u evaluation/isaac_door_k2_sim_ev9r8u.py --world "$WORLD" --with-drawer --subtask-handoff --drawer-park-pose="$D_PARK" --drawer-stow-q="$D_STOW" --base-box-snap --tenure-loop --nav-env-r 0.39 --rolling-handover --gate6-log --nav-start --pub-tf true --hold-confirm-cont --finger-leader-only --finger-loss-cont --ext-forces-every-iter --diag-6d --regrasp --regrasp-ref "$REF" --w2-mode-run-id "$RUN_ID" --contact-v3-out "$DIR/contact_v3.jsonl" --door-asset "$DOOR3" --gc-freeze-before-release --pause-finger-max-effort "$W1E_FEFF" --pause-finger-original-effort "$W1E_FEFF0" --cmd-env --evidence-out "$DIR/evidence.jsonl" --park-gate-mobile --run-id "$RUN_ID" --corridor-table "$W1B_TABLE" --envelope="$W1B_ENV" --sim-limit 1400 --contact-min-n 0.5 \
  --finger-collision split --park-hold --start-pose="$START_S" --park-pose="$STAND" --stow-q="$K2_READY" --out "$DIR"
say "  等場景建起（最多 180 s）"
for i in $(seq 180); do grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null && break; kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null || { say "**場景未建起** ⇒ 中止"; tail -8 "$DIR/sim.log" | tee -a "$DIR/run.log"; exit 83; }
if [ "${SMOKE_SCENE:-0}" = "1" ]; then                                         # 冒煙：只核場景建構，進主迴圈即停
  grep -E '^\[(drawer|door|room)\]' "$DIR/sim.log" | grep -E '抽屜|摩擦核對|接觸視圖|房間|門建於|DY0|拒跑|中止' | tee -a "$DIR/run.log"
  timeout 5 ros2 topic pub --once /room/stop std_msgs/msg/String "{data: 'smoke scene'}" >/dev/null 2>&1
  for i in $(seq 30); do kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
  say "=== SMOKE_SCENE 完成 ==="; exit 0
fi
grep -E '^\[(room|door)\] (零摩擦|夾持面|摩擦核對|門建於|場景求解|主從讀回|初始門角|門扇接觸|牆接觸)' "$DIR/sim.log" | tee -a "$DIR/run.log"

say "[導航] map_server／scan_relay／tracker／gmpc（定版）／平滑器→選擇器→護盾"
spawn rsp ros2 run robot_state_publisher robot_state_publisher "$URDF" --ros-args -p use_sim_time:=true
spawn tf_map ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --qx 0 --qy 0 --qz 0 --qw 1 --frame-id map --child-frame-id odom --ros-args -p use_sim_time:=true
spawn map ros2 run nav2_map_server map_server --ros-args -p use_sim_time:=true -p yaml_filename:="$MAP"
sleep 3
spawn maplife ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_map -p use_sim_time:=true -p autostart:=true -p node_names:="['map_server']"
spawn relay ros2 run ammr_bringup scan_relay --ros-args -p use_sim_time:=true
sleep 3
spawn tracker ros2 run ammr_wholebody_mpc scan_obstacle_tracker --ros-args -p use_sim_time:=true -p scan_topic:=/scan -p map_topic:=/map -p global_frame:=odom -p publish_frame:=odom -p use_map_subtraction:=true -p cluster_gap:=0.30 -p max_radius:=1.20 -p min_cluster_pts:=2 -p static_window:=2.0 -p min_net_speed:=0.05 -p surface_max_pts:=4 -p surface_range:=2.5 -p surface_min_sep:=0.20
sleep 3
spawn gmpc python3 -u -c "
import sys; sys.path.insert(0,'src/ammr_wholebody_mpc')
sys.argv=['gmpc_node','--ros-args','-p','use_sim_time:=true','-p','pose_source:=odom','-p','global_frame:=odom','-p','pose_odom_topic:=/odom','-p','plan_topic:=/plan',
 '-p','cmd_vel_topic:=/cmd_vel_nav','-p','obstacles_topic:=/gmpc/obstacles','-p','cbf_enable:=true','-p','cbf_alpha:=0.5','-p','cbf_safe_margin:=0.69',
 '-p','static_cbf_safe_margin:=0.47','-p','cbf_prune_range:=1.2','-p','cbf_near_steps:=6','-p','cbf_far_stride:=2','-p','horizon:=20','-p','control_frequency:=20.0',
 '-p','Q_xy:=15.0','-p','Qf_mult:=5.0','-p','R_vx:=2.0','-p','R_vy:=2.0','-p','R_w:=1.0','-p','S_vx:=15.0','-p','S_vy:=15.0','-p','S_w:=8.0',
 '-p','heading_enable:=true','-p','wheel_fit_mode:=project','-p','vx_min:=-0.35','-p','ax_max:=1.5','-p','ay_max:=1.0','-p','az_max:=2.0','-p','plan_blend_s:=0.0']
from ammr_wholebody_mpc.gmpc_node import main; main()"
sleep 4
spawn smoother ros2 run nav2_velocity_smoother velocity_smoother --ros-args -p use_sim_time:=true -p max_velocity:="[0.35, 0.25, 0.80]" -p min_velocity:="[-0.35, -0.25, -0.80]" -p max_accel:="[1.5, 1.0, 2.0]" -p max_decel:="[-1.5, -1.0, -2.0]" -p feedback:=OPEN_LOOP -p odom_topic:=/odom -r cmd_vel:=cmd_vel_nav -r cmd_vel_smoothed:=cmd_vel_pre_mux
spawn smlife ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_smoother -p use_sim_time:=true -p autostart:=true -p node_names:="['velocity_smoother']"
# **確認平滑器真的啟用。** 生命週期服務偶爾回應逾時（motm_180116：
# 「failed to send response ... (timeout)」），管理器卡在 Configuring，
# 平滑器一筆都不轉送 ⇒ 導航命令 0 筆、機器人整趟不動。先前沒檢查就往下跑。
SM_OK=0
for i in $(seq 20); do
  grep -q 'Managed nodes are active' "$DIR/smlife.log" 2>/dev/null && { SM_OK=1; break; }
  sleep 1
done
if [ "$SM_OK" != 1 ]; then
  say "**平滑器未回報 active** ⇒ 讀實際狀態並手動 configure／activate"
  ST=$(timeout 10 ros2 lifecycle get /velocity_smoother 2>/dev/null | awk '{print $1}')
  say "  目前狀態：${ST:-（讀不到）}"
  # 轉換中（configuring／activating）有界等待 15 s，不重複下指令（Codex 20261030_044316 第 1 點）
  for j in $(seq 15); do
    case "$ST" in configuring|activating) sleep 1; ST=$(timeout 10 ros2 lifecycle get /velocity_smoother 2>/dev/null | awk '{print $1}') ;; *) break ;; esac
  done
  say "  等待後狀態：${ST:-（讀不到）}"
  [ "$ST" = "unconfigured" ] && timeout 10 ros2 lifecycle set /velocity_smoother configure >>"$DIR/run.log" 2>&1
  ST=$(timeout 10 ros2 lifecycle get /velocity_smoother 2>/dev/null | awk '{print $1}')
  [ "$ST" = "inactive" ] && timeout 10 ros2 lifecycle set /velocity_smoother activate >>"$DIR/run.log" 2>&1
  ST=$(timeout 10 ros2 lifecycle get /velocity_smoother 2>/dev/null | awk '{print $1}')
  if [ "$ST" != "active" ]; then
    say "**平滑器仍未啟用（${ST:-讀不到}）⇒ 中止，不空跑**"; exit 86; fi
  say "  平滑器已手動啟用"
fi
sleep 3
spawn gmpc_diag ros2 topic echo --csv /gmpc/diag std_msgs/msg/Float32MultiArray
spawn shield ros2 run ammr_wholebody_mpc scan_safety_shield --ros-args -p use_sim_time:=true -p enable:=true -p scan_topic:=/scan -p cmd_in_topic:=/cmd_vel_pre_shield -p cmd_out_topic:=/cmd_vel -p robot_radius:=0.39 -p self_return_radius:=0.30 -p alpha:=2.0 -p a_brake:=6.25 -p tau:=0.15 -p vx_max:=0.2775 -p vy_max:=0.2775 -p wz_max:=1.1327
sleep 3
spawn nav1 python3 -u evaluation/door_nav1_mission_node_r9.py --goal="$STAND" --out "$DIR/nav1.json" --rolling --nav-timeout-s 150
spawn wholebody python3 -u evaluation/door_k2_wholebody_node.py --solver-handshake --motm --motm-v-cap 0.030 --motm-a-ref 0.002 --motm-k-yaw 1.0 --motm-w-cap 0.10 --motm-v-creep 0.0 --motm-overshoot 0.0 --pos-tol 0.005 --stow-q "$K2_READY" --park="$STAND" --q-unfold="$Q_PRE" --out "$DIR/wholebody.json"
spawn task python3 -u evaluation/door_coopen_core_task_node10o.py --subtask-mode --m4r --rolling --roll-goal="$STAND" --nav-start --wait-handover-timeout-s 240 --out "$DIR/task.json" --run-id "$RUN_ID" --grasp-basis "$GB" --w1b-params "$PRM" --corridor-table "$W1B_TABLE" --finger-loss-cont --settle-cont --regrasp --regrasp-ref "$REF" --w2-mode-run-id "$RUN_ID" --wall-total-s 1500
ARM_IDENT="$WS/evaluation/results/wgmpc_arm_sp_ident_free4.json"
[ -f "$ARM_IDENT" ] || { say "**找不到手臂辨識檔** ⇒ 中止"; exit 85; }
spawn adapter python3 -u evaluation/arm_vel_adapter_k3m4.py --in-topic /wgmpc/cmd_world --consumer-node /door_k2_sim --cmd-env --cmd-env-v4 --env-in-topic /wgmpc/cmd_env
spawn solver python3 -u evaluation/dl3_wgmpc_wg2_adm23_node.py --tenure-loop --rolling-approach --w2-retreat --w2-hold-r9 --w2-held --w2-held-max-s 60 --w2-mode-run-id "$RUN_ID" --regrasp --require-topic-target --park-hold --offset-observer interval --mode-switch-run-id "$RUN_ID" --cmd-env --run-id "$RUN_ID" --pause --pause-arm-zero --pause-active-hold --pause-arm-win-s 0.5 --coord-topic "$W1B_COORD" --offset-hold-on-free \
  --offset-moving --offset-free \
  --continuous --wait-for-drawer-handover --arm-model setpoint --arm-ident "$ARM_IDENT" \
  --cmd-topic /wgmpc/cmd_world --target-topic /drawer/tcp_target --target-offset 0 0 0 \
  --N 5 --rate 20 --delay-comp-state-cycles 1.6 --delay-comp-cmd-cycles 1.0 \
  --near-target-gamma 1.0 --w-s 0.001 --w-a 0.05 --w-s-arm 0.05 --margin-guard \
  --reach-pos-m 0.005 --reach-rot-rad 0.02 --hold-s 2.0 --duration-s 1600 --duration-sim-s 800 \
  --u-prev-policy strict --assume-initial-rest --out "$DIR/align_solver.json"
spawn lw_observer python3 -u evaluation/door_k3m_observer.py --out "$DIR/lw_observer.json" --door-max-deg "$W1B_OBS"

tsnap(){ local tag="$1"; { for _t in /cmd_vel /cmd_vel_pre_mux /cmd_vel_pre_shield /cmd_vel_nav /scan /tf /tf_static /plan; do echo "== $_t"; timeout 10 ros2 topic info -v "$_t" 2>&1 | grep -E "Publisher count|Subscription count|Node name|Endpoint type" ; done; } > "$DIR/topic_pubs_$tag.txt" 2>&1; say "  話題發布者快照 $tag：$(grep -c 'Node name' "$DIR/topic_pubs_$tag.txt") 筆（topic_pubs_$tag.txt）"; }
( sleep 8; tsnap start ) &
( until [ "$(grep -c '^\[nav1\] 分段 → CRUISE' "$DIR/nav1.log" 2>/dev/null)" -ge 2 ]; do kill -0 "$(awk '$1=="task"{print $2}' "$DIR/procs.tsv")" 2>/dev/null || exit 0; sleep 1; done; tsnap mission2 ) &
say "[任務] 等任務節點結束（牆鐘最多 1600 s）"
pid_of(){ local i; for i in "${!NAMES[@]}"; do [ "${NAMES[$i]}" = "$1" ] && echo "${PIDS[$i]}"; done; }
SIM_PID=$(pid_of sim); TASK_PID=$(pid_of task); SOLVER_PID=$(pid_of solver); OBS_PID=$(pid_of lw_observer); WB_PID=$(pid_of wholebody); NAV1_PID=$(pid_of nav1)
say "  PID：sim $SIM_PID task $TASK_PID solver $SOLVER_PID observer $OBS_PID wholebody $WB_PID nav1 $NAV1_PID"
# （r2 的「交棒後停掉導航計算」已移除：R2 需要導航）
( while kill -0 "$TASK_PID" 2>/dev/null; do kill -0 "$WB_PID" 2>/dev/null || { say "**展開節點已退出**（任務仍在跑；任務端 r3 依命令缺席判定）"; break; }; sleep 1; done ) &
sleep 5; kill -0 "$OBS_PID" 2>/dev/null || { say "**獨立觀察器未存活** ⇒ 中止（沒有觀察期保護不得繼續）"; exit 86; }
wait_or_stop "$TASK_PID" 1600 10; RC_TASK=$STOP_RC; HOW_TASK=$STOP_HOW
grep -E '^\[k3a\]' "$DIR/task.log" | tee -a "$DIR/run.log"
# ================= 門→抽屜交接（S1） =================
DRAWER_PHASE=0; RC_DMISSION=na
if grep -q '^\[k3a\] 結束：DONE' "$DIR/task.log" && grep -q '門段交回導航' "$DIR/task.log"; then
  say "[交接] 門段 DONE 且已交回導航；等執行端門段交接（最多 60 s）"
  for i in $(seq 60); do grep -q '抽屜段開始' "$DIR/sim.log" && break; sleep 1; done
  if grep -q '抽屜段開始' "$DIR/sim.log"; then
    grep -E '門段交接|抽屜段開始' "$DIR/sim.log" | tee -a "$DIR/run.log"
    say "[交接] 門段節點受控結束（solver／wholebody／nav1／lw_observer／gmpc／gmpc_diag）"
    wait_or_stop "$SOLVER_PID" 30 10; RC_SOLVER=$STOP_RC; HOW_SOLVER=$STOP_HOW
    wait_or_stop "$WB_PID" 20 5; RC_WB=$STOP_RC; HOW_WB=$STOP_HOW
    for _n in nav1 lw_observer gmpc gmpc_diag adapter; do _p=$(pid_of $_n); [ -n "$_p" ] && { kill -TERM -- "-$(ps -o pgid= -p "$_p" | tr -d ' ')" 2>/dev/null || kill -TERM "$_p" 2>/dev/null; }; done
    sleep 3
    _ok_iso=0
    for i in $(seq 20); do
      _bad=""
      for _t in /drawer/tcp_target /drawer/solver_start /gripper/cmd /handover/request /plan /cmd_vel_pre_shield /cmd_vel_nav /wgmpc/cmd_world /wgmpc/cmd_env; do
        _o=$(timeout 8 ros2 topic info "$_t" 2>&1); _rc=$?                    # 只有「確讀到 0」或「確認話題不存在」才放行；逾時／查詢失敗算不通過
        if echo "$_o" | grep -q 'Publisher count: 0'; then :; elif echo "$_o" | grep -qi 'Unknown topic'; then :;
        else _bad="$_bad $_t:$(echo "$_o" | awk '/Publisher count/{print $3}')${_rc:+(rc$_rc)}"; fi
      done
      [ -z "$_bad" ] && { _ok_iso=1; break; }; sleep 1
    done
    if [ "$_ok_iso" != 1 ]; then
      say "**門段發布者未全數消失**（$_bad）⇒ 不起抽屜段，整體受控停止"
      timeout 5 ros2 topic pub --once /room/stop std_msgs/msg/String "{data: 'drawer phase refused: door publishers alive'}" >/dev/null 2>&1
    else
      say "[交接] 門段發布者全數消失（隔離核對通過）"
      DPOSE=$(timeout 15 python3 -c "
import rclpy,math
from nav_msgs.msg import Odometry
rclpy.init(); n=rclpy.create_node('pose_once'); r=[]
n.create_subscription(Odometry,'/odom',lambda m:r.append(m),1)
import time; t0=time.time()
while not r and time.time()-t0<10: rclpy.spin_once(n,timeout_sec=0.1)
m=r[-1].pose.pose; q=m.orientation; y=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
print(f'{m.position.x:.6f},{m.position.y:.6f},{y:.6f}')")
      if [ -z "$DPOSE" ]; then
        say "**讀不到 /odom 實際位姿** ⇒ 整體受控停止"
        timeout 5 ros2 topic pub --once /room/stop std_msgs/msg/String "{data: 'drawer phase refused: no odom'}" >/dev/null 2>&1
      else
        DSTART_XY=$(echo "$DPOSE" | cut -d, -f1,2)
        say "[抽屜段] 起點＝實際位姿 $DPOSE；停位 $D_PARK；收臂 $D_STOW"
        DRAWER_PHASE=1
        spawn gmpc_d python3 -u -c "
import sys; sys.path.insert(0,'src/ammr_wholebody_mpc')
sys.argv=['gmpc_node','--ros-args','-p','use_sim_time:=true',
          '-p','pose_source:=odom','-p','global_frame:=odom',
          '-p','control_frequency:=20.0',
          '-p','cbf_enable:=true','-p','plan_topic:=/plan','-p','obstacles_topic:=/gmpc/obstacles',
          '-p','cbf_alpha:=0.5','-p','cbf_safe_margin:=0.69','-p','static_cbf_safe_margin:=0.47',
          '-p','cbf_prune_range:=1.2','-p','cbf_near_steps:=6','-p','cbf_far_stride:=2',
          '-p','cmd_vel_topic:=/cmd_vel_nav','-p','pose_odom_topic:=/odom',
          '-p','Q_xy:=15.0','-p','Qf_mult:=5.0',
          '-p','R_vx:=2.0','-p','R_vy:=2.0','-p','R_w:=1.0',
          '-p','S_vx:=15.0','-p','S_vy:=15.0','-p','S_w:=8.0',
          '-p','vx_min:=-0.35',
          '-p','ax_max:=1.5','-p','ay_max:=1.0','-p','az_max:=2.0',
          '-p','reference_projection:=segment',
          '-p','reference_monotonic:=true',
          '-p','wheel_fit_mode:=project',
          '-p','heading_enable:=true',
          '-p','heading_weight:=2.0',
          '-p','heading_lookahead_m:=1.2',
          '-p','heading_rate_max:=1.0']
from ammr_wholebody_mpc.gmpc_node import main; main()"
        spawn gmpc_diag_d ros2 topic echo --csv /gmpc/diag std_msgs/msg/Float32MultiArray
        sleep 6
        spawn relay_d python3 -u evaluation/cmd_relay_node.py
        sleep 1
        say "[抽屜段] 離門短段（+y 0.4 m；GMPC＋CBF；期限 40 s 模擬時間）"
        python3 -u evaluation/nav_escape_plan_node.py --dy 0.4 --min-progress-m 0.25 --timeout-s 40 --out "$DIR/escape.json" >>"$DIR/escape.log" 2>&1; RC_ESC=$?
        grep -E '^\[escape\]' "$DIR/escape.log" | tee -a "$DIR/run.log"
        ESC_OK=0; [ "$RC_ESC" = 0 ] && ESC_OK=1
        if [ "$ESC_OK" = 1 ]; then
          sleep 3
          DPOSE=$(timeout 15 python3 -c "
import rclpy,math
from nav_msgs.msg import Odometry
rclpy.init(); n=rclpy.create_node('pose_once2'); r=[]
n.create_subscription(Odometry,'/odom',lambda m:r.append(m),1)
import time; t0=time.time()
while not r and time.time()-t0<10: rclpy.spin_once(n,timeout_sec=0.1)
m=r[-1].pose.pose; q=m.orientation; y=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
print(f'{m.position.x:.6f},{m.position.y:.6f},{y:.6f}')")
          DSTART_XY=$(echo "$DPOSE" | cut -d, -f1,2)
          say "[抽屜段] 離門後實測位姿 $DPOSE ⇒ 抽屜段起點"
        else
          say "**離門短段失敗**（rc $RC_ESC；見 escape.json、gmpc_diag_d.log）⇒ 不起抽屜段，整體受控停止"
          timeout 5 ros2 topic pub --once /room/stop std_msgs/msg/String "{data: 'escape failed'}" >/dev/null 2>&1
        fi
      fi
      if [ "$DRAWER_PHASE" = 1 ] && [ "${ESC_OK:-0}" = 1 ]; then
        spawn glide_d python3 -u evaluation/drawer_glide_node.py --start="$DSTART_XY" --park="$D_PARK_XY" --park-yaw "$D_PARK_YAW" --k-yaw 1.0 --wz-max 0.35 --out "$DIR/glide_drawer.json"
        sleep 1
        spawn wholebody_d python3 -u evaluation/drawer_wholebody_node.py --solver-handshake --pos-tol 0.005 --motm --park="$D_PARK" --stow-q="$D_STOW" --out "$DIR/wholebody_drawer.json"
        sleep 3
        spawn task_d python3 -u evaluation/drawer_task_node.py --start="$DPOSE" --park="$D_PARK" --open-m 0.020 --grasp-depth-m 0.01226 --contact-min-n 0.5 --motm --out "$DIR/task_drawer.json"
        sleep 1
        spawn adapter_d python3 -u evaluation/arm_vel_adapter.py --in-topic /wgmpc/cmd_world --consumer-node /door_k2_sim
        spawn solver_d python3 -u evaluation/wgmpc_wg2_node.py --continuous --wait-for-drawer-handover --arm-model setpoint --arm-ident "$ARM_IDENT" \
          --cmd-topic /wgmpc/cmd_world --target-topic /drawer/tcp_target --target-offset 0 0 0 --N 5 --rate 20 \
          --delay-comp-state-cycles 1.6 --delay-comp-cmd-cycles 1.0 --near-target-gamma 1.0 --w-s 0.001 --w-a 0.05 --w-s-arm 0.05 \
          --margin-guard --coord-topic /wgmpc/coord --offset-free --reach-pos-m 0.005 --reach-rot-rad 0.02 --hold-s 2.0 \
          --duration-s 600 --duration-sim-s 400 --u-prev-policy strict --assume-initial-rest --out "$DIR/align_solver_drawer.json"
        sleep 2
        say "[抽屜段] 編排節點（最多 900 s 牆鐘）"
        spawn mission_d python3 -u evaluation/drawer_mission_node.py --out "$DIR/mission_drawer.json" --start="$DSTART_XY" --park="$D_PARK_XY" --park-yaw "$D_PARK_YAW" --stow-q="$D_STOW" --slow-zone 0.0
        wait_or_stop "$(pid_of mission_d)" 900 10; RC_DMISSION=$STOP_RC
        grep -E '^\[(mission|task|drawer)\]' "$DIR/mission_d.log" 2>/dev/null | tail -12 | tee -a "$DIR/run.log"
        wait_or_stop "$(pid_of task_d)" 60 10
        wait_or_stop "$(pid_of solver_d)" 30 10
        timeout 5 ros2 topic pub --once /room/stop std_msgs/msg/String "{data: 'mission door+drawer end'}" >/dev/null 2>&1
      fi
    fi
  else
    say "**執行端未進抽屜段**（60 s 內無「抽屜段開始」）⇒ 照門段收尾"
  fi
fi
if [ "$DRAWER_PHASE" = 1 ]; then
  say "[收尾] 等執行端封存 room_run.json／room_run_drawer.json（最多 90 s）"
  wait_or_stop "$SIM_PID" 90 15; RC_SIM=$STOP_RC; HOW_SIM=$STOP_HOW
  OK_DDONE=0; grep -q '"final_phase": "DONE"' "$DIR/task_drawer.json" 2>/dev/null && OK_DDONE=1
  RC=1; [ "$OK_DDONE" = 1 ] && [ "$RC_DMISSION" = 0 ] && [ "$HOW_SIM" = exited ] && [ "$RC_SIM" = 0 ] && RC=0
  say "=== 結束 rc=$RC（抽屜段）：離門 ${ESC_OK:-na}、抽屜任務 DONE $OK_DDONE、mission rc=$RC_DMISSION、sim $HOW_SIM rc=$RC_SIM；room_run_drawer $( [ -s "$DIR/room_run_drawer.json" ] && echo 1 || echo 0)；返程門與回起點不在本趟範圍 ==="
  exit "$RC"
fi
say "[收尾] 等求解節點停止並封存（最多 30 s，之後 TERM→10 s→KILL）"
wait_or_stop "$SOLVER_PID" 30 10; RC_SOLVER=$STOP_RC; HOW_SOLVER=$STOP_HOW
wait_or_stop "$WB_PID" 20 5; RC_WB=$STOP_RC; HOW_WB=$STOP_HOW
kill -TERM "$NAV1_PID" 2>/dev/null || true; sleep 1
say "[收尾] 等執行端封存 room_run.json（最多 60 s；任務節點已送 /room/stop）"
wait_or_stop "$SIM_PID" 60 15; RC_SIM=$STOP_RC; HOW_SIM=$STOP_HOW
kill -TERM "$OBS_PID" 2>/dev/null || true
wait_or_stop "$OBS_PID" 10 5; RC_OBS=$STOP_RC; HOW_OBS=$STOP_HOW
say "LW_HEALTH sim=$HOW_SIM sim_rc=$RC_SIM obs=$HOW_OBS obs_rc=$RC_OBS task=$HOW_TASK task_rc=$RC_TASK"
grep -qE '"lw_observer_(unhealthy|not_ready)"' "$DIR/task.json" 2>/dev/null && say "**觀察器失效（任務節點記錄）** ⇒ 本趟記觀察提前終止"
say "  結束方式：task $HOW_TASK、solver $HOW_SOLVER、wholebody $HOW_WB、sim $HOW_SIM（非 exited 即非正常結束）"
OK_ROOM=0; [ -s "$DIR/room_run.json" ] && OK_ROOM=1
OK_SOLVER=0; [ -s "$DIR/align_solver.json" ] && OK_SOLVER=1
OK_DONE=0; grep -q '^\[k3a\] 結束：DONE' "$DIR/task.log" && OK_DONE=1
say "  rc：task $RC_TASK、solver $RC_SOLVER、wholebody $RC_WB、sim $RC_SIM；room_run $OK_ROOM、align_solver $OK_SOLVER、DONE $OK_DONE"
RC=1
[ "$HOW_TASK" = exited ] && [ "$HOW_SOLVER" = exited ] && [ "$HOW_SIM" = exited ] && [ "$RC_TASK" = 0 ] && [ "$RC_SOLVER" = 0 ] && [ "$RC_SIM" = 0 ] && [ "$OK_ROOM" = 1 ] && [ "$OK_SOLVER" = 1 ] && [ "$OK_DONE" = 1 ] && RC=0
say "=== 結束 rc=$RC（0 只表示所有節點正常結束且到 DONE；指定初態功能展示，不是正式評估；拉關到約 2° 只能稱近關閉）==="
exit "$RC"
