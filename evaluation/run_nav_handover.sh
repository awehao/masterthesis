#!/usr/bin/env bash
# 導航 → 減速 → 滾動交棒 的實跑。**只跑前半段**，全身接手後即結束。
#
# 範圍（明說）：控制器是**真的** gmpc_node，原樣使用；/odom 給真值（本實驗的
# 明示前提）；/plan 由任務節點依已知靜態房間產生，**未用 nav2 planner**；
# **CBF 關閉** —— 沒有障礙物聚合器在發 /obstacles，房間靜態且路徑已用幾何
# 核過淨空（evaluation/test_drawer_room.py）。這三點都要寫進報告。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
RUN_ID="${RUN_ID:-nav_handover_$(date +%H%M%S)}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
SIM_LIMIT="${SIM_LIMIT:-240}"
# **控制週期要與實際相符。** 節點預設 20 Hz（dt=0.05），但 gmpc 自己的診斷
# 量到 dt_meas p50 0.1800、p90 0.1800 s —— 求解器以 0.05 s 一步規劃，命令卻
# 作用 0.18 s，第一步實際走 3.6 倍遠。本實驗把它設成實測可守住的週期，
# 這**不改任何速度／加速度／輪級限制**，只是讓 a_max 與預測的時間基準為真。
# 凍結的 Phase-1 基準設定不受影響（那是另一個場景的另一組啟動參數）。
# **預設回到節點原本的 20 Hz。** 先前我把預設設成 5.0 想讓 dt 與實測
# 週期相符，實跑證明那會讓底盤停死在 d≈0.73（nav_handover_ctrl5b），
# 而且那是**改掉既有預設值**。要測 dt 假設時明寫 CTRL_HZ=，不改預設。
CTRL_HZ="${CTRL_HZ:-20.0}"
# 航向目標：HEADING=1 開。預設關閉 ＝ 定版登錄的「航向參考 無」。
HEADING_EN=$([ "${HEADING:-0}" = "1" ] && echo true || echo false)
PIDS=(); NAMES=()
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
spawn(){ local n="$1"; shift; "$@" >>"$DIR/$n.log" 2>&1 & PIDS+=("$!"); NAMES+=("$n");
         say "  起 $n PID=${PIDS[-1]}"; }
cleanup(){ say "cleanup（只針對本趟 PID）"
  for i in "${!PIDS[@]}"; do kill -TERM "${PIDS[$i]}" 2>/dev/null || true; done
  sleep 3
  for i in "${!PIDS[@]}"; do kill -KILL "${PIDS[$i]}" 2>/dev/null || true; done; }
trap cleanup EXIT

say "=== 導航交棒 RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID ==="
say "範圍：真 gmpc_node；/odom 真值；/plan 本地產生（非 nav2）；**CBF 關閉**"
say "control_frequency=$CTRL_HZ Hz（dt=$(python3 -c "print(f'{1/$CTRL_HZ:.4f}')") s）；限制值全部沿用"

# **PARK_FIXED=1**：固定底盤操作（停住再展開；展開到交還導航底盤恆為零）。預設不設 ＝ 既有行為。
# 與 MOTM=1 同時指定 ⇒ 拒跑（不靜默選其中一個）。各節點各自帶旗標：sim --park-fixed、glide --park-stop、
# mission／wholebody／task --park-fixed、solver --base-fixed。
if [ "${PARK_FIXED:-0}" = "1" ] && [ "${MOTM:-0}" = "1" ]; then
  say "**PARK_FIXED=1 與 MOTM=1 不可同時指定** ⇒ 拒跑"; exit 87; fi
PARK_ON=$([ "${PARK_FIXED:-0}" = "1" ] && echo 1 || echo 0)
PK_SIM=$([ "$PARK_ON" = "1" ] && echo --park-fixed || true)
PK_GLIDE=$([ "$PARK_ON" = "1" ] && echo --park-stop || true)
PK_NODE=$([ "$PARK_ON" = "1" ] && echo --park-fixed || true)
PK_SOLVER=$([ "$PARK_ON" = "1" ] && echo --base-fixed || true)
if [ "$PARK_ON" = "1" ]; then say "  **PARK_FIXED**：固定底盤操作"; fi
spawn sim "$ISAAC_PY" -u evaluation/isaac_drawer_room_sim.py \
  --sim-limit "$SIM_LIMIT" --cam "${CAM:-false}" --cam-hz "${CAM_HZ:-10}" \
  --contact-min-n "${CONTACT_MIN_N:-0.5}" \
  --finger-collision "${FINGER_COL:-hull}" $PK_SIM \
  ${DRAWER_ASSET:+--drawer-asset "$DRAWER_ASSET"} --out "$DIR"
say "  等場景建起（最多 180 s）"
for i in $(seq 180); do grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null && break; sleep 1; done
if ! grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null; then
  say "**場景未建起** ⇒ 中止"; exit 83; fi
say "  場景已建起"
grep -E '^\[room\] (零摩擦|夾持面|摩擦核對)' "$DIR/sim.log" | tee -a "$DIR/run.log"

spawn gmpc python3 -u -c "
import sys; sys.path.insert(0,'src/ammr_wholebody_mpc')
sys.argv=['gmpc_node','--ros-args','-p','use_sim_time:=true',
          '-p','pose_source:=odom','-p','global_frame:=odom',
          '-p','control_frequency:=$CTRL_HZ',
          '-p','cbf_enable:=false','-p','plan_topic:=/plan',
          '-p','cmd_vel_topic:=/cmd_vel_nav','-p','pose_odom_topic:=/odom',
          # **定版的平滑設定**（CHANGELOG 的定案表）。先前這支 runner
          # 一項都沒帶，等於用節點出廠預設 S=0,0,0 跑 —— QP 完全不懲罰
          # 相鄰命令變化，配上實測 160 ms 的週期就是明顯頓挫。
          '-p','Q_xy:=15.0','-p','Qf_mult:=5.0',
          '-p','R_vx:=2.0','-p','R_vy:=2.0','-p','R_w:=1.0',
          '-p','S_vx:=15.0','-p','S_vy:=15.0','-p','S_w:=8.0',
          '-p','vx_min:=-0.35',
          '-p','ax_max:=1.5','-p','ay_max:=1.0','-p','az_max:=2.0',
          # **參考點改連續投影 ＋ 單調進度**：原本取最近頂點，s 在 5 cm
          # 取樣間跳格，慢速區相鄰命令方向變化 p50 32.7°、最大 103°。
          '-p','reference_projection:=segment',
          '-p','reference_monotonic:=true',
          # **航向目標**（HEADING=1 開）。參數沿用 gmpc_scan_heading 那個
          # 方法的值：權重 2.0、前視 1.2 m、參考斜率上限 1.0 rad/s。
          # 定版基準登錄的是「航向參考 無」，所以這是**對照用的選項**，
          # 不是定版設定；預設關閉。
          #
          # 它走的是 path_processor 文件裡的**目標式**航向：每個樣本都取
          # 同一個 desired_yaw，所以 xi_ref 的角分量仍然恆為零 —— 改變的是
          # e0[2] 這個成本看得到的誤差，以及平移參考所在的本體座標，
          # **不是**把轉速前饋進去。
          # **輪級修正改投影**：線段縮放用單一 λ，一列貼邊就凍結三個自由度，
          # 而 λ=0 的語意是「沿用上一筆」不是「停止」。實測
          # demo_heading_054503 有 229/300 輪（76.3%）輸出與上一筆逐位元
          # 相同，同一筆轉速連續送了 1.1 s，車頭轉過頭還在轉。
          # 投影到**同一組**輪級集合，限制一條未放寬（驗過才用）。
          '-p','wheel_fit_mode:=${WHEEL_FIT:-project}',
          '-p','heading_enable:=$HEADING_EN',
          '-p','heading_weight:=2.0',
          '-p','heading_lookahead_m:=1.2',
          '-p','heading_rate_max:=1.0']
from ammr_wholebody_mpc.gmpc_node import main; main()"
sleep 6

# **velocity_smoother**：gmpc → /cmd_vel_nav → 平滑器 → /cmd_vel。
# 定版的導航鏈本來就有它，先前這支 runner 漏掉，所以命令直接送到執行端。
# 它是 lifecycle 節點，沒有管理器會停在 unconfigured、一筆都不轉送。
spawn smoother ros2 run nav2_velocity_smoother velocity_smoother --ros-args \
  -p use_sim_time:=true \
  -p max_velocity:="[0.35, 0.25, 0.80]" -p min_velocity:="[-0.35, -0.25, -0.80]" \
  -p max_accel:="[1.5, 1.0, 2.0]" -p max_decel:="[-1.5, -1.0, -2.0]" \
  -p feedback:=OPEN_LOOP -p odom_topic:=/odom \
  -r cmd_vel:=cmd_vel_nav -r cmd_vel_smoothed:=cmd_vel
spawn smlife ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args \
  -r __node:=lifecycle_manager_smoother \
  -p use_sim_time:=true -p autostart:=true \
  -p node_names:="['velocity_smoother']"
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
  [ "$ST" = "unconfigured" ] && timeout 10 ros2 lifecycle set /velocity_smoother configure >>"$DIR/run.log" 2>&1
  ST=$(timeout 10 ros2 lifecycle get /velocity_smoother 2>/dev/null | awk '{print $1}')
  [ "$ST" = "inactive" ] && timeout 10 ros2 lifecycle set /velocity_smoother activate >>"$DIR/run.log" 2>&1
  ST=$(timeout 10 ros2 lifecycle get /velocity_smoother 2>/dev/null | awk '{print $1}')
  if [ "$ST" != "active" ]; then
    say "**平滑器仍未啟用（${ST:-讀不到}）⇒ 中止，不空跑**"; exit 86; fi
  say "  平滑器已手動啟用"
fi

# **求解器每輪的輸入**要逐筆留著。實跑量到慢速區命令在 ±a_max·dt 之間
# 反覆擺動，而橫向誤差 p50 僅 8.6 mm、偏航幾乎不動；離線純運動學閉環
# 重現不出來，所以要看求解器自己看到的 e0、xi_ref 與實際週期。
spawn diag python3 -u evaluation/record_gmpc_diag.py \
  --out "$DIR/gmpc_diag.jsonl"
sleep 1

# **減速交接段**：第三個控制權擁有者。導航的成本函數是「到達目標並煞停」，
# 實測交棒區內同時「框內」且「還在動」的步數只有 2 步（0.02 s）。減速段沿
# 已知直線做距離驅動的速率斜坡，降到 30 mm/s 後維持 ⇒ 窗口 6.7 s（666 步）。
# 它與導航共用直寫路徑與同一個變化率上限基準，所以 nav→glide 連續。
# 偏航權限：航向開啟時，底盤在交棒區入口是朝**行進方向**，離停車偏航可達
# 25°（實測 demo_heading_054127：−24.86°）。減速段原本的 k_yaw 0.30／
# wz_max 0.10 在 1.78 s 只轉得了 9°，展開接手時還差 15.78°，最後過衝卡住。
# 把增益與上限提到能在減速段內收完；**交棒當步的角速度框 0.1999 不變**，
# 它仍然是真正的閘門（P 控制在誤差變小時 wz 自然降到框內）。
# **兩個變數不要一起動。** 先前為了吸收航向開啟後的偏航誤差，把 k_yaw/
# wz_max 由 0.3/0.1 提到 1.0/0.35，結果 ON/OFF 兩趟差的不只是航向。
# 現在預設回到節點預設值，要調再用環境變數明寫。
spawn glide python3 -u evaluation/drawer_glide_node.py $PK_GLIDE \
  --k-yaw "${GLIDE_KYAW:-0.30}" --wz-max "${GLIDE_WZMAX:-0.10}" \
  --out "$DIR/glide.json"
sleep 1

# **全身端要在切換前就開始送命令** —— 預核才有東西可核。
# 切換前它沒有控制權，命令會被依控制權拒絕，但命令鏈仍然收下。
# **展開的底盤容差要比 ALIGN 的判準緊。** 預設 0.02 m 比 ALIGN 的 5 mm 寬，
# 於是 ALIGN 一開始就得退掉約 18 mm（實測 nav_align_solve4_023417：底盤停在
# y=0.54734、停車點 0.560，差 12.7 mm；任務誤差 12.17 mm）。
# 這不是放寬任何限制，是讓預定位把自己的工作做完。
# MOTM=1：移動中操作（底盤不停；只在開與關之間停頓）。預設關閉 = 既有行為。
MOTM_FLAG=$([ "${MOTM:-0}" = "1" ] && echo --motm || true)
spawn wholebody python3 -u evaluation/drawer_wholebody_node.py \
  --solver-handshake --pos-tol "${WB_POS_TOL:-0.005}" $MOTM_FLAG $PK_NODE \
  ${MOTM_A_REF:+--motm-a-ref "$MOTM_A_REF"} \
  --out "$DIR/wholebody.json"
sleep 3

# **模擬器還活著嗎** —— 它在主迴圈裡崩潰過一次（stop_req 屬性未初始化），
# 當時運行器照樣往下跑，全身端才在 30 s 後報「收不到狀態」。
if ! kill -0 "${PIDS[0]}" 2>/dev/null; then
  say "**模擬器已結束**，不往下跑"; tail -6 "$DIR/sim.log" | tee -a "$DIR/run.log"
  exit 84
fi

# **ALIGN 的目標來源**：由實測把手位姿算接觸前的退讓目標。
# 這一趟它**只發目標並獨立量測**：目標話題還沒有消費者（求解節點尚未接進
# 本管線），所以本趟不得宣稱 ALIGN 已經在控制。它要回答的是「展開結束時
# 離接觸前目標有多遠」。
# **任務編排節點**（取代只發目標的 drawer_align_node）：ALIGN → 夾持 →
# 開 → 停 → 關 → 停 → 放 → 退 → 收臂 → 交還導航 → 回起點。相位由
# drawer_task_policy 決定、推進一律看實測。提早起，求解節點一上線就有
# 接觸前目標可跟，不必走「保持起始 TCP」的退路。
# OPEN_M：先跑 0.020 的校正趟（不是結果），再跑正式 0.200。
spawn task python3 -u evaluation/drawer_task_node.py \
  --open-m "${OPEN_M:-0.020}" --grasp-depth-m "${GRASP_DEPTH_M:-0.01226}" \
  --contact-min-n "${CONTACT_MIN_N:-0.5}" $MOTM_FLAG $PK_NODE \
  ${MOTM_A_REF:+--motm-a-ref "$MOTM_A_REF"} ${MOTM_TASK_ARGS:-} \
  ${INJECT_ABORT_AT_S:+--inject-abort-at-s "$INJECT_ABORT_AT_S"} \
  --out "$DIR/task.json"
sleep 1

# 提早載入求解器；待命期間不發命令，展開與任務端皆放行才啟動。
say "[待命] 提早起求解節點（目標由 /drawer/tcp_target 來；**無安全層**）"
ARM_IDENT="$WS/evaluation/results/wgmpc_arm_sp_ident_free4.json"
[ -f "$ARM_IDENT" ] || { say "**找不到手臂辨識檔** ⇒ 中止"; exit 85; }
  # 世界座標 → 本體座標，只在這裡做一次
  spawn adapter python3 -u evaluation/arm_vel_adapter.py \
    --in-topic /wgmpc/cmd_world --consumer-node /drawer_room_sim
  # **連續模式、放到背景**：整個抽屜任務都由它跟隨目標，到達後不結束，
  # 由任務節點在退開完成後發 /wgmpc/stop。
  spawn solver python3 -u evaluation/wgmpc_wg2_node.py \
    --continuous --wait-for-drawer-handover \
    --arm-model setpoint --arm-ident "$ARM_IDENT" \
    `# **座標系**：求解節點發的是**世界座標**（_publish_u 的 docstring：` \
    `# 「body → world 後發給安全鏈，adapter 之後會轉回 body」）。` \
    `# 先前我把它直接接到 /wb_vel_cmd —— 那裡被命令鏈當**本體座標**，` \
    `# 底盤的 x/y 於是被旋轉了 θ≈74°。實跑 nav_align_solve5_023901 的` \
    `# 底盤走了 Δx=-59.3 mm／Δθ=-9.71°，而精確解要的是 Δy=-25 mm。` \
    `# 現在改發中性話題，由 adapter 做那一次旋轉（轉換只有一份）。` \
    --cmd-topic /wgmpc/cmd_world \
    --target-topic /drawer/tcp_target \
    `# 啟動參數只是**退路**：偏移 0 = 保持實測起始 TCP。` \
    `# 不用 '--target 0 0 0' —— 那會讓目標話題失效時瞄向原點。` \
    --target-offset 0 0 0 \
    `# 預測步數：WGMPC_N（預設 5 = 既有行為）。分析一律以求解節點落盤的` \
    `# align_solver.json args.N 為準，不信這裡的傳參。` \
    --N "${WGMPC_N:-5}" --rate 20 \
    `# **沿用 Stage A 已驗證的那一組**（到達保持 9/10 的設定），` \
    `# 不自己另配一套：延遲補償 1.6/1.0、近目標整形 γ=1.0、` \
    `# 命令變化率權重分底盤（W_A）與手臂（W_S_ARM）、餘裕守衛開。` \
    `# 先前漏帶這四項，實跑 nav_align_solve2_022109 的結果是閉環發散：` \
    `# 誤差 0→57→35→17→60→64 mm 來回，手臂速率 p50 0.99990 rad/s 飽和。` \
    --delay-comp-state-cycles 1.6 --delay-comp-cmd-cycles 1.0 \
    --near-target-gamma 1.0 --w-s 0.001 --w-a 0.05 --w-s-arm 0.05 \
    --margin-guard \
    `# MotM：任務節點依相位送底盤參考速度／手臂名目姿態／權重；過期即關閉` \
    $([ "${MOTM:-0}" = "1" ] && echo "--coord-topic /wgmpc/coord") \
    ${SOLVER_EXTRA:-} \
    `# WG4-B：SOLVER_KIND=b1 換成單步 QP 對照（預設不帶 = wgmpc，既有行為）` \
    ${SOLVER_KIND:+--solver-kind "$SOLVER_KIND"} ${B1_KP:+--b1-kp "$B1_KP"} \
    `# PARK_FIXED：整個時域 u_base = 0（核心等式）` \
    $PK_SOLVER \
    `# **無偏移追蹤**：手臂在抓取姿態下 j2 穩態下垂 +0.0243 rad、` \
    `# j3 −0.0099 rad，模型不知道 ⇒ ALIGN 停在 12 mm。線上估計補上。` \
    --offset-free \
    `# **不**帶 --use-applied-for-predict：那要執行端發 cmd_env 的終點` \
    `# 封裝，抽屜模擬器沒有這條。延遲補償因此用的是請求值而非套用值，` \
    `# 這個差別要寫進報告。` \
    --reach-pos-m 0.005 --reach-rot-rad 0.02 --hold-s 2.0 \
    --duration-s 600 --duration-sim-s 400 \
    --u-prev-policy strict --assume-initial-rest \
    --out "$DIR/align_solver.json"

say "[任務] 發計畫、監看、備妥時要求轉移"
# **慢速區關掉**（--slow-zone 0）：減速交接段已經負責把速度降進全身的
# 速度框並維持，v_nominal 再降一次只會讓參考視窗縮到 30 mm、參考點貼在
# 機器人身上，反而製造左右修正。
python3 -u evaluation/drawer_mission_node.py --out "$DIR/mission.json" \
  --slow-zone 0.0 $PK_NODE \
  2>&1 | tee -a "$DIR/run.log"
RC=${PIPESTATUS[0]}
say "[展開] 等全身端完成展開（最多 120 s）"
for i in $(seq 120); do
  # **等到進入保持交棒**，不是只等「展開完成」—— 那行在前面就印了，
  # 拿它當條件會在全身端還沒進保持時就往下跑（競態）。
  grep -q '保持交棒\|停在\|保持交棒逾時' "$DIR/wholebody.log" 2>/dev/null && break
  sleep 1
done
grep -E '^\[wb\]' "$DIR/wholebody.log" | tail -6 | tee -a "$DIR/run.log"

# ---- ALIGN：求解節點接手，收掉接觸前的最後誤差 ----------------------------
# **範圍（明說）**：安全層**不在**此管線內。抽屜管線本來就是直接發
# /wb_vel_cmd（展開節點如此），而 WG2 自由空間那條是以
# `freespace_confirmed:=true` 放行的 —— 櫃體就在旁邊，那在抽屜房裡是假前提，
# 不能照搬。本趟的保護是命令鏈的三層：低速介面界限 → 輪級 λ → 底盤變化率
# 上限。把櫃體接進真正的障礙物來源是**碰撞安全主張的前置條件**，尚未做。
# 求解器已提早初始化，控制交棒由 RELEASED + ALIGN 訊號決定。
say "[任務] 等編排節點走完全部相位（最多 900 s）"
for i in $(seq 900); do
  grep -qE '\[task\] 結束' "$DIR/task.log" 2>/dev/null && break
  sleep 1
done
grep -E '^\[task\]' "$DIR/task.log" | tee -a "$DIR/run.log"

# **先等求解節點停止並寫完統計，再停物理（A0）**。中止路徑上任務節點會發 /wgmpc/stop；
# 正常路徑求解節點早已在退開完成時停止 ⇒ 立即通過。等待有上限，不無限等；
# 逾時就記下「未確認停止」再往下走（之後收尾時節點以外部關閉寫出統計）。
SOLVER_PID=''
for i in "${!NAMES[@]}"; do [ "${NAMES[$i]}" = solver ] && SOLVER_PID="${PIDS[$i]}"; done
STOP_WAIT_S="${STOP_WAIT_S:-30}"
say "[收尾] 等求解節點停止並封存（最多 ${STOP_WAIT_S} s）"
stop_ok=0
for i in $(seq "$STOP_WAIT_S"); do
  if [ -s "$DIR/align_solver.json" ] && { [ -z "$SOLVER_PID" ] || ! kill -0 "$SOLVER_PID" 2>/dev/null; }; then
    stop_ok=1; break
  fi
  sleep 1
done
if [ "$stop_ok" = 1 ]; then
  say "  求解節點已停止並寫出 align_solver.json"
else
  say "  **求解節點未在 ${STOP_WAIT_S} s 內確認停止**（不再等；收尾時以外部關閉寫出統計，紀錄標為未確認）"
fi
# 停止後再讓物理跑一段（牆鐘），供核對執行端實際套用命令與實測速度確已歸零／減速。
POST_STOP_HOLD_S="${POST_STOP_HOLD_S:-3}"
say "[收尾] 停止後保留物理 ${POST_STOP_HOLD_S} s（牆鐘）供核對執行端減速"
sleep "$POST_STOP_HOLD_S"

say "[收尾] 請模擬器受控停止並封存"
timeout 10 ros2 topic pub --once /room/stop std_msgs/msg/String \
  "{data: '導航交棒與展開已結束，請封存'}" >>"$DIR/run.log" 2>&1 || true
for i in $(seq 90); do
  [ -f "$DIR/room_run.json" ] && break
  sleep 1
done
if [ -f "$DIR/room_run.json" ]; then
  say "  封存完成 $(wc -c <"$DIR/room_run.json") bytes"
else
  say "  **封存未寫出**"
fi
say "=== 結束 rc=$RC ==="
exit $RC
