#!/usr/bin/env python3
"""瓶子 W1・F1 兩瓶連續任務（isaac_bottle_w1dynsr35.py 的版本化複本，2026-11-03；Codex F1 裁定）：遠瓶先抓：/World/bottle（+0.15）→ 第二桌遠端，自主返回原起點，/World/bottle2（−0.15）→ 第二桌近端。
不重置世界、不瞬移關節。控制器 bottle_w1_f1_ctrl＋任務層 bottle_f1_task（第一輪實例含返回段；切換後第二輪新實例，時間全域連續）。
切換時：作用瓶換 /World/bottle2（瓶位姿、接觸視圖 v_bt／v_b2／v_bs 換第二組；手指、手臂、附加連桿的過濾欄把作用瓶換到第 0 欄，另一瓶留在禁止接觸欄）；鏈路 chain、候選連結 slink（清空本輪狀態）、V2 連結換接收者；
發 /f1/round 讓選物節點與候選節點換全新狀態與第二輪區域；導航監看換新實例。另一瓶位姿逐步記錄，也算進穿透估計。第一輪紀錄另存 s1_report_r0.json、s1_cbf_r0.npz、place_log_r0.npz。
瓶子 W1・F1 兩瓶視覺採集（isaac_bottle_w1dynsr34.py 的版本化複本，2026-11-02；Codex 20261102_225852）：
控制器換成 bottle_w1_f1cap_ctrl（carrygov 基線＋第一輪觀看點＋漸變起點前受控停止，不插入、不閉爪）。場景多一支同型瓶 /World/bottle2（--bottle2 x,y，世界座標；同尺寸、質量、材質與顏色），
併入第二桌腳清單（同 /World/dyn_obs 的作法）：底盤、手臂、手指、作用瓶碰到它都算禁止接觸。真值只供場景與事後評估，不進視覺與控制。其餘同 dynsr34。
瓶子 W1・R-B1＋迎面錯身移動物（isaac_bottle_w1dynsr33.py 的版本化複本，2026-11-02；Codex 20261102_200625）：
控制器同 dynsr33（bottle_w1_carrygov_ctrl，基線凍結）。只多把 bottle_dyn_case_region.cases()（experiments/20261102_rb1_dyn_case.json，世界座標）登錄到
bottle_w1_dyn_case.CASES，讓 --dyn-case rb1_dyn_pass1 生成移動圓柱（世界座標驅動；真值只供場景、評估與安全網，控制器只經掃描→追蹤器）。其餘同 dynsr33。
以下為 dynsr33 說明。
瓶子 W1・運送進度協調（isaac_bottle_w1dynsr32.py 的版本化複本，2026-11-02；Codex 20261102_175049／184223）：
完整組控制器換成 bottle_w1_carrygov_ctrl（gaze＋持瓶段持守點：剩餘手臂包絡對第二桌的幾何持守、收近前限直線段入口前、車速停在持守點前 0.04 m；
停止模型保底〔超越監看 carry_hold_overrun、計畫晚到 return_plan_late、非有限 carry_gov_invalid〕、持守等待上限 10 s；停車求解器 zero_box_clip_tol (1e-9, 1e-9)）。
結束時另印持守摘要。其餘同 dynsr32。
以下為 dynsr32 說明。
瓶子 W1・在線視向候選（isaac_bottle_w1dynsr31.py 的版本化複本，2026-11-02；Codex 20261102_092601）：
完整組控制器換成 bottle_w1_gaze_ctrl（online5＋接近段 U／H 只加 q1 視向偏移：實際相機 FK 對已知區域觀看點、中央帶外才修正、速度 0.6 rad/s、加速度 1.5 rad/s² 限制、
R 段起歸零；GMPC、抓取速度、新鮮度、處置期限、安全規則不變）。結束時另印視向摘要。其餘同 dynsr31。
以下為 dynsr31 說明。
瓶子 W1・區域任務 R-B1：抓取區域（第一桌＋瓶）世界平移（isaac_bottle_w1dynsr30.py 的版本化複本，2026-11-02；Codex 20261102_090007）。
--region-offset ox,oy（預設 0,0＝與 dynsr30 相同）：Isaac 場景的第一桌（桌面、桌腳）與瓶放在 原場景位置＋(ox, oy)；機器人起點、第二桌、原障礙方塊維持世界位置。
控制器與上層任務全程在區域座標（bottle_region：世界＝區域＋O）；邊界只轉一次：
  控制器建構 x_start（區域）、障礙物函式（世界→區域）、掃描射線位姿（區域→世界）、放置先驗 PLACE（世界→區域）、候選 snap_cx／snap_cy（世界→區域）；
  主迴圈 ctrl.step 的底盤位姿與瓶位姿真值、導航監看 mprog（世界→區域）；規劃用 bottle_region.PlanRosRegion（請求起訖區域→世界，回覆先格心換算再轉區域，
  代價格查詢時區域點＋O）與 plan_before_start_region（發布 /odom、TF 用世界位姿，請求與核對用區域位姿）。
世界端照舊世界座標：/odom、TF、地圖（bottle_map_write --region-offset）、planner_server、掃描、相機旁路、真值安全網（第一桌盒＝平移後）、紀錄 s1_steps 的 base_x／bottle_*。
以下為 dynsr30 說明。
瓶子 W1・在線 MotM 第五版：online3＋規劃路徑格心換算接口（isaac_bottle_w1dynsr28.py 的版本化複本，2026-11-02；Codex 20261102_083817／085048）：
1 規劃用戶端換成 bottle_n2_ros_center.PlanRosCenter：起跑前 verify()（planner_server 實際參數、已安裝 nav2_smac_planner 版本、當趟地圖解析度與原點），
  全部符合 SmacPlanner2D／1.3.12／未降採樣才對每個回覆換算一次到 costmap 格心慣例（P＋res/2）；否則照舊不換算。原始與換算後路徑都保存（plan.json 與報告）。
2 控制器換成 bottle_w1_online5_ctrl（＝online3 控制與排程；只多記錄持瓶承接當下的在途前推狀態與換算紀錄）。其餘同 dynsr28。
以下為 dynsr28 說明。
瓶子 W1・在線 MotM 第三版：提前安全轉回、與運送重疊（isaac_bottle_w1dynsr27.py 的版本化複本，2026-11-02；Codex 20261102_080657）：
完整組控制器換成 bottle_w1_online3_ctrl（只改升高／外伸起點排程；速度、時長、淨空閘、停止與收斂照舊）。其餘同 dynsr27。
以下為 dynsr27 說明。
瓶子 W1・區域導引在線 MotM 第二版（isaac_bottle_w1dynsr26.py 的版本化複本，2026-11-02；Codex 20261102_075357）：
完整組控制器換成 bottle_w1_online2_ctrl（承接時觀測年齡以來源擷取戳計、≤ 0.5 s；不新鮮則保留結果等新觀測）。其餘同 dynsr26。
以下為 dynsr26 說明。
瓶子 W1・區域導引在線 MotM（isaac_bottle_w1dynsr25.py 的版本化複本，2026-11-02；Howard 20261102_howard_region_guided_online_motm、Codex 072116／072949）：
完整組控制器換成 bottle_w1_online_ctrl（抓取計畫與轉回計畫由背景子程序 bottle_plan_worker 依本趟量測與當下狀態在線產生，控制拍不阻塞；
承接條件、最晚處置點、時長實例參數見該模組）。與 dynsr25 的差別：
- 不再把生成桌高 Z1 寫入 compactcarry2.TOP_Z／_TOP（收近閘／外伸閘的第一桌改用本趟量測支撐高度＝控制器實例參數；模組常數維持資產先驗）。
- 控制器多傳 planner_log＝<out>/plan_worker.log（工作者 stderr）；結束時列印在線計畫紀錄。
- 控制器的 top_z 參數改傳資產先驗高度（原傳 Z1；只被舊 V2／W1 估計路徑使用，segtarget 下已覆寫且只記錄——改傳先驗以確保生成桌高不進任何目標路徑）。
Z1 仍用於：場景生成、判定器離桌判定（真值安全網）、穿透估計與評估、地圖桌腳（2D 位置＝區域先驗）。
以下為 dynsr25 說明。
瓶子 W1・多任務 v3（跨桌高）：轉回高度依放置桌產生（isaac_bottle_w1dynsr24.py 的版本化複本，2026-11-02；Codex 20261102_064116／070114）：
完整組控制器換成 bottle_w1_vhigh_relwin_xh_ctrl（v2w＋轉回目標高度＝第二桌面＋Z_PAD＋H_CLEAR−持瓶／夾爪包絡最低點、轉回時長 5.1／2.1／5.1 s、下放前 TCP 收斂 ≤ 2 mm）。
案例參數與其餘同 dynsr24。
以下為 dynsr24 說明。
瓶子 W1・多任務 v2w：平移滑移超標改為警告（isaac_bottle_w1dynsr23.py 的版本化複本，2026-11-02；Codex 20261102_055603）：
完整組控制器換成 bottle_w1_vhigh_relwin_slipwarn_ctrl（v2 量測目標相對排程＋平移滑移 > 5 mm 只警告、評分基準固定、幾何用當下量測持瓶位姿；轉角／失抓／接觸等停止照舊）。
案例參數與其餘同 dynsr23。
以下為 dynsr23 說明。
瓶子 W1・多任務第一批 v2：量測目標相對排程（isaac_bottle_w1dynsr18.py 的版本化複本，2026-11-02；Howard 多任務交接、Codex 20261102_050244）：
完整組控制器換成 bottle_w1_vhigh_relwin_ctrl（V_HIGH 接近序列＋首次合格量測時以量測瓶 x 平移抓取窗、預抓、漸變、慢速點與回備妥觸發；窗寬不變、凍結；不重選備妥）。
案例參數照既有介面：--bottle-dx／--bottle-dy（生成瓶位，只進場景與評估）、--table1-z、--dyn-case。其餘同 dynsr18。
以下為 dynsr18 說明。
瓶子 W1・移動接近中的高位手腕視角（isaac_bottle_w1dynsr16.py 的版本化複本，2026-11-02；Howard／Codex 20261102_howard_moving_gaze_no_parked_grasp、023000）：
完整組控制器換成 bottle_w1_segtarget_vhigh_ctrl（segtarget 完整堆疊之上只換接近段手臂參考：K2_READY→V_HIGH→保持觀察→名目備妥，實測 x 觸發＋上一段完成才換段）。
沒有停車分支、沒有 OBSERVE。須由 runner 帶 SEG3D=1。其餘同 dynsr16。
以下為 dynsr16 說明。
瓶子 W1・學習模型供目標＋第一桌高度參數（isaac_bottle_w1dynsr15.py 的版本化複本，2026-11-02；Codex 20261102_005118）：加 --table1-z（預設＝資產 0.60）。
第一桌（桌面、桌腳、瓶初始高度）、判定器的 table_top_z（離桌判定）、compactcarry2 的 TOP_Z／_TOP（收近閘與外伸閘的第一桌）、可見性框、持瓶淨空安全網的第一桌都用同一個 Z1；
第二桌與放置先驗維持資產 0.60；手臂備妥分支（T_nom）維持資產 0.60 名目（不依實際桌高重選）。Z1 只用於場景生成、真值安全網與評估，不送候選估計或抓取目標。
以下為 dynsr15 說明。
瓶子 W1・學習模型提供抓取目標（isaac_bottle_w1dynsr14.py 的版本化複本，2026-11-02；Codex 20261102_002139）：完整組控制器換成 bottle_w1_segtarget_ctrl
（目標只來自 /bottle/target_candidates 的 known_type 快照；V2／W1 舊估計只記錄），另加 SegTargetLink（bridge 節點訂閱、每步轉送、時間換任務時刻）。
須由 runner 帶 SEG3D=1 起旁路與候選節點；其餘同 dynsr14。
以下為 dynsr14 說明。
瓶子 W1・緊湊持物 v3 轉回排程候選（isaac_bottle_w1dynsr13.py 的版本化複本，2026-11-01；Codex 20261101_204716 第 2、4 項）：完整組控制器換成 bottle_w1_compactcarry3_ctrl
（v2.1 第一桌閘＋兩條模型停止軌跡、前方距離用 Shield 強制上界；升高提前到距直線段入口 v_carry×T_UP、外伸仍在 PLACE_APPROACH 且須過外伸淨空閘），其餘同 dynsr13。
以下為 dynsr13 說明。
瓶子 W1・緊湊持物修正版 v2（isaac_bottle_w1dynsr12.py 的版本化複本，2026-11-01；Codex 20261101_201934）：完整組控制器換成 bottle_w1_compactcarry2_ctrl
（三段各自起訖速度為零的過渡、時長依整段路徑關節速度／加速度；第一桌淨空閘＝完整瓶身＋夾爪 × 實測／在途／預測／路徑前方位姿），其餘同 dynsr12。
另加 --scan-cut-front FRAC（預設不注入；同 dynsr11 的掃描函式回傳 None）：收近開始（ctrl.cc t0）後 FRAC×T_TO 起不發布掃描，由既有 scan_stale 或 Shield 過期限速觸發受控停止。
以下為 dynsr12 說明。
瓶子 W1・低近身直立運送姿態候選（isaac_bottle_w1dynsr9.py 的版本化複本，2026-11-01；Howard 修正 reviews/20261101_howard_compact_carry_and_motion.md）：完整組控制器換成 bottle_w1_compactcarry_ctrl
（連續展臂基線＋抓持、離桌判定、清出第一桌後收近降低到緊湊姿態；PLACE_APPROACH 先升後伸回原側持放置姿態），其餘同 dynsr9。名目用 --dyn-case none。
以下為 dynsr9 說明。
瓶子 W1・連續展臂接近候選（isaac_bottle_w1dynsr6.py 的版本化複本，2026-11-01；Howard 確認、Codex 20261101_190727 交接）：完整組控制器換成 bottle_w1_continuous_approach_ctrl
（SHROW 子類：固定 3.4 s 展臂改為依底盤位置 x_unfold→x_blend 進度連續展臂；預抓、兩路視覺、閉爪、安全判準不變），其餘同 dynsr6（紀錄補強、TRACKER_SCRIPT 指定追蹤器）。名目用 --dyn-case none。
以下為 dynsr6 說明。
瓶子 W1・追蹤器候選 2 實跑腳本（isaac_bottle_w1dynsr2.py 的版本化複本，2026-11-01；Codex 20261101_140323）：控制（SHROW）、場景、速度、Shield、判準與 dynsr2 相同；
追蹤器節點由 runner 的 TRACKER_SCRIPT 指定（候選 2＝bottle_tracker_stamped_kfcm.py）。紀錄補強再加 tracker_log.npz：每則 /gmpc/obstacles_stamped 原始內容與收到時任務時刻。
以下為 dynsr2 說明。
瓶子 W1・第一拍 Shield 一致列候選＋紀錄補強（isaac_bottle_w1dynsr.py 的版本化複本，2026-11-01；Codex 20261101_132514）：控制與時序完全同 dynsr（SHROW 凍結），只加：
  --dyn-case none：不生成移動物（名目；逐步紀錄的移動物欄位填 NaN）；
  scan_log.npz：每幀實際發布的原始掃描（任務時刻、ROS 戳＝T0＋t、frame lidar_link、掃描時底盤位姿、360 筆距離）；
  cmd_chain.npz：送往 Shield 的每筆命令（序號、步序 k、送出任務時刻、回覆 ROS 時刻、生效任務時刻＝送出＋D_STEPS·PHYS、輸入、輸出、序號核對）與 /shield/diag 原始列（收到時 ROS 時刻＋全欄）。
以下為 dynsr 說明。
瓶子 W1 第二批・第一拍 Shield 一致列候選（isaac_bottle_w1dynlc.py 的版本化複本，2026-11-01；Codex 20261101_130612）：非 --nopred 時控制器為 bottle_w1_shrow_ctrl
（分段抬升＋進 CARRY 重規劃＋每次求解以最新原始掃描與 Shield 公式加第一拍接受列；Shield 不改），其餘同進 CARRY 重規劃版。
以下為進 CARRY 重規劃版說明。
瓶子 W1 第二批・分段抬升＋進 CARRY 重規劃候選（isaac_bottle_w1dynl.py 的版本化複本，2026-11-01；Codex 20261101_125000）：非 --nopred 時控制器為 bottle_w1_liftc_ctrl
（分段抬升＋持瓶段路徑改在進 CARRY 時依在途前推實測請求一次；等待時同一 GMPC 降速保持；回覆時照原核對承接），其餘同分段抬升版。
以下為分段抬升版說明。
瓶子 W1 第二批・抓持後分段抬升候選（isaac_bottle_w1dynm.py 的版本化複本，2026-11-01；Codex 20261101_123440）：非 --nopred 時控制器為 bottle_w1_liftb_ctrl
（凍結 G＋抓持且離桌可靠後 LIFT 改為底盤座標運送式抬升；切換用的瓶位姿與桌面受力與既有離桌判定同源＝模擬真值），其餘同監看修正版。
以下為監看修正版說明。
瓶子 W1 第二批（監看修正版；isaac_bottle_w1dyn.py 的版本化複本，2026-11-01；Codex 20261101_093009）：導航段路徑監看改用獨立進度
（bottle_monitor_progress.MonitorProgress：實測投影、沿末段抵達入口且在側向管內才結束），每步在代價圖回呼與過期核對前更新；控制器、參考、QP 都不動，兩組共用。
以下為原第二批說明。
瓶子 W1 第二批：移動物錯身（isaac_bottle_w1.py 的版本化複本，2026-11-01；Codex 20261101_090533）：控制器為凍結 G（bottle_w1_ctrl），--nopred 換成瞬時組（bottle_w1_nopred_ctrl），其餘同 W1。
移動物（bottle_w1_dyn_case.CASE）：真剛體圓柱（關重力、離地 3 mm、質量 1000 kg），以速度驅動＋位置回授跟世界軌跡（任務時間 t；起跑前停在起點）；
它就是 PhysX 掃描打到的碰撞體（掃描位置＝碰撞位置），併入底盤／手臂／手指／瓶的禁止接觸過濾（併在第二桌腳清單）；逐步記錄讀回的位置與速度。障礙真值只供驅動與評估，不進控制器。
以下為 W1 說明。
瓶子 W1 手腕視覺抓放（isaac_bottle_v2.py 的版本化複本，2026-11-01；Codex 20261101_074314）：底盤相機遠距凍結初始目標（V2），手腕相機近距修正（bottle_w1_ctrl.W1PlaceController），
同一個 GMPC。手腕擷取同 isaac_bottle_w1c（bottle_w1_cam.CamCapture('w1')、/w1/*、w1_* 紀錄、深度有效範圍＝渲染裁切 0.05–10 m，非實體 D435i 測距能力）；
手腕估計經 V2Link（擷取步位姿互核 < 1.0 mm、< 0.1°）送 ctrl.wrist_intake。閉爪前須有手腕穩定量測且限速後目標已到達鎖定值；無手腕估計不退回底盤目標。
以下為 V2 說明。
瓶子 V2（isaac_bottle_v1.py 的版本化複本，2026-11-01；Codex 20261101_071255）：視覺提供抓取目標、真值輔助安全監看的模擬驗證。
控制器換 V2PlaceController：建構時抓取目標為 NaN（不先放真值），接近時由 V1 估計凍結成世界座標目標；估計經 bottle_v2_link.V2Link 轉送，
只轉送該影格擷取步位姿互核通過的估計（< 1.0 mm、< 0.1°，同起跑前第一張影格核對）。手臂備妥分支用任務名目瓶位（資產檔 center_xy，不含擾動）算，是任務先驗，不是讀模擬真值；
本趟名目場景中名目瓶位與實際生成位置相同，須揭露。5 mm 淨空拒絕線、禁止接觸仍用模擬真值瓶位姿（安全網），底盤定位為真值，第二桌放置點為已知位置。
以下為 V1 說明。
瓶子 V1 視覺旁路採集（isaac_bottle_place.py 的版本化複本，2026-11-01）：抓放控制與物理時序完全同 bottle_p1_3（draft-3），只加底盤相機旁路（bottle_v1_cam.V1Capture）。
相機影格、深度、內參、相機位姿以擷取時刻戳發布到 /v1/*，估計節點（bottle_v1_est_node.py，另一程序）回 /v1/bottle_estimate；本程序只記錄與評估，估計不進控制。
以下為放置版說明。
瓶子第二桌放置（isaac_bottle_n2.py 的版本化複本，2026-11-01；設計 第七次進度報告素材/規格/瓶子第二桌放置_最小設計.md）：
場景加第二桌（同資產，位移 PLACE），控制器換 PlaceController；持瓶段在 LIFT 開始時向 planner_server 規劃（bottle_place_ros.RosCarryPlanner）。
新增量測：瓶↔第二桌面接觸力向量（支撐判定）、手指實際開度；新增禁止通道：手指／手臂↔第二桌（併入 finger_table／arm_table）、底盤↔第二桌腳、瓶↔第二桌腳（併入 bottle_env）、抬離後瓶↔第一桌。
瓶位姿、底盤位姿、第二桌位置用真值（第一版，不是視覺）。以下為 N2 說明。
瓶子 N2（isaac_bottle_ros.py 的版本化複本，2026-11-01）：導航段改用 planner_server（SmacPlanner2D）的路徑，進度用弧長（S1Planned）；
握手後先規劃一次（bottle_n2_ros.plan_before_start），沒有有效路徑就不起跑（離開碼 72），不退回 DetourPath；任務中路徑被新障礙佔到或代價圖過期 ⇒ 受控停止，不重規劃。
--plan-only：握手＋規劃後結束（冒煙用）。以下為 N1 說明。
瓶子 N1（isaac_bottle_s1.py 的版本化複本，2026-11-01；Codex 條件核可）：感測保護鏈改走真 ROS 話題，其餘同 S1。
本程序發 /clock、/odom、/tf、/joint_states、/scan_raw、/cmd_vel_pre_shield；scan_relay、追蹤器（bottle_tracker_stamped）、Shield、map_server、RSP 是獨立節點（run_bottle_ros.sh）。
控制器仍在本程序與物理同步，不是即時部署驗證；路線仍是 DetourPath，不是規劃器。命令時序與逾時處置見 bottle_ros_chain。--handshake-only：只做話題、TF、發布者握手後結束，不跑任務（冒煙用）。
以下為原版說明。
瓶子 S1 候選：繞障 → 回到可操作區 → 移動抓取 → 持瓶進出口區（核心 gmpc_wb；開 Isaac 視窗）。規格：第七次進度報告素材/規格/瓶子S1_候選規格.md。
控制邏輯全在 bottle_s1_ctrl.S1Controller（繼承 S0bController、S0aController、S0Judge，與 test_bottle_s1.py 負測同一份）；本檔只做 Isaac 轉接。
場景同 S0b 加一個固定方塊障礙。**導航資料來源：PhysX 射線掃描 → 既有 scan_obstacle_tracker → gmpc_wb；底盤命令一律經既有 scan_safety_shield 才寫入（bottle_nav_chain，原始節點程式、同程序同步）。** 地圖只有桌腳，方塊是未知物。障礙真值只留作評估，不進控制器。
底盤位姿與瓶位姿仍取真值（無定位、無視覺）；無 ROS 命令鏈與 DDS 延遲；參數差異見 bottle_nav_chain。不得稱完整感知導航整合。
參考路徑是工程給定的升餘弦繞行，不是規劃器輸出；命令延遲 8 物理步是注入設定；底盤速度為工程設定與啟發式調速。
量測有缺一律給 None／NaN；必要接觸對全程有效，起跑前形狀／有限值核對失敗即拒跑；僅完整任務（抓持五項＋實際進入出口區＋底盤停下）回 0。不自動補跑。
    $ISAAC_PY evaluation/isaac_bottle_s1.py --asset src/my_omnibot_description/config/bottle_s0.yaml --out <dir>
"""
import argparse
import json
import math
import os
import sys
import time

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--asset', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--x-start', type=float, default=-3.6, help='底盤起點 x（瓶軸在 x＝0）')
ap.add_argument('--no-tcp-preview', action='store_true', help='消融：時域 TCP 目標固定為 T_seq[0]，其餘不變')
ap.add_argument('--s2-box', default='', help='S2 三維障礙 cx,cy,cz,sx,sy,sz；模擬真值進控制器，LiDAR 看不到')
ap.add_argument('--wall-box', default='', help='側牆 cx,cy,cz,sx,sy,sz：進場景、掃描、地圖與接觸判定')
ap.add_argument('--s2-links', default='', help='納入約束的連桿，逗號分隔，例如 link3,link4,link5')
ap.add_argument('--s2-active', action='store_true', help='開啟預測距離約束（無此旗標時障礙在場但控制器不知道）')
ap.add_argument('--no-inflight', action='store_true', help='消融：求解初態不沿在途命令前推；實際延遲與安全核對不變')
ap.add_argument('--group', default='G', choices=['G', 'P', 'M'], help='對照組：G gmpc_wb（凍結）、P 停車操作、M MotM 式單步 QP')
ap.add_argument('--m-kp', type=float, default=4.0)
ap.add_argument('--m-mu', type=float, default=0.01)
ap.add_argument('--y-start', type=float, default=0.0, help='幾何擾動：底盤起點 y')
ap.add_argument('--yaw-start-deg', type=float, default=0.0, help='幾何擾動：底盤起始偏航')
ap.add_argument('--bottle-dx', type=float, default=0.0, help='幾何擾動：瓶位置 x（瓶位姿仍是真值進控制器）')
ap.add_argument('--bottle-dy', type=float, default=0.0)
ap.add_argument('--x-end', type=float, default=0.85, help='底盤參考終點 x（出口區內）')
ap.add_argument('--obst-x', type=float, default=-2.0)
ap.add_argument('--path', default='-3.4,-0.6,-0.81', help='升餘弦繞行 x1,x2,A')
ap.add_argument('--exit-c', default='1.0,0.0')
ap.add_argument('--exit-r', type=float, default=0.30)
ap.add_argument('--win-lo', type=float, default=-0.35)
ap.add_argument('--win-hi', type=float, default=0.15)
ap.add_argument('--v-c', type=float, default=0.10, help='接近／抓取參考速度上限 m/s（啟發式調速只減速不加速）')
ap.add_argument('--v-nav', type=float, default=0.35)
ap.add_argument('--v-carry', type=float, default=0.25)
ap.add_argument('--x-unfold', type=float, default=-1.55)
ap.add_argument('--t-unfold', type=float, default=3.4)
ap.add_argument('--t-blend', type=float, default=3.0)
ap.add_argument('--t-insert', type=float, default=1.5)
ap.add_argument('--x-slow', type=float, default=-0.90)
ap.add_argument('--q-tuck', default='0,0,0.035,0,-1.5708,0', help='起始與行進姿態：K2_READY（Howard 指定）；開始後不直接設定關節')
ap.add_argument('--sim-s', type=float, default=90.0)
ap.add_argument('--settle-s', type=float, default=1.0, help='開始前保持（量靜態偏移 d̂ 初值）')
ap.add_argument('--stop-tail-s', type=float, default=2.0, help='停止或完成後續跑（核停止有效性）')
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--headless', default='false')
ap.add_argument('--handshake-only', action='store_true')
ap.add_argument('--plan-only', action='store_true')
ap.add_argument('--v1-hz', type=float, default=5.0)
ap.add_argument('--goal', default='-0.9,0.0', help='規劃目標＝直線段入口（任務定義）')
ap.add_argument('--scan-cut-front', type=float, default=None, help='注入：收近開始後 FRAC×T_TO 起停止發布掃描；預設不注入')
ap.add_argument('--nopred', action='store_true', help='瞬時組：動態障礙速度置 0（bottle_w1_nopred_ctrl）')
ap.add_argument('--bottle2', default='', help='第二支同型瓶的世界 x,y（F1）；空＝單瓶')
ap.add_argument('--dyn-case', default='w1_dyn_pass1', help='bottle_w1_dyn_case.CASES 的鍵（封存案例）；none＝不生成移動物（名目）')
ap.add_argument('--x-lane', type=float, default=-0.9)
ap.add_argument('--table1-z', type=float, default=None, help='第一桌桌面頂（m）；預設＝資產 0.60')
ap.add_argument('--region-offset', default='0,0', help='抓取區域（第一桌＋瓶）世界平移 ox,oy（R-B1）；第二桌、起點、原障礙不動')
a = ap.parse_args()

from isaacsim import SimulationApp                                   # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                                   # noqa: E402
import yaml                                                          # noqa: E402
from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.api.objects import DynamicCylinder, FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation        # noqa: E402
from isaacsim.core.utils.types import ArticulationAction             # noqa: E402

HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from isaac_common import bind_friction, bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402
from finger_collision import apply_split                             # noqa: E402
import bottle_branch as BB                                           # noqa: E402
import bottle_ros_chain as RC                                        # noqa: E402
NC, C2, GR, N2C, PCm, V1C, V2, W1C, W1 = RC.import_with_real_ros('bottle_nav_chain', 'bottle_s1_ctrl', 'bottle_s1_groups', 'bottle_n2_ctrl', 'bottle_place_ctrl', 'bottle_v1_cam', 'bottle_v2_ctrl', 'bottle_w1_cam', 'bottle_w1_ctrl')
W1G, W1N = RC.import_with_real_ros('bottle_w1_f1_ctrl', 'bottle_w1_nopred_ctrl')   # 運送進度協調（gaze＋持守點）   # 在線視向候選（online5＋q1 視向）   # 在線第五版（online3＋格心換算接口）   # 在線第三版（提前安全轉回）   # 在線第二版（來源擷取戳新鮮度）   # 區域導引在線 MotM   # 多任務 v3：跨桌高轉回   # 多任務 v2w：平移滑移警告   # 多任務 v2：量測目標相對排程   # 完整組換成 V_HIGH 接近序列   # 完整組換成第一拍 Shield 一致列候選
import bottle_f1_contact as FC                                        # noqa: E402
import bottle_f1_crash as FX                                          # noqa: E402
import bottle_f1_task as FT                                           # noqa: E402
import bottle_w1_dyn_case as DC                                       # noqa: E402
import bottle_dyn_case_region as DCR                                  # noqa: E402
DC.CASES.update(DCR.cases())                                         # R-B1 迎面錯身（世界座標）
import bottle_segtarget_link as SL                                   # noqa: E402
import bottle_monitor_progress as MPG                                 # noqa: E402
W1 = W1N if a.nopred else W1G   # 不讓 ros_stub 蓋掉真 rclpy
import bottle_n2_path as NP                                          # noqa: E402
import bottle_n2_ros as NR                                           # noqa: E402
import bottle_place_ros as PRS                                       # noqa: E402
import bottle_ref as BRF                                             # noqa: E402
import bottle_s0_ctrl as C                                           # noqa: E402
import bottle_s1_geom as G                                           # noqa: E402
import bottle_v2_link as VL                                          # noqa: E402

F1_ROUNDS = (dict(x_look=0.15, target=(2.55, -0.58), region=(0.0, 0.35, 0.36, 0.62)), dict(x_look=-0.15, target=(2.25, -0.58), region=(-0.35, 0.0, 0.36, 0.62)))   # 凍結：遠瓶先抓、遠放置區先放
ROBOT = '/World/omni_bot'
ARM = [f'joint{i}' for i in range(1, 7)]


def yaw_of(q):
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def ray_range(query, origin, direction, max_range, excluded):
    """同 ev9r9a：回傳 (距離, 命中路徑)，未命中為 inf。"""
    best = [float('inf'), None]

    def report(hit):
        coll = str(getattr(hit, 'collision', '') or '')
        body = str(getattr(hit, 'rigidBody', '') or '')
        d = float(getattr(hit, 'distance', 0.0))
        if any(coll == e or (body and e.startswith(body + '/')) for e in excluded):
            return True
        if 0.0 <= d < best[0]:
            best[0], best[1] = d, (coll or body)
        return True

    query.raycast_all(origin, direction, max_range, report)
    return best[0], best[1]


def read_pairs(view, n_filters, dt):
    """回傳 (n_sensor, n_filter, 3) 陣列；讀不到回 None（不當零）。"""
    try:
        M = np.asarray(view.get_contact_force_matrix(dt=dt), float)
        return M.reshape(-1, n_filters, 3)
    except Exception:
        return None


def main():
    S = yaml.safe_load(open(a.asset))
    T, B, PH = S['table'], S['bottle'], S['physics']
    Z1 = float(T['top_z']) if a.table1_z is None else float(a.table1_z)
    T1 = dict(T, top_z=Z1)                                                    # 第一桌（場景、離桌判定、安全網、評估）
    import bottle_region as BRG
    RO = np.array([float(v) for v in a.region_offset.split(',')] + [0.0])
    REG = BRG.RegionFrame(RO[0], RO[1])
    T1TB = [(n_, c_ + RO, sz_) for n_, c_, sz_ in G.table_boxes(dict(G.S1_TABLE, top_z=Z1))]   # 世界：第一桌平移後（真值安全網、評估、可見度）
    print(f'[s1] 區域偏移 {RO[:2].tolist()}（第一桌與瓶；第二桌、起點、原障礙不動）', flush=True)
    T2TB = G.table_boxes(dict(G.S1_TABLE, top_z=float(T['top_z'])))          # 第二桌維持資產高度
    # 在線版：不把生成桌高寫入 compactcarry2.TOP_Z（收近閘／外伸閘的第一桌由控制器以本趟量測支撐高度設定）
    print(f'[s1] 第一桌桌面頂 {Z1:.3f} m（第二桌 {float(T["top_z"]):.3f} m；備妥分支維持資產名目）', flush=True)
    dt = float(PH['dt'])
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics
    scn = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.Scene)]
    if len(scn) != 1:
        print(f'[s1] 物理場景數 {len(scn)} ≠ 1 ⇒ 拒跑', flush=True)
        return 67
    sapi = PhysxSchema.PhysxSceneAPI(scn[0]) if scn[0].HasAPI(PhysxSchema.PhysxSceneAPI) else PhysxSchema.PhysxSceneAPI.Apply(scn[0])
    sapi.CreateEnableExternalForcesEveryIterationAttr().Set(bool(PH.get('ext_forces_every_iter')))
    # 桌與瓶（同 isaac_bottle_smoke）
    FixedCuboid(prim_path='/World/table/top', name='table_top', position=np.array([0.0, T1['near_y'] + T1['size_y'] / 2, T1['top_z'] - T1['thick'] / 2]) + RO,
                scale=np.array([T1['size_x'], T1['size_y'], T1['thick']]), color=np.array([0.55, 0.42, 0.30]))
    hz = T1['top_z'] - T1['thick']
    leg_paths = []
    for sx in (-1, 1):
        lx = sx * (T['size_x'] / 2 - T['leg_inset_x'] - T['leg'] / 2)
        for nm, ly in (('near', T['near_y'] + T['leg_inset_near'] + T['leg'] / 2), ('far', T['near_y'] + T['size_y'] - T['leg_inset_far'] - T['leg'] / 2)):
            pth = f'/World/table/leg_{nm}_{"p" if sx > 0 else "m"}'
            FixedCuboid(prim_path=pth, name=f'leg_{nm}_{sx}', position=np.array([lx, ly, hz / 2]) + RO, scale=np.array([T['leg'], T['leg'], hz]), color=np.array([0.4, 0.3, 0.2]))
            leg_paths.append(pth)
    OB_C, OB_S = np.array([a.obst_x, 0.0, 0.25]), np.array([0.40, 0.40, 0.50])
    FixedCuboid(prim_path='/World/obstacle', name='obstacle', position=OB_C, scale=OB_S, color=np.array([0.7, 0.25, 0.2]))
    WALL = None
    if a.wall_box:
        v_ = [float(x) for x in a.wall_box.split(',')]
        WALL = (np.array(v_[:3]), np.array(v_[3:6]))
        FixedCuboid(prim_path='/World/wall', name='wall', position=WALL[0], scale=WALL[1], color=np.array([0.5, 0.5, 0.55]))
    S2 = None
    if a.s2_box:
        v_ = [float(x) for x in a.s2_box.split(',')]
        S2 = (np.array(v_[:3]), np.array(v_[3:6]))
        FixedCuboid(prim_path='/World/s2_box', name='s2_box', position=S2[0], scale=S2[1], color=np.array([0.2, 0.7, 0.3]))
    T2D = np.array([PCm.PLACE['target'][0] - float(B['center_xy'][0]), PCm.PLACE['lane_y'], 0.0])     # 第二桌＝同資產整體位移
    hz = T['top_z'] - T['thick']                                              # 第二桌桌腳高（資產高度）
    FixedCuboid(prim_path='/World/table2/top', name='table2_top', position=np.array([0.0, T['near_y'] + T['size_y'] / 2, T['top_z'] - T['thick'] / 2]) + T2D,
                scale=np.array([T['size_x'], T['size_y'], T['thick']]), color=np.array([0.45, 0.50, 0.62]))
    leg2_paths = []
    for sx in (-1, 1):
        lx = sx * (T['size_x'] / 2 - T['leg_inset_x'] - T['leg'] / 2)
        for nm, ly in (('near', T['near_y'] + T['leg_inset_near'] + T['leg'] / 2), ('far', T['near_y'] + T['size_y'] - T['leg_inset_far'] - T['leg'] / 2)):
            pth = f'/World/table2/leg_{nm}_{"p" if sx > 0 else "m"}'
            FixedCuboid(prim_path=pth, name=f'leg2_{nm}_{sx}', position=np.array([lx, ly, hz / 2]) + T2D, scale=np.array([T['leg'], T['leg'], hz]), color=np.array([0.35, 0.38, 0.45]))
            leg2_paths.append(pth)
    BX, BY = float(B['center_xy'][0]) + a.bottle_dx + RO[0], float(B['center_xy'][1]) + a.bottle_dy + RO[1]   # 世界瓶位（只交場景與評估）
    bottle = DynamicCylinder(prim_path='/World/bottle', name='bottle', position=np.array([BX, BY, Z1 + B['height'] / 2]),
                             radius=B['radius'], height=B['height'], mass=B['mass'], color=np.array([0.2, 0.6, 0.9]))
    bind_friction(stage, [stage.GetPrimAtPath('/World/table/top'), stage.GetPrimAtPath('/World/table2/top')] + [stage.GetPrimAtPath(p) for p in leg_paths + leg2_paths], T['material']['static_friction'],
                  T['material']['dynamic_friction'], T['material']['restitution'], mat_path='/World/PhysicsMaterials/table')
    bind_friction(stage, [stage.GetPrimAtPath('/World/bottle')], B['material']['static_friction'], B['material']['dynamic_friction'],
                  B['material']['restitution'], mat_path='/World/PhysicsMaterials/bottle')
    bottle2 = None
    if a.bottle2:
        B2X, B2Y = (float(v) for v in a.bottle2.split(','))
        bottle2 = DynamicCylinder(prim_path='/World/bottle2', name='bottle2', position=np.array([B2X, B2Y, Z1 + B['height'] / 2]),
                                  radius=B['radius'], height=B['height'], mass=B['mass'], color=np.array([0.2, 0.6, 0.9]))
        bind_friction(stage, [stage.GetPrimAtPath('/World/bottle2')], B['material']['static_friction'], B['material']['dynamic_friction'],
                      B['material']['restitution'], mat_path='/World/PhysicsMaterials/bottle2')
        leg2_paths.append('/World/bottle2')                                   # 同 dyn_obs：併入第二桌腳清單，底盤、手臂／手指、作用瓶碰到都算禁止接觸
    CA = None if a.dyn_case == 'none' else DC.CASES[a.dyn_case]
    if CA is not None:
        DZ = CA['height'] / 2 + 0.003
        dyn = DynamicCylinder(prim_path='/World/dyn_obs', name='dyn_obs', position=np.array([CA['start'][0], CA['start'][1], DZ]), radius=CA['radius'], height=CA['height'],
                              mass=1000.0, color=np.array([0.85, 0.55, 0.15]))
        _dp = stage.GetPrimAtPath('/World/dyn_obs')
        (PhysxSchema.PhysxRigidBodyAPI(_dp) if _dp.HasAPI(PhysxSchema.PhysxRigidBodyAPI) else PhysxSchema.PhysxRigidBodyAPI.Apply(_dp)).CreateDisableGravityAttr().Set(True)
        leg2_paths.append('/World/dyn_obs')                                       # 併入第二桌腳清單：底盤 base_env、手臂／手指 table、瓶 bottle_env 的禁止接觸都涵蓋移動物
    # 機器人（同 ev9r9a）
    base = np.array([a.x_start, a.y_start, math.radians(a.yaw_start_deg)])
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(base[0]), float(base[1]), 0.0))
    xf.AddRotateZOp().Set(float(a.yaw_start_deg))
    fc = apply_split(stage, ROBOT)
    sph = collision_prims(stage, ROBOT, lambda p: 'support_' in p)
    gr = collision_prims(stage, '/World/ground')
    if bind_frictionless(stage, sph + gr) == 0:
        print('[s1] 零摩擦材質沒綁到 ⇒ 拒跑', flush=True)
        return 19
    grip = [pr for pr in collision_prims(stage, ROBOT) if 'finger' in str(pr.GetPath()).lower()]
    if bind_friction(stage, grip, B['material']['static_friction'], B['material']['dynamic_friction']) == 0:
        print('[s1] 夾持面摩擦沒綁到 ⇒ 拒跑', flush=True)
        return 21
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(600.0)
    bodies, arts = physics_parts(stage, ROBOT)
    if not arts:
        print('[s1] 找不到 articulation root ⇒ 拒跑', flush=True)
        return 15
    fingers = sorted(b for b in bodies if str(b).rstrip('/').split('/')[-1] in ('uflite_finger1', 'uflite_finger2'))
    links = sorted(b for b in bodies if str(b).rstrip('/').split('/')[-1] in tuple(f'link{i}' for i in range(1, 7)))
    extra = [b for b in bodies if str(b).rstrip('/').split('/')[-1] in ('link_eef', 'uflite_gripper_link')]
    if len(fingers) != 2 or len(links) != 6:
        print(f'[s1] 剛體不符：手指 {fingers} 連桿 {links} ⇒ 拒跑', flush=True)
        return 16
    print(f'[s1] 剛體 {len(bodies)}；手指 {fingers}；附加 {extra}；手指拆塊 {list(fc.get("fingers", {}))}', flush=True)
    table_paths = ['/World/table/top'] + leg_paths + ['/World/table2/top'] + leg2_paths + ['/World/obstacle'] + (['/World/s2_box'] if S2 else []) + (['/World/wall'] if WALL else [])   # 含障礙：禁止接觸對涵蓋 桌＋障礙
    # 接觸視圖（須在 world.reset 前建；用字元類別，不用路徑清單）
    v_finger = RigidPrim(prim_paths_expr=str(fingers[0])[:-1] + '[12]', name='v_finger', track_contact_forces=True, max_contact_count=128,
                         prepare_contact_sensors=True, contact_filter_prim_paths_expr=['/World/bottle'] + table_paths)
    # 連桿路徑是巢狀的（link1/link2/…），字元類別只比對到 link1（bottle_s0a_1 拒跑）⇒ 每根連桿各一個視圖
    v_arm = [RigidPrim(prim_paths_expr=str(b), name=f'v_arm{i}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                       contact_filter_prim_paths_expr=['/World/bottle'] + table_paths) for i, b in enumerate(links)]
    v_extra = [RigidPrim(prim_paths_expr=str(b), name=f'v_extra{i}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                         contact_filter_prim_paths_expr=['/World/bottle'] + table_paths) for i, b in enumerate(extra)]
    b_base = [b for b in bodies if str(b).rstrip('/').split('/')[-1] == 'base_link']
    if len(b_base) != 1:
        print(f'[s1] base_link 剛體不符 {b_base} ⇒ 拒跑', flush=True)
        return 16
    v_base = RigidPrim(prim_paths_expr=str(b_base[0]), name='v_base', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                       contact_filter_prim_paths_expr=['/World/obstacle'] + leg_paths + leg2_paths + (['/World/wall'] if WALL else []))
    v_bt = RigidPrim(prim_paths_expr='/World/bottle', name='v_bt', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                     contact_filter_prim_paths_expr=['/World/table/top'])
    b_env = (['/World/s2_box'] if S2 else []) + (['/World/wall'] if WALL else []) + leg2_paths   # 瓶↔S2 方塊、側牆、第二桌腳：缺資料不當零
    v_b2 = RigidPrim(prim_paths_expr='/World/bottle', name='v_b2', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                     contact_filter_prim_paths_expr=['/World/table2/top'])   # 瓶↔第二桌面：指定支撐，力向量做支撐判定
    v_bs = RigidPrim(prim_paths_expr='/World/bottle', name='v_bs', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                     contact_filter_prim_paths_expr=b_env) if b_env else None
    if bottle2 is None:
        print('[s1] F1 需要 --bottle2 ⇒ 拒跑', flush=True)
        return 2
    b_env2 = (['/World/s2_box'] if S2 else []) + (['/World/wall'] if WALL else []) + [p_ for p_ in leg2_paths if p_ != '/World/bottle2'] + ['/World/bottle']   # 第二輪：作用瓶↔已放第一瓶也禁止接觸
    v_bt_2 = RigidPrim(prim_paths_expr='/World/bottle2', name='v_bt_2', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=['/World/table/top'])
    v_b2_2 = RigidPrim(prim_paths_expr='/World/bottle2', name='v_b2_2', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=['/World/table2/top'])
    v_bs_2 = RigidPrim(prim_paths_expr='/World/bottle2', name='v_bs_2', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=b_env2)
    F1_PERM, _ = FC.filter_perm(table_paths)                                  # 第二輪過濾欄：作用瓶換到第 0 欄（/World/bottle 換到原 /World/bottle2 的禁止接觸欄）
    world.reset()
    if v_bs is not None:
        v_bs.initialize()
    for v in [v_finger, v_bt, v_base, v_b2] + v_arm + v_extra:
        v.initialize()
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    if any(j not in names for j in ARM + ['finger_joint1', 'finger_joint2']):
        print(f'[s1] 缺關節 {names} ⇒ 拒跑', flush=True)
        return 16
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    T_nom = C.grasp_target(float(B['center_xy'][0]), float(B['center_xy'][1]), float(T['top_z']) + 0.10)   # 任務名目瓶位（資產檔，不含擾動）：只用來算手臂備妥分支
    T_p = T_nom.copy()
    T_p[:3, 3] -= 0.06 * T_nom[:3, 2]
    br = BB.window_branch(T_p, T_nom, BRF.lift_pose(T_nom, 0.05), a.win_lo, a.win_hi)
    T_g = np.full((4, 4), np.nan)                                             # V2：抓取目標只來自視覺凍結；凍結前任何讀取都會變成非有限值
    q_ready, T_ready_rel, _ = BB.ready_in_branch(np.array([a.win_lo + 0.05, 0.0, 0.0]), T_p, br['q_mid'], br['x_mid'])
    q_tuck = np.array([float(v) for v in a.q_tuck.split(',')])
    q0 = robot.get_joint_positions()
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i, j in zip(idx, range(6)):
        q0[i] = float(q_tuck[j])
        kp[i], kd[i] = a.kp, a.kd
    q0[f1], q0[f2] = C.FINGER_OPEN, C.FINGER_OPEN
    kp[f1], kd[f1] = a.finger_kp, a.finger_kd                                     # 主從：只驅動 finger_joint1（同 --finger-leader-only）
    act = np.array(sorted(idx + [f1]), np.int32)
    robot.set_joint_positions(q0)
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)

    def apply(tgt_full):
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.asarray(tgt_full, np.float32)[act], joint_indices=act))

    def base_zero():
        robot.set_linear_velocity(np.zeros(3, np.float32))
        robot.set_angular_velocity(np.zeros(3, np.float32))

    def base_cmd(u, yw):
        """本體速度 → 世界速度（同 ev9r9a 寫法）。"""
        vx, vy, wz = (float(v) for v in u)
        robot.set_linear_velocity(np.array([vx * math.cos(yw) - vy * math.sin(yw), vx * math.sin(yw) + vy * math.cos(yw), 0.0], dtype=np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, wz], dtype=np.float32))

    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([-0.6, -4.2, 2.6]), target=np.array([-1.0, -0.2, 0.4]))
    except Exception as e:
        print(f'[s1] 視角設定失敗（不影響物理）：{e!r}', flush=True)
    readback = dict(solver_type=str(sapi.GetSolverTypeAttr().Get()), ext_forces_every_iter=bool(sapi.GetEnableExternalForcesEveryIterationAttr().Get()),
                    physics_dt=float(world.get_physics_dt()), bottle_mass=float(bottle.get_mass()), kp=a.kp, kd=a.kd, base=[float(v) for v in base],
                    q_ready=[float(v) for v in q_ready], q_tuck=[float(v) for v in q_tuck], group=a.group, wall_box=a.wall_box, s2_box=a.s2_box, s2_active=a.s2_active, no_tcp_preview=a.no_tcp_preview, no_inflight=a.no_inflight, m=dict(kp=a.m_kp, mu=a.m_mu) if a.group == 'M' else None, perturb=dict(bottle_dx=a.bottle_dx, bottle_dy=a.bottle_dy, x_start=a.x_start, y_start=a.y_start, yaw_start_deg=a.yaw_start_deg), speeds=dict(v_nav=a.v_nav, v_c=a.v_c, v_carry=a.v_carry, x_slow=a.x_slow, x_unfold=a.x_unfold, t_unfold=a.t_unfold, t_blend=a.t_blend, t_insert=a.t_insert), branch_margin=float(br['margin']), finger_split=bool(fc.get('fingers')))
    print(f'[s1] 讀回 {readback}', flush=True)
    if abs(readback['physics_dt'] - dt) > 1e-12 or readback['ext_forces_every_iter'] != bool(PH.get('ext_forces_every_iter')):
        print('[s1] 物理設定讀回不符 ⇒ 拒跑', flush=True)
        return 64

    def q9():
        p, qt = robot.get_world_pose()
        jp = robot.get_joint_positions()
        return np.r_[float(p[0]), float(p[1]), yaw_of(qt), [float(jp[i]) for i in idx]], jp

    # 靜態保持：量 d̂ 初值
    tgt = np.array(robot.get_joint_positions(), float)
    for _ in range(int(round(a.settle_s / dt))):
        apply(tgt)
        base_zero()
        world.step(render=True)
    q_s, _ = q9()
    d0 = q_s[3:] - q_tuck
    M_, M2_ = read_pairs(v_bt, 1, dt), read_pairs(v_bt_2, 1, dt)              # 力方向約定：瓶靜置第一桌時桌面給瓶的力應向上；兩支瓶都核，第二桌與環境應為零（映射不符就拒跑）
    M3_, M4_, M5_, M6_ = read_pairs(v_b2, 1, dt), read_pairs(v_b2_2, 1, dt), read_pairs(v_bs, len(b_env), dt), read_pairs(v_bs_2, len(b_env2), dt)
    fz0, fz2_ = (float('nan') if m_ is None else float(m_[0, 0, 2]) for m_ in (M_, M2_))
    f_tmp_ = [float('nan') if m_ is None else float(np.linalg.norm(m_, axis=2).max()) for m_ in (M3_, M4_, M5_, M6_)]
    ok_s_, why_s_, F_SIGN = FC.support_check(fz0, fz2_, f_tmp_[0], f_tmp_[1], f_tmp_[2], f_tmp_[3], float(B['mass']) * 9.81)
    p1_, _ = bottle.get_world_pose()
    p2_, _ = bottle2.get_world_pose()
    if not ok_s_ or math.hypot(float(p1_[0]) - BX, float(p1_[1]) - BY) > 0.01 or math.hypot(float(p2_[0]) - B2X, float(p2_[1]) - B2Y) > 0.01:
        print(f'[s1] 動作前兩瓶支撐／位置核對未過（{why_s_}）：第一桌 {fz0}／{fz2_}、第二桌與環境 {f_tmp_}、位置 {[float(v) for v in p1_[:2]]}／{[float(v) for v in p2_[:2]]} ⇒ 拒跑', flush=True)
        return 68
    readback['f_sign'] = dict(fz0=fz0, fz0_bottle2=fz2_, t2_env=f_tmp_, sign=F_SIGN, mg=float(B['mass']) * 9.81)
    print(f'[s1] 接觸力方向約定：起始瓶↔第一桌 fz {fz0:.3f} N ⇒ 符號 {F_SIGN:+.0f}', flush=True)
    n_filters = 1 + len(table_paths)
    required = ['finger_table', 'arm_bottle', 'arm_table', 'base_env', 'bottle_table1', 'bottle_table2'] + (['bottle_env'] if b_env else []) + [f'extra{i}_{w}' for i in range(len(v_extra)) for w in ('bottle', 'table')]
    # 動作前：所有接觸視圖形狀、感測數、有限值都須有效，否則拒跑（不當成沒有接觸）
    for nm, v, ns, nf in [('bottle_table_r1', v_bt_2, 1, 1), ('bottle_table2_r1', v_b2_2, 1, 1), ('bottle_env_r1', v_bs_2, 1, len(b_env2)), ('finger', v_finger, 2, n_filters), ('bottle_table', v_bt, 1, 1), ('bottle_table2', v_b2, 1, 1), ('base', v_base, 1, 1 + len(leg_paths) + len(leg2_paths) + (1 if WALL else 0))] + ([('bottle_env', v_bs, 1, len(b_env))] if b_env else []) + [(f'arm{i}', v, 1, n_filters) for i, v in enumerate(v_arm)] \
            + [(f'extra{i}', v, 1, n_filters) for i, v in enumerate(v_extra)]:
        M = read_pairs(v, nf, dt)
        if M is None or M.shape != (ns, nf, 3) or not np.isfinite(M).all():
            print(f'[s1] 接觸視圖 {nm} 無效（形狀 {None if M is None else M.shape}，期望 {(ns, nf, 3)}）⇒ 拒跑', flush=True)
            return 68
    # 掃描鏈：障礙只從這裡進控制器；OB_C 真值只留作評估
    lidar_prim = next((pr for pr in stage.Traverse() if str(pr.GetPath()).startswith(ROBOT) and pr.GetName() == 'lidar_link'), None)
    if lidar_prim is None:
        print('[s1] 找不到 lidar_link ⇒ 拒跑', flush=True)
        return 18
    excl = [str(pr.GetPath()) for pr in stage.Traverse() if str(pr.GetPath()).startswith(ROBOT) and pr.HasAPI(UsdPhysics.CollisionAPI)
            and (str(pr.GetPath()).endswith('/base_link/cylinder') or '/lidar_link/' in str(pr.GetPath()))]
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()
    xfc = UsdGeom.XformCache()

    scan_log, diag_log = [], []
    scut = dict(ctrl=None, t_cut=None, last=None, n_drop=0)

    def scan_fn(t_, q_):
        cc_ = scut['ctrl']
        if a.scan_cut_front is not None and cc_ is not None and getattr(cc_, 'cc', None) is not None:   # 注入：收近過渡中途起不再發布掃描
            if scut['t_cut'] is None:
                scut['t_cut'] = float(cc_.cc['t0'] + a.scan_cut_front * W1G.C2.T_TO)
            if t_ >= scut['t_cut'] - 1e-9:
                scut['n_drop'] += 1
                return None
        r_ = _scan_fn(t_, q_)
        scut['last'] = float(t_)
        scan_log.append((float(t_), [float(v) for v in q_[:3]], r_.copy()))
        return r_

    def _scan_fn(t_, q_):
        xfc.Clear()
        o5 = xfc.GetLocalToWorldTransform(lidar_prim).ExtractTranslation()
        org = [float(o5[0]), float(o5[1]), float(o5[2])]
        ang = NC.scan_angles() + float(q_[2])
        return np.array([ray_range(query, org, [math.cos(v), math.sin(v), 0.0], NC.RANGE_MAX, excl)[0] for v in ang])

    leg_boxes = [(b[1], b[2]) for b in T1TB if b[0] != 'table_top'] + ([WALL] if WALL else [])   # 地圖：桌腳＋側牆
    import rclpy                                                              # 地圖（leg_boxes 同一份）由 run_bottle_ros.sh 寫檔給 map_server
    rclpy.init()
    chain = RC.RosChain()
    _cb_diag0 = chain.cb_diag

    def _cb_diag_rec(d):                                                      # 紀錄補強：原始 diag 列與收到時刻；原回呼照常
        diag_log.append([float(chain.ros_now)] + [float(v) for v in d])
        return _cb_diag0(d)
    chain.cb_diag = _cb_diag_rec
    trk_log = []
    _cb_obs0 = chain.cb_obs

    def _cb_obs_rec(d):                                                       # 紀錄補強：追蹤器原始輸出（序號、掃描戳、動態／靜態值）；原回呼照常
        trk_log.append((float(chain.ros_now), [float(v) for v in d]))
        return _cb_obs0(d)
    chain.cb_obs = _cb_obs_rec
    node = RC.make_bridge(chain)
    chain.tx = node
    from isaacsim.sensors.camera import Camera
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Float64MultiArray
    v1_boxes = [(c_, sz_) for _, c_, sz_ in T1TB] + [(OB_C, OB_S)] + [(c_ + T2D, sz_) for _, c_, sz_ in T2TB]
    v1 = V1C.V1Capture(Camera, str(b_base[0]), a.urdf, node, a.v1_hz, dt, v1_boxes, dict(Image=Image, CameraInfo=CameraInfo, PoseStamped=PoseStamped, Float64MultiArray=Float64MultiArray))
    b_eef = [b for b in bodies if str(b).rstrip('/').split('/')[-1] == 'link_eef']
    if len(b_eef) != 1:
        print(f'[s1] link_eef 剛體不符 {b_eef} ⇒ 拒跑', flush=True)
        return 16
    w1 = W1C.CamCapture('w1', Camera, str(b_eef[0]), a.urdf, node, a.v1_hz, dt, v1_boxes, dict(Image=Image, CameraInfo=CameraInfo, PoseStamped=PoseStamped, Float64MultiArray=Float64MultiArray))
    v1_first, w1_first = None, None                                           # 起跑前保持期確認第一張影格（兩台）：尺寸、有效深度、內參、時間可配對、位姿互核；不過即拒跑
    for _k in range(int(round(2.0 / dt))):
        apply(tgt)
        base_zero()
        world.step(render=True)
        q_c, _ = q9()
        for cap in (v1, w1):
            cap.record_step(float(world.current_time), float('nan'), float('nan'), q_c, None, -1)
        if not (v1_first and v1_first.get('ok')):
            v1_first = v1.poll(dry=True) or v1_first
        if not (w1_first and w1_first.get('ok')):
            w1_first = w1.poll(dry=True) or w1_first
        if v1_first and v1_first.get('ok') and w1_first and w1_first.get('ok'):
            break
    v1.hist.clear()
    w1.hist.clear()
    print(f"[s1] V1 第一張影格核對：{ {k: v for k, v in (v1_first or {}).items() if k != 'K'} }", flush=True)
    print(f"[s1] W1 第一張影格核對：{ {k: v for k, v in (w1_first or {}).items() if k != 'K'} }", flush=True)
    if not (v1_first and v1_first.get('ok')) or not (w1_first and w1_first.get('ok')):
        print('[s1] V1／W1 第一張影格未通過 ⇒ 拒跑', flush=True)
        return 73
    print(f"[s1] W1 手腕相機：外參與可見度程式一致 {w1.meta()['vis_T_matches_urdf']}；深度有效範圍 {w1.meta()['depth_range']}（{w1.meta()['depth_range_note']}）", flush=True)
    print(f"[s1] V1 底盤相機：{v1.meta()['T_mount'][0][3]:.4f},{v1.meta()['T_mount'][1][3]:.4f},{v1.meta()['T_mount'][2][3]:.4f}；外參與可見度程式一致 {v1.meta()['vis_T_matches_urdf']}", flush=True)
    q_chk, _ = q9()
    r0 = scan_fn(0.0, q_chk)
    hits = [ray_range(query, [float(q_chk[0]), float(q_chk[1]), 0.2652], [math.cos(v + float(q_chk[2])), math.sin(v + float(q_chk[2])), 0.0], 1.0, excl) for v in NC.scan_angles()[::30]]
    print(f"[s1] 掃描核對：有效 {int(np.isfinite(r0).sum())}／{r0.size}、最近 {np.nanmin(np.where(np.isfinite(r0), r0, np.nan)) if np.isfinite(r0).any() else float('nan'):.3f} m；近距命中 {[h for h in hits if h[1]][:3]}；排除 {excl}", flush=True)
    if np.isfinite(r0).any() and float(np.min(r0[np.isfinite(r0)])) < 0.32:
        print('[s1] 掃描打到自身（<0.32 m）⇒ 拒跑', flush=True)
        return 69
    p_ = [float(v) for v in a.path.split(',')]
    if a.group != 'G':
        print('[s1] N2 只接 G', flush=True)
        return 2
    cls_ = W1.W1PlaceController
    def make_ctrl(i_, q_tuck_, t0_):
        return cls_(C.S0_K(), ident, T_g, T_ready_rel, q_ready, Z1, float(B['radius']), a.x_start - float(RO[0]), (a.win_lo, a.win_hi), C2.DetourPath(*p_),
                           tuple(float(v) for v in a.exit_c.split(',')), a.exit_r, (lambda t_eff: REG.obs_to_region(chain.obstacles(t_eff))), chain=chain, scan_fn=(lambda t_, q_: scan_fn(t_, REG.q_to_world(q_))), place=dict(REG.place_to_region(PCm.PLACE), target=F1_ROUNDS[i_]['target']), x_look=F1_ROUNDS[i_]['x_look'], t_round0=t0_, round_id=i_, x_exit_stop=a.x_end, v_c=a.v_c, v_nav=a.v_nav, v_carry=a.v_carry, x_slow=a.x_slow, q_tuck=q_tuck_, x_unfold=a.x_unfold, t_unfold=a.t_unfold, t_blend=a.t_blend, t_insert=a.t_insert, no_tcp_preview=a.no_tcp_preview, no_inflight=a.no_inflight, body_boxes=[S2] if (S2 and a.s2_active) else None, body_links=tuple(x for x in a.s2_links.split(',') if x), reg=None, required=required, top_z=float(T['top_z']), win=(a.win_lo, a.win_hi), planner_log=os.path.join(a.out, 'plan_worker.log'), **(dict(kp=a.m_kp) if a.group == 'M' else {}))
    ctrl = make_ctrl(0, q_tuck, 0.0)
    if a.group == 'M':
        ctrl.p1.mu_post = a.m_mu
    ctrl.compact = []                                                         # CBF 證據；反事實重解在實跑後（bottle_s1_cbf_replay.py）
    ctrl.obs.d = np.clip(d0, -ctrl.obs.max, ctrl.obs.max)
    ctrl.cfg.arm_model.bias = ctrl.bias0 + np.asarray(ctrl.cfg.arm_model.alpha, float) * ctrl.obs.d
    print(f'[s1] 靜態偏移 d̂₀ {np.round(ctrl.obs.d, 4).tolist()}；底盤 {np.round(q_s[:3], 4).tolist()}', flush=True)
    chain.ctrl = ctrl
    scut['ctrl'] = ctrl
    if not np.isnan(ctrl.task.T_grasp).all():
        print('[s1] V2 建構後抓取目標不是 NaN ⇒ 拒跑', flush=True)
        return 75
    link = VL.V2Link(v1, ctrl)
    wlink = VL.V2Link(w1, ctrl, sink=ctrl.wrist_intake)                       # 手腕：同一位姿閘，送 wrist_intake
    _ci_world = ctrl.candidate_intake

    def _ci_region(t_recv, c, _f=_ci_world):                                 # 候選 xy：世界 → 區域（只轉一次）
        c = dict(c)
        for k_, o_ in (('snap_cx', RO[0]), ('snap_cy', RO[1])):
            if c.get(k_) is not None and np.isfinite(c[k_]):
                c[k_] = float(c[k_]) - float(o_)
        return _f(t_recv, c)
    ctrl.candidate_intake = _ci_region
    slink = SL.SegTargetLink(node, ctrl, Float64MultiArray)                    # 學習模型候選 → candidate_intake（目標唯一來源）

    def hs_state():
        p_, qt_ = robot.get_world_pose()
        jp_ = robot.get_joint_positions()
        return p_, qt_, [float(jp_[i]) for i in idx]

    def hs_hold():
        apply(tgt)
        base_zero()
        world.step(render=True)
    T0, why = RC.handshake(node, chain, hs_hold, hs_state, lambda: scan_fn(0.0, q9()[0]))
    if T0 is None:
        print(f'[s1] 握手未過 ⇒ 拒跑：{json.dumps(why, ensure_ascii=False, default=str)[:1500]}', flush=True)
        return 70
    print(f"[s1] 握手通過 T0 {T0:.2f}；發布者 {chain.graph['publishers']}；TF 對 {len(chain.graph['tf_pairs'])}", flush=True)
    if a.handshake_only:
        node.destroy_node()
        rclpy.shutdown()
        return 0
    import bottle_n2_ros_center as NRC
    conv_cfg = NRC.verify(node, os.path.join(a.out, 'map.yaml'))
    print(f"[s1] 規劃座標換算設定 {conv_cfg}", flush=True)
    pr = BRG.PlanRosRegion(node, conv_cfg, REG)
    ctrl.planner_conv, ctrl.planner_ros = conv_cfg, pr
    goal = tuple(float(v) for v in a.goal.split(','))
    pp, plan_info = BRG.plan_before_start_region(node, chain, pr, hs_hold, hs_state, lambda: scan_fn(0.0, q9()[0]), goal, a.x_lane, T0 + RC.PHYS, REG)
    json.dump(dict(plan=plan_info, center_conv=dict(cfg=conv_cfg, log=pr.conv_log)), open(os.path.join(a.out, 'plan.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    if pp is None:
        print(f'[s1] 無有效路徑 ⇒ 不起跑：{json.dumps(plan_info, ensure_ascii=False, default=float)[:800]}', flush=True)
        node.destroy_node()
        rclpy.shutdown()
        return 72
    print(f"[s1] 規劃 L {pp.L:.3f} m、σ0 {pp.sig0:.3f}、原始 {plan_info['n_raw']} 點、代價峰 {plan_info['cost_max']}、牆鐘 {plan_info['wall_ms']:.0f} ms；y 最低 {pp.P[:, 1].min():.3f}", flush=True)
    if a.plan_only:
        node.destroy_node()
        rclpy.shutdown()
        return 0
    T0 = plan_info['t_ready'] + RC.PHYS
    mon = NP.PathMonitor(pp)
    mon.on_grid(pr.grid, pp.sig0)                                             # 先用規劃時那張代價圖
    mprog = MPG.MonitorProgress(pp, C2.B.Y_TUBE)                              # 導航段監看的獨立進度（不影響控制參考）
    t_now = [0.0]
    cpl = PRS.RosCarryPlanner(pr, lambda: T0 + t_now[0], lambda: ctrl.csig)
    port = FT.PlanPort(cpl)                                                   # 兩輪共用規劃接口的輪次與用途界線（bottle_f1_1 的 KeyError 根因）
    port.bind(ctrl)
    ctrl.carry_planner, ctrl.exit_c, ctrl.bottle_mass = port, np.array([99.0, 99.0]), float(readback['bottle_mass'])
    ctrl.path, ctrl.monitor = pp, mon
    ctrl.s_ref = ctrl.sig_robot = pp.sig0
    pr.monitor, pr.sig_fn = mon, (lambda: mprog.sig)
    slink.set_T0(T0)
    chain.start(T0)
    pub_round = node.create_publisher(Float64MultiArray, '/f1/round', 10)
    q_at_step = [None]

    def make_round(i_, s_now, d_, t_):
        nonlocal mon, mprog
        n_ = make_ctrl(i_, np.asarray(s_now, float).copy(), float(t_))        # 承接實際設定點，不拉回預設姿態；抓取目標仍是 NaN，只能來自視覺
        n_.compact = []
        n_.obs.d = np.clip(d_, -n_.obs.max, n_.obs.max)
        n_.cfg.arm_model.bias = n_.bias0 + np.asarray(n_.cfg.arm_model.alpha, float) * n_.obs.d
        n_.planner_conv, n_.planner_ros = conv_cfg, pr
        port.bind(n_)
        n_.carry_planner, n_.exit_c, n_.bottle_mass = port, np.array([99.0, 99.0]), float(readback['bottle_mass'])
        mon = NP.PathMonitor(pp)
        mon.on_grid(pr.grid, pp.sig0)
        mprog = MPG.MonitorProgress(pp, C2.B.Y_TUBE)
        n_.path, n_.monitor = pp, mon
        n_.s_ref = n_.sig_robot = pp.project(np.asarray(q_at_step[0][:2], float), pp.sig0)
        pr.monitor, pr.sig_fn = mon, (lambda: mprog.sig)
        print(f'[s1] F1 第二輪實例：t {t_:.2f}、σ {n_.s_ref:.3f}、d̂ {np.round(n_.obs.d, 4).tolist()}', flush=True)
        return n_

    def on_switch(n_, t_):
        chain.ctrl = n_
        scut['ctrl'] = n_
        link.ctrl, link.sink = n_, n_.vision_intake
        wlink.ctrl, wlink.sink = n_, n_.wrist_intake
        slink.rebind(n_, n_.candidate_intake)
        reg_ = F1_ROUNDS[1]['region']
        pub_round.publish(Float64MultiArray(data=[1.0, float(reg_[0]), float(reg_[1]), float(reg_[2]), float(reg_[3]), float(T0 + t_)]))
        print(f'[s1] F1 切換 t {t_:.2f}；已發 /f1/round', flush=True)
    def on_checkpoint(kind, r_):
        FX.atomic_json(os.path.join(a.out, f's1_report_{kind}.json'), r_)
        np.savez_compressed(os.path.join(a.out, f's1_steps_{kind}.npz'), rows=np.array(rows, float), cols=np.array(step_cols()))
        print(f'[s1] F1 檢查點 {kind} 已落盤（{len(rows)} 列）', flush=True)
    task = FT.F1Task(ctrl, [float(pp.P[0][0]), float(pp.P[0][1])], make_round, on_switch, on_checkpoint)
    ctrl = task
    dyn_log = []

    def dyn_drive(t_task):
        if CA is None:                                                        # 名目：無移動物，欄位填 NaN
            dyn_log.append([float('nan')] * 9)
            return
        p_t, v_t = DC.state(t_task, CA)
        p_m, _ = dyn.get_world_pose()
        v_m = dyn.get_linear_velocity()
        e = np.array([p_t[0] - float(p_m[0]), p_t[1] - float(p_m[1]), DZ - float(p_m[2])])
        vc = np.array([v_t[0], v_t[1], 0.0]) + np.clip(2.0 * e, -0.2, 0.2)          # 速度前饋＋位置回授（只為跟軌跡，不是控制器資訊）
        dyn.set_linear_velocity(vc.astype(np.float32))
        dyn.set_angular_velocity(np.zeros(3, np.float32))
        dyn_log.append([float(p_m[0]), float(p_m[1]), float(p_m[2]), float(v_m[0]), float(v_m[1]), float(p_t[0]), float(p_t[1]), float(v_t[0]), float(v_t[1])])
    BP_S = G.bottle_points()

    cols_base_idx = 1 + 3 + 6 + 6                                            # rows：t、底盤 3、q 6、手臂目標 6，之後是底盤命令 3

    def step_cols():
        return ['t', 'base_x', 'base_y', 'base_yaw'] + [f'q{i}' for i in range(1, 7)] + [f's{i}' for i in range(1, 7)] + ['base_cmd_vx', 'base_cmd_vy', 'base_cmd_wz', 'v_b', 's_ref', 'finger1_q', 'finger2_q', 'finger_cmd', 'f_finger1_bottle', 'f_finger2_bottle',
                'f_bottle_table', 'bottle_base_x', 'bottle_base_y', 'bottle_base_z', 'bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz', 'phase_index', 'stopped'] + [f'forb_{kk}' for kk in sorted(forb)] + ['dyn_x', 'dyn_y', 'dyn_z', 'dyn_vx', 'dyn_vy', 'dyn_tx', 'dyn_ty', 'dyn_tvx', 'dyn_tvy'] + ['f1_stage', 'other_base_x', 'other_base_y', 'other_base_z', 'other_qw', 'other_qx', 'other_qy', 'other_qz']
    rows, step_ms = [], []                                                    # step_ms：物理＋渲染，分開計時
    t_end_tail = None
    n = int(round((a.sim_s + a.stop_tail_s + 1.5) / dt))                      # 總期限後仍走停止尾段
    penetration = 0.0
    forb, t = {}, 0.0
    try:
      for k in range(n):
          t = k * dt
          task.check_deadline(t, a.sim_s)                                       # 看 F1 整體，不看單輪 mission_done
          q, jp = q9()
          q_at_step[0] = np.asarray(REG.q_to_region(q), float)
          mprog.update(REG.q_to_region(q), t)                                   # 區域座標；先更新監看進度，再處理代價圖回呼（node.drain）與過期核對
          p_, qt_ = robot.get_world_pose()
          node.publish_state(T0 + t, p_, qt_, q[3:])
          chain.set_now(T0 + t)
          node.drain()
          t_now[0] = t
          cpl.tick()
          if ctrl.pph is None:                                                  # 承接後代價圖只餵持瓶段監看器
              mon.check_age(T0 + t, mprog.sig)
          elif ctrl.cmon is not None:
              ctrl.cmon.check_age(T0 + t, ctrl.csig)
          # 瓶位姿（幾何中心 → 底心座標：z 軸＝瓶軸）
          st_ = task.stage
          pb, qb = (bottle if st_ == 0 else bottle2).get_world_pose()
          po_, qo_ = (bottle2 if st_ == 0 else bottle).get_world_pose()         # 另一瓶：只供穿透估計與事後評估
          Ro_ = quat_R(qo_)
          To_ = np.eye(4)
          To_[:3, :3] = Ro_
          To_[:3, 3] = np.asarray(po_, float) - Ro_[:, 2] * float(B['height']) / 2
          Rb = quat_R(qb)
          Tb = np.eye(4)
          Tb[:3, :3] = Rb
          Tb[:3, 3] = np.asarray(pb, float) - Rb[:, 2] * float(B['height']) / 2
          ok_b = bool(np.isfinite(Tb).all())
          v1.record_step(float(world.current_time), T0 + t, t, q, Tb if ok_b else None, BR_PH.index(ctrl.st.phase))   # V1 旁路：先記本步（＝上一次算圖的狀態），再取新影格
          v1.poll()
          link.tick(t)                                                          # V2：估計（位姿互核通過者）→ 控制器；時間為任務時間
          w1.record_step(float(world.current_time), T0 + t, t, q, Tb if ok_b else None, BR_PH.index(ctrl.st.phase))   # W1 手腕：同樣先記本步再取影格
          w1.poll()
          wlink.tick(t)
          slink.tick(t)                                                         # 候選：任務時刻轉送
          # 接觸
          Mf = read_pairs(v_finger, n_filters, dt)
          Ma_l = [read_pairs(v, n_filters, dt) for v in v_arm]
          Ma = None if any(x is None for x in Ma_l) else np.concatenate(Ma_l, axis=0)   # 任一連桿讀不到 ⇒ 整對無效（不當零）
          Me = [read_pairs(v, n_filters, dt) for v in v_extra]
          Mt = read_pairs(FC.pick(st_, v_bt, v_bt_2), 1, dt)
          Ms = read_pairs(FC.pick(st_, v_bs, v_bs_2), len(b_env), dt) if b_env else None
          Mb = read_pairs(v_base, 1 + len(leg_paths) + len(leg2_paths) + (1 if WALL else 0), dt)
          M2 = read_pairs(FC.pick(st_, v_b2, v_b2_2), 1, dt)
          ff, ft, fp, fp_raw, forb, Mf = FC.summarize(st_, Mf, Ma, Me, Mt, Ms, Mb, M2, F1_PERM, F_SIGN, ctrl.judge.t_lift_ok is not None, has_env=bool(b_env))
          # 穿透估計（每控制週期）：非手指連桿點雲對瓶軸與桌面
          if k % C.NPH == 0:
              W = G.link_world(q, ctrl.f_cmd)
              Pl = np.vstack([P for n_, P in W.items() if n_ not in G.C.FINGERS])
              Pb = (Tb[:3, :3].T @ (Pl - Tb[:3, 3]).T).T
              dmin = float(G.cyl_dist(Pb, 0.0, 0.0, 0.0, float(B['height']), float(B['radius'])).min())
              Po_ = (To_[:3, :3].T @ (Pl - To_[:3, 3]).T).T
              dmin = min(dmin, float(G.cyl_dist(Po_, 0.0, 0.0, 0.0, float(B['height']), float(B['radius'])).min()))
              tb = list(T1TB)
              tb = tb + [(n_, c_ + T2D, sz_) for n_, c_, sz_ in T2TB]
              dmin = min(dmin, min(float(G.box_dist(Pl, c, sz).min()) for _, c, sz in tb), float(G.box_dist(Pl, OB_C, OB_S).min()), float(G.box_dist(Pl, S2[0], S2[1]).min()) if S2 else 9.9, float(G.box_dist(Pl, WALL[0], WALL[1]).min()) if WALL else 9.9)
              if ok_b:                                                          # 瓶身對第二桌面與桌腳的穿透（瓶底支撐的正常接觸遠小於 1 mm 判準）
                  Pb_w2 = (Tb[:3, :3] @ BP_S.T).T + Tb[:3, 3]
                  dmin = min(dmin, min(float(G.box_dist(Pb_w2, c_ + T2D, sz_).min()) for _, c_, sz_ in T2TB))
              if (S2 or WALL) and ok_b:                                         # 實測瓶位姿對 S2 方塊與側牆
                  Pb_w = (Tb[:3, :3] @ BP_S.T).T + Tb[:3, 3]
                  for box_ in [b_ for b_ in (S2, WALL) if b_ is not None]:
                      dmin = min(dmin, float(G.box_dist(Pb_w, box_[0], box_[1]).min()))
              elif S2 or WALL:
                  dmin = float('nan')                                           # 瓶位姿無效：不當零，由資料失效判定處理
              penetration = max(0.0, -dmin) if np.isfinite(dmin) else float('nan')
          out = ctrl.step(t, REG.q_to_region(q), f_fingers=ff, t_contact=(t if ff is not None else None), T_bottle=REG.T_to_region(Tb) if ok_b else None, t_bottle=t if ok_b else None,
                          f_table=ft, forbidden=forb, penetration=penetration, f_place=fp, q_fingers=(float(jp[f1]), float(jp[f2])), f_place_raw=fp_raw, f_finger_vec=None if Mf is None else np.asarray(Mf[:, 0], float))
          tgt = np.array(jp, float)
          for j, i in enumerate(idx):
              tgt[i] = float(out['arm_target'][j])
          tgt[f1] = out['finger_target']
          apply(tgt)
          base_cmd(out['base_twist'], float(q[2]))
          dyn_drive(t)
          t_w = time.perf_counter()
          world.step(render=True)
          step_ms.append((time.perf_counter() - t_w) * 1e3)
          rows.append([t, *q, *out['arm_target'], *out['base_twist'], ctrl.v_b, ctrl.s_ref, float(jp[f1]), float(jp[f2]), out['finger_target'], *(ff if ff else (float('nan'),) * 2),
                       float('nan') if ft is None else ft, *Tb[:3, 3], *[float(v) for v in qb], BR_PH.index(out['phase']), float(out['stopped']),
                       *[forb[kk] for kk in sorted(forb)], *dyn_log[-1], float(st_), *To_[:3, 3], *[float(v) for v in qo_]])
          if (out['stopped'] or out['done']) and t_end_tail is None:
              t_end_tail = t + (a.stop_tail_s if out['done'] else max(a.stop_tail_s, 2.6))   # 停穩判定 1.0–2.0 s 窗，尾段 ≥ 2.5 s  # 停止尾段須涵蓋 1 s 停止有效性
          if t_end_tail is not None and t >= t_end_tail:
              break
    except Exception as ex_:                                                  # 例外：先存例外與狀態，再獨立收尾，最後盡力存資料；原例外不被掩蓋（Codex 20261103）
        ph_ = FX_ph = None
        try:
            ph_, FX_ph = task.st.phase, task.pph
        except Exception:
            pass
        crash = FX.record(a.out, ex_, dict(t=float(t), stage=int(task.stage), phase=ph_, pph=FX_ph, outcome=task.outcome, n_rows=len(rows)))
        print(f"[s1] 例外（已落盤 crash.json）：{crash['type']}: {crash['msg']}；t {t:.2f} 輪次 {task.stage} 相位 {ph_}／{FX_ph}", flush=True)
        u_last_ = np.array(rows[-1][cols_base_idx:cols_base_idx + 3], float) if rows else np.zeros(3)
        tgt_hold_ = np.array(robot.get_joint_positions(), float)

        def stop_step_(u_):
            apply(tgt_hold_)
            q_, _ = q9()
            base_cmd(u_, float(q_[2]))
            world.step(render=True)
        crash['stop'] = FX.guarded(crash, 'independent_stop', lambda: FX.independent_stop(stop_step_, lambda: q9()[0], u_last_, C2.B.A_BASE, dt))
        print(f"[s1] 獨立收尾 {crash['stop']}", flush=True)
        FX.guarded(crash, 'rows', lambda: np.savez_compressed(os.path.join(a.out, 's1_steps.npz'), rows=np.array(rows, float), cols=np.array(step_cols())))
        FX.guarded(crash, 'report', lambda: FX.atomic_json(os.path.join(a.out, 's1_report_crash.json'), task.c.report()))
        FX.guarded(crash, 'reports_task', lambda: FX.atomic_json(os.path.join(a.out, 's1_report_rounds_crash.json'), dict(outcome=task.outcome, stage=task.stage, t_switch=task.t_switch, t_done0=task.t_done0, switch=task.switch_info, rounds=task.reports)))
        FX.atomic_json(os.path.join(a.out, 'crash.json'), crash)
        print(f"[s1] 例外收尾完成：停止 {crash['stop']}；存檔錯誤 {crash['save_errors']}", flush=True)
        node.destroy_node()
        rclpy.shutdown()
        return 78
    f1_rep = task.report()
    rep = task.c.report()
    rep['f1_task'] = {k: f1_rep[k] for k in ('outcome', 'stage', 't_switch', 't_done0', 'switch', 'grip_ok')}
    if task.stage == 1:                                                       # 第一輪完整紀錄另存；主紀錄 s1_report.json 是最後作用的實例
        try:
            json.dump(f1_rep['rounds'][0], open(os.path.join(a.out, 's1_report_r0.json'), 'w'), ensure_ascii=False, indent=1, default=float)
            C2.save_compact(task.prev, os.path.join(a.out, 's1_cbf_r0.npz'))
            np.savez_compressed(os.path.join(a.out, 'place_log_r0.npz'), rows=np.array(task.prev.plog, float).reshape(-1, len(PCm.PLOG_COLS)), cols=np.array(PCm.PLOG_COLS), fail_bits=np.array(json.dumps(PCm.FAIL_BITS)))
        except Exception as e:
            print(f'[s1] 第一輪紀錄存檔失敗：{e!r}', flush=True)
    rep['timing']['world_step'] = dict(p50=round(float(np.percentile(step_ms, 50)), 2), p95=round(float(np.percentile(step_ms, 95)), 2), max=round(float(np.max(step_ms)), 2), n=len(step_ms))
    rep['region'] = dict(offset=RO[:2].tolist(), frame='控制器與 rep 內事件位置為區域座標；s1_steps 的 base_x／bottle_* 為世界座標')
    rep.update(plan=plan_info, readback=readback, asset=S, forb_keys=sorted(forb), d0=[float(v) for v in d0], n_steps=len(rows), obstacle=dict(c=OB_C.tolist(), sz=OB_S.tolist(), r_cbf=0.2 * math.sqrt(2.0), margin=0.47, truth_for_eval_only=True), exit=dict(c=a.exit_c, r=a.exit_r), path=p_, note='W1：底盤遠距凍結＋手腕近距修正（只修 xy、限速 5 mm/s、INSERT＋0.9 s 鎖定、閉爪前須到達）；V2：抓取目標由底盤相機估計凍結（建構時 NaN），只收位姿互核通過的影格；手臂備妥分支用任務名目瓶位；5 mm 淨空與禁止接觸用模擬真值瓶位姿（安全網）。核心 gmpc_wb；PhysX 射線掃描經真 ROS 話題：/scan_raw→scan_relay→/scan→追蹤器與 Shield 獨立節點，障礙走 /gmpc/obstacles_stamped，底盤命令走 /cmd_vel_pre_shield→Shield→/cmd_vel；控制器與物理同步（非即時部署驗證）；底盤與瓶位姿真值；地圖只有桌腳與側牆；參考路徑：導航段起跑前 SmacPlanner2D 規劃到抓取直線入口；抓取直線為任務定義；持瓶段 LIFT 開始時再規劃到放置直線入口（平滑、承接、局部降速、參考加速度核對）；放置直線與停車放置為任務定義；不重規劃。第二桌位置為已知位置，瓶位姿真值只供安全網與評估；命令總延遲 8 物理步＝實測往返＋注入補足（分列於 nav_chain.ros）；停止處置為腳本自有', ros_chain=True)
    cols = step_cols()
    if CA is None:
        rep['dyn'] = dict(case=None, note='無移動物（--dyn-case none）')
    else:
        DL = np.array(dyn_log, float)
        e_pos = np.hypot(DL[:, 0] - DL[:, 5], DL[:, 1] - DL[:, 6])
        e_vel = np.hypot(DL[:, 3] - DL[:, 7], DL[:, 4] - DL[:, 8])
        R_ = np.array(rows, float)
        dsurf = np.hypot(R_[:, 1] - DL[:, 0], R_[:, 2] - DL[:, 1]) - CA['radius']
        rep['dyn'] = dict(case=CA, n=len(DL), track_pos_err_max_m=float(e_pos.max()), track_vel_err_max_mps=float(e_vel.max()), z_range=[float(DL[:, 2].min()), float(DL[:, 2].max())],
                          v_meas_max=float(np.hypot(DL[:, 3], DL[:, 4]).max()), base_center_to_surface_min_m=float(dsurf.min()), at_t=float(R_[int(np.argmin(dsurf)), 0]), nopred=bool(a.nopred),
                          note='障礙位姿為讀回值；只供驅動與評估，不進控制器')
        print(f"[s1] 移動物 {json.dumps({k: v for k, v in rep['dyn'].items() if k != 'case'}, ensure_ascii=False, default=float)}", flush=True)
    if bottle2 is not None:
        p2_, q2_ = bottle2.get_world_pose()
        rep['bottle2'] = dict(start=[B2X, B2Y, Z1 + B['height'] / 2], end=[float(v) for v in p2_], quat_end=[float(v) for v in q2_], moved_xy_m=float(math.hypot(float(p2_[0]) - B2X, float(p2_[1]) - B2Y)), note='讀回值；只供事後評估')
    np.savez_compressed(os.path.join(a.out, 's1_steps.npz'), rows=np.array(rows, float), cols=np.array(cols), phases=np.array(BR_PH))
    try:                                                                      # 紀錄補強（Codex 20261101_132514）：實際掃描與命令時間鏈，可直接重放
        np.savez_compressed(os.path.join(a.out, 'scan_log.npz'), t=np.array([s_[0] for s_ in scan_log], float), stamp=np.array([s_[0] + (chain.T0 or 0.0) for s_ in scan_log], float),
                            pose=np.array([s_[1] for s_ in scan_log], float), ranges=np.array([s_[2] for s_ in scan_log], float), frame=np.array('lidar_link'),
                            angle_min=NC.SCAN_MIN, angle_inc=NC.SCAN_INC, range_min=NC.RANGE_MIN, range_max=NC.RANGE_MAX, T0=float(chain.T0 or 0.0))
        S_ = [e for e in chain.sent if e['k'] is not None]
        np.savez_compressed(os.path.join(a.out, 'cmd_chain.npz'), seq=np.array([e['seq'] for e in S_]), k=np.array([e['k'] for e in S_]), t_send=np.array([e['t'] for e in S_], float),
                            t_rep=np.array([np.nan if e['t_rep'] is None else e['t_rep'] - chain.T0 for e in S_], float), t_effect=np.array([e['t'] + RC.D_STEPS * RC.PHYS for e in S_], float),
                            u=np.array([e['u'] for e in S_], float), out=np.array([np.full(3, np.nan) if e['out'] is None else e['out'] for e in S_], float),
                            chk=np.array([np.nan if e['chk'] is None else float(e['chk']) for e in S_], float),
                            diag=np.array([np.r_[r_[0] - chain.T0, r_[1:]] for r_ in diag_log], float) if diag_log else np.zeros((0, 1)),
                            note=np.array('diag 列＝[收到時任務時刻, /shield/diag 全欄]；t_effect＝送出＋D_STEPS·PHYS（控制器佇列生效步）'))
        L_ = max((len(v) for _, v in trk_log), default=0)
        np.savez_compressed(os.path.join(a.out, 'tracker_log.npz'), t_recv=np.array([r_ - (chain.T0 or 0.0) for r_, _ in trk_log], float),
                            data=np.array([v + [np.nan] * (L_ - len(v)) for _, v in trk_log], float), note=np.array('每列＝[序號, 掃描戳(ROS), 動態值數, 靜態值數, 動態 5 值×n, 靜態 5 值×m]，NaN 補齊；t_recv＝收到時任務時刻'))
        rep['records'] = dict(n_scan=len(scan_log), n_cmd=len(S_), n_diag=len(diag_log), n_tracker=len(trk_log))
    except Exception as e:
        rep['records'] = dict(error=repr(e))
        print(f'[s1] 紀錄補強存檔失敗 {e!r}', flush=True)
    rep['cols'] = cols
    try:                                                                      # 逐物理步放置紀錄（與 s1_steps 同一個 t）
        PL = np.array(ctrl.plog, float).reshape(-1, len(PCm.PLOG_COLS))
        np.savez_compressed(os.path.join(a.out, 'place_log.npz'), rows=PL, cols=np.array(PCm.PLOG_COLS), fail_bits=np.array(json.dumps(PCm.FAIL_BITS)))
        rep['place_log'] = dict(n=int(PL.shape[0]), cols=len(PCm.PLOG_COLS))
    except Exception as e:
        print(f'[s1] 放置紀錄存檔失敗：{e!r}', flush=True)
        rep['place_log'] = dict(error=repr(e))
    rep['reached_end'] = bool(ctrl.mission_done or ctrl.judge.stop is not None)
    rep['monitor_progress'] = mprog.report()
    try:
        rep['w1_cap'] = w1.save(a.out)                                        # 手腕擷取紀錄；rep['w1'] 是控制器的手腕修正紀錄，不可覆寫
        rep['w1_cap']['first_frame'] = w1_first
        print(f"[s1] W1 擷取 { {k: v for k, v in rep['w1_cap'].items() if k != 'meta'} }", flush=True)
    except Exception as e:
        print(f'[s1] W1 存檔失敗：{e!r}', flush=True)
        rep['w1_cap'] = dict(error=repr(e), collect_ok=False)
    rep['v2_link'] = link.report()
    rep['w1_link'] = wlink.report()
    rep['segtarget_link'] = slink.report()
    rep['table1_z'] = Z1
    print(f"[s1] v2 相對排程 {rep.get('relwin')}", flush=True)
    print(f"[s1] v2w 滑移警告 {rep.get('slip_warn')}", flush=True)
    print(f"[s1] v3 跨桌高 {rep.get('xh')}", flush=True)
    print(f"[s1] 在線計畫 {rep.get('online')}", flush=True)
    _cg = rep.get('carry_gov') or {}
    print(f"[s1] 持守 events={_cg.get('events')} hold_s={_cg.get('hold_s')} min_stop_margin={_cg.get('min_stop_margin')} min_hold_minus_brake={_cg.get('min_hold_minus_brake')} park_tol_n={len(_cg.get('park_tol_log') or [])}", flush=True)
    _gz = rep.get('gaze') or {}
    print(f"[s1] 視向 on={_gz.get('on')} max={_gz.get('max')} ev={_gz.get('ev')} counts={_gz.get('counts')} state={_gz.get('state')}", flush=True)
    print(f"[s1] V_HIGH 序列 {(rep.get('vhigh_seq') or {}).get('log')}；等候 {(rep.get('vhigh_seq') or {}).get('waits')}", flush=True)
    print(f"[s1] 候選轉接 {rep['segtarget_link']}", flush=True)
    rep['scan_cut'] = dict(frac=a.scan_cut_front, t_cut=scut['t_cut'], last_published=scut['last'], n_drop=scut['n_drop'])
    print(f"[s1] 掃描中斷注入 {rep['scan_cut']}", flush=True)
    if rep['w1'].get('goal') is not None:
        rep['w1']['goal_err_vs_truth_mm'] = [round((rep['w1']['goal'][0] - (BX - RO[0])) * 1e3, 3), round((rep['w1']['goal'][1] - (BY - RO[1])) * 1e3, 3)]   # 評估用
    print(f"[s1] W1 控制 鎖定 {rep['w1'].get('locked')} 目標 {rep['w1'].get('goal')} 誤差（評估）{rep['w1'].get('goal_err_vs_truth_mm')} mm 修正 {rep['w1'].get('corr_mm')} mm 組 {rep['w1'].get('n_groups')} 拒 {rep['w1'].get('rejected')}；轉送 {rep['w1_link']}", flush=True)
    if rep['v2'].get('fix') is not None:
        fx = rep['v2']['fix']
        rep['v2']['fix_err_vs_truth_mm'] = [round((fx['xy'][0] - (BX - RO[0])) * 1e3, 3), round((fx['xy'][1] - (BY - RO[1])) * 1e3, 3)]   # 評估用：凍結值對生成位置
    print(f"[s1] V2 凍結 {rep['v2']['fix']} 誤差（評估）{rep['v2'].get('fix_err_vs_truth_mm')} mm；收 {rep['v2']['n_intake']} 拒 {rep['v2']['rejected']}；轉送 {rep['v2_link']}", flush=True)
    try:
        rep['v1'] = v1.save(a.out)
        rep['v1']['first_frame'] = v1_first
        print(f"[s1] V1 { {k: v for k, v in rep['v1'].items() if k != 'meta'} }", flush=True)
    except Exception as e:
        print(f'[s1] V1 存檔失敗：{e!r}', flush=True)
        rep['v1'] = dict(error=repr(e), collect_ok=False)
    if not RC.write_report(rep, a.out):                                       # bottle_n1_1 寫到一半崩潰；失敗另存診斷檔、不算完成
        node.destroy_node()
        rclpy.shutdown()
        return 71
    try:                                                                      # 主紀錄存好後才存 CBF 證據：這裡失敗不影響主紀錄
        C2.save_compact(task.c, os.path.join(a.out, 's1_cbf.npz'))
    except Exception as e:
        print(f'[s1] CBF 紀錄存檔失敗：{e!r}', flush=True)
    print(f"[s1] 結果 success={rep['success']} 判定 {rep['result']} 第一阻點 {rep['stop']} 停止有效 {rep['stop_effect']} 出口 {rep['exit_reached']}@{rep['t_exit_in']} 任務 {rep['mission_done']} 瓶淨空 {rep['clear_bottle']} 滑移 {rep['slip_max_mm_deg']} 求解 {rep['solve_ms']} fallback {rep['n_fallback']} 求解器 {rep['qp_counts']} GI 未採用 {rep['gi_fail']} >50ms {rep['n_over_50ms']} 掃描鏈 { {k: rep['nav_chain'][k] for k in ('n_scan', 'n_dyn_max', 'n_static_max', 'shield')} } 分項 {rep['timing']}", flush=True)
    print(f"[s1] 事件 {rep['events']}", flush=True)
    print(f"[s1] ROS { {k: v for k, v in rep['nav_chain']['ros'].items() if k not in ('graph', 'note', 'shield_age_top_s')} }", flush=True)
    node.destroy_node()
    rclpy.shutdown()
    rc_task = C2.exit_code(rep)
    sw = rep.get('slip_warn') or {}
    if rep.get('mission_done') and rep.get('stop') is None and rep.get('success') is not True and sw.get('warn') is not None:
        rc_task = 6                                       # v2w：完成抓放但原抓持品質不合格（平移滑移超標警告）；不是停止，也不是原標準成功
        print(f"[s1] v2w 分類：完成抓放、伴隨滑移超標（原標準不成功）⇒ 結果碼 6；警告 {sw.get('warn')}", flush=True)
    if rc_task == 0 and f1_rep['outcome'] == 'complete' and any(g_ is False for g_ in f1_rep['grip_ok']):
        rc_task = 6                                       # F1：兩輪流程完成，但第一輪原抓持品質不合格；與流程完成分開
    if f1_rep['outcome'] != 'complete' and rc_task in (0, 6):
        rc_task = 4                                       # 流程未完成不得回成功碼
    print(f"[s1] F1 結果 outcome={f1_rep['outcome']} stage={f1_rep['stage']} t_switch={f1_rep['t_switch']} grip_ok={f1_rep['grip_ok']}", flush=True)
    if not (rep.get('v1') or {}).get('collect_ok') or not (rep.get('w1_cap') or {}).get('collect_ok'):   # 採集與任務分開：任一台採集不成立就不能回 0
        print(f"[s1] V1／W1 採集不成立（任務碼 {rc_task}；V1 {(rep.get('v1') or {}).get('collect_ok')}、W1 {(rep.get('w1_cap') or {}).get('collect_ok')}）⇒ 回 74", flush=True)
        return 74
    return rc_task


BR_PH = ['CRUISE', 'PREGRASP', 'INSERT', 'CLOSE', 'LIFT', 'CARRY']

if __name__ == '__main__':
    rc = 1
    try:
        rc = main()
    except BaseException:
        import traceback
        print('[s1] 例外：' + traceback.format_exc().replace('\n', ' | '), flush=True)
    finally:
        # sim_app.close() 會直接結束程序、sys.exit 到不了（bottle_s0a_1 拒跑卻回 0）⇒ 先寫離開碼檔，由跑法讀回
        with open(os.path.join(a.out, 'exit_rc.txt'), 'w') as fh:
            fh.write(str(rc))
        sim_app.close()
    sys.exit(rc)
