# 門→抽屜同趟串接 S1（設計 r1，2026-10-30；Codex m5a 核可 S1 修正版）

目標：同一 Isaac 世界、同一座標，連續跑「門段（m4r_16 定版）→ 交回導航 → 櫃前接近 → 抽屜開關」。返程門與回起點不在本階段。

## 座標
- 世界＝門座標（＝mission_room.sdf 原座標）：鉸鏈在原點；抽屜櫃 (2.4, 5.05)。
- 抽屜管線常數平移 (+2.4, +3.6)（抽屜房 → 門座標，純平移、無旋轉；抽屜軸仍為世界 y）：
  停位 (−0.136412, 0.56, 1.297349) → (2.263588, 4.16, 1.297349)；起點＝門段結束實際位姿（m4r_16：(0.516, 0.530, −0.993)）。
- 牆：mission_room 隔牆 x∈[−4.0,−1.5]∪[2.5,5.0]（y 中心 0.06、厚 0.12），door_asset_v3 牆 x∈[−1.5,2.5] ⇒ 不重疊、不補牆。
- 地圖：mission_room.yaml（門洞空；門扇由掃描看見）。門段原用 door_scene_v3.yaml——**門段地圖改變須揭露**（門段計畫為直線／折線，不依地圖）。

## 執行端 ev9r8t（ev9r8s 版本化複本）
1. `--with-drawer`：門建好後照 isaac_drawer_room_sim 同段建抽屜櫃（drawer_asset 另名匯入）、驅動讀回為零、把手橫桿夾持摩擦、摩擦核對（抽屜子樹另核一次）、DY0、手指↔抽屜接觸視圖；
   主迴圈發 /drawer/state、/drawer/handle_pose、/drawer/sync_state（格式與 drawer_room_sim 逐欄相同，28 欄）。門話題 /door/* 不變。
2. 階段閘 `phase ∈ {door, handoff, drawer}`：
   - 收 `/mission/subtask_done`（TRANSIENT_LOCAL，含 run_id、subtask='door'、門任務 DONE 時刻）＝**交接請求**，不是立即換入口。
   - 准入：門任務已 DONE、權威無中止、整機包絡（EnvelopeMonitor）准入、導航備妥（nav_ready）⇒ 走既有 WB→NAV（DualSourceExecutorR2.request_nav，同 R1_HO 路徑：任期封存、承接最後套用值）。未完成前原控制者續管。
   - 切換完成（owner＝nav）⇒ phase＝drawer：W2 權威結束（不再 step_begin、逾時監督停）、換抓核心脫離、門 K3-M／停位閘停用；一般入口（/wb_vel_cmd）放行；
     **保留**：中止閂鎖、命令鏈通用限制、包絡監看、導航 CBF／護盾（節點端）。
   - drawer 階段 PARK_HOLD 錨點改 `--drawer-park-pose`（抽屜停位）；之後抽屜 NAV→WB 交棒走既有預核（承接導航最後套用值）。
   - 架構不永久禁止重入（返程門以新階段／任期接手，本輪不實作）。
3. 門段成功收尾不送 /room/stop；整體中止仍可 /room/stop。

## 任務端 node10o（node10n 複本）
- `--subtask-mode`：DONE 時發 /mission/subtask_done（不發 /room/stop），封存 task.json 後正常結束。

## 發射檔 run_mission_door_drawer.sh
- 世界 mission_room.sdf＋門＋抽屜；地圖 mission_room.yaml；導航節點全程共用。
- 門段節點（task／solver ADM23／adapter k3m4／door wholebody／nav1／lw_observer）在 subtask_done＋phase＝drawer 後依序受控結束，確認全數結束（話題發布者快照）後才起抽屜段節點
  （drawer_glide／drawer_wholebody／drawer_task／arm_vel_adapter〔consumer /door_k2_sim〕／wgmpc_wg2），參數平移到門座標、--start＝實際位姿。
- 隔離：依序生命週期（門段發布者全結束才起抽屜段）＋執行端階段閘；鎖存訊息隨發布者結束消失。

## 須揭露
- 物理配置沿用門段（主從夾爪、手指碰撞等），與抽屜單獨跑的配置不同之處逐項列。
- 門段地圖由 door_scene_v3 改 mission_room。
- 本階段交付＝連續到抽屜開關完成；返程門、回起點未完成。
