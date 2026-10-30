# 第三階段（返程推門＋回程關門）接線設計 r1（2026-10-30）

範圍：Howard 定案：去程不變（dd_7）；回程推開、穿過、再關門，回原始起點。離線基準：door_return_rev_v2c（A）、door_return_close_v3b（B）。

## 堆疊
沿用抽屜段（同一 Isaac 迴圈、不重設）：plain /wb_vel_cmd、wgmpc_wg2、arm_vel_adapter、GMPC＋CBF、護盾、R2 包絡。不重啟門段 W2／ADM23（模式機不倒放）。
抽屜段迴圈擴充（door_drawer_phase6／ev9r8y）：另發 /door/state、/door/sync_state（門角、門角速度、內外把手位姿、底盤、手臂）；手指接觸改依階段對門扇（既有 finger_v）或抽屜（finger_v_d），/gripper/state 同格式；通用閂鎖、包絡、抓持期限出口保留（抓持期限於門接觸重新起算一次）。

## 進場重用
兩次 NAV→WB 不寫新交棒：沿用 drawer_mission＋drawer_glide＋drawer_wholebody（移動中交棒、邊走邊展臂），停位＝S2 起點／C2 起點底盤，展臂目標＝該起點手臂，--stow-q STOW_F。R0＋S1、C0＋C1 因此由既有機制實現（C1 候選的移動轉彎作為進場路徑與朝向依據；實際軌跡由既有節點決定，須揭露與離線候選的差異）。導航末段與進場方向共線。門角與底盤起點一律取當趟實測（Δ 補償），不寫死 dd_7。

## 操作段
新任務節點 door_return_task_node（單一節點、順向新序號，不用 W2 模式表）：逐段照參考 20 Hz 發 /drawer/tcp_target（FK(參考)）＋/wgmpc/coord（底盤前饋、手臂名目），段間門檻：
- A：S2 滑入→位姿／無非預期接觸准入→S3 閉爪→兩指連續 2 s 證據→S4 推門（門角追蹤容差 2°）→位置差分停穩准入→S5 張爪→解除確認→S6→S7→S8→A 側清空＋包絡＋導航備妥→WB→NAV（request_nav）。
- B：進場→C2→C3 閉爪＋證據→C4 推關至近關閉→位置差分停穩准入→C5 張爪→解除確認→C6→C7→WB→NAV→GMPC 回 (−0.6,−2.8)。
- 監看：GeometryMonitor＋SwitchAwareGeom＋登錄 ds_err；接觸、腕力、門速 0.05、抓持漂移 5 mm／2° 既有門檻；任一中止即受控停止，不自動重試。

## 不做
不改力量／門檻；不宣稱上閂；不倒放 W2；去程 R1 先維持 dd_7（R1 v3 候選另列，58 s、0.42 m、不倒退，待返程後接）。

## 交付
door_return_ref_v1.json（A＋B 20 Hz 列）→ phase6／ev9r8y → door_return_task_node → run_mission_roundtrip.sh → 冒煙（Isaac 場景、節點、真執行層）→ 整包申請一趟。
