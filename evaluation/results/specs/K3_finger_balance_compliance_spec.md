# K3-A-B：兩指承載差驅動的有界法向位移柔順（小規格 draft-3，實作 r2）

實作 r2（Codex 20261023_194140 四項必修；增益、位移上限、模型與安全門檻均不改）：
1. 判定器在替換基礎目標**之前**獨立驗證原始發布證據（應發布列的 target 缺失、形狀錯誤或非有限 ⇒ published_target_invalid）。
2. 時間契約：首個積分步也核對 sim_t − t0 ＝ dt_p（t0＝OPEN 第一輪狀態時間；執行端契約：狀態步 s 的 sim_t＝配對步 s 的 sim_t）；消費配對的 sim_t 不得晚於本輪狀態時間（future_time）；牆鐘新鮮度只在來源步號確實向前進展且內容有效時刷新（重送、遲到丟棄、無效內容不刷新）；判定器同步核對 t0、首步與相鄰步時間、每列最後消費時間 ≤ 該列狀態時間。
3. SETTLE→END_HOLD **轉換輪**即不消費、發布上一筆實際已發布的完整目標；判定器把轉換列（ph0＝SETTLE、phase＝END_HOLD）納入凍結核對。
4. 判定器對逐步紀錄 e、ef、u、v、delta、sat 全部型別核對並重算；room 來源缺手指或 'bar' 欄位不以零代替（pair_source_field_missing）；報告的 Δ／d̂ 對齊改用 task trace 實際發布列（delta_published_m），來源時間回填另名 delta_by_pair_source_time_m。
凍結改為 `freeze_door_k3ab_ev_dev_r2.sha256`（跑法 FRZ 同步指向；r1 清單保留為紀錄）。

draft-3（Codex 20261023_192620：draft-2 可進入版本化實作與離線測試，**尚未核可實跑**）：SETTLE 繼續調整、END_HOLD 凍結完整已發布目標、DONE 照原策略停止發布；量測消費與積分時序明確化（§2、§4）；判定器獨立重算（§7）；雙向幾何核對結果（§3）；§1、§3、§5 措辭補正。

draft-2（Codex 20261022_133343 三項）：開門首筆目標連續、END_HOLD 固定目標不動、新判定器分開核對基礎目標與柔順位移（§4、§5、§7 修訂）。

依 Codex 20261022_133329：在模型診斷封存後，先送審一個**有界承載平衡候選**小規格；**不先實作、不開跑**。
定位：檢驗「OPEN 期間依兩指承載差，作受限且持續的夾合軸目標微調」能否維持雙指承載。
它的必要性只是**候選假說**，不代表已確認負載或順應是根因；本規格也不是 K3-A 任務通過的條件。

## 1. 背景事實（既有封存資料，描述性）
三趟取證資料（ev_1、au_ev_1、ax_ev_1），窗口取 OPEN 起到任一指首次 < 0.5 N 之前，逐物理步：

| 量 | ev_1 | au_ev_1 | ax_ev_1 |
|---|---|---|---|
| F1＋F2（room leaf_contacts.bar，N） | 9.93–10.06 | 9.93–10.08 | 9.33–10.11 |
| 手指關節實測和力（finger_joint1＋2，evidence dof_measured_efforts） | −10.00～−9.95 | −10.00～−9.95 | −10.00～−9.89 |
| 手指關節位置 q_f1／q_f2（mm） | 1.654–1.673／1.675–1.699 | 1.654–1.674／1.675–1.699 | 1.659–1.675／1.678–1.697 |
| F2 − F1（N） | −4.55 → +8.89 | −4.77 → +8.79 | −4.71 → +8.98 |

- 接觸重心在把手座標中：finger1 z ≈ +13.0 mm（橫桿上方），finger2 z ≈ −12.9 mm（下方），桿半徑約 13 mm（`door_k3a_open_bins_r2.json`）。
- **描述**：夾持期間，**兩指法向力幅值之和**約 10 N，手指位置幾乎不動。在和約 10 N 時，「finger1 < 0.5 N」等價於「F2 − F1 > 約 9 N」。
- 不推因果。幅值和為何約 10 N（驅動上限或 mimic 耦合），也不在本規格內判定。

## 2. 量測
- **只用同一來源物理步的兩指橫桿法向力幅值** (F1_k, F2_k)，即 room leaf_contacts 中該步的 finger1／finger2 'bar' 值。
  - 誤差 e_k ＝ F2_k − F1_k。
  - 不用 /door/contacts 的窗口最小／最大值相減（不同時刻）。
  - 不用作用側未確認的切向量，也不用腕力。
- **新增量測流**：執行器另建版本副本 `isaac_door_k2_sim_ev2.py`（只插入，不改原檔），新增 `/door/finger_pair`。
  - 每 50 Hz 一筆 JSON，列出窗口內**每一物理步**的 {step, sim_t, f1, f2, valid}。
  - 與 room 紀錄同一步、同一來源，發布前不做任何聚合。
- **入口驗證**（`door_k3ab_balance.PairBuffer`，原始值轉型前先檢查）：訊息須為 dict 且 pairs 為 list；每筆 5 元素，step 為有限整數值、sim_t 有限、
  valid 為 bool、f1／f2 為有限數或 None。不合格 ⇒ 緩衝 fault＝malformed。
- **重送**：已消費或已待處理的步號再收到 ⇒ 值完全相同則忽略（重送不重積分），不同則 fault＝conflict。
- **消費時序**（Codex 20261023 第 2 項）：
  - 每個來源步只消費一次，依步號順序逐步更新濾波與積分（同批多步各用一個物理步長 dt_p＝0.01 s，不套任務輪 Δt）；
  - 只取步號 ≤ 本輪狀態步號（/door/sync_state 的 step）的配對；較晚的留在緩衝，不提前使用、不丟棄；
  - 首個可更新步＝s0＋1，s0＝OPEN 第一輪的狀態步號；≤ s0 的配對標為已消費但不積分（之後遲到的 ≤ s0 筆只計數）。
- **停止**（立即停止柔順更新並以 `balance_meas_invalid` 中止；Δ 凍結；**不恢復、不跨缺口補積分**；原安全監看全部保留）：
  - 步號不等於上一步＋1（缺步）、sim_t 差不等於 dt_p（容差 1e-4 s）、valid＝False、f1／f2 為 None 或 < 0；
  - 緩衝 fault（malformed／conflict）；
  - 模擬時間：最後消費配對的 sim_t 比本輪狀態 sim_t 落後 > 0.10 s；
  - **牆鐘**：配對流最後一次收到有效內容的牆鐘年齡 > stale_s（0.2 s，與既有監看相同）——模擬時鐘停止時也會中止，不會無限等待。

## 3. 方向（由實際手指幾何核對，不寫死世界 z）
- URDF（`models/omni_bot_manip.urdf`）：
  - finger_joint1 為 prismatic，axis (0, +1, 0)；finger_joint2 axis (0, −1, 0)，皆在 uflite_gripper_link。
  - 張開為正值（0 → 0.0089 m）⇒ finger1 位於夾爪座標 +y 側、finger2 位於 −y 側。
  - joint_tcp 只有 z 平移 0.0836、rpy 0 ⇒ TCP 的 y 軸與夾爪 y 軸同向。
- 夾合軸單位向量（finger1 → finger2）：e₁₂ᵀ ＝ (0, −1, 0)，以 TCP 座標表示。
- 與抓取姿態交叉核對：R_grasp 的 tool z ＝ handle +y、tool x ＝ handle −x ⇒ tool y ＝ handle +z。所以 finger1 在 +z_h，與接觸重心 +13.0 mm 一致。
- **回授正負號（依幾何提出的候選方向，不是已證實的力—位移關係）**：
  - 依剛性夾持的幾何推想：F2 > F1 對應下方 finger2 承載較多；候選修正方向是把目標往 e₁₂（finger2 側）移，Δ 取正值使目標沿 e₁₂ 平移。
  - 實際的力—位移關係未辨識；本規格只登錄方向與界限，不宣稱此方向必然降低承載差。
- **開跑前的離線幾何核對**（`door_k3ab_offline_geom.py` → `results/door/door_k3ab_offline_geom.json`；點雲／盒／膠囊近似，只在取樣點核對）：
  1. HOLD 實測姿態下兩指點雲重心的把手 z：finger1 +11.4 mm、finger2 −11.2 mm ⇒ finger1 在上方，e₁₂ 指向 finger2 側。
  2. 取樣：門角 −1, 0, 1, 2, 4, 6, 8, 9°（登錄 OPEN 範圍 −1…9°）× Δ ∈ {−1, −0.5, 0, +0.5, +1} mm，沿 e₁₂＝R_target·(0, −1, 0)。
     40/40 樣本 IK 成功且符合既有餘裕門檻（非手指 ≥ 5 mm、手指對門扇與立柱 ≥ 5 mm、底盤 ≥ 20 mm）；非手指最小 7.2 mm（uflite_gripper_link）。
  3. 位移只沿 R_target·e₁₂ᵀ：策略測試含旋轉把手與任意姿態的 e₁₂ 反例。
  - 這只支援規格：取樣之間未核、近似幾何、假設 TCP 到達目標；不是碰撞安全證明，也不是力—位移關係。

## 4. 調整律（登錄後固定，不搜尋、不調參）
- 濾波：每個有效物理步更新一次一階低通，ẽ_k ＝ ẽ_{k−1} ＋ β(e_k − ẽ_{k−1})，β ＝ 1 − exp(−0.01/0.05)（τ_f ＝ 0.05 s）。
- 死區：u ＝ sign(ẽ)·max(|ẽ| − 0.5 N, 0)。
- 速率：Δ̇ ＝ clip(k_i·u, ±v_max)，k_i ＝ 0.25 mm/(N·s)，v_max ＝ 1.0 mm/s。
- 總位移：Δ ∈ [−Δ_max, +Δ_max]，Δ_max ＝ 1.0 mm。
  - Δ_max 小於抓取漂移界 5 mm，也小於桿半徑 13 mm。
  - 從 0 走到上限至少需要 1 s；舊趟 OPEN 到首次低值約 1.2–1.4 s。
- 防積分飽和：Δ 達上限時，同向的積分即停止（條件積分），反向立即可動。
- 作用期間：
  - **OPEN 第一輪**：Δ ≡ 0，濾波器也不更新。DoorTargetK3 第一輪不前進，所以首筆 OPEN 發布目標**逐元素等於最後一筆 HOLD 發布目標**（≤ 1e-9），既有連續性契約不變。
  - **OPEN 第二輪起到 SETTLE**（Codex 同意 SETTLE 繼續調整；原到位、門角速率、雙指承載與保持時間條件全部保留）：Δ 逐來源步更新，每輪變化 ≤ v_max·dt_p·（本輪消費步數）。
  - **進入 END_HOLD 的那一輪起**：凍結**最後一筆實際已發布的完整目標**（不只凍結 Δ）；END_HOLD 3.0 s 期間發布目標完全不動、不消費配對。
  - **DONE**：照原策略**停止發布目標並停止求解器**（不是繼續發布固定目標）；紀錄保存最後目標。
  - OPEN 之前 Δ ≡ 0（HOLD 不作用、不疊加 U／X）；中止時 Δ 凍結並記錄。
- **偏移基準固定**：Δ 是唯一的累積狀態，基準為「不含 Δ 的基礎目標」。

## 5. 接入（只作 W-GMPC 外層的有界目標調整）
- 新 `K3ABPolicy(K3APolicy)`，不繼承 K3-A-U／K3-A-X。
- 發布目標 ＝ 基礎目標 ⊕ 平移 Δ·R_base·e₁₂ᵀ。
  - **DoorTargetK3 只旋轉上一筆「基礎目標」**，不旋轉含 Δ 的發布目標，避免 Δ 被併入基礎序列後重複累積（有測試鎖住）。
  - 策略內分存 `last_base`（基礎序列）與 `last_target`（實際發布）。END_HOLD 用 `last_target`；DoorTargetK3 建立時用最後一筆 HOLD 發布目標（此時兩者相同，因為 Δ ≡ 0）。
  - task trace 每列另記 `ph0`（該輪開始時相位）、`state_step`、`base_target`、`delta_m`、`e12_world`、`bal_consumed`（本輪消費的步號範圍）；
    task.json 另存 `balance`：s0、逐步紀錄 {step, sim_t, f1, f2, e, ef, u, v, delta, sat}、停止原因、緩衝 fault、重送與遲到計數。
- 不改 QP、成本、限制、偏移觀測器（interval 照舊）、夾爪命令、摩擦或門模型。
- 速度、加速度、輪級與關節限制沿用現值。
- 每週期記錄：使用的最新 pair 步號、e、ẽ、Δ、飽和旗標、有效性。
- **已登錄的交互風險**：偏移觀測器與本外層承載回授可能交互影響（例如慢振盪或互相抵消）；機制未分析。
  - 位移與速率有界（v_max、Δ_max）**不等於穩定性已證明**；不為此調參。
  - 觀察時列出 Δ 與 d̂ 的同窗時序。

## 6. 安全與停止（全部保留，不放寬）
- 保留既有停止條件：單指 < 0.5 N 持續 0.2 s、腕力、抓取漂移（**仍對原始 G_T_H_grasp**，5 mm／2°，不得重新初始化）、非預期接觸、閂鎖、門角 −1…9° 與 |q̇| ≤ 0.05、92 °C 熱中止。
- 新增停止：`balance_meas_invalid`（見 §2）。
- Δ 飽和不是停止條件，只記錄。

## 7. 判定與報告（兩者分開）
- **K3-A 任務**：沿用既有 check 規則的門檻與其他項目。但既有的 `target_not_rigid_rotation_of_first` 要求已發布目標是首筆的純鉸鏈旋轉，不能直接套在含 Δ 的發布目標上。
  另建 `door_k3ab_check.py`，分開核對（實際 TCP 漂移、接觸、腕力等安全判定仍用 room 實測與原始 G_T_H_grasp，不以基礎目標代替）：
  1. **基礎目標**：`base_target` 序列套用原本的參考檢查（q_ref 速率、超前、目標上限、到達、純鉸鏈旋轉 ≤ 1e-5、END_HOLD 基礎固定）。
  2. **發布＝基礎＋柔順位移**：旋轉部分逐元素相等（≤ 1e-9）；平移差 ＝ delta_m·e12_world（≤ 1e-9）；e12_world ＝ R_base·(0, −1, 0)（≤ 1e-9）；|Δ| ≤ Δ_max。
  3. **時序（依 ph0）**：首個 OPEN 列以前 Δ ＝ 0（精確）；首個 OPEN 列 Δ＝0，且**實際發布目標**＝前一筆已發布目標（≤ 1e-9）；
     END_HOLD 列實際發布目標＝進入前最後一筆已發布目標（精確）、Δ 不變、不消費；DONE 不發布。
  4. **調整律獨立重算**（判定器自寫，不匯入策略程式；常數為登錄值並核 task config 相符）：
     逐步紀錄自 s0＋1 連續；每筆 f1／f2、sim_t 與 room 同一步 leaf_contacts 相同（≤ 1e-9）；重算 ẽ 與 Δ（≤ 1e-12）；
     每列 bal_consumed 連續、不重疊、合起來＝全部紀錄、上界 ≤ 該列 state_step；該列 delta_m＝重算 Δ 於其最後消費步。
  5. **證據完整**：缺 base_target、delta_m、e12_world、ph0、balance 紀錄或含非有限值 ⇒ 不通過；balance 停止或緩衝 fault ⇒ 不通過。
  每一項都有完整判定流程的反例測試（`test_door_k3ab_judge.py`，r2 共 38 項）。
- **承載平衡觀察**（事前登錄、只描述、一趟介入觀察）：
  1. OPEN 期間逐步的 F1、F2、e、ẽ、Δ 時序；
  2. 首次單指 < 0.5 N 的物理步與門角（若有）；
  3. OPEN 期間 F1 的最小值；
  4. OPEN 期間 |e| ≤ 2 N 的步數比例；
  5. Δ 飽和的累計時間；
  6. 與 ev／au／ax 同窗並列。
- 不宣稱根因已定位，也不把承載平衡的結果當成任務通過。

## 8. 實作與測試（draft-2 核可後已完成，見 §9；實跑尚未核可）
- 新檔：
  - `door_k3ab_policy.py`、`door_k3ab_task_node.py`；
  - `isaac_door_k2_sim_ev2.py`（只插入 /door/finger_pair）；
  - `run_door_k3ab_ev.sh`、`door_k3ab_check.py`、`door_k3ab_report.py`；
  - 幾何核對 `door_k3ab_offline_geom.py`。
- 測試需涵蓋：
  - 正負號（含旋轉把手）；
  - 濾波、死區、速率、上限、條件積分；
  - 量測無效即停；
  - 同步配對（不同步的 f1、f2 不可配成一對）；
  - 基礎目標與 Δ 分離；
  - HOLD 期間與 OPEN 第一輪 Δ ≡ 0，首筆 OPEN 目標＝最後 HOLD 目標；
  - END_HOLD／DONE 目標固定、Δ 凍結；
  - 漂移仍對原始抓取；
  - 判定反例。
- 開跑前：凍結雜湊、Codex 確認，只跑一趟。

## 9. 實作狀態（draft-3，2026-10-23；離線測試全部通過，未開跑）
| 檔案 | 內容 | 測試 |
|---|---|---|
| `door_k3ab_balance.py` | PairBuffer（入口驗證、重送、take ≤ 狀態步、丟棄區、只在進展時刷新新鮮度）＋BalanceLaw（逐步、首步核 t0、停止不恢復） | `test_door_k3ab_balance.py` 15 |
| `door_k3ab_policy.py` | K3ABPolicy(K3APolicy)：OPEN 第二輪至 SETTLE 作用、轉換輪起凍結完整發布目標、DONE 停止發布、future_time 停止 | `test_door_k3ab_policy.py` 16 |
| `isaac_door_k2_sim_ev2.py` | EV 執行端只插入 /door/finger_pair（逐步、與 room 同源同 4 位小數） | `test_door_k3ab_nodes.py` 10（含節點、跑法、報告與 Δ／d̂ 對齊） |
| `door_k3ab_task_node.py` | 訂閱配對流直送緩衝；Meas 帶 state_step、pairs_age_wall；trace／balance 紀錄 | 同上＋`test_door_k3ab_task_node_ros.py` 1（ROS 接線） |
| `door_k3ab_check.py` | 版本化判定器（§7＋實作 r2） | `test_door_k3ab_judge.py` 38 |
| `door_k3ab_report.py` | 承載平衡觀察（§7；含對照 run 與 Δ／d̂ 同窗） | 結構測試在對照實錄上可跑 |
| `door_k3ab_offline_geom.py` | 雙向幾何核對（§3） | 結果 40/40 |
| `run_door_k3ab_ev.sh` | 相對 run_door_k3ax_ev.sh 只換執行端、任務節點、凍結清單與字樣；求解器參數全同 | 同上 |
