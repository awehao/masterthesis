# 連續協同通行：核心段 — 規格 draft-8（指定初態隔離驗證，iso2 版）

draft-7（iso r1）保留、不跑。本版依 Codex r16 的兩項必修版本化修正；**目標、範圍、登錄 r8、初始化方式、其餘參數與保護全部同 draft-7**，只列差異。

**成果口徑不變**：「指定初態、經實際夾持取證後的持握開門與出口繞行核心驗證」。
- `chain_pass` 不得自動稱為「邊開邊繞成功」。
- 須分報夾持成立、實測開門、開門同期出口進展。

## 1. INIT_CHECK 連續確認 r2（`door_coopen_init_grasp2.py`；r1 保留）
r1 缺口：只累計位置合格的時間，沒有姿態准入；重複／倒退樣本仍會累積；資料空白不會重置確認窗。

r2 規則（尺度全部沿用既有值）：

| 項目 | 規則 |
|---|---|
| 有效樣本 | T_tcp、H 有限；state_age ≤ 0.2 s（stale_s）；狀態時間嚴格大於上一筆有效樣本（重複／倒退 ⇒ 無效） |
| 合格樣本 | 位置誤差 < 2 mm（grip_tol_m）**且**姿態誤差 ≤ 0.02 rad（登錄 approach_gate.rot_resume_rad） |
| 無效樣本 | 確認窗清空 |
| 資料空白 | 相鄰有效樣本間隔 > 0.2 s ⇒ 確認窗清空 |
| 轉入 ENGAGE | 合格連續 ≥ 0.5 s（grip_settle_s） |

- 每輪 `act['init']` 記 valid、why、err_p_m、err_r_rad、good、win_t0。任務節點 v6 寫入 trace，供判定器核對。
- 監看擴充同 r1：只在 INIT_CHECK 期間暫時擴充 ACTIVE_MONITOR，try/finally 還原（Codex r16 §3 接受）。

## 2. 任務節點 v6（`door_coopen_core_task_node6.py` ＝ v5 ＋ D1–D3）
- 改 3 行：匯入、策略建構 `IG2.init_grasp2(AG2.gated2(PP2.K3MP2Policy))`、grasp_init 模組紀錄。
- 新增 1 行：trace 的 init 欄位。

## 3. 判定器 r7 ＝ r4（J1–J4）＋ J6 r2（`door_coopen_check_core7.py`；r6 保留）

### 證據形式
- **六軸資料**：qa_grasp、stow_q、q_arm_meas、arm_sp 都須恰為 6 個有限數（非 bool），否則 `iso_six_axis_invalid`。修正 r6 的 zip 截短誤收。
- **兩種狀態必須區分**：
  - 有效查詢、沒有接觸點：contacts 為 dict 且 contacts_err 為 None；空 dict 合法。
  - 沒有量到：contacts 缺、不是 dict，或 contacts_err 非 None。
- **逐步覆蓋**：生成至閉爪前的每一個 room 步，都須有同步號、且為有效查詢的證據列。每個接觸點的 fn、sep 須為有限數。缺步、查詢錯誤、NaN 都判 `iso_contact_evidence_invalid`。修正 r6 的 NaN 誤收。
- **room 欄位**：
  - finger_contact_n 為 None 時（實錄前 160 步手指接觸視圖未啟用），只在同步證據為有效查詢時接受，並計數。修正 r6 的 max(None) 拋錯。
  - finger_contact_n 不是 None 時，須為 2 個有限數。
  - leaf／wall 須為 dict。

### 確認窗 `iso_confirm_invalid`
- 取 `init_check:pass` 事件的 win_t0 至 ENGAGE 時刻。
- 窗內每一列 trace 的 init 都須存在、valid 且 good、位置 < 2 mm、姿態 ≤ 0.02 rad，且 win_t0 一致。
- 相鄰列間隔 ≤ 0.2 s，窗長 ≥ 0.5 s，列數 ≥ 2。
- 判定器逐列核對紀錄，不只相信事件。

### 其餘判準與分項報告
- 其餘判準同 r6：設定、實測、設定點、相序（出現 ALIGN／APPROACH 即不通過）、閉爪前接觸、證據缺。
- 分項報告同 r6。

## 4. 測試（全過）

| 測試檔 | 內容 | 結果 |
|---|---|---|
| test_door_coopen_init_grasp2 | 10 項，見下 | 全過 |
| test_door_coopen_check_core7 | 7 項，**全部為合成資料**，見下 | 全過 |
| test_door_coopen_core_iso2_nodes | 4 項：v6 紀錄與 trace init、r6 拒跑、v6 對 v5 差異限定、iso2 對 iso 只差 FRZ／spawn task／結束字樣 | 全過 |
| 回歸 | init_grasp 7、check_core6 3、gate2 13、core_policy 5、check_core4 4、k3a_policy 18、settle_p2 11 | 全過 |

test_door_coopen_init_grasp2：
- K3A 組合：正常；0.6 s 空白（重讀）；姿態 0.03 rad 阻擋；確認期間重讀失效；倒退時間；逾時；缺逾時設定；缺 gcfg；三種監看中止。
- **實際 K3MP2Policy 組合**：正常到 DONE；手指—橫桿／牆／門角三種監看中止都發生在 INIT_CHECK；拿掉擴充的反證。

test_door_coopen_check_core7：
- 正常（finger None 受同步有效查詢支持 160 步）；
- 外層 `judge()` 的 J6 部分通過（iso_ok）；
- Codex 重現三例；7 種證據形式反例；7 種確認窗反例；9 種判準反例；
- g3_1 實錄 ⇒ iso_reg_invalid。

## 5. 實跑（Codex r16 條件式核可：上述修正、完整測試、新凍結全過後）
- 單趟 `door_coopen_core_iso_1`，跑法 `run_door_coopen_core_iso2.sh`，凍結 `freeze_door_coopen_core_iso_dev_r2`（最後建；跑前、跑後核對）。
- 乾淨環境、92 °C 守護；失敗保留，不補跑；不新增影片。
- 不做放手、退離、收臂或穿越；不放寬保護。
