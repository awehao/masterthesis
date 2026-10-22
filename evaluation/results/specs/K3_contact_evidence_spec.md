# K3-A 接觸承載取證：小規格（2026-10-22；依 Codex 20261022_114336 裁定；已實作待審，**未實跑**）

**目的**：只增加診斷欄位，不改控制，以區分單指卸載的候選：
- 拉力經夾持面的傳遞（接觸點、法向與切向分布）；
- 夾爪繞橫桿的滾轉。

**不**用來調參或宣告根因。上位：door_k3a_1／door_k3a_adm_1／door_k3a_int_1 三趟都出現「兩指力分裂 → 繞桿滾轉 → 單指卸載」：**三種已試配置均出現類似失效，尚未獲得功能改善**（修訂 1 措辭）。

## 1. 控制配置
- 沿用 **INT 配置**（run_door_k3a_int.sh；ADM2 求解器加 `--offset-observer interval`）與所有中止條件、門檻、K3-A 範圍。
- 不張爪、不退離、不進 K3-B。
- 唯一差異：執行端換成取證複本 `isaac_door_k2_sim_ev.py`，並加 `--evidence-out "$DIR/evidence.jsonl"`。
  runner 為 `run_door_k3a_int_ev.sh`，凍結 `freeze_door_k3a_ev_dev.sha256`。

## 2. 取證執行端（`isaac_door_k2_sim_ev.py`，isaac_door_k2_sim.py 的複本）
- **只做插入**（6 處）；`--evidence-out` 預設空字串，此時不建檔、不讀任何新 API，與原版逐字同行為。
- 每個物理步，在原有 `leaf_contacts`／`wrist_force` 之後、使用**同一個**把手 prim 位姿 `_HM_s`，寫一列 JSONL：
  - `step`、`t`、`T_WH`（把手 prim 位姿 4×4 列優先）、`arm_sp`（本步手臂設定點 `res.arm_setpoint`）；
  - `contacts`：`get_contact_force_data(dt)`。感測端為門扇剛體，過濾對象為各機器人剛體。
    只保留 uflite_finger1／2、uflite_gripper_link、link_eef、link6，逐點記 `p`（點）、`n`（法向）、`fn`（法向力）、`sep`（分離距離）。
    **法向正負約定未經本專案核對，原樣保存**；
  - `friction`：`get_friction_data(dt)`，逐錨點記 `p` 與 `ft`（切向力向量）。與接觸點不保證一一對應，分開保存；
  - `j6_reaction_wrench`：`get_measured_joint_forces()` 的 joint6 列完整 6 維（入射關節的反作用力與力矩，子關節座標），**不是馬達輸出**；
  - `dof_measured_efforts`：`get_measured_joint_efforts()` 依 DOF 名取 joint1..6、finger_joint1..2。這是解算器在 DOF 方向的關節力，**不是馬達命令值**；
  - 任一項讀不到就記 None，並在 `<欄>_err` 附原因；非有限值記 None，不寫成 0。
- 每項 API 讀取都經 `KEV.safe`；整列寫出與逐步呼叫都包在 try 內，**取證失敗不得中斷模擬**（只計 n_err）。
- room_run.json 另加 `evidence` 摘要：筆數、各欄錯誤次數、路徑、各欄語意。
- 既有欄位、話題、判定器輸入一律不變；q_arm、q_finger、底盤、門角等仍在 room_run 的 steps 內，依同一個 step 對齊。

## 3. 測試（不啟動 Isaac）
- `test_door_k3_evidence.py` **7/7**：
  - pair 切片與 keep 過濾；pair 數不符時拋例外，經 safe 轉為 (None, 原因)；
  - 非有限記 None；摩擦切片；
  - 反作用力列（含 (1, J, 6) 形狀）；依名取 DOF 力（長度不符、名不存在都拋例外）；
  - 轉到把手座標的摘要；torch 型輸入。
- `test_isaac_door_k2_sim_ev.py` **5/5**：
  - 相對原版只有插入；`--evidence-out` 預設空；
  - 新 API（get_friction_data、get_measured_joint_efforts、KEV）只出現在插入行；
  - 逐步呼叫與收尾各在 `_ev_f is not None` 守衛下（AST 核對），且逐步呼叫包在 try 內；
  - `_ev_write` 內四項 API 都經 `KEV.safe`；
  - runner 相對 INT runner 只換執行端並加 --evidence-out。
- **未驗證**：get_friction_data 在本資產與這個視圖下實際回傳的形狀與內容（只有實跑才知道）。所以第一趟取證也兼作 API 可用性核對；讀不到會如實記錄。

## 4. 事前登錄的分析（取證趟跑完後；只讀）
- **窗口**：HOLD、OPEN→首次單指 < 0.5 N（物理步）、首次低接觸→中止、中止後。
- **每指**：
  - 接觸點數；法向力和；
  - 力加權的接觸點與法向，換到把手座標（x 沿桿、y 法向、z 鉛直）；
  - 切向力和（把手座標）。
- **由接觸點與力，依把手中心與夾爪 TCP 求合力與合力矩**，並與 j6 反作用旋量的趨勢並列。
  只比趨勢與符號，不宣稱數值等價：座標系不同，也還有其他接觸。
- **區分兩候選的觀察點**（事前寫下，不據此調參）：
  - 若卸載前，finger1 的接觸點沿 y（拉門方向）或 z 方向朝桿緣移動，且法向逐漸偏轉，傾向「接觸幾何改變／滾轉」；
  - 若接觸點不動，但切向力與法向力比值上升、接近摩擦錐，傾向「拉力經摩擦傳遞」造成的負載重分配；
  - 兩者都可能同時發生，只描述，不排他。
- 結果只稱「取證觀察」，不稱根因確認。

## 5. 申請
審查本規格與實作後，另行申請一次取證趟 `door_k3a_ev_1`。


---

# 修訂 1（2026-10-22；依 Codex 20261022_115319 四項必修；未實跑）

## 6. 故障隔離（`KEV.EvidenceWriter`）
- 開檔、寫出（含 allow_nan=False 的序列化）、週期 flush（每 200 筆）與關檔，每一步都包在 try 內；失敗只記入摘要（n_err、errors），不拋例外。
- 執行端只透過 EvidenceWriter 開檔，不直接 open。開檔失敗時取證停用，但只要給了 `--evidence-out`，room_run.json 一定封存 `evidence` 摘要。
- `complete` 只在「有開檔、無任何錯誤（含各欄讀取錯誤）、已正常關檔」時為 True；其餘一律 False 並列原因，**不靜默當成功**。
- 逐步記錄取證耗時：`step_s_total`／`step_s_max`（讀取加寫出），以及 `write_s_total`／`write_s_max`（序列化加寫出）。
- 實跑後另報漏時槽、過期丟棄、deadline miss 與取證耗時。**不宣稱取證完全無干擾**。
- 修正：先前插入的計時呼叫用了未匯入的 `time`，會在取證啟用時中斷模擬。已改用插入的 `import time as _ev_time`，結構測試以 pyflakes 核對沒有未定義名稱。

## 7. pair 索引驗證
- (count, start) 必須是有限、非負、整數值；有資料的切片須完全落在緩衝 [0, n_buf) 內（n_buf＝各緩衝長度的最小值）。
  不在 keep 中的 pair 越界也算錯，因為索引整體不可信。
- 不符時拋 ValueError，經 `KEV.safe` 記為該欄的取證錯誤，**不當成零接觸**。
- 反例：start＝−1、count＝−1、count＝1.5、count＝nan、切片超出緩衝、非 keep 越界、摩擦切片越界。

## 8. 幅值權重
- 法向力原始符號保留：每點 `fn`，摘要 `fn_signed_sum`。
- 幾何摘要明示用 **|fn| 幅值權重**：`fn_abs_sum`、`p_handle_absw`、`n_handle_absw`，另附未加權中心 `p_handle_unweighted`。
- |fn| 全零時，加權中心記 None（未定），不以等權冒充。
- 反例：fn＝(−1, −3)、位置 (0, 1) ⇒ 幅值加權 0.75（不是 0.5）；全零 ⇒ None。

## 9. 欄位語意（座標、作用側、參考點、單位；未知者明列）
| 欄 | 座標 | 作用側／方向 | 力矩參考點 | 單位 | 未知 |
|---|---|---|---|---|---|
| contacts.p | 世界 | — | — | m | — |
| contacts.n | 世界（單位向量） | **未核**：SDK 未載明由哪一方指向哪一方 | — | — | 正負約定 |
| contacts.fn | 純量 | 依 SDK，對應 n 方向；**可帶負號** | — | N（dt 換算後） | 符號意義 |
| contacts.sep | 純量 | 負值＝穿透（依 SDK 慣例，未核） | — | m | 慣例 |
| friction.ft | 世界 | 作用於哪一方**未核** | — | N | 作用側；與接觸點的對應 |
| friction.p | 世界 | — | — | m | — |
| T_WH | 世界←把手 prim | — | — | m | — |
| arm_sp | 關節 | 本步寫入 API 的設定點 | — | rad | — |
| j6_reaction_wrench | joint6 子關節座標 | 入射關節對子連桿的反作用（SDK 定義）；**非馬達輸出** | 子關節座標原點 | N、N·m | — |
| dof_measured_efforts | 各 DOF | 解算器 DOF 方向關節力；**非馬達命令** | — | N·m（轉動）、N（指） | 驅動與限位成分的分解 |

## 10. 分析界線（取代 §4 中對應句）
- 由接觸點與力求的合力、合力矩，參考點標為「把手 prim 原點」或「TCP」，座標為把手座標。
  j6 旋量是子關節座標、參考點在 joint6。**兩者未完成座標與參考點轉換前，不比較分量符號**，只報各自原始量或大小的趨勢。
- **法向作用側未核清前，不宣稱物理合力矩**；由接觸點求的結果只稱「幾何代理量」。
- pair 層級的「切向合量／法向合量」比只作**代理指標**，不稱接近局部摩擦錐或已滑移。
- 其餘同 §4：只描述、不排他，只稱「取證觀察」。

## 11. 測試
- `test_door_k3_evidence.py` **12/12**，新增：pair 索引驗證、幅值權重與未定、開檔失敗、寫出／flush／關檔失敗、停用與欄位錯誤時 complete＝False。
- `test_isaac_door_k2_sim_ev.py` **7/7**，新增：
  - pyflakes 無未定義名稱且 `_ev_time` 已匯入；
  - 只經 EvidenceWriter 開檔；
  - 收尾摘要在 `_ev_w is not None` 下一定封存。
  - 相對原版仍只有插入。
