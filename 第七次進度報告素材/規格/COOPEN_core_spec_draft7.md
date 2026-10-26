# 連續協同通行：核心段 — 規格 draft-7（指定初態隔離驗證，iso 版）

draft-1～6 保留。本版依 Howard 2026-10-27 確認「先驗證核心動作」與 Codex r15 補充裁定（`reviews/20261027_014242_howard_confirmation.md`）。

**成果口徑（唯一）**：「指定初態、經實際夾持取證後的持握開門與出口繞行核心驗證」。
- 本版跳過自主接近，**不代表接近已解決**，也不稱完整自主開門通行。
- 既有失敗結果（door_coopen_core_1、g_1、g3_1）與凍結全部保留。

## 1. 目標（不變）
- 門持續開啟時，底盤同時沿出口方向繞行。
- 不是先開門、停住、放手退離再起步。本版不要求握著把手完成穿越。

## 2. 與 g2（draft-5）的差異（只列差異）

| 項目 | g2（原流程） | iso（本版） |
|---|---|---|
| 登錄 | r6 | **r8** ＝ r6 ＋ grasp_init（**不含 r7 的接近段 w_a 0.01**） |
| 手臂初始 | --stow-q K2_READY | --stow-q **qa_grasp**（`coopen_core_h0_r4.json`，門關閉 0° 候選抓取構型），手指張開生成 |
| 展開目標 | Q_PRE（預抓取） | qa_grasp（與初始相同 ⇒ 零位移；展開節點 v 不改） |
| UNFOLD 之後 | ALIGN_PRE → ALIGN_HOLD → APPROACH（s 0.10 → 0） | **INIT_CHECK**（目標 ＝ T(H, s＝0)，與 APPROACH 終點同一式） |
| 閉爪准入 | APPROACH 中 s＝0 且誤差 < 2 mm 連續 0.5 s | INIT_CHECK 中誤差 < 2 mm 連續 0.5 s（同 K2Config grip_tol_m／grip_settle_s） |
| 逾時 | APPROACH 15 s | INIT_CHECK 15 s（沿用同值） |
| 任務節點 | v3 | **v5** ＝ v3 ＋ B1–B5（8 行新增 ＋ 策略建構 1 行替換）；不含 v4 |
| 判定器 | r4（J1–J4） | **r6** ＝ r4 ＋ J6 指定初態（不含 J5） |
| 凍結 | g_dev_r2 | **iso_dev_r1** ＝ g_dev_r2 全清單 ＋ 新檔 |

- **不變**：
  - 執行端 ev5、轉接、求解 ADM9 及其全部參數（w_a 0.05）；
  - 展開節點程式；
  - 觀察器、增益、權重、限制、門檻；
  - 手指上限、暫停定義、門速 0.6 °/s、θ 48°、走廊／引導表、包絡、觀察器上限；
  - 閘控 r2（只作用於 ALIGN_HOLD／APPROACH，本版不出現）。
- ENGAGE 起全部沿用原策略：K2 閉爪斜坡、執行端逐物理步接觸證據、G_T_H 量測、HOLD 1 s，以及 OPEN／SETTLE／PAUSE 與既有停止。

## 3. 初態相位模組 `door_coopen_init_grasp.py`（mixin `init_grasp(Base)`）
- **(I1)** UNFOLD 完成 ⇒ INIT_CHECK。紀錄中不出現 ALIGN_PRE／ALIGN_HOLD／APPROACH。
- **(I2)** INIT_CHECK 依 K2 規則轉入 ENGAGE：
  - 凍結把手位姿 H_frozen，並記閉爪起點；
  - 誤差超限即清除持續窗。
- **(I3) 監看範圍**：父類 monitor 只在 `door_k3a_policy.ACTIVE_MONITOR` 列舉的相位執行，原集合不含 INIT_CHECK。
  - 本類只在 INIT_CHECK 呼叫 monitor 期間暫時擴充該集合，呼叫後立即還原，不改原模組檔。
  - 因此 INIT_CHECK 受與 APPROACH 相同的全部監看：門角／門速、意外接觸、牆、手指—橫桿（不在 FINGER_BAR_ALLOWED ⇒ 閉爪前 > 0.1 N 即中止）、腕力、底盤漂移、求解失效、逾時。
  - 測試 I8 為反證：拿掉 (I3) 後，相同注入不會中止 ⇒ (I3) 是必要的。
- 不固定附著、不直接綁住把手、不偽造接觸證據；持握未成立即中止，不進開門段。

## 4. 初始化核對
- **離線（既有，h0 r4）**：qa_grasp 開指淨空為非手指 7.28 mm、手指 16.99 mm、自身 11.42 mm、底座 177 mm。
  - CLI 以六位小數傳遞（與登錄值差 ≤ 5e-7 rad）。
- **執行端**：停車閘門要求手臂與 --stow-q 差 ≤ 0.02 rad 才通過（既有，ev5 不改）。
- **判定器 J6**（必要判準，任一不符即不通過）：

| 理由 | 條件 |
|---|---|
| iso_reg_invalid | 登錄缺 grasp_init，或 task 紀錄與登錄不符 |
| iso_init_config_mismatch | 執行端 config.stow_q 與 qa_grasp 差 > 1e-6 |
| iso_init_meas_mismatch | 首步實測與 qa_grasp 差 > 0.02 rad |
| iso_setpoint_mismatch | 首個 wholebody 步的證據 arm_sp 與實測差 > 0.02 rad |
| iso_phase_sequence | 不含 UNFOLD → INIT_CHECK → ENGAGE，或出現 ALIGN／APPROACH（不得偽造接近） |
| iso_contact_before_close | ENGAGE 前任一證據接觸對 fn > 0 或 sep < 0（穿透），或 finger_contact_n > 0，或 leaf／wall 接觸非空 |
| iso_evidence_missing | 必要紀錄缺 |

- **分項報告**（整體通過仍由 J1–J4 ＋ J6 決定）：
  - 持握成立：ENGAGE → HOLD 且無 grasp／finger 類理由；
  - 實測門角最大值對 θ_goal；
  - 開門同期出口進展（core2 既有欄位）；
  - 閉爪前接觸對最小 sep（只涵蓋進入接觸偏移帶的對，非完整淨空）。

## 5. 測試（全過）

| 測試檔 | 內容 | 結果 |
|---|---|---|
| test_door_coopen_init_grasp | 7 項：正常相序且無接近相位、偏移阻擋閉爪、逾時、監看四種中止、集合還原、缺逾時拒絕、UNFOLD 當輪中止、(I3) 必要性反證 | 全過 |
| test_door_coopen_check_core6 | 3 項：正常；12 個反例各對應一條理由；g3_1 實錄 ⇒ iso_reg_invalid | 全過 |
| test_door_coopen_core_iso_nodes | 4 項：v5 以 r8 啟動紀錄正確且無 approach_cost；r6 拒跑；v5 對 v3 差異限定；iso 對 g2 跑法差異限定 | 全過 |
| 回歸 | gate2 13、core_policy 5、check_core4 4、k3a_policy 18 | 全過 |

## 6. 實跑（待 Codex 核可；禁止自動補跑）
- 單趟 `door_coopen_core_iso_1`，跑法 `run_door_coopen_core_iso.sh`，凍結 iso_dev_r1（最後建；跑前、跑後核對）。
- 乾淨環境、92 °C 守護；失敗保留，不補跑；不新增影片。
- 不做放手、退離、收臂或穿越；不放寬接觸、幾何、漂移或停止保護。
