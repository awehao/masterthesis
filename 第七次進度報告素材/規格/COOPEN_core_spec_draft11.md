# 連續協同通行：核心段 — 規格 draft-11（台架初態啟動，bench3 版）

draft-10（bench2）保留、不跑。本版只依 Codex r22 補正判定器：**執行端 ev9r2、求解端 ADM17、任務節點 v8、登錄 r9、初值、參數與保護全同 draft-10**。

**成果口徑**：指定初態（台架啟動）、經實際夾持取證後的持握開門與出口繞行核心驗證。
- 台架初始化與 `chain_pass` 都不等於核心動作已成功。
- 夾持、實測開門、開門同期出口進展分項報告。

## 1. 判定器 r11（`door_coopen_check_core11.py`；r10 保留）
- **首步原始證據獨立核對**，不再只相信自報摘要：
  - 欄位：必要欄位缺、型別錯或非有限即拒；
  - 身分：owner＝wholebody，kind＝wholebody_applied，套用來源 seq＝宣告，mode 0；
  - **自行重算** s_init＋dt·u_arm（dt＝room config.physics_dt），與 arm_setpoint 比（1e-9）。
- **與 room_run 同步號列交叉核對**：sim_t、owner 碼、kind、base_cmd_body／applied_arm 對九軸、applied_src 的 seq 與模式碼。
- **與 evidence 同步號列交叉核對**：arm_sp 對 arm_setpoint。
- **容差依記錄精度登錄（TOL_REC）**：

| 欄位 | 記錄精度 | 容差 |
|---|---|---|
| room sim_t | 四位小數 | 5e-5 |
| room 命令 | 六位小數 | 5e-7 |
| 證據 arm_sp | 九位小數 | 1e-8 |
| 自行重算 | — | 1e-9 |

- 新理由：`bench_first_step_raw`、`bench_first_step_room`、`bench_first_step_evidence`。
- 其餘同 r10：錨點交叉核對後提供鏈閘門視圖、不豁免、鏈窗口別名。

## 2. 最外層完整正例（**合成資料**；`door_coopen_bench_positive_synth.py`）
- 以真實通過鏈判定的 door_wide_w1f_60_3 為底，轉換成台架流程形式：
  - 從第一筆帶求解端來源的步起裁切；
  - 合成登錄：鏈常數用該趟原值，另加 coopen／閘控／初態／台架各鍵；
  - 相位改為 BENCH_START → INIT_CHECK → ENGAGE…，trace 與 task.json 同步轉換；
  - 補上台架與首步紀錄（以該步實值建立）。
- 真正最外層 r11 `judge()` 跑完：**why＝[]**，chain_exempt＝[]，錨點一致，未 mock 任何子判定器。
- 這是判定器的正例，不是台架實錄，不得當成果引用。

## 3. 測試（全過）
- **test_door_coopen_check_core11**（6 項）：
  - Z1：完整正例 why＝[]。
  - Z2：Codex 重現四例（owner＝nav、kind 錯、九軸 NaN、設定點 100）⇒ raw。
  - Z3：room 列竄改五例（applied_src seq、base_cmd、sim_t、owner 碼、缺列）⇒ room。
  - Z4：證據 arm_sp 偏 1e-6、缺步 ⇒ evidence。
  - Z5：缺欄或型別錯三例 ⇒ raw。
  - Z6：最外層錨點竄改 ⇒ bench_anchor ＋ k3m_room_evidence_invalid。
- **回歸**：check_core10 5、bench2_static 3、bench_start2 4、bench_grasp2 5、check_core8 11、init_grasp2 10、k3a_policy 18，全過。

## 4. 實跑（Codex r22 條件式核可：上述全過、新凍結、跑法更新後執行一次，不另開設計審查）
- 單趟 `door_coopen_core_bench_1`，跑法 `run_door_coopen_core_bench3.sh`，凍結 `freeze_door_coopen_core_bench_dev_r3`。
- 範圍：實際夾持 → 開門至 48° → 觀察開門同期往出口繞行 → 既有受控收尾。
- 不放手、不退離、不收臂、不穿越。
- 乾淨環境、92 °C 守護、跑前後凍結核對；失敗保留、不補跑、不新增影片。
