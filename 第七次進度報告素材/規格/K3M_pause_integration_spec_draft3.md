# K3-M 受控暫停接入 draft-3：開跑前四項必修（Codex 20261024_095031）；凍結 k3m_pause r2；**未開跑**

draft-2 其餘內容不變。本文件只列四項修正與對應反例。

## 1. 停止監督器（`door_k3m_pause_sup.py`）
- **新鮮度**：每週期除了窗口內部間隔合法，**最新合法樣本**的來源模擬時間須 ≥ 本週期 − 0.1 s，且接收牆鐘年齡 ≤ 0.2 s；否則無值。
  - ADM4 餵同步狀態時帶接收牆鐘。
  - 重送（步號或時間不遞增）拒收、不刷新。
- **例外修正**：無效樣本記為斷點，下一筆合法樣本照常比較，不再拋 TypeError。
- **套用回報**：所有階段嚴格驗證步號、時間、3＋6 維、有限值，非法 ⇒ FAIL `applied_invalid`。
  - ADM4 回呼收到錯誤長度或解析失敗 ⇒ 明確 FAIL，不只記事件。
- **檢查順序**：每週期先查求解失敗與逾時，再處理「尚未成窗或不新鮮」。LOWSPEED 無值只重設計時；其他階段無值 ⇒ FAIL `measurement_stale` 或 `measurement_invalid`。
- **反例**（共 13 項，新增 6 項）：
  - 你的反例：只餵 0.10 s 後停餵 ⇒ 不確認任何停止；
  - 缺樣本後再來合法樣本不拋例外；
  - 四個階段的非法套用回報都 FAIL；
  - 無窗口時求解失敗與逾時仍先判；
  - 重送不刷新新鮮度；牆鐘新鮮度。

## 2. 命令入口執行期互斥（`door_k3m_applied_src.EnvIntake` r2、ev4）
- **接管**：第一筆封裝進鏈＝求解端接管（takeover_t）。之後一般命令話題的命令**不進鏈**，並走既有 K2 閂鎖中止（`cmd_entry：plain_cmd_after_takeover`）。
- **進鏈前核對**：本趟 run_id（`ENV.run_id_num(run_id)`）、段別＝adapter、種類＝normal、derived 且 source_seq ≥ 0、序號。
  - 同序同值 ⇒ 冪等，不重新進鏈；
  - 同序異值、倒序、錯趟、錯段、錯種類、非衍生 ⇒ 不進鏈，並走既有閂鎖中止（`cmd_entry：<why>`）。
- **反例**（真 CmdChainE2）：
  - 你的反例：先收 seq 10 零命令，再送另一 run_id、seq 7 的非零命令 ⇒ `wrong_run`，不進鏈；
  - 另有倒序、同序異值、錯段、錯種類、非衍生，鏈收到筆數不變、仍套用 seq 10；
  - 接管前允許舊路徑、接管後不允許；
  - 晚到的舊命令在進鏈前即被擋（監督器的 `old_command_after_stop` 仍是第二道）。
- 執行端接線的違規中止路徑寫在 ev4 主迴圈內，與既有走廊越界的閂鎖同一套；Isaac 端只能在實跑中驗證。

## 3. 零速發布值與求解值分開（ADM4）
- **紀錄**：逐輪另記 `pause_u_solve`（QP 求解值）與 `pause_u_published`（暫停處理後的最終發布值；零速段手臂為 0、底盤為 decel 值）；實際套用值由 room `applied_src` 與 base_cmd／applied_arm 記錄。
- **最終發布值核對**：`publish_feasible` 沿用求解器既有限制，核對速度框（vbox）、加速度框（amax·dt，相對權威 applied u_prev）、輪速、輪加速度。
  - 不合 ⇒ 不發布、暫停 FAIL `published_infeasible:<項目>`。
- 停止序列的來源序號就是這筆最終發布值的封裝序號（note_published 在發布後記錄）。
- **ROS 假世界 P1**：零速段 `pause_u_published` 手臂為 0，同一輪 `pause_u_solve` 手臂約 1 rad/s（替身無物理），可行性核對通過。P1–P4 仍全過。

## 4. 判定器（`door_k3m_check_p.py` r2）與完整凍結目標
- **分項判定**：`check_pause` 回傳 (cmd_ok, phys_ok)，各自只在必要證據齊全且核對通過時為 True；缺證據 ⇒ (False, False)。
- **趟別**：request、兩個 ack、room.k3m.config 的 run_id 都須等於 task.args.run_id。
- **ack**：求解端 ack 的 published 須為 True。
- **凍結目標**：請求時刻以前最後一筆已發布目標的 pub_seq 須等於 frozen_target_seq，完整 4×4 須等於請求。
  - 握手模組、策略、ADM4 的請求與 ack 都改帶完整 `frozen_target_T`（列優先 16）；任務端與判定器同時比較位置與旋轉。
- **反例**（判定器 4 項；握手模組新增旋轉不符、T 與 xyz 不一致、T 長度錯）：
  - 缺證據 ⇒ 分項 False；其他 run_id；published=False；
  - 引用較舊序號；只差旋轉；
  - 底盤漂移 ⇒ 命令停止 True、實際停止 False。

## 5. 測試總表（全部通過）
- 監督器 13；握手 4；套用來源 5；策略 5；判定器 4；任務節點 ROS 1；ADM4 ROS 假世界 P1–P4。
- 回歸：原 K3-M 策略 12、暫停 9。

## 6. 凍結與申請
- `freeze_door_k3m_pause_dev_r2.sha256`：r1 清單＋本文件，修改檔重算雜湊；跑法 FRZ 指向 r2。
- 申請範圍同 draft-2 §4：單趟 `door_k3m_pause_1`；門檻、92 °C、25 s、10 s 觀察與觀察器都不變；成功或失敗都交 side／close 實錄重播。
