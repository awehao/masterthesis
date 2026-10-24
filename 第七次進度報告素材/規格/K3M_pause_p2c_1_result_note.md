# door_k3m_pause_p2c_1：環境錯誤（虛擬環境誤啟用），不計入受測結果（2026-10-24）

- **核可**：依 p2c 規格 §5 由 Claude 依授權核可。凍結 p2c r1（178 檔）在起跑前與結束後都核對通過。CPU 峰值 85 °C，熱中止 0 次。
- **交棒前置條件**：生效。
  - 閘門通過後等 0.36 s，請求時已收到 20 筆、最新一筆 8 ms，交棒在步 171 成功。
  - 展開端全程送出 258 筆。
  - `wb_cmd_t_first` 為 nan：首筆在第一筆同步狀態之前收到，只是紀錄欄位，不影響判斷。
- **相位**：WAIT_HANDOVER 1.32 s → UNFOLD 1.74 s → ALIGN_PRE 7.90 s → ALIGN_HOLD 7.98 s → ABORTED 16.42 s。
  - 中止原因是 `executor_violation`，即停車監看 `dev_yaw 2.002° > 2.00°`。
- **求解端啟動失敗**：`solver.log` 記 `ModuleNotFoundError: No module named 'osqp'`。
  - 求解節點沒有輸出，ALIGN_HOLD 期間底盤偏航由 0.11° 漂到 2.0°，觸發既有保護。
  - 保護如設計運作。
- **環境錯誤**：重開機後 VS Code 終端自動啟用專案虛擬環境 `.venv-dl0`（`VIRTUAL_ENV` 在 PATH 最前）。
  - 跑法的 `python3` 因此指向 venv：numpy 2.5.2、無 osqp。系統 Python 是 numpy 1.26.4、osqp 1.1.1，先前各趟用的都是系統 Python。
  - 本趟以後才查到，**p2b_1、p2b_2 同樣在這個環境下執行**。
- **判定器 check_p2**：`pass` False。
- **處置**：
  - 本趟不計入受測結果，原始紀錄保留。
  - 已在系統 Python 下重跑全部單元與 ROS 測試，全過，含 ADM4 求解端暫停 P1–P4。
  - 下一趟 `door_k3m_pause_p2c_2` 以乾淨環境執行：移除 venv 路徑、`unset VIRTUAL_ENV`。起跑前確認 `python3` 為 `/usr/bin/python3` 且可匯入 osqp。
  - 程式與凍結不變。
- **保護**：完整（LW_HEALTH sim exited/0、obs exited/0）。

## 更正（Codex 20261024_231908）
- 本趟為**環境失敗**，排除於有效操作比較，但保留在全部嘗試紀錄中。
- 機器人**有動作**：交棒、展開完成並進入 ALIGN_HOLD；因求解器未啟動而底盤偏航漂移，觸發保護。
