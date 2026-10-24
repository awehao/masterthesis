# door_k3m_pause_p2b_2：交棒取消再現，未進入受測流程（不改程式重試；2026-10-24）

- **核可**：依 p2b 規格 §6 由 Claude 依授權核可，只重試一次。凍結 p2b r1 在起跑前與結束後都核對通過。CPU 峰值 87 °C，熱中止 0 次。
- **啟動**：暖啟動，起 sim 到起節點 9 s，與 p2_1 相同。
- **結果**：與 p2b_1 相同，未進入受測流程。
  - WAIT_HANDOVER 0.92 s → ABORTED 30.98 s（`timeout:WAIT_HANDOVER`）。
  - 執行端步 92 排定、步 95 切換，步 144 取消，理由「接手方未備妥：命令鏈從未收到全身命令（n_recv = 0）」。
- **時序**：展開端命令約在模擬 4.0 s 才首度到達執行端（3003 步被拒）；p2_1 是約 0.7 s。
  - 所以冷啟動不是唯一條件。同期系統負載 load1 3–6，VS Code renderer 約 196% CPU。只描述，不推定根因。
- **判定器 check_p2**：`pass` False（`aborted:timeout:WAIT_HANDOVER`、`no_unfold_event`、`settle_p2_unverifiable`、`pause_evidence_missing` 等）。
- **保護**：完整（LW_HEALTH sim exited/0、obs exited/0）。
- **處置**：依 p2b §6 停止重試，另立 p2c 交棒前置條件（`results/specs/K3M_pause_p2c_handover_precondition.md`）。
- **影片**：機器人無動作，仍依規定交 side／close 實錄重播。

## 更正（2026-10-24，p2c_1 之後查到）
- 本趟執行時 `python3` 指向專案虛擬環境 `.venv-dl0`（numpy 2.5.2、無 osqp），不是先前各趟所用的系統 Python。原因是重開機後 VS Code 終端自動啟用該環境。
- 因此本趟屬**環境錯誤**，不計入受測結果。展開端啟動較慢與這個環境的關係未驗證。
- 「冷啟動」「系統負載」都只是同期觀察，不作為原因。
- 見 `runs/door_k3m_pause_p2c_1/analysis/NOTE_k3m_pause_p2c_1_result.md`。
