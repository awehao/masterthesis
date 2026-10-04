# 時域消融（H1／H5）：準備資料

2026-10-04。依 `單步與多步比較_下一步計畫.md`，**不啟動模擬、不改控制程式**完成的準備工作。

| 檔案 | 內容 |
|---|---|
| `comparison_inventory.csv` | P0 既有證據清單（23 列）：每列的控制器版本、N、資料完整性、能支持與不能支持的主張 |
| `H1_H5_resolved_config.yaml` | P1 共同設定表與「只因 N 而不同」的差異 D1–D6（含終端權重 ×5 造成的成本結構差） |
| `replay_input_coverage.yaml` | P2 逐輪重播的輸入覆蓋、缺欄、補紀錄規格、H5 重播驗收草案 |
| `experiment_spec_C0.yaml` | P4 C0 實驗規格草案（前置條件、配對、判準、指標、分析規則） |
| `h5_ref_*.json` | b6 四趟（H5）的窗口指標，由 `evaluation/horizon_metrics.py` 產生 |

工具：

- `evaluation/horizon_config_check.py`：一組趟次除 N 外是否相同（N 取落盤值）。b6 MotM 三趟 0 差異；MotM 對停車只差 MotM 開關相關 4 項。
- `evaluation/horizon_metrics.py`：W-GMPC 窗口按相位的追蹤、限位、平順、計算指標。

主要發現：

1. **每輪的 TCP 目標矩陣沒有紀錄、無法忠實重建** ⇒ 現有 b6 不能通過 H5 重播驗收；同狀態 H1／H5 比較要等補紀錄後的新趟次。
2. 逐輪偏移估計 d̂ **可精確重建**（4 趟更新次數相同、最終差 ≤ 4.9e-7 rad）。
3. N=1 時唯一的一步就是終端步（Qf×5），兩組成本結構不同，報告須寫實際成本式。
4. **b6 MotM 手臂命令有間歇性反轉**（修正口徑：相鄰輪反轉 69／94／22 次，佔有效比較 26／23／9%；集中在保持與退開，三趟退開都有）。停車 0 次但有效比較只有 15–21 次，不能直接對比。決定 (a)：保留配置，當作待比較現象；候選原因（MotM 權重）未驗證。
5. 展開節點參數已改為落盤（wholebody.json args）；之前的趟次仍無法核對。

已實作（不改求解行為）：逐輪求解輸入／輸出紀錄（`wgmpc_cycle_record.py`）、`WGMPC_N`、展開節點參數落盤、重播核心 `horizon_replay.py`。
測試：`test_wgmpc_cycle_record.py` 9/9（含逐位元重播）、`test_horizon_metrics.py` 6/6；既有求解節點測試全部通過（`test_wgmpc_wg2_health.py` 需先 `source install/setup.bash`）。
