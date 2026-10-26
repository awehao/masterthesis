# 連續協同通行：核心段接觸驗證 — 規格 draft-5（閘控 r2，g2 版）

draft-1～4 保留。本版依 Codex 20261027_004139（條件式核可單趟 `door_coopen_core_g_1`，但 g r1 不得直接起跑）修正閘控放行缺口與判定器；**門檻、持續窗、抓法、參考、門速、APPROACH 15 s 總逾時均不變**。只列與 draft-4 的差異。

## 1. 措辭更正（模型核對）
`door_coopen_core_1_model_check.py` 用的是**結束時的 d̂ 回套整段 APPROACH**，只能稱「事後常值補償的一步殘差核對」，不是在線逐時補償已驗證。
- 結論縮為：**現有證據不足以支持先前「後仰構型設定點模型偏差」的歸因**（撤回）。
- 不稱「推翻」或「三趟一致」。
- 不再做辨識或擬合。

## 2. 閘控 r1 的放行缺口（Codex 直接呼叫策略重現）
| 情境 | r1 結果 |
|---|---|
| 垂直 0、沿軸落後 10 mm | 未暫停，s 從 0.050 推到 0.049 m |
| 已暫停、s 1 mm、位置到抓取點、姿態 0.03 rad | 0.5 s 後轉入 ENGAGE，閘控仍顯示 paused |
| ALIGN_HOLD 兩筆有效資料間隔 0.6 s | 算成連續確認，進入 APPROACH |

原因：只在垂直超限時暫停；父類執行後才還原 s，無法阻止父類先切換相位；確認窗只要求狀態時間前進。

## 3. 閘控 r2（`door_coopen_approach_gate2.py`，mixin，r1 保留）
1. **推進許可**：本輪量測有效 且 未暫停 且 |沿軸| ≤ 5 mm 且 姿態 ≤ 0.02 rad。
   - 垂直遲滯暫停 3／2 mm、續進 0.5 s 不變。
   - 不許可時，父類推進後還原 s；本輪目標仍為 T(s)，控制照常發布。
   - 沿軸、姿態超限只擋當輪推進，不另啟動 0.5 s 恢復窗。
2. **APPROACH → ENGAGE**：推進許可為假時攔下 `_go('ENGAGE')` 並清除 settle_t0。另外，APPROACH 中任一輪不許可即清除 settle_t0，所以閉爪前 0.5 s 穩定窗只在閘控持續許可下累積（比 Codex 要求更嚴，揭露）。
3. **確認窗連續性**：相鄰有效樣本狀態時間間隔 > 0.2 s（stale_s，既有新鮮度尺度）就清空對準與恢復累積窗。非有限、倒退、重讀、state_age 超限一律算無效：清空累積窗、不許推進。

**預期影響（照實）**：「沿軸 ≤ 5 mm 才推進」等同限制沿軸落後；正面夾法實跑的接近沿軸落後約 11–12 mm。
- 粗估平均推進速率約減半（0.02 → 約 0.01 m/s），接近 0.10 m 約 10 s，加上可能的垂直暫停，逼近 15 s 總逾時。
- 逾時就照實失敗，不加時。

## 4. 判定器 r4（`door_coopen_check_core4.py`，r3 保留）
J4 先檢查型別（int／float、非 bool）與有限性再逐鍵比較。

| 情況 | 理由 |
|---|---|
| 登錄值為非數值、NaN、Inf 或缺鍵 | coopen_gate_invalid_registered |
| 紀錄值為非數值、NaN、Inf 或缺鍵 | coopen_gate_invalid_recorded |
| 缺 coopen_gate | coopen_gate_missing |
| 數值不同 | coopen_gate_mismatch |
| 檢查過程出錯 | coopen_gate_check_error（不拋例外） |

## 5. 接線
- `door_coopen_core_task_node3.py`（v2 複本）：只把閘控換為 r2，coopen_gate 記 r2 模組雜湊。
- `run_door_coopen_core_g2.sh`（g 複本）：只換凍結、任務節點、判定器字樣，非註解差異恰三行。

## 6. 測試
| 測試 | 內容 | 結果 |
|---|---|---|
| `test_door_coopen_approach_gate2.py` | r1 的 G1–G8 以 r2 重跑全過；R1 沿軸落後 10 mm 時 s 不推進；R2 姿態 0.03 rad 時不進 ENGAGE，移除後才進，且 ENGAGE 前 0.5 s 每輪許可；R3 兩筆間隔 0.6 s 不算確認；R4 恢復窗中 0.6 s 空白會清空重算；R5 重讀不推進 | 13／13 |
| `test_door_coopen_check_core4.py` | J4 型別、有限性、缺、不符 | 4／4 |
| `test_door_coopen_core_g2_nodes.py` | 節點 v3 啟動並記 r2 閘控設定與模組雜湊；r5 拒跑；J4；跑法 g→g2 只差三行 | 4／4 |

- 合成世界不是真鏈端到端。

## 7. 凍結與執行
- `freeze_door_coopen_core_g_dev_r2.sha256`＝g r1 清單＋本包新檔（最後建立；g r1 保留）。
- 單趟 `door_coopen_core_g_1`：跑法 `run_door_coopen_core_g2.sh`、判定器 `door_coopen_check_core4.py`。
- 執行條件同 draft-3 §8：只跑一次、乾淨環境、92 °C、起跑前後凍結核對、失敗保留不重試、不新增影片、不放手不收臂。
