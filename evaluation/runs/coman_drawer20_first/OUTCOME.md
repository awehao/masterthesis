# coman_drawer20_first：**啟動中止，未進入任務**

**不是操作失敗** —— 任務從未開始，因此三題（完整操作、拉動期間同動、執行期時效）
**都沒有證據**，不得以本趟填任何一題。

| 項目 | 值 |
|---|---|
| 時間 | 2026-09-30 11:48:27 起，11:51:28 cleanup |
| RUN_ID / domain | `coman_drawer20_first` / 71 |
| 起動前檢查 | **0 項失敗**（`preflight.json`: `{"failed": []}`） |
| 中止點 | Isaac 啟動階段，`return 10` |
| 中止訊息 | `[drawer] **預期恰好 1 個 world→根 固定關節，中止**`（實際 0 個） |
| 進入任務 | **否**（未等到 `/clock`，`exit 3`） |
| 趟次資料 | 無（只有 `preflight.json`、`run.log`） |
| CPU 峰值 | 57.0 °C（限 92 °C，未接近） |
| 殘留程序 | 無，cleanup 完成 |

## 原因（兩項，都在我們的程式）

1. **固定關節守衛與 `--free-base` 相斥**：守衛寫死「恰好 1 個 world→根固定關節」，
   是為固定底座版寫的；`--free-base` 讓匯入器不建該關節（**本來就該是 0 個**）。
   查全部趟次：29 趟皆 `importer_fix_base`，**開放底盤路徑從未真正執行過**。
2. **執行端的目標開度是 200 mm**：`runner` 只把 `--stroke-m 0.020` 傳給求解端，
   執行端沿用案例 `drawer_open_a_fixed` 的 `target_opening_m = 0.2`。

`carb TaskGroup` 斷言與 `Fatal Python error: Aborted` 是 `SimulationApp.close()`
在我們中止後關閉時的雜訊，**不是原因**。

## 本目錄保留原樣

不覆寫、不重用。修正後的重試另開目錄。
