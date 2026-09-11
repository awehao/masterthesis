# v2_on_095657：啟動流程失敗，**不是** ON 導航結果

中止點：就緒檢查，`exit=2`。兩項未過：

* `!! clock 在前進 逾時 60s`
* `!! 靜止檢查未通過`（`stillness.log`：`/odom 無資料`）

## 兩項都是誤判

同一趟的 bag（錄了 116.04 s）顯示這些主題一直在發布：

| topic | 筆數 |
|---|---|
| `/clock` | 10391 |
| `/odom` | 3596 |
| `/scan` | 1078 |
| `/cmd_vel` | 2121 |
| `/wheel_guard/status` | 2121 |
| `/cmd_vel_smoothed` | 2279 |
| `/gmpc/diag_v2` | 2108 |

`heading_enable = True` 也已讀回確認，ON 設定正確。溫度峰值 85 °C。

## 成因與修正

兩個閘都在 DDS 訂閱配對完成前就用完了固定的觀察視窗：

1. **`clock_moved()`**（`run_bigarena_isaac.sh`）每次嘗試開兩個
   `ros2 topic echo --once`，各自是新行程、各自付一次配對，而 timeout 只有 3 s。
   導航鏈十幾個節點起來之後配對常超過 3 s，於是不斷重試把 60 s 預算耗光。
   這個閘本來就在邊緣：`v2_off_093455` 用掉 **56 s**（上限 60）才通過。
   → 改用 `evaluation/clock_advancing.py`：單一行程，配對只付一次，
   再量測模擬時間是否前進；閘的預算放寬到 120 s。

2. **`stillness_check.py`** 建立訂閱後立刻進入 3 s 固定視窗。
   → 加入配對等待（上限 `--discover`，預設 15 s），等到每個主題至少一則
   再清空重新計時。

兩者都把「配對未完成」與「真的沒有資料」分開列印，不再混為一談。

同一類問題稍早已在 `case_start_check.py` 修過（見 `v2_on_095217/ABORTED.md`）。
