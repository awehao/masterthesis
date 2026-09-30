# coman_drawer20_main3：**距離節點在第一個 `_tick` 崩潰**

**不是操作失敗**。三題**都沒有證據**。

| 項目 | 值 |
|---|---|
| **F9** 啟動競態 | **生效** —— 求解端印出「等距離列…」而**不再立刻中止** |
| 距離節點 | 認證取樣與 G2 下界都印出來了，**但第一個 `_tick` 就 TypeError 死亡** |
| 求解端 | 等不到 OK 列，政策端 stall 中止；`0 週期寫入` |
| CPU | 起 53.8 °C，峰值 **78.25 °C**（限 92） |

## 原因（F12）

```
File "arm_link_distance.py", line 652, in _tick
    d.data += ([float(_r['lb']), float(_r['ub']), ...])
TypeError: can only extend array with array (not "list")
```

`Float32MultiArray.data` 在 Jazzy 是 `array.array('f')`，**`+= list` 會 TypeError**。
這是我加 tight 配對診斷欄位時寫的。已改為先組 list 再一次指派
（`d.data = list(d.data) + _tail`），並離線重現舊寫法的 TypeError 確認修法。

## 冒煙測試的盲點（F13）

F8 的冒煙測試**沒抓到這個錯**，因為它沒有發 `/clock`，而節點是
`use_sim_time:=true` ⇒ **計時器從未觸發**，`_tick` 裡的錯照不到；
節點「過得了 `__init__`」就被判存活。

已在冒煙測試加上假時鐘（1 kHz 遞增的 `/clock`）。
**用壞版本實測**：新版冒煙測試正確報出
`arm_link_distance **啟動即死**` 並印出該 TypeError。
