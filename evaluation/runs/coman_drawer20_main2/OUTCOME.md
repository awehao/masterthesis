# coman_drawer20_main2：**啟動競態，求解端在第一個週期 fail closed**

**不是操作失敗**。三題**都沒有證據**。

| 項目 | 值 |
|---|---|
| 節點冒煙測試 | **全部存活**（F8 新增的檢查生效） |
| **距離節點** | **首次真正起來** —— 認證取樣 11 連桿、G2 下界（殼—橫桿 8812 三角形）、六個連桿的必要配對列全部建立 |
| R1.1 十五組 | 兩端載入一致 |
| 求解端 | `中止（fail closed）：QP: 沒有可用的距離列`，`0 週期寫入` |
| `stop_reason` | `sim_limit`；最終開度 0.00 mm |
| CPU | 起 55.9 °C，峰值 **76.375 °C**（限 92） |

## 原因：啟動競態（F9）

父類別 `wholebody_pregrasp.main()` 的等待條件是

```python
while ... and (nd.q_arm is None or nd.base is None or nd.rows is None):
```

只看「**有沒有雲**」，不看「有沒有 `STATUS_OK` 的列」。距離節點在
`1790791668.15` 才就緒，而求解端 `1790791663.16` 就啟動 —— 它拿到的第一批雲
整片 NODATA（TF 尚未暖、抽屜位姿未到），`_constraints` 於是立刻 fail closed。

**修法不放寬 fail closed**：協同求解端在解之前檢查 OK 列數，
**啟動期**（尚未成功解過一次）最多等 `rows_wait_s = 15 s` 且期間不下命令；
**任務中**列消失仍然交由 `_constraints` 中止。父類別是凍結檔，未更動。

## 另一個被順帶發現的缺陷（F10）

```
[isaac_drawer_sim] New publisher discovered on topic '/arm_link_distance/diag',
offering incompatible QoS. ... Last incompatible policy: DURABILITY
```

執行端以 `TRANSIENT_LOCAL` 訂閱 `diag` 與 `cmd_meta`，但發布端是 `VOLATILE`
⇒ **一則都收不到**。本趟 `diag_record.json` 完全沒有產生，
O4 要求的各節點耗時也會全缺。已改用相容 QoS。
