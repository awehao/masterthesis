# coman_drawer20_main1：**距離節點與安全節點啟動即死，任務未開始**

**不是操作失敗**。三題**都沒有證據**。

| 項目 | 值 |
|---|---|
| **G8a** `/odom` ＋ TF | **生效** —— 求解端 `base=True`（先前 False）、`base_footprint prim` 找到 |
| **G8b** `robot_state_publisher` | 啟動並 `Robot initialized` |
| F1–F5 | 全部通過 |
| `stop_reason` | `sim_limit`；最終開度 0.00 mm |
| 命令 | 未下（`資料不齊（arm=True, base=True, rows=False）`） |
| CPU | 起 54.6 °C，峰值 **71.0 °C**（限 92） |

## 兩個啟動即死的節點（都是本次工作引入的缺陷）

### 1 距離節點：`obstacles` 參數格式錯

```
RCLError: Couldn't parse parameter override rule:
'-p obstacles:=[drawer_front_panel:drawer_body:box:0.575,0.018,0.26:...
```

每條障礙物規格**本身含逗號**（尺寸與座標），runner 卻用
`[$(IFS=,; echo "${OBS[*]}")]` 直接串接 ⇒ ROS 無法分辨規格邊界。
已改為逐項加引號（`["spec1","spec2",…]`，14 個規格）。

**這個缺陷從第一趟就在** —— 先前每一趟距離節點都沒起來，
只是被更早的中止掩蓋（首趟守衛、retry1 NameError、retry2 未 source）。

### 2 安全節點：`_on_src_meta` 方法不存在

```
AttributeError: 'WholeBodySafetyNode' object has no attribute '_on_src_meta'
```

加 `/coman/cmd_meta` 訂閱時，替換字串寫成 `def _on_phase(self, msg):`，
實際簽章是 `def _on_phase(self, msg) -> None:` ⇒ **靜默沒有替換**，
方法從未被加進類別。已補上，並用 AST 確認在 `WholeBodySafetyNode` 內。

## 新增防護

`evaluation/coman_node_smoke.py`：用 **runner 的實際參數**啟動
`arm_link_distance`／`wholebody_safety`／`robot_state_publisher`／
`coman_diag_record`，提早退出即失敗。

**離線套件與入口核對都抓不到這兩類錯** —— 它們只在 `rclpy.init()` 與
`Node.__init__()` 時才會爆。冒煙測試是專為此補的。
