# 封存導航版本（2026-11-27 起：住宅任務版）

Howard 11/27 指定：封存導航改成**住宅任務實際在用的這一版**，N1 導航比較的「本研究方法」也是這版。
舊的封存說法（只有 `gmpc.py`／`gmpc_node.py`，9 月 `chain_final120.sh` 的鎖定參數）不再代表目前方法，留作歷史。

## 這一版是什麼

| 層 | 檔案（本資料夾的捷徑） | 說明 |
|---|---|---|
| 啟動整套 | `omni_bot_dynamic.launch.py` | 規劃器、GMPC、追蹤器、Shield、GUARD、參數 |
| 導航控制器 | `gmpc_task_node.py` | 任務擴充節點（`GMPC_TASK=1`），繼承 `gmpc_node.py` |
| 求解核心 | `gmpc_wb.py` | 共同核心（`GMPC_WB_CORE=1`，預設），底盤問題與封存 `gmpc.py` 逐位元相同 |
| 底層類別（不改） | `gmpc.py`、`gmpc_node.py` | 仍被上面兩個繼承／沿用 |
| 會車層 | `yield_side.py` | 8 s 等速預測選「減速 × 側移」、鎖邊（`D2_YIELD_SIDE=1 YS_PRED=1`） |
| 感知 | `scan_obstacle_tracker.py` | 雷射 → 動態障礙物（位置、速度、半徑） |
| 安全層 | `scan_safety_shield.py`、`wheel_limit_guard.py` | 原始雷射防護、輪級限制（四個比較方法共用） |
| 路徑 | `goal_to_plan_relay.py` | 目標 → 全域規劃（1 Hz 重規劃） |
| 基線（比較用） | `omni_bot_baseline.launch.py` | MPPI／DWB／RPP，同一套 Shield＋GUARD |

## 開關（照這樣開才是這一版）

住宅任務（`evaluation/run_d2_isaac.sh`）：

```
HEADING=1 GMPC_TASK=1 GUARD=1 SHIELD=1          # 導航
D2_YIELD_SIDE=1 YS_PRED=1                       # 會車層（mission/config/task.env）
```

純導航比較（N1，`evaluation/run_n1_batch.sh`／`run_n1_par.sh`）另加：

```
NAV_YIELD=1 M8_MAP=<布局>    # 會車層接在導航控制器的參考路線（住宅任務是接在全身控制器的參考路線，做法相同）
NAV_VMAX=0.26                # 四個方法平移速度上限一致（各軸＋GUARD 合成上限）
```

CBF／規劃的 6 個參數（`PLANNER_SCAN`、`CBF_ALPHA`、`CBF_SAFE_MARGIN`、`CBF_PRUNE_RANGE`、`CBF_FAR_STRIDE`、`PLAN_BLEND`）**不設**＝程式預設，與住宅任務相同。
`NAV_YIELD`、`NAV_VMAX`、GUARD 的 `v_xy_max` 都是不設就與原行為相同。

## 檔案雜湊（git 4e40d53e5；之後改了任何一個，這一版就不再成立，要更新本檔）

```
2a1e2148b0d4bc22  src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc.py
887bbdd591b40ed1  src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_node.py
f31ef3ffcbed0e56  src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_task_node.py
e6c8e82a668f28b9  src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_wb.py
e6db2380a74f041c  src/ammr_wholebody_mpc/ammr_wholebody_mpc/scan_obstacle_tracker.py
075746e562acf529  src/ammr_wholebody_mpc/ammr_wholebody_mpc/scan_safety_shield.py
11fc2ffc37290adb  src/ammr_wholebody_mpc/ammr_wholebody_mpc/wheel_limit_guard.py
7f52031211741748  src/ammr_wholebody_mpc/ammr_wholebody_mpc/goal_to_plan_relay.py
12ab03c40313f4c6  evaluation/yield_side.py
e54c24c2b0931538  src/my_omnibot_description/launch/omni_bot_dynamic.launch.py
3900dae17c82cfed  src/ammr_wholebody_mpc/config/gmpc_params.yaml
898dce238b9e237a  src/my_omnibot_description/config/nav2_localization.yaml
```

核對：`cd ~/masterthesis && sha256sum <檔案> | cut -c1-16`
