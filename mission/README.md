# 住宅任務（m8 house）操作入口

開門 → 茶几圓柱（移動中抓取）→ 放車尾盒 → 開抽屜取筆 → 關抽屜 → 回程推門穿過再關門 → 回起點，全程四位行人。

## 怎麼啟動

所有指令在專案根目錄（`~/masterthesis`）下執行。單趟臨時改設定就在最後加 `KEY=VAL`。

| 要做什麼 | 指令 | 大約時間 |
|---|---|---|
| 替身跑完整任務（不開 Isaac，感測為合成） | `mission/launch/run_stub.sh s300d` | 20 分 |
| 替身換圓柱位置、關掉預測會車 | `mission/launch/run_stub.sh s301d CYL_SEED=2 YS_PRED=0` | 20 分 |
| 比較替身結果（成敗、時間、離行人距離） | `mission/launch/compare_walkers.sh s300d s301d` | 秒 |
| Isaac 實跑：先登錄（凍結程式雜湊） | `mission/launch/register_isaac.sh 88 s300d "說明"` | 秒 |
| Isaac 實跑 | `mission/launch/run_isaac.sh 88` | 20–25 分 |
| 出俯視＋跟車影片 | `mission/launch/make_videos.sh 88 成功 "描述"` | 15–20 分 |
| 收掉所有程序（中止、卡住時） | `mission/launch/stop_all.sh` | 秒 |
| Mask R-CNN 收訓練資料（開 Isaac） | `mission/launch/seg_collect.sh evaluation/house_seg_data/v1 53,54,55` | 每趟約 1 分 |
| Mask R-CNN 訓練（GPU，不和 Isaac 同時） | `mission/launch/seg_train.sh evaluation/house_seg_data/v1 evaluation/house_seg_ckpt/run1` | 約 30 分 |
| Isaac 改用 Mask R-CNN 視覺 | `mission/launch/run_isaac.sh 89 D2_SEG=1` | 20–25 分 |

規則：一次只能一個 Isaac（含重播）。`run_isaac.sh` 會等 CPU ≤55 °C 才起跑，92 °C 熱中止。改過程式後要重新 `register_isaac.sh`，否則起跑前雜湊核對會拒跑。

進度與結果：
- 替身：`evaluation/results/motm_direct/d2/<TAG>.sup.log`（監督事件）、`<TAG>_stub.json`（成敗）、`<TAG>_d2w.json`（控制事件）
- Isaac：`mission/logs/isaac_<N>.out`、`evaluation/runs/m8house_d2_unified_<N>/result.json`

## 設定檔（改行為先看這裡）

| 檔案 | 內容 |
|---|---|
| `mission/config/task.env` | 任務共用：場景、任務開關與順序、圓柱位置、移動中抓取、穿門、會車、即時化 |
| `mission/config/stub.env` | 替身專用：起始姿態、模擬速度、控制期限 |
| `mission/config/isaac.env` | Isaac 專用：模擬速度上限、模擬秒上限、控制期限 |

## 功能地圖：要改什麼去哪裡

### 場景
| 功能 | 檔案 |
|---|---|
| 牆、家具、門、抽屜櫃、筆的位置 | `evaluation/experiments/20261121_m8_house_r2w3_scene.json` |
| 行人路線、未知箱 | `evaluation/results/mission/mission_obstacles_m8_house_r2w3.json` |
| 行人怎麼走（來回、速度比例） | `evaluation/walkers.py` |
| 門、抽屜、筆、收納盒的 3D 資產 | `evaluation/door_asset_v3.py`、`drawer_asset.py`＋`house_assets.py`、`marker_proxy_asset.py`、`rear_bin.py` |
| 導航地圖 | `src/ammr_bringup/maps/mission_house_m8b.*` |

### 任務流程（做什麼、什麼順序、什麼時候切換）
| 功能 | 檔案 |
|---|---|
| 任務狀態機：開門→茶几→抽屜→回程關門→返回；導航目標、送封包時機 | `evaluation/motm_b/b_d2_node.py` |
| 回程關門流程（找門扇、共轉關門） | `b_d2_node.py` 的 `door_tick` |
| 控制工作程序、交權、控制期限、停止 | `evaluation/motm_dry/world_d2_node.py` |

### 導航（封存版＝住宅任務版，11/27 起；捷徑在 `mission/code/nav/`，版本說明 `mission/code/nav/NAV_VERSION.md`）
| 功能 | 檔案 |
|---|---|
| 導航整套啟動（規劃、GMPC、Shield、GUARD、追蹤器、參數） | `src/my_omnibot_description/launch/omni_bot_dynamic.launch.py` |
| 導航控制器（任務擴充節點，`GMPC_TASK=1`） | `src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_task_node.py` |
| 求解核心（共同核心） | `src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_wb.py` |
| 底層類別（被繼承，不改） | `src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc.py`、`gmpc_node.py` |
| 會車層 | `evaluation/yield_side.py`（`D2_YIELD_SIDE=1 YS_PRED=1`） |
| 導航比較（N1）跑法 | `evaluation/run_n1_par.sh`（平行車道）、`run_n1_batch.sh`；情境 `src/ammr_bringup/scripts/make_n1_scenario.py` |

### 全身控制（W-GMPC 單一控制器）
| 功能 | 檔案 |
|---|---|
| 求解核心（底盤＋手臂、CBF、限制） | `src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_wb.py` |
| 導航參考視窗、移動中抓取基底 | `evaluation/door_motm_grasp_ctrl.py` |
| 茶几抓取、放盒、返回 | `evaluation/door_pick_ctrl.py` |
| 抽屜：開、取筆、關 | `evaluation/door_drawer_ctrl.py` |
| 穿門、收臂 | `evaluation/door_pass_ctrl.py` |
| 回程關門 | `evaluation/door_close_ctrl.py` |
| 停止軌跡核對 | `evaluation/door_cstop.py` |

### 行人會車
| 功能 | 檔案 |
|---|---|
| 會車層（預測選動作；舊規則保留） | `evaluation/yield_side.py`（參數在檔頭 `YS_*`，可用環境變數覆蓋） |

### 抓取選位、可達性
| 功能 | 檔案 |
|---|---|
| 茶几圓柱抓取站位 | `evaluation/cyl_reach_map.py` |
| 抽屜把手、取筆站位 | `evaluation/drawer_reach.py`、`drawer_motm.py` |

### 視覺（目前為深度幾何法；Mask R-CNN 進行中）
| 物件 | 檔案 |
|---|---|
| 門把 | `evaluation/d1_handle_detect_v2.py`、`door_far_link.py` |
| 茶几圓柱 | `evaluation/cyl_detect.py` |
| 抽屜把手 | `evaluation/handle_detect.py` |
| 筆 | `evaluation/marker_detect.py` |
| 門扇角度 | `evaluation/door_leaf_detect.py` |
| Mask R-CNN 訓練資料收集 | `evaluation/isaac_m8_house_seg_collect.py` |
| Mask R-CNN 模型（訓練、評估、推論） | `evaluation/house_seg.py`；推論服務 `house_seg_server.py`（.venv-dl0 子程序） |
| 遮罩＋深度 → 物體 3D 位置（`D2_SEG=1` 時取代上面各偵測器的搜尋範圍） | `evaluation/seg_vision.py` |

### 碰撞、淨空
| 功能 | 檔案 |
|---|---|
| 手臂、底盤對家具／門的淨空 | `evaluation/door_clear_door_fast.py`、`door_clear_fast.py` |
| 手指碰撞 | `evaluation/finger_collision.py` |

### 模擬
| 功能 | 檔案 |
|---|---|
| Isaac 場景建構、物理、手腕相機、視覺工作程序 | `evaluation/isaac_m8_house_d2.py` |
| Isaac 啟動鏈、熱保護 | `evaluation/run_d2_isaac.sh`、`run_guarded_v3.sh` |
| 替身（合成觀測、行人碰撞判定） | `evaluation/motm_dry/stub_isaac_d2.py`、`run_d2_stub.sh` |
| 影片重播 | `evaluation/isaac_m8_house_d2_replay.py` |

### 工具（`mission/tools/`）
`tick_budget.py <d2w.npz> [50]`：各階段控制拍計算時間與超過預算的拍數（即時性檢查）；`nav_watchdog.sh`：替身 Nav2 啟動卡住自動重跑一次。
`wait_clean.sh`（起跑前確認沒有殘留程序）、`kill_all.sh`（收整個行程群組）、`make_registration.py`（登錄＋凍結）、`compare_walkers.py`、`replay_walkers.py`（重播行人與機器人距離）。

## 歸檔

`evaluation/` 裡主線用不到的 1,158 個歷史 .py 已移到 `evaluation/archive/`（只搬不刪，說明見該資料夾 README）。
