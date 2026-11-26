# 程式依功能分類（符號連結）

這裡每個檔案都是指向實際檔案的符號連結（實際位置在 `evaluation/`、`src/`），在這裡打開、修改就是改真正的檔案。
實際檔案沒有搬動，所以執行路徑、Isaac 凍結雜湊、歷史紀錄都不受影響。★＝目前主線最常改的檔案。

| 資料夾 | 功能 | 主要檔案 |
|---|---|---|
| `supervisor/` | 任務流程：做什麼、什麼順序、何時切換、送導航目標與控制封包 | ★`b_d2_node.py` |
| `control/` | 全身控制（W-GMPC 單一控制器）：各任務段的參考與限制 | ★`gmpc_wb.py`（求解核心）、★`door_pick_ctrl.py`（茶几、放盒、返回）、★`door_drawer_ctrl.py`（抽屜）、★`door_pass_ctrl.py`（穿門、收臂）、`door_close_ctrl.py`（回程關門）、`door_motm_grasp_ctrl.py`（導航參考視窗） |
| `avoidance/` | 行人會車、Shield 例外 | ★`yield_side.py`（預測選動作）、`door_leaf_scan_mask.py` |
| `planning/` | 抓取站位、可達性、門把接近規劃 | ★`cyl_reach_map.py`（茶几）、`drawer_reach.py`、`drawer_motm.py`（抽屜） |
| `perception/` | 視覺偵測 | `cyl_detect.py`、`handle_detect.py`、`marker_detect.py`、`door_leaf_detect.py`、`d1_handle_detect_v2.py`（門把）、`isaac_m8_house_seg_collect.py`（Mask R-CNN 資料） |
| `collision/` | 手臂／底盤淨空、自碰撞、停止掃掠 | `door_clear_door_fast.py`、`door_clear_fast.py`、`drawer_scene.py` |
| `scene/` | 場景登錄、行人與未知箱、門／抽屜／筆／收納盒資產 | ★`20261121_m8_house_r2w3_scene.json`、★`mission_obstacles_m8_house_r2w3.json`、`walkers.py` |
| `sim/` | Isaac 執行器、替身、控制工作程序、啟動鏈 | ★`isaac_m8_house_d2.py`、`stub_isaac_d2.py`、★`world_d2_node.py`、`run_d2_isaac.sh` |
| `nav/` | **封存導航＝住宅任務版**（11/27 起；N1 比較也用這版；開關與檔案雜湊見 `nav/NAV_VERSION.md`） | `omni_bot_dynamic.launch.py`、`gmpc_task_node.py`、`gmpc_wb.py`、`gmpc.py`／`gmpc_node.py`（底層不改）、`yield_side.py`（會車層）、`scan_obstacle_tracker.py`、`scan_safety_shield.py`、`wheel_limit_guard.py`、`goal_to_plan_relay.py`、`omni_bot_baseline.launch.py`（基線） |
| `legacy_bottle/` | 瓶子線舊模組，目前仍被主線間接引用（勿刪） | — |

清單由入口程式（監督、world、替身、Isaac 執行器、重播、資料收集）的 import 追出來，共 121 個現役模組；`evaluation/` 裡其餘約 1,900 個 .py 是歷史實驗，主線不會用到。
