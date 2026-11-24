# 第一至三章新增圖表之來源

所有新增圖表為研究內容整理或本研究自行繪製。模擬遮罩示例是固定模型之真實離線輸出；甘特圖來自 Isaac 實驗事件，其餘為幾何／方法示意或設計曲線。

## 文獻比較表

引用出版書目沿用 `thesis_references.bib`。方法核對使用原論文或作者版本：

| 引用鍵 | 核對來源與範圍 |
|---|---|
| tang2024gmpc | [作者版本](https://arxiv.org/abs/2403.07317)、[期刊 DOI](https://doi.org/10.1109/LRA.2024.3381088)：輪式底盤幾何追蹤 |
| burgess2023motm | [作者全文](https://arxiv.org/html/2212.06991)：III 架構、共同速度 QP 及 RGB 感知；不把 2024 動態避障擴充回填至此文 |
| mittal2022articulated | [作者全文](https://arxiv.org/html/2103.10534)：V 運動學全身 MPC，VII-C 反應式動態避障、未來位置預測留待後續 |
| minniti2019wholebody | [作者全文](https://arxiv.org/html/1902.10415v2)：全身動力學、III-C 頻率整形、開門與接觸力規劃 |
| du2026wholebody | [期刊頁](https://onlinelibrary.wiley.com/doi/10.1002/rob.70320)：公開摘要記述分層全身 MPC、混合位置／速度命令一致性及動態避障；摘要未提供的視覺等內容標為未明示 |
| sun2026motion | [期刊全文](https://www.mdpi.com/2227-7080/14/4/210)：2.1 分階段協調、SC-VS／CSR-RRT-Connect；第 5 節動態障礙即時反應限制 |

「未明示」與「未評估」不表示方法無此能力。表格中的致動／介面處理亦區分剛體動力學、頻率整形及命令追蹤，不能視為同一種致動器模型。

## 平台與導航

- [UFACTORY 技術規格](https://docs.lite6.ufactory.cc/9.technical_specifications.html)：Lite 6 工作範圍、負載及 LG1000 行程／夾持力。
- `evaluation/models/omni_bot_wholebody_expanded.urdf`：輪半徑 0.05 m、L 0.245 m、手臂安裝、TCP 與相機固定變換。
- `src/my_omnibot_description/urdf/lidar.xacro`、`omni_bot.urdf.xacro`：雷射模擬參數與相對底盤安裝位置。
- `evaluation/isaac_m8_house_d2.py`：相機、雷射參數及 PD 設定；第 97 趟 `result.json` 的 readback 核對實際增益與送偵測頻率。
- `src/my_omnibot_description/config/nav2_localization.yaml`：`nav2_smac_planner::SmacPlanner2D`。
- `src/my_omnibot_description/launch/omni_bot_dynamic.launch.py`、`config/ekf_fusion.yaml`：AMCL 位姿觀測與 EKF 融合里程之分工。

## 圖檔與資料

| 圖檔 | 直接來源／核對紀錄 |
|---|---|
| ch02/robot_coordinate_frames | URDF 網格與固定變換；`{B}` 對應底盤平面基準，非 `base_link` |
| ch02/command_execution_chains | 執行鏈配置與 run97 增益讀回；PD 參數屬模擬設定 |
| ch02/mask_rcnn_class_examples | `make_mask_examples.py`；權重、影像及來源雜湊見 `ch02/mask_examples_evidence.json` |
| ch02/residential_task_state_machine | 現行住宅任務流程；任務層狀態與第三章細部操作轉移分別呈現 |
| ch02/run97_task_gantt | Isaac run97 `world.log`；`ch02/run97_stage_times.json` 保存事件邊界與來源雜湊 |
| ch03/omni_wheel_velocity_feasible_set | 第 3.3.2 節四輪映射矩陣；菱形為 vx–omega 剖面，非固定轉速的 vx–vy 區域 |
| ch03/mask_rcnn_target_pipeline | `house_seg.py`、`seg_vision.py` 與物件幾何偵測流程；包含門把 ROI 例外 |
| ch03/mobile_approach_distance_profile | run101 參考封包與 `door_pick_ctrl.py` 距離規則；`ch03/approach_profile_evidence.json` 保存圓柱階段事件及設計參數 |

資料集表與逐類驗證 IoU：`evaluation/house_seg_ckpt/run2/meta.json`、`best_dev.json`。測試平均 0.754：`evaluation/experiments/20261119_m8_house_d2_unified91_result.json`。測試集建置張數與既有測試摘要分別揭露，未推定其全部逐張評測紀錄。

重製向量圖：`python3 預口試論文/overleaf/figsrc/build_chapter_additions.py`。遮罩重製使用既有深度學習環境 `.venv-dl0/bin/python 預口試論文/overleaf/figsrc/make_mask_examples.py`，僅離線推論，不訓練。腳本依本研究完整儲存庫的模型、權重及實驗檔案讀取資料；Overleaf 排版只需 PDF。
