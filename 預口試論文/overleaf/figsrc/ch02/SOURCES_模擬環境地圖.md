# 室內模擬環境地圖

整理日期：2026-11-21。用於第二章 2.3.2「Isaac Sim 物理模擬與場景建置」。

圖檔為依既有 `m8_house_r2` 登錄資料重繪之原生向量平面圖，非 AI 生成圖片或模擬器截圖；沒有改寫場景幾何或執行模擬。

- 登錄檔：`evaluation/experiments/20261118_m8_house_r2_scene.json`。
- 場景牆面與家具：`src/ammr_bringup/worlds/mission_house_m8b.sdf`；輸出前核對其 SHA-256 與登錄資料一致。
- 門資產：`src/my_omnibot_description/config/door_local_v3.yaml`；另補入由門資產生成、未寫在 SDF 的兩段門牆。
- 抽屜櫃：採登錄檔的世界座標外框，未假設新的櫃體尺寸。
- 圖中文字：`overleaf/fonts/kaiu.ttf`，標楷體（DFKai-SB）；PDF 嵌入字型，SVG 文字輸出為向量路徑。

圖中呈現室內範圍 19.50 m × 10.00 m、門洞名目淨寬 1.00 m，以及客廳與右側房間間的 2.00 m 開口。此尺寸為模擬場景的設計值。平面展示座標為 `(u,v)=(世界 y,-世界 x)`，使臥室、客廳與右側房間由左至右排列；圖內另有世界座標方向示意。

門扇實線表示關閉配置，虛線為建築平面圖慣用的 90 度開啟示意，並非量測軌跡、實驗結果或關節角度上限。茶几與抽屜櫃以相同色彩標示，代表操作設備。

本圖介紹共用空間配置，不加入各次實驗的行人路線、未知障礙配置、圓柱位置、機器人路徑或起終點座標。這些條件依第四章個別實驗提供，不能由本圖推定任務已完成。

本輪找到的 `第八次進度報告素材/圖表/m8_house_scene/r2/m8_house_top.png`，其 metadata 仍登錄為較小的 `m8_house_r1`；因此未將該截圖誤標為大房子 r2。

## 檔案

- `simulation_environment_map.pdf`：Overleaf 插圖用向量檔。
- `simulation_environment_map.png`：360 dpi 預覽。
- `simulation_environment_map.svg`：可編輯向量檔。
- `draw_simulation_environment_map.py`：重繪程式，只讀取登錄資料。
- `simulation_environment_map_metadata.json`：來源、尺寸、字型與輸出檔雜湊。
- `overleaf/第二章_模擬環境地圖插入.tex`：正文導引、圖說與標籤。

重繪方式：在工作區執行 `python3 預口試論文/overleaf/figsrc/ch02/draw_simulation_environment_map.py`。此為本研究自行重繪的場景圖，圖說不需新增外部文獻。
