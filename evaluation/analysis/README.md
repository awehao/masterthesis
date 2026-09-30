# 分析與繪圖框架（第六次報告用）

**只讀趟次資料，輸出到獨立目錄；不覆寫原始資料、不新增正式通過標準。**
正式資格由 v1 狀態機在趟次中判定，本目錄的分類與統計**不替代它**。

## 檔案

| 檔案 | 用途 |
|---|---|
| `FIELD_INVENTORY.md` | 資料欄位盤點（來源／單位／座標系／時間基準／是否實測／缺值處理）＋ **缺項 G1–G7** |
| `FIXED_BASE_GAP.md` | F 組（固定底盤）對照的實作缺口與**最小修改方案 P1–P6**（本輪未實作） |
| `coman_run_analyze.py` | 單趟分析 |
| `coman_run_plots.py` | 三類繪圖 |
| `test_run_analyze.py` | 離線測試（合成資料） |

## 使用

```bash
# 單趟分析（輸出不得放在趟次目錄內，程式會拒絕）
python3 evaluation/analysis/coman_run_analyze.py \
    --run evaluation/runs/<RUN_ID> \
    --spec evaluation/results/specs/wb_coman_drawer20_criteria_v1.yaml \
    --out evaluation/analysis/out/<RUN_ID>

# 該趟的圖 1／圖 2
python3 evaluation/analysis/coman_run_plots.py \
    --analysis evaluation/analysis/out/<RUN_ID>/analysis.json \
    --run evaluation/runs/<RUN_ID> \
    --out evaluation/analysis/out/<RUN_ID>/fig

# 圖 3：F／M 比較（多趟分析輸出；**失敗趟次也要放進來**）
python3 evaluation/analysis/coman_run_plots.py \
    --compare evaluation/analysis/out/*/analysis.json \
    --out evaluation/analysis/out/_compare

# 合成資料自我驗證（圖面標「合成測試資料，非實驗成果」）
python3 evaluation/analysis/coman_run_plots.py --synthetic \
    --out evaluation/analysis/out/_synthetic_test

# 離線測試
python3 evaluation/analysis/test_run_analyze.py
```

## 分類（**不是通過標準**）

| 值 | 意義 |
|---|---|
| `not_started` | 未進入任務（無趟次資料） |
| `insufficient_data` | 有資料但必要欄位／門檻／時間基準不明 ⇒ **拒絕計算** |
| `task_incomplete` | 進了任務但相位未走完 |
| `complete_data` | 具備完整分析資料 |

## 規則

* 同動**只算「已確認連接且相位為 pull」的交集**；接近／保持／退出另列，不併入占比。
* 只用**實測**運動；命令與設定點不得代替。
* 同動門檻**讀適用規格**；讀不到就 `cannot_determine`，**不硬編**。
* 缺測**不補零**、**不跨越資料空隙**累積連續同動（空隙由規格的
  `continuity.max_gap_s` 切開，本輪讀到 0.15 s）。
* 模擬時間（`sim`）與牆鐘（`wall`）**分開**；耗時一律 `wall`。
* 延遲只標示為「**依值配對的延遲估計**」，並列出候選數與未配對數。
* 每個彙總值都能追到趟次、相位窗、欄位與單位；輸入來源與 sha 記在
  `analysis.json` 的 `inputs`。

## 目前已知的阻點

* **G3**：底盤實測速度未記錄 ⇒ 第二題無法計算。
* **G7**：同動門檻不在任何已核准規格 ⇒ 第二題無門檻可依。
* 兩者都要補，見 `FIXED_BASE_GAP.md` 的 P5／P6。
