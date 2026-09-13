# 本趟依 v1 判準 **判定未通過**（20/21）

未通過項：`G4b_凍結後 joint2 位移(原始)` = **4.7490 mrad**，門檻 2 mrad。

* **權威記錄是 `run.log`**。`cutoff_check_v1_summary_rebuilt_from_log.txt`
  是**從日誌重建的判定摘要**，**不是**原始判定檔 ——
  原始 `cutoff_check.json` 已被覆蓋且**未能復原**。
* `criteria_used.yaml` 為 **v1** 規格（`wb_cutoff_criteria_v1.yaml`）
* `cutoff_check_v2_retrospective.json` 是**事後以 v2 重算的回溯分析**，
  依 v2 的 `not_retroactive` 條款，**不追認本趟為通過**

## 操作事故記錄

2026-09-13，我在 v1 趟次目錄上直接執行了 v2 檢查器，
覆蓋掉原本的 `cutoff_check.json` 與 `criteria_used.yaml`。
已做的處置：v1 規格重新複製、**從日誌重建**判定摘要、v2 輸出改名為回溯分析。
**原始 v1 `cutoff_check.json` 無法復原**，不宣稱已完整還原。

檢查器已加入兩道防護：
* **不同規格版本** —— 未加 `--retrospective` 即拒絕（exit 4），
  加了則寫到獨立檔名，原規格副本與原判定都不動
* **同一規格版本** —— 未加 `--allow-recheck` 也拒絕覆寫（exit 5），
  避免既有結果被靜默蓋掉
