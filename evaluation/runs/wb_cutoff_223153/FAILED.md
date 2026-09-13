# 本趟依 v1 判準 **判定未通過**（20/21）

未通過項：`G4b_凍結後 joint2 位移(原始)` = **4.7490 mrad**，門檻 2 mrad。

* 權威記錄：`run.log` 內的 v1 判定輸出，另抄一份於 `cutoff_check_v1_verdict.txt`
* `criteria_used.yaml` 為 **v1** 規格（`wb_cutoff_criteria_v1.yaml`）
* `cutoff_check_v2_retrospective.json` 是**事後以 v2 重算的回溯分析**，
  依 v2 的 `not_retroactive` 條款，**不追認本趟為通過**

## 操作事故記錄

2026-09-13，我在 v1 趟次目錄上直接執行了 v2 檢查器，
覆蓋掉原本的 `cutoff_check.json` 與 `criteria_used.yaml`。
已復原：v1 規格重新複製、v1 判定自 `run.log` 抄出、v2 輸出改名為回溯分析。
檢查器已加入防護：對既有判定檔為不同規格版本者，必須明確指定
`--retrospective` 才會寫入，且寫到獨立檔名。
