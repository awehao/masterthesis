# 單趟分析：drawer_220102_offset20

**分類：`insufficient_data`**  
（分類不是正式通過判定 —— 本檔的分類與統計**不是**正式通過判定；正式資格由 v1 狀態機於趟次中判定）

## 輸入與版本

| 項目 | 值 |
|---|---|
| `run_dir` | evaluation/runs/drawer_220102_offset20 |
| `spec` | /home/howardchen/masterthesis/evaluation/results/specs/wb_coman_drawer20_criteria_v1.yaml |
| `spec_sha256_16` | c9f303a8630d4d0e |
| `preflight` | None |
| `has_outcome_note` | False |
| `drawer_run_sha256_16` | 549dca53c20c084d |

## 趟次

| 項目 | 值 |
|---|---|
| `grasp_model` | fixed_attachment |
| `stop_reason` | sim_limit |
| `sim_time_s` | 55.209998765960336 |
| `wall_s` | 54.70576485496713 |
| `target_opening_used_m` | 0.02 |
| `target_opening_case_m` | 0.2 |
| `temp_max_c` | 86.25 |
| `cpu_limit_c` | 92.0 |
| `monitor_failure` | None |
| `base_mode` | importer_fix_base |
| `asset_sha` | {'urdf': 'fd3252f41bd3d6ef', 'spec': '9cad296fd6e21c43', 'cases': 'bc4ba54f52d51b3a', 'poses': '685ee536c51f898d'} |
| `rtf` | 1.0092 |

## 相位窗（sim）

| 相位 | t0 | t1 | 長度 s | 樣本 | 空隙切分 |
|---|---|---|---|---|---|
| idle | 0.33 | 6.01 | 5.68 | 569 | 0 |
| reach | 6.02 | 11.13 | 5.11 | 512 | 0 |
| approach | 11.14 | 13.73 | 2.59 | 260 | 0 |
| engage | 13.74 | 15.74 | 2.0 | 200 | 0 |
| postengage | 15.75 | 16.73 | 0.98 | 99 | 0 |
| pull | 16.74 | 18.77 | 2.03 | 204 | 0 |
| hold | 18.78 | 20.77 | 1.99 | 200 | 0 |
| release | 20.78 | 21.77 | 0.99 | 99 | 0 |
| retreat | 21.78 | 23.15 | 1.37 | 138 | 0 |
| settle | 23.16 | 55.01 | 31.85 | 3186 | 0 |

## 拉動期間同動（**只含已連接且 pull**）

判定：**cannot_determine**  
原因：適用規格 wb_coman_drawer20_criteria_v1.yaml 未提供同動門檻（simultaneity.base_lin_min_mps／joint_rate_min_rps／min_continuous_s）⇒ **無法判定，不硬編新值**。見 FIELD_INVENTORY G7：同動門檻目前不在任何已核准規格中。  
缺項：趟次無底盤實測速度或可定向位置欄位（只有 base_drift 距離純量與base_dyaw_deg 絕對值）⇒ **無法判定同動**，見 FIELD_INVENTORY G3  

## 缺失原因（**不補零**）

| 代號 | 說明 |
|---|---|
| `no_sim_thresholds` | 適用規格 wb_coman_drawer20_criteria_v1.yaml 未提供同動門檻（simultaneity.base_lin_min_mps／joint_rate_min_rps／min_continuous_s）⇒ **無法判定，不硬編新值**。見 FIELD_INVENTORY G7：同動門檻目前不在任何已核准規格中。 |
| `no_base_velocity` | 趟次無底盤實測速度或可定向位置欄位（只有 base_drift 距離純量與base_dyaw_deg 絕對值）⇒ **無法判定同動**，見 FIELD_INVENTORY G3 |
| `no_diag_record` | diag_record.json 不存在 ⇒ 無節點耗時 |
| `no_solver_out` | solver_out.json 不存在或無 log ⇒ 無求解耗時 |
| `no_e2e_log` | coman_e2e_log 不存在 ⇒ 無延遲估計 |

## 三題

| 問題 | 本趟 |
|---|---|
| 完整操作 | 見 classification 與 phase_windows；**正式通過由 v1 狀態機判定，本檔不替代** |
| 拉動期間同動 | cannot_determine |
| 執行期時效 | cost_wall 與 latency_value_paired_estimate；各節點分開，不相加當作端到端驗收 |
