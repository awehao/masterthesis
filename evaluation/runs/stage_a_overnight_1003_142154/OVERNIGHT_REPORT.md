# 階段 A 夜間工作計畫 — 隔日交付

**工作階段** `stage_a_overnight_1003_142154`
**用時** 601 s（0.17 h）　**Isaac 啟動** 6 次
**錄影分支** 相機診斷 `pass`、錄影趟次 `pass`

> 未遇阻滯。

## 1 入口核對

判定 **pass**。
- `threshold`：{'file': 'evaluation/runs/wgmpc_stage_a_baseline_long120_020143/drawer_threshold.json', 'opening_tol_m': 1e-09, 'contact_tol_n': 1e-09, 'fingerprint_matches_asset_and_pose': True}
- `target`：{'tcp_world': [0.0, 1.0597, 0.55], 'backoff_m': 0.12, 'R_des_rowmajor': [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]}
- `margin_and_mode`：{'solver': 0.05, 'endpoint': 0.05, 'mode': 'solver_drawer'}
- `distance_node_params`：{'n_obstacles': 14, 'n_approved': 14, 'pair_rows': ['link4:*', 'link5:*', 'link6:*', 'uflite_finger1:*', 'uflite_finger2:*', 'uflite_gripper_link:*'], 'scene_truth': True}
- `config_vs_runner`：{'n_compared': 17, 'all_match': True}
- `handover_config`：{'quiet_s': 0.4, 'max_cmd_age_s': 0.2, 'u_prev': '取自實際套用回報；**不傳 --assume-initial-rest**'}
- `files`：{'all_present': True, 'n_checked': 11}
- `installed_matches_src`：{'n_modules': 29, 'all_match': True, 'scene_truth_declared_in_installed': True}
- `scene_ready_marker`：{'ready_marker': '[wb] articulation root', 'false_match_removed': True}

靜止基線**未重跑**（已完成，門檻沿用 `long120_020143/drawer_threshold.json`）。

## 2 各趟結果

### `stage_a_overnight_1003_142154_r1`

判定 **pass**　首次失敗：（無）

- 資料路徑 `evaluation/runs/stage_a_overnight_1003_142154_r1/`（config `run_config.json`、程式版本 `code_versions.txt`）
- 前置調姿：{'ok': True, 'n_commands': 123, 'q_goal': [0.0, 0.0, 1.4372784010000001, 0.0, 0.0, 0.0], 'last_cmd_sim_t': 36.089999193000004, 'duration_s': 4.609982011003003}
- 交棒：{'ok': True, 'quiet_s': 6.469999856000001, 'setpoint_meas_max_dev_rad': 0.00235715258281379, 'u_prev': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'u_prev_source': 'applied_report', 'exec_mode': 0, 'q_meas': [3.8229700294323266e-06, 3.412416481296532e-05, 1.4371657371520996, 1.9113706002826802e-05, 3.3954196965169103e-07, 6.482455461309655e-09], 'setpoint': [-9.725420449935604e-05, -0.0023230284180008245, 1.438584245931355, -0.001559916778727961, -1.148035088351126e-05, 2.8183817151636626e-08]}
- 到達：{'node_reported_reached_held': True, 'reached_held_measured': True, 'longest_in_tol_s': 4.580000000000005, 't_first_in_tol_s': 58.66, 't_reach_s': 58.66, 'err_p_min_mm': 1.39720399369601, 'err_r_min_rad': 0.0030000011249887615, 'err_p_final_mm': 1.4268321555109262, 'err_r_final_rad': 0.0030000011249887615, 'tol': {'pos_m': 0.005, 'rot_rad': 0.02, 'hold_s': 2.0}, 'n_samples_in_tol': 459, 'basis': '由 log 的 tcp_x/y/z 與 tcp_r** 對 target.json 重算；姿態用測地角。**不照抄節點回報的 reached_held**'}
- 命令鏈：{'fail': None, 'received': 755, 'rejected': 0, 'frozen_steps': 29, 'stop_reason': 'stop_request'}
- 有效餘裕：{'setpoint_min_effective_slack_rad': 0.009973, 'setpoint_min_per_joint': [1.686298, 0.784176, 0.009973, 1.924186, 0.954885, 1.133178], 'measured_min_effective_slack_rad': 0.009871999999999999, 'measured_min_per_joint': [1.692792, 0.77927, 0.009872, 1.944274, 0.976122, 1.133656], 'n_setpoint_samples': 3818, 'n_measured_samples': 6322, 'note': '執行端那一層只保護**設定點**；實測角在一階遲滯下仍可能落在界外，故實測角另行核實、不列入成功判準'}
- 封存：ok=True
- 目標：{'tcp_world': [0.0, 1.0596999999999999, 0.55], 'source': '交棒時刻的抽屜位姿真值（0.0,1.45，開度 0.0）', 'stamp_sim_t': 42.559999049000005}
- 抽屜：{'verdict': 'displacement_within_baseline|no_contact_evidence', 'opening_max_abs_m': 0.0, 'opening_tol_m': 1e-09, 'over_baseline': False, 'contact_verdict': 'no_contact_evidence', 'contact_fmag_max_n': 0.0}
- 缺項：[]

- 實測同動：{'window': '交棒後', 'n_steps': 2068, 'base_moving_frac': 0.7978723404255319, 'arm_moving_frac': 0.6247582205029013, 'both_moving_frac': 0.6237911025145068, 'both_moving_frac_of_any': 0.7808716707021792, 'base_speed_p50_mm_s': 28.98168679337716, 'arm_rate_p50_mrad_s': 84.6999999999854, 'thresholds': {'base_m_s': 0.0005, 'arm_rad_s': 0.005}, 'basis': '由**實測位姿與實測關節角**差分算，不用回報的 twist（靜止後會停更）'}
- 保持窗振動：{'window_s': [58.66, 63.24], 'n': 458, 'tcp_speed_p50_mm_s': 0.10000000000002097, 'tcp_speed_p95_mm_s': 5.685812561051372, 'tcp_speed_max_mm_s': 32.64015931332831, 'basis': 'TCP 世界位置差分除以**實際**取樣間隔', 'arm_rate_sign_flip_frac': 0.48253275109170307}

### `stage_a_overnight_1003_142154_r2`

判定 **pass**　首次失敗：（無）

- 資料路徑 `evaluation/runs/stage_a_overnight_1003_142154_r2/`（config `run_config.json`、程式版本 `code_versions.txt`）
- 前置調姿：{'ok': True, 'n_commands': 121, 'q_goal': [0.0, 0.0, 1.4372784010000001, 0.0, 0.0, 0.0], 'last_cmd_sim_t': 36.759999178, 'duration_s': 4.6099818992442945}
- 交棒：{'ok': True, 'quiet_s': 5.809999869999999, 'setpoint_meas_max_dev_rad': 0.0023589553953279278, 'u_prev': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'u_prev_source': 'applied_report', 'exec_mode': 0, 'q_meas': [4.605802132573444e-06, 4.41462907474488e-05, 1.4397008419036865, 2.5082994397962466e-05, 4.4870512283523567e-07, 1.022912776704743e-08], 'setpoint': [-9.676507730470603e-05, -0.002314809104580479, 1.441122075515616, -0.0015544077510545947, -1.1365400437347629e-05, 3.156097564145003e-08]}
- 到達：{'node_reported_reached_held': True, 'reached_held_measured': True, 'longest_in_tol_s': 3.6699999999999946, 't_first_in_tol_s': 58.2, 't_reach_s': 58.2, 'err_p_min_mm': 1.4190757555536608, 'err_r_min_rad': 0.0030000011249887615, 'err_p_final_mm': 1.4394943556680817, 'err_r_final_rad': 0.0030000011249887615, 'tol': {'pos_m': 0.005, 'rot_rad': 0.02, 'hold_s': 2.0}, 'n_samples_in_tol': 368, 'basis': '由 log 的 tcp_x/y/z 與 tcp_r** 對 target.json 重算；姿態用測地角。**不照抄節點回報的 reached_held**'}
- 命令鏈：{'fail': None, 'received': 691, 'rejected': 0, 'frozen_steps': 35, 'stop_reason': 'stop_request'}
- 有效餘裕：{'setpoint_min_effective_slack_rad': 0.009973, 'setpoint_min_per_joint': [1.63324, 0.784327, 0.009973, 2.044167, 1.132439, 1.103745], 'measured_min_effective_slack_rad': 0.009871999999999999, 'measured_min_per_joint': [1.63811, 0.779402, 0.009872, 2.073397, 1.167027, 1.103972], 'n_setpoint_samples': 3484, 'n_measured_samples': 6185, 'note': '執行端那一層只保護**設定點**；實測角在一階遲滯下仍可能落在界外，故實測角另行核實、不列入成功判準'}
- 封存：ok=True
- 目標：{'tcp_world': [0.0, 1.0596999999999999, 0.55], 'source': '交棒時刻的抽屜位姿真值（0.0,1.45，開度 0.0）', 'stamp_sim_t': 42.569999048}
- 抽屜：{'verdict': 'displacement_within_baseline|no_contact_evidence', 'opening_max_abs_m': 0.0, 'opening_tol_m': 1e-09, 'over_baseline': False, 'contact_verdict': 'no_contact_evidence', 'contact_fmag_max_n': 0.0}
- 缺項：[]

- 實測同動：{'window': '交棒後', 'n_steps': 1930, 'base_moving_frac': 0.8321243523316062, 'arm_moving_frac': 0.6569948186528497, 'both_moving_frac': 0.6569948186528497, 'both_moving_frac_of_any': 0.7895392278953923, 'base_speed_p50_mm_s': 30.672869644578448, 'arm_rate_p50_mrad_s': 97.64999999998658, 'thresholds': {'base_m_s': 0.0005, 'arm_rad_s': 0.005}, 'basis': '由**實測位姿與實測關節角**差分算，不用回報的 twist（靜止後會停更）'}
- 保持窗振動：{'window_s': [58.2, 61.87], 'n': 367, 'tcp_speed_p50_mm_s': 0.10000000000289547, 'tcp_speed_p95_mm_s': 8.039396408276538, 'tcp_speed_max_mm_s': 29.36085829809801, 'basis': 'TCP 世界位置差分除以**實際**取樣間隔', 'arm_rate_sign_flip_frac': 0.4055404178019982}

### `stage_a_overnight_1003_142154_r3`

判定 **pass**　首次失敗：（無）

- 資料路徑 `evaluation/runs/stage_a_overnight_1003_142154_r3/`（config `run_config.json`、程式版本 `code_versions.txt`）
- 前置調姿：{'ok': True, 'n_commands': 122, 'q_goal': [0.0, 0.0, 1.4372784010000001, 0.0, 0.0, 0.0], 'last_cmd_sim_t': 35.629999204, 'duration_s': 4.609982041270987}
- 交棒：{'ok': True, 'quiet_s': 3.9099999120000035, 'setpoint_meas_max_dev_rad': 0.0023579675233992557, 'u_prev': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'u_prev_source': 'applied_report', 'exec_mode': 0, 'q_meas': [4.184909812465776e-06, 3.8587782910326496e-05, 1.4379152059555054, 2.1851350538781844e-05, 3.763944391721452e-07, 7.62221219474668e-09], 'setpoint': [-9.697946954829488e-05, -0.002319379740488929, 1.4393348260449106, -0.0015574941033936016, -1.1439356369125246e-05, 2.9472433184745618e-08]}
- 到達：{'node_reported_reached_held': True, 'reached_held_measured': True, 'longest_in_tol_s': 3.760000000000005, 't_first_in_tol_s': 53.66, 't_reach_s': 53.66, 'err_p_min_mm': 1.4359470742336264, 'err_r_min_rad': 0.003162278977795912, 'err_p_final_mm': 1.4501065478094943, 'err_r_final_rad': 0.003162278977795912, 'tol': {'pos_m': 0.005, 'rot_rad': 0.02, 'hold_s': 2.0}, 'n_samples_in_tol': 377, 'basis': '由 log 的 tcp_x/y/z 與 tcp_r** 對 target.json 重算；姿態用測地角。**不照抄節點回報的 reached_held**'}
- 命令鏈：{'fail': None, 'received': 651, 'rejected': 0, 'frozen_steps': 0, 'stop_reason': 'stop_request'}
- 有效餘裕：{'setpoint_min_effective_slack_rad': 0.009973, 'setpoint_min_per_joint': [1.622752, 0.784302, 0.009973, 2.034214, 1.155296, 1.091471], 'measured_min_effective_slack_rad': 0.009871999999999999, 'measured_min_per_joint': [1.628279, 0.77937, 0.009872, 2.068786, 1.180867, 1.091776], 'n_setpoint_samples': 3251, 'n_measured_samples': 5740, 'note': '執行端那一層只保護**設定點**；實測角在一階遲滯下仍可能落在界外，故實測角另行核實、不列入成功判準'}
- 封存：ok=True
- 目標：{'tcp_world': [0.0, 1.0596999999999999, 0.55], 'source': '交棒時刻的抽屜位姿真值（0.0,1.45，開度 0.0）', 'stamp_sim_t': 39.539999116000004}
- 抽屜：{'verdict': 'displacement_within_baseline|no_contact_evidence', 'opening_max_abs_m': 0.0, 'opening_tol_m': 1e-09, 'over_baseline': False, 'contact_verdict': 'no_contact_evidence', 'contact_fmag_max_n': 0.0}
- 缺項：[]

- 實測同動：{'window': '交棒後', 'n_steps': 1788, 'base_moving_frac': 0.8176733780760627, 'arm_moving_frac': 0.7069351230425056, 'both_moving_frac': 0.7069351230425056, 'both_moving_frac_of_any': 0.8645690834473324, 'base_speed_p50_mm_s': 32.55806974654411, 'arm_rate_p50_mrad_s': 106.74999999998319, 'thresholds': {'base_m_s': 0.0005, 'arm_rad_s': 0.005}, 'basis': '由**實測位姿與實測關節角**差分算，不用回報的 twist（靜止後會停更）'}
- 保持窗振動：{'window_s': [53.66, 57.42], 'n': 376, 'tcp_speed_p50_mm_s': 0.1000000000000203, 'tcp_speed_p95_mm_s': 8.632781086666233, 'tcp_speed_max_mm_s': 28.990688160164023, 'basis': 'TCP 世界位置差分除以**實際**取樣間隔', 'arm_rate_sign_flip_frac': 0.4020390070921986}

### `stage_a_overnight_1003_142154_rec`

判定 **pass**　首次失敗：（無）

- 資料路徑 `evaluation/runs/stage_a_overnight_1003_142154_rec/`（config `run_config.json`、程式版本 `code_versions.txt`）
- 前置調姿：{'ok': True, 'n_commands': 124, 'q_goal': [0.0, 0.0, 1.4372784010000001, 0.0, 0.0, 0.0], 'last_cmd_sim_t': 21.379999522000002, 'duration_s': 4.609982095154651}
- 交棒：{'ok': True, 'quiet_s': 2.039999954999999, 'setpoint_meas_max_dev_rad': 0.0023576257464001573, 'u_prev': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'u_prev_source': 'applied_report', 'exec_mode': 0, 'q_meas': [4.0602535591460764e-06, 3.724468115251511e-05, 1.4376744031906128, 2.1026120521128178e-05, 3.6682985182778793e-07, 7.898325549149376e-09], 'setpoint': [-9.707426271376695e-05, -0.002320381065247642, 1.4390936038219642, -0.0015581565346451858, -1.1452843871707194e-05, 2.9184393904317294e-08]}
- 到達：{'node_reported_reached_held': True, 'reached_held_measured': True, 'longest_in_tol_s': 2.490000000000002, 't_first_in_tol_s': 36.79, 't_reach_s': 36.79, 'err_p_min_mm': 1.4336171036926837, 'err_r_min_rad': 0.003162278977795912, 'err_p_final_mm': 1.4415505540909708, 'err_r_final_rad': 0.003162278977795912, 'tol': {'pos_m': 0.005, 'rot_rad': 0.02, 'hold_s': 2.0}, 'n_samples_in_tol': 250, 'basis': '由 log 的 tcp_x/y/z 與 tcp_r** 對 target.json 重算；姿態用測地角。**不照抄節點回報的 reached_held**'}
- 命令鏈：{'fail': None, 'received': 529, 'rejected': 0, 'frozen_steps': 0, 'stop_reason': 'stop_request'}
- 有效餘裕：{'setpoint_min_effective_slack_rad': 0.009973, 'setpoint_min_per_joint': [1.625206, 0.784353, 0.009973, 2.044177, 1.136207, 1.090895], 'measured_min_effective_slack_rad': 0.009871999999999999, 'measured_min_per_joint': [1.62956, 0.779419, 0.009872, 2.075973, 1.170474, 1.091128], 'n_setpoint_samples': 2642, 'n_measured_samples': 3926, 'note': '執行端那一層只保護**設定點**；實測角在一階遲滯下仍可能落在界外，故實測角另行核實、不列入成功判準'}
- 封存：ok=True
- 目標：{'tcp_world': [0.0, 1.0596999999999999, 0.55], 'source': '交棒時刻的抽屜位姿真值（0.0,1.45，開度 0.0）', 'stamp_sim_t': 23.419999477}
- 抽屜：{'verdict': 'displacement_over_baseline|no_contact_evidence', 'opening_max_abs_m': 1.19e-07, 'opening_tol_m': 1e-09, 'over_baseline': True, 'contact_verdict': 'no_contact_evidence', 'contact_fmag_max_n': 0.0}
- 缺項：[]

- 實測同動：{'window': '交棒後', 'n_steps': 1586, 'base_moving_frac': 0.8720050441361917, 'arm_moving_frac': 0.7969735182849937, 'both_moving_frac': 0.7969735182849937, 'both_moving_frac_of_any': 0.9139551699204628, 'base_speed_p50_mm_s': 33.897933087899304, 'arm_rate_p50_mrad_s': 127.0000000000246, 'thresholds': {'base_m_s': 0.0005, 'arm_rad_s': 0.005}, 'basis': '由**實測位姿與實測關節角**差分算，不用回報的 twist（靜止後會停更）'}
- 保持窗振動：{'window_s': [36.79, 39.28], 'n': 249, 'tcp_speed_p50_mm_s': 0.10000000000289547, 'tcp_speed_p95_mm_s': 16.280913961964465, 'tcp_speed_max_mm_s': 28.48666354629411, 'basis': 'TCP 世界位置差分除以**實際**取樣間隔', 'arm_rate_sign_flip_frac': 0.29250334672021416}

## 3 判定彙總

- 成功 **4/4** 趟
- 前置調姿：全數完成
- 交棒：全數通過
- 到達 5 mm／0.02 rad 保持 2 s（**實錄重算**）：r1=True（保持 4.580000000000005 s）; r2=True（保持 3.6699999999999946 s）; r3=True（保持 3.760000000000005 s）; rec=True（保持 2.490000000000002 s）

> 多趟成功**只支持這批案例的可重現性**，不宣稱統計穩定性。

### 指標彙總

| 趟次 | 前置誤差 mrad | 到達 t (sim) | 連續保持 s | 保持窗誤差 p50/p95/max mm | 姿態 max rad | 底盤位移/路徑 m | yaw rad | 同時動 % | 保持 TCPv p50/p95/max mm/s | 手臂翻號 % |
|---|---|---|---|---|---|---|---|---|---|---|
| `r1` | 0.11 | 58.66 | 4.580000000000005 | 1.413/1.455/4.788 | 0.00300 | 0.4087/0.4640 | 1.261 | 62.4 | 0.100/5.686/32.640 | 48.3 |
| `r2` | 2.42 | 58.2 | 3.6699999999999946 | 1.433/1.625/4.748 | 0.00316 | 0.4033/0.4600 | 1.278 | 65.7 | 0.100/8.039/29.361 | 40.6 |
| `r3` | 0.64 | 53.66 | 3.760000000000005 | 1.442/1.784/4.731 | 0.00316 | 0.4010/0.4568 | 1.287 | 70.7 | 0.100/8.633/28.991 | 40.2 |
| `rec` | 0.40 | 36.79 | 2.490000000000002 | 1.442/2.277/4.841 | 0.00316 | 0.4018/0.4563 | 1.287 | 79.7 | 0.100/16.281/28.487 | 29.3 |

同動由**實測位姿與實測關節角**差分算（不用回報的 twist）；門檻底盤 0.5 mm/s、手臂 5 mrad/s。
到達與保持由 **TCP 實錄對 target.json 重算**，不照抄節點回報。

## 4 抽屜讀值

- `stage_a_overnight_1003_142154_r1`：開度峰值 0.000e+00 m（門檻 1.000e-09）、超過基線 False；接觸 no_contact_evidence、淨合力峰值 0.0
- `stage_a_overnight_1003_142154_r2`：開度峰值 0.000e+00 m（門檻 1.000e-09）、超過基線 False；接觸 no_contact_evidence、淨合力峰值 0.0
- `stage_a_overnight_1003_142154_r3`：開度峰值 0.000e+00 m（門檻 1.000e-09）、超過基線 False；接觸 no_contact_evidence、淨合力峰值 0.0
- `stage_a_overnight_1003_142154_rec`：開度峰值 1.190e-07 m（門檻 1.000e-09）、超過基線 True；接觸 no_contact_evidence、淨合力峰值 0.0

**淨接觸力全零不作為「完全沒有接觸」的證明** —— 正向對照（已知接觸確認讀數非零）尚未完成。

## 5 影片

- 相機診斷通過：`evaluation/runs/stage_a_overnight_1003_142154_camdiag`
- 錄影趟次 `stage_a_overnight_1003_142154_rec`：影格 393 張、mp4 ['evaluation/runs/stage_a_overnight_1003_142154_rec/video.mp4']
- 可支持的說法：**完成展示**

## 6 有證據支持的下一步

補齊接觸感測的**正向對照**（已知接觸確認讀數非零）—— 那是目前唯一阻止「抽屜未被碰到」成為可支持結論的缺口。

## 7 殘留程序與封存完整性

- 殘留程序：無
- `stage_a_overnight_1003_142154_r1` 封存核對：{'run_dir': '/home/howardchen/masterthesis/evaluation/runs/stage_a_overnight_1003_142154_r1', 'checks': [{'name': 'sim/wb_run.json 可解析', 'ok': True, 'detail': '4.91 MB'}, {'name': '執行端必要紀錄齊備', 'ok': True, 'detail': '12 項'}, {'name': '執行端有步進紀錄', 'ok': True, 'detail': '6322 列 × 62 欄'}, {'name': '停止原因已記錄', 'ok': True, 'detail': 'stop_request'}, {'name': 'CPU 溫度已記錄且未超限', 'ok': True, 'detail': '峰 79.375 / 限 92.0'}, {'name': '本趟未要求錄影', 'ok': True, 'detail': ''}, {'name': '追蹤 JSONL 每行可解析', 'ok': True, 'detail': '2690 行可解析／0 行壞'}, {'name': '有 header 與 **summary**（錄製器已受控關檔）', 'ok': True, 'detail': 'header 1／summary 1'}, {'name': 'summary 筆數與**實際筆數一致**', 'ok': True, 'detail': "summary {'solver': 414, 'safety': 766, 'adapter': 755, 'endpoint': 753}／實際 {'safety': 766, 'adapter': 755, 'endpoint': 753, 'solver': 414}"}, {'name': '錄製器未拒絕任何封裝', 'ok': True, 'detail': '拒絕 0'}, {'name': '四段都有紀錄', 'ok': True, 'detail': 'adapter:755 endpoint:753 safety:766 solver:414'}, {'name': '**至少有一筆可逐段重建關聯**', 'ok': True, 'detail': '332 / 332 筆 source_seq 四段齊全'}, {'name': '執行端事件也已落盤（wb_run.json）', 'ok': True, 'detail': '753 筆'}, {'name': '失效停止事件的 source_seq 一律 −1', 'ok': True, 'detail': '失效事件 0 筆'}, {'name': 'wg2_out.json 可解析', 'ok': True, 'detail': '1.24 MB'}, {'name': '節點必要紀錄齊備', 'ok': True, 'detail': ''}, {'name': '實際模型選擇 == setpoint', 'ok': True, 'detail': 'arm_model=setpoint'}, {'name': '節點確實啟動（非拒絕啟動）', 'ok': True, 'detail': 'started=True\u3000'}, {'name': '有發布週期', 'ok': True, 'detail': 'published=298'}], 'cmd_env': {'path': '/home/howardchen/masterthesis/evaluation/runs/stage_a_overnight_1003_142154_r1/cmd_env.jsonl', 'n_lines': 2690, 'n_env': 2688, 'counts': {'safety': 766, 'adapter': 755, 'endpoint': 753, 'solver': 414}, 'n_traceable_full_chain': 332, 'n_endpoint_events_in_wb_run': 753}, 'node_format': 'current', 'arm_model': 'setpoint', 'node_stats': {'published': 298, 'dropped_stale': 0, 'no_solution': 0, 'reached_held': True, 'stop_why': 'loop_end', 'task_sim_span_s': 15.17, 'n_step_mismatch': 0, 'stopped_on_chain_fail': False}}
- `stage_a_overnight_1003_142154_r2` 封存核對：{'run_dir': '/home/howardchen/masterthesis/evaluation/runs/stage_a_overnight_1003_142154_r2', 'checks': [{'name': 'sim/wb_run.json 可解析', 'ok': True, 'detail': '4.69 MB'}, {'name': '執行端必要紀錄齊備', 'ok': True, 'detail': '12 項'}, {'name': '執行端有步進紀錄', 'ok': True, 'detail': '6185 列 × 62 欄'}, {'name': '停止原因已記錄', 'ok': True, 'detail': 'stop_request'}, {'name': 'CPU 溫度已記錄且未超限', 'ok': True, 'detail': '峰 83.875 / 限 92.0'}, {'name': '本趟未要求錄影', 'ok': True, 'detail': ''}, {'name': '追蹤 JSONL 每行可解析', 'ok': True, 'detail': '2517 行可解析／0 行壞'}, {'name': '有 header 與 **summary**（錄製器已受控關檔）', 'ok': True, 'detail': 'header 1／summary 1'}, {'name': 'summary 筆數與**實際筆數一致**', 'ok': True, 'detail': "summary {'solver': 401, 'safety': 737, 'adapter': 691, 'endpoint': 686}／實際 {'safety': 737, 'adapter': 691, 'endpoint': 686, 'solver': 401}"}, {'name': '錄製器未拒絕任何封裝', 'ok': True, 'detail': '拒絕 0'}, {'name': '四段都有紀錄', 'ok': True, 'detail': 'adapter:691 endpoint:686 safety:737 solver:401'}, {'name': '**至少有一筆可逐段重建關聯**', 'ok': True, 'detail': '317 / 326 筆 source_seq 四段齊全'}, {'name': '執行端事件也已落盤（wb_run.json）', 'ok': True, 'detail': '686 筆'}, {'name': '失效停止事件的 source_seq 一律 −1', 'ok': True, 'detail': '失效事件 0 筆'}, {'name': 'wg2_out.json 可解析', 'ok': True, 'detail': '1.31 MB'}, {'name': '節點必要紀錄齊備', 'ok': True, 'detail': ''}, {'name': '實際模型選擇 == setpoint', 'ok': True, 'detail': 'arm_model=setpoint'}, {'name': '節點確實啟動（非拒絕啟動）', 'ok': True, 'detail': 'started=True\u3000'}, {'name': '有發布週期', 'ok': True, 'detail': 'published=317'}], 'cmd_env': {'path': '/home/howardchen/masterthesis/evaluation/runs/stage_a_overnight_1003_142154_r2/cmd_env.jsonl', 'n_lines': 2517, 'n_env': 2515, 'counts': {'safety': 737, 'adapter': 691, 'endpoint': 686, 'solver': 401}, 'n_traceable_full_chain': 317, 'n_endpoint_events_in_wb_run': 686}, 'node_format': 'current', 'arm_model': 'setpoint', 'node_stats': {'published': 317, 'dropped_stale': 0, 'no_solution': 0, 'reached_held': True, 'stop_why': 'loop_end', 'task_sim_span_s': 16.12, 'n_step_mismatch': 0, 'stopped_on_chain_fail': False}}
- `stage_a_overnight_1003_142154_r3` 封存核對：{'run_dir': '/home/howardchen/masterthesis/evaluation/runs/stage_a_overnight_1003_142154_r3', 'checks': [{'name': 'sim/wb_run.json 可解析', 'ok': True, 'detail': '4.39 MB'}, {'name': '執行端必要紀錄齊備', 'ok': True, 'detail': '12 項'}, {'name': '執行端有步進紀錄', 'ok': True, 'detail': '5740 列 × 62 欄'}, {'name': '停止原因已記錄', 'ok': True, 'detail': 'stop_request'}, {'name': 'CPU 溫度已記錄且未超限', 'ok': True, 'detail': '峰 80.375 / 限 92.0'}, {'name': '本趟未要求錄影', 'ok': True, 'detail': ''}, {'name': '追蹤 JSONL 每行可解析', 'ok': True, 'detail': '2389 行可解析／0 行壞'}, {'name': '有 header 與 **summary**（錄製器已受控關檔）', 'ok': True, 'detail': 'header 1／summary 1'}, {'name': 'summary 筆數與**實際筆數一致**', 'ok': True, 'detail': "summary {'solver': 423, 'safety': 662, 'adapter': 651, 'endpoint': 651}／實際 {'safety': 662, 'adapter': 651, 'endpoint': 651, 'solver': 423}"}, {'name': '錄製器未拒絕任何封裝', 'ok': True, 'detail': '拒絕 0'}, {'name': '四段都有紀錄', 'ok': True, 'detail': 'adapter:651 endpoint:651 safety:662 solver:423'}, {'name': '**至少有一筆可逐段重建關聯**', 'ok': True, 'detail': '274 / 303 筆 source_seq 四段齊全'}, {'name': '執行端事件也已落盤（wb_run.json）', 'ok': True, 'detail': '651 筆'}, {'name': '失效停止事件的 source_seq 一律 −1', 'ok': True, 'detail': '失效事件 0 筆'}, {'name': 'wg2_out.json 可解析', 'ok': True, 'detail': '1.23 MB'}, {'name': '節點必要紀錄齊備', 'ok': True, 'detail': ''}, {'name': '實際模型選擇 == setpoint', 'ok': True, 'detail': 'arm_model=setpoint'}, {'name': '節點確實啟動（非拒絕啟動）', 'ok': True, 'detail': 'started=True\u3000'}, {'name': '有發布週期', 'ok': True, 'detail': 'published=298'}], 'cmd_env': {'path': '/home/howardchen/masterthesis/evaluation/runs/stage_a_overnight_1003_142154_r3/cmd_env.jsonl', 'n_lines': 2389, 'n_env': 2387, 'counts': {'safety': 662, 'adapter': 651, 'endpoint': 651, 'solver': 423}, 'n_traceable_full_chain': 274, 'n_endpoint_events_in_wb_run': 651}, 'node_format': 'current', 'arm_model': 'setpoint', 'node_stats': {'published': 298, 'dropped_stale': 0, 'no_solution': 0, 'reached_held': True, 'stop_why': 'loop_end', 'task_sim_span_s': 14.87, 'n_step_mismatch': 0, 'stopped_on_chain_fail': False}}
- `stage_a_overnight_1003_142154_rec` 封存核對：{'run_dir': '/home/howardchen/masterthesis/evaluation/runs/stage_a_overnight_1003_142154_rec', 'checks': [{'name': 'sim/wb_run.json 可解析', 'ok': True, 'detail': '3.35 MB'}, {'name': '執行端必要紀錄齊備', 'ok': True, 'detail': '12 項'}, {'name': '執行端有步進紀錄', 'ok': True, 'detail': '3926 列 × 62 欄'}, {'name': '停止原因已記錄', 'ok': True, 'detail': 'stop_request'}, {'name': 'CPU 溫度已記錄且未超限', 'ok': True, 'detail': '峰 84.25 / 限 92.0'}, {'name': '錄影影格數 > 0', 'ok': True, 'detail': '索引 393 格'}, {'name': '索引與磁碟影格數一致', 'ok': True, 'detail': '索引 393／磁碟 393'}, {'name': '索引的模擬時間單調不減', 'ok': True, 'detail': '393 筆'}, {'name': '所有影格檔非空', 'ok': True, 'detail': '非空 393／共 393'}, {'name': '追蹤 JSONL 每行可解析', 'ok': True, 'detail': '2019 行可解析／0 行壞'}, {'name': '有 header 與 **summary**（錄製器已受控關檔）', 'ok': True, 'detail': 'header 1／summary 1'}, {'name': 'summary 筆數與**實際筆數一致**', 'ok': True, 'detail': "summary {'solver': 424, 'safety': 537, 'adapter': 529, 'endpoint': 527}／實際 {'safety': 537, 'adapter': 529, 'endpoint': 527, 'solver': 424}"}, {'name': '錄製器未拒絕任何封裝', 'ok': True, 'detail': '拒絕 0'}, {'name': '四段都有紀錄', 'ok': True, 'detail': 'adapter:529 endpoint:527 safety:537 solver:424'}, {'name': '**至少有一筆可逐段重建關聯**', 'ok': True, 'detail': '293 / 302 筆 source_seq 四段齊全'}, {'name': '執行端事件也已落盤（wb_run.json）', 'ok': True, 'detail': '527 筆'}, {'name': '失效停止事件的 source_seq 一律 −1', 'ok': True, 'detail': '失效事件 0 筆'}, {'name': 'wg2_out.json 可解析', 'ok': True, 'detail': '1.23 MB'}, {'name': '節點必要紀錄齊備', 'ok': True, 'detail': ''}, {'name': '實際模型選擇 == setpoint', 'ok': True, 'detail': 'arm_model=setpoint'}, {'name': '節點確實啟動（非拒絕啟動）', 'ok': True, 'detail': 'started=True\u3000'}, {'name': '有發布週期', 'ok': True, 'detail': 'published=297'}], 'recording': {'n': 393, 'dir': '/home/howardchen/masterthesis/evaluation/runs/stage_a_overnight_1003_142154_rec/frames', 'total_mb': 158.1, 'sim_span_s': 39.2, 'eff_fps': 10.03}, 'cmd_env': {'path': '/home/howardchen/masterthesis/evaluation/runs/stage_a_overnight_1003_142154_rec/cmd_env.jsonl', 'n_lines': 2019, 'n_env': 2017, 'counts': {'safety': 537, 'adapter': 529, 'endpoint': 527, 'solver': 424}, 'n_traceable_full_chain': 293, 'n_endpoint_events_in_wb_run': 527}, 'node_format': 'current', 'arm_model': 'setpoint'}
