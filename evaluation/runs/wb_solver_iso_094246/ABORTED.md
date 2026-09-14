# 啟動失敗，非任務結果（起動前檢查的時鐘設定錯誤）

場景條件、link_tcp、下游速度框讀回**都通過**：

* 語意場景檢查：機器人與地面 20 個碰撞體、其他位置 0 個
* 下游 `wholebody_safety` 執行期生效值
  `base_lin=0.035255 base_ang=0.199900 arm_max=0.999900`，與求解端同源
* `obstacles_configured = 0`

失敗項是 **TF 新鮮度**：11 個連桿的 TF 全部查得到、有限、身分正確
（`odom → <link>`，xyz 合理），但 age 算出 1.789e9 s ——
`wb_freespace_preflight.py` 的節點**沒有設 `use_sim_time`**，
拿牆鐘去減 sim 時戳。是檢查程式的缺陷，不是場景或 TF 的問題。

處置：檢查節點設 `use_sim_time=True`，並等 `/clock` 有值才做查核。
本目錄不計為任務結果。
