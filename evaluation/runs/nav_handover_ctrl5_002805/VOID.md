# 這一趟作廢（對 control_frequency 的問題**沒有**證據力）

* 設定：`control_frequency=5.0`（dt=0.200 s），其餘沿用。
* 結果：底盤整趟**一步沒動**，d 固定 4.607 m。
* 原因**不是** 5 Hz：gmpc 整趟 `state='no_plan'`（1144 輪全是），
  也就是它從頭到尾沒有收到 `/plan`。
  * mission 的 `/plan` 發布是 `TRANSIENT_LOCAL`，gmpc 的訂閱是預設
    `VOLATILE`（RELIABLE/KEEP_LAST/1）。兩者**相容**所以不報 QoS 不符，
    但 VOLATILE 訂閱者**收不到 latch 回放**，只收現場訊息。
  * 計畫原本只發一次（程式裡還寫著「latched，不重發」），發布端比訂閱端
    先就緒時就會掉。這是競態，所以前幾趟碰巧收到、這趟沒有。
* gmpc.log 末端的 `ExternalShutdownException` 是收尾 SIGTERM，不是跑中崩潰。
* 已修：`drawer_mission_node.publish_plan()` 改成先等訂閱者（最多 10 s）
  再發，並重發 5 次；`mission.json` 記下 `plan_n_sub_at_publish`。
* 結論：**5 Hz 這個假設仍未驗證**，要重跑。
