# 第六次進度報告：12 頁圖片預覽

這是版面與內容參考，不是已完成的可編輯 PowerPoint。沿用第五次白底、紫色標題線及藍綠圖框；主體限 W-GMPC 實作與成果。

- `第六次簡報圖片參考.pdf`：完整 12 頁。
- `第六次簡報12頁PNG.zip`：12 張 1600×900 PNG。
- `overview_1.png`／`overview_2.png`：前、後六頁縮圖總覽。
- `slides.html`：可編輯的本地 HTML/CSS 版面，開啟後可查看全稿；網址加 `?slide=8` 可只看第 8 頁。
- `render_slides.py`：本地 Chrome 輸出圖片、總覽、PDF 與 ZIP，不啟動模擬。

| 頁 | 圖片 | 內容 |
|---|---|---|
| 1 | `01_cover.png` | 封面 |
| 2 | `02_agenda.png` | 報告大綱 |
| 3 | `03_goal.png` | 研究目標 |
| 4 | `04_review.png` | 進度回顧 |
| 5 | `05_model.png` | 增廣預測模型 |
| 6 | `06_controller.png` | 成本、限制與回授架構 |
| 7 | `07_sequence.png` | 完整操作流程與影片位置 |
| 8 | `08_parking.png` | 嚴格停車 vs MotM |
| 9 | `09_horizon.png` | H1/H5 時域消融 |
| 10 | `10_architecture.png` | B1/P 架構比較 |
| 11 | `11_next.png` | 未來工作 |
| 12 | `12_conclusion.png` | 結語 |

第 4 頁只回顧第五次的全身模型、Isaac 閉迴路、底盤朝向與定點操作，不與本次成果並列；第六次實作從第 5 頁開始。

## 資料與展示界線

- 第 8 頁來源為 `evaluation/results/motm_speed/park_hold_formal_summary.yaml` 及正式比較表；平均 80.72／93.62 s 是三趟平均，配對差來自正式各趟結果。
- 第 9 頁採 C0 正式結果範圍，C1a 僅補充；不跨批次混算。
- 第 10 頁使用既有圖 15 與 WG4-B 正式結果，未重新執行分析或控制。
- 影像從既有正式配對 A 影片截取：`assets/formal_room.png` 為全景並排影片 12 s，`assets/formal_operation.png` 為側面並排影片 16 s。原片為 3 倍速實錄重播；圖片只占影片位置，不代表新增驗證趟。
- 封面報告日期待填。大學名稱為文字標示，不是重新繪製的校徽。
- 已檢查 12 頁內容高度與圖片載入，並目視前後六頁總覽。原實驗資料、第五次 PDF 與控制程式未改動。

完整口述與備用頁見工作區根目錄 `第六次進度報告_逐頁簡報參考.md`。
