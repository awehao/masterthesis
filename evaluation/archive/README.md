# 歸檔的歷史程式（2026-11-23）

主線（住宅任務：替身、Isaac 實跑、重播、Mask R-CNN）用不到的 `evaluation/*.py`，共 1,158 個，從 `evaluation/` 搬到這裡；**沒有刪除**，git 歷史跟著走。

判定方式：從主線入口程式追 import，加上現役程式／啟動腳本／`mission/`／第 80 趟以後凍結清單裡以字串提到的模組，反覆擴張；不在其中的才歸檔（保留得偏寬）。

要重跑這裡的舊程式，把兩個資料夾都放進搜尋路徑：

    PYTHONPATH=evaluation:evaluation/archive python3 evaluation/archive/<檔名>.py

舊趟次（第 80 趟以前）的凍結清單路徑指向 `evaluation/<檔名>.py`，若要依舊登錄重跑，需先把對應檔案搬回。
