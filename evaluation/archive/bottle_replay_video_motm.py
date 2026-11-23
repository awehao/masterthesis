#!/usr/bin/env python3
"""多任務第一批 v2 實錄重播製片（bottle_replay_video_w1vhigh.py 原樣沿用，只加 v2 事件；2026-11-02，Howard 多任務交接「影片 A1、B2、C1」）：
關鍵時刻加 relwin_shift（依量測瓶 x 一次平移抓取窗與接近排程，顯示 Δx 與平移後窗口）。片名由呼叫端給：當趟配置（桌高、瓶位偏移、動態物）與結果
（「抓取完成、運送未完成」等依實際第一阻點）。版面、圖表、滑移圖、「實錄重播（非即時錄影）」標示全部同 w1vhigh。
    python3 evaluation/bottle_replay_video_motm.py <replay_run_id> <src_run_id> <out.mp4> --title 文字
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bottle_replay_video_w1vhigh as V                  # noqa: E402

_moments = V.moments


def moments(r):
    out = _moments(r)
    out += [(e['t'], f"依量測瓶位平移抓取排程 Δx {e['dx'] * 1e3:+.1f} mm（窗 {e['win'][0]:+.3f}～{e['win'][1]:+.3f} m）") for e in r['events'] if e['what'] == 'relwin_shift']
    return sorted(out)


V.moments = moments

if __name__ == '__main__':
    sys.exit(V.main())
