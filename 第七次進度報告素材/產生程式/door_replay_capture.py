#!/usr/bin/env python3
"""重播單幀擷取順序（純函式；Codex 20261024_003051）：取影像 →（無圖時）同一姿態重新 render，最多 max_retry 次（不推進物理）
→ **最後一次 render 之後**才回讀實際姿態 → 回傳（影像、重試次數、姿態）。呼叫端再存影格。
姿態偏差因此對應實際擷取狀態，而不是暖機前狀態。"""


def has_image(img):
    return img is not None and bool(getattr(img, 'size', 0))


def capture(get_img, render, read_pose, max_retry=3):
    img = get_img()
    retry = 0
    while not has_image(img) and retry < max_retry:
        retry += 1
        render()
        img = get_img()
    pose = read_pose()
    return img, retry, pose
