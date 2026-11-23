#!/usr/bin/env python3
"""Mask R-CNN 推論服務（在 .venv-dl0 執行；Isaac 的 Python 環境 torchvision 版本不合，不能直接推論）。
由實跑視覺工作程序以子程序啟動，stdin／stdout 傳長度前綴的 npz：
  請求：rgb（H×W×3 uint8）
  回覆：各類別「聯集遮罩」packbits＋最高分數＋實例數；ms＝推論耗時
    .venv-dl0/bin/python evaluation/house_seg_server.py <weights.pth>"""
import io
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import house_seg as HS                                               # noqa: E402


def rd(f, n):
    b = b''
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            return None
        b += c
    return b


def main():
    model = HS.load_model(sys.argv[1])
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    fout.write(b'READY\n'); fout.flush()
    while True:
        h = rd(fin, 8)
        if h is None:
            break
        n = struct.unpack('<Q', h)[0]
        req = np.load(io.BytesIO(rd(fin, n)))
        t0 = time.perf_counter()
        pr = HS.predict(model, req['rgb'])
        out = dict(ms=np.float32((time.perf_counter() - t0) * 1e3), shape=np.array(req['rgb'].shape[:2]))
        for c, lst in pr.items():
            u = np.zeros(req['rgb'].shape[:2], bool)
            for d in lst:
                u |= d['mask']
            out[f'm_{c}'] = np.packbits(u)
            out[f's_{c}'] = np.float32(max(d['score'] for d in lst))
            out[f'n_{c}'] = np.int32(len(lst))
        bio = io.BytesIO(); np.savez(bio, **out); b = bio.getvalue()
        fout.write(struct.pack('<Q', len(b)) + b); fout.flush()


if __name__ == '__main__':
    main()
