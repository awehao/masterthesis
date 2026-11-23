#!/usr/bin/env python3
"""door_replay_capture 反例：姿態回讀一定在最後一次 render 之後；重試上限 3；有圖時不重試；仍無圖時回傳 None 並記 retry＝3。

    python3 evaluation/test_door_replay_capture.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_replay_capture as C                          # noqa: E402


class Img:
    size = 1


def run(empties):
    log, seq = [], list(empties)

    def get_img():
        log.append('get')
        return None if (seq and seq.pop(0)) else Img()

    def render():
        log.append('render')

    def read_pose():
        log.append('pose')
        return 'P'
    img, retry, pose = C.capture(get_img, render, read_pose)
    return img, retry, pose, log


class T(unittest.TestCase):
    def test_no_retry_when_image(self):
        img, retry, pose, log = run([False])
        self.assertEqual((retry, log), (0, ['get', 'pose']))

    def test_pose_after_last_render(self):
        img, retry, pose, log = run([True, True, False])
        self.assertEqual(retry, 2)
        self.assertEqual(log, ['get', 'render', 'get', 'render', 'get', 'pose'])
        self.assertIsInstance(img, Img)

    def test_cap_three(self):
        img, retry, pose, log = run([True] * 10)
        self.assertIsNone(img)
        self.assertEqual(retry, 3)
        self.assertEqual(log.count('render'), 3)
        self.assertEqual(log[-1], 'pose')
        self.assertEqual(log.index('pose'), len(log) - 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
