#!/usr/bin/env python3
"""重播核對與影片時間的反例：
  door_replay_check.check：一一對應時 complete；缺圖（img_ok＝0 或檔案缺）、姿態超差、步號錯位、少幀、多幀、來源雜湊未記或不符 ⇒ 不完整並列原因。
  door_replay_video：缺圖不靜默略過——影片總長仍等於（末幀模擬時間 − 首幀）÷ 倍速＋末幀時長；keymoments 記缺圖數與 replay_complete。

    python3 evaluation/test_door_replay.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_replay_check as RC                          # noqa: E402

STEPS = [[k, round(0.01 * k, 4)] for k in range(100)]


def rows_for(exp, over=None):
    rows = [{'frame': str(i), 'step': str(s), 'sim_t': f'{t:.4f}', 'img_ok': '1', 'pose_ok': '1'} for i, (s, t) in enumerate(exp, start=1)]
    for (i, k), v in (over or {}).items():
        rows[i][k] = v
    return rows


class Check(unittest.TestCase):
    def setUp(self):
        self.exp = RC.expected_frames(STEPS, 10.0, 0.01, 0.0, 1e9)

    def test_complete(self):
        r = RC.check(self.exp, rows_for(self.exp), lambda f: True, 'a' * 64, 'a' * 64)
        self.assertTrue(r['complete'], r)
        self.assertEqual(r['n_expected'], 10)

    def test_failures(self):
        cases = [
            (rows_for(self.exp, {(3, 'img_ok'): '0'}), lambda f: True, 'missing_images_1'),
            (rows_for(self.exp), lambda f: f != 5, 'missing_images_1'),
            (rows_for(self.exp, {(2, 'pose_ok'): '0'}), lambda f: True, 'pose_out_of_tolerance_1'),
            (rows_for(self.exp, {(4, 'step'): '41'}), lambda f: True, 'frame_step_mismatch_1'),
            (rows_for(self.exp)[:-1], lambda f: True, 'frame_count_9_vs_expected_10'),
            (rows_for(self.exp) + [{'frame': '11', 'step': '100', 'sim_t': '1.0', 'img_ok': '1', 'pose_ok': '1'}], lambda f: True,
             'frame_count_11_vs_expected_10'),
        ]
        for rows, png, why in cases:
            r = RC.check(self.exp, rows, png, 'a' * 64, 'a' * 64)
            self.assertFalse(r['complete'], why)
            self.assertIn(why, r['reasons'])
        self.assertIn('source_sha_not_logged', RC.check(self.exp, rows_for(self.exp), lambda f: True, None, 'a' * 64)['reasons'])
        self.assertIn('source_sha_mismatch', RC.check(self.exp, rows_for(self.exp), lambda f: True, 'b' * 64, 'a' * 64)['reasons'])


class Video(unittest.TestCase):
    def test_missing_frame_keeps_timing(self):
        from PIL import Image
        rid = '_replaytest_tmp'
        D = os.path.join(HERE, 'runs', rid)
        shutil.rmtree(D, ignore_errors=True)
        os.makedirs(os.path.join(D, 'frames'))
        src = 'door_k3ab_lw_1'
        room = json.load(open(os.path.join(HERE, 'runs', src, 'room_run.json')))
        S = room['steps'][2000:2400:10]                      # 40 幀、模擬 0.1 s 間隔
        with open(os.path.join(D, 'frames.csv'), 'w') as f:
            f.write('frame,sim_t,step,owner,kind,door_q_deg,x,y,yaw,img_ok,pose_ok\n')
            for i, s in enumerate(S, start=1):
                ok = 0 if i == 20 else 1
                if ok:
                    Image.new('RGB', (320, 180), (80, 80, 80)).save(os.path.join(D, 'frames', f'f_{i:05d}.png'))
                f.write(f'{i},{s[1]:.4f},{s[0]},wholebody,x,0,0,0,0,{ok},1\n')
        json.dump({'complete': False, 'reasons': ['missing_images_1']}, open(os.path.join(D, 'replay_check.json'), 'w'))
        out = os.path.join(tempfile.mkdtemp(), 'v.mp4')
        try:
            subprocess.run([sys.executable, os.path.join(HERE, 'door_replay_video.py'), rid, src, out, '--speed', '2'], check=True,
                           capture_output=True)
            dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', out],
                                       capture_output=True, text=True).stdout)
            expect = (S[-1][1] - S[0][1]) / 2 + 0.1 / 2
            self.assertAlmostEqual(dur, expect, delta=0.15)          # 缺一幀不縮短影片
            km = json.load(open(out + '.keymoments.json'))
            self.assertEqual(km['n_missing_images'], 1)
            self.assertFalse(km['replay_complete'])
        finally:
            shutil.rmtree(D, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
