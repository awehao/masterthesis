#!/usr/bin/env python3
"""判定器 r4 J4 反例：一致 ⇒ 無理由；記錄值 NaN／Inf／字串／bool／缺鍵 ⇒ coopen_gate_invalid_recorded；登錄值 NaN（暫存登錄檔）⇒
coopen_gate_invalid_registered；不同 ⇒ coopen_gate_mismatch；缺 coopen_gate ⇒ coopen_gate_missing；皆不拋例外。

    python3 evaluation/test_door_coopen_check_core4.py
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core4 as J                    # noqa: E402

PRM = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r6.json')
REL = os.path.relpath(PRM, WS)
REG = json.load(open(PRM))['approach_gate']


def chk(cfg, rel=REL):
    why = []
    J.check_gate({'coopen': {'params_path': rel}, 'coopen_gate': None if cfg is None else {'config': cfg}}, why)
    return why


class K(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(chk(dict(REG)), [])

    def test_recorded_invalid(self):
        for bad in (float('nan'), float('inf'), '0.003', True, None):
            self.assertIn('coopen_gate_invalid_recorded', chk(dict(REG, perp_pause_m=bad)), bad)
        c = dict(REG)
        c.pop('rot_resume_rad')
        self.assertIn('coopen_gate_invalid_recorded', chk(c))

    def test_registered_invalid(self):
        p = json.load(open(PRM))
        p['approach_gate']['perp_pause_m'] = float('nan')
        with tempfile.NamedTemporaryFile('w', suffix='.json', dir=os.path.join(HERE, 'results', 'door'), delete=False) as f:
            json.dump(p, f)
            path = f.name
        try:
            self.assertIn('coopen_gate_invalid_registered', chk(dict(REG), os.path.relpath(path, WS)))
        finally:
            os.remove(path)

    def test_mismatch_and_missing(self):
        self.assertIn('coopen_gate_mismatch', chk(dict(REG, perp_pause_m=0.004)))
        self.assertIn('coopen_gate_missing', chk(None))


if __name__ == '__main__':
    unittest.main(verbosity=2)
