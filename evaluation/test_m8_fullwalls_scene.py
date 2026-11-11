#!/usr/bin/env python3
"""Offline XML geometry checks only: no Isaac, rendering, ROS, or simulation."""
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORLDS = ROOT / 'src/ammr_bringup/worlds'
BASE = WORLDS / 'mission_room.sdf'
FULL = WORLDS / 'mission_room_m8_fullwalls.sdf'
OUTER = ('wall_north', 'wall_south', 'wall_east', 'wall_west')
PARTITIONS = ('partition_w', 'partition_e')
HEIGHT = 2.4


def models(path):
    return {m.attrib['name']: m for m in ET.parse(path).findall('./world/model')}


def values(model, path):
    return tuple(float(v) for v in model.findtext(path).split())


def box(model):
    return values(model, './link/collision/geometry/box/size')


def rect(model):
    p, s = values(model, 'pose'), box(model)
    return (p[0] - s[0] / 2, p[0] + s[0] / 2,
            p[1] - s[1] / 2, p[1] + s[1] / 2)


class FullWallSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old, cls.new = models(BASE), models(FULL)

    def test_same_models_and_unchanged_other_assets(self):
        self.assertEqual(set(self.old), set(self.new))
        for name in self.old.keys() - set(OUTER):
            with self.subTest(model=name):
                self.assertEqual(ET.tostring(self.old[name]), ET.tostring(self.new[name]))

    def test_outer_changes_are_only_height_and_center_z(self):
        for name in OUTER:
            with self.subTest(wall=name):
                expected = ET.fromstring(ET.tostring(self.old[name]))
                p = values(expected, 'pose')
                expected.find('pose').text = ' '.join(str(v) for v in (*p[:2], HEIGHT / 2, *p[3:]))
                actual = self.new[name]
                self.assertEqual(values(expected, 'pose'), values(actual, 'pose'))
                for kind in ('collision', 'visual'):
                    path = f'./link/{kind}/geometry/box/size'
                    old_size = values(expected, path)
                    self.assertEqual((*old_size[:2], HEIGHT), values(actual, path))
                self.assertEqual(actual.findtext('static'), 'true')

    def test_complete_height_from_floor_and_visual_collision_match(self):
        for name in OUTER + PARTITIONS:
            with self.subTest(wall=name):
                m = self.new[name]
                p, s = values(m, 'pose'), box(m)
                self.assertAlmostEqual(p[2] - s[2] / 2, 0.0)
                self.assertAlmostEqual(p[2] + s[2] / 2, HEIGHT)
                self.assertEqual(s, values(m, './link/visual/geometry/box/size'))
                self.assertEqual(p[3:], (0.0, 0.0, 0.0))

    def test_perimeter_corners_have_no_gaps(self):
        for a in ('wall_north', 'wall_south'):
            for b in ('wall_east', 'wall_west'):
                with self.subTest(corner=(a, b)):
                    ra, rb = rect(self.new[a]), rect(self.new[b])
                    self.assertGreater(min(ra[1], rb[1]) - max(ra[0], rb[0]), 0)
                    self.assertGreater(min(ra[3], rb[3]) - max(ra[2], rb[2]), 0)

    def test_partition_extensions_meet_outer_and_door_asset_bounds(self):
        west, east = rect(self.new['partition_w']), rect(self.new['partition_e'])
        self.assertAlmostEqual(west[0], rect(self.new['wall_west'])[1])
        self.assertAlmostEqual(east[1], rect(self.new['wall_east'])[0])
        self.assertAlmostEqual(west[1], -1.5)
        self.assertAlmostEqual(east[0], 2.5)
        for r in (west, east):
            self.assertAlmostEqual(r[2], 0.0)
            self.assertAlmostEqual(r[3], 0.12)

    def test_all_static_box_footprints_unchanged(self):
        for name, m in self.old.items():
            if m.find('./link/collision/geometry/box') is None:
                continue
            with self.subTest(model=name):
                p0, p1 = values(m, 'pose'), values(self.new[name], 'pose')
                self.assertEqual(p0[:2] + p0[3:], p1[:2] + p1[3:])
                self.assertEqual(box(m)[:2], box(self.new[name])[:2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
