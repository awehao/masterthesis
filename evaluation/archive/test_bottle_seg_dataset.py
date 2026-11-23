#!/usr/bin/env python3
"""Small analytic checks; no Isaac or rendering launch."""
import numpy as np
from bottle_seg_dataset import label


def main():
    K = np.array([[100., 0, 32], [0, 100., 32], [0, 0, 1]])
    T = np.eye(4)
    # Cylinder axis along optical Z: visible front cap is z=1.0.
    dep = np.full((64, 64), 1.0)
    mask, st = label(dep, K, T, np.array([0., 0, 1.]), np.array([0., 0, 1.]), radius=.10, height=.2)
    assert st['visible'] > 200 and mask[31, 31] == 1, st
    assert mask[0, 0] == 0
    _, st2 = label(dep * .5, K, T, np.array([0., 0, 1.]), np.array([0., 0, 1.]), radius=.1)
    assert st2['visible'] == 0 and st2['occluded'] > 200, st2
    _, st3 = label(dep * 0., K, T, np.array([0., 0, 1.]), np.array([0., 0, 1.]), radius=.1)
    assert st3['visible'] == 0 and st3['unknown'] > 200, st3
    _, st4 = label(dep, K, T, np.array([0., 0, -2.]), np.array([0., 0, 1.]))
    assert st4['expected'] == 0
    # World-frame translation of both camera and cylinder changes no pixel.
    T2 = T.copy(); T2[:3, 3] = [1., 2., 3.]
    mask2, _ = label(dep, K, T2, np.array([1., 2., 4.]), np.array([0., 0, 1.]), radius=.1, height=.2)
    assert np.array_equal(mask, mask2)
    try:
        label(dep, K, T, np.full(3, np.nan), np.array([0., 0, 1.]))
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite geometry accepted')
    print('6 bottle label geometry checks passed')


if __name__ == '__main__':
    main()
