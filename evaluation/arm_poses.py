"""The arm's named fixed configurations, from the one file that holds them.

`config/arm_initial_pose.yaml` exists so these numbers live in a single place.
Three evaluation scripts had each grown their own copy of the run's start pose,
which is how the file's own docstring describes the failure it was created to
prevent. This is the reader that keeps that promise for the Python side; xacro
and the launch read the same file directly.

    spawn               what Gazebo spawns at and the visualisation shows
    test_start          where a RUN starts (see the file for why they differ)
    pregrasp_reference  posture reference for the whole-body solver
"""
from __future__ import annotations

import os

import yaml

_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'src/my_omnibot_description/config/arm_initial_pose.yaml')
JOINTS = [f'joint{i}' for i in range(1, 7)]


def pose(name: str = 'spawn', path: str | None = None) -> list[float]:
    d = yaml.safe_load(open(path or _YAML, encoding='utf-8'))
    src = d if name == 'spawn' else d.get(name)
    if src is None:
        raise KeyError(f'{name!r} 不在 {path or _YAML}；可用：spawn, '
                       + ', '.join(k for k in d if isinstance(d[k], dict)))
    missing = [j for j in JOINTS if j not in src]
    if missing:
        raise KeyError(f'{name!r} 缺少 {missing}')
    return [float(src[j]) for j in JOINTS]


if __name__ == '__main__':
    import sys
    print(' '.join(f'{v:g}' for v in pose(sys.argv[1] if len(sys.argv) > 1
                                          else 'spawn')))
