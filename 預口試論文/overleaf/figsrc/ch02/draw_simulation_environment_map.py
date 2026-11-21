#!/usr/bin/env python3
"""Draw the registered m8_house_r2 layout as a thesis vector figure.

Only reads scene geometry. Does not run a simulator or alter original images.
The presentation transform is (u, v) = (world_y, -world_x), matching the report.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'masterthesis_map_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Arc, Circle, Rectangle
import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
REG = ROOT / 'evaluation/experiments/20261118_m8_house_r2_scene.json'
FONT = OUT.parents[1] / 'fonts/kaiu.ttf'
INK = '#24282c'
WALL = '#3b4146'
FURN = '#e1ded8'
EDGE = '#92908b'
ACCENT = '#1f607c'
TINT = '#e4f0f4'
MUTED = '#62696e'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reg = json.loads(REG.read_text())
    door_path = ROOT / reg['door']['asset']
    door = yaml.safe_load(door_path.read_text())
    world = ROOT / reg['world']['path']
    assert digest(world) == reg['world']['sha256'], 'Registered world geometry changed'
    assert digest(door_path) == reg['door']['sha256'], 'Registered door asset changed'
    assert FONT.is_file(), 'DFKai-SB font file is required'
    font = FontProperties(fname=str(FONT))
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'path', 'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(8.2, 5.05))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor('white')
    ax.set(xlim=(-8.65, 13.05), ylim=(-7.15, 6.2), aspect='equal')
    ax.set_axis_off()
    texts = []

    def text(u, v, label, size=10, color=INK, **kw):
        artist = ax.text(u, v, label, fontproperties=font, fontsize=size,
                         color=color, ha=kw.pop('ha', 'center'),
                         va=kw.pop('va', 'center'), zorder=10, **kw)
        texts.append((label, artist))
        return artist

    def footprint(c, s, face=FURN, edge=EDGE, lw=0.65, z=3):
        patch = Rectangle((c[1] - s[1]/2, -c[0] - s[0]/2), s[1], s[0],
                          facecolor=face, edgecolor=edge, linewidth=lw, zorder=z)
        ax.add_patch(patch)
        return patch

    def callout(label, point, label_at, size=10, color=INK, align='center'):
        artist = ax.annotate(label, xy=point, xytext=label_at,
                             fontproperties=font, fontsize=size, color=color,
                             ha=align, va='center', linespacing=1.25, zorder=12,
                             arrowprops={'arrowstyle': '-', 'lw': 0.7,
                                         'color': color, 'shrinkA': 3, 'shrinkB': 2})
        texts.append((label, artist))
        return artist

    # Room backgrounds and exact wall and furniture footprints.
    rooms = [('臥室', '起始區域', reg['dims']['bedroom_y'], '#faf9f6'),
             ('客廳', '導航與茶几取物區域', reg['dims']['living_y'], '#f8fafb'),
             ('右側房間', '抽屜操作區域', reg['dims']['right_y'], '#faf9f6')]
    for name, subtitle, (a, b), color in rooms:
        ax.add_patch(Rectangle((a, -5), b-a, 10, facecolor=color,
                               edgecolor='none', zorder=0))
        text((a+b)/2, 5.92, name, size=12.5)
        text((a+b)/2, 5.45, subtitle, size=9, color=MUTED)

    for item in reg['walls']:
        footprint(item['c'], item['s'], face=WALL, edge=WALL, lw=0, z=4)

    # These two sections are created by the door asset, outside the SDF walls.
    lo, hi = door['wall']['extent_x']
    recess = door['wall'].get('hinge_side_recess', 0.0)
    thickness = door['wall']['thickness']
    clear_width = door['frame_clear_width']
    for xa, xb in [(lo, -recess), (clear_width, hi)]:
        footprint([(xa+xb)/2, thickness/2, 0], [xb-xa, thickness, 0],
                  face=WALL, edge=WALL, lw=0, z=4)

    furniture = {item['name']: item for item in reg['furniture']}
    for name, item in furniture.items():
        if name == 'coffee_table':
            footprint(item['c'], item['s'], face=TINT, edge=ACCENT, lw=1.1)
        elif name == 'tv':
            footprint(item['c'], item['s'], face=WALL, edge=WALL, lw=0.4, z=5)
        else:
            footprint(item['c'], item['s'])

    # Small interior details retain the same source footprint and aid recognition.
    bed = furniture['bed']
    bu, bv = bed['c'][1], -bed['c'][0]
    ax.plot([bu-0.55, bu-0.55], [bv-0.60, bv+0.60], color=EDGE, lw=0.6, zorder=4)
    text(bu+0.12, bv, '床', size=10)
    for name in ['desk', 'dresser', 'tv_stand', 'sofa', 'cabinet_a', 'cabinet_b']:
        item = furniture[name]
        label = {'desk': '書桌', 'dresser': '矮櫃', 'tv_stand': '電視櫃',
                 'sofa': '沙發', 'cabinet_a': '櫃', 'cabinet_b': '櫃'}[name]
        text(item['c'][1], -item['c'][0], label, size=7.6)
    text(furniture['loveseat']['c'][1], -furniture['loveseat']['c'][0], '雙人\n沙發', size=8.5)
    shelf = furniture['shelf']
    callout('層架', (shelf['c'][1]-shelf['s'][1]/2, -shelf['c'][0]),
            (11.15, -1.0), size=9.2)

    # Drawer footprint comes directly from the frozen scene registry.
    xa, xb, ya, yb = reg['drawer']['footprint']
    footprint([(xa+xb)/2, (ya+yb)/2, 0], [xb-xa, yb-ya, 0],
              face=TINT, edge=ACCENT, lw=1.1)
    callout('抽屜櫃', ((ya+yb)/2, -xb), (10.3, 3.5), color=ACCENT, size=10.5)
    coffee = furniture['coffee_table']
    callout('茶几取物區', (coffee['c'][1]+coffee['s'][1]/2, -coffee['c'][0]),
            (6.05, -1.50), size=10.5, color=ACCENT)

    # Closed leaf and a conventional 90-degree opening symbol, not a trajectory.
    length = door['leaf']['length']
    leaf_t = door['leaf']['thickness']
    footprint([length/2, leaf_t/2, 0], [length, leaf_t, 0],
              face=ACCENT, edge=ACCENT, lw=0.7, z=6)
    ax.add_patch(Circle((0, 0), 0.035, facecolor=ACCENT, edgecolor='none', zorder=7))
    ax.add_patch(Arc((0, 0), 2*length, 2*length, theta1=180, theta2=270,
                     color=ACCENT, lw=0.8, linestyle=(0, (3, 2)), zorder=5))
    ax.plot([-length, 0], [0, 0], color=ACCENT, lw=0.8,
            linestyle=(0, (3, 2)), zorder=5)
    callout(f'通行門\n淨寬 {clear_width:.2f} m', (0.02, -0.5),
            (-1.72, -1.50), size=9.5, color=ACCENT)
    opening = reg['dims']['opening2_x']
    callout(f'通行開口\n淨寬 {opening[1]-opening[0]:.2f} m', (8.06, 0),
            (9.65, 0.0), size=9.5)

    # Overall clear interior envelope; dimension numbers come from the registry.
    ymin, ymax = reg['dims']['bedroom_y'][0], reg['dims']['right_y'][1]
    xmin, xmax = reg['dims']['interior_x']
    for u in [ymin, ymax]:
        ax.plot([u, u], [-5.22, -5.85], color=MUTED, lw=0.55)
    ax.annotate('', xy=(ymin, -5.65), xytext=(ymax, -5.65),
                arrowprops={'arrowstyle': '|-|', 'lw': 0.65, 'color': MUTED})
    text((ymin+ymax)/2, -5.99, f'{ymax-ymin:.2f} m', size=9.5, color=MUTED)
    for v in [-xmax, -xmin]:
        ax.plot([-7.22, -7.93], [v, v], color=MUTED, lw=0.55)
    ax.annotate('', xy=(-7.75, -xmax), xytext=(-7.75, -xmin),
                arrowprops={'arrowstyle': '|-|', 'lw': 0.65, 'color': MUTED})
    text(-8.13, 0, f'{xmax-xmin:.2f} m', size=9.5, color=MUTED, rotation=90)

    # A compact world-axis indicator makes the rotated presentation explicit.
    ou, ov = -5.4, -3.2
    ax.annotate('', xy=(ou+0.8, ov), xytext=(ou, ov),
                arrowprops={'arrowstyle': '->', 'lw': 0.7, 'color': MUTED})
    ax.annotate('', xy=(ou, ov-0.8), xytext=(ou, ov),
                arrowprops={'arrowstyle': '->', 'lw': 0.7, 'color': MUTED})
    text(ou+0.95, ov, 'y', size=9, color=MUTED)
    text(ou, ov-0.98, 'x', size=9, color=MUTED)

    # Figure legend, positioned outside the actual environment footprint.
    legend_y = -6.66
    for u, label, fc, ec in [(-6.65, '牆體', WALL, WALL),
                              (-2.50, '固定家具', FURN, EDGE),
                              (2.50, '操作設備', TINT, ACCENT)]:
        ax.add_patch(Rectangle((u, legend_y-0.095), 0.34, 0.19,
                               facecolor=fc, edgecolor=ec, lw=0.7))
        text(u+0.55, legend_y, label, size=9.3, ha='left')
    ax.plot([7.05, 7.55], [legend_y, legend_y], color=ACCENT, lw=0.8,
            linestyle=(0, (3, 2)))
    text(7.78, legend_y, '門扇開啟示意', size=9.3, ha='left')

    # Check that labels stay on the output canvas before exporting.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for label, artist in texts:
        box = artist.get_window_extent(renderer)
        assert box.x0 >= bounds.x0 and box.x1 <= bounds.x1, f'Horizontal text clipping: {label}'
        assert box.y0 >= bounds.y0 and box.y1 <= bounds.y1, f'Vertical text clipping: {label}'

    stem = OUT / 'simulation_environment_map'
    fig.savefig(stem.with_suffix('.pdf'), metadata={
        'Title': 'Indoor simulation environment and task areas',
        'Subject': 'Registered m8_house_r2 geometry; floor-plan illustration',
        'Creator': 'Matplotlib; DFKai-SB embedded'})
    fig.savefig(stem.with_suffix('.png'), dpi=360, facecolor='white')
    fig.savefig(stem.with_suffix('.svg'), metadata={'Date': None})
    plt.close(fig)

    meta = {
        'scene': reg['name'], 'source_registry': str(REG.relative_to(ROOT)),
        'registry_sha256': digest(REG), 'world': str(world.relative_to(ROOT)),
        'world_sha256': digest(world), 'door_asset': str(door_path.relative_to(ROOT)),
        'door_sha256': digest(door_path), 'font': 'DFKai-SB / 標楷體',
        'font_sha256': digest(FONT), 'presentation_transform': '(u,v)=(world_y,-world_x)',
        'interior_envelope_m': [ymax-ymin, xmax-xmin],
        'door_clear_width_m': clear_width, 'door_leaf_length_m': length,
        'internal_opening_width_m': opening[1]-opening[0],
        'source_wall_count': len(reg['walls']), 'door_asset_wall_sections': 2,
        'furniture_count': len(reg['furniture']), 'drawer_footprint': reg['drawer']['footprint'],
        'purpose': 'Chapter 2 environment layout; no measured paths or trial results',
        'door_arc': 'Conventional 90-degree opening symbol, not a measured angle or joint limit',
        'not_drawn': ['per-trial robot paths', 'walker paths', 'trial-specific unknown obstacles',
                      'cylinder coordinates', 'start/end coordinates'],
        'outputs': {p.name: digest(p) for p in [stem.with_suffix(s) for s in ['.pdf', '.png', '.svg']]}}
    (OUT/'simulation_environment_map_metadata.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'scene': reg['name'], 'font': font.get_name(),
                      'outputs': list(meta['outputs']), 'canvas_checks': 'PASS'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
