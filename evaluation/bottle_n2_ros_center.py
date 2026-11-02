#!/usr/bin/env python3
"""Smac2D 路徑座標換算到 costmap 格心慣例（版本限定、可選；Codex 20261102_083817／085048）。bottle_n2_ros.PlanRos 原樣保留。
已確認（results/bottle/planner_offset_check.json、持瓶路徑承接偏差_定位與最小修正設計.md）：nav2_smac_planner 1.3.12 的 SmacPlanner2D 以 getWorldCoords＝原點＋mx·res
（無 +0.5）輸出，整條路徑相對 costmap 格心慣例偏 (−res/2, −res/2)。
verify()：起跑前查 planner_server 實際參數（GridBased.plugin、GridBased.downsample_costmap，經 /planner_server/get_parameters）、已安裝套件版本
（/opt/ros/jazzy/share/nav2_smac_planner/package.xml）、當趟地圖解析度與原點；全部符合 SmacPlanner2D／1.3.12／未降採樣才 enabled，否則照舊不換算並記原因。
PlanRosCenter.poll()：每個新回覆只換算一次（P＋res/2，res＝地圖解析度，且須等於當下代價圖解析度、原點一致，否則該回覆不換算並記錄）；
原始路徑存 P_raw、換算資訊存 conv；不移動地圖、定位、任務目標或視覺座標。之後的 validate_plan、平滑、承接、代價核對全部照舊作用在換算後路徑。
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml

import bottle_n2_ros as NR

REQ = dict(plugin='nav2_smac_planner::SmacPlanner2D', version='1.3.12', downsample=False)
PKG_XML = '/opt/ros/jazzy/share/nav2_smac_planner/package.xml'


def smac_version():
    try:
        return ET.parse(PKG_XML).getroot().findtext('version')
    except Exception as e:
        return f'unreadable:{e!r}'


def verify(node, map_yaml, timeout=5.0):
    import rclpy
    from rcl_interfaces.srv import GetParameters
    out = dict(required=REQ, version=smac_version(), plugin=None, downsample=None, map_res=None, map_origin=None, enabled=False, why=None)
    try:
        my = yaml.safe_load(open(map_yaml))
        out['map_res'], out['map_origin'] = float(my['resolution']), [float(v) for v in my['origin'][:2]]
    except Exception as e:
        out['why'] = f'map_yaml:{e!r}'
        return out
    cli = node.create_client(GetParameters, '/planner_server/get_parameters')
    if not cli.wait_for_service(timeout_sec=timeout):
        out['why'] = 'param_service_unavailable'
        return out
    req = GetParameters.Request()
    req.names = ['GridBased.plugin', 'GridBased.downsample_costmap']
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    r = fut.result()
    if r is None or len(r.values) != 2:
        out['why'] = 'param_query_failed'
        return out
    out['plugin'], out['downsample'] = r.values[0].string_value, bool(r.values[1].bool_value)
    bad = [k for k, v in (('plugin', out['plugin']), ('version', out['version']), ('downsample', out['downsample'])) if v != REQ[k]]
    out['enabled'] = not bad
    out['why'] = None if not bad else 'mismatch:' + ','.join(bad)
    return out


class PlanRosCenter(NR.PlanRos):
    def __init__(self, node, conv):
        super().__init__(node)
        self.conv, self.conv_log = dict(conv), []

    def poll(self):
        before = self.result
        super().poll()
        r = self.result
        if r is None or r is before or r.get('conv') is not None:
            return
        P = r.get('P')
        info = dict(id=r.get('id'), applied=False, why=None)
        g = self.grid
        if not self.conv.get('enabled'):
            info['why'] = 'disabled:' + str(self.conv.get('why'))
        elif P is None or len(P) == 0 or r.get('frame') != 'map':
            info['why'] = 'no_path_or_frame'
        elif g is None or abs(g.res - self.conv['map_res']) > 1e-6 or abs(g.ox - self.conv['map_origin'][0]) > 1e-6 or abs(g.oy - self.conv['map_origin'][1]) > 1e-6:
            info['why'] = 'costmap_mismatch' if g is not None else 'no_costmap'
            if g is not None:
                info.update(costmap_res=g.res, costmap_origin=[g.ox, g.oy])
        else:
            h = 0.5 * self.conv['map_res']
            r['P_raw'] = np.asarray(P, float).round(5).tolist()
            r['P'] = np.asarray(P, float) + h
            info.update(applied=True, shift=[h, h], n=len(P), raw_first=r['P_raw'][0], conv_first=r['P'][0].round(5).tolist())
        r['conv'] = info
        self.conv_log.append(dict(info, P_raw=r.get('P_raw')))
