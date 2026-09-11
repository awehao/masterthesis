"""執行期核對：robot_state_publisher 實際載入的模型，與 Isaac 載入的那份是否一致。

離線比對兩個檔案只能證明「檔案內容相同」。要說「執行中的模型已統一」，必須
在跑起來之後，把 robot_state_publisher 的 robot_description 參數取回來，
和 Isaac 實際匯入的那個檔案比對。robot_state_publisher 會重新序列化 XML，
所以逐位元組比對會誤判；這裡比的是**結構**：非固定關節集合、link 集合、
以及 TCP 的固定變換鏈。

用法：
    python3 evaluation/check_model_runtime.py --file evaluation/models/omni_bot_manip.urdf \
        [--node /robot_state_publisher] [--tcp link_tcp]
"""
import argparse, math, sys, xml.etree.ElementTree as ET
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters


def parse(xml):
    r = ET.fromstring(xml)
    links = {l.get('name') for l in r.findall('link')}
    joints = {}
    for j in r.findall('joint'):
        o = j.find('origin')
        xyz = [float(v) for v in (o.get('xyz') or '0 0 0').split()] if o is not None else [0, 0, 0]
        joints[j.get('name')] = (j.get('type'), j.find('parent').get('link'),
                                 j.find('child').get('link'), tuple(xyz))
    return links, joints


def tcp_chain(joints, tcp, stop='link6'):
    up = {v[2]: (k, v) for k, v in joints.items()}
    out, cur = [], tcp
    while cur in up and cur != stop:
        k, v = up[cur]; out.append((k, v[3])); cur = v[1]
    return list(reversed(out)) if cur == stop else None


ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--node', default='/robot_state_publisher')
ap.add_argument('--param', default='robot_description')
ap.add_argument('--tcp', default='link_tcp')
ap.add_argument('--timeout', type=float, default=30.0)
a = ap.parse_args()

rclpy.init()
n = Node('check_model_runtime')
cli = n.create_client(GetParameters, f'{a.node}/get_parameters')
if not cli.wait_for_service(timeout_sec=a.timeout):
    print(f'  !! 等不到 {a.node}/get_parameters'); sys.exit(1)
req = GetParameters.Request(); req.names = [a.param]
fut = cli.call_async(req)
rclpy.spin_until_future_complete(n, fut, timeout_sec=a.timeout)
if fut.result() is None or not fut.result().values:
    print(f'  !! 取不到參數 {a.param}'); sys.exit(1)
live_xml = fut.result().values[0].string_value
n.destroy_node(); rclpy.shutdown()

if not live_xml.strip():
    print(f'  !! {a.node} 的 {a.param} 是空的'); sys.exit(1)

fl, fj = parse(open(a.file).read())
ll, lj = parse(live_xml)
fm = {k for k, v in fj.items() if v[0] != 'fixed'}
lm = {k for k, v in lj.items() if v[0] != 'fixed'}

print(f'  檔案   {a.file}')
print(f'  執行中 {a.node} 的 {a.param}（{len(live_xml)} 字元）')
print(f'  非固定關節  檔案 {len(fm)}  執行中 {len(lm)}')
print(f'  link 數     檔案 {len(fl)}  執行中 {len(ll)}')

bad = 0
if fm != lm:
    print(f'  !! 非固定關節不同：只在檔案 {sorted(fm-lm)}；只在執行中 {sorted(lm-fm)}'); bad += 1
if fl != ll:
    print(f'  !! link 不同：只在檔案 {sorted(fl-ll)[:8]}；只在執行中 {sorted(ll-fl)[:8]}'); bad += 1

for tag, links, joints in (('檔案', fl, fj), ('執行中', ll, lj)):
    if a.tcp not in links:
        print(f'  !! {a.tcp} 不在{tag}的模型裡'); bad += 1
    else:
        ch = tcp_chain(joints, a.tcp)
        if ch is None:
            print(f'  !! {tag}的 {a.tcp} 無法由 link6 沿固定關節到達'); bad += 1
        else:
            tot = [sum(c[1][i] for c in ch) for i in range(3)]
            print(f'  {tag}的 {a.tcp}：link6 → ' + ' → '.join(c[0] for c in ch)
                  + f'   累計平移 {math.dist([0,0,0], tot):.5f} m')

print(f'  --- 結果：{"執行中的模型一致" if bad == 0 else "**不一致**"} ---')
sys.exit(0 if bad == 0 else 1)
