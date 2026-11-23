#!/usr/bin/env python3
"""WG4-B：預設 P 路徑的離線回歸（建構層）。不開模擬器。

以 mt_b1_01_M 實際的求解節點參數，分別建構「目前節點」與「加入 B1 之前的節點版本」（git show 取出），
比較：參數（新版只多 solver_kind／b1_kp 兩項且為預設值）、WGMPCConfig、節點全部純量／容器狀態
（ROS 控制代碼與牆鐘時間本來就逐次不同，列出但不算差異）。迴圈內的改動另由 diff 審查與
horizon_replay_check 對既有 P 趟的重播（核心未改）支持；本檢查不宣稱整個迴圈逐位元不變。

    git show 323b514d6:evaluation/wgmpc_wg2_node.py > /tmp/node_preb1.py
    python3 evaluation/wg4b_p_parity_check.py /tmp/node_preb1.py   （需 source ROS）
"""
import sys, json, argparse, importlib.util, dataclasses, numpy as np
BASE = sys.argv[1]
sys.argv=['x']; sys.path.insert(0,'evaluation')
import rclpy
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; spec.loader.exec_module(m); return m
NEW=load('evaluation/wgmpc_wg2_node.py','node_new'); OLD=load(BASE,'node_old')
A=json.load(open('evaluation/runs/mt_b1_01_M/align_solver.json'))['args']
rclpy.init()
def ns(mod):
    cap={}
    orig=argparse.ArgumentParser.parse_args
    def fake(self,*a,**k):
        cap['ns']=orig(self,[]); raise SystemExit
    argparse.ArgumentParser.parse_args=fake
    try: mod.main()
    except SystemExit: pass
    argparse.ArgumentParser.parse_args=orig
    n=cap['ns']
    for k,v in A.items():
        if hasattr(n,k): setattr(n,k,v)
    n.out=''
    return n
an, ao = ns(NEW), ns(OLD)
extra=set(vars(an))-set(vars(ao))
print('new-only args:', {k:getattr(an,k) for k in sorted(extra)})
print('shared args equal:', all(getattr(an,k)==getattr(ao,k) for k in vars(ao)))
nn=NEW.WGMPCNode(an); no=OLD.WGMPCNode(ao)
diff=[]
for k in sorted(set(vars(nn))|set(vars(no))):
    if k in ('_b1','_b1p','_n_b1_mu'): continue
    a_,b_=getattr(nn,k,'<absent>'),getattr(no,k,'<absent>')
    if not isinstance(a_,(int,float,str,bool,type(None),list,tuple,dict,np.ndarray)): continue
    try: eq=(np.array_equal(a_,b_) if isinstance(a_,np.ndarray) else bool(a_==b_))
    except Exception: eq=repr(a_)==repr(b_)
    if not eq: diff.append(k)
print('scalar/container state diffs (excluding _b1*):', diff)
print('cfg repr equal:', repr(nn.cfg)==repr(no.cfg))
print('new b1 flags:', nn._b1, nn._b1p, nn.b1_report())
