import json, math, numpy as np
def T3(p):
    c,s=np.cos(p[2]),np.sin(p[2]); T=np.eye(4); T[:2,:2]=[[c,-s],[s,c]]; T[:2,3]=p[:2]; return T
def ang(A,B): return math.degrees(math.acos(max(-1,min(1,(np.trace(A.T@B)-1)/2))))
Th=np.array(json.load(open('evaluation/runs/m8house_far_capture_3/truth.json'))['T_handle_start'])
out={}
for run in ('m8house_far_door_1','m8house_far_door_2','m8house_far_door_3'):
    R=f'evaluation/runs/{run}/'; J=json.load(open(R+'ros_merge.json')); F=json.load(open(R+'frames_meta.json'))['frames']
    Z=np.load(R+'ros_merge.npz'); Rr=Z['R']; c=[str(x) for x in Z['cols']]; i=c.index
    an=J['anchor']; Tf=np.array(an['T_odom_handle'])
    t_obs0=[e['t'] for e in J['events'] if e['what']=='arm_prep_done_observe'][0]
    tt=Rr[:,0]; al=Rr[:,i('along')]
    res={}
    for gate_deg in (3.0,10.0):
        hist=[]; lock=None; tlast=-1
        for f in F:
            if f['t_obs']<t_obs0-0.1: continue
            a=float(np.interp(f['t_processed'],tt,al))
            if not f['valid']:
                pass
            else:
                To=T3(f['odom_base_at_capture'])@np.array(f['T_base_handle'])
                dp=np.linalg.norm(To[:3,3]-Tf[:3,3]); da=ang(Tf[:3,:3],To[:3,:3])
                if hist and f['t_obs']-hist[-1][0]>0.3: hist=[]
                if dp<=0.02 and da<=gate_deg: hist=(hist+[(f['t_obs'],To)])[-3:]
                else: hist=[]
            if a<=0.25 and lock is None:
                ok=len(hist)==3 and f['t_processed']-hist[-1][0]<=0.3
                if ok:
                    P=np.array([h[1][:3,3] for h in hist]); sp=max(np.linalg.norm(P[x]-P[y]) for x in range(3) for y in range(3))
                    H=[h[1][:3,:3] for h in hist]; pw=max(ang(H[x],H[y]) for x in range(3) for y in range(x+1,3))
                    U,_,V=np.linalg.svd(sum(H)); Rm=U@np.diag([1,1,np.linalg.det(U@V)])@V
                    lock=dict(along=round(a,3),spread_mm=round(sp*1e3,2),pairwise_deg=round(pw,3),ang_budget_wo_tcp=round(0.885+pw,3),eval_ang_err=round(ang(Rm,Th[:3,:3]),3),eval_pos_err_mm=round(float(np.linalg.norm(np.median(P,axis=0)-Th[:3,3]))*1e3,2),ok=bool(sp<=0.005))
                else:
                    lock=dict(along=round(a,3),ok=False,n_hist=len(hist))
        res['gate_%g'%gate_deg]=lock
    res['far_ref_ang_err_eval']=round(ang(Tf[:3,:3],Th[:3,:3]),2)
    out[run]=res; print(run,res)
json.dump(dict(note='三趟實際影格離線重播 OBSERVE 規則（鎖定點 0.25）：粗差閘門姿態門檻 3°（現行，對遠處固定參考）對 10°（改用目標關聯容許）；eval 欄為事後真值，只評估',runs=out),open('第八次進度報告素材/資料/far_observe_gate_replay.json','w'),ensure_ascii=False,indent=1)
