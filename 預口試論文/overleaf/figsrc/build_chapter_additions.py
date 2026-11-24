"""Reproducible thesis additions from URDF, configuration and recorded Isaac events."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE/'ch03'))
from method_diagrams import plt, FONT, text, node, arrow, line, plain, robot_plan
from method_scene_geometry import Scene, frames, URDF, origin, mesh_for
from matplotlib.patches import Circle, Polygon, Rectangle

MANIFEST=[]

def save(fig,chapter,name,source):
    folder=HERE/chapter;folder.mkdir(exist_ok=True)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
    for artist in fig.findobj(plt.matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            b=artist.get_window_extent(renderer)
            assert b.x0>=-1 and b.y0>=-1 and b.x1<=w+1 and b.y1<=h+1,(name,artist.get_text())
    for ext in ('pdf','svg','png'):fig.savefig(folder/(name+'.'+ext),dpi=220)
    plt.close(fig)
    MANIFEST.append(dict(chapter=chapter,file=name,source=source,font='DFKai-SB / STIX',text_bounds='passed'))
    print(name,flush=True)

def coord_frames():
    fig=plt.figure(figsize=(9,5.6))
    ax=fig.add_axes([.015,.10,.61,.85]);sc=Scene(azimuth=-65,elevation=19)
    q=np.array([-.4,-.4,1.2,.2,.1,-.1]);ts=frames((0,0,0),q)
    sc.robot((0,0,0),q)
    tl=ts['base_link'].copy();tl[:3,3]+=[0,0,.2152]
    sc.cylinder(tl[:3,3],.03615,.0444,gray=.30)
    tc=ts['link6']
    for jointname in ('joint_eef','camera_link_joint','camera_color_joint','camera_color_optical_joint'):
        tc=tc@origin(URDF.find(f"joint[@name='{jointname}']"))
    te=ts['link_tcp']
    tmesh=ts['link6'];sc.faces(mesh_for('link_eef')@tmesh[:3,:3].T+tmesh[:3,3],.55)
    sc.draw(ax,margin=.16)
    tb=np.eye(4)
    targets=[(tb,'底盤平面基準 {B}',(.15,.10)),(ts['link_base'],'手臂基座 {A}',(.12,.86)),
             (tl,'光達 {L}',(.85,.12)),(te,'TCP {E}',(.86,.91)),(tc,'相機光學座標 {C}',(.86,.73))]
    for t,label,pos in targets:
        xy=sc.p(t[:3,3]);ax.plot(*xy,'ko',ms=3,zorder=16)
        for k,sty in enumerate(('-','--',':')):
            end=sc.p(t[:3,3]+.075*t[:3,k]);arrow(ax,xy,end,lw=1.0,ls=sty,scale=6)
        ax.annotate(label,xy=xy,xytext=pos,textcoords='axes fraction',fontproperties=FONT,
            fontsize=11,ha='center',va='center',zorder=30,
            bbox=dict(fc='white',ec='none',pad=2),arrowprops=dict(arrowstyle='-',lw=.8))
    top=plain(fig,[.67,.40,.30,.50],(-.36,.52),(-.36,.43),True)
    robot_plan(top,(0,0));top.plot(.11,0,'ko',ms=5)
    arrow(top,(0,0),(.42,0),scale=8);text(top,.44,.05,'$x_B$',11)
    arrow(top,(0,0),(0,.35),scale=8);text(top,-.055,.37,'$y_B$',11)
    text(top,-.09,-.05,'{B, L}',10);text(top,.17,-.055,'{A}',11)
    line(top,[(0,-.06),(0,-.40)],ls=':',lw=.7);line(top,[(.11,-.06),(.11,-.40)],ls=':',lw=.7)
    top.set_ylim(-.49,.43)
    top.annotate('',xy=(0,-.37),xytext=(.11,-.37),arrowprops=dict(arrowstyle='<->',lw=.8))
    text(top,.075,-.455,'前移 0.11 m',11)
    text(top,.04,.41,'俯視：四輪十字配置',12)
    a=plain(fig,[.65,.08,.33,.23],(0,1),(0,1))
    text(a,.5,.90,'$x$：實線　$y$：虛線　$z$：點線',10)
    text(a,.5,.62,r'${}^{B}p_A=(0.11,\;0,\;0.33)$ m',12)
    text(a,.5,.34,'base_link 離平面基準 0.05 m',10.5)
    text(a,.5,.06,'光達與車心同軸；高度不同',10.5)
    save(fig,'ch02','robot_coordinate_frames','expanded URDF; lidar.xacro; camera optical fixed-joint chain')

def command_chains():
    fig=plt.figure(figsize=(9,4.7));ax=plain(fig,[.02,.02,.96,.96],(0,16),(0,8))
    text(ax,1.8,7.4,'底盤速度命令',13)
    xs=[1.9,6.0,10.1,14.2]
    for x,title,detail in zip(xs,['速度輸出','命令平滑','雷射防護','輪級限制'],
        [r'$v_x,\;v_y,\;\omega$','加速度／斜坡','原始掃描與預測碰撞','輪速與輪速變化']):
        node(ax,x,6.2,3.3,1.25,title,detail,size=11.5)
    for a,b in zip(xs,xs[1:]):arrow(ax,(a+1.65,6.2),(b-1.65,6.2))
    text(ax,8.0,5.1,'最終放行速度回報 → 後續預測與命令差分',11)
    text(ax,1.8,4.15,'手臂位置命令',13)
    for x,title,detail in zip(xs,['設定點變化率','離散積分','關節 PD','實際關節狀態'],
        ['$u_a$',r'$s_{a,k+1}=s_{a,k}+\Delta t\,u_{a,k}$',r'$k_p=600,\;k_d=60$',r'$q_a,\;\dot q_a$']):
        node(ax,x,2.9,3.3,1.25,title,detail,size=11.5)
    for a,b in zip(xs,xs[1:]):arrow(ax,(a+1.65,2.9),(b-1.65,2.9))
    line(ax,[(14.2,2.275),(14.2,1.4),(10.1,1.4)]);arrow(ax,(10.1,1.4),(10.1,2.275))
    text(ax,12.3,1.0,'位置及速度回授',11)
    text(ax,6.5,.24,'PD 為 Isaac 模擬致動設定；設定點與實際位置分別保存',11)
    save(fig,'ch02','command_execution_chains','Isaac gains readback, run97; navigation command chain configuration')

def state_machine():
    fig=plt.figure(figsize=(8.8,5.7));ax=plain(fig,[.02,.02,.96,.96],(0,15),(0,10))
    titles=['起點與任務啟動','開門與穿門','圓柱移動中抓取','圓柱車尾收納',
            '開抽屜並觀測筆','取筆與車尾收納','關抽屜並退離','返程推門與通行','關門並返回起點']
    details=['地圖與狀態有效','視覺定位／抓持','抬升與持物確認','解綁／張爪／收臂',
             '把手定位／拉開','觀測／抓取／釋放','推回／釋放／退離','門扇觀測與導航','關閉確認／起點到位']
    coords=[(2.5,8.8),(7.5,8.8),(12.5,8.8),(12.5,5.9),(7.5,5.9),(2.5,5.9),
            (2.5,3),(7.5,3),(12.5,3)]
    for i,((x,y),a,b) in enumerate(zip(coords,titles,details)):
        node(ax,x,y,4.35,1.65,str(i+1)+'  '+a,b,size=11.4)
    for (x,y),(u,v) in zip(coords,coords[1:]):
        if abs(y-v)<.01:arrow(ax,(x+np.sign(u-x)*2.175,y),(u-np.sign(u-x)*2.175,v))
        else:arrow(ax,(x,y-.825),(u,v+.825))
    text(ax,7.5,7.35,'各階段依觀測、到位及抓持／釋放條件切換',11)
    text(ax,7.5,4.45,'導航與手臂動作可重疊；方框表示任務階段',11)
    node(ax,7.5,.7,12.0,.9,'任一階段：資訊過期、逾時或候選未通過 → 停止尾段／中止',size=11)
    save(fig,'ch02','residential_task_state_machine','current residential task specification; detailed controller transitions in chapter3')

def events(n):
    path=REPO/f'evaluation/runs/m8house_d2_unified_{n}/world.log';out=[]
    for ln in path.read_text().splitlines():
        if '[d2w] {' not in ln:continue
        try:out.append(json.loads(ln.split('[d2w] ',1)[1]))
        except json.JSONDecodeError:pass # truncated diagnostic lines are not stage boundaries
    return out

def gantt():
    es=events(97)
    def ts(what):return [e['t'] for e in es if e['what']==what]
    starts=ts('pick_start');stows=ts('pick_stowed')
    cuts=[0.,ts('pass_start')[0],starts[0],ts('pick_binned')[0],stows[0],
          ts('pick_dr_pulled')[0],stows[1],stows[2],starts[-1],stows[-1],ts('world_end')[0]]
    labels=['啟動、導航與開門','放手、穿門與前往茶几','圓柱抓取、抬升與放盒',
            '圓柱放盒後收臂','前往抽屜、定位與拉開','觀測筆、取筆、放盒與收臂',
            '關抽屜、退離與收臂','返程導航、通行與門扇觀測','關門、退離與收臂','返回起點與流程結束']
    assert np.all(np.diff(cuts)>0)
    fig,ax=plt.subplots(figsize=(9,5.5));fig.subplots_adjust(left=.37,bottom=.13,top=.92,right=.98)
    for i,(a,b) in enumerate(zip(cuts,cuts[1:])):
        ax.barh(i,b-a,left=a,height=.63,color=str(.82-.055*(i%3)),edgecolor='black',hatch='//' if i==6 else None)
        ax.text(b+2,i,f'{b-a:.1f} s',va='center',fontsize=10,fontproperties=FONT)
    ax.set_yticks(range(len(labels)),labels,fontproperties=FONT,fontsize=10.5);ax.invert_yaxis()
    ax.set_xlim(0,435);ax.set_xticks(range(0,401,50));ax.grid(axis='x',color='.86',lw=.6);ax.set_axisbelow(True)
    ax.set_xlabel('Isaac 模擬時間（s）',fontproperties=FONT,fontsize=12)
    ax.set_title('第 97 趟：以事件邊界彙整的任務時間',fontproperties=FONT,fontsize=13)
    stages=[dict(stage=s,start_s=a,end_s=b,duration_s=round(b-a,2)) for s,a,b in zip(labels,cuts,cuts[1:])]
    p=HERE/'ch02/run97_stage_times.json';p.write_text(json.dumps(dict(
        run='m8house_d2_unified_97',clock='Isaac simulation time; world.log t (not ctrl_t)',
        source='evaluation/runs/m8house_d2_unified_97/world.log',
        source_sha256=hashlib.sha256((REPO/'evaluation/runs/m8house_d2_unified_97/world.log').read_bytes()).hexdigest(),
        end_event_s=cuts[-1],simulator_final_s=388.71,
        limitation='drawer residual opening 11.1 mm; completed sequence is not full-quality success',stages=stages),ensure_ascii=False,indent=2)+'\n')
    save(fig,'ch02','run97_task_gantt','Isaac run97 world.log t; exact stage boundaries in run97_stage_times.json')

def wheel_polygon():
    r,L,wmax=.05,.245,5.55;c=r*wmax
    fig,axes=plt.subplots(1,2,figsize=(8.9,4.55));fig.subplots_adjust(left=.09,right=.97,bottom=.21,top=.82,wspace=.36)
    ax=axes[0]
    for w,style,gray in [(0,'--','none'),(.6,'-', '.88')]:
        h=c-L*abs(w);ax.add_patch(Rectangle((-h,-h),2*h,2*h,fc=gray,ec='black',ls=style,lw=1.4))
    ax.set_aspect('equal');ax.set_xlim(-.32,.32);ax.set_ylim(-.32,.32)
    ax.set_xticks([-.3,0,.3]);ax.set_yticks([-.3,0,.3])
    ax.set_xlabel('$v_x$ (m/s)',fontsize=12);ax.set_ylabel('$v_y$ (m/s)',fontsize=12)
    ax.text(0,.225,r'$\omega=0$',ha='center',fontsize=12)
    ax.text(0,0,r'$\omega=0.6$'+'\nrad/s',ha='center',va='center',fontsize=11)
    ax.set_title('固定旋轉速度：平移方框收縮',fontproperties=FONT,fontsize=12,pad=14)
    ax=axes[1];wm=c/L
    ax.add_patch(Rectangle((-c,-wm),2*c,2*wm,fill=False,ec='black',ls='--',lw=1.2))
    ax.add_patch(Polygon([(c,0),(0,wm),(-c,0),(0,-wm)],fc='.87',ec='black',lw=1.4))
    ax.plot(.24,.8,'kx',ms=8,mew=1.5);ax.annotate('輪速超限',xy=(.24,.8),xytext=(.04,.87),
        fontproperties=FONT,fontsize=11,arrowprops=dict(arrowstyle='-',lw=.7))
    ax.set_xlim(-.32,.32);ax.set_ylim(-1.3,1.3)
    ax.set_xticks([-.3,0,.3]);ax.set_yticks([-1,0,1])
    ax.set_xlabel('$v_x$ (m/s)',fontsize=12);ax.set_ylabel(r'$\omega$ (rad/s)',fontsize=12)
    ax.set_title('$v_y=0$：平移與旋轉共同限制',fontproperties=FONT,fontsize=12,pad=14)
    for ax in axes:ax.axhline(0,color='.65',lw=.5,zorder=0);ax.axvline(0,color='.65',lw=.5,zorder=0);ax.tick_params(labelsize=10)
    fig.text(.5,.055,'虛線：各軸獨立速度框　　灰色：輪速限制下的可行區域',ha='center',fontproperties=FONT,fontsize=12)
    save(fig,'ch03','omni_wheel_velocity_feasible_set','chapter3 W matrix; r=.05 m, L=.245 m, wheel limit5.55 rad/s; no margins')

def vision_pipeline():
    fig=plt.figure(figsize=(9,4.75));ax=plain(fig,[.02,.02,.96,.96],(0,16),(0,8))
    xs=[2.1,8.,13.9]
    for x,a,b in zip(xs,['RGB 影像','Mask R-CNN','類別與實例遮罩'],['影像擷取時間 $t_c$','六類目標分割','分數及遮罩門檻均為 0.5']):node(ax,x,6.7,3.8,1.35,a,b,size=11.3)
    for a,b in zip(xs,xs[1:]):arrow(ax,(a+1.9,6.7),(b-1.9,6.7))
    node(ax,2.1,3.9,3.8,1.35,'對齊的深度與內參',r'$D(u,v),\;K$')
    node(ax,8.,3.9,4.6,1.35,'選取有效深度點','門把：遮罩外框擴張 ROI',size=11)
    arrow(ax,(4,3.9),(5.7,3.9));line(ax,[(13.9,6.025),(13.9,5.25),(8,5.25)]);arrow(ax,(8,5.25),(8,4.575))
    node(ax,13.6,3.9,4.35,1.35,'幾何估測','平面／圓柱／直線與方向',size=11)
    arrow(ax,(10.3,3.9),(11.425,3.9))
    node(ax,4.0,1.15,7.1,1.35,'擷取時刻的機器人位姿與外參','以 $t_c$ 對齊底盤、關節與相機姿態',size=11)
    node(ax,12.,1.15,6.0,1.35,'帶時間戳的世界座標目標','位置／姿態／來源時間／有效性',size=11)
    arrow(ax,(7.55,1.15),(9.,1.15));line(ax,[(13.6,3.225),(13.6,1.825)]);arrow(ax,(13.6,2.0),(13.6,1.825))
    save(fig,'ch03','mask_rcnn_target_pipeline','house_seg.py and Isaac online geometry detectors; handle ROI exception included')

def approach_profile():
    es=events(101);packet=next(e for e in es if e['what']=='pick_app')['app']['flow']
    assert any(e['what']=='pick_jblend_start' for e in es)
    end=next(e for e in es if e['what']=='pick_phase' and e.get('phase')=='CLOSE')
    assert abs(end['ep_mm']-1.27)<1e-6
    d=np.linspace(0,1.7,500);smooth=lambda x:x*x*(3-2*x)
    p=np.clip((1.2-d)/(.9),0,1);beta=p**3*(10-15*p+6*p*p)
    x=np.clip((d-packet['r_in'])/(packet['r_turn']-packet['r_in']),0,1)
    vc=packet['v_cap']+(.26-packet['v_cap'])*smooth(x)
    fig,axes=plt.subplots(2,1,figsize=(8.2,5.5),sharex=True);fig.subplots_adjust(left=.13,right=.95,top=.92,bottom=.18,hspace=.35)
    axes[0].plot(d,vc,'k',lw=1.8);axes[0].set_ylim(0,.30)
    axes[0].set_ylabel('平移逐軸限速（m/s）',fontproperties=FONT,fontsize=11)
    axes[0].set_title('圓柱接近段的距離剖面（設計曲線）',fontproperties=FONT,fontsize=13)
    axes[0].text(.85,.075,'近段 0.06 m/s',fontproperties=FONT,fontsize=11)
    axes[1].plot(d,beta,'k',lw=1.8);axes[1].axhline(.55,ls='--',color='.5',lw=.9)
    axes[1].text(1.1,.61,r'$\beta_0=0.55$：開始後段併線',fontproperties=FONT,fontsize=11)
    axes[1].set_ylabel(r'距離展開進度 $\beta_{dist}$',fontproperties=FONT,fontsize=11)
    axes[1].set_xlabel('底盤至操作位置的剩餘距離（m）　接近方向 →',fontproperties=FONT,fontsize=11)
    for ax in axes:ax.set_xlim(1.7,0);ax.set_xticks([0,.4,.8,1.2,1.6]);ax.grid(color='.87',lw=.6);ax.tick_params(labelsize=10)
    axes[0].set_yticks([0,.1,.2,.3]);axes[1].set_yticks([0,.5,1])
    fig.text(.5,.055,'實際進度另受最短展開時間限制；最終命令仍經導航、避障及輪級限制',
             ha='center',fontproperties=FONT,fontsize=11)
    (HERE/'ch03/approach_profile_evidence.json').write_text(json.dumps(dict(run='m8house_d2_unified_101',
        branch_event=next(e for e in es if e['what']=='pick_jblend_start'),close_event=end,
        flow=packet,distance_progress=dict(outer_m=1.2,inner_m=.3,beta_merge=.55),
        curve='design envelope, not measured velocity or a reconstruction of logged beta',
        scope='cylinder segment completed; full run failed on return door perception'),ensure_ascii=False,indent=2)+'\n')
    save(fig,'ch03','mobile_approach_distance_profile','run101 flow packet + door_pick_ctrl.py distance and speed envelope; design curve')

def main():
    coord_frames();command_chains();state_machine();gantt();wheel_polygon();vision_pipeline();approach_profile()
    (HERE/'chapter_additions_manifest.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
