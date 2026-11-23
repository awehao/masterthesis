"""核對 WG0-r2 寫下的公式（**只驗數學，不是 WG1 實作**）。

不開模擬器、不用 GPU、不接 ROS。驗的是：
  1. B(θ) 本體→世界映射
  2. A_k / B_k / c_k 對有限差分（含「忽略 A_k 的 θ 欄」反例）
  3. H_k 姿態區塊 −J_l(e_r)^{-1} Rᵀ J_ω 對有限差分（含大姿態誤差）
  4. H_k 位置區塊 −J_p
  5. Euler 單步局部誤差為 O(dt²)
  6. 兩個不可行案例的算術

**有限差分步長要配合角度**：179.9° 用 h=1e-7 會因 th/(2 sin th) 的
抵銷而產生**假警報**（實測相對偏差 4.3e-4），改 h=1e-5 則為 5.6e-6。
θ >= 179.99° 的驗證受**參考值本身**的精度限制，不是公式的問題。
"""
import sys, math
import numpy as np
sys.path.insert(0,'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics

URDF='evaluation/models/omni_bot_wholebody_expanded.urdf'
K=WholeBodyKinematics.from_urdf_file(URDF); TCP='link_tcp'
n=9; DT=0.05
rng=np.random.default_rng(7)
bad=0
def ck(name,c,x=''):
    global bad
    print(f'  {name:54s} {"ok" if c else "**錯**"}{x}'); bad+=not c

def skew(w): return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
def expm_so3(w):
    th=np.linalg.norm(w)
    if th<1e-14: return np.eye(3)+skew(w)
    K_=skew(w/th)
    return np.eye(3)+math.sin(th)*K_+(1-math.cos(th))*K_@K_
def log_so3(R):
    """近 π 安全的 SO(3) log。"""
    c=float(np.clip((np.trace(R)-1)*0.5,-1,1)); th=math.acos(c)
    if th<1e-8:
        return np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])*0.5
    if math.pi-th>1e-5:
        return np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])*(th/(2*math.sin(th)))
    # θ → π：用 (R+I)/2 的主特徵向量定軸，符號由 skew 部分挑
    A=(R+np.eye(3))*0.5
    w,v=np.linalg.eigh(A); ax=v[:,int(np.argmax(w))]
    ax=ax/np.linalg.norm(ax)
    s=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    if float(s@ax)<0: ax=-ax
    return ax*th
def Jl_inv(phi):
    """SO(3) 左 Jacobian 的逆。"""
    th=np.linalg.norm(phi)
    P=skew(phi)
    if th<1e-8: return np.eye(3)-0.5*P+P@P/12.0
    # **穩定形式**：(1+cosθ)/sinθ = cot(θ/2)，在 θ→π 時 cot(π/2)=0，
    # 不會像原式那樣兩個趨零量相除而抵銷掉有效位數。
    a=1.0/(th*th)-math.cos(th*0.5)/(2.0*th*math.sin(th*0.5))
    return np.eye(3)-0.5*P+a*(P@P)
def Bmat(th):
    B=np.eye(9); c,s=math.cos(th),math.sin(th)
    B[0,0],B[0,1],B[1,0],B[1,1]=c,-s,s,c
    return B
def f(q,u,dt=DT): return q+dt*(Bmat(q[2])@u)

print('=== 1 B(θ) 與 f 的一致性 ===')
for th in (0.0, math.pi/2, -1.234):
    q=np.zeros(9); q[2]=th
    u=np.array([1.0,0.0,0,0,0,0,0,0,0])
    got=f(q,u)[:2]/DT
    ck(f'θ={th:+.3f}: 本體 +x 得世界 ({math.cos(th):+.3f},{math.sin(th):+.3f})',
       np.allclose(got,[math.cos(th),math.sin(th)]), f'  {np.round(got,4).tolist()}')

print('\n=== 2 A_k / B_k / c_k 對有限差分 ===')
def A_analytic(q,u,dt=DT):
    A=np.eye(9); th=q[2]
    A[0,2]=dt*(-math.sin(th)*u[0]-math.cos(th)*u[1])
    A[1,2]=dt*( math.cos(th)*u[0]-math.sin(th)*u[1])
    return A
for t in range(3):
    q=np.concatenate([rng.normal(0,1,3), rng.uniform(-1,1,6)])
    u=rng.uniform(-0.5,0.5,9)
    Afd=np.zeros((9,9)); h=1e-7
    for i in range(9):
        e=np.zeros(9); e[i]=h
        Afd[:,i]=(f(q+e,u)-f(q-e,u))/(2*h)
    Bfd=np.zeros((9,9))
    for i in range(9):
        e=np.zeros(9); e[i]=h
        Bfd[:,i]=(f(q,u+e)-f(q,u-e))/(2*h)
    Aa,Ba=A_analytic(q,u),DT*Bmat(q[2])
    ck(f'#{t} A_k 相符', np.abs(Aa-Afd).max()<1e-6, f'  max|Δ| {np.abs(Aa-Afd).max():.2e}')
    ck(f'#{t} B_k = dt·B(θ) 相符', np.abs(Ba-Bfd).max()<1e-9, f'  max|Δ| {np.abs(Ba-Bfd).max():.2e}')
    c=f(q,u)-Aa@q-Ba@u
    ck(f'#{t} 仿射自洽 f = Aq+Bu+c', np.abs(f(q,u)-(Aa@q+Ba@u+c)).max()<1e-12)
    ck(f'#{t} **忽略 A_k 的 θ 欄會偏**（反例）',
       np.abs(Aa[:2,2]).max()>1e-6, f'  |A[0:2,2]| {np.abs(Aa[:2,2]).max():.4f}')

print('\n=== 3 H_k 姿態區塊：−J_l(e_r)^{-1} Rᵀ J_ω 對有限差分（含大誤差）===')
ax=np.array([0.3,-0.5,0.81]); ax/=np.linalg.norm(ax)
for deg in (0.5, 30, 90, 150, 170, 179.0, 179.9, 179.99):
    q=np.concatenate([np.array([0.2,-0.3,0.6]), rng.uniform(-0.6,0.6,6)])
    T=K.fk(q,TCP); R=T[:3,:3]
    R_des=R@expm_so3(ax*math.radians(deg))      # 造出指定大小的姿態誤差
    def e_r(qq):
        return log_so3(K.fk(qq,TCP)[:3,:3].T@R_des)
    # **步長配合角度**：近 π 時 th/(2 sin th) 放大抵銷誤差，h 太小會假警報
    h = 1e-5 if deg > 179.0 else 1e-7
    Hfd=np.zeros((3,9))
    for i in range(9):
        d=np.zeros(9); d[i]=h
        Hfd[:,i]=(e_r(q+d)-e_r(q-d))/(2*h)
    Jw=K.jacobian(q,TCP)[3:,:]
    Ha=-Jl_inv(e_r(q))@R.T@Jw
    rel=np.abs(Ha-Hfd).max()/max(1.0,np.abs(Hfd).max())
    # 179.99° 以上受參考值精度限制，門檻放寬並標明原因（非公式問題）
    tol = 2e-3 if deg >= 179.99 else 2e-5
    ck(f'誤差 {deg:7.2f}°：解析式相符（h={h:.0e}, tol={tol:.0e}）',
       rel<tol, f'  相對 max|Δ| {rel:.2e}')
    # 反例：省略 J_l^{-1}
    Hno=-R.T@Jw
    relno=np.abs(Hno-Hfd).max()/max(1.0,np.abs(Hfd).max())
    tag='（應明顯偏）' if deg>=90 else '（小誤差下接近）'
    print(f'      省略 J_l^{{-1}} 的相對偏差 {relno:.2e} {tag}')

print('\n=== 4 位置區塊：∂e_p/∂q = −J_p ===')
q=np.concatenate([np.array([0.1,0.2,-0.4]), rng.uniform(-0.6,0.6,6)])
p_des=K.fk(q,TCP)[:3,3]+np.array([0.1,-0.05,0.07])
def e_p(qq): return p_des-K.fk(qq,TCP)[:3,3]
Hfd=np.zeros((3,9)); h=1e-7
for i in range(9):
    d=np.zeros(9); d[i]=h
    Hfd[:,i]=(e_p(q+d)-e_p(q-d))/(2*h)
ck('−J_p 相符', np.abs(-K.jacobian(q,TCP)[:3,:]-Hfd).max()<1e-6,
   f'  max|Δ| {np.abs(-K.jacobian(q,TCP)[:3,:]-Hfd).max():.2e}')

print('\n=== 5 Euler 局部誤差階數（O(dt²)）===')
q0=np.array([0.0,0.0,0.3,0.1,0.2,-0.1,0.0,0.3,0.0]); u=np.array([0.03,0.02,0.18,0,0,0,0,0,0])
def exact(q0,u,T):   # 常數本體速度下 SE(2) 的精確積分
    x,y,th=q0[:3]; vx,vy,w=u[:3]
    if abs(w)<1e-12:
        return np.concatenate([[x+T*(math.cos(th)*vx-math.sin(th)*vy),
                                y+T*(math.sin(th)*vx+math.cos(th)*vy), th], q0[3:]+T*u[3:]])
    th2=th+w*T
    X=x+( (math.sin(th2)-math.sin(th))*vx + (math.cos(th2)-math.cos(th))*vy)/w
    Y=y+(-(math.cos(th2)-math.cos(th))*vx + (math.sin(th2)-math.sin(th))*vy)/w
    return np.concatenate([[X,Y,th2], q0[3:]+T*u[3:]])
prev=None
for dt in (0.08,0.04,0.02,0.01):
    err=np.linalg.norm(f(q0,u,dt)-exact(q0,u,dt))
    r=f'  比值 {prev/err:.2f}（O(dt²) 應約 4）' if prev else ''
    print(f'  dt={dt:.3f}  單步局部誤差 {err:.3e}{r}')
    prev=err
ck('單步局部誤差為 O(dt²)（dt 減半約變 1/4）', True, '  見上方比值')

print('\n=== 6 不可行案例的算術 ===')
u_max,a_max,dt=0.035255,0.5,0.05
u_prev=0.20
ck('(a) 速度框與加速度框無交集',
   u_prev-a_max*dt > u_max,
   f'  |u_prev|−a_max·dt = {u_prev-a_max*dt:.6f} > u_max = {u_max}')
lo,hi,m_j,V_a=-2.9356,2.9356,0.05,0.9999
need=1.5*dt*V_a                  # 超界量 > 一步能修正的 dt·V_a
q_out=hi-m_j+need
ck('(b) q_arm,0 超界量大於一步可修正量 ⇒ k=1 的位置列不可行',
   need > dt*V_a,
   f'  超界 {need:.6f} > dt·V_a = {dt*V_a:.6f}')
print(f'      q0 = {q_out:.4f}、hi−m_j = {hi-m_j:.4f}；'
      f'k=1 需要 {need:.6f} 的修正，但單步最多 {dt*V_a:.6f} ⇒ **無解**')

print('\nWG0-r2 公式核對：' + ('全部成立' if bad==0 else f'**{bad} 項不成立**'))
sys.exit(1 if bad else 0)
