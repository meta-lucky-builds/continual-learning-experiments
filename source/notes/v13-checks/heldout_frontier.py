# Held-out frontier check, written by the independent v1.3 reviewer (originally chk2.py). Damage measured on 200 items not used to build the protection.
import numpy as np, math, json
from scipy.optimize import nnls
rng=np.random.default_rng(11)
def qp_exact(C,d0,s):
    # B=I. Dual: min .5 mu^T G mu + mu^T r, mu>=0 ; solve by projected Newton/active set via scipy lsq? use quadprog-like loop with cvx check
    from scipy.optimize import minimize
    G=C.T@C; r=C.T@d0+s
    res=minimize(lambda mu:(.5*mu@G@mu+mu@r,G@mu+r),np.zeros(len(s)),jac=True,bounds=[(0,None)]*len(s),method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-12,'maxiter':5000})
    mu=res.x
    for _ in range(60):
        F=mu>1e-14; muF=np.zeros_like(mu)
        if F.any(): muF[F]=np.linalg.solve(G[np.ix_(F,F)],-r[F])
        g=G@muF+r
        if (muF>=-1e-13).all() and (g[~F]>=-1e-11).all(): mu=np.maximum(muF,0);break
        mu=np.maximum(muF,0); mu[(~F)&(g<0)]=1e-12
    d=d0+C@mu
    return d, float(np.max(-(C.T@d)-s)), float(np.max(np.abs(mu*(C.T@d+s))))
N,n,K=200,20,10; T=20; r_sh=20
adam=[]; out={}
def add(k,v): out.setdefault(k,[]).append(v)
for trial in range(T):
    # structured Jacobians: shared low-rank part so pool constraints can transfer to held-out items
    V=rng.normal(size=(r_sh,N))/math.sqrt(N)
    def mkJ(m): return np.einsum('ikr,rn->ikn',rng.normal(size=(m,K,r_sh)),V)*0.8+rng.normal(size=(m,K,N))/math.sqrt(N)*0.6
    J=mkJ(n); Jh=mkJ(200)
    L=rng.normal(size=(n,K))*2; y=L.argmax(1); Lh=rng.normal(size=(200,K))*2; yh=Lh.argmax(1)
    def Mg_of(J,y): return np.stack([J[i][y[i]]-J[i][j] for i in range(len(y)) for j in range(K) if j!=y[i]],1)
    Mg=Mg_of(J,y); Mh=Mg_of(Jh,yh)
    mars=np.array([L[i,y[i]]-L[i,j] for i in range(n) for j in range(K) if j!=y[i]])
    a=rng.normal(size=N); d0=-a; adam.append(-(a@d0))
    H=sum(J[i].T@(np.eye(K)-1/K)@J[i] for i in range(n))/n
    dmg=lambda M,d: float(np.sum(np.maximum(0,-(M.T@d))))/M.shape[1]*Mg.shape[1]
    add('adam',(dmg(Mg,d0),dmg(Mh,d0),-(a@d0)))
    for lam in [3,30,300,3000]:
        d=np.linalg.solve(np.eye(N)+lam*H,-a); add(f'quad{lam}',(dmg(Mg,d),dmg(Mh,d),-(a@d)))
    d,fv,cs=qp_exact(Mg,d0,np.zeros(Mg.shape[1])); add('cons_all_s0',(dmg(Mg,d),dmg(Mh,d),-(a@d))); add('kkt',(fv,cs,0))
    s=0.2*np.maximum(0,mars-0.5); d,fv,cs=qp_exact(Mg,d0,s); add('cons_all_slack',(dmg(Mg,d),dmg(Mh,d),-(a@d)))
print('mean adam progress',np.mean(adam),' (spec quotes 187)')
print(json.dumps({k:[round(float(np.mean([p[i] for p in v])),2) for i in range(3)] for k,v in out.items()},indent=0))
