"""Damage/progress frontier: quadratic exact-metric brake vs one-sided margin constraints (toy, first order)."""
import numpy as np, math, json, sys
sys.path.insert(0,'.')
rng=np.random.default_rng(11)
from scipy.optimize import minimize
def qp(C,d0,s):
    G=C.T@C; r=C.T@d0+s
    res=minimize(lambda mu:(.5*mu@G@mu+mu@r,G@mu+r),np.zeros(len(s)),jac=True,bounds=[(0,None)]*len(s),method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-12,'maxiter':5000})
    mu=res.x
    for _ in range(60):
        F=mu>1e-14; muF=np.zeros_like(mu)
        if F.any(): muF[F]=np.linalg.solve(G[np.ix_(F,F)],-r[F])
        g=G@muF+r
        if (muF>=-1e-13).all() and (g[~F]>=-1e-11).all(): mu=np.maximum(muF,0);break
        mu=np.maximum(muF,0); mu[(~F)&(g<0)]=1e-12
    return d0+C@mu
N,n,K=200,20,10; T=40
out={'quad':{},'qp':{},'centered_quad':{}}
for trial in range(T):
    J=rng.normal(size=(n,K,N))/math.sqrt(N); L=rng.normal(size=(n,K))*2; y=L.argmax(1)
    riv=[[j for j in np.argsort(-L[i]) if j!=y[i]] for i in range(n)]
    Mg=np.stack([J[i][y[i]]-J[i][j] for i in range(n) for j in riv[i]],1)
    a=rng.normal(size=N); d0=-a
    Hraw=sum(J[i].T@J[i] for i in range(n))/n
    Hc=sum(J[i].T@(np.eye(K)-1/K)@J[i] for i in range(n))/n
    for lam in [1,3,10,30,100,300,1000]:
        for nm,H in [('quad',Hraw),('centered_quad',Hc)]:
            d=np.linalg.solve(np.eye(N)+lam*H,-a)
            out[nm].setdefault(lam,[]).append((np.sum(np.maximum(0,-(Mg.T@d))),-(a@d)))
    for k in range(1,K):
        C=np.stack([J[i][y[i]]-J[i][j] for i in range(n) for j in riv[i][:k]],1)
        d=qp(C,d0,np.zeros(C.shape[1]))
        out['qp'].setdefault(k,[]).append((np.sum(np.maximum(0,-(Mg.T@d))),-(a@d)))
base=-(a@d0)
summ={nm:{str(k):[round(float(np.mean([p[0] for p in v])),2),round(float(np.mean([p[1] for p in v])),1)] for k,v in d_.items()} for nm,d_ in out.items()}
print('format: setting: [mean first-order margin damage, mean progress]; adam progress ~', round(float(base),1))
print(json.dumps(summ,indent=0))
