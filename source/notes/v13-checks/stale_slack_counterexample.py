import numpy as np, math
from math import comb
from scipy.optimize import minimize
rng=np.random.default_rng(0)
def qp(Bd,C,d0,s):
    G=C.T@(Bd[:,None]*C); r=C.T@d0+s
    if np.all(r>=0): return d0
    # exact active set by enumeration via NNLS-like loop
    res=minimize(lambda mu:(.5*mu@G@mu+mu@r,G@mu+r),np.zeros(len(s)),jac=True,bounds=[(0,None)]*len(s),method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-13,'maxiter':5000})
    return d0+Bd*(C@res.x)
# 1. (Q) is not a projection: idempotence fails
B=np.ones(1); H=np.array([[1.]]); lam=1.
P=lambda d0: np.linalg.solve(np.diag(1/B)+lam*H, d0/B)  # Delta* as function of Delta0 = -Ba
d0=np.array([1.]); print('Q prox(d0)=',P(d0),' prox(prox(d0))=',P(P(d0)))
# 2. Stale slack: linear margin m=theta, Adam pushes down, s fixed for R_pre=25 steps
for R in [1,25]:
    th=2.0; mf=.5; ks=.2; C=np.array([[1.]])
    for t in range(100):
        if t%R==0: s=np.array([ks*max(0,th-mf)])
        d=qp(np.ones(1),C,np.array([-1.]),s); th+=d[0]
    print('R_pre',R,'margin after 100 steps',round(th,4),'(floor',mf,')')
# 3. Uniform rescale preserves feasibility, coordinate clip does not
C=np.array([[1.],[-1.]]); s=np.array([0.]); d=np.array([5.,5.]) # c^T d = 0 feasible
dc=np.clip(d,-1,10); print('clip: c^T d before',C[:,0]@d,'after clip',C[:,0]@np.clip(np.array([5.,-0.]),-1,1) if False else C[:,0]@np.minimum(d,[1,10]))
# 4. Markov bound with new label
fref=np.array([1.,0.]); fnow=np.array([1.,0.,2.]); d=fnow[:2]-fref; Pd=d-d.mean()
print('new-label: ||P d||=',np.linalg.norm(Pd),'wrong now:',fnow.argmax()!=0)
# 5. L_min equality / multi-state
def Lmin(h,pi): O=pi/(1-pi); return O*(h*O+1-h)/((1-h)*O+h)
h=.02; L=Lmin(h,.9); O=h/(1-h); mx=0
for t in range(200000): O=L*((1-h)*O+h)/(h*O+1-h); mx=max(mx,O/(1+O))
print('at L=L_min exactly: sup posterior',mx,'reaches 0.9?',mx>=0.9)
# multi-state filter: 3 contexts + new_in + new_cond(c) ; target context gets constant LR L vs all others (others equal)
def multi(L,h=.02,pinew=.1,nctx=3,steps=5000):
    S=nctx+2; p=np.zeros(S); p[0]=1  # start in context 0; target = context 1
    def T():
        M=np.zeros((S,S))
        for c in range(nctx):
            M[c,c]=1-h
            for c2 in range(nctx):
                if c2!=c: M[c,c2]=h*(1-pinew)/(nctx-1)
            M[c,nctx]=h*pinew/2; M[c,nctx+1]=h*pinew/2
        for k in (nctx,nctx+1): M[k,k]=1-h; M[k,:nctx]=h/nctx  # assumption for novelty rows
        return M
    M=T(); lik=np.ones(S); lik[1]=L; mx=0
    for t in range(steps):
        p=p@M; p=p*lik; p/=p.sum(); mx=max(mx,p[1])
    return mx
for lL in [0.15,0.166,0.17,0.2,0.25]:
    print('multi-state log L',lL,'max posterior of target',round(multi(math.exp(lL)),4))
# 6. McNemar minimal n (one-sided exact), (0.125,0)
def pw(n,pp,pm,alpha=.05):
    tot=0.
    for b in range(n+1):
        for c in range(n-b+1):
            nd=b+c
            if nd==0: continue
            tail=sum(comb(nd,k) for k in range(b,nd+1))/2**nd
            if tail<alpha: tot+=comb(n,b)*comb(n-b,c)*pp**b*pm**c*(1-pp-pm)**(n-b-c)
    return tot
print('power n=40..60:',[(n,round(pw(n,.125,0),3)) for n in range(40,61)])
def env(n): return min(pw(n,(dd+.125)/2,(dd-.125)/2) for dd in np.linspace(.125,.4,12))
print('envelope 160..175:',[(n,round(env(n),3)) for n in range(160,176)])
