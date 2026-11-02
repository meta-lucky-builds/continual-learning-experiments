import numpy as np, math
rng=np.random.default_rng(0)
# 1. Woodbury exact solve: (D^-1 + w/n G G^T) d = -eta m
P,n=400,16; Dinv=np.exp(rng.uniform(np.log(1e-6),np.log(1e-2),P)); D=1/Dinv
G=rng.normal(size=(P,n)); m=rng.normal(size=P); eta=1e-3
for w in [0.0,0.1,10,1e4]:
    Dm=D*m; S=(n/w)*np.eye(n)+G.T@(D[:,None]*G) if w>0 else None
    d=-eta*(Dm-(D[:,None]*G)@np.linalg.solve(S,G.T@Dm)) if w>0 else -eta*Dm
    K=np.diag(Dinv)+w/n*G@G.T; dd=np.linalg.solve(K,-eta*m)
    print('woodbury w',w,'relerr',np.linalg.norm(d-dd)/np.linalg.norm(dd),'descent',m@d<0, 'equals adam at 0', np.allclose(d,-eta*Dm) if w==0 else '')
# 2. PCG from 0 with preconditioner D: descent every iterate; w=0 exact in 1 iter
def pcg(Kmv,b,Mi,k):
    x=np.zeros_like(b); r=b.copy(); z=Mi*r; p=z.copy(); out=[]
    for _ in range(k):
        Kp=Kmv(p); a=(r@z)/(p@Kp); x=x+a*p; rn=r-a*Kp; zn=Mi*rn
        out.append(x.copy()); bt=(rn@zn)/(r@z); p=zn+bt*p; r,z=rn,zn
        if np.linalg.norm(r)<1e-14: break
    return out
bad=0
for t in range(200):
    A=rng.normal(size=(P,30)); H=A@A.T/30; w=10**rng.uniform(-2,3)
    Kmv=lambda v: Dinv*v+w*H@v; mm=rng.normal(size=P)
    for x in pcg(Kmv,-eta*mm,D,10):
        if mm@x>=0: bad+=1
print('PCG nonDescent iterates',bad)
x=pcg(lambda v:Dinv*v,-eta*m,D,1)[0]; print('w=0 PCG 1 iter equals adam', np.allclose(x,-eta*D*m))
# 3. token coupling: logits f = Wh * mean_p( relu? ) linear test: f = V mean_p(W z_p); grad of xi^T f wrt W = sum_p (V^T xi /P) z_p^T
din,dout,Pt,K=5,4,3,3; W=rng.normal(size=(dout,din)); V=rng.normal(size=(K,dout)); Z=rng.normal(size=(din,Pt))
f=lambda W: V@(W@Z).mean(1); Dl=rng.normal(size=(dout,din))
exact=np.linalg.norm(f(W+Dl)-f(W))**2
est=np.mean([ (np.sum((V.T@xi/Pt)[:,None]*Z.sum(1)[None,:]*Dl))**2 for xi in rng.normal(size=(200000,K))])
print('token-coupled exact',exact,'xi-gradient estimate',est)
# 4. power: n for 80% power at delta, exact one-sided McNemar no regressions
from math import comb
def pw(n,dl,alpha=0.05):
    # need b>=5 when no regressions; general: discordant b improvements, c regressions; assume only improvements w.p. dl
    return sum(comb(n,k)*dl**k*(1-dl)**(n-k) for k in range(5,n+1))
for n in [40,53,60,80]: print('n',n,'power@0.125',round(pw(n,0.125),3),'@0.1',round(pw(n,0.1),3))
