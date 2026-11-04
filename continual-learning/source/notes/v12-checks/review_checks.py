import numpy as np
from math import comb
class binom:
    @staticmethod
    def sf(k,n,p): return sum(comb(n,j)*p**j*(1-p)**(n-j) for j in range(k+1,n+1))
rng=np.random.default_rng(0)
def cg(Kf,b,x0,k=10,tol=1e-3):
    x=x0.copy(); r=b-Kf(x); p=r.copy(); rs=r@r; nb=np.linalg.norm(b)
    for i in range(k):
        if np.sqrt(rs)<tol*nb: break
        Kp=Kf(p); a=rs/(p@Kp); x+=a*p; r-=a*Kp; rsn=r@r; p=r+rsn/rs*p; rs=rsn
    return x, np.sqrt(rs)/nb
# 1. CG with omega=0, Adam-like D^-1 spread
n=5000
Dinv=10**rng.uniform(-6,-2,n)  # sqrt(vhat)+eps
m=rng.normal(size=n)*Dinv  # m ~ grad scale
eta=1e-3
b=-eta*m
d0=-eta*m/Dinv
x,res=cg(lambda v:Dinv*v,b,np.zeros(n))
print("omega=0 CG from 0: rel err vs Adam step",np.linalg.norm(x-d0)/np.linalg.norm(d0),"resid",res,"rho",(x@x)/(d0@d0),"descent",m@x)
# 2. warm start descent failure: warm start = previous step with m flipped
nn=200
A=rng.normal(size=(nn,nn)); Q,_=np.linalg.qr(A)
ev=10**rng.uniform(-3,2,nn); K=(Q*ev)@Q.T
fails=0
for trial in range(200):
    mm=rng.normal(size=nn); bb=-mm
    x0=-np.linalg.solve(K,-rng.normal(size=nn))*1.0  # previous step for an unrelated m
    x,res=cg(lambda v:K@v,bb,x0,k=10)
    if mm@x>=0: fails+=1
print("warm-start CG ascent cases /200:",fails)
fails0=0
for trial in range(200):
    mm=rng.normal(size=nn); x,res=cg(lambda v:K@v,-mm,np.zeros(nn),k=10)
    fails0+= mm@x>=0
print("zero-start ascent cases:",fails0)
# 3. rho-hat > 1 possible?
mx=0
for t in range(2000):
    d=10**rng.uniform(-2,2,3); Dm=np.diag(1/d)  # D^-1 = diag(1/d) -> D=diag(d)
    z=rng.normal(size=3); H=np.outer(z,z)
    mm=rng.normal(size=3)
    dw=-np.linalg.solve(np.diag(1/d)+5*H,mm); dz=-d*mm
    mx=max(mx,(dw@dw)/(dz@dz))
print("max rho-hat",mx)
# 4. M2b null space with single xi vs exact
din,dout,K_,nH=6,5,4,3
Z=rng.normal(size=(nH,din)); J=rng.normal(size=(nH,K_,dout)); xi=rng.normal(size=(nH,K_))
B=np.einsum('nkd,nk->nd',J,xi)
Gex=sum(np.kron(np.outer(Z[i],Z[i]),J[i].T@J[i]) for i in range(nH))
Gb=sum(np.kron(np.outer(Z[i],Z[i]),np.outer(B[i],B[i])) for i in range(nH))
print("rank exact",np.linalg.matrix_rank(Gex),"rank xi",np.linalg.matrix_rank(Gb))
# find Delta in null(Gb) not in null(Gex)
U,S,Vt=np.linalg.svd(Gb); N=Vt[np.linalg.matrix_rank(Gb):]
v=N[0]; print("xi-null vec: exact metric",v@Gex@v)
# traces
print("trace check",np.trace(Gb), sum((B[i]@B[i])*(Z[i]@Z[i]) for i in range(nH)))
# token cross terms: per-token treatment vs exact
P=3; Zt=rng.normal(size=(P,din)); Bt=rng.normal(size=(P,dout)); D=rng.normal(size=(dout,din))
print("exact (sum_p b^T D z)^2",(sum(Bt[p]@D@Zt[p] for p in range(P)))**2,"per-token sum",sum((Bt[p]@D@Zt[p])**2 for p in range(P)))
# 5. McNemar power
for nq in [40,50]:
    print(nq,"P(b>=5) at Delta=.125/.1:",binom.sf(4,nq,0.125),binom.sf(4,nq,0.1))
for nq in range(40,200):
    if binom.sf(4,nq,0.125)>=0.8: print("n for 80% power at 12.5pts:",nq);break
# 6. filter one batch: prior h=0.02, kappa=0.25
