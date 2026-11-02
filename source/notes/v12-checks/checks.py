import numpy as np
np.random.seed(0)
# Claim 6: Adam-composed preconditioner can be uphill
g=np.array([1,0.1]); D=np.diag([1,100]); Pinv=np.array([[.5,-.4],[-.4,.5]])
print("eig Pinv",np.linalg.eigvalsh(Pinv)); U=D@g; print("Pinv U",Pinv@U,"g.PinvU",g@Pinv@U)
# fix: (D^-1 + w H)^-1 g is descent
H=np.linalg.inv(Pinv)-np.eye(2); print("H psd eig",np.linalg.eigvalsh(H))
d=np.linalg.solve(np.linalg.inv(D)+1.0*H,g); print("fixed g.d",g@d)
# Claim 3: K-FAC overprotects
Z=[np.eye(2)[0],np.eye(2)[1]]; B=Z
A=sum(np.outer(z,z) for z in Z)/2; G=sum(np.outer(b,b) for b in B)/2
for Dl in [np.array([[0,1],[1,0.]]),np.eye(2)]:
    exact=np.mean([(b@Dl@z)**2 for z,b in zip(Z,B)]); kfac=np.trace(G@Dl@A@Dl.T)
    print("Delta",Dl.ravel(),"exact",exact,"kfac",kfac)
# Claim 11 McNemar
print("p 3/0",0.5**3,"p 5/0",0.5**5)
# Claim 15 margin bound check by simulation
n=20000;K=5
f=np.random.randn(n,K)*2; y=np.random.randint(K,size=n); f[np.arange(n),y]+=3
df=np.random.randn(n,K)*0.4
def err(F): return np.mean(F.argmax(1)!=y)
m=f[np.arange(n),y]-np.max(np.where(np.eye(K)[y]==1,-1e9,f),1)
for eps in [0.3,0.6,1.0]:
    bound=err(f)+np.mean((m>0)&(m<=2*eps))+np.mean((df**2).sum(1))/eps**2
    print("eps",eps,"after",err(f+df),"bound",round(bound,3))
# Proposed fix: exact per-sample layer metric, CG solve, descent + floor blend
din,dout,n=6,4,10
Zs=np.random.randn(n,din); Bs=np.random.randn(n,dout); u=np.random.rand(n)
def Hv(Dm): s=np.einsum('ni,ij,nj->n',Bs,Dm,Zs); return np.einsum('n,n,ni,nj->ij',u,s,Bs,Zs)/n
Dg=np.random.rand(dout,din)*3+.1  # Adam diag scaling
gr=np.random.randn(dout,din); w=5.0
def op(X): return X/Dg + w*Hv(X)   # (D^-1 + w H) X
# CG
X=np.zeros_like(gr); r=gr-op(X); p=r.copy()
for _ in range(50):
    Ap=op(p); a=(r*r).sum()/(p*Ap).sum(); X+=a*p; rn=r-a*Ap
    if np.sqrt((rn*rn).sum())<1e-10: break
    p=rn+((rn*rn).sum()/(r*r).sum())*p; r=rn
# check against dense solve
M=np.zeros((din*dout,din*dout))
for k in range(din*dout):
    E=np.zeros(din*dout);E[k]=1;M[:,k]=op(E.reshape(dout,din)).ravel()
Xd=np.linalg.solve(M,gr.ravel()).reshape(dout,din)
print("CG vs dense err",np.abs(X-Xd).max(),"descent g.X",(gr*X).sum()>0)
D0=Dg*gr; rho=(X**2).sum()/(D0**2).sum(); print("passed energy",rho)
rmin=0.8
if rho<rmin:
    a2=((D0-X)**2).sum(); b=2*(X*(D0-X)).sum(); c=(X**2).sum()-rmin*(D0**2).sum()
    lam=(-b+np.sqrt(b*b-4*a2*c))/(2*a2); Xf=X+lam*(D0-X)
    print("lam",lam,"rho after",(Xf**2).sum()/(D0**2).sum(),"descent",(gr*Xf).sum()>0)
# Claim 7 snapshot drift
