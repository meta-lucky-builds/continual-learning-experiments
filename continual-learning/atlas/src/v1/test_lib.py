import numpy as np, sys
sys.path.insert(0,'.')
from atlas_lib import *
rng=np.random.default_rng(0)
fails=[]
def check(name, cond, info=''):
    print(('PASS' if cond else 'FAIL'), name, info); 
    if not cond: fails.append(name)

# L1 Ledoit-Wolf: shrinkage in [0,1]; identity data -> near mu*I; large n -> near sample cov
X=rng.normal(size=(5000,10)); C,s=ledoit_wolf(X-X.mean(0)); check('LW_large_n', np.abs(C-np.cov(X.T,bias=True)).max()<0.05 and 0<=s<=1, f's={s:.3f}')
X=rng.normal(size=(20,50)); C,s=ledoit_wolf(X-X.mean(0)); check('LW_p>n_wellcond', np.linalg.cond(C)<1e3 and s>0.3, f's={s:.3f}')

# L2 LDA equals brute-force Gaussian log-likelihood argmax
d,Cn=8,5; M=rng.normal(size=(Cn,d))*2; A=rng.normal(size=(d,d)); Sig=A@A.T+np.eye(d)
X=rng.normal(size=(300,d))
Si=np.linalg.inv(Sig); ll=np.stack([-0.5*np.einsum('nd,de,ne->n',X-M[c],Si,X-M[c]) for c in range(Cn)],1)
check('LDA_bruteforce', np.array_equal(lda_predict(X,M,Sig), ll.argmax(1)))

# L3 LDA accuracy on well separated Gaussians ~ Bayes
L=np.repeat(np.arange(Cn),400); Z=M[L]+rng.multivariate_normal(np.zeros(d),Sig,size=len(L))
acc=(lda_predict(Z,M,Sig)==L).mean(); check('LDA_acc_sane', acc>0.5, f'acc={acc:.3f}')

# L4 GREG exact under exact linear relation, no noise (truth recovered)
da,dy,Cn=6,4,10; W=rng.normal(size=(da,dy)); off=rng.normal(size=(Cn,dy))*0  # pure linear: y = a@W + b
b=rng.normal(size=dy)
mu_a={c:rng.normal(size=da)*3 for c in range(Cn)}
yb=np.repeat(np.arange(Cn),12); Ab=np.stack([mu_a[c] for c in yb])+rng.normal(size=(len(yb),da)); Yb=Ab@W+b
est,fb=greg_means(list(range(Cn)),mu_a,Ab,Yb,yb,rng)
err=max(np.abs(est[c]-(mu_a[c]@W+b)).max() for c in range(Cn)); check('GREG_exact_linear', err<1e-3, f'maxerr={err:.2e}')

# L5 GREG variance reduction ~ (1-R2) (Monte Carlo, scalar target, known population means)
res={}
for R2 in (0.0,0.8):
    errs_g=[];errs_b=[]
    for t in range(300):
        N=2000; a=rng.normal(size=(N,3)); e=rng.normal(size=N)
        w=np.array([1.,0.5,-0.5]); w=w/np.linalg.norm(w)*np.sqrt(R2)
        yv=(a@w+np.sqrt(1-R2)*e)[:,None]
        idx=rng.choice(N,30,replace=False)
        est,_=greg_means([0],{0:a.mean(0)},a[idx],yv[idx],np.zeros(30,int),rng)
        errs_g.append(est[0][0]-yv.mean()); errs_b.append(yv[idx].mean()-yv.mean())
    res[R2]=(np.var(errs_g)/np.var(errs_b))
check('GREG_varratio_R2=0', 0.8<res[0.0]<1.35, f'ratio={res[0.0]:.2f} (expect ~1)')
check('GREG_varratio_R2=0.8', res[0.8]<0.4, f'ratio={res[0.8]:.2f} (expect ~0.2 + slope noise)')

# L6 fallback path flagged for classes absent from bridge
est,fb=greg_means([0,1,99],{0:np.zeros(da),1:np.ones(da),99:np.ones(da)},Ab,Yb,yb,rng)
check('GREG_fallback_flag', fb=={99})

# L7 chained exact linear hops recover truth (two hops, invertible maps, exact)
mu0={c:rng.normal(size=5)*3 for c in range(6)}
W1=rng.normal(size=(5,5)); W2=rng.normal(size=(5,5))
yb=np.repeat(np.arange(6),10); F0=np.stack([mu0[c] for c in yb])+rng.normal(size=(60,5))
F1=F0@W1+1.0; F2=F1@W2-2.0
m1,_=greg_means(list(range(6)),mu0,F0,F1,yb,rng); m2,_=greg_means(list(range(6)),m1,F1,F2,yb,rng)
err=max(np.abs(m2[c]-((mu0[c]@W1+1)@W2-2)).max() for c in range(6)); check('chain_exact', err<1e-3, f'{err:.1e}')
g=global_affine_means(list(range(6)),mu0,F0,F1,yb,rng); err=max(np.abs(g[c]-(mu0[c]@W1+1)).max() for c in range(6)); check('global_affine_exact', err<1e-3, f'{err:.1e}')

# L8 within-class R2: ~1 for exact linear, ~0 for independent
A=rng.normal(size=(400,6)); yy=np.repeat(np.arange(20),20)
check('R2_linear', within_class_r2(A,A@rng.normal(size=(6,3)),yy,rng)>0.99)
r0=within_class_r2(A,rng.normal(size=(400,3)),yy,rng); check('R2_indep', abs(r0)<0.1, f'{r0:.3f}')

# L9 paired rule
check('paired_sig', paired([2,2.1,1.9,2.2,2.0],[0,0,0,0,0])['sig'] and not paired([1,-1,1,1,1],[0]*5)['sig'])
# L10 ridge equals closed form both regimes
for n,p in ((200,30),(30,200)):
    X=rng.normal(size=(n,p)); Y=rng.normal(size=(n,3))
    from atlas_lib import _ridge_factory
    lam=3.7; B=_ridge_factory(X,Y)(lam); Bc=np.linalg.solve(X.T@X+lam*np.eye(p),X.T@Y)
    check(f'ridge_closed_form_{n}x{p}', np.abs(B-Bc).max()<1e-8)
# L11 block ridge: kappa=0 path equals plain ridge on the second block; useless wide block is dropped or shrunk
Xa=rng.normal(size=(300,8)); Yb=Xa@rng.normal(size=(8,4))+0.1*rng.normal(size=(300,4))
Xn=rng.normal(size=(300,200))
Bb,lamb,kap=ridge_cv(np.hstack([Xn,Xa]),Yb,np.random.default_rng(1),split=200)
r2_block=1-((np.hstack([Xn,Xa])@Bb-Yb)**2).sum()/(Yb**2).sum()
check('block_ridge_drops_noise', kap<=0.1 and r2_block>0.95, f'kappa={kap} R2={r2_block:.3f}')
Bz=np.zeros((208,4)); 
B0,lam0,_=ridge_cv(Xa,Yb,np.random.default_rng(1))
check('block_ridge_shape', Bb.shape==(208,4))
print('FAILS:',fails)

import sys as _s
if fails: _s.exit(1)
