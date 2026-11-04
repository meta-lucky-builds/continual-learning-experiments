import numpy as np, sys; sys.path.insert(0,'.')
from atlas_analysis import run_analysis
fails=[]
def check(n,c,i=''):
    print('PASS' if c else 'FAIL', n, i)
    if not c: fails.append(n)
def synth(noise, seed=0, T=5, per=60, nres=10, M=200, dz=12, d=12, dr=40):
    rng=np.random.default_rng(seed); C=T*3
    order=rng.permutation(C); tasks=list(order.reshape(T,3))
    mz=rng.normal(size=(C,dz))*1.2
    y=np.repeat(np.arange(C),per+nres); Z=mz[y]+rng.normal(size=(len(y),dz))
    yte=np.repeat(np.arange(C),30); Zte=mz[yte]+rng.normal(size=(len(yte),dz))
    W=rng.normal(size=(dz,d)); F=[];
    Ws=[]
    for t in range(T):
        W=W+0.4*rng.normal(size=W.shape); Ws.append(W.copy())
        F.append((Z@W+0.3*t+noise*rng.normal(size=(len(y),d))).astype(np.float32))
    Wr=rng.normal(size=(dz,dr)); R=(np.tanh(Z@Wr*0.5)+0.05*rng.normal(size=(len(y),dr))).astype(np.float32)
    Rte=np.tanh(Zte@Wr*0.5); Fte=Zte@Ws[-1]+0.3*(T-1)+noise*rng.normal(size=(len(yte),d))
    br=[];ctl=[];st=[]
    for c in range(C):
        ids=np.where(y==c)[0]; ids=rng.permutation(ids)
        br+=list(ids[:nres//2]); ctl+=list(ids[nres//2:nres]); st+=list(ids[nres:])
    st=np.array(st); bufs=[]; buf=np.array([],int)
    for t in range(T):
        seen=st[np.isin(y[st],np.concatenate(tasks[:t+1]))]
        buf=rng.choice(seen,min(M,len(seen)),replace=False); bufs.append(buf)
    return dict(tasks=tasks,y_train=y,stream_ids=st,bridge_ids=np.array(br),control_ids=np.array(ctl),F=F,R=R,
                buffers=bufs,Ftest=Fte,Rtest=Rte,y_test=yte,head_acc=0.0)
if __name__=="__main__":
    # exact linear features, enough bridge items: every transport arm must recover the oracle
    o=run_analysis(synth(0.0, nres=40),0)
    for k in ('CH_cv_buf','CH_cv_res','GR_prev_buf','GR_prev_res','GR_multi_buf','GR_multi_res'):
        check('exact_'+k, o['err_all'][k]<1e-3, f"err={o['err_all'][k]:.2e} acc={o['acc'][k]:.1f} oracle={o['acc']['O_full']:.1f}")
    check('exact_hopR2', o['r2']['hop_min']>0.999, o['r2']['hop_min'])
    check('exact_fallbacks_zero', sum(o['fallbacks'].values())==0)
    o=run_analysis(synth(1.0),0)
    for k,v in sorted(o['err_all'].items(), key=lambda kv: kv[1]): print(f'   {k:14s} err={v:.4f} acc={o["acc"][k]:.1f}')
    check('noisy_oracle_err0', o['err_all']['O_full']==0)
    check('noisy_greg_beats_bonly', o['err_all']['GR_prev_buf']<o['err_all']['B_only_buf'])
    check('noisy_greg_res_beats_bonly_res', o['err_all']['GR_multi_res']<o['err_all']['B_only_res'])
    check('block_R2_not_worse', o['r2']['fT|[r,f(T-1)]'] >= o['r2']['fT|f(T-1)'] - 0.02, (o['r2']['fT|[r,f(T-1)]'], o['r2']['fT|f(T-1)']))
    print('FAILS',fails)
    if fails: sys.exit(1)
