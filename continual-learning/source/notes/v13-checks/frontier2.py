import numpy as np, math, json
exec(open('frontier.py').read().split("N,n,K=200,20,10")[0])
rng=np.random.default_rng(11)
N,n,K=200,20,10; T=40
out={'gem_task':[], 'gem_item':[], 'margin_k1':[], 'margin_k3':[], 'margin_all':[]}
for trial in range(T):
    J=rng.normal(size=(n,K,N))/math.sqrt(N); L=rng.normal(size=(n,K))*2; y=L.argmax(1)
    P=np.exp(L-L.max(1,keepdims=True)); P/=P.sum(1,keepdims=True)
    riv=[[j for j in np.argsort(-L[i]) if j!=y[i]] for i in range(n)]
    Mg=np.stack([J[i][y[i]]-J[i][j] for i in range(n) for j in riv[i]],1)
    a=rng.normal(size=N); d0=-a
    gce=np.stack([-(J[i].T@(P[i]-np.eye(K)[y[i]])) for i in range(n)],1)   # -grad CE = direction that raises log p_y
    sets={'gem_task':gce.mean(1,keepdims=True),'gem_item':gce,
          'margin_k1':np.stack([J[i][y[i]]-J[i][riv[i][0]] for i in range(n)],1),
          'margin_k3':np.stack([J[i][y[i]]-J[i][j] for i in range(n) for j in riv[i][:3]],1),'margin_all':Mg}
    for k,C in sets.items():
        d=qp(C,d0,np.zeros(C.shape[1])); out[k].append((np.sum(np.maximum(0,-(Mg.T@d))),-(a@d)))
print(json.dumps({k:[round(float(np.mean([p[0] for p in v])),2),round(float(np.mean([p[1] for p in v])),1)] for k,v in out.items()},indent=0))
