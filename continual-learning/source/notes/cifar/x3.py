"""X3 (x3_plan.md): dormant class statistics reactivated through a small key set. Offline replay of the T-S5 erp runs.
Run: CL_K=100 CL_NTASK=20 CIFAR_DIR=data100 RESV=20 LR=0.03 MEM=2000 python3 x3.py selftest | <seed>   -> results/x3/x3_<seed>.json"""
import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C
from x2 import lda_scores, class_means, within_scatter, ridge, transport, mknet

def fit_back(Xn, Yo):
    """Affine ridge map new->old: returns f(X) = [X,1] Aa^T."""
    Xa = torch.cat([Xn, torch.ones(len(Xn), 1, dtype=torch.float64)], 1); Aa = ridge(Xa, Yo)
    return lambda X: torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64)], 1) @ Aa.T

def acc_within(S, y, cls):
    """Accuracy on items with label in cls, argmax restricted to cls; S has one column per entry of cls."""
    cls = torch.as_tensor(cls); return float((cls[S.argmax(1)] == y).double().mean())

def keys(lab, classes, k):
    """Positions in the buffer of the first k items (by buffer position) of each class in `classes`."""
    out = []
    for c in classes: p = torch.where(lab == c)[0]; out += p[:k].tolist() if k else p.tolist()
    return torch.tensor(out)

def selftest():
    ok = True
    def chk(c, name, *info):
        nonlocal ok; print('PASS' if c else 'FAIL', name, *info, flush=True); ok &= bool(c)
    g = torch.Generator().manual_seed(3); d, K5 = 16, 5
    mu_new = torch.randn(K5, d, generator=g, dtype=torch.float64) * 0.5  # overlapping classes: accuracy well below 1
    yt = torch.arange(K5).repeat_interleave(200); Fn = mu_new[yt] + torch.randn(len(yt), d, generator=g, dtype=torch.float64)
    yk = torch.arange(K5).repeat_interleave(40); Fk = mu_new[yk] + torch.randn(len(yk), d, generator=g, dtype=torch.float64)
    for name, Q, noise in [('S1 rotation+noise, many keys', torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0], 0.1), ('S2 identity, no noise', torch.eye(d, dtype=torch.float64), 0.0)]:
        old = lambda F: F @ Q.T + noise * torch.randn(len(F), d, generator=g, dtype=torch.float64)
        Fo_t = old(Fn); mu_o = class_means(Fo_t, yt, K5); cov_o = within_scatter(Fo_t, yt) / (len(yt) - K5)
        a_old = acc_within(lda_scores(Fo_t, mu_o, cov_o), yt, list(range(K5)))
        mp = fit_back(Fk, old(Fk)) if noise else fit_back(Fk, Fk @ Q.T)
        a_re = acc_within(lda_scores(mp(Fn), mu_o, cov_o), yt, list(range(K5)))
        chk(a_old < 0.95 and abs(a_old - a_re) <= (0.02 if noise else 1e-9), name, a_old, a_re)
    S = torch.randn(30, 3, generator=g, dtype=torch.float64); cls = [7, 2, 9]; y = torch.tensor(cls * 10)
    full = torch.full((30, C.K), -1e9, dtype=torch.float64); full[:, cls] = S
    chk(abs(acc_within(S, y, cls) - C.acc_seen(full, y, cls)) < 1e-6, 'S3 within-task accuracy equals C.acc_seen')
    print('SELFTEST', 'PASS' if ok else 'FAIL'); return ok

def main(s):
    torch.set_num_threads(4); K, T = C.K, C.NTASK - 1; d = 16 * C.W
    Xtr, ytr, Xte, yte = C.load(); tasks, order, tid, resv = C.make_stream(ytr, s); bl = C.batches(order, tid, resv)
    E = torch.load(f'results/ts5_probe/enc_{s}.pt', weights_only=False); assert E['seed'] == s
    nets = [mknet(v['state']) for v in E['versions']] + [mknet(E['final'])]
    buf = C.Reservoir(C.MEM, np.random.default_rng(s * 1000 + 3)); Bt = []
    for i, (t, bi) in enumerate(bl):
        if buf.idx: buf.sample(C.RB)
        buf.add(torch.as_tensor(bi), ytr[bi])
        if i + 1 == len(bl) or bl[i + 1][0] != t: Bt.append((np.array(buf.idx), torch.tensor(buf.lab)))
    feats = lambda net, idx: C.chunked(net.feats, Xtr[idx]).double()
    for t, v in enumerate(E['versions']):  # gR as in x2
        idx, lab = Bt[t]; Fb = feats(nets[t], idx); old = sorted(c for tt in tasks[:t + 1] for c in tt)
        S1 = torch.zeros(K, d, dtype=torch.float64).index_add_(0, lab, Fb)
        assert torch.equal(v['n'][old], torch.bincount(lab, minlength=K).double()[old]) and torch.allclose(v['S1'][old], S1[old], atol=1e-6, rtol=0), 'gR'
    print('gR PASS', flush=True)
    src_mu, src_cov = {}, {}
    idx0, lab0 = Bt[0]; F0 = feats(nets[0], idx0); src_mu[0] = class_means(F0, lab0, K)
    for u in range(1, C.NTASK):
        bis = [bi for t, bi in bl if t == u]
        with torch.no_grad(): Fu = torch.cat([nets[u - 1].feats(Xtr[bi]) for bi in bis]).double()
        yu = torch.cat([ytr[bi] for bi in bis]); src_mu[u] = class_means(Fu, yu, K); src_cov[u] = within_scatter(Fu, yu) / (len(yu) - len(tasks[u]))
        v = E['versions'][u - 1]; cs = tasks[u]
        assert torch.equal(v['n'][cs], torch.bincount(yu, minlength=K).double()[cs]) and torch.allclose(v['S1'][cs], torch.zeros(K, d, dtype=torch.float64).index_add_(0, yu, Fu)[cs], rtol=1e-4, atol=1e-4), 'gS'
    print('gS PASS', flush=True)
    fin = nets[T]; Ft = C.chunked(fin.feats, Xte).double(); idxF, labF = Bt[T]; FbF = feats(fin, idxF)
    _, mu_buf_all, cov_buf = C.lda_from(FbF.float(), labF).stats()
    fit = order[~resv]; _, muF, covF = C.lda_from(C.chunked(fin.feats, Xtr[fit]), ytr[fit]).stats()
    aO = float((lda_scores(Ft, muF, covF).argmax(1) == yte).double().mean())
    pj = [e for e in json.load(open(f'results/ts5_probe/probe_{s}.json'))['enc'] if e['name'] == 'final'][0]['PF']
    assert abs(aO - pj) < 1e-6, ('gO', aO, pj); print('gO PASS', flush=True)
    # X2's online chained cv means (for TRANS)
    mu = torch.full((K, d), float('nan'), dtype=torch.float64); mu[tasks[0]] = src_mu[0][tasks[0]]
    for t in range(T):
        mu[tasks[t + 1]] = src_mu[t + 1][tasks[t + 1]]; idx, lab = Bt[t + 1]
        mu, _, _ = transport(mu, feats(nets[t], idx), feats(nets[t + 1], idx), lab, 'cv', K)
    mu_tr = mu
    rows = []
    for u in range(1, C.NTASK):
        cs = tasks[u]; mt = torch.isin(yte, torch.tensor(cs)); yt = yte[mt]; Fte_f = Ft[mt]
        old = nets[u - 1]; Fte_o = C.chunked(old.feats, Xte[mt]).double()
        r = {'u': u, 'age': T - u, 'OLD': acc_within(lda_scores(Fte_o, src_mu[u][cs], src_cov[u]), yt, cs),
             'TRANS': acc_within(lda_scores(Fte_f, mu_tr[cs], cov_buf), yt, cs)}
        prev = sorted(c for tt in tasks[:u + 1] for c in tt); Fb_old = feats(old, idxF)
        for k in [1, 5, 0]:
            kk = 'all' if k == 0 else str(k)
            kp = keys(labF, prev, k); mp = fit_back(FbF[kp], Fb_old[kp])
            r[f'REACT_{kk}'] = acc_within(lda_scores(mp(Fte_f), src_mu[u][cs], src_cov[u]), yt, cs)
            ku = keys(labF, cs, k); mk = class_means(FbF[ku], labF[ku], K)[cs]
            r[f'BUF_{kk}'] = acc_within(lda_scores(Fte_f, mk, cov_buf), yt, cs)
        rows.append(r); print(json.dumps(r), flush=True)
    os.makedirs('results/x3', exist_ok=True); json.dump({'seed': s, 'rows': rows}, open(f'results/x3/x3_{s}.json', 'w'), indent=1)

if __name__ == '__main__':
    if sys.argv[1] == 'selftest': sys.exit(0 if selftest() else 1)
    main(int(sys.argv[1]))
