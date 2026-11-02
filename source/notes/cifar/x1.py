"""X1 (x1_plan.md): how much of the readout bias of the final ER-trained net is weight scale. Readout-side only.
Run: CL_K=100 CL_NTASK=20 CIFAR_DIR=data100 RESV=20 LR=0.03 MEM=2000 python3 x1.py selftest | <seed>   -> results/x1/x1_<seed>.json"""
import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C
from ts5_probe import probe

def acc_all(S, y):
    """Accuracy, argmax over all classes present in y (all 100 at the final task end)."""
    return float((S.argmax(1) == y).double().mean())

def bc(S_test, S_res, yr):
    """Subtract the balanced mean score vector over the reserve (same estimator as bc_resv)."""
    return S_test - C.balanced_mean_logit(S_res, yr)

def arms(Ft, Fr, yr, W, b):
    """Score matrices (test, reserve) for every arm. Ft/Fr float64 features, W (K,d), b (K)."""
    nW = W.norm(dim=1, keepdim=True)
    out = {'A': (Ft @ W.T + b, Fr @ W.T + b), 'B': (Ft @ W.T, Fr @ W.T),
           'C05': (Ft @ (W / nW ** 0.5).T, Fr @ (W / nW ** 0.5).T), 'C1': (Ft @ (W / nW).T, Fr @ (W / nW).T)}
    nz = lambda F: F / F.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cls = torch.unique(yr); assert len(cls) == W.shape[0]
    P = torch.stack([nz(Fr)[yr == c].mean(0) for c in cls]); P = nz(P)
    out['D'] = (nz(Ft) @ P.T, nz(Fr) @ P.T)
    M = torch.stack([Fr[yr == c].mean(0) for c in cls])
    out['E'] = (-torch.cdist(Ft, M) ** 2, -torch.cdist(Fr, M) ** 2)
    return out

def evaluate(Ft, yt, Fr, yr, W, b):
    res = {}
    for k, (St, Sr) in arms(Ft, Fr, yr, W, b).items():
        res[k] = acc_all(St, yt)
        if k != 'E': res[k + '+bc'] = acc_all(bc(St, Sr, yr), yt)
    return res

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])

def selftest():
    g = torch.Generator().manual_seed(0); K, d, n = 20, 64, 200
    mu = torch.randn(K, d, generator=g, dtype=torch.float64); mu = 3 * mu / mu.norm(dim=1, keepdim=True)
    y = torch.arange(K).repeat_interleave(n); X = mu[y] + torch.randn(len(y), d, generator=g, dtype=torch.float64)
    yr = torch.arange(K).repeat_interleave(20); Xr = mu[yr] + torch.randn(len(yr), d, generator=g, dtype=torch.float64)
    ok = True
    # P1: C1 equals argmax cosine to mu directions (any positive gains)
    gain = torch.linspace(1, 4, K, dtype=torch.float64); W = mu * gain[:, None]; b = torch.zeros(K, dtype=torch.float64)
    A = arms(X, Xr, yr, W, b); c1 = (X @ (mu / mu.norm(dim=1, keepdim=True)).T).argmax(1)
    p1 = bool(torch.equal(A['C1'][0].argmax(1), c1)); ok &= p1; print('P1', p1)
    # P2: C1 >= A on gain-biased head; A falls as gain spread grows
    accs = []
    for top in [1.0, 2.0, 4.0]:
        Wg = mu * torch.linspace(1, top, K, dtype=torch.float64)[:, None]; r = evaluate(X, y, Xr, yr, Wg, b); accs.append((r['A'], r['C1']))
    p2 = accs[2][1] >= accs[2][0] and accs[0][0] > accs[1][0] > accs[2][0]; ok &= p2; print('P2', p2, accs)
    # P3: bc equals C.balanced_mean_logit on an unbalanced reserve
    yu = torch.cat([yr, yr[:7]]); Su = torch.randn(len(yu), K, generator=g, dtype=torch.float64)
    ref = torch.stack([Su[yu == c].mean(0) for c in range(K)]).mean(0)
    p3 = float((C.balanced_mean_logit(Su, yu) - ref).abs().max()) < 1e-12 and torch.allclose(bc(Su[:5], Su, yu), Su[:5] - ref, atol=1e-12, rtol=0)
    ok &= p3; print('P3', p3)
    # P4: acc_all equals C.acc_seen with all classes seen
    S = torch.randn(len(y), K, generator=g, dtype=torch.float64); p4 = abs(acc_all(S, y) - C.acc_seen(S, y, list(range(K)))) < 1e-6; ok &= p4; print('P4', p4)  # C.acc_seen averages in float32
    # P5: with equal gains and equal-norm means, A = B = C1 exactly in argmax
    r5 = arms(X, Xr, yr, mu, b); p5 = bool(torch.equal(r5['A'][0].argmax(1), r5['B'][0].argmax(1)) and torch.equal(r5['A'][0].argmax(1), r5['C1'][0].argmax(1)))
    ok &= p5; print('P5', p5)
    print('SELFTEST', 'PASS' if ok else 'FAIL'); return ok

def main(s):
    torch.set_num_threads(4)
    Xtr, ytr, Xte, yte = C.load(); tasks, order, tid, resv = C.make_stream(ytr, s)
    E = torch.load(f'results/ts5_probe/enc_{s}.pt', weights_only=False); assert E['seed'] == s
    net = C.make_net(0); net.load_state_dict(E['final']); net.eval()
    fit = order[~resv]; yf = ytr[fit]; ridx = order[resv]; yr = ytr[ridx]
    assert (np.bincount(yr.numpy(), minlength=C.K) == C.RESV).all() and not np.isin(fit, ridx).any()
    Ft = C.chunked(net.feats, Xte).double(); Fr = C.chunked(net.feats, Xtr[ridx]).double()
    # g1: the loaded net reproduces its saved matched probe PF('final')
    Ff = C.chunked(net.feats, Xtr[fit]); pf = probe(Ff, Ft.float(), yf, yte, np.arange(len(fit)), list(range(C.K)))
    pj = [e for e in json.load(open(f'results/ts5_probe/probe_{s}.json'))['enc'] if e['name'] == 'final'][0]['PF']
    assert abs(pf - pj) < 1e-6, ('g1 failed', pf, pj); print('g1 PASS', pf, pj, flush=True)
    W = net.head.weight.detach().double(); b = net.head.bias.detach().double()
    r = evaluate(Ft, yte, Fr, yr, W, b)
    lda = C.StreamLDA(Fr.shape[1]); lda.update(Fr.float(), yr); r['R'] = acc_all(lda.scores(Ft.float())[1], yte)
    task_of = np.zeros(C.K, int)
    for t, cs in enumerate(tasks): task_of[cs] = t
    nW = W.norm(dim=1).numpy(); cosW = (W / W.norm(dim=1, keepdim=True)) @ (W / W.norm(dim=1, keepdim=True)).T
    off = cosW[~torch.eye(C.K, dtype=torch.bool)]
    diag = {'norm_ratio': float(nW.max() / nW.min()), 'rho_norm_task': spearman(nW, task_of), 'rho_bias_task': spearman(b.numpy(), task_of),
            'cos_min': float(off.min()), 'cos_max': float(off.max()), 'cos_sd': float(off.std()),
            'norm_by_task': [float(nW[task_of == t].mean()) for t in range(C.NTASK)], 'bias_by_task': [float(b.numpy()[task_of == t].mean()) for t in range(C.NTASK)]}
    out = {'seed': s, 'acc': r, 'diag': diag, 'g1_PF': pf}
    os.makedirs('results/x1', exist_ok=True); json.dump(out, open(f'results/x1/x1_{s}.json', 'w'), indent=1); print(json.dumps(out['acc']), json.dumps({k: v for k, v in diag.items() if 'by_task' not in k}), flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'selftest': sys.exit(0 if selftest() else 1)
    main(int(sys.argv[1]))
