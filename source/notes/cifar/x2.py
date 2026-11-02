"""X2 (x2_plan.md): transporting stranded class statistics into new encoders. Offline replay of the saved T-S5 erp runs.
Run: CL_K=100 CL_NTASK=20 CIFAR_DIR=data100 RESV=20 LR=0.03 MEM=2000 python3 x2.py selftest | <seed>   -> results/x2/x2_<seed>.json"""
import sys, os, json, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C

# ---------------------------------------------------------------- moment-based LDA (same estimator as C.StreamLDA)
def lda_scores(F, mu, cov):
    d = cov.shape[0]; S = (1 - C.SHRINK) * cov + (C.SHRINK * torch.trace(cov) / d + C.FLOOR) * torch.eye(d, dtype=torch.float64)
    L = torch.linalg.cholesky(S); PM = torch.cholesky_solve(mu.T.contiguous(), L)
    return F.double() @ PM - 0.5 * (mu * PM.T).sum(1)

def class_means(F, y, K):
    F = F.double(); return torch.stack([F[y == c].mean(0) if (y == c).any() else torch.full((F.shape[1],), float('nan'), dtype=torch.float64) for c in range(K)])

def within_scatter(F, y):
    F = F.double(); W = torch.zeros(F.shape[1], F.shape[1], dtype=torch.float64)
    for c in torch.unique(y): X = F[y == c] - F[y == c].mean(0); W += X.T @ X
    return W

def ridge(X, Y, rel=1e-3):
    """A with Y ~ X A^T, ridge lambda = rel * tr(X^T X) / d."""
    G = X.T @ X; lam = rel * torch.trace(G) / G.shape[0]
    return torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=torch.float64), X.T @ Y).T

# ---------------------------------------------------------------- transport maps
def transport(mu_src, Ps, Pd, yb, method, K):
    """mu_src (K,d) with NaN rows for classes absent; Ps/Pd buffer features in src/dst coords (float64); returns mu_dst, A, resid_cov."""
    ms, md = class_means(Ps, yb, K), class_means(Pd, yb, K); have = ~torch.isnan(ms[:, 0])
    out = torch.full_like(mu_src, float('nan')); valid = ~torch.isnan(mu_src[:, 0])
    Xc = Ps - ms[yb]; Yc = Pd - md[yb]; A = ridge(Xc, Yc); resid = Yc - Xc @ A.T
    rcov = resid.T @ resid / max(len(yb) - int(have.sum()), 1)
    if method == 'shift':
        gshift = (Pd - Ps).mean(0)
        for c in torch.where(valid)[0]: out[c] = mu_src[c] + ((md[c] - ms[c]) if have[c] else gshift)
    elif method == 'cv':
        Ag = A
        for c in torch.where(valid)[0]:
            out[c] = (md[c] + Ag @ (mu_src[c] - ms[c])) if have[c] else (Ag @ (mu_src[c] - Ps.mean(0)) + Pd.mean(0))
    elif method == 'aff':
        Xa = torch.cat([Ps, torch.ones(len(Ps), 1, dtype=torch.float64)], 1); Aa = ridge(Xa, Pd)
        out[valid] = torch.cat([mu_src[valid], torch.ones(int(valid.sum()), 1, dtype=torch.float64)], 1) @ Aa.T
    else: raise ValueError(method)
    return out, A, rcov

# ---------------------------------------------------------------- self-test
def selftest():
    ok = True
    def chk(c, name, *info):
        nonlocal ok; print('PASS' if c else 'FAIL', name, *info, flush=True); ok &= bool(c)
    g = torch.Generator().manual_seed(1); K, d = 10, 16
    musrc = torch.randn(K, d, generator=g, dtype=torch.float64) * 3; yb = torch.arange(K).repeat_interleave(30)
    Ps = musrc[yb] + torch.randn(len(yb), d, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    # T1: rotation only, no noise -> cv and aff exact, shift not
    Pd = Ps @ Q.T; true = musrc @ Q.T
    e = {m: float((transport(musrc, Ps, Pd, yb, m, K)[0] - true).abs().max()) for m in ['shift', 'cv', 'aff']}
    sc = float(true.abs().max()); chk(e['cv'] < 1e-2 * sc and e['aff'] < 1e-2 * sc and e['shift'] > 10 * max(e['cv'], e['aff']), 'T1 rotation: cv, aff recover (ridge bias < 1% of scale); shift error >10x larger', e, sc)
    # T2: identity plus class offsets, no noise -> shift and cv exact, aff worse than shift
    delta = torch.randn(K, d, generator=g, dtype=torch.float64) * 2; Pd = Ps + delta[yb]; true = musrc + delta
    e = {m: float((transport(musrc, Ps, Pd, yb, m, K)[0] - true).abs().max()) for m in ['shift', 'cv', 'aff']}
    chk(e['shift'] < 1e-9 and e['cv'] < 1e-2 and e['aff'] > 10 * e['shift'] + 0.1, 'T2 class offsets: shift, cv recover; aff does not', e)
    # T3: moment LDA equals C.StreamLDA on the same data (full 100-class K to match C.K)
    K2 = C.K; y3 = torch.arange(K2).repeat_interleave(8); F3 = torch.randn(len(y3), d, generator=g) + torch.randn(K2, d, generator=g)[y3]
    ref = C.StreamLDA(d); ref.update(F3, y3); sn, mu, cov = ref.stats(); Ft = torch.randn(50, d, generator=g)
    chk(float((lda_scores(Ft, mu, cov) - ref.scores(Ft)[1]).abs().max()) < 1e-9, 'T3 moment LDA equals StreamLDA scores')
    # T4: rotation + offsets + noise, few buffer items, accurate source means -> cv beats buffer-only on means
    errs = []
    for rep in range(20):
        gg = torch.Generator().manual_seed(100 + rep); Qr, _ = torch.linalg.qr(torch.randn(d, d, generator=gg, dtype=torch.float64))
        yb4 = torch.arange(K).repeat_interleave(5); Ps4 = musrc[yb4] + torch.randn(len(yb4), d, generator=gg, dtype=torch.float64) * 2
        Pd4 = Ps4 @ Qr.T + 0.3 * torch.randn(len(yb4), d, generator=gg, dtype=torch.float64); true4 = musrc @ Qr.T
        cvm = transport(musrc, Ps4, Pd4, yb4, 'cv', K)[0]; bo = class_means(Pd4, yb4, K)
        errs.append((float((cvm - true4).norm(dim=1).mean()), float((bo - true4).norm(dim=1).mean())))
    e4 = np.array(errs).mean(0); chk(e4[0] < e4[1], 'T4 cv beats buffer-only means under noise', e4)
    print('SELFTEST', 'PASS' if ok else 'FAIL'); return ok

# ---------------------------------------------------------------- real run
def mknet(state): n = C.make_net(0); n.load_state_dict(state); n.eval(); return n
def softmax(L): L = L - L.max(1, keepdims=True).values; P = L.exp(); return P / P.sum(1, keepdims=True)
def acc(S, y): return float((S.argmax(1) == y).double().mean())

def main(s):
    torch.set_num_threads(4); K, T = C.K, C.NTASK - 1
    Xtr, ytr, Xte, yte = C.load(); tasks, order, tid, resv = C.make_stream(ytr, s); bl = C.batches(order, tid, resv)
    E = torch.load(f'results/ts5_probe/enc_{s}.pt', weights_only=False); assert E['seed'] == s and len(E['versions']) == T
    nets = [mknet(v['state']) for v in E['versions']] + [mknet(E['final'])]   # nets[t] = encoder at end of task t
    assert all(v['t'] == t for t, v in enumerate(E['versions']))
    # reservoir reconstruction (cifar2.run order: sample before add, every step; epochs=1, no other RNG use)
    buf = C.Reservoir(C.MEM, np.random.default_rng(s * 1000 + 3)); Bt = []
    for i, (t, bi) in enumerate(bl):
        if buf.idx: buf.sample(C.RB)
        buf.add(torch.as_tensor(bi), ytr[bi])
        if i + 1 == len(bl) or bl[i + 1][0] != t: Bt.append((np.array(buf.idx), torch.tensor(buf.lab)))
    assert len(Bt) == C.NTASK
    feats = lambda net, idx: C.chunked(net.feats, Xtr[idx]).double()
    # gR: version v_t's statistics on classes of tasks <= t equal the reconstructed buffer at promotion t
    okR = True
    for t, v in enumerate(E['versions']):
        idx, lab = Bt[t]; Fb = feats(nets[t], idx); old = sorted(c for tt in tasks[:t + 1] for c in tt)
        cnt = torch.bincount(lab, minlength=K).double(); S1 = torch.zeros(K, Fb.shape[1], dtype=torch.float64).index_add_(0, lab, Fb)
        okR &= bool(torch.equal(v['n'][old], cnt[old])) and bool(torch.allclose(v['S1'][old], S1[old], atol=1e-6, rtol=0))
    assert okR, 'gR failed'; print('gR PASS: reconstructed buffers reproduce every version statistic on its old classes', flush=True)
    # source statistics: task-0 classes from buffer at end of task 0 in v0; task-u classes from their stream in v_{u-1}
    d = 16 * C.W; mu_src_by_t = {}; scat_by_t = {}; n_by_t = {}; okS = True
    idx0, lab0 = Bt[0]; F0 = feats(nets[0], idx0); mu_src_by_t[0] = class_means(F0, lab0, K); scat_by_t[0] = within_scatter(F0, lab0); n_by_t[0] = len(lab0)
    for u in range(1, C.NTASK):
        bis = [bi for t, bi in bl if t == u]
        with torch.no_grad(): Fu = torch.cat([nets[u - 1].feats(Xtr[bi]) for bi in bis]).double()
        yu = torch.cat([ytr[bi] for bi in bis]); mu_src_by_t[u] = class_means(Fu, yu, K); scat_by_t[u] = within_scatter(Fu, yu); n_by_t[u] = len(yu)
        v = E['versions'][u - 1]; cs = tasks[u]; S1 = torch.zeros(K, d, dtype=torch.float64).index_add_(0, yu, Fu)
        okS &= bool(torch.equal(v['n'][cs], torch.bincount(yu, minlength=K).double()[cs])) and bool(torch.allclose(v['S1'][cs], S1[cs], rtol=1e-4, atol=1e-4))
    assert okS, 'gS failed'; print('gS PASS: recomputed stream statistics match the saved version statistics', flush=True)
    def src_means(u):
        m = torch.full((K, d), float('nan'), dtype=torch.float64); m[tasks[u]] = mu_src_by_t[u][tasks[u]]; return m
    # final-coordinate references
    fin = nets[T]; Ft = C.chunked(fin.feats, Xte).double(); idxF, labF = Bt[T]; FbF = feats(fin, idxF)
    fit = order[~resv]; Ffit = C.chunked(fin.feats, Xtr[fit]); yfit = ytr[fit]
    full = C.lda_from(Ffit, yfit); _, mu_full, cov_full = full.stats()
    bl_ = C.lda_from(FbF.float(), labF); _, mu_buf, cov_buf = bl_.stats()
    R = {'O_full': acc(lda_scores(Ft, mu_full, cov_full), yte), 'O_mean': acc(lda_scores(Ft, mu_full, cov_buf), yte), 'B_only': acc(lda_scores(Ft, mu_buf, cov_buf), yte)}
    pj = [e for e in json.load(open(f'results/ts5_probe/probe_{s}.json'))['enc'] if e['name'] == 'final'][0]['PF']
    assert abs(R['O_full'] - pj) < 1e-6, ('gO failed', R['O_full'], pj); print('gO PASS', R['O_full'], pj, flush=True)
    # chained (online) transport
    for m in ['shift', 'cv', 'aff']:
        mu = src_means(0); Wsc = scat_by_t[0].clone(); N = n_by_t[0]
        for t in range(T):  # at end of task t+1: add task t+1 stream (in v_t coords), then move everything into v_{t+1}
            new = src_means(t + 1); mu[tasks[t + 1]] = new[tasks[t + 1]]; Wsc = Wsc + scat_by_t[t + 1]; N += n_by_t[t + 1]
            idx, lab = Bt[t + 1]; Ps, Pd = feats(nets[t], idx), feats(nets[t + 1], idx)
            mu, A, rcov = transport(mu, Ps, Pd, lab, m, K)
            ncls = len([c for tt in tasks[:t + 2] for c in tt]); Wsc = A @ Wsc @ A.T + rcov * (N - ncls)
        assert not torch.isnan(mu).any()
        R[f'CH_{m}'] = acc(lda_scores(Ft, mu, cov_buf), yte)
        if m == 'cv': R['CH_cv_tr'] = acc(lda_scores(Ft, mu, Wsc / (N - K)), yte); mu_ch_cv = mu
        if m == 'shift': mu_ch_shift = mu
    # direct (non-online) transport: each class from its source encoder straight to final, final buffer as probes
    for m in ['shift', 'cv', 'aff']:
        mu = torch.full((K, d), float('nan'), dtype=torch.float64)
        for u in range(C.NTASK):
            srcnet = nets[0] if u == 0 else nets[u - 1]; Ps = feats(srcnet, idxF)
            mu_u, _, _ = transport(src_means(u), Ps, FbF, labF, m, K); mu[tasks[u]] = mu_u[tasks[u]]
        R[f'DI_{m}'] = acc(lda_scores(Ft, mu, cov_buf), yte)
    # mixtures and the recomputed bank
    with torch.no_grad(): Lte = fin.head(Ft.float()).double(); Lb = fin.head(FbF.float()).double()
    PA = softmax(Lte - C.balanced_mean_logit(Lb, labF)); R['ER+bc_buf'] = acc(PA, yte)
    rc = np.load(f'results/ts5/rc_{s}.npz'); PB = torch.from_numpy(rc[f'e{T}_test_lda_lp'].astype(np.float64)).exp(); R['RC'] = acc(PB, yte)
    def vprob(vi):
        v = E['versions'][vi]; l = C.StreamLDA(d); l.S1, l.n, l.XX = v['S1'], v['n'], v['XX']; return l.logprobs(C.chunked(nets[vi].feats, Xte)).double().exp()
    V = {vi: vprob(vi) for vi in [3, 8, 13, 18]}
    R['BANK_4spread'] = acc((PA + PB + sum(V.values())) / 6, yte)
    P = lambda mu: softmax(lda_scores(Ft, mu, cov_buf))
    R['MIX3_B_only'] = acc((PA + PB + P(mu_buf)) / 3, yte); R['MIX3_CH_cv'] = acc((PA + PB + P(mu_ch_cv)) / 3, yte); R['MIX3_CH_shift'] = acc((PA + PB + P(mu_ch_shift)) / 3, yte)
    R['MIX2_CH_cv_RC'] = acc((PB + P(mu_ch_cv)) / 2, yte)
    os.makedirs('results/x2', exist_ok=True); json.dump({'seed': s, 'acc': R}, open(f'results/x2/x2_{s}.json', 'w'), indent=1); print(json.dumps(R), flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'selftest': sys.exit(0 if selftest() else 1)
    main(int(sys.argv[1]))
