"""Sharp-test marathon harness (plan: notes/marathon/plan.md).

One training loop, shared by every test, for two datasets:
  toy   : the v1.3 toy (functions imported from notes/v13-checks/toy_cl3.py), SEP/NMEM/REP from cfg
  mnist : Split-MNIST, 5 tasks of 2 digits, MLP 784-256-256-10
The loop is a byte-for-byte generalisation of toy_cl3's ER loop (manual Adam in float64 on the flat
parameter vector), so with every switch off it must reproduce toy_cl3's per-seed ER numbers (selftest S1).

Usage:  python mar.py selftest
        python mar.py run '<json cfg>' seeds   -> one JSON line per seed
"""
import sys, os, json, time, math, importlib.util, struct, copy
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
TC_PATH = os.path.join(HERE, '..', 'v13-checks', 'toy_cl3.py')
MNIST_DIR = os.environ.get('MNIST_DIR', '/tmp/claude-0/-home-user/da7a0050-f410-5db2-8360-29a4b9e95c98/scratchpad/data/mnist')
torch.set_num_threads(int(os.environ.get('NT', '1')))

def load_tc(sep, nmem, rep):
    os.environ['SEP'] = str(sep); os.environ['NMEM'] = str(nmem); os.environ['REP'] = str(rep)
    spec = importlib.util.spec_from_file_location('tc_%s' % sep, TC_PATH); tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
    torch.set_num_threads(int(os.environ.get('NT', '1')))   # toy_cl3 sets 4 threads on import; results do not depend on it
    return tc

# ---------------------------------------------------------------- data
_MN = {}
def _idx(path):
    with open(path, 'rb') as f:
        magic, = struct.unpack('>I', f.read(4)); nd = magic & 0xff
        dims = struct.unpack('>' + 'I' * nd, f.read(4 * nd))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)
def mnist():
    if not _MN:
        Xtr = _idx(os.path.join(MNIST_DIR, 'train-images-idx3-ubyte')).reshape(-1, 784).astype(np.float32) / 255.
        ytr = _idx(os.path.join(MNIST_DIR, 'train-labels-idx1-ubyte')).astype(np.int64)
        Xte = _idx(os.path.join(MNIST_DIR, 't10k-images-idx3-ubyte')).reshape(-1, 784).astype(np.float32) / 255.
        yte = _idx(os.path.join(MNIST_DIR, 't10k-labels-idx1-ubyte')).astype(np.int64)
        _MN.update(Xtr=torch.from_numpy(Xtr), ytr=torch.from_numpy(ytr), Xte=torch.from_numpy(Xte), yte=torch.from_numpy(yte))
    return _MN['Xtr'], _MN['ytr'], _MN['Xte'], _MN['yte']

def setup(cfg, seed):
    if cfg['data'] == 'toy':
        tc = load_tc(cfg.get('sep', 0.6), cfg.get('nmem', 5), cfg.get('rep', 4))
        Xtr, ytr, Xte, yte = tc.make_data(seed); m = tc.model(seed + 1000)
    else:
        Xtr, ytr, Xte, yte = mnist(); torch.manual_seed(seed + 1000)
        m = torch.nn.Sequential(torch.nn.Linear(784, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))
    return Xtr, ytr, Xte, yte, m

class Head(torch.nn.Module):
    """Follow-up F5 heads. 'center': logits = W (h - mu) + b, mu = class-balanced mean buffer feature (no grad),
    updated every step. 'cos': logits = s * cos(W, h) + b. 'cos_center': both. Initialised from the same nn.Linear
    draw as the default head, so streams and initial weights are identical across arms."""
    def __init__(self, lin, mode, scale=10.0):
        super().__init__(); self.weight = torch.nn.Parameter(lin.weight.detach().clone()); self.bias = torch.nn.Parameter(lin.bias.detach().clone())
        self.mode = mode; self.scale = scale; self.register_buffer('mu', torch.zeros(lin.weight.shape[1]))
    def forward(self, h):
        if 'center' in self.mode: h = h - self.mu
        if 'cos' in self.mode:
            return self.scale * torch.nn.functional.normalize(h, dim=-1) @ torch.nn.functional.normalize(self.weight, dim=-1).T + self.bias
        return h @ self.weight.T + self.bias

def flat(m): return torch.cat([p.reshape(-1) for p in m.parameters()])
def set_flat(m, v):
    i = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n
def feats(m, X):
    with torch.no_grad(): return torch.nn.Sequential(*list(m.children())[:-1])(X)
def class_acc(pred, y, classes):
    classes = list(classes)
    if not classes: return None
    return float(np.mean([(pred[y == c] == c).float().mean().item() for c in classes]))

# ---------------------------------------------------------------- readouts (T2, T3)
def refit(F, Y, Fte, K=10):
    torch.manual_seed(0); lin = torch.nn.Linear(F.shape[1], K)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=500, line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); l = torch.nn.functional.cross_entropy(lin(F), Y) + 1e-3 * (lin.weight ** 2).sum(); l.backward(); return l
    opt.step(cl)
    with torch.no_grad(): return lin(Fte).argmax(1)
def ncm_predict(Fb, yb, Fte, K=10):
    """Nearest class mean, cosine, means from buffer features."""
    Fb = torch.nn.functional.normalize(Fb, dim=1); Ft = torch.nn.functional.normalize(Fte, dim=1)
    M = torch.stack([Fb[yb == c].mean(0) for c in range(K)]); M = torch.nn.functional.normalize(M, dim=1)
    return (Ft @ M.T).argmax(1)
def wa_logits(m, X, new_cls, old_cls):
    """Weight aligning: rescale the newest classes' head rows to the old rows' mean norm (bias unchanged)."""
    W = m[-1].weight.detach().clone(); b = m[-1].bias.detach()
    gamma = W[old_cls].norm(dim=1).mean() / W[new_cls].norm(dim=1).mean(); W[new_cls] *= gamma
    return feats(m, X) @ W.T + b, float(gamma)
def lda_predict(means, cov, Fte, shrink=0.1):
    """Equal-prior LDA with a shared covariance shrunk toward (tr/d) I."""
    d = cov.shape[0]; S = (1 - shrink) * cov + shrink * torch.trace(cov) / d * torch.eye(d, dtype=cov.dtype)
    P = torch.linalg.inv(S); M = torch.stack([means[k] for k in sorted(means)]).to(cov.dtype); Ft = Fte.to(cov.dtype)
    sc = Ft @ P @ M.T - 0.5 * ((M @ P) * M).sum(1)
    keys = torch.tensor(sorted(means)); return keys[sc.argmax(1)]
def pooled_cov(F, y):
    F = F.double(); R = torch.cat([F[y == k] - F[y == k].mean(0) for k in torch.unique(y)]); return R.T @ R / max(1, len(R) - 1)

def biascorr_predict(m, Xb, Xte):
    """Subtract each class's mean logit over the (class-balanced) buffer items."""
    with torch.no_grad(): Lb = m(Xb); Lt = m(Xte)
    return (Lt - Lb.mean(0, keepdim=True)).argmax(1)

# ---------------------------------------------------------------- wave 2 helpers (plan2.md)
def affine_fit(Hthen, Hnow, alpha):
    """Ridge-to-identity affine map (row vectors): min ||Hnow - Hthen A - 1c'||^2 + lam ||A - I||^2,
    lam = alpha * tr(Hc'Hc)/d. Returns (A, c)."""
    Ht, Hn = Hthen.double(), Hnow.double(); mt, mn = Ht.mean(0), Hn.mean(0); Tc, Nc = Ht - mt, Hn - mn; d = Ht.shape[1]
    G = Tc.T @ Tc; lam = alpha * torch.trace(G) / d; I = torch.eye(d, dtype=torch.float64)
    A = torch.linalg.solve(G + lam * I, Tc.T @ Nc + lam * I); return A, mn - mt @ A
def drift_explained(pred, then, now):
    """1 - ||pred - now||^2 / ||then - now||^2 (1 = drift fully predicted, 0 = no better than the stale value)."""
    return float(1 - ((pred - now) ** 2).sum() / ((then - now) ** 2).sum())
def rank_corr(a, b):
    """Spearman correlation (average ranks for ties)."""
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)
def la_predict(L, cnt, tau=1.0):
    """Logit adjustment: argmax_k L_k - tau log pi_k, pi = training-stream class frequencies (unseen classes excluded)."""
    cnt = torch.as_tensor(cnt, dtype=torch.float64); pi = cnt / cnt.sum()
    adj = torch.where(cnt > 0, tau * torch.log(pi.clamp_min(1e-300)), torch.full_like(pi, float('inf')))
    return (L.double() - adj).argmax(1)
def lda_fit_chunked(fmap, X, y, classes, chunk=5000):
    """Class means and pooled within-class covariance of fmap(X), accumulated in chunks (exact; streaming-LDA statistics)."""
    S1, n, XX = {}, {}, None
    for i in range(0, len(X), chunk):
        F = fmap(X[i:i + chunk]).double(); yy = y[i:i + chunk]
        if XX is None: XX = torch.zeros(F.shape[1], F.shape[1], dtype=torch.float64)
        XX += F.T @ F
        for k in classes:
            s = yy == k
            if s.any(): S1[k] = S1.get(k, 0) + F[s].sum(0); n[k] = n.get(k, 0) + int(s.sum())
    means = {k: S1[k] / n[k] for k in classes}; N = sum(n.values())
    W = XX - sum(n[k] * torch.outer(means[k], means[k]) for k in classes)
    return means, W / (N - len(classes))
def lda_predict_chunked(means, cov, fmap, X, shrink=0.1, chunk=5000):
    return torch.cat([lda_predict(means, cov, fmap(X[i:i + chunk]), shrink) for i in range(0, len(X), chunk)])
def loo_affine_pred(Fthen, Fnow, labels, k, alpha, mean_then):
    """Fit the affine map on every item whose label is not k, then map class k's stored mean."""
    keep = torch.from_numpy(np.asarray(labels) != k); A, cc = affine_fit(Fthen[keep], Fnow[keep], alpha); return mean_then @ A + cc
def rp_map(d_in, d_out, seed):
    """Random ReLU expansion (RanPAC-style), fixed by its own generator."""
    g = torch.Generator().manual_seed(seed); R = torch.randn(d_in, d_out, generator=g) / math.sqrt(d_in); bvec = torch.randn(d_out, generator=g) * 0.1
    return lambda X: torch.relu(X @ R + bvec)

# ---------------------------------------------------------------- per-sample Fisher (T4)
def diag_fisher(m, X, y):
    params = {n: p.detach() for n, p in m.named_parameters()}
    def lossf(pr, x, t): return torch.nn.functional.cross_entropy(torch.func.functional_call(m, pr, (x.unsqueeze(0),)), t.unsqueeze(0))
    G = torch.func.vmap(torch.func.grad(lossf), in_dims=(None, 0, 0))(params, X, y)
    return torch.cat([(G[n] ** 2).mean(0).reshape(-1) for n, _ in m.named_parameters()]).double()

# ---------------------------------------------------------------- main loop
def run(cfg, seed):
    c = dict(dict(data='toy', sep=0.6, nmem=5, rep=4, bs=16, steps=300, lr=1e-3, w=1.0, loss='ce', buf='end', init='default',
                  warm=0, ewc='none', lam=0.0, ewc_every=25, stream=[0, 1, 2, 3, 4], snap='none', fresh=False,
                  probe_curve=False, readouts=False, curve_every=0), **cfg)
    Xtr, ytr, Xte, yte, m = setup(c, seed)
    K = 10; yn = ytr.numpy(); rng = np.random.default_rng(seed)
    hm = c.get('head', 'linear')
    if hm != 'linear': m[-1] = Head(m[-1], hm)
    def update_mu(Xnew):
        if 'center' not in hm: return
        with torch.no_grad():
            if len(buf) > 0:
                Fb = feats(m, Xtr[buf]); yb = ytr[buf]; m[-1].mu.copy_(torch.stack([Fb[yb == k].mean(0) for k in torch.unique(yb)]).mean(0))
            else: m[-1].mu.copy_(feats(m, Xnew).mean(0))
    names = [n for n, _ in m.named_parameters()]; sizes = [p.numel() for p in m.parameters()]; offs = np.cumsum([0] + sizes)
    head = np.zeros(int(offs[-1]), bool); head[offs[-3]:] = True        # last Linear: weight + bias
    Np = int(offs[-1]); mv = torch.zeros(Np, dtype=torch.float64); vv = torch.zeros(Np, dtype=torch.float64); t = 0
    beta1, beta2, eps = .9, .999, 1e-8
    buf = []; have = {}; snaps = {}; stats = {}; anchor = None; F = None; out = {'curve': [], 'probe_curve': [], 'ret': None}
    stream = [0] if c['fresh'] else c['stream']
    snaps['init'] = flat(m).detach().clone()
    # wave 2 measurement state; every measurement uses its own RNG and never touches the training draws
    rng_i = np.random.default_rng(seed + 31337); rng_d = np.random.default_rng(seed + 27183)
    cnt_all = np.zeros(K); cnt_last = np.zeros(K); cnt_last_rep = np.zeros(K); cnt_last_w = np.zeros(K)
    drift_store = {}; out['interf'] = []; out['rets'] = []; out['cyc'] = []; ref_bc = {}; resv = []
    if c.get('reserve'): assert c['buf'] == 'end' and 0 <= c['reserve'] <= c['nmem'], 'reserve needs buf=end and 0 <= reserve <= nmem'
    if c.get('drift') or c.get('stats2'): c['stats'] = True
    if c.get('stats'): assert len(set(stream)) == len(stream), 'stats/drift store each class once: not defined for streams that revisit a task'
    if c.get('floors'): assert stream[0] == 0, 'floors use the task-0-end snapshot'
    for ti, task in enumerate(stream):
        cls = [2 * task, 2 * task + 1]; idx = np.where(np.isin(yn, cls))[0]
        is_return = task in stream[:ti]
        cnt_last = np.zeros(K); cnt_last_rep = np.zeros(K); cnt_last_w = np.zeros(K)
        if c.get('interf') and ti > 0 and not is_return:   # W7: similarity of each new class to each old class BEFORE any update
            with torch.no_grad():
                Fbuf = feats(m, Xtr[buf]).double(); yb = ytr[buf]
                oldk = sorted(set(k for tk in stream[:ti] for k in (2 * tk, 2 * tk + 1)))    # classes of earlier tasks only
                hbar = torch.stack([Fbuf[yb == k].mean(0) for k in oldk]).mean(0)
                Xbr = Xtr[buf].double(); xbar = torch.stack([Xbr[yb == k].mean(0) for k in oldk]).mean(0)
                cs = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
                isamp = {k: rng_i.choice(np.where(yn == k)[0], 200, replace=False) for k in cls}
                interf_task = []
                for j in cls:
                    Fj = feats(m, Xtr[isamp[j]]).double().mean(0) - hbar; Xj = Xtr[isamp[j]].double().mean(0) - xbar
                    for k in oldk:
                        interf_task.append({'ti': ti, 'old': k, 'new': j, 'sim_feat': cs(Fbuf[yb == k].mean(0) - hbar, Fj),
                                            'dist_feat': float((Fbuf[yb == k].mean(0) - hbar - Fj).norm()),
                                            'sim_raw': cs(Xbr[yb == k].mean(0) - xbar, Xj)})
        if c['buf'] == 'start':
            for k in cls:
                if have.get(k, 0) < c['nmem']:
                    ci = np.where(yn == k)[0]; buf += rng.choice(ci, c['nmem'], replace=False).tolist(); have[k] = c['nmem']
        if is_return and c['snap'] != 'none':      # T5: act on a recurrence (oracle boundary; the filter is not under test here)
            # sources: right = task-1-end snapshot; 'wrong' = task-3-end snapshot (a STALE copy: replay kept task-1 rows
            # partly up to date, so it is not a null); 'init' = the initial model (a true null for the rows); noop = no change
            src = snaps[2] if 'wrong' in c['snap'] else snaps['init'] if 'init' in c['snap'] else snaps[0]
            cur = flat(m).detach().clone()
            if c['snap'] == 'noop': new = cur.clone()
            elif c['snap'].startswith('full'): new = src.clone()
            else:                                  # restore only this task's head rows (weight rows and biases)
                new = cur.clone(); Wo, bo = offs[-3], offs[-2]; d_in = sizes[-2] // K
                for k in cls:
                    new[Wo + k * d_in: Wo + (k + 1) * d_in] = src[Wo + k * d_in: Wo + (k + 1) * d_in]; new[bo + k] = src[bo + k]
            set_flat(m, new); assert torch.equal(flat(m).detach(), new)
            out['restore_changed'] = np.where((new != cur).numpy())[0].tolist() if c.get('_log_changed') else None
            mv.zero_()                              # spec: reset m on a switch, keep v (noop arm isolates this reset)
        if is_return:
            ref = out['ref'][task]; thr = 0.9 * ref; ttt = None
            def acc_task():
                with torch.no_grad(): return class_acc(m(Xte).argmax(1), yte, cls)
            a0 = acc_task(); ret_curve = [a0]; ret_steps = [0]
            if a0 >= thr: ttt = 0
            if c.get('ret_bc'):   # FU-d: savings measured through a bias-corrected readout (ER's raw ttt is dominated by recency)
                seen_sofar = sorted(set(k for tk in stream[:ti + 1] for k in (2 * tk, 2 * tk + 1)))
                def acc_bc():
                    with torch.no_grad(): pb_ = biascorr_predict(m, Xtr[buf], Xte); pr_ = m(Xte).argmax(1)
                    return class_acc(pb_, yte, cls), class_acc(pb_, yte, seen_sofar), class_acc(pr_, yte, seen_sofar)
                thr_bc = 0.9 * ref_bc[task]; b0 = acc_bc(); bc_curve = [b0]; ttt_bc = 0 if b0[0] >= thr_bc else None
        if c['fresh']:
            with torch.no_grad(): out['fresh_curve'] = [(0, class_acc(m(Xte).argmax(1), yte, cls))]
        for st in range(c['steps']):
            b = rng.choice(idx, c['bs'], replace=False); X, y = Xtr[b], ytr[b]
            if st == 0 and c['init'] == 'mean' and ti > 0 and not is_return:   # T1(b): new rows at the class-mean feature
                with torch.no_grad():
                    Fb = feats(m, X); W = m[-1].weight; bb = m[-1].bias
                    old = [k for k in range(K) if k < 2 * task]; nrm = W[old].norm(dim=1).mean()
                    for k in cls:
                        if (y == k).any():
                            mu = Fb[y == k].mean(0); W[k] = mu / (mu.norm() + 1e-12) * nrm; bb[k] = bb[old].mean()
            if st == 0 and c['init'] == 'cmean' and ti > 0 and not is_return:  # T1(b'), fixed after review: CENTRED class mean
                with torch.no_grad():                  # ReLU features share a large mean direction; an uncentred mean row
                    Fb = feats(m, X); W = m[-1].weight; bb = m[-1].bias   # aligns with it and outranks every old class
                    Fbuf = feats(m, Xtr[buf]); yb = ytr[buf]
                    hbar = torch.stack([Fbuf[yb == k].mean(0) for k in torch.unique(yb)]).mean(0)
                    old = [k for k in range(K) if k < 2 * task]; nrm = W[old].norm(dim=1).mean()
                    for k in cls:
                        if (y == k).any():
                            v = Fb[y == k].mean(0) - hbar; W[k] = v / (v.norm() + 1e-12) * nrm
                    # safe bias: on every buffer item x, the new logit stays 0.1 below the item's own-class logit
                    Lb = m(Xtr[buf]); own = Lb[torch.arange(len(buf)), yb]
                    for k in cls:
                        if (y == k).any(): bb[k] = (own - (Fbuf @ W[k] + bb[k]) + bb[k]).min() - 0.1
            if st == 0 and c['curve_every'] and ti > 0:
                with torch.no_grad(): out['curve'].append((ti, 0, class_acc(m(Xte).argmax(1), yte, range(2 * task))))
            nb = c['bs']; update_mu(X)
            if c.get('interf') and st == 0 and ti > 0 and not is_return:   # first-batch-only version of the new-class mean
                with torch.no_grad():
                    Fb1 = feats(m, X).double()
                    for e in interf_task:
                        s1 = y == e['new']
                        e['sim_feat_b1'] = cs(Fbuf[yb == e['old']].mean(0) - hbar, Fb1[s1].mean(0) - hbar) if s1.any() else None
            if len(buf) > 0:
                rb = rng.choice(buf, c['rep'], replace=len(buf) < c['rep']); X = torch.cat([X, Xtr[rb]]); y = torch.cat([y, ytr[rb]])
            bc_ = np.bincount(y.numpy(), minlength=K); cnt_all += bc_; cnt_last += bc_
            cnt_last_rep += np.bincount(y[nb:].numpy(), minlength=K)   # replay items only (the only cross-class competition under ACE)
            # audit 2026-10-29: the prior the LOSS induces = per-item loss coefficients (replay items carry weight w; CE path w = 1)
            cnt_last_w += np.bincount(y[:nb].numpy(), minlength=K) + c['w'] * np.bincount(y[nb:].numpy(), minlength=K)
            m.zero_grad(); o = m(X)
            if c['loss'] == 'ace':
                mask = torch.full((K,), float('-inf')); mask[cls] = 0; o = torch.cat([o[:nb] + mask, o[nb:]])
            if c['w'] == 1.0 and c['loss'] == 'ce':
                loss = torch.nn.functional.cross_entropy(o, y)          # identical op to toy_cl3 (S1)
            else:
                ce = torch.nn.functional.cross_entropy(o, y, reduction='none'); wts = torch.ones(len(y)); wts[nb:] = c['w']
                loss = (ce * wts).sum() / wts.sum()
            loss.backward()
            g = torch.cat([p.grad.reshape(-1) for p in m.parameters()]).double(); t += 1
            if c['ewc'] != 'none' and anchor is not None and c['lam'] > 0:
                if c['ewc'] == 'fresh' and st % c['ewc_every'] == 0 and len(buf) > 0 and not c.get('_test_zeroF'):
                    F = diag_fisher(m, Xtr[buf], ytr[buf]); F = F / F.mean()
                g = g + c['lam'] * F * (flat(m).detach().double() - anchor)
            mv = beta1 * mv + (1 - beta1) * g; vv = beta2 * vv + (1 - beta2) * g * g
            mh = mv / (1 - beta1 ** t); vh = vv / (1 - beta2 ** t)
            Bd = (c['lr'] / (vh.sqrt() + eps)).numpy(); d = -Bd * mh.numpy()
            if c['warm'] > 0 and ti > 0 and not is_return and st < c['warm']:
                d = np.where(np.ones_like(head) if c.get('_test_mask_all') else head, d, 0.0)   # T1(c): head-only warm-up
            set_flat(m, flat(m).detach().double().add(torch.from_numpy(d)).float())
            if c['curve_every'] and ti > 0 and (st + 1) % c['curve_every'] == 0:
                with torch.no_grad(): out['curve'].append((ti, st + 1, class_acc(m(Xte).argmax(1), yte, range(2 * task))))
            dense = (st + 1) <= 50 or (st + 1) % 5 == 0          # every step for the first 50 steps, then every 5
            if c['fresh'] and dense:
                with torch.no_grad(): out.setdefault('fresh_curve', []).append((st + 1, class_acc(m(Xte).argmax(1), yte, cls)))
            if is_return and dense:
                a = acc_task(); ret_curve.append(a); ret_steps.append(st + 1)
                if ttt is None and a >= thr: ttt = st + 1
                if c.get('ret_bc'):
                    bb_ = acc_bc(); bc_curve.append(bb_)
                    if ttt_bc is None and bb_[0] >= thr_bc: ttt_bc = st + 1
        # end of task
        if c['buf'] == 'end':
            for k in cls:
                if have.get(k, 0) < c['nmem']:
                    ci = np.where(yn == k)[0]; drawn = rng.choice(ci, c['nmem'], replace=False).tolist(); have[k] = c['nmem']
                    nr = int(c.get('reserve', 0))   # audit follow-up: the last nr drawn items are stored but NEVER replayed
                    buf += drawn[:c['nmem'] - nr]; resv += drawn[c['nmem'] - nr:]
        with torch.no_grad(): pr = m(Xte).argmax(1)
        out.setdefault('ref', {})
        if not is_return: out['ref'][task] = class_acc(pr, yte, cls)
        if c.get('ret_bc') and not is_return: ref_bc[task] = class_acc(biascorr_predict(m, Xtr[buf], Xte), yte, cls)
        if task in (0, 2) and not is_return: snaps[task] = flat(m).detach().clone()
        if c.get('stats') and not is_return:      # F7: class statistics written once, at the class's own task end
            Fall = feats(m, Xtr[idx]).double(); yall = ytr[idx]
            for k in cls:
                stats.setdefault('mean', {})[k] = Fall[yall == k].mean(0)
                bk = [i for i in buf if yn[i] == k]; stats.setdefault('bufmean_then', {})[k] = feats(m, Xtr[bk]).double().mean(0)
            stats.setdefault('covs', []).append((pooled_cov(Fall, yall), len(idx)))
        if c.get('interf') and ti > 0 and not is_return:   # W7: confusion old -> new at task end, test set
            for e in interf_task:
                sel = yte == e['old']; e['conf_end'] = float((pr[sel] == e['new']).float().mean()); out['interf'].append(e)
        if c.get('drift') and not is_return:              # W2: features at this time of the buffer and of a fixed subsample
            seen_now = sorted(set(k for tk in stream[:ti + 1] for k in (2 * tk, 2 * tk + 1)))
            pool = np.where(np.isin(yn, seen_now))[0]; sub = rng_d.choice(pool, min(4000, len(pool)), replace=False)
            suba = rng_d.choice(len(yn), min(4000, len(yn)), replace=False)   # all 10 classes, incl. not-yet-seen ones (oracle support for LOO)
            drift_store[ti] = {'buf': list(buf), 'Fbuf': feats(m, Xtr[buf]).double(), 'sub': sub, 'Fsub': feats(m, Xtr[sub]).double(),
                               'suba': suba, 'Fsuba': feats(m, Xtr[suba]).double()}
            for k in cls: stats.setdefault('t_of', {})[k] = ti
        if c.get('cyc'):                                  # W6: per task end, accuracy on everything seen so far, raw and bias-corrected
            seen_now = sorted(set(k for tk in stream[:ti + 1] for k in (2 * tk, 2 * tk + 1)))
            out['cyc'].append({'ti': ti, 'task': task, 'acc_seen': class_acc(pr, yte, seen_now), 'acc_task': class_acc(pr, yte, cls),
                               'bc_seen': class_acc(biascorr_predict(m, Xtr[buf], Xte), yte, seen_now) if len(buf) else None})
        if c['ewc'] != 'none':
            anchor = flat(m).detach().double().clone()
            Fn = diag_fisher(m, Xtr[buf], ytr[buf]); Fn = Fn / Fn.mean()
            if c['ewc'] == 'stale_avg': F = Fn if F is None else (F * ti + Fn) / (ti + 1)   # running mean over tasks (separate arm)
            else: F = Fn            # 'stale': latest task-end Fisher, frozen until the next task end; 'fresh' starts from it too
            if c.get('_test_zeroF'): F = torch.zeros_like(Fn)
        if c['probe_curve'] and ti > 0 and not is_return:
            seen = list(range(2 * task + 2)); old = list(range(2 * task))
            Ft, Fe = feats(m, Xtr), feats(m, Xte)
            sel = np.where(np.isin(yn, seen))[0]; selte = np.isin(yte.numpy(), seen)
            pp = torch.full_like(yte, -1); pp[selte] = refit(Ft[sel], ytr[sel], Fe[selte])
            bsel = [i for i in buf if yn[i] in seen]
            pb = torch.full_like(yte, -1); pb[selte] = refit(Ft[bsel], ytr[bsel], Fe[selte])
            out['probe_curve'].append({'after_task': ti, 'model_old': class_acc(pr, yte, old), 'probe_full_old': class_acc(pp, yte, old),
                                       'probe_buf_old': class_acc(pb, yte, old)})
        if is_return:
            with torch.no_grad(): pr2 = m(Xte).argmax(1)
            others = [k for k in range(K) if k not in cls]
            out['ret'] = {'ref': ref, 'ttt': ttt if ttt is not None else c['steps'] + 5, 'censored': ttt is None,
                          'acc_task_start': ret_curve[0], 'acc_task_end': ret_curve[-1], 'acc_others_end': class_acc(pr2, yte, others),
                          'auc50': float(np.mean(ret_curve[:51])), 'curve': ret_curve, 'curve_steps': ret_steps}
            if c.get('ret_bc'):
                tb = [x[0] for x in bc_curve]
                out['ret']['bc'] = {'ref_bc': ref_bc[task], 'ttt_bc': ttt_bc if ttt_bc is not None else c['steps'] + 5, 'censored_bc': ttt_bc is None,
                                    'start_bc': tb[0], 'auc50_bc': float(np.mean(tb[:51])), 'end_bc': tb[-1],
                                    'seen_bc_curve': [x[1] for x in bc_curve], 'seen_raw_curve': [x[2] for x in bc_curve]}
            out['rets'].append(dict(out['ret'], ti=ti, task=task, occurrence=stream[:ti + 1].count(task)))
    with torch.no_grad(): pr = m(Xte).argmax(1)
    seen = sorted(set(k for tk in stream for k in (2 * tk, 2 * tk + 1)))
    out['acc'] = class_acc(pr, yte, seen)
    last = [2 * stream[-1], 2 * stream[-1] + 1]; old = [k for k in seen if k not in last]
    out['old_acc'] = class_acc(pr, yte, old) if old else None; out['last_acc'] = class_acc(pr, yte, last)
    if c.get('reserve') is not None:   # LDA readouts from replayed items, from never-replayed reserve items, and from both
        def lda_items(ii, cov_ii=None):
            """Class means from items ii; pooled within-class covariance from cov_ii (default ii). None if a class is
            missing, or if the covariance items have no within-class scatter (one item per class: the audit's 7.11 case)."""
            cov_ii = ii if cov_ii is None else cov_ii
            if len(ii) == 0 or len(set(ytr[ii].tolist())) < len(seen) or len(cov_ii) - len(set(ytr[cov_ii].tolist())) < 1: return None
            F_ = feats(m, Xtr[ii]).double(); y_ = ytr[ii]; mus = {k: F_[y_ == k].mean(0) for k in seen}
            Fc = feats(m, Xtr[cov_ii]).double(); yc = ytr[cov_ii]; kc = sorted(set(yc.tolist()))
            R_ = torch.cat([Fc[yc == k] - Fc[yc == k].mean(0) for k in kc]); cov = R_.T @ R_ / max(1, len(R_) - len(kc))
            p_ = lda_predict(mus, cov, feats(m, Xte)); return {'all': class_acc(p_, yte, seen), 'old': class_acc(p_, yte, old), 'last': class_acc(p_, yte, last)}
        out['resv'] = {'n_buf': len(buf), 'n_resv': len(resv), 'lda_buf': lda_items(buf), 'lda_resv': lda_items(resv), 'lda_both': lda_items(buf + resv),
                       'lda_resvmean_bothcov': lda_items(resv, buf + resv) if resv else None,
                       'biascorr_buf': class_acc(biascorr_predict(m, Xtr[buf], Xte), yte, seen) if buf else None,
                       'biascorr_resv': class_acc(biascorr_predict(m, Xtr[resv], Xte), yte, seen) if resv else None}
    if c['readouts']:
        Fb, yb = feats(m, Xtr[buf]), ytr[buf]; Fte = feats(m, Xte)
        if hm == 'linear': lw, gamma = wa_logits(m, Xte, last, old); wa = class_acc(lw.argmax(1), yte, seen)
        else: wa, gamma = None, None                       # WA is defined for the plain linear head only
        out['ro'] = {'wa': wa, 'wa_gamma': gamma,
                     'biascorr': class_acc(biascorr_predict(m, Xtr[buf], Xte), yte, seen),
                     'ncm_buf': class_acc(ncm_predict(Fb, yb, Fte), yte, seen),
                     'refit_buf': class_acc(refit(Fb, yb, Fte), yte, seen),
                     'refit_full': class_acc(refit(feats(m, Xtr), ytr, Fte), yte, seen)}
    if c.get('stats'):
        # F7: can a readout built from STORED statistics beat one built from the 5-per-class buffer? Does staleness hurt (H1)?
        Fte = feats(m, Xte); Fb = feats(m, Xtr[buf]).double(); yb = ytr[buf]
        cov_stale = sum(C * n for C, n in stats['covs']) / sum(n for _, n in stats['covs'])
        m_buf = {k: Fb[yb == k].mean(0) for k in seen}; cov_buf = pooled_cov(Fb, yb)
        Fall = feats(m, Xtr).double(); m_full = {k: Fall[ytr == k].mean(0) for k in seen}; cov_full = pooled_cov(Fall, ytr)
        m_sdc = {k: stats['mean'][k] + (m_buf[k] - stats['bufmean_then'][k]) for k in seen}   # drift-compensated stored means
        for k in last: assert torch.allclose(stats['mean'][k], m_full[k], atol=1e-4), 'last-task stored mean must equal the current mean'
        if c.get('stats2'):   # W1: which statistic carries the buffer-vs-oracle gap: the means or the covariance?
            Id = torch.eye(cov_full.shape[0], dtype=torch.float64)
            out['stats2'] = {'lda_fullmean_bufcov': class_acc(lda_predict(m_full, cov_buf, Fte), yte, seen),
                             'lda_bufmean_fullcov': class_acc(lda_predict(m_buf, cov_full, Fte), yte, seen),
                             'euc_fullmean': class_acc(lda_predict(m_full, Id, Fte, shrink=1.0), yte, seen),
                             'euc_bufmean': class_acc(lda_predict(m_buf, Id, Fte, shrink=1.0), yte, seen),
                             'ncm_cos_fullmean': class_acc(ncm_predict(Fall.float(), ytr, Fte), yte, seen)}
        if c.get('drift'):    # W2: can the CURRENT mean of an old class be predicted from statistics stored at its task end?
            fits = {}
            for t, ds in drift_store.items():
                Hnb = feats(m, Xtr[ds['buf']]).double(); Hns = feats(m, Xtr[ds['sub']]).double()
                fits[t] = {'trans_global': (torch.eye(Hnb.shape[1], dtype=torch.float64), (Hnb - ds['Fbuf']).mean(0)),
                           'aff_oracle': affine_fit(ds['Fsub'], Hns, 1e-3)}
                for a in (0.1, 1.0, 10.0): fits[t][f'aff_buf_{a}'] = affine_fit(ds['Fbuf'], Hnb, a)
            preds = {'stale': {k: stats['mean'][k] for k in old}, 'sdc_class': {k: m_sdc[k] for k in old}}
            for name in fits[0]:
                preds[name] = {k: stats['mean'][k] @ fits[stats['t_of'][k]][name][0] + fits[stats['t_of'][k]][name][1] for k in old}
            # leave-class-out: an affine map has enough parameters to move ~10 class means anywhere, so a fit that sees
            # class k's own items is weak evidence; these fit the map WITHOUT class k and then predict class k's mean
            preds['aff_oracle_loo'], preds['aff_oracle_allcls_loo'], preds['aff_buf_loo_1.0'] = {}, {}, {}
            for k in old:
                ds = drift_store[stats['t_of'][k]]
                preds['aff_oracle_loo'][k] = loo_affine_pred(ds['Fsub'], feats(m, Xtr[ds['sub']]).double(), yn[ds['sub']], k, 1e-3, stats['mean'][k])
                preds['aff_oracle_allcls_loo'][k] = loo_affine_pred(ds['Fsuba'], feats(m, Xtr[ds['suba']]).double(), yn[ds['suba']], k, 1e-3, stats['mean'][k])
                bk = np.array(ds['buf'])
                preds['aff_buf_loo_1.0'][k] = loo_affine_pred(ds['Fbuf'], feats(m, Xtr[bk]).double(), yn[bk], k, 1.0, stats['mean'][k])
            preds['oracle_now'] = {k: m_full[k] for k in old}
            out['drift'] = {}
            for name, P in preds.items():
                E = [drift_explained(P[k], stats['mean'][k], m_full[k]) for k in old]
                num = sum(float(((P[k] - m_full[k]) ** 2).sum()) for k in old); den = sum(float(((stats['mean'][k] - m_full[k]) ** 2).sum()) for k in old)
                mm = {**P, **{k: m_full[k] for k in last}}
                out['drift'][name] = {'E_mean': float(np.mean(E)), 'E_min': float(np.min(E)), 'E_pooled': 1 - num / den,
                                      'resid_norm_frac': math.sqrt(num / den),     # residual drift norm / drift norm (pooled)
                                      'E_by_task': {int(t): float(np.mean([e for e, k in zip(E, old) if stats['t_of'][k] == t])) for t in sorted(set(stats['t_of'][k] for k in old))},
                                      'lda_fullcov': class_acc(lda_predict(mm, cov_full, Fte), yte, seen)}
            out['drift']['rel_drift'] = float(np.mean([float((stats['mean'][k] - m_full[k]).norm() / m_full[k].norm()) for k in old]))
        out['stats'] = {'lda_buf_fresh': class_acc(lda_predict(m_buf, cov_buf, Fte), yte, seen),
                        'lda_stored_stale': class_acc(lda_predict(stats['mean'], cov_stale, Fte), yte, seen),
                        'lda_stored_driftcomp': class_acc(lda_predict(m_sdc, cov_stale, Fte), yte, seen),
                        'lda_bufmean_storedcov': class_acc(lda_predict(m_buf, cov_stale, Fte), yte, seen),
                        'lda_full_fresh_oracle': class_acc(lda_predict(m_full, cov_full, Fte), yte, seen)}
    if c.get('mech'):
        # Mechanism of the offset (follow-up F4): split each logit as f_c(x) = w_c.hbar + w_c.(h(x) - hbar) + b_c,
        # hbar = mean penultimate feature of the class-balanced buffer. For old-class test items predicted as a
        # newest-task class j, average the gap (f_j - f_y) into the common term, the item-specific term and the bias term.
        assert hm in ('linear', 'center'), 'mech decomposition needs a linear (possibly centred) head'
        sh = m[-1].mu if hm == 'center' else 0.0           # centred head: logits are linear in h - mu
        Fb = feats(m, Xtr[buf]) - sh; hbar = Fb.mean(0); Fte = feats(m, Xte) - sh; W = m[-1].weight.detach(); bb = m[-1].bias.detach()
        with torch.no_grad(): prd = m(Xte).argmax(1)
        sel = torch.isin(yte, torch.tensor(old)) & torch.isin(prd, torch.tensor(last))
        yy, jj = yte[sel], prd[sel]; Hs = Fte[sel]
        common = ((W[jj] - W[yy]) @ hbar); specific = ((W[jj] - W[yy]) * (Hs - hbar)).sum(1); bias = bb[jj] - bb[yy]
        tot = common + specific + bias
        assert torch.allclose(tot, (Hs * W[jj]).sum(1) + bb[jj] - (Hs * W[yy]).sum(1) - bb[yy], atol=1e-4)
        cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        out['mech'] = {'n_items': int(sel.sum()), 'gap_total': float(tot.mean()), 'gap_common': float(common.mean()),
                       'gap_specific': float(specific.mean()), 'gap_bias': float(bias.mean()),
                       'cos_newrows_hbar': float(np.mean([cos(W[k], hbar) for k in last])),
                       'cos_oldrows_hbar': float(np.mean([cos(W[k], hbar) for k in old])),
                       'norm_new': float(W[last].norm(dim=1).mean()), 'norm_old': float(W[old].norm(dim=1).mean())}
    if c.get('floors'):       # W3/R1: how much of the 'retained' feature information is just what untrained features carry?
        sel = np.where(np.isin(yn, seen))[0]; Xs, ys = Xtr[sel], ytr[sel]
        def lda_acc(fmap):
            mu, cv = lda_fit_chunked(fmap, Xs, ys, seen); return class_acc(lda_predict_chunked(mu, cv, fmap, Xte), yte, seen)
        m0 = copy.deepcopy(m); set_flat(m0, snaps['init']); m1 = copy.deepcopy(m); set_flat(m1, snaps[0])
        fm = lambda mm: (lambda X: feats(mm, X))
        rp = rp_map(Xtr.shape[1], 1024 if c['data'] == 'mnist' else 512, seed + 777)
        out['floors'] = {'lda_now': lda_acc(fm(m)), 'lda_init': lda_acc(fm(m0)), 'lda_task1': lda_acc(fm(m1)),
                         'lda_raw': lda_acc(lambda X: X), 'lda_rp': lda_acc(rp),
                         'refit_now': class_acc(refit(feats(m, Xs), ys, feats(m, Xte)), yte, seen),
                         'refit_init': class_acc(refit(feats(m0, Xs), ys, feats(m0, Xte)), yte, seen),
                         'refit_raw': class_acc(refit(Xs, ys, Xte), yte, seen)}
    if c.get('la'):           # R12: closed-form logit adjustment by the training-stream class frequencies (no fitted parameter)
        with torch.no_grad(): Lt = m(Xte)
        out['la'] = {f'{wn}_tau{tau}': class_acc(la_predict(Lt, cn, tau), yte, seen)
                     for wn, cn in (('all', cnt_all), ('last', cnt_last), ('last_rep', cnt_last_rep)) for tau in (0.5, 1.0, 2.0)}
        out['la']['pi_last'] = (cnt_last / cnt_last.sum()).tolist()
        for tau in (0.5, 1.0, 2.0): out['la'][f'lastw_tau{tau}'] = class_acc(la_predict(Lt, cnt_last_w, tau), yte, seen)
        out['la']['pi_last_w'] = (cnt_last_w / cnt_last_w.sum()).tolist()
        # diagnostics (oracle, test set): best tau for LA(last_w); and the temperature that calibrates the 2-way softmax
        # among the newest classes on their own test items. A pure global logit scale a predicts tau* ~ T_new ~ a.
        taus = np.round(np.arange(0.0, 4.01, 0.1), 2); accs = [class_acc(la_predict(Lt, cnt_last_w, t), yte, seen) for t in taus]
        out['la']['tau_star'] = float(taus[int(np.argmax(accs))]); out['la']['acc_tau_star'] = float(max(accs))
        sel = torch.isin(yte, torch.tensor(last)); L2 = Lt[sel][:, last].double(); y2 = (yte[sel] == last[1]).long()
        Ts = np.exp(np.linspace(np.log(0.25), np.log(16), 81))
        nll = [float(torch.nn.functional.cross_entropy(L2 / T, y2)) for T in Ts]; out['la']['T_new'] = float(Ts[int(np.argmin(nll))])
    if c.get('sleep') is not None and c.get('sleep') is not False:   # R13: end-of-stream consolidation on the buffer alone (copy)
        out['sleep'] = {}
        for mode in ('head', 'full'):
            ms_ = copy.deepcopy(m); prm = list(ms_[-1].parameters()) if mode == 'head' else list(ms_.parameters())
            opt = torch.optim.Adam(prm, 1e-3); rs = np.random.default_rng(seed + 555)
            for _ in range(int(c.get('sleep_steps', 200))):
                b = rs.choice(buf, 20, replace=True); opt.zero_grad(); torch.nn.functional.cross_entropy(ms_(Xtr[b]), ytr[b]).backward(); opt.step()
            with torch.no_grad(): out['sleep'][mode] = class_acc(ms_(Xte).argmax(1), yte, seen)
    if c.get('save_model'):   # checkpoint for frozen-checkpoint readout tests (readout_checks.py)
        torch.save({'state': m.state_dict(), 'buf': list(buf), 'cfg': {k: v for k, v in c.items() if k != 'save_model'}, 'seed': seed},
                   c['save_model'].format(seed=seed))
    out['buf_size'] = len(buf); out.pop('ref', None) if not out.get('ret') else None
    return out

# ---------------------------------------------------------------- self-tests
def selftest():
    ok = True
    def check(name, cond, info=''):
        nonlocal ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info, flush=True)
    ER = [0.3805, 0.4805, 0.371, 0.42, 0.424]
    r = [round(run({}, s)['acc'], 4) for s in range(5)]
    check('S1 toy ER reproduces toy_cl3 per seed', r == ER, r)
    a, b = run({'init': 'mean', 'warm': 30}, 1), run({'init': 'mean', 'warm': 30}, 1)
    check('S2 determinism (same cfg twice)', a['acc'] == b['acc'] and a['old_acc'] == b['old_acc'], (a['acc'], b['acc']))
    base = run({}, 2)['acc']
    offs = {k: run(v, 2)['acc'] for k, v in {'w=1': {'w': 1.0}, 'warm=0': {'warm': 0}, 'init=default': {'init': 'default'},
                                              'ewc lam=0': {'ewc': 'fresh', 'lam': 0.0}}.items()}
    check('S3 null switches reproduce ER exactly', all(v == base for v in offs.values()), (base, offs))
    ce_w = run({'w': 1.0 + 1e-12}, 2)['acc']
    check('S3b weighted-CE code path at w~1 equals ER (to rounding)', abs(ce_w - base) < 0.006, (ce_w, base))
    ACE = [0.557, 0.641, 0.553, 0.517, 0.622]
    r = [round(run({'loss': 'ace', 'buf': 'start'}, s)['acc'], 3) for s in range(5)]
    check('S4 ER-ACE start-insert reproduces diag.py', r == ACE, r)
    # S5 NCM: separable synthetic features
    g = torch.Generator().manual_seed(0); Mu = torch.randn(10, 32, generator=g) * 5
    Fb = torch.cat([Mu[k] + torch.randn(5, 32, generator=g) * .3 for k in range(10)]); yb = torch.arange(10).repeat_interleave(5)
    Ft = torch.cat([Mu[k] + torch.randn(50, 32, generator=g) * .3 for k in range(10)]); yt = torch.arange(10).repeat_interleave(50)
    check('S5 NCM = 100% on separable features', (ncm_predict(Fb, yb, Ft) == yt).all().item())
    pr = refit(Fb, yb, Ft); check('S5b refit probe = 100% on separable features', (pr == yt).all().item())
    mus = {k: Fb[yb == k].double().mean(0) for k in range(10)}
    check('S11 LDA readout = 100% on separable features', (lda_predict(mus, pooled_cov(Fb, yb), Ft) == yt).all().item())
    Rr = torch.randn(32, 32, generator=g, dtype=torch.float64); Ct = Rr @ Rr.T / 32
    Z = torch.distributions.MultivariateNormal(torch.zeros(32, dtype=torch.float64), Ct).sample((20000,)); yz = torch.arange(2).repeat(10000)
    check('S11b pooled_cov recovers a known covariance (rel err < 5%)', float((pooled_cov(Z, yz) - Ct).norm() / Ct.norm()) < 0.05)
    # S6 WA unchanged when norms equal
    _, _, Xte, _, m = setup({'data': 'toy', 'sep': .6, 'nmem': 5, 'rep': 4}, 0)
    with torch.no_grad(): W = m[-1].weight; W.copy_(torch.nn.functional.normalize(W, dim=1))
    lw, gm = wa_logits(m, Xte, [8, 9], list(range(8)))
    with torch.no_grad(): check('S6 WA no-op when row norms equal', abs(gm - 1) < 1e-6 and torch.equal(lw.argmax(1), m(Xte).argmax(1)), gm)
    # S7 snapshots: return task, full restore must equal snapshot bit-for-bit (asserted inside run); wrong != right
    s_full = run({'stream': [0, 1, 2, 3, 4, 0], 'snap': 'full', 'steps': 300}, 0)['ret']
    s_wrong = run({'stream': [0, 1, 2, 3, 4, 0], 'snap': 'full_wrong', 'steps': 300}, 0)['ret']
    check('S7 snapshot restore runs; right and wrong snapshots differ at return start',
          s_full['acc_task_start'] != s_wrong['acc_task_start'], (s_full['acc_task_start'], s_wrong['acc_task_start']))
    check('S7b full restore recovers the reference accuracy at return start (within 1e-9)',
          abs(s_full['acc_task_start'] - s_full['ref']) < 1e-9, (s_full['acc_task_start'], s_full['ref']))
    # S3c/S3d: non-vacuous null switches
    check('S3c warm-up with an all-true mask equals ER exactly', run({'warm': 30, '_test_mask_all': True}, 2)['acc'] == base)
    check('S3d EWC with lam > 0 but F = 0 equals ER exactly', run({'ewc': 'fresh', 'lam': 10.0, '_test_zeroF': True}, 2)['acc'] == base)
    # S6b WA with unequal norms: new rows' mean norm becomes the old mean
    with torch.no_grad(): m[-1].weight[8:] *= 3.0
    lw, gm = wa_logits(m, Xte, [8, 9], list(range(8)))
    W = m[-1].weight.detach(); check('S6b WA gamma = 1/3 when new rows are 3x larger', abs(gm - 1/3) < 1e-5, gm)
    # S7c rows restore touches only the task's head rows (weights + biases)
    r = run({'stream': [0, 1, 2, 3, 4, 0], 'snap': 'rows', '_log_changed': True}, 0)
    _, _, _, _, m2 = setup({'data': 'toy'}, 0); sz = [p.numel() for p in m2.parameters()]; of = np.cumsum([0] + sz); din = sz[-2] // 10
    allowed = set(range(of[-3], of[-3] + 2 * din)) | {of[-2], of[-2] + 1}
    ch = set(r['restore_changed']); check('S7c rows restore changes only rows 0-1 and their biases', len(ch) > 0 and ch <= allowed, (len(ch), len(allowed)))
    rn = run({'stream': [0, 1, 2, 3, 4, 0], 'snap': 'noop', '_log_changed': True}, 0)
    check('S7d noop restore changes no parameter', rn['restore_changed'] == [])
    # S10 sanity of the centred init (reviewer found the uncentred one collapses old classes to 0 at step 0)
    rc = run({'init': 'cmean', 'curve_every': 10}, 0); rd = run({'curve_every': 10}, 0)
    a0c = [a for ti, st, a in rc['curve'] if st == 0]; a0d = [a for ti, st, a in rd['curve'] if st == 0]
    # S10 is informational (a property of the init method, not of harness correctness): the T1 dip metric includes step 0,
    # so any step-0 damage caused by an init is charged to that arm.
    print('  info S10: step-0 old-class acc, centred init', np.round(a0c, 3), 'default', np.round(a0d, 3))
    check('S10b step-0 point recorded once per task in both arms', len(a0c) == 4 and len(a0d) == 4)
    ru = run({'init': 'mean', 'curve_every': 10}, 0); a0u = [a for ti, st, a in ru['curve'] if st == 0]
    print('  info: uncentred mean init step-0 old-class acc', np.round(a0u, 3))
    # S8 MNIST sanity
    Xtr, ytr, Xte, yte = mnist()
    check('S8 MNIST shapes, classes, range', Xtr.shape == (60000, 784) and Xte.shape == (10000, 784) and
          sorted(set(ytr.tolist())) == list(range(10)) and float(Xtr.min()) == 0. and float(Xtr.max()) == 1.)
    j = run({'data': 'mnist', 'stream': [0], 'steps': 1}, 0)   # smoke
    jt = joint_mnist(0); ft = run({'data': 'mnist', 'rep': 0, 'nmem': 0}, 0)
    check('S8b MNIST joint upper bound > 0.95', jt > 0.95, jt)
    check('S8c MNIST fine-tune keeps only the last task (old acc < 0.05, last > 0.95)', ft['old_acc'] < 0.05 and ft['last_acc'] > 0.95, (ft['old_acc'], ft['last_acc']))
    print('SELFTEST', 'ALL PASS' if ok else 'FAILURES', flush=True); return ok

def selftest2():
    """Wave-2 checks (plan2.md, N1-N10)."""
    ok = True
    def check(name, cond, info=''):
        nonlocal ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info, flush=True)
    g = torch.Generator().manual_seed(1)
    # N2 affine fit recovers a planted map; huge ridge gives identity + mean shift
    Ht = torch.randn(60, 8, generator=g, dtype=torch.float64); At = torch.randn(8, 8, generator=g, dtype=torch.float64); ct = torch.randn(8, generator=g, dtype=torch.float64)
    A, cc = affine_fit(Ht, Ht @ At + ct, 1e-12)
    check('N2 affine_fit recovers a planted affine map', float((A - At).norm() / At.norm()) < 1e-6 and float((cc - ct).norm()) < 1e-6)
    A, cc = affine_fit(Ht, Ht @ At + ct, 1e10); sh = (Ht @ At + ct).mean(0) - Ht.mean(0)
    check('N2b affine_fit with huge ridge = identity + mean shift', float((A - torch.eye(8, dtype=torch.float64)).norm()) < 1e-6 and float((cc - sh).norm()) < 1e-6)
    a, b = torch.randn(8, dtype=torch.float64, generator=g), torch.randn(8, dtype=torch.float64, generator=g)
    check('N3 drift_explained: 1 at the truth, 0 at the stale value, 0.75 halfway', abs(drift_explained(b, a, b) - 1) < 1e-12 and abs(drift_explained(a, a, b)) < 1e-12 and abs(drift_explained((a + b) / 2, a, b) - 0.75) < 1e-12)
    check('N5 rank_corr: +1 monotone, -1 reversed', abs(rank_corr([1, 2, 3, 4], [10, 20, 35, 90]) - 1) < 1e-12 and abs(rank_corr([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12)
    L = torch.randn(50, 10, generator=g)
    check('N7 logit adjustment with uniform counts = plain argmax', torch.equal(la_predict(L, np.full(10, 7.0)), L.argmax(1)))
    cnt = np.ones(10); cnt[3] = 0
    check('N7b logit adjustment never predicts a class with zero count', not (la_predict(L, cnt) == 3).any().item())
    Lp = torch.zeros(4, 10); cnt = np.ones(10); cnt[0] = 100.0
    check('N7c logit adjustment lowers a frequent class (ties broken away from it)', not (la_predict(Lp + 1e-3 * torch.arange(10).flip(0), cnt) == 0).any().item())
    # N9 chunked streaming-LDA statistics equal the direct computation
    Z = torch.randn(1234, 6, generator=g, dtype=torch.float64); yz = torch.randint(0, 3, (1234,), generator=g)
    mu, cv = lda_fit_chunked(lambda X: X, Z, yz, [0, 1, 2], chunk=100)
    R = torch.cat([Z[yz == k] - Z[yz == k].mean(0) for k in range(3)]); cvd = R.T @ R / (1234 - 3)
    check('N9 chunked LDA statistics equal direct', all(torch.allclose(mu[k], Z[yz == k].mean(0)) for k in range(3)) and torch.allclose(cv, cvd))
    # N1 measurement flags do not change training (toy, two seeds); N4 built-in sanity values
    allf = {'floors': True, 'stats': True, 'stats2': True, 'drift': True, 'interf': True, 'la': True, 'sleep': True, 'readouts': True}
    for sd in (0, 3):
        r0, r1 = run({}, sd), run(allf, sd)
        check(f'N1 seed {sd}: all wave-2 measurements leave acc/old/last unchanged', all(r0[k] == r1[k] for k in ('acc', 'old_acc', 'last_acc')), (r0['acc'], r1['acc']))
    d = r1['drift']
    check('N4 drift bookkeeping (definitional): stale E = 0, oracle_now E = 1, oracle_now LDA = lda_full_fresh_oracle',
          abs(d['stale']['E_mean']) < 1e-12 and abs(d['oracle_now']['E_mean'] - 1) < 1e-12 and d['oracle_now']['lda_fullcov'] == r1['stats']['lda_full_fresh_oracle'], d['oracle_now'])
    print('  info N4b: stale-mean LDA with current covariance', d['stale']['lda_fullcov'], 'vs stored-cov', r1['stats']['lda_stored_stale'])
    fl = r1['floors']
    check('N10 floors: lda_now equals stats lda_full_fresh_oracle (same statistics, chunked)', abs(fl['lda_now'] - r1['stats']['lda_full_fresh_oracle']) < 1e-9, (fl['lda_now'], r1['stats']['lda_full_fresh_oracle']))
    _, _, Xte, _, mi = setup({'data': 'toy'}, 3); m0 = copy.deepcopy(mi)
    with torch.no_grad(): mi[0].weight.mul_(2.0)
    set_flat(mi, flat(m0).detach().clone())
    with torch.no_grad(): check('N10b set_flat from the init snapshot reproduces the initial model exactly', torch.equal(mi(Xte), m0(Xte)))
    it = r1['interf']
    check('N6 interference: 2 new x (2,4,6,8) old pairs = 40 rows, each with first-batch and end confusion', len(it) == 40 and all('conf_end' in e and 'sim_feat_b1' in e for e in it), len(it))
    # N8 sleep with 0 steps equals the model's own accuracy
    r2 = run({'sleep': True, 'sleep_steps': 0}, 0)
    check('N8 sleep with 0 steps = model accuracy', r2['sleep']['head'] == r2['acc'] and r2['sleep']['full'] == r2['acc'])
    # N6b cyclic stream bookkeeping
    rc = run({'stream': [0, 1, 2, 3, 4, 0], 'cyc': True}, 0); rp_ = run({'stream': [0, 1, 2, 3, 4, 0]}, 0)
    check('N6b one return -> one rets entry equal to ret; cyc flag changes nothing', len(rc['rets']) == 1 and rc['rets'][0]['ttt'] == rc['ret']['ttt'] and rc['acc'] == rp_['acc'] and len(rc['cyc']) == 6)
    rcc = run({'stream': [0, 1, 0, 1, 0], 'cyc': True}, 0)
    check('N6c stream 0,1,0,1,0 -> 3 returns with occurrences 2,2,3', [(e['task'], e['occurrence']) for e in rcc['rets']] == [(0, 2), (1, 2), (0, 3)], [(e['task'], e['occurrence']) for e in rcc['rets']])
    # N1b ACE start-insert with interf: old classes exclude the newly inserted ones
    ra = run({'loss': 'ace', 'buf': 'start', 'interf': True}, 1)
    check('N1b interf under buf=start: no pair has old == a new class of its task', all(e['old'] < 2 * [0, 1, 2, 3, 4][e['ti']] for e in ra['interf']) and len(ra['interf']) == 40)
    # N11 LOO really excludes class k: shared planted map + a class-specific offset for class 2 only
    lab = np.repeat(np.arange(4), 100); Ft = torch.randn(400, 6, generator=g, dtype=torch.float64) + torch.from_numpy(lab).double()[:, None] * torch.randn(6, generator=g, dtype=torch.float64)
    A0 = torch.eye(6, dtype=torch.float64) + 0.3 * torch.randn(6, 6, generator=g, dtype=torch.float64); Fn = Ft @ A0 + 1.0
    off = torch.zeros(6, dtype=torch.float64); off[0] = 5.0; Fn[torch.from_numpy(lab == 2)] += off
    mt = {k: Ft[torch.from_numpy(lab == k)].mean(0) for k in range(4)}; mn = {k: Fn[torch.from_numpy(lab == k)].mean(0) for k in range(4)}
    Fn0 = Ft @ A0 + 1.0; mn0 = Fn0[torch.from_numpy(lab == 1)].mean(0)       # shared map only (no offset anywhere)
    e_shared = drift_explained(loo_affine_pred(Ft, Fn0, lab, 1, 1e-9, mt[1]), mt[1], mn0)
    p2 = loo_affine_pred(Ft, Fn, lab, 2, 1e-9, mt[2]); miss = float((p2 - mn[2]).norm())
    Afull, cfull = affine_fit(Ft, Fn, 1e-9); miss_full = float((mt[2] @ Afull + cfull - mn[2]).norm())
    check('N11 LOO: a shared map predicts a held-out class; a class-specific offset is NOT predicted (excluded from the fit), '
          'while the non-LOO fit absorbs part of it', e_shared > 0.999 and abs(miss - 5.0) < 0.05 and miss_full < 4.5, (e_shared, miss, miss_full))
    # N1c ACE + start-insert + curve: measurement flags change nothing, incl. the training curve
    r0 = run({'loss': 'ace', 'buf': 'start', 'curve_every': 10}, 4); r1 = run({'loss': 'ace', 'buf': 'start', 'curve_every': 10, **allf}, 4)
    check('N1c ACE start-insert: acc and the full training curve unchanged by all flags', r0['acc'] == r1['acc'] and r0['curve'] == r1['curve'])
    try: run({'stream': [0, 1, 0], 'drift': True}, 0); check('N12 stats on a revisiting stream is refused', False)
    except AssertionError as e: check('N12 stats on a revisiting stream is refused', 'revisit' in str(e))
    # N13 ret_bc is measurement-only, and its raw seen-class curve at return start equals an independent evaluation
    a = run({'stream': [0, 1, 2, 3, 4, 0], 'snap': 'rows'}, 1); b = run({'stream': [0, 1, 2, 3, 4, 0], 'snap': 'rows', 'ret_bc': True}, 1)
    check('N13 ret_bc changes no training output (acc, ret curve, ttt)', a['acc'] == b['acc'] and a['ret']['curve'] == b['ret']['curve'] and a['ret']['ttt'] == b['ret']['ttt'])
    bc = b['ret']['bc']; L = len(b['ret']['curve'])
    check('N13b ret_bc curves aligned with the raw return curve; ref_bc in [0,1]; end_bc = last point',
          len(bc['seen_bc_curve']) == L and len(bc['seen_raw_curve']) == L and 0 <= bc['ref_bc'] <= 1, (L, len(bc['seen_bc_curve'])))
    # N14 loss-weighted prior: with w = 1 it equals the raw-count prior exactly; with w = 4 replay classes carry 4x mass
    a = run({'la': True}, 0); b = run({'la': True, 'w': 4.0}, 0)
    check('N14 lastw == last at w = 1 (all taus)', all(a['la'][f'lastw_tau{t}'] == a['la'][f'last_tau{t}'] for t in (0.5, 1.0, 2.0)) and a['la']['pi_last_w'] == a['la']['pi_last'])
    pw, pr_ = np.array(b['la']['pi_last_w']), np.array(b['la']['pi_last'])
    check('N14b at w = 4 the new/old per-class prior ratio drops by 4x vs raw counts', abs((pw[8] / pw[:8].mean()) / (pr_[8] / pr_[:8].mean()) - 0.25) < 0.02, (pw[8] / pw[:8].mean(), pr_[8] / pr_[:8].mean()))
    import tempfile; tf = os.path.join(tempfile.mkdtemp(), 'ck_{seed}.pt'); r = run({'save_model': tf}, 2)
    ck = torch.load(tf.format(seed=2)); _, _, Xte, yte_, mm = setup(ck['cfg'], 2); mm.load_state_dict(ck['state'])
    with torch.no_grad(): accl = class_acc(mm(Xte).argmax(1), yte_, range(10))
    check('N15 saved checkpoint reproduces the run accuracy exactly; run unchanged by saving', accl == r['acc'] and r['acc'] == run({}, 2)['acc'], (accl, r['acc']))
    # N16 reserve = 0 is ER exactly; reserve = 2 stores 2/class unreplayed, disjoint from the replay buffer
    a = run({}, 1); b = run({'reserve': 0}, 1); r2 = run({'reserve': 2}, 1)
    check('N16 reserve=0 equals ER exactly; reserve=2 gives 3+2 per class', a['acc'] == b['acc'] and r2['resv']['n_buf'] == 30 and r2['resv']['n_resv'] == 20, (r2['resv']['n_buf'], r2['resv']['n_resv']))
    r1 = run({'reserve': 1}, 1)
    check('N16b reserve=1: reserve-only LDA refused (no scatter), reserve means + pooled memory covariance defined', r1['resv']['lda_resv'] is None and r1['resv']['lda_resvmean_bothcov'] is not None)
    rb = r2['resv']; check('N16c lda_items(buf) unchanged by the refactor at reserve=0', b['resv']['lda_buf'] is not None)
    print('SELFTEST2', 'ALL PASS' if ok else 'FAILURES', flush=True); return ok

def joint_mnist(seed, steps=1500):
    Xtr, ytr, Xte, yte, m = setup({'data': 'mnist'}, seed); rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(steps):
        b = rng.choice(len(ytr), 64, replace=False); opt.zero_grad(); torch.nn.functional.cross_entropy(m(Xtr[b]), ytr[b]).backward(); opt.step()
    with torch.no_grad(): return class_acc(m(Xte).argmax(1), yte, range(10))

if __name__ == '__main__':
    if sys.argv[1] == 'selftest': sys.exit(0 if (selftest() & selftest2()) else 1)
    if sys.argv[1] == 'selftest2': sys.exit(0 if selftest2() else 1)
    cfg = json.loads(sys.argv[2]); seeds = [int(x) for x in sys.argv[3].split(',')]
    for s in seeds:
        t0 = time.time(); r = run(cfg, s); r['sec'] = round(time.time() - t0)
        print(json.dumps({'cfg': cfg, 'seed': s, **r}), flush=True)
