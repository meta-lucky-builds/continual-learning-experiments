"""Sharp-test marathon harness (plan: notes/marathon/plan.md).

One training loop, shared by every test, for two datasets:
  toy   : the v1.3 toy (functions imported from notes/v13-checks/toy_cl3.py), SEP/NMEM/REP from cfg
  mnist : Split-MNIST, 5 tasks of 2 digits, MLP 784-256-256-10
The loop is a byte-for-byte generalisation of toy_cl3's ER loop (manual Adam in float64 on the flat
parameter vector), so with every switch off it must reproduce toy_cl3's per-seed ER numbers (selftest S1).

Usage:  python mar.py selftest
        python mar.py run '<json cfg>' seeds   -> one JSON line per seed
"""
import sys, os, json, time, math, importlib.util, struct
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
    for ti, task in enumerate(stream):
        cls = [2 * task, 2 * task + 1]; idx = np.where(np.isin(yn, cls))[0]
        is_return = task in stream[:ti]
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
            if len(buf) > 0:
                rb = rng.choice(buf, c['rep'], replace=len(buf) < c['rep']); X = torch.cat([X, Xtr[rb]]); y = torch.cat([y, ytr[rb]])
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
        # end of task
        if c['buf'] == 'end':
            for k in cls:
                if have.get(k, 0) < c['nmem']:
                    ci = np.where(yn == k)[0]; buf += rng.choice(ci, c['nmem'], replace=False).tolist(); have[k] = c['nmem']
        with torch.no_grad(): pr = m(Xte).argmax(1)
        out.setdefault('ref', {})
        if not is_return: out['ref'][task] = class_acc(pr, yte, cls)
        if task in (0, 2) and not is_return: snaps[task] = flat(m).detach().clone()
        if c.get('stats') and not is_return:      # F7: class statistics written once, at the class's own task end
            Fall = feats(m, Xtr[idx]).double(); yall = ytr[idx]
            for k in cls:
                stats.setdefault('mean', {})[k] = Fall[yall == k].mean(0)
                bk = [i for i in buf if yn[i] == k]; stats.setdefault('bufmean_then', {})[k] = feats(m, Xtr[bk]).double().mean(0)
            stats.setdefault('covs', []).append((pooled_cov(Fall, yall), len(idx)))
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
    with torch.no_grad(): pr = m(Xte).argmax(1)
    seen = sorted(set(k for tk in stream for k in (2 * tk, 2 * tk + 1)))
    out['acc'] = class_acc(pr, yte, seen)
    last = [2 * stream[-1], 2 * stream[-1] + 1]; old = [k for k in seen if k not in last]
    out['old_acc'] = class_acc(pr, yte, old) if old else None; out['last_acc'] = class_acc(pr, yte, last)
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

def joint_mnist(seed, steps=1500):
    Xtr, ytr, Xte, yte, m = setup({'data': 'mnist'}, seed); rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(steps):
        b = rng.choice(len(ytr), 64, replace=False); opt.zero_grad(); torch.nn.functional.cross_entropy(m(Xtr[b]), ytr[b]).backward(); opt.step()
    with torch.no_grad(): return class_acc(m(Xte).argmax(1), yte, range(10))

if __name__ == '__main__':
    if sys.argv[1] == 'selftest': sys.exit(0 if selftest() else 1)
    cfg = json.loads(sys.argv[2]); seeds = [int(x) for x in sys.argv[3].split(',')]
    for s in seeds:
        t0 = time.time(); r = run(cfg, s); r['sec'] = round(time.time() - t0)
        print(json.dumps({'cfg': cfg, 'seed': s, **r}), flush=True)
