"""CIFAR-10 CPU harness (2026-10-29): E0 gap check and E1 Split CIFAR-10 two-path test. See plan.md (pre-declared).

Usage:
  python cifar.py selftest
  python cifar.py e1 <method> <seeds> [outdir]     method in er, ace, frozen, joint1, rc, rp      (LR env sets the step size)
  python cifar.py e0 <seeds> [outdir]
Each run writes <outdir>/<method>_<seed>.npz (raw per-item outputs for analysis) and prints one JSON summary line.
"""
import os, sys, json, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('CIFAR_DIR', os.path.join(HERE, 'data'))
K, NTASK, B, RB, W = int(os.environ.get('CL_K', '10')), int(os.environ.get('CL_NTASK', '5')), 10, 10, 32  # CIFAR-100 (T-S5): CL_K=100 CL_NTASK=20
MEM = int(os.environ.get('MEM', '500')); CPT = K // NTASK; assert CPT * NTASK == K
RESV = int(os.environ.get('RESV', '5'))  # reserve items per class (E1: 5; E2: 20)
LR = float(os.environ['LR']) if 'LR' in os.environ else None  # required for learner methods (tuned on seed 100)
SHRINK, FLOOR, EPS_UNSEEN = 0.1, 1e-6, 1e-3
torch.set_num_threads(int(os.environ.get('NT', '1')))

# ---------------------------------------------------------------- data and stream
def to_t(X): return torch.from_numpy(((X.astype(np.float32) / 255 - 0.5) / 0.25).transpose(0, 3, 1, 2).copy())
def load():
    tr = np.load(os.path.join(DATA, 'train.npz')); te = np.load(os.path.join(DATA, 'test.npz'))
    return to_t(tr['X']), torch.from_numpy(tr['y']), to_t(te['X']), torch.from_numpy(te['y'])

def make_stream(ytr, seed):
    """Per-seed class permutation into NTASK tasks of CPT classes (default 5 of 2), shuffled within task; the first RESV arrivals of each class are reserve."""
    rng = np.random.default_rng(seed + 4242); y = ytr.numpy()
    perm = rng.permutation(K); tasks = [sorted(perm[CPT * t:CPT * t + CPT].tolist()) for t in range(NTASK)]
    order, tid = [], []
    for t, cl in enumerate(tasks):
        idx = rng.permutation(np.where(np.isin(y, cl))[0]); order.append(idx); tid.append(np.full(len(idx), t))
    order, tid = np.concatenate(order), np.concatenate(tid)
    resv = np.zeros(len(order), bool); cnt = np.zeros(K, int)
    for p, i in enumerate(order):
        if cnt[y[i]] < RESV: resv[p] = True; cnt[y[i]] += 1
    return tasks, order, tid, resv

def batches(order, tid, resv):
    """List of (task, train-index array) learning batches of size B; batches never straddle tasks; reserve excluded."""
    out = []
    for t in range(NTASK):
        p = np.where((tid == t) & ~resv)[0]
        out += [(t, order[p[i:i + B]]) for i in range(0, len(p), B)]
    return out

# ---------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(s, w=W):
        super().__init__(); g = lambda c: nn.GroupNorm(4, c)
        s.body = nn.Sequential(nn.Conv2d(3, w, 3, padding=1), g(w), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(w, 2 * w, 3, padding=1), g(2 * w), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(2 * w, 4 * w, 3, padding=1), g(4 * w), nn.ReLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten())
        s.head = nn.Linear(16 * w, K)
    def feats(s, x): return s.body(x)
    def forward(s, x): return s.head(s.body(x))
def make_net(seed, w=W): torch.manual_seed(seed); return Net(w)

def chunked(f, X, chunk=1000):
    with torch.no_grad(): return torch.cat([f(X[i:i + chunk]) for i in range(0, len(X), chunk)])

def rp_map(d_in, d_out, seed):
    """Random ReLU expansion of flattened pixels (same construction as marathon mar2.rp_map)."""
    g = torch.Generator().manual_seed(seed); R = torch.randn(d_in, d_out, generator=g) / math.sqrt(d_in); bvec = torch.randn(d_out, generator=g) * 0.1
    return lambda X: torch.relu(X.reshape(len(X), -1) @ R + bvec)

# ---------------------------------------------------------------- streaming LDA
class StreamLDA:
    """Exact streaming class sums, counts and raw second moment (float64); equal-prior LDA over seen classes."""
    def __init__(s, d):
        s.d = d; s.S1 = torch.zeros(K, d, dtype=torch.float64); s.n = torch.zeros(K, dtype=torch.float64)
        s.XX = torch.zeros(d, d, dtype=torch.float64); s._c = None
    def update(s, F, y):
        F = F.double(); s.XX += F.T @ F; s.S1.index_add_(0, y, F); s.n += torch.bincount(y, minlength=K).double(); s._c = None
    def stats(s):
        sn = s.n > 0; mu = s.S1[sn] / s.n[sn, None]; N = s.n.sum(); C = int(sn.sum())
        Wsc = s.XX - (mu.T * s.n[sn]) @ mu
        return sn, mu, Wsc / max(float(N) - C, 1.0)
    def _prep(s):
        if s._c is None:
            sn, mu, cov = s.stats()
            S = (1 - SHRINK) * cov + (SHRINK * torch.trace(cov) / s.d + FLOOR) * torch.eye(s.d, dtype=torch.float64)
            L = torch.linalg.cholesky(S); PM = torch.cholesky_solve(mu.T.contiguous(), L)
            s._c = (sn, PM, -0.5 * (mu * PM.T).sum(1))
        return s._c
    def scores(s, F):
        sn, PM, c = s._prep(); return sn, F.double() @ PM + c
    def logprobs(s, F):
        """(n, K) log-probabilities: softmax over seen classes carrying 1-EPS; unseen share EPS; uniform if nothing seen."""
        if s.n.sum() == 0: return torch.full((len(F), K), -math.log(K), dtype=torch.float64)
        sn, sc = s.scores(F); C = int(sn.sum()); out = torch.empty(len(F), K, dtype=torch.float64)
        lp = torch.log_softmax(sc, 1)
        if C == K: out[:] = lp
        else: out[:, sn] = lp + math.log1p(-EPS_UNSEEN); out[:, ~sn] = math.log(EPS_UNSEEN / (K - C))
        return out

def lda_from(F, y):
    """Batch LDA (same estimator as StreamLDA) on features F and labels y."""
    m = StreamLDA(F.shape[1])
    for i in range(0, len(F), 5000): m.update(F[i:i + 5000], y[i:i + 5000])
    return m

# ---------------------------------------------------------------- evaluation helpers
def seen_of(tasks, t): return sorted(c for tt in tasks[:t + 1] for c in tt)
def acc_seen(scores, y, seen):
    """Accuracy on items whose label is in `seen`, argmax restricted to `seen`; scores (n, K)."""
    seen = torch.tensor(seen); msk = torch.isin(y, seen)
    pred = seen[scores[msk][:, seen].argmax(1)]; return float((pred == y[msk]).float().mean())
def balanced_mean_logit(L, y):
    """Mean over classes present in y of each class's mean logit vector (equals plain mean on a balanced set)."""
    cl = torch.unique(y); return torch.stack([L[y == c].mean(0) for c in cl]).mean(0)

class Reservoir:
    def __init__(s, M, rng): s.M, s.rng, s.idx, s.lab, s.n = M, rng, [], [], 0
    def add(s, idx, lab):
        for i, l in zip(idx.tolist(), lab.tolist()):
            s.n += 1
            if len(s.idx) < s.M: s.idx.append(i); s.lab.append(l)
            else:
                j = int(s.rng.integers(s.n))
                if j < s.M: s.idx[j] = i; s.lab[j] = l
    def sample(s, k):
        j = s.rng.choice(len(s.idx), min(k, len(s.idx)), replace=False)
        return [s.idx[a] for a in j], torch.tensor([s.lab[a] for a in j])

def ace_loss(ls, yb):
    """Incoming-batch CE over only the classes present in the incoming batch (ER-ACE)."""
    present = torch.zeros(K, dtype=torch.bool); present[torch.unique(yb)] = True
    return Fn.cross_entropy(ls.masked_fill(~present, float('-inf')), yb)

# ---------------------------------------------------------------- E1 learners
def run_learner(method, seed, D, flip_step=None, max_steps=None, eval_tasks=True):
    """method in er / ace / frozen. Returns dict with prequential probs and task-end outputs.
    flip_step: labels of that learning batch are altered (causality self-test). max_steps: truncate (self-tests)."""
    Xtr, ytr, Xte, yte = D; tasks, order, tid, resv = make_stream(ytr, seed); bl = batches(order, tid, resv)
    net = make_net(seed * 1000 + 1); opt = torch.optim.SGD(net.parameters(), lr=LR)
    rng = np.random.default_rng(seed * 1000 + 3); buf = Reservoir(MEM, rng)
    ridx = order[resv]; ry = ytr[ridx]
    freeze_at = sum(1 for t, _ in bl if t == 0) // 2 if method == 'frozen' else None
    lda = StreamLDA(16 * W) if method == 'frozen' else None
    assert LR is not None, 'set LR'
    preq, preq_lp, ylearn, ends, nfb, t0 = [], [], [], [], 0, time.time()
    nsteps = len(bl) if max_steps is None else max_steps
    for step in range(nsteps):
        t, bi = bl[step]; xb = Xtr[bi]; yb = ytr[bi].clone()
        if step == flip_step: yb = (yb + 1) % K
        if freeze_at is not None and step >= freeze_at:
            with torch.no_grad(): F = net.feats(xb)
            lp = lda.logprobs(F); preq.append(lp.exp().float()); preq_lp.append(lp[torch.arange(len(yb)), yb]); lda.update(F, yb)
        else:
            if buf.idx: rj, ryb = buf.sample(RB); x = torch.cat([xb, Xtr[rj]])
            else: ryb = None; x = xb
            out = net(x); ls = out[:len(bi)]; nfb += len(x)
            lp = torch.log_softmax(ls.detach().double(), 1); preq.append(lp.exp().float()); preq_lp.append(lp[torch.arange(len(yb)), yb])
            lin = ace_loss(ls, yb) if method == 'ace' else Fn.cross_entropy(ls, yb)
            loss = lin + (Fn.cross_entropy(out[len(bi):], ryb) if ryb is not None else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        buf.add(torch.as_tensor(bi), yb); ylearn.append(yb)
        last_of_task = step + 1 == len(bl) or bl[step + 1][0] != t
        if eval_tasks and last_of_task: ends.append(eval_learner(net, lda, buf, Xtr, Xte, yte, Xtr[ridx], ry, tasks, t))
    return {'tasks': tasks, 'preq': torch.cat(preq).numpy(), 'preq_lp': torch.cat(preq_lp).numpy(), 'ylearn': torch.cat(ylearn).numpy(), 'ends': ends,
            'resv_y': ry.numpy(), 'fwdbwd_imgs': nfb, 'secs': time.time() - t0, 'freeze_at': freeze_at}

def eval_learner(net, lda, buf, Xtr, Xte, yte, Xr, ry, tasks, t):
    seen = seen_of(tasks, t)
    with torch.no_grad():
        Fte = chunked(net.feats, Xte); Lte = net.head(Fte)
        Xb = Xtr[buf.idx]; yb = torch.tensor(buf.lab); Fb = chunked(net.feats, Xb); Lb = net.head(Fb)
        Fr = net.feats(Xr); Lr = net.head(Fr)
    e = {'t': t, 'test_logits': Lte.numpy(), 'resv_logits': Lr.numpy(), 'b_buf': balanced_mean_logit(Lb, yb).numpy(),
         'buf_counts': torch.bincount(yb, minlength=K).numpy()}
    bsel = torch.isin(yb, torch.tensor(seen))
    e['acc_buflda'] = acc_seen(lda_from(Fb[bsel], yb[bsel]).logprobs(Fte).float(), yte, seen)
    if lda is not None:  # frozen expert: LDA probabilities on test and reserve
        e['test_lda_lp'] = lda.logprobs(Fte).float().numpy(); e['resv_lda_lp'] = lda.logprobs(Fr).float().numpy()
    return e

# ---------------------------------------------------------------- E1 fixed-feature experts
def expert_fmap(kind, seed):
    if kind == 'rc': net0 = make_net(seed * 1000 + 2); return net0.feats, 16 * W
    if kind == 'rp': return rp_map(3 * 32 * 32, 512, seed * 1000 + 4), 512
    if kind == 'rcw': net0 = make_net(seed * 1000 + 13, 64); return net0.feats, 16 * 64  # = E0 init_wide encoder
    raise ValueError(kind)

def run_expert(kind, seed, D, flip_step=None, max_steps=None):
    Xtr, ytr, Xte, yte = D; tasks, order, tid, resv = make_stream(ytr, seed); bl = batches(order, tid, resv)
    fmap, d = expert_fmap(kind, seed); t0 = time.time()
    lda = StreamLDA(d); ridx = order[resv]; ry = ytr[ridx]
    Fte = chunked(fmap, Xte); Fr = chunked(fmap, Xtr[ridx])
    nsteps = len(bl) if max_steps is None else max_steps
    allidx = np.concatenate([bi for _, bi in bl[:nsteps]]); Fall = chunked(fmap, Xtr[allidx])
    preq, preq_lp, ylearn, ends, p = [], [], [], [], 0
    for step in range(nsteps):
        t, bi = bl[step]; F = Fall[p:p + len(bi)]; p += len(bi); yb = ytr[bi].clone()
        if step == flip_step: yb = (yb + 1) % K
        lp = lda.logprobs(F); preq.append(lp.exp().float()); preq_lp.append(lp[torch.arange(len(yb)), yb]); lda.update(F, yb); ylearn.append(yb)
        if max_steps is None and (step + 1 == len(bl) or bl[step + 1][0] != t):
            ends.append({'t': t, 'test_lda_lp': lda.logprobs(Fte).float().numpy(), 'resv_lda_lp': lda.logprobs(Fr).float().numpy()})
    return {'tasks': tasks, 'preq': torch.cat(preq).numpy(), 'preq_lp': torch.cat(preq_lp).numpy(), 'ylearn': torch.cat(ylearn).numpy(), 'ends': ends,
            'resv_y': ry.numpy(), 'fwdbwd_imgs': 0, 'secs': time.time() - t0, 'stat_dim': d}

def run_joint1(seed, D):
    """IID single pass over the same learning items (no reserve), batch 2B, no replay: offline-shuffled reference."""
    Xtr, ytr, Xte, yte = D; tasks, order, tid, resv = make_stream(ytr, seed)
    idx = np.random.default_rng(seed * 1000 + 5).permutation(order[~resv]); net = make_net(seed * 1000 + 1)
    assert LR is not None, 'set LR'
    opt = torch.optim.SGD(net.parameters(), lr=LR); t0 = time.time(); preq, preq_lp, ylearn = [], [], []
    for i in range(0, len(idx), 2 * B):
        bi = idx[i:i + 2 * B]; out = net(Xtr[bi]); lp = torch.log_softmax(out.detach().double(), 1); yb = ytr[bi]
        preq.append(lp.exp().float()); preq_lp.append(lp[torch.arange(len(yb)), yb]); ylearn.append(yb)
        loss = Fn.cross_entropy(out, ytr[bi]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): Lte = chunked(net, Xte)
    return {'tasks': tasks, 'preq': torch.cat(preq).numpy(), 'preq_lp': torch.cat(preq_lp).numpy(), 'ylearn': torch.cat(ylearn).numpy(),
            'ends': [{'t': NTASK - 1, 'test_logits': Lte.numpy()}], 'resv_y': ytr[order[resv]].numpy(), 'fwdbwd_imgs': len(idx), 'secs': time.time() - t0}

def summarize(method, seed, r, yte):
    s = {'method': method, 'seed': seed, 'lr': LR, 'tasks': r['tasks'], 'secs': round(r['secs'], 1), 'fwdbwd_imgs': r['fwdbwd_imgs']}
    P = r['preq']; yl = r['ylearn']
    s['preq_nll'] = float(-r['preq_lp'].mean()); s['preq_lp_min'] = float(r['preq_lp'].min()); s['preq_acc'] = float((P.argmax(1) == yl).mean())
    acc = {}
    for e in r['ends']:
        seen = seen_of(r['tasks'], e['t']); row = {}
        if 'test_logits' in e:
            L = torch.from_numpy(e['test_logits']); row['raw'] = acc_seen(L, yte, seen)
            if 'b_buf' in e:
                row['bc_buf'] = acc_seen(L - torch.from_numpy(e['b_buf']), yte, seen)
                rl, ry = torch.from_numpy(e['resv_logits']), torch.from_numpy(r['resv_y']); m = torch.isin(ry, torch.tensor(seen))
                row['bc_resv'] = acc_seen(L - balanced_mean_logit(rl[m], ry[m]), yte, seen); row['buflda'] = e['acc_buflda']
        if 'test_lda_lp' in e: row['lda'] = acc_seen(torch.from_numpy(e['test_lda_lp']), yte, seen)
        acc[e['t']] = row
    s['acc_by_task_end'] = acc; return s

def save(path, r):
    flat = {'preq': r['preq'], 'preq_lp': r['preq_lp'], 'ylearn': r['ylearn'], 'resv_y': r['resv_y'], 'tasks': np.array(r['tasks'])}
    for e in r['ends']:
        for k, v in e.items():
            if k != 't': flat[f"e{e['t']}_{k}"] = np.asarray(v)
    np.savez_compressed(path, **flat)

# ---------------------------------------------------------------- E0 gap check
def train_joint(seed, Xtr, ytr, epochs=10, w=W, bs=64):
    net = make_net(seed * 1000 + 11, w); opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    steps = epochs * math.ceil(len(Xtr) / bs); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed * 1000 + 12)
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), bs):
            bi = perm[i:i + bs]; x = Xtr[bi]
            flip = torch.from_numpy(rng.random(len(bi)) < 0.5); x = torch.where(flip[:, None, None, None], x.flip(3), x)
            xp = Fn.pad(x, (4, 4, 4, 4), value=-2.0)  # pad with black (pixel 0 maps to -2), as torchvision pads before normalising
            off = rng.integers(0, 9, (len(bi), 2)); x = torch.stack([xp[j, :, off[j, 1]:off[j, 1] + 32, off[j, 0]:off[j, 0] + 32] for j in range(len(bi))])
            loss = Fn.cross_entropy(net(x), ytr[bi]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net

def lda_acc_map(fmap, Xtr, ytr, Xte, yte, ks, ndraw, rng):
    Ftr = chunked(fmap, Xtr); Fte = chunked(fmap, Xte); res = {}; yn = ytr.numpy()
    for k in ks:
        a = []
        for _ in range(ndraw):
            idx = np.concatenate([rng.choice(np.where(yn == c)[0], k, replace=False) for c in range(K)])
            a.append(acc_seen(lda_from(Ftr[idx], ytr[idx]).scores(Fte)[1], yte, list(range(K))))
        res[str(k)] = float(np.mean(a))
    res['all'] = acc_seen(lda_from(Ftr, ytr).scores(Fte)[1], yte, list(range(K))); return res

def run_e0(seed, D):
    Xtr, ytr, Xte, yte = D; t0 = time.time(); net = train_joint(seed, Xtr, ytr); net.eval()
    out = {'seed': seed, 'joint10_head': acc_seen(chunked(net, Xte), yte, list(range(K))), 'train_secs': round(time.time() - t0, 1), 'lda': {}}
    maps = {'joint10': net.feats, 'init': make_net(seed * 1000 + 2).feats, 'init_wide': make_net(seed * 1000 + 13, 64).feats,
            'raw': lambda X: X.reshape(len(X), -1)}
    maps['rp512'] = rp_map(3072, 512, seed * 1000 + 4)  # the same map as the E1 RP-LDA expert
    for w in (2048, 4096): maps[f'rp{w}'] = rp_map(3072, w, seed * 1000 + 4 + w)
    for name, f in maps.items():
        out['lda'][name] = lda_acc_map(f, Xtr, ytr, Xte, yte, [5, 20, 100, 500], 5, np.random.default_rng(seed * 1000 + 14)); print(name, out['lda'][name], file=sys.stderr, flush=True)
    return out

# ---------------------------------------------------------------- mixtures (used by analysis; tested here)
def bayes_mix_stream(la, lb):
    """Sequential Bayes mixture, uniform prior; la, lb are per-item float64 log-probabilities of the true label (prequential).
    Returns per-item mixture log-probability and the log posterior weights of A and B before each item (from items < t only)."""
    ca = np.concatenate([[0.0], np.cumsum(la)[:-1]]); cb = np.concatenate([[0.0], np.cumsum(lb)[:-1]])
    z = np.logaddexp(ca, cb); lwa, lwb = ca - z, cb - z
    return np.logaddexp(lwa + la, lwb + lb), lwa, lwb

# ---------------------------------------------------------------- self-tests
def selftest():
    ok = True
    def chk(c, name, *info):
        nonlocal ok; print('PASS' if c else 'FAIL', name, *info, flush=True); ok &= bool(c)
    D = load(); Xtr, ytr, Xte, yte = D
    chk(Xtr.shape == (50000, 3, 32, 32) and Xte.shape == (10000, 3, 32, 32) and (torch.bincount(ytr) == 5000).all() and (torch.bincount(yte) == 1000).all()
        and abs(float(Xtr.min()) + 2.0) < 1e-6 and abs(float(Xtr.max()) - 2.0) < 1e-6, 'C1 data shapes, per-class counts, pixel range [-2,2]')
    # C2 stream
    tasks, order, tid, resv = make_stream(ytr, 0); bl = batches(order, tid, resv); y = ytr.numpy()
    learn = np.concatenate([bi for _, bi in bl])
    c2 = (sorted(c for t in tasks for c in t) == list(range(K)) and len(order) == 50000 and len(np.unique(order)) == 50000
          and len(learn) == 50000 - K * RESV and len(np.unique(learn)) == len(learn) and not np.isin(learn, order[resv]).any()
          and all(set(y[bi]) <= set(tasks[t]) and len(bi) == B for t, bi in bl)
          and all(np.isin(y[order[tid == t]], tasks[t]).all() for t in range(NTASK))
          and (np.bincount(y[order[resv]], minlength=K) == RESV).all())
    first_ok = all(set(np.where(y[order] == c)[0][:RESV]) == set(np.where(resv & (y[order] == c))[0]) for c in range(K))
    t2, o2, _, r2 = make_stream(ytr, 0); t3, o3, _, _ = make_stream(ytr, 1)
    chk(c2 and first_ok and t2 == tasks and (o2 == order).all() and (r2 == resv).all() and not (o3 == order).all(),
        'C2 stream: partition, task purity, reserve = first 5 per class, deterministic, seeds differ', tasks)
    # C3 streaming LDA = batch, order invariance
    g = torch.Generator().manual_seed(1); Z = torch.randn(700, 6, generator=g, dtype=torch.float64) * 3 + 1; yz = torch.randint(0, 4, (700,), generator=g)
    m1 = lda_from(Z, yz); m2 = StreamLDA(6); perm = torch.randperm(700, generator=g)
    for i in range(0, 700, 13): m2.update(Z[perm[i:i + 13]], yz[perm[i:i + 13]])
    sn, mu, cov = m1.stats(); _, mu2, cov2 = m2.stats()
    mb = torch.stack([Z[yz == c].mean(0) for c in range(4)]); R = torch.cat([Z[yz == c] - mb[c] for c in range(4)]); cb = R.T @ R / (700 - 4)
    chk(torch.allclose(mu, mb, atol=1e-10) and torch.allclose(cov, cb, atol=1e-9) and torch.allclose(mu2, mu, atol=1e-10) and torch.allclose(cov2, cov, atol=1e-9),
        'C3 streaming LDA stats equal batch two-pass stats and are order-invariant', float((cov2 - cb).abs().max()))
    # C4 equals marathon lda_predict (inverse-based)
    sys.path.insert(0, os.path.join(HERE, '..', 'marathon'))
    import importlib.util; sp = importlib.util.spec_from_file_location('mar2', os.path.join(HERE, '..', 'marathon', 'mar2.py')); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
    pm = M.lda_predict({c: mb[c] for c in range(4)}, cb, Z, shrink=SHRINK); pc = m1.scores(Z)[1].argmax(1)
    chk((pm == pc).all(), 'C4 Cholesky LDA argmax equals marathon lda_predict on all 700 items')
    # C5 planted separable, probabilities, unseen eps
    Xa = torch.cat([torch.randn(300, 2, generator=g) + 6, torch.randn(300, 2, generator=g) - 6]).double(); ya = torch.cat([torch.zeros(300), torch.ones(300)]).long()
    i5 = list(range(5)) + list(range(300, 305)); mp = lda_from(Xa[i5], ya[i5])
    lp = mp.logprobs(Xa); acc = float((lp.argmax(1) == ya).float().mean())
    chk(acc > 0.999 and torch.allclose(lp.exp().sum(1), torch.ones(600, dtype=torch.float64)) and torch.allclose(lp[:, 2:].exp().sum(1), torch.full((600,), EPS_UNSEEN, dtype=torch.float64))
        and torch.allclose(StreamLDA(2).logprobs(Xa).exp(), torch.full((600, K), 0.1, dtype=torch.float64)),
        'C5 planted 5-shot LDA ~100%, probs sum to 1, unseen mass = eps, empty = uniform', acc)
    # C6 causality (ER, ACE, LDA expert): flip labels at step t; outputs up to t identical, later differ
    T, ft = 40, 25
    for meth, T, ft in [('er', 40, 25), ('ace', 40, 25), ('frozen', 520, 505)]:  # frozen: flip lands after its freeze step (499)
        a = run_learner(meth, 0, D, max_steps=T, eval_tasks=False); b = run_learner(meth, 0, D, flip_step=ft, max_steps=T, eval_tasks=False)
        same = np.array_equal(a['preq'][:(ft + 1) * B], b['preq'][:(ft + 1) * B]); diff = not np.array_equal(a['preq'][(ft + 1) * B:], b['preq'][(ft + 1) * B:])
        chk(same and diff, f'C6 causality {meth}: prequential preds up to flipped batch identical, later ones change')
    T, ft = 40, 25
    a = run_expert('rc', 0, D, max_steps=T); b = run_expert('rc', 0, D, flip_step=ft, max_steps=T)
    chk(np.array_equal(a['preq'][:(ft + 1) * B], b['preq'][:(ft + 1) * B]) and not np.array_equal(a['preq'][(ft + 1) * B:], b['preq'][(ft + 1) * B:]),
        'C6 causality rc expert')
    # C7 determinism
    a2 = run_expert('rc', 0, D, max_steps=T); e1 = run_learner('er', 0, D, max_steps=T, eval_tasks=False); e2 = run_learner('er', 0, D, max_steps=T, eval_tasks=False)
    chk(np.array_equal(a['preq'], a2['preq']) and np.array_equal(e1['preq'], e2['preq']), 'C7 determinism (rc expert, er learner)')
    # frozen: before freeze identical to ER, check freeze point
    fz = run_learner('frozen', 0, D, max_steps=T, eval_tasks=False)
    chk(fz['freeze_at'] == math.ceil((10000 - 2 * RESV) / B) // 2 and np.array_equal(fz['preq'], e1['preq']), 'C7b frozen learner equals ER before its freeze step (half of task-1 batches)')
    # C8 mixture bound and nulls
    rg = np.random.default_rng(0); la = np.log(rg.uniform(0.01, 1, 3000)); lb = np.minimum(la + np.log(rg.uniform(0.2, 3, 3000)), 0.0)
    lm, lwa, _ = bayes_mix_stream(la, lb); reg = -lm.sum() - min(-la.sum(), -lb.sum())
    # direct loop: w_t = prod_{s<t} pa / (prod pa + prod pb), mixture p_t = w pa + (1-w) pb; computed with mpmath-free exact log recursion
    ref_w = np.empty(3000); A = Bv = 0.0
    for t in range(3000): ref_w[t] = 1 / (1 + math.exp(Bv - A)) if abs(Bv - A) < 700 else float(A > Bv); A += la[t]; Bv += lb[t]
    ref_lm = np.log(ref_w * np.exp(la) + (1 - ref_w) * np.exp(lb))
    lm2, _, _ = bayes_mix_stream(la, la); lm3, lwa3, _ = bayes_mix_stream(la, la - 50.0)  # B far worse -> weight on A -> 1
    chk(reg <= math.log(2) + 1e-9 and np.allclose(np.exp(lwa), ref_w, atol=1e-12) and np.allclose(lm, ref_lm, atol=1e-9) and np.allclose(lm2, la)
        and lwa[0] == math.log(0.5) and np.exp(lwa3[1:]).min() > 1 - 1e-12 and np.allclose(lm3[1:], la[1:]),
        'C8 Bayes mixture = direct loop (weights use items < t), regret <= log 2, identical -> same, dominated B -> A, prior 1/2', reg)
    # leak detector: a mixture whose weight uses item t's own label must fail the direct-loop comparison
    cl = np.cumsum(la); cbl = np.cumsum(lb); leak_w = 1 / (1 + np.exp(np.clip(cbl - cl, -700, 700)))
    chk(not np.allclose(leak_w, ref_w, atol=1e-6), 'C8b direct-loop comparison detects an off-by-one (leaky) weight')
    # C13 stored log-probabilities: float64, finite, consistent with stored probabilities where those do not underflow
    lp_ok = all(r_['preq_lp'].dtype == np.float64 and np.isfinite(r_['preq_lp']).all() for r_ in (a, e1, fz))
    pt = a['preq'][np.arange(len(a['ylearn'])), a['ylearn']].astype(np.float64); big = pt > 1e-30
    chk(lp_ok and np.allclose(np.log(pt[big]), a['preq_lp'][big], atol=1e-5), 'C13 per-item true-label log-probs stored in float64, finite, match probs', float(a['preq_lp'].min()))
    # C9 reservoir uniformity
    cnt = np.zeros(1000)
    for trial in range(2000):
        r = Reservoir(100, np.random.default_rng(trial)); r.add(torch.arange(1000), torch.zeros(1000, dtype=torch.long)); cnt[r.idx] += 1
    p = cnt / 2000; z = (p - 0.1) / math.sqrt(0.1 * 0.9 / 2000)
    chk(np.abs(z).max() < 4.5 and abs(np.corrcoef(np.arange(1000), p)[0, 1]) < 0.1,
        'C9 reservoir inclusion uniform (max |z| over 1000 items, corr with position)', float(np.abs(z).max()), float(np.corrcoef(np.arange(1000), p)[0, 1]))
    # C10 ACE gradient on absent logits is zero
    ls = torch.randn(10, K, requires_grad=True); yb = torch.tensor([2, 3] * 5); ace_loss(ls, yb).backward()
    chk(ls.grad[:, [c for c in range(K) if c not in (2, 3)]].abs().max() == 0 and ls.grad[:, [2, 3]].abs().max() > 0, 'C10 ACE: zero gradient on absent-class logits')
    # C11 biascorr equals marathon definition on balanced buffer
    Lb = torch.randn(40, K, generator=g); yb4 = torch.arange(4).repeat(10)
    yim = torch.tensor([0] * 30 + [1] * 5 + [2] * 5); Lim = torch.randn(40, K, generator=g)
    hand = (Lim[:30].mean(0) + Lim[30:35].mean(0) + Lim[35:].mean(0)) / 3
    chk(torch.allclose(balanced_mean_logit(Lb, yb4), Lb.mean(0)) and torch.allclose(balanced_mean_logit(Lim, yim), hand) and not torch.allclose(hand, Lim.mean(0)),
        'C11 balanced mean logit = marathon plain mean on balanced buffer; = mean of class means on imbalanced buffer')
    # C12 accuracy metric
    sc = torch.zeros(6, K); sc[0, 1] = 1; sc[1, 9] = 5; sc[1, 0] = 1; sc[2, 2] = 1; sc[3, 3] = 1
    yy = torch.tensor([1, 0, 2, 3, 7, 8])  # items 4,5 unseen class -> excluded; item1: class 9 unseen, so argmax over seen gives 0 (correct)
    chk(abs(acc_seen(sc, yy, [0, 1, 2, 3]) - 1.0) < 1e-12 and abs(acc_seen(sc, torch.tensor([0, 0, 2, 3, 7, 8]), [0, 1, 2, 3]) - 0.75) < 1e-12, 'C12 seen-restricted accuracy on hand example')
    print('SELFTEST', 'ALL PASS' if ok else 'FAILURES'); return ok

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'selftest': sys.exit(0 if selftest() else 1)
    D = load()
    if cmd == 'e1':
        meth = sys.argv[2]; outdir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, 'results')
        for s in [int(x) for x in sys.argv[3].split(',')]:
            r = run_learner(meth, s, D) if meth in ('er', 'ace', 'frozen') else run_joint1(s, D) if meth == 'joint1' else run_expert(meth, s, D)
            tag = f"{meth}_lr{LR}" if meth in ('er', 'ace', 'frozen', 'joint1') else meth
            save(os.path.join(outdir, f'{tag}_{s}.npz'), r); print(json.dumps(summarize(meth, s, r, D[3])), flush=True)
    elif cmd == 'e0':
        for s in [int(x) for x in sys.argv[2].split(',')]: print(json.dumps(run_e0(s, D)), flush=True)
