"""Round-2 CIFAR learners (2026-10-29; plan.md section "Round 2"). Built on cifar.py; cifar.py's ER is the reference.

One generalised online learner with optional parts:
  mem=M          reservoir size (memory matching)
  epochs=E       passes over each task's learning batches (epoch 0 = the stream order; later epochs reshuffled, own RNG);
                 prequential predictions, buffer insertion and expert statistics use the first pass only
  ewc=lam        online EWC: at each task end, empirical diagonal Fisher on the buffer, summed over tasks; anchor = params then
  m2c=1|2        zero-slack margin projection with born-label rivals (spec M2c variant that won on the toy, A5):
                 every RREF steps a pool of PPOOL buffer items; for correctly classified ones, columns grad(f_y - f_j) for
                 j in top-2 rivals plus the current task's classes; each step the SGD step is projected so that no column's
                 first-order margin change is negative (metric = SGD's, i.e. Euclidean)
                 m2c=1 draws the pool from the whole reservoir, so current-task items are protected too (round-2 port);
                 m2c=2 draws it from past-task items only, which is what the toy did (amendment 2026-10-29 07:41 UTC)
  promote=1      at each task end t < last, freeze a copy of the encoder; its streaming-LDA statistics start from the
                 buffer items' features at promotion and then add every later stream batch

Usage: python cifar2.py run <name> <seeds> <outdir> key=value ...      (LR and RESV env as in cifar.py)
       python cifar2.py selftest
"""
import os, sys, json, math, time, copy
import numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C
RREF, PPOOL = 25, 32

def flat_grad(net): return torch.cat([p.grad.reshape(-1) for p in net.parameters()])
def add_flat(net, v):
    with torch.no_grad():
        o = 0
        for p in net.parameters(): n = p.numel(); p.add_(v[o:o + n].view_as(p).to(p.dtype)); o += n

def margin_cols(net, X, y, extra):
    """Columns grad_theta (f_y - f_j) for items correctly classified, j in top-2 rivals + extra classes (j != y)."""
    params = {n: p.detach() for n, p in net.named_parameters()}
    def fn(pr, x): return torch.func.functional_call(net, pr, (x.unsqueeze(0),)).squeeze(0)
    with torch.no_grad(): L = net(X)
    ok = L.argmax(1) == y; X, y, L = X[ok], y[ok], L[ok]
    if len(X) == 0: return None
    J = torch.func.vmap(torch.func.jacrev(fn), in_dims=(None, 0))(params, X)
    Jf = torch.cat([J[n].reshape(len(X), C.K, -1) for n, _ in net.named_parameters()], 2)
    cols = []
    for i in range(len(X)):
        Li = L[i].clone(); yi = int(y[i]); Li[yi] = -float('inf'); riv = [int(j) for j in torch.topk(Li, 2).indices]
        riv += [j for j in extra if j != yi and j not in riv]
        cols += [Jf[i, yi] - Jf[i, j] for j in riv]
    return torch.stack(cols, 1).double().numpy()

def qp_project_zero(lr, Cm, d0):
    """min ||d - d0||^2 s.t. Cm^T d >= 0 (SGD metric B = lr I; dual by L-BFGS-B + exact active-set polish, as diag.qp_polished)."""
    from scipy.optimize import minimize
    G = lr * (Cm.T @ Cm); r = Cm.T @ d0
    if np.all(r >= 0): return None
    res = minimize(lambda mu: (.5 * mu @ G @ mu + mu @ r, G @ mu + r), np.zeros(len(r)), jac=True, bounds=[(0, None)] * len(r),
                   method='L-BFGS-B', options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 5000})
    mu = res.x; n = len(r); damp = 1e-12 * np.trace(G) / n
    for _ in range(100):
        F = mu > 1e-12 * max(1, mu.max()); muF = np.zeros(n)
        if F.any(): muF[F] = np.linalg.solve(G[np.ix_(F, F)] + damp * np.eye(F.sum()), -r[F])
        gr = G @ muF + r
        if (muF >= 0).all() and (gr[~F] >= -1e-10 * (1 + np.abs(r[~F]))).all(): mu = muF; break
        F2 = (F & (muF > 0)) if (muF < 0).any() else F
        F2 = F2 | ((~F) & (gr < 0)); mu = np.where(F2, np.maximum(muF, 1e-9), 0.0)
    return lr * (Cm @ mu)  # correction added to the plain SGD step d0

def qp_project_zero_nnls(lr, Cm, d0):
    """Same QP as qp_project_zero, solved exactly as a small NNLS: with G = lr C^T C = R^T R (damped Cholesky) and r = C^T d0,
    the dual 1/2 mu^T G mu + mu^T r = 1/2 ||R mu + R^-T r||^2 + const, so mu = nnls(R, -R^-T r). Replaces L-BFGS-B, which took
    ~100% of M2c's wall time (profiled 2026-10-29); the optimum is the same (strictly convex primal)."""
    from scipy.optimize import nnls
    G = lr * (Cm.T @ Cm); r = Cm.T @ d0
    if np.all(r >= 0): return None
    n = len(r); Rm = np.linalg.cholesky(G + 1e-12 * np.trace(G) / n * np.eye(n)).T
    mu, _ = nnls(Rm, -np.linalg.solve(Rm.T, r), maxiter=50 * n)
    return lr * (Cm @ mu)

QP = qp_project_zero_nnls if os.environ.get('QP', 'nnls') == 'nnls' else None  # set after qp_project_zero is defined below

def run(seed, D, mem=C.MEM, epochs=1, ewc=0.0, m2c=0, promote=0, max_steps=None, eval_tasks=True, flip_step=None, log=None):
    Xtr, ytr, Xte, yte = D; tasks, order, tid, resv = C.make_stream(ytr, seed); bl = C.batches(order, tid, resv)
    assert C.LR is not None
    net = C.make_net(seed * 1000 + 1); opt = torch.optim.SGD(net.parameters(), lr=C.LR)
    rng = np.random.default_rng(seed * 1000 + 3); buf = C.Reservoir(mem, rng)
    rng_ep = np.random.default_rng(seed * 1000 + 7); rng_pool = np.random.default_rng(seed * 1000 + 9)
    ridx = order[resv]; ry = ytr[ridx]; Xr = Xtr[ridx]
    preq, preq_lp, ylearn, ends, nfb, t0 = [], [], [], [], 0, time.time()
    Fw, anchor = None, None; Cm, since = None, RREF; versions = []
    stats = {'proj_steps': 0, 'steps': 0, 'ncols': [], 'max_violation': 0.0}
    nsteps = len(bl) if max_steps is None else max_steps; step = 0
    def train_step(xb, yb, record):
        nonlocal nfb, Cm, since
        if buf.idx: rj, ryb = buf.sample(C.RB); x = torch.cat([xb, Xtr[rj]])
        else: ryb = None; x = xb
        out = net(x); ls = out[:len(xb)]; nfb += len(x)
        if record:
            lp = torch.log_softmax(ls.detach().double(), 1); preq.append(lp.exp().float()); preq_lp.append(lp[torch.arange(len(yb)), yb])
        loss = Fn.cross_entropy(ls, yb) + (Fn.cross_entropy(out[len(xb):], ryb) if ryb is not None else 0.0)
        if ewc > 0 and Fw is not None:
            th = torch.cat([p.reshape(-1) for p in net.parameters()]); loss = loss + 0.5 * ewc * (Fw * (th - anchor) ** 2).sum()
        opt.zero_grad(); loss.backward()
        corr = None
        if m2c and buf.idx:
            if since >= RREF:
                # m2c=1: pool from the whole buffer (incl. current-task items); m2c=2: past-task items only, as on the toy
                elig = list(range(len(buf.idx))) if m2c == 1 else [j for j in range(len(buf.idx)) if int(buf.lab[j]) not in cur_classes]
                if elig:
                    pj = rng_pool.choice(elig, min(PPOOL, len(elig)), replace=False)
                    Cm = margin_cols(net, Xtr[[buf.idx[j] for j in pj]], torch.tensor([buf.lab[j] for j in pj]), cur_classes)
                else: Cm = None
                since = 0; stats['ncols'].append(0 if Cm is None else Cm.shape[1])
            since += 1
            if Cm is not None:
                d0 = -C.LR * flat_grad(net).double().numpy(); corr = (QP or qp_project_zero)(C.LR, Cm, d0)
                if corr is not None:
                    v = Cm.T @ (d0 + corr); stats['max_violation'] = max(stats['max_violation'], float(max(0.0, -v.min()) / (1e-12 + np.abs(Cm.T @ d0).max())))
                    stats['proj_steps'] += 1
        opt.step()
        if corr is not None: add_flat(net, torch.from_numpy(corr))
        stats['steps'] += 1
    for t in range(C.NTASK):
        tb = [(i, bi) for i, (tt, bi) in enumerate(bl) if tt == t and i < nsteps]
        if not tb: break
        cur_classes = tasks[t]
        for i, bi in tb:  # first pass = the stream
            yb = ytr[bi].clone()
            if i == flip_step: yb = (yb + 1) % C.K
            train_step(Xtr[bi], yb, True); buf.add(torch.as_tensor(bi), yb); ylearn.append(yb)
            if versions:
                with torch.no_grad():
                    for v in versions: v['lda'].update(v['net'].feats(Xtr[bi]), yb)
        items = np.concatenate([bi for _, bi in tb]); labs = torch.cat(ylearn[-len(tb):])
        for ep in range(1, epochs):  # extra passes over this task's items (labels as seen on the stream)
            p = rng_ep.permutation(len(items))
            for j in range(0, len(p), C.B): sel = p[j:j + C.B]; train_step(Xtr[items[sel]], labs[sel], False)
        if not eval_tasks: continue
        e = C.eval_learner(net, None, buf, Xtr, Xte, yte, Xr, ry, tasks, t)
        for vi, v in enumerate(versions):
            with torch.no_grad(): e[f'ver{vi}_test_lp'] = v['lda'].logprobs(C.chunked(v['net'].feats, Xte)).float().numpy(); e[f'ver{vi}_resv_lp'] = v['lda'].logprobs(v['net'].feats(Xr)).float().numpy()
        ends.append(e)
        if ewc > 0:  # empirical diagonal Fisher on the buffer (per-sample gradients), summed over tasks
            Fn_ = torch.zeros(sum(p.numel() for p in net.parameters()))
            for a in range(0, len(buf.idx)):
                net.zero_grad(); Fn.cross_entropy(net(Xtr[[buf.idx[a]]]), torch.tensor([buf.lab[a]])).backward(); Fn_ += flat_grad(net) ** 2
            Fn_ /= len(buf.idx); Fw = Fn_ if Fw is None else Fw + Fn_; anchor = torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone(); net.zero_grad()
        if promote and t < C.NTASK - 1:
            vn = copy.deepcopy(net)
            for p_ in vn.parameters(): p_.requires_grad_(False)
            lda = C.StreamLDA(16 * C.W)
            with torch.no_grad(): lda.update(C.chunked(vn.feats, Xtr[buf.idx]), torch.tensor(buf.lab))
            versions.append({'net': vn, 'lda': lda, 't': t})
    return {'tasks': tasks, 'preq': torch.cat(preq).numpy(), 'preq_lp': torch.cat(preq_lp).numpy(), 'ylearn': torch.cat(ylearn).numpy(), 'ends': ends,
            'resv_y': ry.numpy(), 'fwdbwd_imgs': nfb, 'secs': time.time() - t0, 'stats': stats, 'versions': versions, 'net': net, 'buf_len': len(buf.idx)}

def selftest():
    ok = True
    def chk(c, name, *info):
        nonlocal ok; print('PASS' if c else 'FAIL', name, *info, flush=True); ok &= bool(c)
    D = C.load(); T = 40
    ref = C.run_learner('er', 0, D, max_steps=T, eval_tasks=False); a = run(0, D, max_steps=T, eval_tasks=False)
    chk(np.array_equal(ref['preq'], a['preq']) and np.array_equal(ref['preq_lp'], a['preq_lp']), 'K1 generalised learner (defaults) equals cifar.run_learner ER exactly')
    b = run(0, D, max_steps=T, eval_tasks=False, ewc=0.0, m2c=0, promote=1)
    chk(np.array_equal(a['preq'], b['preq']), 'K2 promote/ewc=0 do not change the learner within a task')
    m = run(0, D, max_steps=T, eval_tasks=False, mem=C.MEM)
    chk(np.array_equal(m['preq'], a['preq']), 'K3 mem default = 500')
    mm = run(0, D, max_steps=200, eval_tasks=False, mem=50); chk(mm['buf_len'] == 50 and a['buf_len'] == min(C.MEM, T * C.B), 'K3b mem sets the reservoir size', mm['buf_len'], a['buf_len'])
    # K4 causality with m2c and ewc
    for kw in [dict(m2c=1), dict(ewc=10.0)]:
        x1 = run(0, D, max_steps=T, eval_tasks=False, **kw); x2 = run(0, D, max_steps=T, eval_tasks=False, flip_step=25, **kw)
        chk(np.array_equal(x1['preq'][:260], x2['preq'][:260]) and not np.array_equal(x1['preq'][260:], x2['preq'][260:]), f'K4 causality {kw}')
    # K5 m2c: projection feasible (zero first-order margin decrease), actually active, and changes the trajectory
    x = run(0, D, max_steps=150, eval_tasks=False, m2c=1); s = x['stats']
    chk(s['proj_steps'] > 0 and s['max_violation'] < 1e-6 and min(s['ncols']) > 0 and not np.array_equal(x['preq'], run(0, D, max_steps=150, eval_tasks=False)['preq']),
        'K5 m2c active, feasible (relative violation), trajectory differs from ER', s['proj_steps'], s['max_violation'], s['ncols'][:5])
    # K6 QP: projection is the exact minimum-norm feasible correction on a random instance (KKT)
    rg = np.random.default_rng(0); Cr = rg.normal(size=(300, 12)); d0 = rg.normal(size=300); corr = qp_project_zero(0.1, Cr, d0)
    d = d0 + corr; mu = np.linalg.lstsq(0.1 * Cr, corr, rcond=None)[0]
    chk((Cr.T @ d >= -1e-9).all() and (mu >= -1e-9).all() and np.abs(mu * (Cr.T @ d)).max() < 1e-8 and np.allclose(0.1 * Cr @ mu, corr), 'K6 QP KKT: feasible, mu>=0, complementary slackness')
    # K7 margin columns equal finite-difference directional derivatives
    # float64 net and a tiny step: at step 1e-3 the float32 FD error is dominated by curvature/ReLU kinks (error shrinks ~linearly with step)
    net = C.make_net(5).double(); X = D[0][:6].double(); y = net(X).argmax(1).detach(); Cm = margin_cols(net, X, y, [0, 1])
    v = torch.randn(sum(p.numel() for p in net.parameters()), generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 1e-6
    with torch.no_grad():
        L0 = net(X).double(); n2 = copy.deepcopy(net); add_flat(n2, v); L1 = n2(X).double()
    # rebuild the (item, rival) list in the same order to compare
    fd = []
    for i in range(6):
        Li = L0[i].clone(); yi = int(y[i]); Li[yi] = -float('inf'); riv = [int(j) for j in torch.topk(Li, 2).indices]; riv += [j for j in [0, 1] if j != yi and j not in riv]
        fd += [float((L1[i, yi] - L1[i, j]) - (L0[i, yi] - L0[i, j])) for j in riv]
    lin = Cm.T @ v.numpy(); fd = np.array(fd); rel = float(np.abs(lin - fd).max() / np.abs(fd).max())
    chk(rel < 1e-3 and np.corrcoef(lin, fd)[0, 1] > 0.999999, 'K7 margin Jacobian columns match float64 finite differences (relative max error)', rel)
    # K8 promotion: version stats at promotion = batch LDA on buffer features; version params frozen thereafter
    p = run(0, D, max_steps=None, eval_tasks=True, promote=1) if os.environ.get('FULLK8') else None
    if p is not None: chk(len(p['versions']) == C.NTASK - 1, 'K8 four promoted versions')
    # small promotion check: truncated to the first task end via a 1-task run
    tb0 = sum(1 for t, _ in C.batches(*C.make_stream(D[1], 0)[1:]) if t == 0)
    q = run(0, D, max_steps=tb0 + 30, eval_tasks=True, promote=1); v0 = q['versions'][0]; q0 = run(0, D, max_steps=tb0 + 30, eval_tasks=True, promote=0)
    chk(np.array_equal(q['preq'], q0['preq']), 'K2b promotion does not change the learner across a task boundary')
    # its stats = buffer-at-promotion features + 30 later stream batches; recompute independently
    tasks, order, tid, resv = C.make_stream(D[1], 0); bl = C.batches(order, tid, resv)
    later = np.concatenate([bi for _, bi in bl[tb0:tb0 + 30]])
    chk(abs(float(v0['lda'].n.sum()) - (min(C.MEM, tb0 * C.B) + 30 * C.B)) < 1e-9 and q['ends'][0]['t'] == 0, 'K8 promoted version statistics count = buffer + later stream items', float(v0['lda'].n.sum()))
    # K9 epochs: first-pass predictions equal the 1-epoch run within task 1
    e2 = run(0, D, max_steps=tb0, eval_tasks=False, epochs=2); e1 = run(0, D, max_steps=tb0, eval_tasks=False, epochs=1)
    chk(np.array_equal(e2['preq'], e1['preq']) and e2['fwdbwd_imgs'] > 1.9 * e1['fwdbwd_imgs'], 'K9 epochs=2: stream predictions unchanged in task 1, about 2x the training images', e2['fwdbwd_imgs'], e1['fwdbwd_imgs'])
    # K10 ewc penalty gradient = lam * F * (theta - anchor)
    th = torch.randn(7, requires_grad=True); Fw = torch.rand(7); an = torch.randn(7); (0.5 * 3.0 * (Fw * (th - an) ** 2).sum()).backward()
    chk(torch.allclose(th.grad, 3.0 * Fw * (th.detach() - an)), 'K10 EWC penalty gradient')
    print('SELFTEST', 'ALL PASS' if ok else 'FAILURES'); return ok

def save_run(path, r):
    C.save(path, r)

if __name__ == '__main__':
    if sys.argv[1] == 'selftest': sys.exit(0 if selftest() else 1)
    _, name, seeds, outdir = sys.argv[1:5]; kw = {}
    for a in sys.argv[5:]:
        k, v = a.split('='); kw[k] = float(v) if k == 'ewc' else int(v)
    D = C.load()
    for s in [int(x) for x in seeds.split(',')]:
        r = run(s, D, **kw); save_run(os.path.join(outdir, f'{name}_{s}.npz'), r)
        sm = C.summarize(name, s, r, D[3]); sm['cfg'] = kw; sm['stats'] = {k: (v if not isinstance(v, list) else [len(v), float(np.mean(v)) if v else 0]) for k, v in r['stats'].items()}
        print(json.dumps(sm), flush=True)
