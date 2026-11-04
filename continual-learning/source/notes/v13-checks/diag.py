"""Diagnostics: why did M2c tie with ER, and why did compute-matched ER win?
Same toy, data, model and streams as toy_cl3.py (functions imported from it).

Methods:
  er        : ER, 16 new + 4 replay per step, mean CE over the 20 items
  er_w<w>   : ER with the SAME 4 replay items, but replay CE weighted by w:
              loss = (sum_new CE + w*sum_rep CE) / (16 + 4w). Same compute as ER.
              w = 9 gives replay the same loss share as er_cm (36/52). w = 1 must reproduce ER exactly (sanity).
  er_cm     : ER with 36 replay items per step (compute-matched to M2c's 32 pool items)
  margin_qp : M2c as specified (k_r = 2, correct-only pool of 32, tracked margins, R_pre from env RREF)
  mqp_cm    : M2c on top of er_cm (does protection add anything once replay share is high?)
  er_ace    : ER-ACE: incoming items' CE restricted to current-task logits (targets the recency bias); same compute as ER

Diagnostics (every 25 steps from task 1 on, measured on a fixed HELD-OUT set: 64 old-class training
items never in the buffer, drawn from their own RNG):
  first-order damage D(d) = sum over correct held-out items, top-2 rivals, of max(0, -(grad margin)^T d)
  For M2c steps: the same damage on the constrained pool, for d0 (Adam) and d (projected), giving
  the fraction of damage removed in-sample vs held out.
Per-step for M2c: fraction of steps where the projection changed the step, and ||d - d0||_{B^-1}/||d0||_{B^-1}.
End of run: old-class vs last-task accuracy, share of old-class test errors predicted as a last-task class,
old-class accuracy after each task (forgetting curve)."""
import sys, os, json, time, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('tc', os.path.join(HERE, 'toy_cl3.py')); tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
K, TASKS = tc.K, tc.TASKS
R_REF = int(os.environ.get('RREF', '25'))

def qp_polished(Bd, C, d0, s):
    """Dual QP by L-BFGS-B, then an exact active-set polish (solve G_FF mu_F = -r_F on the active set, iterate).
    Used for the zero-slack variant, where L-BFGS-B alone left ~1e-6 violations that the feasibility assert caught."""
    G = C.T @ (Bd[:, None] * C); r = C.T @ d0 + s
    if np.all(r >= 0): return d0
    from scipy.optimize import minimize
    res = minimize(lambda mu: (.5 * mu @ G @ mu + mu @ r, G @ mu + r), np.zeros(len(s)), jac=True,
                   bounds=[(0, None)] * len(s), method='L-BFGS-B', options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 5000})
    mu = res.x; n = len(s); damp = 1e-12 * np.trace(G) / n
    for _ in range(100):
        F = mu > 1e-12 * max(1, mu.max())
        muF = np.zeros(n)
        if F.any(): muF[F] = np.linalg.solve(G[np.ix_(F, F)] + damp * np.eye(F.sum()), -r[F])
        gr = G @ muF + r
        if (muF >= 0).all() and (gr[~F] >= -1e-10 * (1 + np.abs(r[~F]))).all(): mu = muF; break
        if (muF < 0).any(): F2 = F & (muF > 0)
        else: F2 = F
        F2 = F2 | ((~F) & (gr < 0))
        mu = np.where(F2, np.maximum(muF, 1e-9), 0.0)
    return d0 + Bd * (C @ mu)

def damage(Cm, d):
    return float(np.maximum(0, -(Cm.T @ d)).sum())

def run(method, seed, n_mem=5, steps=300, bs=16, rep=4, lr=1e-3, k_riv=2, n_pool=32, kappa=0.2, mfloor=0.5):
    early = method.endswith('_s')   # buffer policy: insert each task's memory items at task START (so the current task is replayed too)
    if early: method = method[:-2]
    w = 1.0
    if method.startswith('er_w'): w = float(method[4:])
    if method in ('er_cm', 'mqp_cm'): rep = rep + 32
    if method == 'mqp_k0': kappa = 0.0; method = 'margin_qp'   # M2c with zero slack: no constrained margin may decrease at first order
    protect = method in ('margin_qp', 'mqp_cm')
    Xtr, ytr, Xte, yte = tc.make_data(seed)
    m = tc.model(seed + 1000); rng = np.random.default_rng(seed); rng_pool = np.random.default_rng(seed + 7919)
    rng_ho = np.random.default_rng(seed + 31337)
    Np = tc.flat(m).numel(); mv = torch.zeros(Np, dtype=torch.float64); vv = torch.zeros(Np, dtype=torch.float64); t = 0
    beta1, beta2, eps = .9, .999, 1e-8
    buf = []; Ctrack = None; mt = None; since = 0
    log = {'ho_red': [], 'pool_red': [], 'ho_d0': [], 'changed': 0, 'psteps': 0, 'relchg': [], 'curve': []}
    yn = ytr.numpy()
    for task in range(TASKS):
        cls = [2 * task, 2 * task + 1]; idx = np.where(np.isin(yn, cls))[0]
        # held-out set for this task: old-class train items not in the buffer (fixed for the task)
        if task > 0:
            old = np.where((yn < 2 * task) & ~np.isin(np.arange(len(yn)), buf))[0]
            ho = rng_ho.choice(old, 64, replace=False)
        if early:
            for c in cls:
                ci = np.where(yn == c)[0]; buf += rng.choice(ci, n_mem, replace=False).tolist()
        for st in range(steps):
            b = rng.choice(idx, bs, replace=False); X, y = Xtr[b], ytr[b]
            nrep = 0
            if len(buf) > 0:
                rb = rng.choice(buf, rep, replace=len(buf) < rep); X = torch.cat([X, Xtr[rb]]); y = torch.cat([y, ytr[rb]]); nrep = rep
            m.zero_grad()
            out = m(X)
            if method == 'er_ace':   # ER-ACE (Caccia et al. 2022): incoming items' CE only over the current task's logits
                mask = torch.full((K,), float('-inf')); mask[cls] = 0
                out = torch.cat([out[:bs] + mask, out[bs:]])
            ce = torch.nn.functional.cross_entropy(out, y, reduction='none')
            wts = torch.ones(len(y)); wts[bs:] = w
            loss = (ce * wts).sum() / wts.sum(); loss.backward()
            g = torch.cat([p.grad.reshape(-1) for p in m.parameters()]).double(); t += 1
            mv = beta1 * mv + (1 - beta1) * g; vv = beta2 * vv + (1 - beta2) * g * g
            mh = mv / (1 - beta1 ** t); vh = vv / (1 - beta2 ** t)
            Bd = (lr / (vh.sqrt() + eps)).numpy(); d0 = -Bd * mh.numpy(); d = d0
            Cuse = None
            if protect and len(buf) > 0:
                pool = rng_pool.choice(buf, min(n_pool, len(buf)), replace=False)
                if Ctrack is None or since >= R_REF:
                    with torch.no_grad(): ok = (m(Xtr[pool]).argmax(1) == ytr[pool]).numpy()
                    pc = pool[ok]
                    if len(pc) == 0: Ctrack = np.zeros((Np, 0)); mt = np.zeros(0)
                    else: Ctrack, mt = tc.margin_grads(m, Xtr[pc], ytr[pc], k_riv)
                    since = 0
                since += 1
                s = kappa * np.maximum(mt - mfloor, 0)
                if Ctrack.shape[1] > 0:
                    d = qp_polished(Bd, Ctrack, d0, s) if kappa == 0 else tc.qp_project(Bd, Ctrack, d0, s)
                    assert mh.numpy() @ d <= 1e-12
                    assert np.all(Ctrack.T @ d >= -s - 1e-6 * (1 + np.abs(s))), 'projected step infeasible'
                    mt = mt + Ctrack.T @ d; Cuse = Ctrack
                log['psteps'] += 1
                ch = not np.array_equal(d, d0); log['changed'] += int(ch)
                log['relchg'].append(float(np.sqrt(((d - d0) ** 2 / Bd).sum() / ((d0 ** 2 / Bd).sum()))))
            if task > 0 and st % 25 == 0:
                with torch.no_grad(): okh = (m(Xtr[ho]).argmax(1) == ytr[ho]).numpy()
                if okh.sum() > 0:
                    Ch, _ = tc.margin_grads(m, Xtr[ho[okh]], ytr[ho[okh]], k_riv)
                    D0 = damage(Ch, d0); log['ho_d0'].append(D0)
                    if protect and D0 > 0: log['ho_red'].append(1 - damage(Ch, d) / D0)
                if protect and Cuse is not None and Cuse.shape[1] > 0:
                    P0 = damage(Cuse, d0)
                    if P0 > 0: log['pool_red'].append(1 - damage(Cuse, d) / P0)
            tc.set_flat(m, tc.flat(m).detach().double().add(torch.from_numpy(d)).float())
        if not early:
            for c in cls:
                ci = np.where(yn == c)[0]; buf += rng.choice(ci, n_mem, replace=False).tolist()
        with torch.no_grad(): pr = m(Xte).argmax(1)
        if task > 0:
            log['curve'].append(round(float(np.mean([(pr[yte == c] == c).float().mean().item() for c in range(2 * task)])), 4))
    with torch.no_grad():
        pr = m(Xte).argmax(1); bacc = (m(Xtr[buf]).argmax(1) == ytr[buf]).float().mean().item()
    accs = [(pr[yte == c] == c).float().mean().item() for c in range(K)]
    # Where is the loss? Refit ONLY the output layer on frozen penultimate features:
    #  probe_buf : fitted on the buffer (what the learner could do with its own memory), class-balanced by construction
    #  probe_full: fitted on all training data of all classes (how much the features still support)
    feat = torch.nn.Sequential(*list(m.children())[:-1])
    with torch.no_grad(): Ftr, Fte = feat(Xtr), feat(Xte)
    def refit(F, Y):
        torch.manual_seed(0); lin = torch.nn.Linear(F.shape[1], K)
        opt = torch.optim.LBFGS(lin.parameters(), max_iter=500, line_search_fn='strong_wolfe')
        def cl():
            opt.zero_grad(); l = torch.nn.functional.cross_entropy(lin(F), Y) + 1e-3 * (lin.weight ** 2).sum(); l.backward(); return l
        opt.step(cl)
        with torch.no_grad(): p2 = lin(Fte).argmax(1)
        return float(np.mean([(p2[yte == c] == c).float().mean().item() for c in range(K)]))
    allbuf = list(set(buf))
    probe_buf = refit(Ftr[allbuf], ytr[allbuf]); probe_full = refit(Ftr, ytr)
    oldmask = yte < 2 * (TASKS - 1); err = oldmask & (pr != yte)
    rec = float(((pr >= 2 * (TASKS - 1)) & err).sum() / max(1, err.sum()))
    return {'acc': float(np.mean(accs)), 'old_acc': float(np.mean(accs[:8])), 'last_acc': float(np.mean(accs[8:])),
            'recency_err_share': rec, 'probe_buf': probe_buf, 'probe_full': probe_full, 'buffer_acc': bacc, 'curve': log['curve'],
            'ho_d0_mean': float(np.mean(log['ho_d0'])) if log['ho_d0'] else None,
            'ho_reduction': float(np.mean(log['ho_red'])) if log['ho_red'] else None,
            'pool_reduction': float(np.mean(log['pool_red'])) if log['pool_red'] else None,
            'changed_frac': log['changed'] / max(1, log['psteps']) if protect else None,
            'relchg_mean': float(np.mean(log['relchg'])) if log['relchg'] else None}

if __name__ == '__main__':
    meth = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(',')]
    for s in seeds:
        t0 = time.time(); r = run(meth, s); r['sec'] = round(time.time() - t0)
        print(json.dumps({'method': meth, 'seed': s, **r}), flush=True)
