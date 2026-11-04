# ===== atlas_analysis: everything after the features exist (pure numpy) =====
import numpy as np, time

def run_analysis(D, seed):
    """D: dict with
         tasks       list of arrays of class ids (task t = tasks[t])
         y_train     labels of all train items
         stream_ids  train ids the learner saw (excludes the reserve)
         bridge_ids, control_ids   reserve splits (never trained on)
         F           list of T arrays, F[t] = features of ALL train items under snapshot f_t (end of task t)
         R           stable random-chart features of all train items
         buffers     list of T arrays, train ids in the replay buffer at the end of task t
         Ftest, Rtest, y_test     test features under the final f_T, under r, and labels
         head_acc    accuracy of the learner's own softmax head (sanity only)
    Returns a dict of results for this run."""
    rng = np.random.default_rng(10_000 + seed)
    t0 = time.time()
    tasks = [np.asarray(c) for c in D['tasks']]; T = len(tasks)
    classes = np.concatenate(tasks)
    y = np.asarray(D['y_train']); F = D['F']; R = D['R']
    stream = np.asarray(D['stream_ids']); br = np.asarray(D['bridge_ids']); ct = np.asarray(D['control_ids'])
    res_ids = np.concatenate([br, ct])
    task_of = {int(c): t for t, cs in enumerate(tasks) for c in cs}
    # ---- leakage / integrity asserts ----
    assert len(np.intersect1d(stream, res_ids)) == 0, 'reserve leaked into stream'
    assert len(np.intersect1d(br, ct)) == 0
    for b in D['buffers']:
        assert np.isin(b, stream).all(), 'buffer holds a non-stream item'
    assert set(np.unique(y[res_ids]).tolist()) == set(classes.tolist())
    f64 = lambda a: np.asarray(a, np.float64)
    ids_of = {int(c): stream[y[stream] == c] for c in classes}
    FT = F[T - 1]
    # ---- exact statistics ----
    mu_r = {c: f64(R[ids]).mean(0) for c, ids in ids_of.items()}                 # stable chart, all confirmed data
    mu_rec = {c: f64(F[task_of[c]][ids]).mean(0) for c, ids in ids_of.items()}   # recorded at end of own task
    oracle = {c: f64(FT[ids]).mean(0) for c, ids in ids_of.items()}              # ceiling: all stream data in f_T
    Sig_shared = pooled = None
    from atlas_lib import pooled_cov, lda_predict, greg_means, global_affine_means, within_class_r2
    Sig_shared = pooled_cov(f64(FT[res_ids]), y[res_ids])                        # same Sigma for every arm
    Sig_oracle = pooled_cov(f64(FT[stream]), y[stream])
    Sinv = np.linalg.inv(Sig_shared)
    ytest = np.asarray(D['y_test']); keep = np.isin(ytest, classes)
    Ftest = f64(D['Ftest'])[keep]; Rtest = f64(D['Rtest'])[keep]; ytest = ytest[keep]
    old_cls = set(np.concatenate(tasks[: T // 2]).tolist())

    def evaluate(means, Sig=Sig_shared, Xte=Ftest):
        M = np.stack([means[int(c)] for c in classes])
        return float((classes[lda_predict(Xte, M, Sig)] == ytest).mean() * 100)

    def mean_err(means, subset=None):
        cs = [int(c) for c in classes if subset is None or int(c) in subset]
        d = np.stack([means[c] - oracle[c] for c in cs])
        return float(np.einsum('nd,de,ne->n', d, Sinv, d).mean())

    # ---- arms ----
    arms, fallbacks = {}, {}
    arms['O_full'] = oracle

    def bonly(ids_pool):
        g = f64(FT[ids_pool]).mean(0); out = {}; nfb = 0
        for c in classes:
            m = ids_pool[y[ids_pool] == c]
            if len(m) == 0: out[int(c)] = g; nfb += 1
            else: out[int(c)] = f64(FT[m]).mean(0)
        return out, nfb
    arms['B_only_buf'], fallbacks['B_only_buf'] = bonly(np.asarray(D['buffers'][T - 1]))
    arms['B_only_res'], fallbacks['B_only_res'] = bonly(br)

    def bridge_at(t_new, kind):
        """bridge items available when hopping into f_{t_new}"""
        if kind == 'buf':
            return np.asarray(D['buffers'][t_new])
        seen = np.concatenate(tasks[: t_new + 1])
        return br[np.isin(y[br], seen)]

    def chain(kind, bkind):
        est = {int(c): mu_rec[int(c)] for c in tasks[0]}; nfb = 0
        for t in range(T - 1):
            old = [int(c) for c in np.concatenate(tasks[: t + 1])]
            ids = bridge_at(t + 1, bkind)
            A = f64(F[t][ids]); Y = f64(F[t + 1][ids]); yb = y[ids]
            if kind == 'multi':
                A = np.hstack([f64(R[ids]), A])
                mu_aux = {c: np.concatenate([mu_r[c], est[c]]) for c in old}
            else:
                mu_aux = est
            split = R.shape[1] if kind == 'multi' else None
            if kind == 'aff':
                est = global_affine_means(old, mu_aux, A, Y, yb, rng)
            else:
                est, fb = greg_means(old, mu_aux, A, Y, yb, rng, split=split); nfb += len(fb)
            for c in tasks[t + 1]:
                est[int(c)] = mu_rec[int(c)]
        return est, nfb

    for kind, name in (('aff', 'CH_cv'), ('prev', 'GR_prev'), ('multi', 'GR_multi')):
        for bk in ('buf', 'res'):
            arms[f'{name}_{bk}'], fallbacks[f'{name}_{bk}'] = chain(kind, bk)
    for bk in ('buf', 'res'):
        ids = bridge_at(T - 1, bk)
        est, fb = greg_means([int(c) for c in classes], mu_r, f64(R[ids]), f64(FT[ids]), y[ids], rng)
        arms[f'GR_stable_{bk}'], fallbacks[f'GR_stable_{bk}'] = est, len(fb)

    out = {'acc': {}, 'acc_oracleSigma': {}, 'err_all': {}, 'err_old': {}, 'err_recent': {}, 'fallbacks': fallbacks}
    recent = set(classes.tolist()) - old_cls
    for k, m in arms.items():
        out['acc'][k] = evaluate(m)
        out['acc_oracleSigma'][k] = evaluate(m, Sig_oracle)
        out['err_all'][k] = mean_err(m); out['err_old'][k] = mean_err(m, old_cls); out['err_recent'][k] = mean_err(m, recent)
    # sanity references (not part of the hypotheses)
    Sig_r = pooled_cov(f64(R[stream]), y[stream])
    out['acc']['RC_LDA'] = evaluate(mu_r, Sig_r, Rtest)
    out['acc']['ER_head'] = float(D['head_acc'])
    # ---- Test 1: within-class R^2 on reserve items (never trained on) ----
    r2 = {}
    Yr = f64(FT[res_ids]); yr = y[res_ids]
    r2['fT|r'] = within_class_r2(f64(R[res_ids]), Yr, yr, rng)
    for lag in (1, 5, T - 1):
        if 1 <= lag <= T - 1:
            r2[f'fT|f(T-{lag})'] = within_class_r2(f64(F[T - 1 - lag][res_ids]), Yr, yr, rng)
    if T >= 2:
        r2['fT|[r,f(T-1)]'] = within_class_r2(np.hstack([f64(R[res_ids]), f64(F[T - 2][res_ids])]), Yr, yr, rng, split=R.shape[1])
        hops = [within_class_r2(f64(F[t][res_ids]), f64(F[t + 1][res_ids]), yr, rng) for t in range(T - 1)]
        r2['hop_mean'] = float(np.mean(hops)); r2['hop_min'] = float(np.min(hops)); r2['hops'] = [float(h) for h in hops]
    out['r2'] = {k: (float(v) if not isinstance(v, list) else v) for k, v in r2.items()}
    out['seconds'] = time.time() - t0
    return out
