# ===== atlas_analysis v2: everything after the features exist (pure numpy) =====
import numpy as np, time
from atlas_lib import (pooled_cov, lda_scores, log_softmax, fit_temperature, fit_poe, eps_mixture_logp,
                       greg_means, global_affine_means, within_class_r2)

EPS = 0.05          # epsilon of the stable-expert mixture, fixed in advance
MATCH_CAP = 10      # items per class in the matched-size bridge test

def run_analysis(D, seed):
    """D: dict with
         tasks, y_train, stream_ids, bridge_ids (may be empty), control_ids (>= 2 per class)
         F[t]   features of ALL train items under snapshot f_t (end of task t);  R  stable-chart features of all train items
         buffers[t]  train ids in the replay buffer at the end of task t
         Ftest, Rtest, y_test, head_acc
         sigma_from  'reserve' (bridge+control items) or 'buffer' (final buffer + control): the ONE Sigma all arms share
    """
    rng = np.random.default_rng(10_000 + seed)
    t0 = time.time()
    tasks = [np.asarray(c) for c in D['tasks']]; T = len(tasks)
    classes = np.concatenate(tasks); C = len(classes)
    cidx = {int(c): i for i, c in enumerate(classes)}
    y = np.asarray(D['y_train']); F = D['F']; R = D['R']
    stream = np.asarray(D['stream_ids'])
    br = np.asarray(D['bridge_ids'], dtype=np.int64); ct = np.asarray(D['control_ids'], dtype=np.int64)
    has_bridge = len(br) > 0
    # ---- integrity asserts (leakage, coverage) ----
    res_ids = np.concatenate([br, ct])
    assert len(np.intersect1d(stream, res_ids)) == 0, 'reserve leaked into stream'
    assert len(np.intersect1d(br, ct)) == 0, 'bridge and control overlap'
    for b in D['buffers']:
        assert np.isin(b, stream).all(), 'buffer holds a non-stream item'
    assert set(np.unique(y[ct]).tolist()) == set(classes.tolist()), 'control must cover every class'
    if has_bridge:
        assert set(np.unique(y[br]).tolist()) == set(classes.tolist()), 'bridge must cover every class'
    f64 = lambda a: np.asarray(a, np.float64)
    task_of = {int(c): t for t, cs in enumerate(tasks) for c in cs}
    ids_of = {int(c): stream[y[stream] == c] for c in classes}
    FT = F[T - 1]
    final_buf = np.asarray(D['buffers'][T - 1])
    # ---- exact statistics ----
    mu_r = {c: f64(R[ids]).mean(0) for c, ids in ids_of.items()}
    mu_rec = {c: f64(F[task_of[c]][ids]).mean(0) for c, ids in ids_of.items()}
    oracle = {c: f64(FT[ids]).mean(0) for c, ids in ids_of.items()}
    sig_ids = res_ids if D['sigma_from'] == 'reserve' else np.concatenate([final_buf, ct])
    Sig = pooled_cov(f64(FT[sig_ids]), y[sig_ids])
    Sig_r = pooled_cov(f64(R[stream]), y[stream])          # stable chart keeps exact sufficient statistics
    ytest = np.asarray(D['y_test']); keep = np.isin(ytest, classes)
    Ftest = f64(D['Ftest'])[keep]; Rtest = f64(D['Rtest'])[keep]; ytest = ytest[keep]
    yti = np.array([cidx[int(c)] for c in ytest]); yci = np.array([cidx[int(c)] for c in y[ct]])
    old_mask = np.isin(ytest, np.concatenate(tasks[: T // 2]))
    Fct, Rct = f64(FT[ct]), f64(R[ct])
    Mr = np.stack([mu_r[int(c)] for c in classes])

    # ---- arms (class means in f_T) ----
    arms, fallbacks = {'O_full': oracle}, {}

    def bonly(pool):
        g = f64(FT[pool]).mean(0); out = {}; nfb = 0
        for c in classes:
            m = pool[y[pool] == c]
            if len(m) == 0: out[int(c)] = g; nfb += 1
            else: out[int(c)] = f64(FT[m]).mean(0)
        return out, nfb
    arms['B_only_buf'], fallbacks['B_only_buf'] = bonly(final_buf)

    def chain(kind, bridges):
        """bridges[t] = item ids used for the hop f_{t-1} -> f_t (t = 1..T-1)"""
        est = {int(c): mu_rec[int(c)] for c in tasks[0]}; nfb = 0
        for t in range(T - 1):
            old = [int(c) for c in np.concatenate(tasks[: t + 1])]
            ids = bridges[t + 1]
            A = f64(F[t][ids]); Y = f64(F[t + 1][ids]); yb = y[ids]
            if kind == 'multi':
                A = np.hstack([f64(R[ids]), A]); mu_aux = {c: np.concatenate([mu_r[c], est[c]]) for c in old}
                est, fb = greg_means(old, mu_aux, A, Y, yb, rng, split=R.shape[1]); nfb += len(fb)
            elif kind == 'aff':
                est = global_affine_means(old, est, A, Y, yb, rng)
            else:
                est, fb = greg_means(old, est, A, Y, yb, rng); nfb += len(fb)
            for c in tasks[t + 1]:
                est[int(c)] = mu_rec[int(c)]
        return est, nfb

    buf_bridges = [np.asarray(b) for b in D['buffers']]
    arms['CH_cv_buf'], fallbacks['CH_cv_buf'] = chain('aff', buf_bridges)
    arms['GR_prev_buf'], fallbacks['GR_prev_buf'] = chain('prev', buf_bridges)
    arms['GR_multi_buf'], fallbacks['GR_multi_buf'] = chain('multi', buf_bridges)
    est, fb = greg_means([int(c) for c in classes], mu_r, f64(R[final_buf]), f64(FT[final_buf]), y[final_buf], rng)
    arms['GR_stable_buf'], fallbacks['GR_stable_buf'] = est, len(fb)

    matched_n = None
    if has_bridge:
        # Matched-size bridge test: at every hop, per class, the SAME number k_c = min(cap, #buffer, #reserve) of items
        # from the buffer (trained on) and from the reserve (never trained on). Only the provenance differs.
        mrng = np.random.default_rng(20_000 + seed)
        mb, mr = [None], [None]; ks = []
        for t in range(1, T):
            seen = np.concatenate(tasks[: t + 1]); bb = np.asarray(D['buffers'][t]); sb, sr = [], []
            for c in seen:
                pb = bb[y[bb] == c]; pr = br[y[br] == c]
                k = min(MATCH_CAP, len(pb), len(pr))
                if k == 0: continue
                sb.append(mrng.choice(pb, k, replace=False)); sr.append(mrng.choice(pr, k, replace=False)); ks.append(k)
            mb.append(np.concatenate(sb)); mr.append(np.concatenate(sr))
        matched_n = float(np.mean(ks))
        arms['GR_prev_buf_m'], fallbacks['GR_prev_buf_m'] = chain('prev', mb)
        arms['GR_prev_res_m'], fallbacks['GR_prev_res_m'] = chain('prev', mr)
        arms['B_only_buf_m'], fallbacks['B_only_buf_m'] = bonly(mb[-1])
        arms['B_only_res_m'], fallbacks['B_only_res_m'] = bonly(mr[-1])

    # ---- scoring: one Sigma for all learned arms; experts calibrated and combined on CONTROL items only ----
    out = {'acc': {}, 'acc_old': {}, 'acc_recent': {}, 'nll': {}, 'fallbacks': fallbacks, 'poe_w': {}, 'matched_n': matched_n}
    def record(name, Ltest):
        pred = Ltest.argmax(1)
        out['acc'][name] = float((pred == yti).mean() * 100)
        out['acc_old'][name] = float((pred[old_mask] == yti[old_mask]).mean() * 100)
        out['acc_recent'][name] = float((pred[~old_mask] == yti[~old_mask]).mean() * 100)
        out['nll'][name] = float(-Ltest[np.arange(len(yti)), yti].mean())
    tau_r = fit_temperature(lda_scores(Rct, Mr, Sig_r), yci)
    L_rc_ct = log_softmax(lda_scores(Rct, Mr, Sig_r) / tau_r); L_rc_te = log_softmax(lda_scores(Rtest, Mr, Sig_r) / tau_r)
    record('RC_LDA', L_rc_te)
    viol = {}
    for name, means in arms.items():
        M = np.stack([means[int(c)] for c in classes])
        S_ct, S_te = lda_scores(Fct, M, Sig), lda_scores(Ftest, M, Sig)
        tau = fit_temperature(S_ct, yci)
        L_ct, L_te = log_softmax(S_ct / tau), log_softmax(S_te / tau)
        record(name, L_te)
        if name in ('O_full', 'B_only_buf', 'CH_cv_buf', 'GR_prev_buf', 'GR_multi_buf'):
            w, tau_p = fit_poe(L_rc_ct, L_ct, yci)
            Lp = log_softmax((w * L_rc_te + (1 - w) * L_te) / tau_p)
            record(f'PoE[RC+{name}]', Lp); out['poe_w'][name] = w
            Lm = eps_mixture_logp(L_rc_te, Lp, EPS)
            record(f'Mix[RC,PoE+{name}]', Lm)
            idx = np.arange(len(yti))
            viol[name] = int((-Lm[idx, yti] > -L_rc_te[idx, yti] - np.log(EPS) + 1e-9).sum())
    out['mixture_bound_violations'] = viol
    out['acc']['ER_head'] = float(D['head_acc'])
    # ---- Test 1: within-class R^2 on held-out items (only with >= 10 held-out items per class) ----
    r2 = {}
    if len(res_ids) >= 10 * C:
        Yr = f64(FT[res_ids]); yr = y[res_ids]
        r2['fT|r'] = within_class_r2(f64(R[res_ids]), Yr, yr, rng)
        if T >= 2:
            r2['fT|f(T-1)'] = within_class_r2(f64(F[T - 2][res_ids]), Yr, yr, rng)
            r2['fT|[r,f(T-1)]'] = within_class_r2(np.hstack([f64(R[res_ids]), f64(F[T - 2][res_ids])]), Yr, yr, rng, split=R.shape[1])
            r2['fT|f1'] = within_class_r2(f64(F[0][res_ids]), Yr, yr, rng)
            hops = [within_class_r2(f64(F[t][res_ids]), f64(F[t + 1][res_ids]), yr, rng) for t in range(T - 1)]
            r2['hop_mean'] = float(np.mean(hops)); r2['hop_min'] = float(np.min(hops))
    out['r2'] = {k: float(v) for k, v in r2.items()}
    out['seconds'] = time.time() - t0
    return out
