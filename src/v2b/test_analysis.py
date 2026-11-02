import numpy as np, sys; sys.path.insert(0, '.')
from atlas_analysis import run_analysis
fails = []
def check(n, c, i=''):
    print('PASS' if c else 'FAIL', n, i)
    if not c: fails.append(n)

def synth(noise, seed=0, T=5, per=60, nbr=10, nct=10, M=200, dz=12, d=12, dr=40, buf_bias=0.0, sigma_from='reserve'):
    """Synthetic stream with known truth. buf_bias shifts the features of items that ever enter the buffer
    (a stand-in for 'the learner trained on these items'), so the matched bridge test has a known answer."""
    rng = np.random.default_rng(seed); C = T * 3
    tasks = list(rng.permutation(C).reshape(T, 3))
    mz = rng.normal(size=(C, dz)) * 1.2
    y = np.repeat(np.arange(C), per + nbr + nct); Z = mz[y] + rng.normal(size=(len(y), dz))
    yte = np.repeat(np.arange(C), 30); Zte = mz[yte] + rng.normal(size=(len(yte), dz))
    br, ctl, st = [], [], []
    for c in range(C):
        ids = rng.permutation(np.where(y == c)[0]); br += list(ids[:nbr]); ctl += list(ids[nbr:nbr + nct]); st += list(ids[nbr + nct:])
    st = np.array(st); bufs = []
    for t in range(T):
        seen = st[np.isin(y[st], np.concatenate(tasks[: t + 1]))]
        bufs.append(rng.choice(seen, min(M, len(seen)), replace=False))
    inbuf = np.zeros(len(y), bool); inbuf[np.concatenate(bufs)] = True
    W = rng.normal(size=(dz, d)); F = []; Ws = []
    for t in range(T):
        W = W + 0.4 * rng.normal(size=W.shape); Ws.append(W.copy())
        Ft = Z @ W + 0.3 * t + noise * rng.normal(size=(len(y), d))
        Ft[inbuf] += buf_bias * (t + 1) / T          # buffer items drift toward a common direction
        F.append(Ft.astype(np.float32))
    Wr = rng.normal(size=(dz, dr)); R = (np.tanh(Z @ Wr * 0.5) + 0.05 * rng.normal(size=(len(y), dr))).astype(np.float32)
    Rte = np.tanh(Zte @ Wr * 0.5); Fte = Zte @ Ws[-1] + 0.3 * (T - 1) + noise * rng.normal(size=(len(yte), d))
    return dict(tasks=tasks, y_train=y, stream_ids=st, bridge_ids=np.array(br, dtype=np.int64), control_ids=np.array(ctl),
                F=F, R=R, buffers=bufs, Ftest=Fte, Rtest=Rte, y_test=yte, head_acc=0.0, sigma_from=sigma_from)

if __name__ == '__main__':
    o = run_analysis(synth(0.0, nbr=40), 0)                     # exact linear features: transport recovers the oracle
    for k in ('CH_cv_buf', 'GR_prev_buf', 'GR_multi_buf', 'GR_prev_res_m', 'GR_prev_buf_m'):
        check('exact_' + k, abs(o['acc'][k] - o['acc']['O_full']) < 0.5, f"acc={o['acc'][k]:.1f} oracle={o['acc']['O_full']:.1f}")
    check('exact_hopR2', o['r2']['hop_min'] > 0.999)
    check('exact_no_fallbacks', sum(o['fallbacks'].values()) == 0, o['fallbacks'])
    check('bound_never_violated', sum(o['mixture_bound_violations'].values()) == 0)
    o = run_analysis(synth(1.0), 0)
    for k in sorted(o['acc'], key=lambda k: -o['acc'][k]): print(f"   {k:28s} acc={o['acc'][k]:.1f} nll={o['nll'].get(k, float('nan')):.3f}")
    check('noisy_greg_beats_bonly', o['acc']['GR_prev_buf'] >= o['acc']['B_only_buf'] - 0.5)
    check('poe_not_worse_than_parts', o['acc']['PoE[RC+GR_prev_buf]'] >= max(o['acc']['RC_LDA'], o['acc']['GR_prev_buf']) - 2.0)
    check('mixture_nll_bound_mean', o['nll']['Mix[RC,PoE+GR_prev_buf]'] <= o['nll']['RC_LDA'] - np.log(0.05) + 1e-9)
    check('bound_never_violated_noisy', sum(o['mixture_bound_violations'].values()) == 0)
    check('matched_n_close_to_cap', 8 < o['matched_n'] <= 10, o['matched_n'])
    # matched bridge with a known provenance bias: reserve must win; with no bias: no systematic winner
    gaps0, gaps1 = [], []
    for s in range(4):
        a = run_analysis(synth(0.5, seed=s, buf_bias=0.0), s); b = run_analysis(synth(0.5, seed=s, buf_bias=3.0), s)
        gaps0.append(a['acc']['GR_prev_res_m'] - a['acc']['GR_prev_buf_m']); gaps1.append(b['acc']['GR_prev_res_m'] - b['acc']['GR_prev_buf_m'])
    check('matched_detects_buffer_bias', np.mean(gaps1) > 3 and min(gaps1) > 0, np.round(gaps1, 1))
    check('matched_unbiased_no_gap', abs(np.mean(gaps0)) < 3, np.round(gaps0, 1))
    # equal-bytes config: no bridge, 2 control items per class, Sigma from buffer
    o = run_analysis(synth(1.0, nbr=0, nct=2, sigma_from='buffer'), 0)
    check('no_bridge_config_runs', 'GR_prev_res_m' not in o['acc'] and o['r2'] == {} and np.isfinite(list(o['acc'].values())).all())
    print('FAILS', fails)
    if fails: sys.exit(1)
