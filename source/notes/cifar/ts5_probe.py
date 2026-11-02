"""A1 probes (audit_plan.md): matched-data LDA probes on every saved T-S5 encoder. Offline diagnostics, not online results.
Every probe fits C.StreamLDA (same shrinkage as the online readouts) on the SAME items for every encoder.
Run: CL_K=100 CL_NTASK=20 CIFAR_DIR=data100 RESV=20 LR=0.03 MEM=2000 python3 ts5_probe.py <seed>   -> results/ts5_probe/probe_<seed>.json"""
import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C
assert C.K == 100 and C.NTASK == 20 and C.RESV == 20
def probe(F, Ft, yf, yte, sel_fit, classes):
    """Fit LDA on F[sel_fit] restricted to `classes`; accuracy on test items of `classes`, argmax over `classes` only."""
    m = np.isin(yf.numpy()[sel_fit], classes); lda = C.StreamLDA(F.shape[1]); lda.update(F[sel_fit][m], yf[sel_fit][m])
    mt = np.isin(yte.numpy(), classes); sc = lda.scores(Ft[mt])[1]; cls = torch.where(lda.n > 0)[0]
    assert sorted(cls.tolist()) == sorted(classes)
    return float((cls[sc.argmax(1)] == yte[mt]).double().mean())

def main():
    torch.set_num_threads(1)
    s = int(sys.argv[1]); Xtr, ytr, Xte, yte = C.load(); tasks, order, tid, resv = C.make_stream(ytr, s)
    E = torch.load(f'results/ts5_probe/enc_{s}.pt'); z = np.load(f'results/ts5/erp_{s}.npz'); T = C.NTASK - 1
    assert tasks == z['tasks'].tolist()
    # g3: saved version LDA states reproduce the online version logprobs at the final task end exactly
    def mknet(state): n = C.make_net(0); n.load_state_dict(state); n.eval(); return n
    ok3 = True
    for vi, v in enumerate(E['versions']):
        lda = C.StreamLDA(16 * C.W); lda.S1, lda.n, lda.XX = v['S1'], v['n'], v['XX']
        lp = lda.logprobs(C.chunked(mknet(v['state']).feats, Xte)).float().numpy(); ok3 &= bool(np.array_equal(lp, z[f'e{T}_ver{vi}_test_lp']))
    assert ok3, 'g3 failed'
    print('g3 PASS: saved version LDA states reproduce the online logprobs exactly', flush=True)
    fit = order[~resv]; assert len(fit) == len(ytr) - C.RESV * C.K and not np.isin(fit, order[resv]).any()
    yf = ytr[fit]; rng = np.random.default_rng(777 + s)
    small = np.concatenate([rng.choice(np.where(yf.numpy() == c)[0], 20, replace=False) for c in range(C.K)])
    t0c, t03c = tasks[0], sorted(c for t in tasks[:4] for c in t)
    allc = list(range(C.K)); allfit = np.arange(len(fit)); out = {'seed': s, 'enc': []}
    encs = [(f"v{vi}", v['t'], v['state']) for vi, v in enumerate(E['versions'])] + [('final', T, E['final']), ('rc', -1, None)]
    for name, t, st in encs:
        net = C.make_net(s * 1000 + 2) if st is None else mknet(st)
        F = C.chunked(net.feats, Xtr[fit]); Ft = C.chunked(net.feats, Xte)
        r = {'name': name, 't': t, 'PF': probe(F, Ft, yf, yte, allfit, allc), 'PS': probe(F, Ft, yf, yte, small, allc), 'PO': probe(F, Ft, yf, yte, allfit, t0c), 'PO4': probe(F, Ft, yf, yte, allfit, t03c)}
        if name == 'rc':  # g4: RC's streaming LDA used exactly the non-reserve stream items, so PF(rc) must equal the rc arm's final accuracy
            rcz = np.load(f'results/ts5/rc_{s}.npz'); a_rc = C.acc_seen(torch.from_numpy(rcz[f'e{T}_test_lda_lp'].astype(np.float64)), yte, allc)
            assert abs(a_rc - r['PF']) < 5e-4, ('g4 failed', a_rc, r['PF']); print('g4 PASS: PF(rc) =', r['PF'], 'rc arm =', a_rc, flush=True)
        if name.startswith('v'): r['online'] = float(C.acc_seen(torch.from_numpy(np.exp(z[f'e{T}_{name.replace("v", "ver")}_test_lp'].astype(np.float64))), yte, allc))
        out['enc'].append(r); print(json.dumps(r), flush=True)
    json.dump(out, open(f'results/ts5_probe/probe_{s}.json', 'w'), indent=0)

if __name__ == '__main__': main()
