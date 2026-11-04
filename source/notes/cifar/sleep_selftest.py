"""Self-tests for sleep.py (written before any sleep result). Run: RESV=20 LR=0.03 python sleep_selftest.py"""
import sys, os, copy, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C, sleep as Z

def main():
    ok = True
    def chk(c, name, *info):
        nonlocal ok; print('PASS' if c else 'FAIL', name, *info, flush=True); ok &= bool(c)
    torch.set_num_threads(1); D = C.load(); Xtr, ytr, Xte, yte = D
    tasks, order, tid, resv = C.make_stream(ytr, 0); bl = C.batches(order, tid, resv)
    tb = [sum(1 for t, _ in bl if t == k) for k in range(C.NTASK)]; M = tb[0] + tb[1] + 30; n12 = sum(len(bi) for t, bi in bl if t < 2)
    # Z1 sleep=0 equals cifar.py ER exactly; sleep=1 equals ER through the end of task 2 and differs after the first sleep
    ref = C.run_learner('er', 0, D, max_steps=M, eval_tasks=False)
    a = Z.run(0, D, sleep=0, max_steps=M, eval_tasks=False)
    chk(np.array_equal(ref['preq'], a['preq']) and np.array_equal(ref['preq_lp'], a['preq_lp']), 'Z1a sleep=0 equals cifar.run_learner ER exactly')
    s1 = Z.run(0, D, sleep=1, rank=16, dream='buf', max_steps=M, eval_tasks=False)
    chk(np.array_equal(ref['preq'][:n12], s1['preq'][:n12]) and not np.array_equal(ref['preq'][n12:], s1['preq'][n12:]) and len(s1['sleeps']) == 1 and s1['sleeps'][0]['t'] == 1,
        'Z1b sleep=1: identical to ER through task 2, one sleep at t=1, differs afterwards')
    # Z9 causality and Z10 determinism with sleep on
    s2 = Z.run(0, D, sleep=1, rank=16, dream='buf', max_steps=M, eval_tasks=False)
    chk(np.array_equal(s1['preq'], s2['preq']) and s1['sleeps'][0]['agree_resv'] == s2['sleeps'][0]['agree_resv'], 'Z10 determinism with sleep')
    k = tb[0] + tb[1] + 10; s3 = Z.run(0, D, sleep=1, rank=16, dream='buf', max_steps=M, eval_tasks=False, flip_step=k)
    chk(np.array_equal(s1['preq'][:(k + 1) * C.B], s3['preq'][:(k + 1) * C.B]) and not np.array_equal(s1['preq'][(k + 1) * C.B:], s3['preq'][(k + 1) * C.B:]), 'Z9 causality after a sleep (label flip)')
    # Z9b across a sleep, with label-using mix dreams: flip inside task 3, run through the end of task 3 (second sleep)
    M3 = tb[0] + tb[1] + tb[2] + 1; k3 = tb[0] + tb[1] + 100; nk = sum(len(bi) for bi in [b_ for _, b_ in bl[:k3]])
    m1 = Z.run(0, D, sleep=1, rank=4, dream='mix', dep=2, max_steps=M3, eval_tasks=False)
    m2 = Z.run(0, D, sleep=1, rank=4, dream='mix', dep=2, max_steps=M3, eval_tasks=False, flip_step=k3)
    sl0 = lambda r: {k: v for k, v in r['sleeps'][0].items() if k != 'sleep_secs'}
    chk(len(m1['sleeps']) == 2 and sl0(m1) == sl0(m2) and np.array_equal(m1['preq'][:nk + C.B], m2['preq'][:nk + C.B])
        and (m1['sleeps'][1]['agree_resv'] != m2['sleeps'][1]['agree_resv'] or m1['sleeps'][1]['kl_resv'] != m2['sleeps'][1]['kl_resv']),
        'Z9b mix dreams: first sleep and predictions before the flip identical; the second sleep (after it) differs')
    # Z14 task-end path: student/teacher saving and ordering (eval_tasks=True), through the first sleep
    M2 = tb[0] + tb[1]; er = C.run_learner('er', 0, D, max_steps=M2); sv = Z.run(0, D, sleep=1, rank=16, dream='buf', dep=2, max_steps=M2)
    e0, e1 = sv['ends'][0], sv['ends'][1]
    chk(len(sv['ends']) == 2 and np.array_equal(e0['test_logits'], er['ends'][0]['test_logits']) and 'T_test_logits' not in e0
        and np.array_equal(e1['T_test_logits'], er['ends'][1]['test_logits']) and np.array_equal(e1['T_b_buf'], er['ends'][1]['b_buf'])
        and np.array_equal(e1['T_resv_logits'], er['ends'][1]['resv_logits']) and not np.array_equal(e1['test_logits'], er['ends'][1]['test_logits']),
        'Z14 task ends: task 1 = ER; at the first sleep T_* = ER exactly, saved student logits differ')
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        C.save(os.path.join(td, 'x.npz'), sv); zz = np.load(os.path.join(td, 'x.npz'))
        chk(all(f'e1_T_{k}' in zz for k in ('test_logits', 'resv_logits', 'b_buf')) and 'e0_T_test_logits' not in zz, 'Z14b npz holds e1_T_* keys (and none for task 1)')
    # Z12 reserve never in the buffer
    chk(not set(s1['buf_idx']) & set(order[resv].tolist()), 'Z12 reserve disjoint from the buffer (and so from buf/mix dreams)')
    # ---- unit tests of sleep_phase on a controlled pair (slow net S, current net = S + F)
    buf = C.Reservoir(200, np.random.default_rng(0)); idx = torch.as_tensor(order[~resv][:200]); buf.add(idx, ytr[idx])
    Xr = Xtr[order[resv][:100]]
    def pair(seed=7):
        net = C.make_net(seed); net._slow = {n: p.detach().clone() for n, p in net.named_parameters()}
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed + 1)
            for p in net.parameters(): p.add_(0.05 * torch.randn(p.shape, generator=g) * (p.abs().mean() + 1e-3))
        return net
    # Z2 dep=0: the net becomes exactly S (fast update reset, B = 0 so the expert is zero)
    net = pair(); S0 = copy.deepcopy(net._slow); Z.sleep_phase(net, 16, 'buf', 1e-3, 0, np.random.default_rng(1), torch.Generator().manual_seed(1), Xtr, buf, Xr)
    chk(all(torch.equal(p, S0[n]) for n, p in net.named_parameters()), 'Z2 dep=0: net := S exactly (reset of the fast update)')
    # Z3 + Z4: teacher and old S untouched; rank of the consolidated weight change <= r; biases/norms changed freely
    net = pair(); S0 = copy.deepcopy(net._slow); cur = copy.deepcopy(net.state_dict())
    T, st = Z.sleep_phase(net, 4, 'buf', 0.03, 5, np.random.default_rng(1), torch.Generator().manual_seed(1), Xtr, buf, Xr)
    chk(all(torch.equal(T.state_dict()[n], cur[n]) for n in cur), 'Z3a teacher = the net before sleep, unchanged by distillation')
    rk = {n: int(torch.linalg.matrix_rank((p.detach() - S0[n]).reshape(p.shape[0], -1).double(), rtol=1e-3)) for n, p in net.named_parameters() if p.dim() >= 2}  # rtol: float32 rounding of (S+d)-S sits ~1e-5 below the top singular value
    chk(all(v <= 4 for v in rk.values()) and max(rk.values()) == 4, 'Z4 rank-4 expert: every weight change has rank <= 4 (and = 4 somewhere)', rk)
    chk(all(torch.equal(net._slow[n], p.detach()) for n, p in net.named_parameters()), 'Z3b slow weights updated to the merged student')
    # Z5 planted null: teacher == S (no fast update) -> KL 0 throughout, net stays at S up to float error
    net = C.make_net(9); net._slow = {n: p.detach().clone() for n, p in net.named_parameters()}; S0 = copy.deepcopy(net._slow)
    _, st = Z.sleep_phase(net, 0, 'buf', 0.03, 3, np.random.default_rng(1), torch.Generator().manual_seed(1), Xtr, buf, Xr)
    chk(max(st['kl_first_batch_by_epoch']) < 1e-9 and max(float((p - S0[n]).abs().max()) for n, p in net.named_parameters()) < 1e-5, 'Z5 nothing to consolidate -> KL 0 and S unchanged', st['kl_first_batch_by_epoch'])
    # Z6 planted effect: full-rank distillation on the buffer recovers the teacher on the dreams; the leak arm (dreams = reserve)
    #     raises reserve agreement above the buf arm, so the held-out measurement would detect leakage
    #     the teacher = S plus a real fast update: 300 plain SGD steps on task-1 items (so there is a lot to consolidate)
    tr_idx = torch.as_tensor(order[~resv][200:3200])
    def trained_pair(seed=7):
        net = C.make_net(seed); net._slow = {n: p.detach().clone() for n, p in net.named_parameters()}
        o = torch.optim.SGD(net.parameters(), lr=0.03)
        for a in range(0, 3000, 10):
            b = tr_idx[a:a + 10]; o.zero_grad(); torch.nn.functional.cross_entropy(net(Xtr[b]), ytr[b]).backward(); o.step()
        return net
    Xr_u = Xtr[order[~resv][3200:3400]]  # unit-test 'reserve': 200 held-out task-1 items, same size and classes as the buffer
    base = trained_pair(); res = {}
    for dream in ['buf', 'resv']:
        net = copy.deepcopy(base); net._slow = {n: v.clone() for n, v in base._slow.items()}
        S_net = C.make_net(7); before = Z.agreement(copy.deepcopy(net), S_net, Xtr[buf.idx], list(range(10)))
        T, st = Z.sleep_phase(net, 0, dream, 0.03, 30, np.random.default_rng(1), torch.Generator().manual_seed(1), Xtr, buf, Xr_u)
        res[dream] = (Z.agreement(T, net, Xtr[buf.idx], list(range(10))), Z.agreement(T, net, Xr_u, list(range(10))), before)
    (ab, kb), (ar_buf, kr_buf), (a0, k0) = res['buf']; (_, _), (ar_leak, kl_leak), _ = res['resv']
    chk(ab >= 0.97 and kb < 0.2 * k0, 'Z6a full-rank distillation fits the teacher on its dreams', ab, kb, 'start', a0, k0)
    # Z6b leak detector. With 200 iid dreams the buffer arm already generalises almost perfectly (first v3 run: held-out KL
    #     0.0125 leak vs 0.0185 buf, agreement 0.970 vs 0.975), so that planted case is too weak to show anything. Small sets
    #     (40 dreams, 40 held-out items, same classes) make generalisation imperfect, so a leak must show as clearly lower KL.
    bs_ = C.Reservoir(40, np.random.default_rng(0)); i40 = torch.as_tensor(order[~resv][:40]); bs_.add(i40, ytr[i40])
    Xh = Xtr[order[~resv][3200:3240]]; kl6 = {}
    for dream in ['buf', 'resv']:
        net = copy.deepcopy(base); net._slow = {n: v.clone() for n, v in base._slow.items()}
        T, _ = Z.sleep_phase(net, 0, dream, 0.03, 150, np.random.default_rng(1), torch.Generator().manual_seed(1), Xtr, bs_, Xh)
        kl6[dream] = (Z.agreement(T, net, Xtr[bs_.idx], list(range(10)))[1], Z.agreement(T, net, Xh, list(range(10)))[1])
    chk(kl6['resv'][1] < 0.5 * kl6['buf'][1] and kl6['buf'][1] > kl6['buf'][0], 'Z6b leak detector (40-item sets): dreaming on the held-out set halves held-out KL; buffer arm shows a held-out gap',
        'buf (dream KL, held-out KL)', kl6['buf'], 'leak', kl6['resv'], 'v3 200-item case', (ar_leak, ar_buf, kl_leak, kr_buf))
    # Z7 mix dreams, checked from the images alone: synthetic data where training image i has channels (v, v^2, v^3), v = (i+1)/n.
    #     A dream lam*x_i + (1-lam)*x_j then identifies (i, j, lam) uniquely; recover them by brute force over buffer pairs
    nI = len(Xtr); v = (torch.arange(nI, dtype=torch.float64) + 1) / nI
    Xs = torch.stack([v, v ** 2, v ** 3], 1).float().view(-1, 3, 1, 1).expand(nI, 3, 32, 32)
    Xd = Z.make_dreams('mix', np.random.default_rng(5), Xs, buf, Xr).double()
    bidx = np.array(buf.idx); bl_ = np.array(buf.lab); vb = v[bidx].numpy(); okm = True; lams = []
    VI, VJ = vb[:, None], vb[None, :]
    for d in range(len(Xd)):
        m0, m1, m2 = (float(Xd[d, c, 0, 0]) for c in range(3))
        assert torch.allclose(Xd[d], Xd[d, :, :1, :1].expand(3, 32, 32))  # every pixel equal: a mix of constant images
        with np.errstate(divide='ignore', invalid='ignore'): lam = (m0 - VJ) / (VI - VJ)
        err = np.abs(lam * VI ** 2 + (1 - lam) * VJ ** 2 - m1) + np.abs(lam * VI ** 3 + (1 - lam) * VJ ** 3 - m2)
        err[~np.isfinite(err)] = np.inf; i, j = np.unravel_index(np.argmin(err), err.shape)
        hit = err[i, j] < 1e-5 and -1e-4 <= lam[i, j] <= 1 + 1e-4
        okm &= bool(hit) and bl_[i] != bl_[j]; lams.append(float(lam[i, j]) if lam[i, j] >= 0.5 else 1 - float(lam[i, j]))
    chk(okm and 0.70 < np.mean(lams) < 0.80, 'Z7 mix dreams (recovered from the images): every dream is an exact convex pair of different-label buffer items; '
        'max(lam, 1-lam) has mean ~0.75 as for lam ~ U(0,1)', round(float(np.mean(lams)), 3), len(lams))
    # Z8 noise dreams carry no data: identical with the training images zeroed
    chk(torch.equal(Z.make_dreams('noise', np.random.default_rng(3), Xtr, buf, Xr), Z.make_dreams('noise', np.random.default_rng(3), torch.zeros_like(Xtr), buf, Xr)), 'Z8 noise dreams independent of the data')
    # Z11 agreement() on a hand example: seen = {0,1}; teacher picks 0,1; student picks 0,0 (class 2 masked out)
    class Fix(torch.nn.Module):
        def __init__(s, L): super().__init__(); s.L = L
        def forward(s, x): return s.L[x[:, 0, 0, 0].long()]
    Lt = torch.zeros(2, 10); Ls = torch.zeros(2, 10); Lt[0, :3] = torch.tensor([2., 0, 5]); Lt[1, :3] = torch.tensor([0., 2, 9])
    Ls[0, :3] = torch.tensor([2., 0, 0]); Ls[1, :3] = torch.tensor([3., 2, 0]); Lt[:, 3:] = -20; Ls[:, 3:] = -20; Xi2 = torch.zeros(2, 3, 32, 32); Xi2[1] = 1
    ag, kl = Z.agreement(Fix(Lt), Fix(Ls), Xi2, [0, 1])
    pt = torch.log_softmax(Lt.double(), 1); ps = torch.log_softmax(Ls.double(), 1)
    chk(ag == 0.5 and abs(kl - float((pt.exp() * (pt - ps)).sum(1).mean())) < 1e-12, 'Z11 agreement metric (seen-masked argmax, full KL)', ag, kl)
    # Z13 width
    w = Z.run(0, D, width=64, max_steps=tb[0])
    chk(w['net'].head.in_features == 1024 and w['ends'][0]['test_logits'].shape == (len(yte), 10), 'Z13 run(width=64) builds and evaluates a 1024-d net')
    print('SELFTEST', 'ALL PASS' if ok else 'FAILURES'); return ok

if __name__ == '__main__': sys.exit(0 if main() else 1)
