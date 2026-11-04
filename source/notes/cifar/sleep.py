"""Sleep-inspired CIFAR tests T-S1-T-S4 (plan: sleep_plan.md, pre-declared 2026-10-29). Built on cifar.py, whose ER is the reference.

Learner = cifar.py's online ER, plus an optional "sleep" at the end of tasks 2..5 (t = 1..4), a CNN analogue of the
consolidation stage of Behrouz et al. (arXiv 2606.03979):
  S        slow weights: the network at the end of task 1, plus every earlier consolidated expert (frozen during sleep)
  F        fast update: what plain ER learned since the last sleep (the current net = S + F)
  teacher  the current net S + F
  student  S + new expert. The expert is a rank-r additive update B@A on every conv/linear weight (r = 0 means full rank)
           plus full deltas on biases and GroupNorm affines; B = 0 at start, so the student starts exactly at S (F reset)
  distill  only the expert's parameters, by KL(teacher || student) on "dreams", SGD momentum 0.9 (DLR), DEP epochs of len(buffer)
           (Adam was dropped before any result: self-test Z5 showed it drifts away from an exact optimum, since it normalises ~0 gradients)
  update   net := S + expert (merged), i.e. the fast update is replaced by its consolidated version; ER then continues
Dreams: buf (buffer images), mix (pairs of buffer images with different labels, lambda ~ U(0,1), fresh per epoch),
        noise (iid N(0,1) in the normalised input space, fresh per epoch), resv (reserve images: LEAK, self-tests only).
The optimiser is plain SGD (no state), so replacing the weights leaves nothing stale. Sleep uses its own RNGs, so the
replay sampling sequence is identical to ER's: arms differ only through the weights.

Usage: python sleep.py run <name> <seeds> <outdir> key=value ...  keys: sleep (0/1) rank dream dlr dep width
       python sleep.py selftest                                     (LR and RESV env as in cifar.py)
"""
import os, sys, json, time, copy
import numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C

DREAMS = ('buf', 'mix', 'noise', 'resv')

def is_lowrank(name, p): return p.dim() >= 2  # conv and linear weights; biases and GroupNorm affines get full deltas

class Expert:
    """Additive update on every parameter of a template net. rank r > 0: weights get B@A (B = 0, A ~ N(0, 1/fan_in))."""
    def __init__(s, net, rank, gen):
        s.parts = {}
        for n, p in net.named_parameters():
            if rank > 0 and is_lowrank(n, p):
                fan = p[0].numel(); r = min(rank, p.shape[0], fan)
                s.parts[n] = ('lr', torch.zeros(p.shape[0], r, requires_grad=True), (torch.randn(r, fan, generator=gen) / fan ** 0.5).requires_grad_(True))
            else: s.parts[n] = ('full', torch.zeros_like(p, requires_grad=True))
    def params(s): return [t for v in s.parts.values() for t in v[1:]]
    def delta(s, n, shape):
        v = s.parts[n]
        return (v[1] @ v[2]).view(shape) if v[0] == 'lr' else v[1]
    def nparams(s): return sum(t.numel() for t in s.params())

def student_params(S, ex): return {n: S[n] + ex.delta(n, S[n].shape) for n in S}

def make_dreams(kind, rng, Xtr, buf, Xr):
    n = len(buf.idx)
    if kind == 'buf': return Xtr[buf.idx][torch.from_numpy(rng.permutation(n))]
    if kind == 'resv': return Xr[torch.from_numpy(rng.permutation(len(Xr)))]
    if kind == 'noise': return torch.from_numpy(rng.standard_normal((n,) + tuple(Xtr.shape[1:])).astype(np.float32))
    if kind == 'mix':
        lab = np.array(buf.lab); i = rng.integers(n, size=n); j = rng.integers(n, size=n)
        bad = lab[i] == lab[j]
        while bad.any(): j[bad] = rng.integers(n, size=int(bad.sum())); bad = lab[i] == lab[j]
        lam = torch.from_numpy(rng.uniform(0, 1, size=n).astype(np.float32)).view(-1, 1, 1, 1)
        bi, bj = torch.tensor([buf.idx[a] for a in i]), torch.tensor([buf.idx[a] for a in j])
        return lam * Xtr[bi] + (1 - lam) * Xtr[bj]
    raise ValueError(kind)

def sleep_phase(net, rank, dream, dlr, dep, rng, gen, Xtr, buf, Xr, bs=64, log=None):
    """Consolidate the current net (teacher) into S + new expert, then set net := merged student. Returns stats."""
    teacher = copy.deepcopy(net)
    for p in teacher.parameters(): p.requires_grad_(False)
    S = {n: net._slow[n].clone() for n, _ in net.named_parameters()}  # slow weights = merged net after the last sleep (see run)
    ex = Expert(net, rank, gen); opt = torch.optim.SGD(ex.params(), lr=dlr, momentum=0.9); nfb = nt = 0; losses = []; ts = time.time()
    for ep in range(dep):
        X = make_dreams(dream, rng, Xtr, buf, Xr)
        for a in range(0, len(X), bs):
            xb = X[a:a + bs]
            with torch.no_grad(): pt = torch.log_softmax(teacher(xb), 1)
            ls = torch.log_softmax(torch.func.functional_call(net, student_params(S, ex), (xb,)), 1)
            loss = (pt.exp() * (pt - ls)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step(); nfb += len(xb); nt += len(xb)
            if a == 0: losses.append(loss.item())
    with torch.no_grad():
        P = student_params(S, ex)
        for n, p in net.named_parameters(): p.copy_(P[n]); net._slow[n] = P[n].detach().clone()
    ranks = {n: (v[1].shape[1] if v[0] == 'lr' else 'full') for n, v in ex.parts.items() if v[0] == 'lr' or v[1].dim() >= 2}
    return teacher, {'kl_first_batch_by_epoch': losses, 'expert_params': ex.nparams(), 'expert_ranks': ranks,
                     'student_fwdbwd_imgs': nfb, 'teacher_fwd_imgs': nt, 'sleep_secs': time.time() - ts}

def agreement(teacher, student, X, seen):
    """Teacher-student agreement on X: argmax match over seen classes, KL(teacher || student) over all classes."""
    with torch.no_grad():
        Lt, Ls = C.chunked(teacher, X), C.chunked(student, X)
    m = torch.full((C.K,), float('-inf')); m[seen] = 0
    agree = float(((Lt + m).argmax(1) == (Ls + m).argmax(1)).float().mean())
    pt = torch.log_softmax(Lt.double(), 1); ps = torch.log_softmax(Ls.double(), 1)
    return agree, float((pt.exp() * (pt - ps)).sum(1).mean())

def run(seed, D, sleep=0, rank=16, dream='buf', dlr=0.03, dep=10, width=C.W, max_steps=None, eval_tasks=True, flip_step=None, sleep_from=1):
    Xtr, ytr, Xte, yte = D; tasks, order, tid, resv = C.make_stream(ytr, seed); bl = C.batches(order, tid, resv)
    assert C.LR is not None and dream in DREAMS
    net = C.make_net(seed * 1000 + 1, width); opt = torch.optim.SGD(net.parameters(), lr=C.LR)
    rng = np.random.default_rng(seed * 1000 + 3); buf = C.Reservoir(C.MEM, rng)
    rng_d = np.random.default_rng(seed * 1000 + 21); gen = torch.Generator().manual_seed(seed * 1000 + 22)
    ridx = order[resv]; ry = ytr[ridx]; Xr = Xtr[ridx]
    preq, preq_lp, ylearn, ends, sleeps, nfb, t0 = [], [], [], [], [], 0, time.time()  # nfb: ER images only; sleep compute is in sleeps
    nsteps = len(bl) if max_steps is None else max_steps
    for step in range(nsteps):
        t, bi = bl[step]; xb = Xtr[bi]; yb = ytr[bi].clone()
        if step == flip_step: yb = (yb + 1) % C.K
        if buf.idx: rj, ryb = buf.sample(C.RB); x = torch.cat([xb, Xtr[rj]])
        else: ryb = None; x = xb
        out = net(x); ls = out[:len(bi)]; nfb += len(x)
        lp = torch.log_softmax(ls.detach().double(), 1); preq.append(lp.exp().float()); preq_lp.append(lp[torch.arange(len(yb)), yb])
        loss = Fn.cross_entropy(ls, yb) + (Fn.cross_entropy(out[len(bi):], ryb) if ryb is not None else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        buf.add(torch.as_tensor(bi), yb); ylearn.append(yb)
        last_of_task = step + 1 == len(bl) or bl[step + 1][0] != t
        if not last_of_task: continue
        if sleep and t == sleep_from - 1:  # slow weights start as the net at the end of task sleep_from (task 1 by default)
            net._slow = {n: p.detach().clone() for n, p in net.named_parameters()}
        if sleep and t >= sleep_from:
            seen = C.seen_of(tasks, t)
            teacher, st = sleep_phase(net, rank, dream, dlr, dep, rng_d, gen, Xtr, buf, Xr)
            st['t'] = t
            st['agree_buf'], st['kl_buf'] = agreement(teacher, net, Xtr[buf.idx], seen)
            sm = torch.isin(ry, torch.tensor(seen)); st['agree_resv'], st['kl_resv'] = agreement(teacher, net, Xr[sm], seen)
            tm = torch.isin(yte, torch.tensor(seen)); st['agree_test'], st['kl_test'] = agreement(teacher, net, Xte[tm], seen)
            if eval_tasks:
                te = C.eval_learner(teacher, None, buf, Xtr, Xte, yte, Xr, ry, tasks, t)
                ends_T = {f'T_{k}': v for k, v in te.items() if k in ('test_logits', 'resv_logits', 'b_buf')}
            sleeps.append(st)
        if eval_tasks:
            e = C.eval_learner(net, None, buf, Xtr, Xte, yte, Xr, ry, tasks, t)
            if sleep and t >= sleep_from: e.update(ends_T)
            ends.append(e)
    return {'tasks': tasks, 'preq': torch.cat(preq).numpy(), 'preq_lp': torch.cat(preq_lp).numpy(), 'ylearn': torch.cat(ylearn).numpy(), 'ends': ends,
            'resv_y': ry.numpy(), 'fwdbwd_imgs': nfb, 'secs': time.time() - t0, 'sleeps': sleeps, 'net': net, 'buf_idx': list(buf.idx)}

if __name__ == '__main__':
    if sys.argv[1] == 'selftest':
        import sleep_selftest; sys.exit(0 if sleep_selftest.main() else 1)
    _, name, seeds, outdir = sys.argv[1:5]; kw = {}
    for a in sys.argv[5:]:
        k, v = a.split('='); kw[k] = v if k == 'dream' else (float(v) if k == 'dlr' else int(v))
    assert kw.get('dream', 'buf') != 'resv', "the 'resv' dream leaks the reserve: self-tests only"
    torch.set_num_threads(1); D = C.load()
    for s in [int(x) for x in seeds.split(',')]:
        r = run(s, D, **kw); C.save(os.path.join(outdir, f'{name}_{s}.npz'), r)
        sm = C.summarize(name, s, r, D[3]); sm['cfg'] = kw; sm['sleeps'] = r['sleeps']; sm['resv'] = C.RESV; sm['mem'] = C.MEM
        print(json.dumps(sm), flush=True)
