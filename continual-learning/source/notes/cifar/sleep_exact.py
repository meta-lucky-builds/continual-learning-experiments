"""A3 (audit_plan.md): exact-transfer Sleep control. Same as sleep.py run(sleep=1, rank=0), except that each sleep sets the
expert to delta = teacher - S exactly (no distillation), then merges as usual. Logs max |logit(merged) - logit(teacher)|.
Run: RESV=20 LR=0.03 python3 sleep_exact.py <seed> <outdir>"""
import sys, os, json, copy, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C, sleep as SL

def exact_phase(net, rank, dream, dlr, dep, rng, gen, Xtr, buf, Xr, bs=64, log=None):
    teacher = copy.deepcopy(net)
    for p in teacher.parameters(): p.requires_grad_(False)
    S = {n: net._slow[n].clone() for n, _ in net.named_parameters()}; ts = time.time()
    with torch.no_grad():
        for n, p in net.named_parameters():
            P = S[n] + (p.detach() - S[n]); p.copy_(P); net._slow[n] = P.detach().clone()
        Xb = Xtr[buf.idx]
        md = {k: float((C.chunked(net, X) - C.chunked(teacher, X)).abs().max()) for k, X in [('buf', Xb), ('resv', Xr)]}
    return teacher, {'maxdiff': md, 'student_fwdbwd_imgs': 0, 'teacher_fwd_imgs': 0, 'sleep_secs': time.time() - ts}

SL.sleep_phase = exact_phase
if __name__ == '__main__':
    s, outdir = int(sys.argv[1]), sys.argv[2]; torch.set_num_threads(1); D = C.load()
    assert C.K == 10 and C.RESV == 20 and C.MEM == 500 and C.LR == 0.03
    kw = {'sleep': 1, 'rank': 0, 'dream': 'buf'}
    r = SL.run(s, D, **kw); C.save(os.path.join(outdir, f'slp_exact_{s}.npz'), r)
    sm = C.summarize('slp_exact', s, r, D[3]); sm['cfg'] = kw; sm['sleeps'] = r['sleeps']; sm['resv'] = C.RESV; sm['mem'] = C.MEM
    print(json.dumps(sm), flush=True)
