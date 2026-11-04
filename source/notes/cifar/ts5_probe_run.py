"""A1 re-run (audit_plan.md): the T-S5 erp learner exactly as run_ts5.sh ran it, plus its frozen encoders, their
streaming-LDA states and the final learner, saved to results/ts5_probe/enc_<seed>.pt.
Run: CL_K=100 CL_NTASK=20 CIFAR_DIR=data100 RESV=20 LR=0.03 MEM=2000 python3 ts5_probe_run.py <seed>"""
import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C, cifar2 as C2
assert C.K == 100 and C.NTASK == 20 and C.MEM == 2000 and C.RESV == 20 and C.LR == 0.03
s = int(sys.argv[1]); D = C.load(); kw = {'mem': 2000, 'promote': 1}
r = C2.run(s, D, **kw); C2.save_run(f'results/ts5/erp_{s}.npz', r)
sm = C.summarize('erp', s, r, D[3]); sm['cfg'] = kw; sm['stats'] = {k: (v if not isinstance(v, list) else [len(v), float(np.mean(v)) if v else 0]) for k, v in r['stats'].items()}
print(json.dumps(sm), flush=True)
torch.save({'versions': [{'t': v['t'], 'state': v['net'].state_dict(), 'S1': v['lda'].S1, 'n': v['lda'].n, 'XX': v['lda'].XX} for v in r['versions']],
            'final': r['net'].state_dict(), 'seed': s}, f'results/ts5_probe/enc_{s}.pt')
print('SAVED', flush=True)
