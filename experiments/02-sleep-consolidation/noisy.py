"""Noisy repeats: every sighting uses a noisy key; queries use the clean key. count_gate and replay are given
item identity (an advantage a real system would not have); slot rules use only per-slot address mass."""
import sys, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '/mnt/project-files/memory-consolidation-test')
from pkmsleep import run, summarize
CONDS = {'single': {}, 'single2N': {}, 'oracle': {}, 'count_gate': {}, 'copy_sel': {'tau': 0.4}, 'copy_soft': {'tau': 3.0}, 'replay': {}, 'replay_all': {}}
SEEDS = [0, 1, 2, 3, 4]
def job(a):
    nz, c, sd = a
    M, C = run(c, sd, n_dis=40, key_noise=nz, **CONDS[c]); s = summarize(M)
    return nz, c, sd, {k: (float(v) if not isinstance(v, list) else v) for k, v in s.items()}
if __name__ == '__main__':
    jobs = [(nz, c, sd) for nz in [0.3, 0.6] for c in CONDS for sd in SEEDS]
    with Pool(4) as p: res = p.map(job, jobs)
    out = {}
    for nz, c, sd, s in res: out.setdefault(str(nz), {}).setdefault(c, {})[sd] = s
    json.dump(out, open('/mnt/project-files/memory-consolidation-test/results_noisy.json', 'w'), indent=1)
    for nz in out:
        print('== key noise', nz)
        for c in CONDS:
            f = np.array([out[nz][c][s]['final'] for s in SEEDS]); a = np.array([out[nz][c][s]['acq'] for s in SEEDS])
            print(f"{c:11s} final {f.mean():.3f}±{f.std(ddof=1):.3f}  acq {a.mean():.3f}")
