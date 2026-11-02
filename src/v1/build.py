import json
rd = lambda f: open(f).read().rstrip('\n')
cells = []
def md(s): cells.append(dict(cell_type='markdown', metadata={}, source=s.strip('\n')))
def code(s): cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s.strip('\n')))

md(r'''
# Atlas ARCH-v2: first tests (CIFAR-100, online class-incremental)

**How to run:** Runtime → Change runtime type → **T4 GPU**. Then Runtime → **Run all**. Expect roughly 1 to 1.5 hours.
At the end a file `atlas_v2_results.json` downloads. Send that file back.

**What this notebook does, in order**
1. Runs every piece of analysis code against synthetic data with known answers, and against scikit-learn. If any check fails, the notebook stops before touching CIFAR.
2. Runs the audit's design tests (synthetic).
3. Runs a small smoke test of the full pipeline on CIFAR-100 (3 tasks, 15 classes), including a determinism check.
4. Runs the real experiment: 2 configs × 5 seeds, 20 tasks of 5 classes, single online pass. Results are saved after every run, so if Colab disconnects, re-run all and finished runs are skipped.
5. Prints the tables and the pre-registered comparisons, and downloads the JSON.

**Honesty notes (read before interpreting)**
- These are **fresh runs, not a reproduction of the T-S5 runs in ewt.md**. That code was never uploaded. The learner here is my re-implementation (a ~107k-parameter 3-block GroupNorm CNN, ER with reservoir buffer, SGD lr 0.03, batch 10 stream + 10 replay). Absolute numbers can differ from ewt's 24.9 / 30.56 / 35.3. Only comparisons *inside this notebook* are clean.
- All arms are scored by the **same LDA with the same pooled covariance**, so differences come only from the class means (the X3 lesson). A second score with the oracle covariance is reported as a robustness check.
- Class statistics are recorded at the **end of each class's own task** (all its stream items, in that task's final encoder). This is a declared simplification that applies equally to every arm.
''')
code(r'''
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   # must be set before CUDA starts (determinism)
import sys, json, time, gc, importlib, subprocess, platform
import numpy as np, torch
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print('torch', torch.__version__, '| numpy', np.__version__, '| python', platform.python_version(), '| GPU', GPU)
if GPU is None:
    print('WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU. On CPU this will take many hours.')
''')
md(r'''
## Pre-registration (written before any CIFAR result)

**Arms.** Every arm estimates the 100 class means in the final encoder f_T. Same LDA, same Σ for all.

| Arm | What it is |
|---|---|
| `O_full` | Ceiling: true means of all stream items in f_T (ewt's 35.3 analogue) |
| `B_only_buf` / `B_only_res` | Mean of the class's final-buffer items / its 10 (or 5) reserve-bridge items in f_T |
| `CH_cv_buf` / `CH_cv_res` | X2-style chained transport: each task, one global ridge map old→new fitted on the bridge, applied to every carried mean |
| `GR_prev_*` | Chained per-class regression (GREG): bridge mean of the class + slope × (carried mean − bridge mean in old coordinates) |
| `GR_stable_*` | One hop from the stable random chart r (exact means in r) straight into f_T |
| `GR_multi_*` | ARCH-v2's estimator: chained GREG with auxiliaries [r, f_prev]; r's block gets its own penalty (it can be dropped by CV) |
| `RC_LDA`, `ER_head` | Sanity references only: LDA in r with exact stats (ewt 28.2), the learner's own head (ewt ER 17.4) |

`_buf` = bridge is the replay buffer (items the learner trained on). `_res` = bridge is the held-out reserve (never trained on).

**Configs.** A: buffer M=2000 + reserve 20/class (10 bridge, 10 control). This spends extra bytes on the reserve. B: buffer M=1000 + reserve 10/class (5 + 5): same 2000 stored images as an M=2000 ER baseline.

**Test 1 (R²).** Out-of-fold within-class R² on reserve items of f_T from r, from f_{T-1}, f_{T-5}, f_1, from [r, f_{T-1}], and per hop.

**Comparisons and the rule.** Paired over 5 seeds; significant only if |mean diff| > 2·SE **and** all 5 seeds share the sign. Not significant is not a tie.
- **H1** A: `GR_multi_res` > `CH_cv_buf`. *Falsifier of ARCH-v2's migration estimator.*
- **H2** A: `GR_multi_res` > `GR_multi_buf` (reserve bridge beats buffer bridge). *Falsifier of the reserve-as-bridge choice.*
- **H3** A: `GR_multi_res` > `CH_cv_res` (per-class correction plus r beats a global map, same bridge).
- **H4** equal bytes: B `GR_multi_res` > A `CH_cv_buf`.
- **S1** A: `CH_cv_buf` > `B_only_buf` (does X2's direction replicate here?). **S2** A: `GR_stable_res` > `B_only_res` (does r help at all?).

**My predictions, with honest confidence.** R²(f_T | r) between 0.2 and 0.5, so r is a weak auxiliary (60%). Hop R² ≥ 0.8 (75%). S1 replicates (80%). H3 supported (65%). H1 supported but small, +0.5 to +3 points (55%). H2 a coin flip (50%): the reserve is unbiased but has fewer items per class than the buffer. H4 less likely (40%): halving the buffer may cost the learner more than the estimator gains.

**Decisions fixed on synthetic data before any CIFAR run:** GREG slope fitted on all bridge items (cross-fitting lost badly with 5 items/class); ridge penalty by 5-fold CV over a wide grid; r's block gets a separate penalty from {drop, 0.1, 0.3, 1}.
''')
md('## 1. Analysis library (pure numpy)')
code('%%writefile atlas_lib.py\n' + rd('atlas_lib.py'))
code('%%writefile atlas_analysis.py\n' + rd('atlas_analysis.py'))
code('%%writefile atlas_summary.py\n' + rd('atlas_summary.py'))
code('%%writefile atlas_torch.py\n' + rd('atlas_torch.py'))
code(r'''
sys.path.insert(0, '.')
import atlas_lib, atlas_analysis, atlas_summary, atlas_torch
for m in (atlas_lib, atlas_analysis, atlas_summary, atlas_torch): importlib.reload(m)
from atlas_analysis import run_analysis
from atlas_summary import summarize
from atlas_torch import load_cifar100, run_one
''')
md('## 2. Checks with known answers (the notebook stops here if anything fails)')
code('%%writefile test_lib.py\n' + rd('test_lib.py'))
code('%%writefile test_analysis.py\n' + rd('test_analysis.py'))
code('%%writefile sk_check.py\n' + rd('sk_check.py'))
code('%%writefile audit_tests.py\n' + rd('audit_tests.py'))
code(r'''
def run_check(script):
    r = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(f'===== {script} =====\n' + r.stdout + (r.stderr[-3000:] if r.returncode else ''))
    if r.returncode != 0:
        raise SystemExit(f'{script} FAILED: stop here and send me this output.')
for s in ('test_lib.py', 'test_analysis.py', 'sk_check.py', 'audit_tests.py'):
    run_check(s)
print('All library, end-to-end, scikit-learn and audit checks passed.')
''')
md('### GPU eigendecomposition (speed only). Used only if it gives the same ridge answers as numpy.')
code(r'''
def gpu_eigh(A):
    ev, V = torch.linalg.eigh(torch.from_numpy(np.ascontiguousarray(A)).to('cuda', torch.float64))
    return ev.cpu().numpy(), V.cpu().numpy()
atlas_lib.EIGH = np.linalg.eigh
if torch.cuda.is_available():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 1152)); Y = rng.normal(size=(2000, 128))
    t = time.time(); B_cpu = atlas_lib.ridge_cv(X, Y, np.random.default_rng(1), split=1024)[0]; t_cpu = time.time() - t
    atlas_lib.EIGH = gpu_eigh
    t = time.time(); B_gpu = atlas_lib.ridge_cv(X, Y, np.random.default_rng(1), split=1024)[0]; t_gpu = time.time() - t
    same = np.allclose(B_cpu, B_gpu, rtol=1e-6, atol=1e-9 * np.abs(B_cpu).max())
    if not (same and t_gpu < t_cpu):
        atlas_lib.EIGH = np.linalg.eigh
    print(f'cpu {t_cpu:.1f}s, gpu {t_gpu:.1f}s, identical={same} -> using', 'GPU' if atlas_lib.EIGH is gpu_eigh else 'CPU')
''')
md('## 3. Data and smoke test (small, fast; also checks determinism)')
code(r'''
data = load_cifar100()
SMOKE = dict(n_classes=15, n_tasks=3, M=200, reserve_per_class=10, lr=0.03, batch=10, stream_per_class=60)
D = run_one(0, SMOKE, data)
o = run_analysis(D, 0)
assert o['err_all']['O_full'] == 0
assert all(np.isfinite(v) and 0 <= v <= 100 for v in o['acc'].values())
assert all(np.isfinite(v) for v in o['err_all'].values())
D2 = run_one(0, SMOKE, data, log=lambda *a: None)
dmax = float(np.abs(D['F'][-1] - D2['F'][-1]).max())
print('determinism: max feature difference between two identical runs =', dmax,
      '(OK)' if dmax < 1e-3 else '(WARNING: not bit-reproducible on this GPU; results are still valid as a 5-seed statistic)')
print('smoke accuracies:', {k: round(v, 1) for k, v in o['acc'].items()})
del D, D2; gc.collect()
''')
md('## 4. The experiment (2 configs × 5 seeds). Saved after every run; re-running skips finished runs.')
code(r'''
SEEDS = [0, 1, 2, 3, 4]
BASE = dict(n_classes=100, n_tasks=20, lr=0.03, batch=10, stream_per_class=None)
CONFIGS = {'A': dict(BASE, M=2000, reserve_per_class=20),
           'B': dict(BASE, M=1000, reserve_per_class=10)}
RESULTS_PATH = 'atlas_v2_results_partial.json'
results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
t_all = time.time()
for cname, cfg in CONFIGS.items():
    results.setdefault(cname, {})
    for seed in SEEDS:
        if str(seed) in results[cname]:
            print(f'{cname} seed {seed}: already done, skipping'); continue
        print(f'{cname} seed {seed}: running ...')
        D = run_one(seed, cfg, data)
        out = run_analysis(D, seed)
        out['config'] = cfg
        results[cname][str(seed)] = out
        del D; gc.collect()
        json.dump(results, open(RESULTS_PATH, 'w'))
        print(f'   analysis {out["seconds"]:.0f}s | O_full {out["acc"]["O_full"]:.1f} | CH_cv_buf {out["acc"]["CH_cv_buf"]:.1f}'
              f' | GR_multi_res {out["acc"]["GR_multi_res"]:.1f} | RC_LDA {out["acc"]["RC_LDA"]:.1f} | ER_head {out["acc"]["ER_head"]:.1f}'
              f' | elapsed {(time.time()-t_all)/60:.0f} min')
''')
md('## 5. Results and the pre-registered comparisons')
code(r'''
summary = summarize(results)
final = dict(meta=dict(torch=torch.__version__, numpy=np.__version__, gpu=GPU, seeds=SEEDS, configs=CONFIGS,
                       eigh='gpu' if atlas_lib.EIGH is not np.linalg.eigh else 'cpu', finished=time.strftime('%Y-%m-%d %H:%M:%S')),
             summary=summary, runs=results)
json.dump(final, open('atlas_v2_results.json', 'w'), indent=1)
try:
    from google.colab import files; files.download('atlas_v2_results.json')
except Exception as e:
    print('Download it from the Files panel on the left (atlas_v2_results.json).', e)
''')
nb = dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(provenance=[], gpuType='T4'),
          kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python')),
          nbformat=4, nbformat_minor=0)
for c in nb['cells']:
    c['source'] = [l + '\n' for l in c['source'].split('\n')]
    c['source'][-1] = c['source'][-1].rstrip('\n')
json.dump(nb, open('atlas_v2_tests.ipynb', 'w'), indent=1)
print(len(cells), 'cells')
