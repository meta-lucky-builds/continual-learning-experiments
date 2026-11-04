"""Planted checks for ts5_probe.probe (audit_plan.md A1)."""
import os; os.environ.setdefault('CL_K', '100'); os.environ.setdefault('CL_NTASK', '20'); os.environ.setdefault('RESV', '20')
import numpy as np, torch, ts5_probe as P
g = torch.Generator().manual_seed(0); K = 100; n = 30
yf = torch.arange(K).repeat_interleave(n); yte = torch.arange(K).repeat_interleave(10)
def feats(y, sig): return torch.nn.functional.one_hot(y, K).float() * 3 + sig * torch.randn(len(y), K, generator=g)
F, Ft = feats(yf, 0.3), feats(yte, 0.3); a = np.arange(len(yf))
r1 = P.probe(F, Ft, yf, yte, a, list(range(K))); r2 = P.probe(torch.randn(len(yf), K, generator=g), torch.randn(len(yte), K, generator=g), yf, yte, a, list(range(K)))
r3 = P.probe(F, Ft, yf, yte, a, [3, 7, 11, 50, 99]); r4 = P.probe(torch.randn(len(yf), K, generator=g), torch.randn(len(yte), K, generator=g), yf, yte, a, [3, 7, 11, 50, 99])
# subset fit: only rows of classes in the list are used even if sel has more
half = a[(yf.numpy() % 2) == 0]
try: P.probe(F, Ft, yf, yte, half, [1, 2]); r5 = False
except AssertionError: r5 = True  # class 1 has no fitting rows -> must refuse
checks = {'P1 informative 100-way ~1': r1 > 0.99, 'P2 noise 100-way ~chance': r2 < 0.05, 'P3 informative 5-way ~1': r3 > 0.99,
          'P4 noise 5-way ~0.2': 0.05 < r4 < 0.45, 'P5 refuses a class with no fitting data': r5}
print({k: v for k, v in checks.items()}, r1, r2, r3, r4); assert all(checks.values()); print('PROBE SELFTEST PASS')
