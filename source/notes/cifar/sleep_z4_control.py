"""Z4 control (review item B1): the rank test at rtol=1e-3 must be able to FAIL. Same setup as sleep_selftest Z4, but a
full-rank expert (rank=0): the measured rank must exceed 4 on the body convs. Run: RESV=20 LR=0.03 python sleep_z4_control.py"""
import sys, os, copy, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C, sleep as Z
torch.set_num_threads(1); D = C.load(); Xtr, ytr = D[0], D[1]; tasks, order, tid, resv = C.make_stream(ytr, 0)
buf = C.Reservoir(200, np.random.default_rng(0)); idx = torch.as_tensor(order[~resv][:200]); buf.add(idx, ytr[idx]); Xr = Xtr[order[resv][:100]]
out = {}
for rank in (4, 0):
    net = C.make_net(7); net._slow = {n: p.detach().clone() for n, p in net.named_parameters()}
    with torch.no_grad():
        g = torch.Generator().manual_seed(8)
        for p in net.parameters(): p.add_(0.05 * torch.randn(p.shape, generator=g) * (p.abs().mean() + 1e-3))
    S0 = copy.deepcopy(net._slow); Z.sleep_phase(net, rank, 'buf', 0.03, 5, np.random.default_rng(1), torch.Generator().manual_seed(1), Xtr, buf, Xr)
    out[rank] = {n: int(torch.linalg.matrix_rank((p.detach() - S0[n]).reshape(p.shape[0], -1).double(), rtol=1e-3)) for n, p in net.named_parameters() if p.dim() >= 2}
ok = max(out[4].values()) == 4 and all(out[0][n] > 4 for n in ('body.0.weight', 'body.4.weight', 'body.8.weight'))
print('PASS' if ok else 'FAIL', 'Z4 control: rank-4 measured', out[4], '; full-rank measured', out[0]); sys.exit(0 if ok else 1)
