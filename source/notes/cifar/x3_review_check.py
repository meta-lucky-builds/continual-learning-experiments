"""Review checks for x3 (one seed). Run from the cifar folder."""
import sys, os, json
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import cifar as C
from x2 import lda_scores, class_means, within_scatter, ridge, transport, mknet
from x3 import fit_back, acc_within, keys

s = int(sys.argv[1]); torch.set_num_threads(4); K, T = C.K, C.NTASK - 1; d = 16 * C.W
Xtr, ytr, Xte, yte = C.load(); tasks, order, tid, resv = C.make_stream(ytr, s); bl = C.batches(order, tid, resv)
E = torch.load(f'results/ts5_probe/enc_{s}.pt', weights_only=False)
nets = [mknet(v['state']) for v in E['versions']] + [mknet(E['final'])]
buf = C.Reservoir(C.MEM, np.random.default_rng(s * 1000 + 3)); Bt = []
for i, (t, bi) in enumerate(bl):
    if buf.idx: buf.sample(C.RB)
    buf.add(torch.as_tensor(bi), ytr[bi])
    if i + 1 == len(bl) or bl[i + 1][0] != t: Bt.append((np.array(buf.idx), torch.tensor(buf.lab)))
feats = lambda net, idx: C.chunked(net.feats, Xtr[idx]).double()
idxF, labF = Bt[T]
# leakage checks
resv_idx = set(order[resv].tolist()); print('final buffer ∩ reserve:', len(resv_idx & set(idxF.tolist())))
cnt = torch.bincount(labF, minlength=K); print('final buffer per-class min/max', int(cnt.min()), int(cnt.max()))
prevn = [int(sum(cnt[c] for tt in tasks[:u + 1] for c in tt)) for u in range(1, C.NTASK)]
print('keys(all) per u:', prevn, 'mean', np.mean(prevn))
# survival check: are task<u items in final buffer also in buffer at end of task u-1?
bad = 0
for u in range(1, C.NTASK):
    old_set = set(Bt[u - 1][0].tolist()); prevcls = set(c for tt in tasks[:u] for c in tt)
    bad += sum(1 for i, l in zip(idxF.tolist(), labF.tolist()) if l in prevcls and i not in old_set)
print('final-buffer items of tasks<u absent from buffer at end of task u-1:', bad)
src_mu, src_cov, src_mu_fin, src_cov_fin = {}, {}, {}, {}
fin = nets[T]
for u in range(1, C.NTASK):
    bis = [bi for t, bi in bl if t == u]; ii = np.concatenate(bis)
    Fu = feats(nets[u - 1], ii); yu = ytr[ii]
    src_mu[u] = class_means(Fu, yu, K); src_cov[u] = within_scatter(Fu, yu) / (len(yu) - 5)
    Fuf = feats(fin, ii); src_mu_fin[u] = class_means(Fuf, yu, K); src_cov_fin[u] = within_scatter(Fuf, yu) / (len(yu) - 5)
Ft = C.chunked(fin.feats, Xte).double(); FbF = feats(fin, idxF)
_, _, cov_buf = C.lda_from(FbF.float(), labF).stats()
idx0, lab0 = Bt[0]; F0 = feats(nets[0], idx0); m0 = class_means(F0, lab0, K)
mu = torch.full((K, d), float('nan'), dtype=torch.float64); mu[tasks[0]] = m0[tasks[0]]
for t in range(T):
    mu[tasks[t + 1]] = src_mu[t + 1][tasks[t + 1]]; idx, lab = Bt[t + 1]
    mu, _, _ = transport(mu, feats(nets[t], idx), feats(nets[t + 1], idx), lab, 'cv', K)
print('TRANS mu NaN:', bool(torch.isnan(mu).any()))
J = {r['u']: r for r in json.load(open(f'results/x3/x3_{s}.json'))['rows']}
rows = []
for u in range(1, C.NTASK):
    cs = tasks[u]; mt = torch.isin(yte, torch.tensor(cs)); yt = yte[mt]; Ff = Ft[mt]; old = nets[u - 1]
    Fo = C.chunked(old.feats, Xte[mt]).double(); Fb_old = feats(old, idxF)
    v = E['versions'][u - 1]; l = C.StreamLDA(d); l.S1, l.n, l.XX = v['S1'], v['n'], v['XX']; _, _, vcov = l.stats()  # v_{u-1}'s own pooled cov (all classes it held)
    sel = torch.isin(labF, torch.tensor(cs)); cov_buf5 = within_scatter(FbF[sel], labF[sel]) / max(int(sel.sum()) - 5, 1)
    prev = sorted(c for tt in tasks[:u + 1] for c in tt); kp = keys(labF, prev, 0)
    mp = fit_back(FbF[kp], Fb_old[kp])
    # unpenalized intercept variant for k=1
    kp1 = keys(labF, prev, 1); X1, Y1 = FbF[kp1], Fb_old[kp1]; mx, my = X1.mean(0), Y1.mean(0); A1 = ridge(X1 - mx, Y1 - my)
    r = dict(u=u,
        OLD=acc_within(lda_scores(Fo, src_mu[u][cs], src_cov[u]), yt, cs),
        OLD_vcov=acc_within(lda_scores(Fo, src_mu[u][cs], vcov), yt, cs),
        REACT_all=acc_within(lda_scores(mp(Ff), src_mu[u][cs], src_cov[u]), yt, cs),
        REACT_all_vcov=acc_within(lda_scores(mp(Ff), src_mu[u][cs], vcov), yt, cs),
        REACT_1=acc_within(lda_scores(fit_back(X1, Y1)(Ff), src_mu[u][cs], src_cov[u]), yt, cs),
        REACT_1_freeint=acc_within(lda_scores((Ff - mx) @ A1.T + my, src_mu[u][cs], src_cov[u]), yt, cs),
        TRANS=acc_within(lda_scores(Ff, mu[cs], cov_buf), yt, cs),
        TRANS_buf5cov=acc_within(lda_scores(Ff, mu[cs], cov_buf5), yt, cs),
        TRANS_fin5cov=acc_within(lda_scores(Ff, mu[cs], src_cov_fin[u]), yt, cs),
        FIN_oracle=acc_within(lda_scores(Ff, src_mu_fin[u][cs], src_cov_fin[u]), yt, cs),
        FIN_oracle_bufcov=acc_within(lda_scores(Ff, src_mu_fin[u][cs], cov_buf), yt, cs))
    r['repro_maxdiff'] = max(abs(r[a] - J[u][a]) for a in ['OLD', 'REACT_all', 'REACT_1', 'TRANS'])
    rows.append(r); print(json.dumps(r), flush=True)
print('MEANS', json.dumps({k: round(100 * float(np.mean([r[k] for r in rows])), 2) for k in rows[0] if k != 'u'}))
json.dump(rows, open(f'/tmp/claude-0/-home-user/da7a0050-f410-5db2-8360-29a4b9e95c98/scratchpad/chk3_{s}.json', 'w'))
