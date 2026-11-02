"""Fast/slow product-key memory with sleep consolidation: a toy test.

Model (follows FwPKM / PKM / delta-rule math):
  address(q): split q into halves, score each half against sqrt(N) sub-keys,
              take top-k per half, combine into k*k candidates, keep top-k
              slots, weights w = softmax(score / temp).
  read:       y = sum_i w_i * (S_i + F_i)            (slow + fast, shared addresses)
  fast write: NLMS / Kaczmarz one-step rewrite on the residual of the combined
              read: F_i += w_i * (v - y) / ||w||^2   -> y becomes exactly v.
  sleep:      at each task boundary the fast store F is cleared (volatile, like
              hippocampus / per-document TTT reset). Before clearing, a
              consolidation rule may move content into S. Sleep rules:
                none      : nothing moves (floor)
                copy_all  : S += F for all slots (provably identical to one
                            memory with no sleep; a sanity check)
                copy_sel  : S += F only on slots whose accumulated address
                            weight this session >= tau (counter per slot only)
                replay    : for keys seen >= 2 times this session (needs a buffer
                            of this session's keys), NLMS-write the fast read
                            into S (CLS-style replay / distillation)
Baselines without hierarchy:
  single    : one memory, NLMS writes on every occurrence, never cleared
  single2N  : same with ~2x slots (parameter-matched to S+F)
  oracle    : single memory that writes only target items (upper bound on
              filtering distractors)
  count_gate: single memory that skips the first sighting of a key in a
              session (needs the same per-session key record as `replay`;
              tests whether the fast store is only acting as a counter)
  copy_soft : S += min(1, mass/tau) * F (soft version of copy_sel)
  replay_all: like replay, but replays every stored sighting of a repeated item

Stream: T tasks. Each task has n_tgt target items, each shown `reps` times,
plus n_dis distractor items shown once, in random order. Only targets are
ever queried. After each task (after sleep) every target learned so far is
queried; accuracy = nearest-neighbour decode among all target values.
"""
import numpy as np, json, sys, time


class PKM:
    def __init__(self, rng, d=32, n_sub=64, k=8, temp=0.05, sub=None):
        self.d, self.n_sub, self.k, self.temp = d, n_sub, k, temp
        h = d // 2
        if sub is None:
            s1 = rng.standard_normal((n_sub, h)); s2 = rng.standard_normal((n_sub, h))
            s1 /= np.linalg.norm(s1, axis=1, keepdims=True); s2 /= np.linalg.norm(s2, axis=1, keepdims=True)
            sub = (s1, s2)
        self.s1, self.s2 = sub
        self.N = n_sub * n_sub

    def address(self, Q):
        """Q: (B,d) unit-ish queries -> idx (B,k), w (B,k)."""
        h = self.d // 2
        a = Q[:, :h] @ self.s1.T; b = Q[:, h:] @ self.s2.T          # (B,n_sub)
        k = self.k
        ia = np.argpartition(-a, k - 1, axis=1)[:, :k]; ib = np.argpartition(-b, k - 1, axis=1)[:, :k]
        sa = np.take_along_axis(a, ia, 1); sb = np.take_along_axis(b, ib, 1)
        comb = sa[:, :, None] + sb[:, None, :]                          # (B,k,k)
        cidx = ia[:, :, None] * self.n_sub + ib[:, None, :]
        comb = comb.reshape(len(Q), -1); cidx = cidx.reshape(len(Q), -1)
        top = np.argpartition(-comb, k - 1, axis=1)[:, :k]
        sc = np.take_along_axis(comb, top, 1); idx = np.take_along_axis(cidx, top, 1)
        sc = sc / self.temp; sc -= sc.max(1, keepdims=True)
        w = np.exp(sc); w /= w.sum(1, keepdims=True)
        return idx, w


def make_stream(rng, T, n_tgt, reps, n_dis, d, dv):
    keys, vals, tasks = [], [], []
    tgt_ids = []
    nid = 0
    for t in range(T):
        ev = []
        for _ in range(n_tgt):
            ev += [nid] * reps; tgt_ids.append((nid, t)); nid += 1
        for _ in range(n_dis):
            ev.append(nid); nid += 1
        ev = np.array(ev); rng.shuffle(ev); tasks.append(ev)
    K = rng.standard_normal((nid, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.standard_normal((nid, dv)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    is_tgt = np.zeros(nid, bool); task_of = np.full(nid, -1)
    for i, t in tgt_ids:
        is_tgt[i] = True; task_of[i] = t
    return K, V, tasks, is_tgt, task_of


def run(cond, seed, T=100, n_tgt=10, reps=3, n_dis=40, d=32, dv=32, n_sub=64, k=8,
        temp=0.05, tau=1.5, check=False, key_noise=0.0):
    rng = np.random.default_rng(seed)
    K, Vt, tasks, is_tgt, task_of = make_stream(rng, T, n_tgt, reps, n_dis, d, dv)
    n_sub_eff = int(round(n_sub * np.sqrt(2))) if cond == 'single2N' else n_sub
    pk = PKM(np.random.default_rng(seed + 10_000), d, n_sub_eff, k, temp)
    idx_all, w_all = pk.address(K)                                    # clean addresses (used for queries)
    nrng = np.random.default_rng(seed + 20_000)
    S = np.zeros((pk.N, dv)); F = np.zeros((pk.N, dv))
    hier = cond in ('none', 'copy_all', 'copy_sel', 'copy_soft', 'replay', 'replay_all')
    tgt_list = np.where(is_tgt)[0]
    tgt_vals = Vt[tgt_list] / np.linalg.norm(Vt[tgt_list], axis=1, keepdims=True)
    M = np.full((T, T), np.nan)                                       # M[t, j] acc on task j after task t
    C = np.full((T, T), np.nan)                                       # mean cosine
    for t, ev in enumerate(tasks):
        mass = np.zeros(pk.N); seen = {}; last_addr = {}
        for i in ev:
            if cond == 'oracle' and not is_tgt[i]:
                continue
            if cond == 'count_gate' and seen.get(i, 0) == 0:   # skip first sighting this session
                seen[i] = 1
                continue
            if key_noise > 0:   # each sighting is a noisy version of the item's key
                kn = K[i] + key_noise * nrng.standard_normal(d) / np.sqrt(d); kn /= np.linalg.norm(kn)
                ii, ww = pk.address(kn[None]); idx, w = ii[0], ww[0]
            else:
                idx, w = idx_all[i], w_all[i]
            y = w @ (S[idx] + F[idx])
            r = (Vt[i] - y) / (w @ w)
            if hier:
                F[idx] += w[:, None] * r[None, :]
            else:
                S[idx] += w[:, None] * r[None, :]
            if check:
                y2 = w @ (S[idx] + F[idx]); assert np.allclose(y2, Vt[i], atol=1e-8)
            mass[idx] += w; seen[i] = seen.get(i, 0) + 1
            if cond == 'replay': last_addr[i] = (idx, w)
            if cond == 'replay_all': last_addr.setdefault(i, []).append((idx, w))
        # ---- sleep ----
        if hier:
            if cond == 'copy_all':
                S += F
            elif cond == 'copy_sel':
                sel = mass >= tau
                S[sel] += F[sel]
            elif cond == 'copy_soft':
                S += np.minimum(1.0, mass / tau)[:, None] * F
            elif cond == 'replay':
                rep = [i for i, c in seen.items() if c >= 2]
                # replays the last stored (possibly noisy) sighting; item identity is given (an oracle advantage under noise)
                outs = {i: last_addr[i][1] @ (S[last_addr[i][0]] + F[last_addr[i][0]]) for i in rep}
                for i in rep:
                    idx, w = last_addr[i]
                    y = w @ S[idx]
                    S[idx] += w[:, None] * ((outs[i] - y) / (w @ w))[None, :]
            elif cond == 'replay_all':
                # replays every stored sighting of each repeated item (larger buffer; identity given)
                rep = [(i, a) for i, c in seen.items() if c >= 2 for a in last_addr[i]]
                outs = [a[1] @ (S[a[0]] + F[a[0]]) for i, a in rep]
                for (i, (idx, w)), o in zip(rep, outs):
                    y = w @ S[idx]
                    S[idx] += w[:, None] * ((o - y) / (w @ w))[None, :]
            F[:] = 0.0
        # ---- evaluate all targets learned so far ----
        q = tgt_list[task_of[tgt_list] <= t]
        Y = np.einsum('bk,bkd->bd', w_all[q], S[idx_all[q]] + F[idx_all[q]])
        Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
        cos_true = np.sum(Yn * Vt[q], 1)
        pred = np.argmax(Yn @ tgt_vals.T, 1)
        correct = tgt_list[pred] == q
        tq = task_of[q]
        for j in range(t + 1):
            m = tq == j
            M[t, j] = correct[m].mean(); C[t, j] = cos_true[m].mean()
    return M, C


def summarize(M):
    T = M.shape[0]
    diag = np.nanmean(np.diag(M))
    final = np.nanmean(M[-1])
    curve = np.array([np.nanmean([M[j + a, j] for j in range(T - a)]) for a in range(T)])
    half = next((a for a in range(T) if curve[a] < 0.5 * curve[0]), T)   # T means "never within horizon"
    forget = np.nanmean([np.nanmax(M[j:, j]) - M[-1, j] for j in range(T)])
    return dict(acq=diag, final=final, half_life=half, forgetting=forget,
                ret_age=[float(curve[a]) for a in (0, 1, 5, 10, 25, 50, 99) if a < T])


if __name__ == '__main__':
    print('ok')
