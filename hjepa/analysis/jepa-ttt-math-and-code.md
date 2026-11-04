# JEPA-TTT (arXiv 2610.00722v1): technical reference for reimplementation

Paper: "JEPA-TTT: Persistent Test-Time Training of Latent World Models for Planning under Dynamics Shifts", Zhang, Ye, Agarwal, Nourkhiz Mahjoub, Moradi Pari, Khashabi, Shu, Tadiparthi (Honda Research Institute USA / JHU), 30 Sep 2026. No code released.

Sources used:
- Full paper text (32 pages, all appendices A1 to A7): a local text extract
- LeWorldModel code, cloned read-only (shallow, LFS skipped) to a local checkout. The files are `jepa.py`, `module.py`, `train.py`, `eval.py`, `utils.py` and `config/`.
- H-JEPA repo at a local checkout (commit `97054ec`).

Convention: **(inferred)** marks anything the paper does not state. Line references are `file:line`. The paper contains no theory section, theorems or proofs. Its mathematics is Eqs. (1) to (4), the window definition in §3.4, the shift definitions (A5) to (A11), and the metric definitions.

---

## 1. Notation

| Symbol | Meaning | Value / shape |
|---|---|---|
| $s_t\in\mathcal S$ | physical simulator state at raw step $t$ | — |
| $o_t$ | RGB observation at raw step $t$ | 224×224×3 |
| $a_t\in\mathcal A$ | **commanded** (executed) raw action at raw step $t$. This is the action before the shift transform. | $d_a$-dim, continuous |
| $a^{\text{apply}}_t$ | action actually applied by the shifted simulator | per shift (§6) |
| $P_{\text{train}},P_{\text{test}}$ | transition kernels before and after the shift | Eq. (1) |
| $r(s)$, $y$ | scalar task score in $[0,1]$, with $y=r(s)$ | per task (§6) |
| $T$ | episode horizon (raw steps) | PushT 300, others 50 |
| $K$ | frame skip, i.e. raw steps per action block | 5 |
| $k$ | keyframe (block) index; keyframes are $K$ raw steps apart | — |
| $x_k$ | observation at keyframe $k$ | — |
| $b_k$ | action block: the $K$ raw actions between $x_k$ and $x_{k+1}$, flattened | $K d_a$-dim (10 for 2-D actions) **(inferred)** that it is flattened, following LeWM `train.py:68` |
| $E_\phi$ | visual encoder: ViT-Tiny/14, CLS token, then the latent projector (2-layer MLP with BN) | **frozen** at test time |
| $z_k=E_\phi(x_k)$ | latent | $d_z=192$ |
| $F_\theta$ | latent dynamics predictor: action encoder, AR transformer predictor and prediction projection | **trainable** at test time |
| $\theta_0$ | pretrained predictor parameters | — |
| $\theta_{e,t},\nu_{e,t},Q_{e,t}$ | predictor params, AdamW state and replay buffer at interaction $t$ of episode $e$ | persist across episodes |
| $C$ | predictor context length (keyframes) | 3 |
| $\hat z_{k+1}$ | predicted latent | $\mathbb R^{192}$ |
| $R_\omega$ | reward head, a sigmoid MLP $192\to256\to256\to1$ | fit offline, $\omega^\star$, **frozen** |
| $\mathcal D^{r}_{\text{train}}$ | offline (observation, score) pairs under $P_{\text{train}}$ | episode-disjoint split |
| $H$ | CEM planning horizon (action blocks) | 5 blocks = 25 raw steps |
| $\gamma$ | planning discount | 0.99 |
| $\hat J_k(\mathbf b)$ | predicted discounted return of candidate $\mathbf b=(b_k,\dots,b_{k+H-1})$ | Eq. (3) |
| $W_t$ | dense training window starting at raw step $t$ | §3.4 |
| $z_{w,j},b_{w,j}$ | latent $j\in\{0..C\}$ and action block $j\in\{0..C-1\}$ of window $w$ | — |
| $\mathcal B$ | minibatch of windows | $\lvert\mathcal B\rvert=64$ |
| $Q$ | growing buffer of all valid dense windows | unbounded |
| $N,N_\tau$ | number of unique dense windows (available after replanning step $\tau$) | — |
| $\mathrm{sg}(\cdot)$ | stop-gradient | — |
| $R(\alpha)$ | 2-D rotation matrix | Eq. (A5) |
| $\tilde q$ | wrapped Reacher joint-angle error in $[-\pi,\pi)^2$ | — |

---

## 2. Equations

### Eq. (1): dynamics shift
$$
P_{\text{test}}(\cdot\mid s,a)\neq P_{\text{train}}(\cdot\mid s,a)\quad\text{for some }(s,a)\in\mathcal S\times\mathcal A. \tag{1}
$$
There is a single shift and it stays fixed for the whole test time. State space, action space, observation mapping, score function, initial-state distribution and horizon are all unchanged. The score $r(s)$ is available offline and for evaluation, but it is **never observed** during test-time training or planning.

### Encoder, latent and predictor (§3.2, unnumbered)
$$
z_k=E_\phi(x_k),\qquad \hat z_{k+1}=F_\theta\big(z_{k-C+1:k},\;b_{k-C+1:k}\big).
$$
- "Up to $C$" previous latents and action blocks are used. With fewer than $C$ available (at episode start), the predictor runs on the shorter prefix. LeWM's `ARPredictor` supports this through `pos_embedding[:, :T]` (`module.py:281-282`) **(inferred)**.
- Action block $b_k$ is paired with $z_k$, and the pair predicts $z_{k+1}$. This is the LeWM alignment: `predict(emb[:, :C], act_emb[:, :C])` is matched to `emb[:, 1:C+1]` (`train.py:32-36`).
- $\theta$ = action encoder + autoregressive predictor + prediction projection (paper §3.2; Table A1).
- In LeWM, $E_\phi$ is the ViT CLS token followed by `projector` (`jepa.py:38-39`). The latent the predictor consumes and the reward head scores is the **post-projector** 192-d embedding **(inferred)**. Both the CLS and projector outputs are 192-d, but predicted latents only exist in projector space, so the reward head must read that space.

### Eq. (2): reward head (offline fit)
$$
\omega^\star=\arg\min_\omega\;\mathbb E_{(o,y)\sim\mathcal D^{r}_{\text{train}}}\Big[\big(R_\omega(E_\phi(o))-y\big)^2\Big]. \tag{2}
$$
- The fit is per-frame MSE regression from frozen encoder features to the instantaneous task score.
- The encoder is frozen during the fit (the head is fitted after pretraining).
- Details: §4.2 of this document.

### Eq. (3): planning objective
$$
\hat J_k(\mathbf b)=\sum_{h=1}^{H}\gamma^{h}\,R_{\omega^\star}\big(\hat z_{k+h}(\mathbf b,\theta_k)\big),\qquad \mathbf b=(b_k,\dots,b_{k+H-1}). \tag{3}
$$
- $\hat z_{k+h}$ comes from an autoregressive latent rollout of the **current** $\theta_k$, starting from the real encoded context at keyframe $k$. $\hat z_{k+1}$ uses $(z_{k-C+1:k}, b_{k-C+1:k-1}\text{(executed)}, b_k\text{(candidate)})$. Later steps feed predictions back in, with a sliding window of $C$.
- The sum starts at $h=1$, so the current state is not scored. The discount is $\gamma^h$, not $\gamma^{h-1}$.
- $H=5$ blocks, $\gamma=0.99$. PushT multiplies $R$ by 10 ("reward-head scale"), which does not change the CEM ranking.
- CEM maximises $\hat J$. Only the first block is executed, then the agent replans.

### Eq. (4): test-time training loss
$$
\mathcal L_{\text{TTT}}(\theta,\mathcal B)=\frac{1}{|\mathcal B|\,C\,d_z}\sum_{w\in\mathcal B}\sum_{j=0}^{C-1}\big\|\hat z_{w,j+1}(\theta)-\mathrm{sg}(z_{w,j+1})\big\|_2^2. \tag{4}
$$
Exact definition:
- Window $w$ holds latents $(z_{w,0},\dots,z_{w,C})$, which are $C+1=4$ keyframes, and action blocks $(b_{w,0},\dots,b_{w,C-1})$, which are $C=3$ blocks.
- At position $j$, $\hat z_{w,j+1}(\theta)=F_\theta(z_{w,0:j},b_{w,0:j})$. The inputs are **ground-truth** encoded latents, so this is teacher forcing, not autoregressive. All $C$ predictions come out of **one causal forward pass** over 3 tokens. This is identical to LeWM's `pred_loss` (`train.py:32-39`, with `history_size=3`, `num_preds=1`): `pred = model.predict(emb[:, :3], act_emb[:, :3])`, `tgt = emb[:, 1:4]`, `(pred - tgt).pow(2).mean()`.
- Averaging: a plain mean over batch × positions × latent dims, i.e. `.mean()` over a `(64, 3, 192)` tensor.
- Targets go through stop-gradient. $E_\phi$ (ViT + latent projector) is frozen, so the targets and context latents are constants and gradients only reach $\theta$.
- There is **no SIGReg term at test time**. SIGReg acts on encoder embeddings, which are frozen, so it would give $\theta$ zero gradient anyway.
- There is no multi-step rollout loss, even though the prediction metric is 5-step autoregressive.
- Optimizer: AdamW, lr $5\times10^{-5}$, weight decay $10^{-3}$, gradient clipping 1.0, batch 64.
- At most one update per replanning step, and only when a complete minibatch is "scheduled" (see §3).
- The trainable modules run in **evaluation mode**. Predictor dropout (0.1) is therefore off, and the BatchNorm running statistics in the prediction projection stay fixed. The BN affine parameters are still trainable **(inferred)**.

### Dense window (§3.4, unnumbered)
$$
W_t=\Big(\big(o_{t+jK}\big)_{j=0}^{C},\;\big(a_{t+jK\,:\,t+(j+1)K-1}\big)_{j=0}^{C-1}\Big),\qquad a_{u:v}=(a_u,\dots,a_v).
$$
- **Dense** construction starts a window at **every** raw step $t$. **Sparse** construction starts one only at action-block boundaries ($t\equiv 0 \bmod K$). Dense therefore covers all $K$ offsets and yields about $K\times$ as many windows.
- Validity (A1.2): a window start $t$ becomes valid only once a clip of $(C+1)K=20$ raw observation rows from $t$ is available, i.e. rows $t..t+19$. This is one action block more than the window strictly needs (rows $t..t+15$). It mirrors LeWM clips, which hold $C+1$ keyframes each carrying its own $K$-action block, with the last block unused. "The loss uses the keyframes and first $C$ action blocks in $W_t$."
- Sparse construction "creates its first window after $C$ complete action blocks". This rule is stated differently from the dense rule; see §9.
- Windows never cross episode boundaries **(inferred)**. Only test-time deployment episodes contribute windows; held-out evaluation trajectories never enter the buffer.
- Window counts per episode **(inferred)**: if an episode yields $R$ raw observation rows, the dense count is $R-19$. With $T=50$ that is about 31 to 32 windows; with PushT's $T=300$, about 281 to 282.

### Update schedule (A1.2, as a formula)
After $N$ unique dense windows have been collected, both the dense stream and dense replay have performed $\lfloor N/64\rfloor$ updates. Equivalently, at replanning step $\tau$:
$$
\text{update iff } U_\tau<\lfloor N_\tau/64\rfloor,\quad\text{at most 1 update per replan},\quad U_{\tau+1}=U_\tau+\mathbb 1[\text{update}].
$$
- Each replan adds $K=5$ new dense windows (once past the first 20 rows), so the backlog never exceeds 1 and the "at most 1 per replan" cap never binds in practice **(inferred)**.
- Leftover windows that do not complete a batch stay pending at scheduled evaluations and at episode 500. They are never flushed as a smaller batch.
- Dense replay samples each batch **uniformly from all of $Q$, without replacement within the batch**. Windows are never removed. The dense stream instead consumes new windows in chronological order, once each.
- Compute-matched sparse stream: each new sparse window is repeated $N_\tau-N_{\tau-1}$ times. That is the number of new dense windows at that step, typically 5.

### Persistence (§3.3)
$$
(\theta_{e+1,0},\nu_{e+1,0},Q_{e+1,0})=(\theta_{e,T_e},\nu_{e,T_e},Q_{e,T_e}).
$$
Predictor parameters, optimizer state and buffer all carry over unchanged across episode boundaries. They are initialised once: $\theta\leftarrow\theta_0$, $Q\leftarrow\varnothing$, fresh AdamW.

### Metrics (§4.1, A5; unnumbered)
Let $S^{(r)}(e)$ be run $r$'s mean held-out score over the 100 fixed held-out episodes, evaluated at $e\in\{0,50,\dots,500\}$.
- **Best score**: $\max_{e\in\{50,\dots,500\}}\frac13\sum_{r=1}^3 S^{(r)}(e)$. This is the maximum of the 3-run mean and excludes episode 0. For the reported $\pm$, the std across runs at the argmax is a guess **(inferred)**.
- **Normalised AUC** per run: $\frac{1}{500}\int_0^{500}S^{(r)}(e)\,de$, computed with the trapezoid rule over the 11 points $0,50,\dots,500$. The result is then averaged over the 3 runs. Aggregates average across shifts.
- **5-block AR latent MSE** (A5.1), per window, horizon $h=1..5$:
$$
\mathrm{MSE}_h=\frac1{d_z}\big\|\hat z^{\text{AR}}_{k+h}-z_{k+h}\big\|_2^2,
$$
  - Context is 3 observed keyframes; 5 future keyframes are predicted autoregressively from the executed actions, with no intermediate observations. Keyframes are 5 raw steps apart.
  - The window set is "every sliding window"; whether the stride is 1 raw step or 5 is unspecified (see §9).
  - Aggregation: average windows within each episode, then average over episodes, then average over the 5 horizons.
  - The trajectory set is fixed. PushT uses 100 seeded 300-step behaviour-policy trajectories. Two-Room, Reacher and Cube use 100 50-step trajectories generated by CEM with the *frozen* JEPA. The same trajectories are used for the frozen and adapted models, and their hashes are verified.
  - The "adapted" model is each of the 3 dense-replay models after episode 500.

### Shift equations (A5 to A11): see §6.

---

## 3. Algorithm (pseudocode)

```python
# ---------- offline (before deployment) ----------
wm = LeWM pretrained 20 epochs on 3000 train-dynamics episodes      # E_phi, F_theta0
R  = fit_reward_head(E_phi frozen, D_r_train)                        # Eq. (2)
freeze(wm.encoder, wm.projector, R)                                  # phi, latent projector, omega*
theta = [wm.action_encoder, wm.predictor, wm.pred_proj]              # trainable
wm.eval()                     # dropout OFF, BN running stats fixed (Table A1)
for p in theta.parameters(): p.requires_grad_(True)  # LeWM eval.py:91 sets requires_grad_(False); undo for theta
opt = AdamW(theta.parameters(), lr=5e-5, weight_decay=1e-3)   # betas default (inferred)
Q = []                        # list of windows (store latents; encoder is frozen -> equivalent) (inferred)
N = 0; U = 0                  # unique windows, updates done
B = 64; C = 3; K = 5; H = 5

def evaluate_heldout():       # at episodes 0,50,...,500; NO updates, NO buffer writes (inferred: no updates)
    scores = [run_episode(seed, learn=False, terminate_on_success=True) for seed in HELDOUT_SEEDS_100]
    return mean(scores)

# ---------- deployment: 500 episodes under P_test ----------
for e in range(500):
    if e % 50 == 0: log(evaluate_heldout())
    obs = env.reset(seed=deploy_seed[e])                     # deployment seeds differ per run
    ep_z   = [E_phi(obs)]          # latent of every raw frame (needed for every-offset windows)
    ep_a   = []                    # commanded raw actions (normalised, pre-shift)
    for k in range(T // K):        # one replan per action block: 60 (PushT) or 10 (others)
        ctx_z = ep_z[::-1][::K][:C][::-1]              # keyframes t, t-K, t-2K (as available)
        ctx_b = last (len(ctx_z)-1) executed blocks    # aligned with ctx_z[:-1]
        with torch.no_grad():
            b_star = CEM(theta, R, ctx_z, ctx_b)       # §4; maximises Eq. (3)
        for i in range(K):                             # execute FIRST block only
            obs, _, done, _ = env.step(denorm(b_star[0][i]))   # reward NOT observed
            ep_a.append(b_star[0][i]); ep_z.append(E_phi(obs))
            # Dense windows: a start t is valid once rows t..t+(C+1)K-1 exist
            t_new = len(ep_z) - (C + 1) * K            # newest start that just became valid
            if t_new >= 0:
                Q.append(make_window(ep_z, ep_a, t_new)) # z[t+jK] j=0..C ; a[t+jK : t+(j+1)K] j=0..C-1
                N += 1
            # NOTE: test-time interaction does NOT stop at success; runs to fixed horizon T
        # ---- update schedule: at most one update per replan ----
        if U < N // B:
            idx = random.sample(range(len(Q)), B)      # uniform over whole buffer, w/o replacement in batch
            z, b = stack(Q[idx])                       # z: (64, C+1, 192), b: (64, C, K*d_a)
            pred = theta.pred_proj(theta.predictor(z[:, :C], theta.action_encoder(b)))  # (64, C, 192)
            loss = ((pred - z[:, 1:].detach()) ** 2).mean()                             # Eq. (4)
            opt.zero_grad(); loss.backward()
            clip_grad_norm_(theta.parameters(), 1.0)   # norm clipping (inferred)
            opt.step(); U += 1
    # episode boundary: theta, opt state, Q all persist (no reset)
log(evaluate_heldout())   # episode 500; pending (<64) windows are not flushed
```

Ablation update rules (Table 1). All of them update only when a full batch of 64 is available.
- **Sparse stream**: windows start only at $t\equiv0\bmod K$. They are consumed in chronological order, once each. The update count is the natural one, $\lfloor N_{\text{sparse}}/64\rfloor$, roughly 1/5 of dense.
- **Compute-matched sparse stream**: every new sparse window is repeated $N_\tau-N_{\tau-1}$ times, so the presentation budget matches dense. Temporal coverage stays sparse.
- **Dense stream**: all dense windows, consumed in chronological order, once each, giving $\lfloor N/64\rfloor$ updates.
- **Dense replay** (JEPA-TTT): uniform sampling from all accumulated dense windows, with the same $\lfloor N/64\rfloor$ updates.

Matched persistence control (A6.1):
- Only dense windows from the **current episode** are used, with **batch size 8**. Loss, optimizer hyperparameters, planner and number of predictor updates are the same.
- *Persistent* carries $\theta$ and the optimizer state across episodes.
- *Episodic reset* restores $\theta_0$ and re-initialises the optimizer at every episode boundary.
- How the episode-local windows are sampled is not stated: stream or replay within the episode. See §9.

Error gating and thresholds: **none**. The paper describes no error-triggered updates, no loss thresholds, no early stopping at test time, and no buffer capacity or eviction. The only gate is "a full minibatch is scheduled", which is the $\lfloor N/64\rfloor$ counter.

Derived update budget **(inferred)**, from window counts per episode:

| Task | Raw steps/episode | Replans/episode | Dense windows/episode | Total dense windows (500 eps) | Total updates $\lfloor N/64\rfloor$ |
|---|---|---|---|---|---|
| PushT | 300 | 60 | ~281 | ~140,500 | ~2,195 |
| Two-Room / Reacher / Cube | 50 | 10 | ~31 | ~15,500 | ~242 (about 0.5 updates per episode) |

---

## 4. Planner

### 4.1 CEM (Table A1)

| Setting | Value |
|---|---|
| Candidates | 256 |
| Iterations | 6 |
| Elites | 32 |
| Horizon | 5 world-model blocks (25 raw steps) |
| Execution | 1 block, then replan (receding horizon 1 block) |
| Action block | 5 raw actions |
| Proposal | zero-centred Gaussian, initial std 1.0, std floor 0.05 |
| Objective | Eq. (3), $\gamma=0.99$, $h=1..H$ |
| Reward-head scale | PushT ×10, others ×1 (rank-invariant) |

Not specified; proposed defaults (all **(inferred)**):
- The Gaussian is over the normalised action space, with shape $(H, K\cdot d_a)$ and independent per dimension. LeWM z-scores actions (`utils.py:186-205`, `eval.py:72-80`).
- Refit uses the elite mean and std, with no momentum or smoothing. The std is clamped at 0.05 after each refit.
- The mean is re-initialised to zero at every replan ("zero-centred"), with no warm-start shifting.
- The executed block is the first block of the final CEM mean. That is how `stable_worldmodel.CEMSolver` behaves; returning the best elite would be an alternative.
- Sampled actions are clipped to the action bounds after de-normalisation.
- Upstream LeWM CEM defaults differ (`config/eval/solver/cem.yaml`: 300 samples, topk 30, 30 iterations, `receding_horizon: 5`). JEPA-TTT uses 256 / 32 / 6 / receding 1.
- LeWM's planning cost is goal-embedding MSE (`jepa.py:112-126`). JEPA-TTT **replaces** it with $-\hat J$ from the reward head, and **no goal image is used**.

### 4.2 Reward head (A1.1)
- Form: $R_\omega(z)=\sigma\big(W_3\,\mathrm{ReLU}(W_2\,\mathrm{ReLU}(W_1 z+c_1)+c_2)+c_3\big)$, with dims $192\to256\to256\to1$. The output lies in $(0,1)$, matching scores in $[0,1]$.
- Input: frozen encoder features $E_\phi(o)$ (192-d, post-projector **(inferred)**). At planning time the input is predicted latents $\hat z$.
- Data: (observation, score) pairs collected under $P_{\text{train}}$, on an **episode-disjoint split**. A PushT diagnostic mentions "90,300 validation frames" $=300$ episodes × 301 frames. This suggests the 10% held-out episodes of the 3,000 are used with every raw frame including $o_0$ **(inferred)**.
- Fitting: MSE (Eq. 2), AdamW, batch 512, lr $10^{-3}$, wd $10^{-5}$, early stopping with patience 15, at most 100 epochs. Selected epochs: PushT 67, Two-Room 64, Reacher 82, Cube 61.
- Labels: the instantaneous score $r(s)$. For PushT this is the instantaneous coverage; the episode score is its peak.
- Known weakness: the PushT head underpredicts high scores. Of 90,300 validation frames, 86,466 have score < 0.1 and only 114 have score ≥ 0.5. MSE is 0.0154 on [0.5, 0.7) and 0.0683 on [0.7, 1.0). A score-balanced head moves training-dynamics CEM coverage from 0.2682 to 0.3137, with no strict successes; the original head was kept.

### 4.3 Planning context
The predictor sees up to $C=3$ latest keyframes: the current observation and the frames $K$ and $2K$ steps earlier. Because the agent replans every block, those are exactly the previous executed block boundaries. It also sees the executed blocks between them, then the candidate blocks. LeWM `JEPA.rollout(info, action_sequence, history_size=3)` (`jepa.py:61-110`) does exactly this when `info["pixels"]` carries the history frames and `action_sequence` is prefixed with the executed history blocks.

---

## 5. Hyperparameter tables

### Table A1 (reproduced exactly): shared test-time planning and training settings

| Component | Setting |
|---|---|
| Test-time interaction horizon | 500 episodes under the test-time dynamics |
| Episode horizon | PushT: 300 raw steps, other tasks: 50 |
| Held-out evaluations | episodes 0, 50, . . . , 500 |
| Evaluation size | 100 held-out episodes per evaluation |
| CEM | 256 candidates, 6 iterations, 32 elites |
| Planning horizon | 5 world-model blocks |
| Execution | 1 block, then replan |
| Action block | 5 raw actions |
| CEM proposal | zero-centered Gaussian, initial std. 1.0, std. floor 0.05 |
| Reward discount | 0.99 |
| Reward-head scale | PushT: 10, other tasks: 1 (rank-invariant for CEM) |
| Test-time optimizer | AdamW, LR 5 × 10⁻⁵, WD 10⁻³ |
| Test-time batch size | 64 |
| Gradient clipping | 1.0 |
| Trainable-module mode | evaluation mode; normalization statistics remain fixed |
| Update rate | at most 1 update per replan |
| Trainable components | action encoder, predictor, prediction projection |
| Frozen components | image encoder, latent projector, reward head, decoder |
| Test-time inputs | current RGB observations and executed actions |

### Pretraining (A1.1 prose, tabulated; not a table in the paper)

| Item | Value |
|---|---|
| Observations | 224×224 RGB |
| Encoder | ViT-Tiny, 14×14 patches, 192-d outputs |
| AR predictor | 6 layers, 16 heads × 64 dim, hidden 192, MLP 2048, dropout 0.1 |
| Action encoder | one 5-action block → 192-d |
| Latent and prediction projections | 2-layer MLP, hidden 2048, BatchNorm |
| Sample | 3 context keyframes + 1 prediction keyframe, 5 raw steps apart |
| Data | 3,000 complete episodes per task under train dynamics |
| Collection policy | PushT: equal mix of block-local and arena-wide target actions held 4 steps. Two-Room: upstream noisy heuristic. Reacher: random. OGB-Cube: equal mix of guided and Gaussian exploration, with 30% uniform actions in guided episodes |
| Max collection horizon | PushT 300, Two-Room 100, Reacher 200, Cube 50 |
| Optimisation | from scratch, 20 epochs, AdamW, batch 128, max LR 5×10⁻⁵, WD 10⁻³, grad clip 1.0, linear warmup + cosine, bf16 |
| Split / checkpoint | random 90/10 window split; predeclared epoch-20 checkpoint |
| Loss | upstream LeWM prediction + SIGReg (weight 0.09, 17 knots, 1,024 projections) |
| Reward head | sigmoid MLP 192→256→256→1, ReLU; AdamW, batch 512, LR 10⁻³, WD 10⁻⁵, early stop patience 15, ≤100 epochs; selected epochs 67/64/82/61 |
| Decoder (visualisation only) | 20 epochs, AdamW, batch 32, LR 2×10⁻⁴, WD 10⁻⁴, clip 1.0, bf16 autocast |

These match the LeWM configs (`config/train/lewm.yaml`, `config/train/model/lewm.yaml`) except for the epoch count: LeWM's config has `max_epochs: 100`, the paper uses 20.

### Baseline hyperparameters (A3, tabulated)

| Baseline | Settings |
|---|---|
| AdaJEPA (adapted to the goal-free setting) | Same pretrained LeWM, reward head and CEM. Every episode restores $\theta_0$ and clears the history of the **5 most recent executed action segments**. **New Adam optimizer at every replanning step, one update.** Predictor LR 5×10⁻⁴; encoder-head LR 10⁻⁵ when adapted. No weight decay, no clipping. Trains the last predictor layer, either with the encoder head frozen (pred-last+enc-frozen) or with the trainable projection head (pred-last+enc-last, the **default**). Evaluated as 3 runs × 100 episodes; one per-episode score is used for both metrics. |
| PPO-TTT | CNN encoder with 32/64/64 channels, 256-d feature layer, diagonal-Gaussian actor and value heads. PushT requested 10M steps, stopped at 2.40M by a predeclared validation plateau, checkpoint at 1.30M. Two-Room/Reacher/Cube trained 2M steps; checkpoints 1.5M / 1.2M / 1.7M. At test time the policy, value function and a **fresh** optimizer persist across 500 episodes using the **oracle** online reward. Every 10 episodes: Adam 10 epochs, LR 3×10⁻⁴, minibatch 512, γ 0.99, GAE λ 0.95, clip 0.2, value coefficient 0.5, entropy 0, grad clip 0.5. Stochastic test-time actions, deterministic held-out actions. |

---

## 6. Environments, shifts, data, metrics

All four environments are the LeWorldModel ones. Each task score lies in $[0,1]$. Every shift changes only how the commanded action produces the next state. The visual goal, score, action bounds and initial-state distribution are unchanged. Rotation matrix:
$$R(\alpha)=\begin{pmatrix}\cos\alpha&-\sin\alpha\\ \sin\alpha&\cos\alpha\end{pmatrix}\tag{A5}$$

### PushT (horizon 300)
The action is a normalised 2-D target for the circular agent. Score: overlap coverage between the T block and the green target. The **episode score is the peak** coverage.
- **Action rotation**: the target is rotated 90° about the current agent position $p_t$ before the base controller sees it:
$$u^{\text{apply}}_t=p_t+R(\pi/2)(u_t-p_t).\tag{A6}$$
This is an actuator-frame shift.
- **Contact rotation**: the agent follows its command normally. At every physics substep, only the contact-induced block displacement $\Delta q$ is replaced with $R(\pi/2)\Delta q$. This is a contact-response shift.

### Two-Room (horizon 50)
The agent starts in the left room. Goal $g=(164,112)$ px, doorway waypoint $d=(112,49)$.
- Score: $\exp(-D(s,g)/64)$, where $D=\|s-d\|_2+\|d-g\|_2$ while the agent is left of the wall, and $D=\|s-g\|_2$ after crossing.
- Success: within 16 px of $g$.
- Both shifts apply $a^{\text{apply}}_t=R(\alpha(s_t))a_t$.
- **Spatial wave field**: $\alpha(s)=75^\circ\sin\!\big(\tfrac{2\pi(x-20)}{88}\big)$ (A7). The rotation varies with horizontal position.
- **Grid (rotation) field**: $\alpha(s)=75^\circ\sin\!\big(\tfrac{2\pi(x-20)}{96}\big)\cos\!\big(\tfrac{2\pi(y-20)}{96}\big)$ (A8). The rotation varies with both coordinates.

### Reacher (DMC, horizon 50)
Goal: fixed joint configuration $(0,0)$. $\tilde q\in[-\pi,\pi)^2$ is the wrapped error.
- Score: $\exp(-\mathrm{mean}(\tilde q^2)/2)$.
- Success: $|\tilde q_i|<0.05$ for both joints.
- The motor command is rotated by a configuration-dependent angle that vanishes near the goal:
$$h(\tilde q)=1-\exp\!\Big(-\frac{\mathrm{mean}(\tilde q^2)}{2(0.20)^2}\Big),\qquad \alpha(\tilde q)=\alpha_{\max}\,h(\tilde q)\tanh\!\Big(\frac{\psi(\tilde q)}{\kappa}\Big).\tag{A9}$$
- **Joint-phase field**: $\psi=\tilde q_1+0.5\tilde q_2$, $\kappa=0.35$, $\alpha_{\max}=155^\circ$.
- **Harmonic field**: $\psi=\sin\tilde q_1+0.7\sin\tilde q_2$, $\kappa=0.30$, $\alpha_{\max}=165^\circ$.
- Both shifts apply $a^{\text{apply}}_t=\mathrm{clip}(R(\alpha(\tilde q_t))a_t,-1,1)$.

### OGBench-Cube (horizon 50)
OGBench-Cube simulator with a UR5e, the same camera and table, and the physical red cube. The task is changed to a **3-D reach**: target hover point 0.10 m above the cube, at $(0.425,0,0.12)$ m.
- Action: Cartesian $x,y,z$. Yaw and gripper are fixed at 0. Whether the action vector stays 5-D with zeros or is cut to 3-D is not stated; see §9.
- Score: $\exp(-\|e\|_2/0.10)$ for end-effector error $e$.
- Success: $\|e\|_2\le0.05$ m.
- **Quadratic crosstalk** (pairwise nonlinear cross-axis terms):
$$a^{\text{apply}}=\mathrm{clip}\big(a+3(a_ya_z,\;a_za_x,\;a_xa_y),-1,1\big).\tag{A10}$$
- **Delayed cyclic actuation** (one-step delay plus fixed axis routing; $a_{-1}=0$ at reset):
$$a^{\text{apply}}_t=(a_{t-1,y},\;a_{t-1,z},\;a_{t-1,x}).\tag{A11}$$

### Training data, protocol and metrics
- **Training data**: 3,000 complete episodes per task under $P_{\text{train}}$, using the collection policies in §5. These policies use no task rewards and do not need to match the evaluation reset distribution. The reward head uses an episode-disjoint split.
- **Deployment**: 500 episodes per run under one fixed $P_{\text{test}}$. Interaction continues to the fixed horizon after success, so the amount of data does not depend on the model. Each shift has 3 runs; they share the same pretrained model and differ only in environment-seed sequences.
- **Held-out evaluation**: 100 fixed episodes, from a seed suite disjoint from deployment, at episodes 0, 50, …, 500. Normal task termination applies (the episode ends at success). Evaluation data never enters $Q$.
- **Metrics**: best score, normalised AUC and 5-block AR latent MSE, defined in §2. Reported std is the sample std over 3 runs and reflects interaction variability for a single pretrained model.
- **Offline reference**: 3,000 shifted-dynamics episodes, a model trained from scratch for 20 epochs, and a new reward head, evaluated once on 100 held-out episodes. It is neither a deployable baseline nor an upper bound.

---

## 7. Results tables (reproduced exactly)

### Table 1: update-rule ablation (mean ± std over 3 runs)

| Shift | Metric | Sparse stream | Compute-matched sparse stream | Dense stream | Dense replay |
|---|---|---|---|---|---|
| PushT: action rotation | Best | 0.253 ± 0.063 | 0.236 ± 0.049 | 0.511 ± 0.009 | 0.490 ± 0.022 |
| | AUC | 0.169 ± 0.007 | 0.169 ± 0.021 | 0.359 ± 0.044 | 0.378 ± 0.022 |
| PushT: contact rotation | Best | 0.317 ± 0.258 | 0.521 ± 0.319 | 0.646 ± 0.044 | 0.619 ± 0.033 |
| | AUC | 0.205 ± 0.082 | 0.320 ± 0.157 | 0.490 ± 0.041 | 0.555 ± 0.028 |
| Two-Room: spatial wave | Best | 0.505 ± 0.031 | 0.790 ± 0.002 | 0.797 ± 0.002 | 0.799 ± 0.002 |
| | AUC | 0.427 ± 0.034 | 0.672 ± 0.020 | 0.697 ± 0.012 | 0.728 ± 0.012 |
| Two-Room: grid rotation | Best | 0.786 ± 0.004 | 0.794 ± 0.003 | 0.782 ± 0.000 | 0.799 ± 0.003 |
| | AUC | 0.662 ± 0.009 | 0.717 ± 0.026 | 0.686 ± 0.020 | 0.725 ± 0.019 |
| Reacher: joint phase | Best | 0.557 ± 0.025 | 0.768 ± 0.062 | 0.725 ± 0.018 | 0.758 ± 0.012 |
| | AUC | 0.401 ± 0.007 | 0.545 ± 0.051 | 0.593 ± 0.013 | 0.619 ± 0.019 |
| Reacher: harmonic | Best | 0.446 ± 0.046 | 0.588 ± 0.048 | 0.610 ± 0.039 | 0.688 ± 0.075 |
| | AUC | 0.398 ± 0.007 | 0.488 ± 0.014 | 0.492 ± 0.006 | 0.518 ± 0.012 |
| OGB-Cube: quadratic | Best | 0.520 ± 0.053 | 0.652 ± 0.010 | 0.653 ± 0.004 | 0.628 ± 0.020 |
| | AUC | 0.390 ± 0.005 | 0.528 ± 0.015 | 0.528 ± 0.013 | 0.530 ± 0.009 |
| OGB-Cube: delayed cyclic | Best | 0.591 ± 0.044 | 0.646 ± 0.006 | 0.646 ± 0.004 | 0.643 ± 0.003 |
| | AUC | 0.290 ± 0.022 | 0.522 ± 0.006 | 0.542 ± 0.006 | 0.513 ± 0.009 |
| Mean | Best | 0.497 ± 0.028 | 0.625 ± 0.025 | 0.671 ± 0.008 | 0.678 ± 0.013 |
| | AUC | 0.368 ± 0.007 | 0.495 ± 0.025 | 0.548 ± 0.007 | 0.571 ± 0.004 |

Caption excerpt: the sparse stream uses its natural number of updates. The Mean rows summarise the three run-level averages across shifts. Bold in the paper marks the highest unrounded mean; that formatting is lost in the text extraction. Dense replay has the highest aggregate mean best score and AUC.

### Table A2: exact values for Figure 5

| Shift | Metric | Frozen JEPA | PPO-TTT | AdaJEPA | JEPA-TTT |
|---|---|---|---|---|---|
| PushT: action rotation | Best | 0.080 | 0.119 ± 0.009 | 0.121 ± 0.015 | 0.490 ± 0.022 |
| | AUC | 0.080 | 0.110 ± 0.006 | 0.121 ± 0.015 | 0.378 ± 0.022 |
| PushT: contact rotation | Best | 0.127 | 0.237 ± 0.029 | 0.186 ± 0.016 | 0.619 ± 0.033 |
| | AUC | 0.127 | 0.219 ± 0.005 | 0.186 ± 0.016 | 0.555 ± 0.028 |
| Two-Room: spatial wave | Best | 0.557 | 0.554 ± 0.334 | 0.579 ± 0.013 | 0.799 ± 0.002 |
| | AUC | 0.557 | 0.421 ± 0.295 | 0.579 ± 0.013 | 0.728 ± 0.012 |
| Two-Room: grid rotation | Best | 0.211 | 0.456 ± 0.015 | 0.203 ± 0.010 | 0.799 ± 0.003 |
| | AUC | 0.211 | 0.426 ± 0.019 | 0.203 ± 0.010 | 0.725 ± 0.019 |
| Reacher: joint phase | Best | 0.367 | 0.336 ± 0.005 | 0.426 ± 0.056 | 0.758 ± 0.012 |
| | AUC | 0.367 | 0.332 ± 0.009 | 0.426 ± 0.056 | 0.619 ± 0.019 |
| Reacher: harmonic | Best | 0.369 | 0.379 ± 0.009 | 0.418 ± 0.049 | 0.688 ± 0.075 |
| | AUC | 0.369 | 0.377 ± 0.007 | 0.418 ± 0.049 | 0.518 ± 0.012 |
| OGB-Cube: quadratic | Best | 0.257 | 0.212 ± 0.043 | 0.271 ± 0.020 | 0.628 ± 0.020 |
| | AUC | 0.257 | 0.199 ± 0.031 | 0.271 ± 0.020 | 0.530 ± 0.009 |
| OGB-Cube: delayed cyclic | Best | 0.172 | 0.162 ± 0.049 | 0.173 ± 0.019 | 0.643 ± 0.003 |
| | AUC | 0.172 | 0.151 ± 0.025 | 0.173 ± 0.019 | 0.513 ± 0.009 |
| Mean | Best | 0.267 | 0.307 | 0.297 | 0.678 |
| | AUC | 0.267 | 0.279 | 0.297 | 0.571 |

Headline claims checked against the tables:
- Best score: +153%, since 0.678/0.267 − 1 ≈ 1.54. The small gap from 153% is presumably rounding of unrounded means.
- AUC: +113% (0.571/0.267 − 1 ≈ 1.14).
- Latent MSE: −83%, which is the mean of the 8 reductions in Table A4 (82.99%).

The two-row label order in the Figure 7 bar labels is garbled in the text extraction; Table A4 is authoritative.

### Table A3: AdaJEPA trainable components (mean ± sample std over 3 runs; same value for best and AUC)

| Shift | pred-last + enc-last (default) | pred-last + enc-frozen |
|---|---|---|
| PushT: action rotation | 0.1210 ± 0.0155 | 0.1042 ± 0.0163 |
| PushT: contact rotation | 0.1864 ± 0.0156 | 0.1527 ± 0.0145 |
| Two-Room: spatial wave | 0.5791 ± 0.0134 | 0.5850 ± 0.0125 |
| Two-Room: grid rotation | 0.2031 ± 0.0101 | 0.2470 ± 0.0028 |
| Reacher: joint phase | 0.4259 ± 0.0565 | 0.4266 ± 0.0552 |
| Reacher: harmonic | 0.4176 ± 0.0490 | 0.4183 ± 0.0487 |
| OGB-Cube: quadratic | 0.2711 ± 0.0199 | 0.2965 ± 0.0144 |
| OGB-Cube: delayed cyclic | 0.1730 ± 0.0191 | 0.1608 ± 0.0204 |
| Mean | 0.2972 | 0.2989 |

The difference between the two configurations is 0.0017. All main comparisons use the default.

### Table A4: five-block autoregressive latent MSE (frozen vs after 500 episodes of dense replay)

| Shift | Frozen JEPA MSE | Adapted MSE | Reduction |
|---|---|---|---|
| PushT action rotation | 1.82540 | 0.19988 ± 0.00491 | 89.05% |
| PushT contact rotation | 0.32117 | 0.17751 ± 0.00953 | 44.73% |
| Two-Room spatial wave | 1.74541 | 0.18697 ± 0.02870 | 89.29% |
| Two-Room grid field | 1.48501 | 0.02205 ± 0.00345 | 98.52% |
| Reacher joint phase | 2.11969 | 0.48927 ± 0.02162 | 76.92% |
| Reacher harmonic | 2.00316 | 0.35488 ± 0.00216 | 82.28% |
| OGB-Cube quadratic | 2.10429 | 0.24754 ± 0.01766 | 88.24% |
| OGB-Cube delayed | 2.10517 | 0.10688 ± 0.01710 | 94.92% |

Matched MSE under the training dynamics, using the same behaviour families: PushT 0.10458, Two-Room 0.11622, Reacher 1.27430, OGBench-Cube 0.02131. Raw magnitudes are not comparable across tasks.

### Table A5: matched persistence control (episode-local dense windows, batch 8, equal update count)

| Shift | Metric | Persistent | Episodic reset | Difference |
|---|---|---|---|---|
| PushT action rotation | Best | 0.4683 ± 0.0239 | 0.1049 ± 0.0168 | +0.3634 |
| | AUC | 0.3863 ± 0.0247 | 0.0934 ± 0.0016 | +0.2929 |
| PushT contact rotation | Best | 0.6393 ± 0.0292 | 0.1735 ± 0.0924 | +0.4658 |
| | AUC | 0.5324 ± 0.0352 | 0.1254 ± 0.0127 | +0.4069 |
| Two-Room spatial wave | Best | 0.7971 ± 0.0016 | 0.5972 ± 0.0135 | +0.1999 |
| | AUC | 0.7692 ± 0.0041 | 0.5614 ± 0.0089 | +0.2078 |
| Two-Room grid field | Best | 0.7892 ± 0.0157 | 0.2178 ± 0.0061 | +0.5714 |
| | AUC | 0.7189 ± 0.0193 | 0.2127 ± 0.0043 | +0.5062 |
| Reacher joint phase | Best | 0.9394 ± 0.0717 | 0.3931 ± 0.0275 | +0.5463 |
| | AUC | 0.7329 ± 0.1311 | 0.3816 ± 0.0162 | +0.3513 |
| Reacher harmonic | Best | 0.9153 ± 0.1150 | 0.3909 ± 0.0379 | +0.5244 |
| | AUC | 0.7426 ± 0.1098 | 0.3810 ± 0.0211 | +0.3616 |
| OGB-Cube quadratic | Best | 0.6385 ± 0.0085 | 0.2622 ± 0.0013 | +0.3763 |
| | AUC | 0.5996 ± 0.0067 | 0.2569 ± 0.0017 | +0.3427 |
| OGB-Cube delayed | Best | 0.6465 ± 0.0010 | 0.1723 ± 0.0064 | +0.4742 |
| | AUC | 0.6119 ± 0.0042 | 0.1686 ± 0.0021 | +0.4433 |
| Mean | Best | 0.7292 | 0.2890 | +0.4402 |
| | AUC | 0.6367 | 0.2726 | +0.3641 |

Note the anomaly: the persistent episode-local variant with batch 8 beats main JEPA-TTT on mean best (0.729 vs 0.678) and mean AUC (0.637 vs 0.571), and is much stronger on Reacher (0.94 / 0.92 vs 0.76 / 0.69). The paper does not comment on this. A likely cause is that batch 8 gives about 8× more updates **(inferred)**.

### Table A6: continued training under unchanged training dynamics (no-shift control)

| Task | Frozen JEPA best/AUC | Continued training best | Continued training AUC |
|---|---|---|---|
| PushT | 0.2682 | 0.3599 ± 0.0187 | 0.3247 ± 0.0264 |
| Two-Room | 0.8039 | 0.8013 ± 0.0019 | 0.7894 ± 0.0084 |
| Reacher | 0.9995 | 0.9520 ± 0.0065 | 0.9438 ± 0.0042 |
| OGB-Cube | 0.6439 | 0.6492 ± 0.0010 | 0.6465 ± 0.0003 |

Averages: best 0.679 → 0.691, AUC 0.679 → 0.676. The paper concludes there is no consistent benefit. Reacher degrades slightly.

### Table A7: training-dynamics checks

| Task | Planning score | Strict success | Reward-head MSE | Decoder MSE |
|---|---|---|---|---|
| PushT | 0.2682 | 0% | 2.54 × 10⁻⁴ | 8.10 × 10⁻⁴ |
| Two-Room | 0.8039 | 100% | 1.24 × 10⁻⁶ | 5.21 × 10⁻⁶ |
| Reacher | 0.9995 | 100% | 1.36 × 10⁻⁶ | 3.85 × 10⁻⁵ |
| OGB-Cube | 0.6439 | 100% | 3.34 × 10⁻⁵ | 5.77 × 10⁻⁴ |

Epoch-20 upstream validation prediction losses: PushT 0.01291, Two-Room 0.00802, Reacher 0.02702, OGBench-Cube 0.01534. These are on overlapping random clips.

### Table A8: offline reference trained under the test-time dynamics

| Shift | Frozen JEPA | JEPA-TTT best | JEPA-TTT AUC | Offline model [95% CI] |
|---|---|---|---|---|
| PushT action rotation | 0.0804 | 0.490 ± 0.022 | 0.378 ± 0.022 | 0.5667 [0.5447, 0.5886] |
| PushT contact rotation | 0.1268 | 0.619 ± 0.033 | 0.555 ± 0.028 | 0.6321 [0.6145, 0.6497] |
| Two-Room spatial wave | 0.5572 | 0.799 ± 0.002 | 0.728 ± 0.012 | 0.7993 [0.7970, 0.8017] |
| Two-Room grid field | 0.2106 | 0.799 ± 0.003 | 0.725 ± 0.019 | 0.8021 [0.7993, 0.8050] |
| Reacher joint phase | 0.3673 | 0.758 ± 0.012 | 0.619 ± 0.019 | 0.9994 [0.9993, 0.9994] |
| Reacher harmonic | 0.3687 | 0.688 ± 0.075 | 0.518 ± 0.012 | 0.9994 [0.9994, 0.9995] |
| OGB-Cube quadratic | 0.2572 | 0.628 ± 0.020 | 0.530 ± 0.009 | 0.6375 [0.6322, 0.6428] |
| OGB-Cube delayed | 0.1717 | 0.643 ± 0.003 | 0.513 ± 0.009 | 0.6527 [0.6465, 0.6589] |

CI = mean ± 1.96 SE over the 100 episodes. JEPA-TTT best is within 0.08 of the reference on 6 shifts, with larger gaps on both Reacher shifts.

Figures without tables: Fig. A1 (held-out curves for the 4 update rules) and Fig. A4 (online scores in bins of 50 episodes). Fig. A4 says only that dense replay has the highest aggregate online score; no numbers are given.

---

## 8. Code mapping: θ, φ, ω in LeWM and H-JEPA level-1

### 8.1 LeWM

| Paper | LeWM object | Definition | Notes |
|---|---|---|---|
| $E_\phi$ image encoder (frozen) | `JEPA.encoder` (HF ViT-tiny, patch 14) | `jepa.py:23`; config `config/train/model/lewm.yaml:3-9` | CLS token `last_hidden_state[:,0]` at `jepa.py:38` |
| $E_\phi$ latent projector (frozen) | `JEPA.projector` = `module.MLP(192→2048→192, BatchNorm1d, GELU)` | `jepa.py:26,39`; `module.py:217-241`; `lewm.yaml:29-36` | BN; must be in eval mode |
| $\theta$: action encoder | `JEPA.action_encoder` = `module.Embedder` (Conv1d k=1 `input_dim→smoothed_dim=10`, then Linear 10→768, SiLU, Linear 768→192) | `jepa.py:25,43`; `module.py:189-214`; `lewm.yaml:24-27` | `input_dim = frameskip*action_dim` (`train.py:68`) |
| $\theta$: AR predictor | `JEPA.predictor` = `module.ARPredictor` (learned `pos_embedding` of `num_frames=history_size=3`, 6 `ConditionalBlock`s with AdaLN-zero action conditioning, causal attention, dropout 0.1) | `jepa.py:24`; `module.py:244-285`, `88-111`, `56-85`, `131-187`; `lewm.yaml:11-22` | Causal (`is_causal=True`, `module.py:83`) gives teacher-forced next-step outputs at every position. Dropout depends on `self.training`. |
| $\theta$: prediction projection | `JEPA.pred_proj` = `module.MLP(192→2048→192, BN)` | `jepa.py:27,53`; `lewm.yaml:38-45` | BN running stats frozen at test time (eval mode) |
| $F_\theta$ forward | `JEPA.predict(emb, act_emb)` | `jepa.py:47-55` | Takes `act_emb = action_encoder(action)` |
| Eq. (4) | `lejepa_forward` prediction term | `train.py:32-39` | Drop the `sigreg_loss` term and freeze encoder/projector |
| SIGReg (pretraining only) | `module.SIGReg(knots=17, num_proj=1024)`, weight 0.09 | `module.py:10-36`; `lewm.yaml:38-43` | Not used at test time |
| Planning rollout | `JEPA.rollout(info, action_sequence, history_size=3)` | `jepa.py:61-110` | Re-encodes the init frames; autoregressive with a window of 3 |
| Planning cost | `JEPA.criterion` / `get_cost` = goal-embedding MSE | `jepa.py:112-153` | **Replace** with $-\sum_h\gamma^hR_{\omega^\star}(\hat z_{k+h})$ over `predicted_emb[:, :, H_hist:]` |
| $R_\omega$ | — | not present | New module |
| Action normalisation | z-score (`StandardScaler`) from dataset stats | `eval.py:72-80`; `utils.py:186-205` | **(inferred)** that JEPA-TTT keeps this |
| Model freeze at eval | `model.requires_grad_(False)` | `eval.py:91` | Must re-enable for θ |
| Upstream CEM defaults | `CEMSolver`: 300 samples, topk 30, 30 iterations, horizon 5, receding 5 | `config/eval/solver/cem.yaml`; `config/eval/*.yaml` `plan_config` | JEPA-TTT: 256/32/6, receding 1 block |

θ parameter selection in LeWM: `list(m.action_encoder.parameters()) + list(m.predictor.parameters()) + list(m.pred_proj.parameters())`.

### 8.2 H-JEPA level-1 (`h_jepa/`)

The level-1 model is `world_model.get_level(1)`, which equals `world_model.jepas[0]` (`models/hjepa.py:39-49`). It is a `models.jepa.JEPA` built in `hjepa_utils.py:182-337`.

| Paper | H-JEPA attribute | Definition / build site | Notes |
|---|---|---|---|
| $E_\phi$ image encoder (frozen) | `jepa.encoder.encoder` (ViT) | `ProjectedEncoder` `models/jepa.py:46-61`; built `hjepa_utils.py:256-269` | With `use_proprio` it is a `FusionEncoder` (`jepa.py:64-102`); the paper uses pixels only |
| $E_\phi$ latent projector (frozen) | `jepa.encoder.projector` | `build_projector` `models/module.py:11-33` (MLP hidden 2048 + BN) | Same as LeWM `projector` |
| $\theta$: action encoder | **`jepa.action_embed`** (`Embedder`) | `models/module.py:189`; built `hjepa_utils.py:303-312` (`input_dim = frameskip*action_dim`); used `jepa.py:351,399,409` | **Naming trap:** H-JEPA `jepa.action_encoder` is `nn.Identity()` at level 1 (`jepa.py:137`). LeWM's `action_encoder` corresponds to H-JEPA's `action_embed`. |
| $\theta$: AR predictor | `jepa.predictor.predictor` (`ARPredictor`) | `models/predictors/predictors.py:6-49`; built `hjepa_utils.py:277-283` | Same code as LeWM, with `output_dim = encoder hidden_dim` |
| $\theta$: prediction projection | `jepa.predictor.projector` | `ProjectedPredictor` `jepa.py:105-116`; built `hjepa_utils.py:284-288` | `jepa.predictor` = ARPredictor + pred_proj together |
| $F_\theta$ | `jepa.predict(emb, act_emb)` | `jepa.py:355-360` | `act_emb = jepa.action_embed(action)` |
| Eq. (4) | teacher-forcing loss (`rollout_n == 1`) | `hjepa_utils.py:77-80`, `115-118` | Identical to Eq. (4) when `history_size=3`; omit SIGReg and the level-2+ terms |
| Planning rollout | `jepa.rollout(info, action_sequence, history_size=1)` | `jepa.py:376-418` | **Default history 1**, and `config/eval/pusht_flat.yaml:8` sets `history_size: 1`. JEPA-TTT needs 3. |
| Planning cost | `jepa.criterion` (goal MSE) | `jepa.py:420-436` | Replace with the reward head |
| Model load / eval mode | `load_model` → `model.eval()` | `planning_eval.py:179-199` | Eval mode is what JEPA-TTT wants during updates; also enable grads for θ |

θ in H-JEPA: `jepa = model.get_level(1)`, then `params = [*jepa.action_embed.parameters(), *jepa.predictor.parameters()]`. The frozen part is `jepa.encoder`, covering both ViT and projector.

Mismatches to resolve when porting:
1. **Predictor size.** H-JEPA level-1 configs use `depth 4, heads 8, mlp_dim 1024` (`config/train/base/{pusht,cube,fourroom,ant}.yaml`, e.g. `pusht.yaml:110-119`). The paper and LeWM use `6 / 16 / 2048` (`lewm.yaml:17-19`).
2. **No CEM in H-JEPA.** The vendored `stable_worldmodel/solver/` contains only `GradientSolver` (`gd.py:13`), and flat evals use it (`config/eval/pusht_flat.yaml:23-36`). CEM must be implemented, or taken from the upstream `stable_worldmodel` package.
3. **Environments.** H-JEPA has configs for pusht, cube, fourroom, ant and droid, but none for **Two-Room or Reacher**. Its cube task is goal-conditioned manipulation, while the paper uses a fixed-target 3-D reach.
4. **Training epochs.** H-JEPA base uses `max_epochs: 9` (`pusht.yaml:54`); the paper uses 20.
5. **Precision.** H-JEPA uses `bf16-mixed`; LeWM uses `bf16`.

---

## 9. Ambiguities and missing details (with proposed defaults)

| # | Gap | Proposed default |
|---|---|---|
| 1 | Raw rows per episode: does an episode yield $T$ or $T+1$ observation rows? This sets the window count (31 vs 32). | Store $o_0..o_T$ ($T+1$ rows). Start $t$ is valid if $t+(C+1)K-1\le T$. |
| 2 | Why the dense validity rule asks for $(C+1)K$ rows when the loss needs only $CK+1$. | Follow the paper literally: require 20 rows. This is consistent with LeWM clip loading. |
| 3 | Sparse validity: "first window after $C$ complete action blocks" vs the dense rule. | Sparse starts are $t\in\{0,K,2K,\dots\}$ with the same 20-row validity as dense. |
| 4 | Whether the buffer stores raw RGB or latents. | Store per-frame latents (192-d) and normalised actions. Re-encoding is equivalent because $E_\phi$ is frozen and in eval mode. |
| 5 | How $b_k$ is laid out (concatenated $K\cdot d_a$ vector, normalised). | Use z-scored actions from the training-data stats, flattened to `(K*d_a)`, as LeWM does. |
| 6 | Whether adaptation also happens during held-out evaluation episodes. | No updates and no buffer writes during evaluation. The paper confirms only that the buffer is untouched. |
| 7 | Whether the update happens before or after the next CEM call. | Algorithm 1 order: CEM, execute block, add windows, then optional update. The new θ is used at the next replan. |
| 8 | AdamW betas/eps, LR schedule, test-time precision. | Defaults (0.9, 0.999, 1e-8), constant LR, fp32 (or bf16 autocast to match pretraining). |
| 9 | Type of gradient clipping. | `clip_grad_norm_(…, 1.0)` over θ. |
| 10 | BN affine params in `pred_proj` under eval mode. | Trainable, with running stats frozen. "Prediction projection" is listed as trainable. |
| 11 | CEM mean/std update, warm start, clipping, which action is returned. | Elite mean/std, no smoothing, std floor 0.05 after refit, zero mean re-init every replan, clip to action bounds, return the first block of the final mean. |
| 12 | Planning history at episode start (fewer than 3 frames). | Use the available frames, without padding. |
| 13 | Where the reward head reads features (CLS or post-projector). | Post-projector 192-d $z$, the same space as $\hat z$. |
| 14 | Reward-head data: which frames, and the exact split. | Every raw frame of a 10% episode-held-out split, plus held-out validation for early stopping. The 90,300 = 300×301 PushT count supports this. |
| 15 | Episode score for Two-Room/Reacher/Cube (peak vs final); PushT is stated as peak. | Held-out episodes terminate at success, so use the score at the final step. Table A7 values (Cube 0.644 ≈ just above exp(−0.5)) are consistent with this. |
| 16 | Cube action dimensionality (5-D with yaw/gripper fixed at 0, or 3-D). | Keep the upstream 5-D action, with yaw and gripper fixed to 0 when executing and planning. CEM samples only x, y, z, so the block is 25-D with 10 dims constant. |
| 17 | PushT action rotation (A6) applies to $u_t$ in pixel coordinates. Whether normalised targets are de-normalised first is not stated. | Rotate in the simulator's native coordinates, after de-normalisation. |
| 18 | Two-Room $x,y$ units in (A7)/(A8). | Pixel coordinates, the same frame as $g$ and $d$. |
| 19 | Whether the 5-block MSE uses a sliding stride of 1 raw step or 5. | Stride 1 raw step ("every sliding window"), with keyframes 5 apart inside each window. |
| 20 | How the "±" on the best score is computed. | Sample std across the 3 runs at the episode that maximises the 3-run mean. |
| 21 | Persistence control (A6.1): sampling scheme for episode-local windows and update count per episode. | Dense stream within the episode at batch 8, $\lfloor N_{\text{ep}}/8\rfloor$ updates, at most 1 per replan. The 1-per-replan cap now binds, since 5 windows arrive per replan against a batch of 8. |
| 22 | Whether the deployment (online) score is computed with continuation past success. | Continue to the horizon, with episode score defined as for held-out episodes. Online scores matter only for Fig. A4. |
| 23 | Seeds: initial-state sampling for deployment vs held-out. | Use separate RNG streams. The held-out suite is fixed across checkpoints and runs; deployment seeds differ per run. |
| 24 | AdaJEPA's "last layer of the predictor" and "encoder head". | Last `ConditionalBlock` of the ARPredictor (or `pred_proj`) plus the encoder `projector`. Ambiguous; it only matters for the baseline. |
| 25 | Whether CEM planning runs in eval mode and under `no_grad`. | Yes to both, since the model is always in eval mode. |
