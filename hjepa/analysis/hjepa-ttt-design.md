# Implementing JEPA-TTT inside H-JEPA: a first-principles design

Sources read in full: `h-jepa.pdf` (Zhang, Terver, Rabbat, LeCun, Balestriero, arXiv 2610.06805, incl. appendices A–M) and `jepa-tttt.pdf` (Zhang, Ye, et al., arXiv 2610.00722, incl. appendices A1–A7). Section and equation numbers below refer to those papers. Anything marked *hypothesis* is my inference, not a result from either paper.

---

## TL;DR

1. **Level 1 of H-JEPA is the exact model JEPA-TTT adapts.** Both papers use LeWorldModel (ViT-Tiny + causal transformer predictor + SIGReg, same codebase). So JEPA-TTT applies verbatim at level 1: freeze the encoder, update the action encoder and predictor with a stop-gradient latent prediction loss, use dense windows, replay, and persistence.
2. **A dynamics shift reaches every level**, because each upper level predicts how the world changes under a macro-action that is ultimately a pooled function of raw actions. However, **not every level needs re-learning.** A level needs adaptation only if *its own* transition kernel, expressed in its own latent space, changes. Since higher levels discard fast detail, many shifts are invisible at the top.
3. **The hierarchy gives a natural place to absorb the shift.** In the simulated H-JEPA models, upper action encoders pool *level-1 action embeddings*. If the level-1 action encoder `A(1)` adapts to undo a shift such as an action rotation, the correction propagates up to every level for free. If the shift instead changes *what is reachable* (speed, friction, contact, obstacles), the upper predictors must adapt too.
4. **Freeze every encoder `E(ℓ)` and the goal cost, and adapt only predictors (plus `A(1)`).** In H-JEPA the encoders are the TTT targets, the goal embeddings, *and* the "lift" used to compare lower rollouts with upper subgoals. Moving any of them silently changes all three. With encoders frozen, SIGReg and IDM are unnecessary at test time because nothing can collapse.
5. **Upper levels are data-starved at test time, so dense windows matter even more there.** A level-3 window spans 61 env steps, while JEPA-TTT's episodes were 50 steps. Use every env-step offset (5·∏s offsets per upper step) and, because H-JEPA plans with context 1, use one-step pair windows at upper levels.
6. **A robust fallback exists.** H-JEPA's "projected-cost" flat planner (Table 1) needs only the adapted level-1 predictor plus frozen upper encoders. It cannot be hurt by stale upper predictors, so it is the safe planner while upper levels catch up.

---

## 1. What each paper actually does

### H-JEPA (hierarchical JEPA world model)

| Piece | What it is |
|---|---|
| Level 1 | LeWM: `z(1)=E(1)(o)` (ViT-Tiny CLS → MLP+BN projector, plus proprio on AntMaze), action encoder `A(1)` on a 5-step action block, causal transformer predictor `F(1)` with AdaLN action conditioning |
| Level ℓ>1 | `z(ℓ)_t = E(ℓ)(z(ℓ-1)_{t·s})` (2-layer MLP, window w=1 so pointwise); `a(ℓ)_t = A(ℓ)(s lower action embeddings)` (1-layer transformer → 4-d macro-action); `F(ℓ)` causal transformer. Stride s=2, so each level doubles the time per step (5, 10, 20, 40 env steps) |
| Objective (Eq. 2–3) | Per level: teacher-forced one-step latent MSE + λ·SIGReg, summed over levels, **end-to-end**. Upper-level gradients flow into lower encoders. No stop-gradient and no EMA. DROID adds an IDM term (Eq. 8) against slow-feature collapse |
| What emerges (§3) | When the data has separated timescales (≥ ~2× frequency gap), upper levels drop fast factors (ant leg pose, teleporting distractor) and keep slow ones (position). Push-T and Cube show no such separation |
| Planning (§2.2, App. E) | Goal image encoded at every level. The top level minimises ‖ẑ_H − g(L)‖² over macro-actions. Each lower level matches the first upper subgoal after **lifting** its own rollout through `E(ℓ+1)` (Eq. 5–6, K=1 closed loop). The solver is gradient descent (AdamW, ≤90 iterations) on actions. Upper macro-actions are initialised from, and clipped to the 2nd–98th percentiles of, the **last 4000 training macro-actions**. Replanning happens every 10 env steps, with **planning context c=1** at every level |
| Why it helps (§4.2) | (a) Abstract goal costs are smoother (Fig. 7, Table 1: level-1 dynamics with an upper-projected cost already beats native cost). (b) Temporal decomposition: subgoal-tracking cost is monotone (Spearman 0.89–1.00 vs 0.48–0.75) |

### JEPA-TTT (persistent test-time training)

| Piece | What it is |
|---|---|
| Setting | One fixed shift `P_test ≠ P_train`. Observation map, task, and action space are unchanged. No goal image and no online reward |
| What adapts | Action encoder, predictor, and prediction projection (θ). **Frozen:** image encoder, latent projector, reward head, BN stats (eval mode) |
| Loss (Eq. 4) | Teacher-forced latent MSE over C=3 context positions, with targets `sg(E(o))` |
| Data | **Dense windows** start at every raw env step (all K=5 offsets inside an action block), giving ~5× more windows than block-aligned windows. A **growing buffer** is sampled uniformly. Batch 64, ≤1 AdamW step per replan, lr 5e-5 |
| Persistence | θ, optimizer state, and buffer carry across episodes |
| Planner | CEM (256×6, 32 elites, H=5 blocks) maximising a frozen, offline-fitted reward head on predicted latents |
| Results | +153% best score and −83% 5-step latent MSE over 500 episodes. **Persistence is the biggest lever** (0.289 → 0.729 best score at matched updates). Dense coverage is next, and replay gives a small extra gain. TTT under unchanged dynamics neither helps nor hurts (App. A6.2) |
| Stated limits | Visual or reward shifts are not handled because the encoder is frozen. Forgetting and sequences of shifts are not studied |

---

## 2. Component mapping

| JEPA-TTT concept | H-JEPA counterpart | Notes |
|---|---|---|
| Visual encoder `Eϕ` (frozen) | `E(1)` **and** every `E(ℓ)`, ℓ≥2 | In H-JEPA the encoders also define the goal cost and the lift in Eq. 5, so freezing them is even more important |
| Predictor `Fθ` (+ action encoder, pred. projection) | `F(ℓ)`, `A(ℓ)` per level | There are L predictors to adapt. The upper "actions" are learned embeddings, not environment actions |
| Window `(z_0..z_C, b_0..b_{C-1})` | Level-ℓ window of `c_ℓ+1` level-ℓ latents and macro-actions | Upper windows span `(c+1)·5·∏s` env steps |
| Dense windows (every env-step offset) | The same idea at every level, with `5·∏_{k≤ℓ} s_k` offsets per level-ℓ step | Pointwise frozen encoders mean every frame has a latent at every level, so dense upper windows cost nothing extra |
| Reward head `Rω*` + CEM | Goal-image latent cost + gradient planner, top-down | TTT is planner-agnostic. Keep H-JEPA's goal cost unless you need the goal-free setting (§6.4) |
| Planning context C=3 | c=1 at every level | Matters for history-dependent shifts (§6.6) |
| Episode horizon 50 to 300 steps | Budgets of 30 (Cube) to 350 (AntMaze) steps | Short episodes starve upper levels, the same failure H-JEPA reports for 3-level Push-T (Table 3) |

---

## 3. First principles: what does a dynamics shift do to a hierarchy?

Write the true raw dynamics as `s' ~ P(·|s,a)`. Level ℓ models

`z(ℓ)_{t+1} ≈ F(ℓ)( z(ℓ)_t , A(ℓ)(A(ℓ-1)(…A(1)(a_{raw block})…)) )`.

A shift changes the map from *raw action sequences* to *future states*, so in principle every level's predictions break. That leads to three observations.

**(i) A level only needs fixing if its own transition kernel changes.** In planning, an upper level is used for one thing: proposing subgoals `ẑ(ℓ+1)*` that are reachable within one upper step. The macro-action itself is never executed, and it is only a free search variable for the upper planner. So what matters at level ℓ is the set of level-ℓ states reachable in one step and how they are parameterised.

**(ii) Shifts fall into classes with different "home levels".**

| Shift class | Example (from JEPA-TTT's suite or analogous) | Reachable set at upper levels | Where adaptation is needed |
|---|---|---|---|
| State-independent action reparameterisation | PushT 90° action rotation; Cube axis routing | Unchanged | **`A(1)` only, in principle.** One linear map undoes it (*hypothesis*) |
| State-dependent reparameterisation | Two-Room wave/grid field; Reacher phase fields | Mostly unchanged | `F(1)` (it needs the state). `A(1)` alone cannot fix it because `A(1)` sees only actions |
| Reachability / capability change | Speed or gain scaling, friction, blocked doors, contact rotation (changes which block poses follow from which pushes) | **Changes** | `F(1)` **and** the upper `F(ℓ)`s, or the upper planner proposes subgoals level 1 cannot reach |
| Temporal / non-Markov | Delayed actuation | ≈ unchanged (a 1-step delay is 1/20 of a level-3 step) | Level 1, and it needs history context (§6.6) |
| Fast-factor-only change | AntMaze leg dynamics that don't change the gait's net speed | Invisible at levels where those factors were abstracted away | Level 1 only |

**(iii) The correction propagates upward through the action encoders.** In the simulated H-JEPA models, `A(ℓ)` for ℓ>1 pools the *level-1 action embeddings* `a(1)`. If TTT drives `A(1)` toward `A(1)'(a) ≈ A(1)(R·a)`, so that the embedding of a raw action again means "what that action did in training", then every macro-action computed from executed actions already means the same thing as before. The upper predictors, and the upper planners' training-based clip boxes, stay valid without touching them. This argues for including `A(1)` in the TTT parameters (JEPA-TTT already does) and for testing an `A(1)`-only variant. **Caveat:** on DROID, `A(2)` consumes raw commands directly, so this free propagation does not happen there.

The consequence is that adaptation should be **bottom-heavy and gated**. Always adapt level 1. Adapt upper predictors only when their own online prediction error rises above their training-dynamics baseline.

---

## 4. Recommended design (minimal version first)

### 4.1 Parameters

| | Level 1 | Levels ℓ ≥ 2 |
|---|---|---|
| Encoder `E(ℓ)`, projectors, BN stats | frozen | frozen |
| Action encoder `A(ℓ)` | **train** | **frozen** by default (keeps the planner's macro-action prior valid, §6.3) |
| Predictor `F(ℓ)` + prediction projection | **train** | **train**, gated by error (§4.4) |
| SIGReg / IDM | off (frozen targets cannot collapse) | off |

### 4.2 Loss

Apply JEPA-TTT Eq. 4 per level, each in its own latent space:

`L_TTT = Σ_ℓ w_ℓ · (1/(|B_ℓ| c_ℓ d_ℓ)) Σ_{w∈B_ℓ} Σ_j ‖ F(ℓ)(z(ℓ)_{w,≤j}, a(ℓ)_{w,≤j}) − sg(z(ℓ)_{w,j+1}) ‖²`

with **no cross-level gradient**. The macro-action fed to `F(ℓ)` is computed from executed actions through `A(1..ℓ)` under `no_grad`, so sparse, noisy upper losses cannot distort level-1 action semantics. This is the opposite of H-JEPA pretraining, which is end-to-end, and it is deliberate: at test time the encoders are fixed and each level should be fit only to its own evidence. (Letting the upper losses also train `A(1)` is a legitimate open choice, §6.2.)

### 4.3 Data: one raw buffer, per-level dense windows

- Store raw `(o_t, a_t)` trajectories persistently across episodes, as JEPA-TTT does. Because the encoders are frozen, **cache `z(ℓ)_t` for every frame at every level** once. Recompute action and macro-action embeddings at sample time, because `A(1)` changes.
- A level-ℓ window is valid at **every env step t** whose span fits in the episode: `(c_ℓ+1)` latents at env offsets `t, t+Δ_ℓ, …` with `Δ_ℓ = 5·∏_{k=2..ℓ} s_k` (5, 10, 20, 40 env steps).
- **Use pair windows (c_ℓ = 1) at upper levels.** H-JEPA plans with context 1 at every level, and the causal mask means c=1 was covered in training. A pair window at level 3 needs 21 env steps instead of 61, which roughly triples usable upper data in 50 to 90-step episodes. Keep c=3 at level 1, matching both papers.
- As in JEPA-TTT, keep acting until the episode horizon even after success, so the data supply doesn't depend on how good the current model is.

### 4.4 Schedule

- Per level, follow JEPA-TTT's rule: one AdamW step whenever a full batch (64) of *new* level-ℓ windows has accumulated, sampling uniformly from that level's whole buffer. Each upper level then updates more slowly on its own, at a rate proportional to its data supply.
- Learning rates: level 1 at 5e-5 (JEPA-TTT). Upper levels at 1e-5, which mirrors H-JEPA's pretraining ratio (5e-5 vs 1e-5, App. C.1).
- **Gate:** keep a running mean of each level's online one-step loss. Enable updates for level ℓ≥2 only when it exceeds that level's held-out training-dynamics loss by a margin. This avoids touching levels the shift doesn't reach. JEPA-TTT's no-shift control (App. A6.2) suggests ungated updates are roughly harmless, so the gate mainly saves compute and limits forgetting.
- Persist θ, optimizer state, and buffers across episodes. This was JEPA-TTT's largest effect.

### 4.5 Planning during adaptation

- Keep H-JEPA's top-down goal-conditioned gradient planner. The goal cost stays correct under a dynamics-only shift because the encoders and the goal image are unchanged.
- **Fallback while upper levels are stale:** use flat level-1 planning with the *upper-projected goal cost* (Table 1). It uses only `F(1)`, which adapts fastest, plus frozen `E(2..L)` for the cost, and on AntMaze it already beats flat LeWM. Switch to full hierarchical planning once the upper-level online error is back near baseline.
- Monitor **subgoal-tracking monotonicity**, the online version of H-JEPA's Table 10 (Spearman of level-1 cost-to-subgoal vs time within each segment). If it drops, the upper levels are proposing subgoals that level 1 can no longer reach. That is a direct, label-free signal that upper adaptation is needed.

### 4.6 Pseudocode

```
init θ(1..L) ← pretrained; A(ℓ≥2), E(·) frozen; buffers Q ← ∅; optimizers persist
for each test episode:
  for each replan step (every 10 env steps):
     plan = HJEPA_topdown(o_t, goal; F(1..L))           # or projected-cost flat planner while gated levels are stale
     execute first 2 level-1 blocks; append raw (o,a) to Q; cache z(ℓ) for new frames
     for ℓ in 1..L:
        add all newly valid dense level-ℓ windows to Q_ℓ   # every env-step offset
        if gate_open(ℓ) and new_windows_ℓ ≥ 64:
           B ← uniform_sample(Q_ℓ, 64)
           macro ← no_grad(A(ℓ)∘…∘A(1))(raw actions)      # for ℓ=1 this is A(1) with grad
           step AdamW_ℓ on L_TTT^(ℓ)(B)
# θ, optimizer state, Q persist across episodes
```

---

## 5. Extensions worth trying after the baseline works

1. **Upward distillation for data-starved upper levels** (*hypothesis*). Level 1 sees far more windows than level 3. Once `F(1)` has adapted, generate imagined lower rollouts from buffer states and train `F(ℓ+1)(z(ℓ+1), A(ℓ+1)(a(ℓ)))` toward `E(ℓ+1)(Roll(ℓ)(z(ℓ), a))`. Upper levels then learn the shifted dynamics from imagination. **Check first** that this composition matches `F(ℓ+1)` under the *training* dynamics, because H-JEPA never trained for cross-level consistency. If they disagree before the shift, the distillation target is biased. Weight it below the real-data loss.
2. **`A(1)`-only TTT.** It has the fewest parameters, carries the least forgetting risk, and propagates upward for free. It should fully recover state-independent reparameterisation shifts and fail on state-dependent ones. That makes it a clean diagnostic of shift class.
3. **Adapt `A(ℓ≥2)` too**, but then refresh each upper action encoder's "last 4000 macro-actions" buffer with macro-actions computed from *online* executed actions. Otherwise the planner's initialisation Gaussian and clip box are stale and it searches the wrong region.
4. **Forgetting control.** JEPA-TTT flags this as unstudied. The hierarchy suggests per-level residual adapters (or LoRA) on `F(ℓ)` that can be kept, reset, or swapped per dynamics regime. Slow upper levels probably need little change and can be regularised toward pretrained weights more strongly than level 1.

---

## 6. Open design choices (with my default)

| # | Choice | Options | Default and reason |
|---|---|---|---|
| 6.1 | Which levels adapt | L1 only / all / gated | **Gated** (L1 always). The shift's home level depends on its class (§3) |
| 6.2 | Cross-level gradients at TTT | Detached / end-to-end like pretraining | **Detached.** Sparse upper losses shouldn't rewrite level-1 action semantics. End-to-end is an ablation |
| 6.3 | Upper action encoders | Frozen / trained + prior refresh | **Frozen.** Keeps the planner's macro-action box valid, and `A(1)` adaptation already propagates up |
| 6.4 | Planning objective | Goal-image latent cost (H-JEPA) / frozen reward head (JEPA-TTT) | **Goal cost** when a goal image exists. For the goal-free setting, fit the reward head on the **top-level** latent `z(L)`, since slow task-relevant state lives there, and use it as the top-level cost. Lower levels still track subgoals |
| 6.5 | Planner | Gradient (H-JEPA) / CEM (JEPA-TTT) | Keep **gradient**, but watch for this risk (*hypothesis*): a briefly adapted predictor can fit values before its action-Jacobians are smooth, which hurts gradient planning more than CEM. Compare against CEM at level 1 early in deployment |
| 6.6 | Planning context | c=1 (H-JEPA) / c=3 (JEPA-TTT) | **c=3 at level 1.** With c=1 a delayed-actuation shift is unlearnable, because the model never sees the previous action |
| 6.7 | Upper window length | c=3 / c=1 pairs | **Pairs**, for 3× more upper windows in short episodes |
| 6.8 | Learning rate per level | Uniform / scaled | **Upper = level-1 / 5**, mirroring pretraining |
| 6.9 | Encoder adaptation | Never / with SIGReg+IDM | **Never** for dynamics-only shifts. A visual shift would need it, and then the goal embeddings, the lift, and the macro-action priors all move together, which is a separate problem |

---

## 7. Experiments that would test this

- **Environments:** AntMaze and FourRoom, where H-JEPA's hierarchy helps most and selective abstraction exists, plus PushT and Cube with JEPA-TTT's exact shifts so both papers' numbers serve as anchors.
- **Shifts covering the four classes:** FourRoom action rotation (state-independent), FourRoom spatial wave field (state-dependent, ported from Two-Room), FourRoom/AntMaze speed or gain halving (reachability), AntMaze one-step actuation delay (temporal).
- **Arms:** frozen H-JEPA; L1-only TTT; `A(1)`-only TTT; all-levels TTT; gated TTT; flat LeWM + JEPA-TTT (the paper's method); projected-cost fallback planner.
- **Core prediction to falsify:** on reparameterisation shifts, L1-only ≈ all-levels. On reachability shifts, all-levels > L1-only, and the gap shows up as falling subgoal-tracking monotonicity.
- **Metrics:** success and AUC over 500 persistent episodes (JEPA-TTT protocol, fixed held-out set every 50 episodes); per-level 5-step autoregressive latent MSE before and after; online subgoal-tracking Spearman; planner TFLOPs (H-JEPA's axis); TTT compute per replan.
- **Watch:** 3-level Push-T will be data-starved for level 3, as in H-JEPA Table 3. Pair windows and the projected-cost fallback are the mitigations.
