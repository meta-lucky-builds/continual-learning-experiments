# "StateLoop: Persistent State for Looped Transformers" (arXiv 2610.35544): deep read and relevance to H-JEPA

**Paper:** Song, Zeng, Li, … , Lin. Shanghai Jiao Tong University (LUMIA Lab), 26 Oct 2026. https://arxiv.org/abs/2610.35544

**Coverage:** I read the full text: main body, ablations, analysis and conclusion. The paper has no appendix. Points marked **(inference)** are my reasoning, not the paper's.

## 1. The method, in math

The setup is a looped decoder-only LM. A shared backbone T (24 layers at 1.4B) is applied K times. Each token additionally carries a persistent state S ∈ R^{d_s}, which is updated across loop iterations.

**Initialisation.** E = √d·Embed(x), S⁰ = 0, H⁰ = T(E).

**At each iteration k:**

| Step | Equation |
|---|---|
| Normalise the backbone output | U^k = RMSNorm(H^k) |
| Write projection | P^k = U^k W_uᵀ |
| Input-dependent step size | Δ^k = softplus(U^k W_Δᵀ + b_Δ) |
| Per-channel rate | λ = exp(A_log) |
| Retention gate | g^k = exp(−λ ⊙ Δ^k) |
| State update | S̃^{k+1} = g^k ⊙ S^k + (1 − g^k) ⊙ P^k |
| Optional mixing across tokens | S^{k+1} = X_ψ(S̃^{k+1}): identity, causal attention, or Mamba-2 |
| Next backbone input | Z^{k+1} = H^k + RMSNorm(S^{k+1}) W_{s→z}ᵀ, then H^{k+1} = T(Z^{k+1}) |
| Final readout | Y = H^{K−1} + RMSNorm(S^K) W_{s→out}ᵀ, then the LM head |

**How to read it.**
- The state is a selective exponential moving average, run along the **loop axis** independently for each token.
- Retention and write gates are tied (g and 1−g).
- The decay rate depends on the input.
- Every parameter is shared across iterations.
- **Initialisation:**
  - W_Δ = 0 and b_Δ is chosen so that Δ = 0.3.
  - λ is spread log-uniformly over [0.01, 4] across channels.
  - So the channels start with write rates 1−exp(−0.3λ), roughly 0.003 to 0.7. That is a built-in spread of time-scales.
- **The direct path is kept.** The next input is H^k plus a state read. The state does not replace H^k.
- **The embedding E is not re-injected** at later iterations.

## 2. Results

**1.4B scale, 30B Pile tokens, K = 4, against their own Vanilla Looped:**

| Model | Params | Pile PPL | Zero-shot Avg-8 | Five-shot Avg-8 |
|---|---|---|---|---|
| Vanilla Looped | 1420.9M | 7.79 | 53.31 | 55.12 |
| Vanilla Looped + FFN (parameter-matched) | +3.3% | 7.75 | 52.85 | 55.54 |
| MeSH (strongest baseline) | +0.04% | 7.74 | 53.87 | 56.50 |
| StateLoop | +1.2% | **7.61** | 54.28 | 56.60 |
| StateLoop + Attn mixer | +2.4% | 7.60 | 54.74 | 57.15 |
| StateLoop + Mamba-2 mixer | +3.3% | 7.60 | 54.75 | 57.13 |

**Loop depth, at 410M with d = d_s = 1024.** The perplexity gap over Vanilla Looped grows with K. These are read from Fig. 2, so the axis values are approximate:
- K = 3: about 9.45 against 9.59.
- K = 7: 9.04 against 9.34.
- Vanilla appears to plateau from K = 5 to 7, while StateLoop keeps improving.

**Ablations, at 410M with K = 7:**

| Ablation | Pile PPL |
|---|---|
| Vanilla Looped | 9.34 |
| State only (state replaces H as the next input) | 9.44, worse than vanilla |
| Reset state every iteration (no persistence) | 9.25 |
| StateLoop with a GRU update instead of the selective SSM | 9.09 |
| StateLoop | **9.04** |

- **State width.** d_s = 128 already gives 9.16, about 60% of the gain at about 0.1% extra parameters. d_s = 512 gives 9.08.
- **Decoding cost.** +1.4 to 2.1% latency at K = 4, against +25% for adding a fifth loop.
- **Analysis on the trained model:**
  - Exact Shapley attribution over all 2⁷ read masks: the last reads (R5 to R7) carry most of the value. Feedback reads account for 64% and the terminal read for 36%.
  - Freezing the state at S³ raises perplexity to 18.4. A single calibrated replacement of S⁴ by S³ gives 9.08.
  - So nearby states are interchangeable, but the state has to keep updating.

## 3. How much to trust it

**Strengths:**
- All models are trained in one codebase.
- There is a parameter-matched FFN control.
- Persistence is isolated cleanly: reset versus persistent at equal parameters.
- The update rule is checked with both an SSM and a GRU.
- Latency is actually measured.
- The Shapley analysis is exact, not sampled.

**Weaknesses:**
1. **One seed per configuration, and no error bars.** Downstream deltas are about 1 pp on Avg-8. Single tasks such as ARC-C, with about 1,170 questions, have a standard error of roughly 1.3 pp.
   - The noise floor is visible in the paper itself: widening the FFN *lowers* zero-shot Avg-8 by 0.46 while improving perplexity.
   - The perplexity gains (−0.18 at 1.4B, −0.30 at 410M with K = 7) are more trustworthy than the accuracy gains.
2. **No non-looped baseline.** No dense model is compared at equal parameters or equal FLOPs. The paper only shows that StateLoop beats other looped models, not that looping is worth it.
3. **No input re-injection baseline.** "How to Loop Your Transformer" found that re-injecting the normalised embedding at every loop is the strongest simple fix. StateLoop never compares against it.
   - Part of StateLoop's gain may be the same thing: giving each iteration a second, scale-controlled input channel. **(inference)**
4. **Baselines are reimplementations.** Partial-loop baselines (Hyperloop, MeSH, Parcae) keep their own prelude/core/coda structure, so they are not like-for-like with full-backbone looping.
5. **Scope.** Language modelling only, from scratch, K ≤ 7, up to 1.4B. Nothing on extrapolating beyond the trained depth or on adaptive halting.

## 4. What it means for our H-JEPA combination (inference)

### 4.1 The looped planner: the most useful link

In `hjepa-ttt-loop-math.md` the planner loop is a learned operator applied K times to the action sequence u:

u^{k+1} = u^k + T_ψ(u^k, ∇_u J_θ(u^k), z₀, g)

StateLoop's update has *exactly* the form of an optimiser's moment estimate:

| StateLoop | Adam |
|---|---|
| S^{k+1} = g ⊙ S^k + (1−g) ⊙ P^k | m^{k+1} = β₁ m^k + (1−β₁) ∇J |

- Adam is the special case with a fixed g = β₁, and with P set to the raw gradient.
- StateLoop's version adds a learned, input-dependent gate, many channels with different time-scales, and a learned write projection.

**So the principled design for the learned planner is a looped T_ψ plus a per-action-token persistent state.** That state is a learned generalisation of Adam's moment buffers. Concretely:

- **Initialise to match AdamW.** Set the gates so the state behaves like an EMA with H-JEPA's planner betas. With T_ψ's correction initialised to zero, step 0 is then exactly the current AdamW planner. This is the "starts as the baseline" rule from before.
- **Keep the direct path.** The direct update must stay; the state is an extra read. State-only feedback was *worse than vanilla* in their ablation.
- **Use a small state.** d_s = 128 already captured most of the gain. Our planner tokens are 4-dimensional macro-actions or raw actions, so a state of a few dozen channels is likely enough.
- **Let the state persist across replanning steps (warm start).** Carrying S across receding-horizon replans is the analogue of their "persistence beats reset" result. It is also cheap.
- **A token-axis mixer here means mixing along the plan horizon.** It would let state at action step h see steps ≤ h. A causal or bidirectional attention layer over the plan is the natural choice.

This is a stronger and more specific design than the generic "learned optimiser" I wrote before. It also keeps the property that matters for TTT: the operator reads ∇J of the *current* adapted model.

### 4.2 Predictor looping (still the lowest priority)

If we ever loop the H-JEPA predictor, the two papers give complementary pieces that nobody has tested together:
- re-inject the anchors (z_t, u_t) with separate RMSNorm and a gate, from 2610.40325;
- carry a small persistent state with an RMSNorm read back into the stream, from this paper.

Both papers normalise every branch before adding it to the stream. That is the one consistent lesson for scale control in recurrent depth.

### 4.3 Interaction with TTT

- The persistent state is activations, not weights, so it does not conflict with predictor TTT.
- Freeze the state-branch parameters (W_u, W_Δ, A_log, W_{s→z}) at test time. Adapt only the predictor's low-rank Δ, so the planner's "optimiser dynamics" stay fixed while the model adapts.
- Their freezing result (the state must keep updating, or perplexity jumps from 9 to 18) argues against reusing a stale planner state after TTT changes the model a lot. Reset S when Δ moves beyond a threshold.

## 5. Bottom line

- **What it is:** a clean, cheap architectural idea with good internal controls, but small, single-seed gains and missing baselines (no dense comparison, no input re-injection).
- **Its main value for us:** it shows how to give our looped planner memory across iterations. A gated, selective EMA state per action token is a learned generalisation of Adam's moments. It can be initialised to equal the current AdamW planner and only improve from there.
- **Where it goes in the plan:** this goes into the planner design. It does not change the order of the plan: shift test → predictor-only TTT → looped planner.
