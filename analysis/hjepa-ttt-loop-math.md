# H-JEPA + JEPA-TTT + looping: the mathematically right combination (my honest view)

This is an opinion grounded in the two papers' equations and the H-JEPA code. Notation follows `hjepa-math-and-code.md` and `jepa-ttt-math-and-code.md`. Anything marked **(inference)** is my reasoning and has not been tested.

## 0. One-paragraph answer

The three pieces solve three different problems on three different time-scales, and the right combination keeps them separate:

- **H-JEPA's encoders stay frozen.** They define the latent spaces and every interface between levels.
- **JEPA-TTT adapts only the predictors**, as small, regularized, level-weighted fast weights. This fixes *model error* under a shift.
- **Looping belongs mainly in the planner, not the predictor.** The planner is already an iterative solver (90 GD iterations), and a weight-tied learned update operator is exactly a looped transformer. This fixes *optimization error* and planner cost.
- **Looping the predictor is optional and the lowest priority.** It needs a fine-tune, and I see no evidence that predictor depth is H-JEPA's bottleneck.

## 1. What each piece is, as math

**H-JEPA (fixed after pretraining).** For levels ℓ = 1..L:

- Encode the state: z^(ℓ)_t = E^(ℓ)(z^(ℓ−1) window). The encoders E are φ, frozen.
- Encode the action: u^(ℓ)_t = A^(ℓ)(u^(ℓ−1) chunk). This is the macro-action.
- Predict the next latent: ẑ^(ℓ)_{t+1} = F^(ℓ)_θℓ(z^(ℓ)_{t−2:t}, u^(ℓ)_{t−2:t}), with a history of 3.

Planning is top-down:

- Top level: min_u ‖Roll^(L)(z^(L)_0, u) − z^(L)_goal‖².
- Level ℓ < L: min_u ‖E^(ℓ+1)(Roll^(ℓ)(z^(ℓ)_0, u)) − ẑ^(ℓ+1),*‖², i.e. track the subgoal from the level above.
- Each level is solved by AdamW gradient descent on u. That is 90 iterations, seeded from the latent-action queue.

**JEPA-TTT.** Online minimization over the deployment transitions seen so far:

θ_k = argmin_θ (1/|Q_k|) Σ_{w∈Q_k} Σ_j ‖F_θ(z_{w,≤j}, b_{w,≤j}) − sg(z_{w,j+1})‖²

This is approximated by ⌊N/64⌋ AdamW steps at learning rate 5e-5. Since the encoder is frozen, the targets are constants. That is why SIGReg is unnecessary: collapse is impossible when the targets cannot move. θ then tracks argmin over E_{p_env}‖F_θ(z,b) − z′‖², the best predictor for *this* environment's transition kernel.

**Looping.** The same block f_W is applied L times with input re-injection:

h^(i) = f_W(Norm(h^(i−1)) ⊕_g Norm(x)), with optional per-loop low-rank deltas W_i = W + B_i A_i.

This adds compute depth without adding parameters.

## 2. Separate the failure modes before combining anything

A planning failure on a task decomposes into two parts.

**Model error.** The model's dynamics differ from the true dynamics. Under a dynamics shift (friction, mass, gear) this term grows. **TTT is the only piece that targets it.**

**Optimization error.** Write J_θ(u) for the planning cost. Optimization error is J_θ(u_found) − min_u J_θ(u): the solver stops short within its fixed iteration and FLOP budget. **A looped or learned planner targets this.**

Predictor looping targets neither directly. It can only reduce model error if predictor *capacity* is the limit, and the one-step losses on these simulated environments suggest it is not **(inference)**.

**A cheap diagnostic we can run on the existing checkpoints (inference).** The evaluation tasks come from dataset trajectories, so a reference action sequence u_ref exists for each task. Compare J_θ(u_ref) with J_θ(u_found) on failed tasks:

- If J_θ(u_ref) < J_θ(u_found), the model "knows" a better plan and the solver missed it. That is optimization error, so a looped planner should help.
- If J_θ(u_ref) ≥ J_θ(u_found) while the true outcome of u_ref is better, the model is wrong. That is model error, so TTT should help.

This tells us which of the two pieces is worth building first, for each environment and each shift.

## 3. TTT inside a hierarchy: what is mathematically correct

1. **Adapt predictors only. Keep every E^(ℓ) and A^(ℓ) frozen.**
   - In flat JEPA-TTT, the action encoder is adapted.
   - In H-JEPA that breaks interfaces. A^(ℓ)'s output is the input of A^(ℓ+1) (gradients flow across levels in training, as verified in code).
   - The latent-action queue holds macro-actions encoded by the *old* A.
   - The upper level's subgoals are compared through the frozen E^(ℓ+1).
   - Changing A^(ℓ) therefore silently changes the meaning of every macro-action above it and of the planner's seeds.
   - Predictor-only adaptation leaves all latent spaces, goal embeddings and seeds valid.
2. **Use a proximal, low-rank parameterization.**
   - θ_ℓ = θ_ℓ,0 + Δ_ℓ, with Δ_ℓ = B_ℓ A_ℓ at rank ≤ 8, minimizing L_TTT^(ℓ)(Δ_ℓ) + (μ/2)‖Δ_ℓ‖².
   - The proximal term is the principled replacement for JEPA-TTT's "tiny learning rate" safeguard.
   - Low rank keeps the update well-posed on a few hundred windows.
   - It is also a mechanism for reversing adaptation when the shift ends.
3. **Weight the levels by the data they actually get.**
   - One level-ℓ step spans 5·2^(ℓ−1) environment steps, so N_ℓ ≈ N_1 / 2^(ℓ−1).
   - The top level of a 3-level model gets about 4× fewer windows than level 1.
   - Adapt level 1 first and most. Adapt upper levels with a smaller step and a stronger μ, or not at all.
4. **Upper levels still need adapting under a physical shift.**
   - A macro-action u^(ℓ) is computed from the *executed* low-level actions, but the outcome it leads to changes with friction or mass.
   - So F^(ℓ) is wrong too, and its subgoals can become unreachable.
   - Adapting level 1 alone lets the agent track subgoals better, but cannot fix infeasible subgoals. This is a testable prediction **(inference)**.
5. **Match the TTT loss to how the planner uses the model.**
   - The planner uses short autoregressive rollouts plus lifted costs, but JEPA-TTT trains one-step teacher-forced.
   - The consistent loss is the h-step rollout loss Σ_{h≤H_ℓ} w_h ‖Roll_h − z_{t+h}‖², with H_ℓ the level's planning horizon.
   - For H-JEPA's short horizons the gap is small, so one-step training is a fine first version.
6. **Engineering requirements (verified in code):** call `model.zero_grad()` before each TTT step, and run in eval mode (BatchNorm, dropout, and the latent-action-queue writes in `encode()`).

## 4. Looping: where it is mathematically justified

**Planner looping (strongest).**
- GD planning is u^{k+1} = u^k − η·Adam(∇_u J_θ(u^k)), which is already a weight-tied loop.
- Replace it with a learned operator: u^{k+1} = u^k + T_ψ(u^k, ∇_u J_θ(u^k), z_0, g, level ℓ).
- T_ψ is one small transformer applied K times (K ≈ 4 to 8), trained by unrolling on the planning objective (meta-learning), with input re-injection as in the looping paper.
- Why this is the right place:
  - (a) Looped transformers are known to emulate iterative optimizers.
  - (b) H-JEPA is compared at matched planner FLOPs, and each GD iteration costs a forward and backward pass through the rollout. Cutting 90 iterations to about 8 is a real compute gain.
  - (c) **It composes with TTT.** T_ψ consumes ∇_u J_θ of the *current*, adapted θ, so when TTT changes the model, the operator still follows the new model's gradients. A pure amortized policy u = π(z_0, g) would not, because it bakes in the old dynamics.
- It needs training data for ψ: planning problems from the offline dataset. That is cheap with frozen world models, and doable on a T4 **(inference)**.

**Predictor looping (weakest).**
- F_θ becomes f_W^L with normalized, gated re-injection of z_{t−2:t} and u_{t−2:t}.
- The looping paper shows how to do it correctly (full-stack, L = 2 to 4, separate RMSNorm on each input, plus a gate), but it trains from scratch.
- On our pretrained checkpoints it requires a fine-tune, and it doubles or quadruples the planner cost, since every GD iteration rolls the predictor.
- I would only try it if the §2 diagnostic shows model error that TTT does not remove.

**Where looping and TTT meet, if predictor looping is used.** TTT should adapt only the per-loop deltas {Δ_i} (Loop-LoRA or Loop-Matrix), with the proximal term. That gives one consistent object: shared slow weights W from pretraining, plus fast per-loop low-rank weights from the test environment.

## 5. The combined system in one picture

```
slow weights (pretrained, frozen):  E^(ℓ), A^(ℓ), W^(ℓ) for every level
fast weights (TTT, online):         Δ^(ℓ) = B A, rank ≤ 8, prox-regularized, level-weighted
iteration (inference compute):      planner u^{k+1} = u^k + T_ψ(u^k, ∇_u J_{θ0+Δ}(u^k), z0, g)
                                    [optional] predictor depth loop with per-loop Δ_i
```

## 6. What I'd claim and in what order

1. **AntMaze shift, unadapted:** measure the success drop and run the §2 diagnostic. This tells us whether model error dominates under the shift.
2. **Predictor-only, low-rank, proximal TTT at level 1, then at all levels:** this recovers success under the shift. It is the most likely real result, and it is cheap on a T4.
3. **Learned looped planner at matched planner FLOPs**, then the same planner on top of TTT. If it holds up, this is the novel part.
4. **Predictor looping only if steps 1 to 3 leave model error on the table.**

**Honest risks:**
- TTT gains could be small if H-JEPA is already robust to the shifts we pick. That is why step 1 comes first.
- A learned planner may not beat 90 AdamW iterations at equal FLOPs on small problems.
- Everything here is single-environment simulation. Claims should be "under shift X on AntMaze", not general.
