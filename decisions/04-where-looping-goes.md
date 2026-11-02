# Decision 04: Where looping goes in H-JEPA, and why

**Dates:** 2026-10-20 (feasibility), 2026-10-31 (three papers read in full)

**Details:**
- [`analysis/hjepa-looping-feasibility.md`](../analysis/hjepa-looping-feasibility.md)
- [`analysis/how-to-loop-transformer-analysis.md`](../analysis/how-to-loop-transformer-analysis.md)
- [`analysis/stateloop-analysis.md`](../analysis/stateloop-analysis.md)
- [`analysis/probing-spatial-state-analysis.md`](../analysis/probing-spatial-state-analysis.md)

## The question

Looped transformers reuse one block K times, so depth becomes adjustable compute at fixed size. This is an emerging approach in LLMs. Could it help H-JEPA, and where?

## Decision, ranked

1. **Loop the planner first.**
   - H-JEPA's planner is already a loop: 90 AdamW steps on the action sequence.
   - A small transformer applied about 8 times as a learned update step is literally a looped transformer.
   - Form: u^{k+1} = u^k + T_ψ(u^k, ∇_u J_θ(u^k), z₀, g).
2. **Second: adaptive looping of the upper-level predictors.** A 20-env-step jump is a "hard token", and the literature finds that extra loops help hard items, not easy ones.
3. **Last, and only if real model error remains: uniform looping of every predictor.**
4. **Rejected: merging the levels into one HRM-style loop.** That rebuilds HWM, which lost to H-JEPA on AntMaze (38.7% against 73.3%).

## Why the planner and not the predictor

- **What limits H-JEPA is getting stuck in gradient planning and long horizons, not predictor capacity.** The predictors are small (4–6 layers, width 192), and nothing we've seen suggests depth is the bottleneck.
- **Looping a predictor multiplies planning cost.** The planner backpropagates through every rollout step, for every candidate, for up to 90 iterations. Two loops roughly doubles FourRoom's about 65 s per task.
- **Contractive loops are stable but shrink action gradients**, which are exactly what the planner needs.
- **A looped planner composes with TTT.** It reads gradients of the *current, adapted* model, so it keeps working as TTT changes the model.

## What each paper contributed

**How to Loop Your Transformer (2610.40325)**
- Loop the whole stack, 2–4 times; at 32 loops small models stall or go unstable.
- Re-inject the input at every loop, normalized separately from the hidden state.
- Per-loop LoRA (rank ≤ 8) helps at 4 or more loops.
- More real layers beat more loops at the same compute.
- **For us:** if we ever loop a predictor, loop the whole thing 2–4 times and re-inject z_t and the action each loop. The per-loop LoRA is a natural TTT target.
- **Weaknesses:** language modeling only, mostly one run per setting, never loops a model that was trained without loops (our situation), so a fine-tune would be needed.

**StateLoop (2610.35544)**
- Each token carries a gated moving-average state across loops: S ← g⊙S + (1−g)⊙P.
- Perplexity 7.79 → 7.61 at 1.4B; the gain grows with loop count; resetting the state is worse.
- **For us:** this update has the form of Adam's momentum, m ← β·m + (1−β)·grad. Adam is the special case with a fixed gate.
- So the looped planner gets a small per-action-step state that acts as a learned Adam moment. We initialize it to behave exactly like AdamW, so it starts at the baseline, and carry the state across replans.

**Probing Spatial State in VFMs (2610.40636)** (not about looping, but about testing honestly)
- V-JEPA2's predictor lost to copy-last-frame (13.0% against 19.9%).
- Most predictions barely changed with the action.
- Predicted features drift from real encoder features.
- **For us:** every prediction claim must report copy-last-latent and shuffled-action baselines and predicted-versus-encoded drift. All three are in the shift benchmark.

## Training the looped planner

It needs offline training: unroll the planner on planning problems sampled from the dataset, with the world model frozen.

- **Stabilize it as plain gradient descent plus a learned correction that starts at zero**, so at step 0 it is exactly the current planner.
- **Train it against slightly perturbed world models**, so it still works after TTT changes the model.
- **It must win at equal planning compute.** The headline would be success against planner compute.
