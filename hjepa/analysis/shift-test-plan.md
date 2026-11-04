# Shift test for H-JEPA: does one exist, and what we should measure

Checked against the H-JEPA repo at commit 97054ec and our JEPA-TTT notes, 2026-10-31. Points marked (inference) are my reasoning, not something stated in the papers.

## 1. Does a shift test already exist?

**There is no dynamics-shift test for H-JEPA.** The paper evaluates only in-distribution: the same dynamics, with 50 fixed start/goal tasks per environment.

What does exist:

- **The repo's "variation spaces"** (`stable_worldmodel/envs/*`) are mostly visual: colors, lighting, camera angle, cube size, and start/goal positions.
  - The one dynamics knob is FourRoom's agent speed, `variation_space['agent']['speed']` (default 5.0). It is used directly in the step function: `pos_next = pos + action * speed` (`four_room/env.py:299-300`).
  - FourRoom also randomizes door positions.
  - AntMaze, a MuJoCo ant through `ogbench/pointmaze_env.py`, has no physics knobs. Friction, mass and gear would mean editing the MjModel ourselves after reset.
- **JEPA-TTT defines 8 shifts as equations, with no code.** All of them change only how the commanded action maps to the next state:

| Environment | Shift | Equation |
|---|---|---|
| Push-T | Action rotation | Target rotated 90° about the agent |
| Push-T | Contact rotation | Block displacement from contact rotated 90° |
| Two-Room | Wave field | Action rotated by α(x) = 75°·sin(...) |
| Two-Room | Grid field | Action rotated by α(x, y) |
| Reacher | Joint-phase field | Configuration-dependent rotation, up to 155°–165° |
| Reacher | Harmonic field | Configuration-dependent rotation, up to 155°–165° |
| Cube (3-D reach variant) | Quadratic crosstalk | a + 3·(a_y·a_z, a_z·a_x, a_x·a_y) |
| Cube (3-D reach variant) | Delayed cyclic actuation | One-step delay plus axis permutation |

- Push-T and Cube overlap with H-JEPA's environments, but JEPA-TTT changed the Cube task to a 3-D reach, so its Cube numbers are not comparable.
- FourRoom is a 2-D point agent like Two-Room, so the Two-Room action-rotation fields port to it directly.

**So we write the shift test ourselves.** That is a feature: it is a new evaluation that the H-JEPA paper does not have.

## 2. Shifts to implement

All shifts change dynamics only, never visuals (unless marked as a control). They are fixed for a whole run, as in JEPA-TTT.

| Environment | Shift | How | Effort |
|---|---|---|---|
| FourRoom | Agent speed ×0.5, ×1.5, ×2 | Native variation knob | Trivial |
| FourRoom | Action rotation field, Two-Room-style α(x, y) | Wrapper on `step` | Small |
| FourRoom | Constant action rotation 45°/90° | Wrapper | Small |
| AntMaze | Floor friction ×0.3 / ×3 | `model.geom_friction` after reset | Medium |
| AntMaze | Body mass ×2 | `model.body_mass` | Medium |
| AntMaze | Actuator gear ×0.5 (weak motors) | `model.actuator_gear` | Medium |
| AntMaze | One leg disabled (gear = 0 on 2 actuators) | `model.actuator_gear` | Medium |
| AntMaze | Joint-action permutation (swap two legs' commands) | Wrapper | Small |
| Cube | Crosstalk / delay, JEPA-TTT A10/A11 adapted to H-JEPA's Cube action space | Wrapper | Small to medium |

**Controls:**

- **No shift** (sanity check). It must reproduce the paper's numbers.
- **Visual-only shift** (color or lighting, native knobs). This is a negative control. The encoder is frozen, so predictor-only TTT should *not* fix it. If it does, something is leaking.

**Severity sweep:** run each shift at 2 or 3 magnitudes. We want shifts where the frozen model drops clearly but not to 0%. FourRoom's 96% gives no headroom unshifted, but it gains headroom under shift.

## 3. Every metric

### A. Task outcome (primary)

1. **Success rate** on the 50 fixed tasks, for 3 model seeds, reported as mean ± SE, with a Wilson CI per seed.
2. **Paired per-task comparison** of shifted against unshifted, using the same task IDs and a McNemar test. With 50 tasks this is far more sensitive than comparing two rates.
3. **Final distance to the goal**, in true state space: (x, y) for the ant or the agent. This is continuous, so it detects degradation before success flips.
4. **Steps to success / path efficiency.** Our run took 175 steps on average against a 210-step budget, so a shift can push successes into timeouts.
5. **Failure type:** timeout while still progressing, stuck, or fell over (AntMaze).
6. **Success against the number of deployment episodes**, i.e. the adaptation curve, plus a normalized AUC as in JEPA-TTT. These apply to the TTT phase only; the frozen model gives a flat line.

### B. Model error, measured per level ℓ = 1, 2, 3 on shifted transitions

7. **k-step autoregressive latent MSE** at k = 1, 2, 5. JEPA-TTT uses 5-step error; we report it per level.
8. **Copy-last-latent baseline:** ‖z_t − z_{t+k}‖². The predictor must beat it, or it is not really predicting. This comes from the probing paper.
9. **Action-sensitivity gap:** the error with shuffled actions minus the error with true actions. A shift should *not* shrink this gap, and TTT should restore it.
10. **One-step teacher-forced loss**, which is the TTT loss itself, on held-out shifted windows.
11. **Predicted-versus-encoded drift.** Distance, and norm ratio, between predicted and encoded latents per rollout step. The planner compares predictions to an encoded goal, so this drift biases its cost directly.

### C. Planner / optimization

12. **Optimization error versus model error.** On failed tasks, compare J(u_ref) against J(u_found). This decides whether TTT or a looped planner comes first (see `hjepa-ttt-loop-math.md`).
   - u_ref is the dataset action sequence for that task. I haven't yet checked that the eval files store it (inference).
13. **Subgoal-tracking error per level:** the distance between the achieved lifted latent and the subgoal from the level above.
   - If upper-level subgoals become infeasible under the shift, this grows at the lower levels even when their predictors are fine.
14. **Final planning cost and GD iterations used**: does it hit early stopping or the iteration cap?

### D. Representation (does the shift break the encoder?)

15. **Linear probe from each level's latent to the true (x, y, heading)**, fit on unshifted data and tested on shifted data.
   - Visuals don't change, so this should hold (inference). If it does, any failure is in the predictor or the planner, which is the part TTT can fix.

### E. For the TTT phase (later, same harness)

16. **Recovery fraction:** (S_TTT − S_shift) / (S_clean − S_shift).
17. **No-shift control:** TTT under clean dynamics must not hurt. JEPA-TTT reports this in its Table A6.
18. **Forgetting:** success back on clean dynamics after adapting to a shift.
19. **Visual-shift negative control:** TTT should not help here (see §2).
20. **Cost:**
    - wall-clock per task;
    - planner TFLOPs per episode, since H-JEPA's headline numbers sit at ≤ 100;
    - TTT update time.

## 4. Order of runs (inference on cost)

1. **FourRoom, speed and rotation shifts, frozen H-JEPA L3.** No new physics code is needed; this is about 1 h per 50-task run on a T4. This validates the harness and all the metrics.
2. **AntMaze, friction / mass / gear / leg shifts, frozen.** This needs the MjModel edit wrapper and AntMaze per-task timing, which I haven't measured.
3. Then TTT on whichever shifts show a real drop.
