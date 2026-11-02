# Test 1: World-model geometry, Lucky's table vs the earlier combination

**When:** 2026-10-23, thread "Repository review and connections".
**Files in this folder:**
- `wmtest.py` (code)
- `result_seed0.json`, `result_seed1.json`, `result_seed2.json` (raw numbers)
- `RESULTS.md` (the original write-up from the day)

## 1. Why we ran it

The test grew out of one conversation about JEPA world models:

1. **What the geometry of a JEPA world model should be.** Lucky asked this after reading `geomoflearn.md`, the CL files and the brain files. My answer was that no single geometry is best: each job of the model needs its own. I proposed this combination:
   - a sphere-plus-radius state;
   - **structured port-Hamiltonian / phase-space dynamics**, where the state is (position, momentum);
   - a separate **quasimetric planning head**;
   - a **precision-weighted belief** that fuses prediction with evidence;
   - Fisher-geometry updates for learning over time.
2. **Lucky's own design.** Lucky then proposed a more conservative table:

   | Component | Geometry |
   |---|---|
   | Recurrent state | Euclidean coordinates |
   | Prediction projection | Gaussian-regularized Euclidean space |
   | Belief | Diagonal Gaussian |
   | Planning head | Directed quasimetric |
   | Physical submodule | Configuration space and its tangent bundle (TQ) |

3. **My review of the table.** I said it was a good v1. I flagged one inconsistency: a diagonal belief in a basis that is only defined up to a rotation. I also listed four gaps.
4. **The ask.** Lucky asked: *"run the smallest test which represents this, compare it to the combination, and make it the smallest test that represents them accurately."*

So the purpose was a **first empirical comparison of the two designs on one shared task**, small enough to run on CPU in minutes.

## 2. What we built

### The toy world: smallest world that exercises every component

| Feature of the world | Which component it tests |
|---|---|
| 2D point mass with damping; actions are forces in [-1, 1]² | dynamics |
| **Velocity is hidden.** Observations show position only, as a fixed random 16-d `tanh` view plus noise σ = 0.05 | recurrent state / phase space; whether the model must infer velocity |
| 8 pure-noise distractor channels | anti-collapse; the JEPA "ignore the unpredictable" property |
| Random 6-step occlusions (about 25% of frames), with an occlusion flag | belief: it should widen when there is no evidence |
| A one-way "river" in \|x\| < 0.25 pushing +y at 2.0, stronger than the maximum action | asymmetric reachability, which is what a quasimetric is for |
| Walls at ±1 | none (keeps the state bounded) |

**Data:**
- 3,000 training episodes of 64 steps with smooth random (OU) actions.
- 400 separate test episodes.

### The two models (same data, same losses where the designs agree)

Shared by both models:
- **Losses:** the JEPA loss (prior state predicts the stop-grad embedding of the next observation), an 8-step open-loop embedding loss, SIGReg (a sliced Epps–Pulley test toward N(0, I)) on the encoder output, and a belief NLL over the next observation.
- **Grounding:** the same 5% of frames carry true position labels.
- **Planner:** the same quasimetric trainer, QRL-style (maximise random-pair distance subject to one-step distance ≤ 1, with a Lagrange multiplier), using an MRN-style head (max-ReLU asymmetric part plus a Euclidean part).

| | **A: Lucky's table** | **B: earlier combination** |
|---|---|---|
| State | 64-d Euclidean GRU: a posterior GRU that sees observations, and a prior GRU for imagination with a consistency loss | Phase space (q, p), each 4-d |
| Dynamics | Free GRU | Control-affine, explicit PSD damping, symplectic Euler: p' = p + dt(F(q) − D(q)p + G(q)u), q' = q + dt p' |
| Evidence fusion | The GRU learns it | Learned Kalman-like gain on the position innovation, zeroed when occluded (precision-weighted fusion) |
| Belief | Diagonal Gaussian | Low-rank (rank 4) + diagonal Gaussian |
| Physical part | TQ heads (q, q̇) with a kinematic-consistency loss | The phase-space state itself |
| Parameters | 65,428 | 59,084 |

### Metrics (identical for both)

1. **Velocity linear-probe R²** on filtered states. Can the latent recover the hidden velocity?
2. **Open-loop rollout position RMSE** at 1, 5, 10 and 20 imagined steps, decoded with a linear probe fit on filtered states.
3. **Belief NLL per dimension** and a **calibration ratio z²** at those horizons. z² is the mean squared standardized error on the informative dimensions; 1.0 means calibrated, and above 1 means overconfident.
4. **Planning success** with CEM model-predictive control:
   - 30 tasks per seed, with start and goal on opposite sides of the river;
   - success means reaching within 0.15 of the goal within 60 steps;
   - two costs: the quasimetric, and plain L1 distance in latent space.
5. **Reference:** a greedy controller using the *true* state.

## 3. Every run we did, in order

| # | Run | What happened |
|---|---|---|
| 1 | **Smoke test**: 60 training steps, seed 0 | The pipeline ran end to end in about 1 minute. B was already ahead (planning 0.55 vs 0.05), but 60 steps is meaningless for conclusions. It only showed the code worked. |
| 2 | **First full launch**: seeds 0, 1 and 2 in parallel, 3,000 steps each | Seeds 1 and 2 finished. **Seed 0 crashed** while evaluating model A: `torch.linalg.solve: input matrix is singular`, inside the linear position/velocity probe. |
| 3 | **Fix and rerun seed 0** | The probe was changed to float64 with ridge 1e-2 (it had been float32 with ridge 1e-3), and seed 0 was rerun. Seeds 1 and 2 were *not* rerun, so they used the old probe. Ridge regression with that tiny regulariser barely changes a well-conditioned fit, so the difference should be negligible. It is still a real inconsistency, and it is recorded here. |
| 4 | **Code rebuild check** (2026-11-02) | The original `wmtest.py` was missing from the project files, so it was rebuilt from the session record. To prove the rebuild is the same code, seed 0 was rerun and its output compared with the original `result_seed0.json`. See §7. |

**Why did seed 0 crash?** This is our best explanation, not verified. Model A's GRU state is 64-d and passes through `tanh`. In that seed, several hidden units were saturated or almost perfectly collinear across the test set. That made XᵀX numerically singular in float32, and ridge 1e-3 was too small to rescue it. It is a measurement bug, not a model result, but it hints that model A's state is poorly conditioned.

## 4. What we got (mean ± std over 3 seeds)

| Metric | A (table) | B (combination) |
|---|---|---|
| Velocity probe R² | 0.55 ± 0.07 | **0.88 ± 0.02** |
| Filtered position RMSE | 0.099 | **0.057** |
| Open-loop position RMSE at 5 / 10 / 20 steps | 1.38 / 2.95 / 8.18 (seed 2 diverged to 21.6 at 20 steps) | **0.13 / 0.24 / 0.44** |
| Belief NLL per dim at 1 / 10 / 20 steps | −0.40 / 0.64 / 2.94 | **−0.45 / −0.15 / 0.19** |
| Calibration z² at 1 / 10 / 20 steps | 0.93 / 3.7 / 10.5 | **0.97 / 1.23 / 1.47** |
| Planning success, quasimetric cost | 0.34 ± 0.16 | **0.87 ± 0.19** |
| Planning success, latent L1 cost | **0.53 ± 0.14** | 0.18 ± 0.23 |
| Greedy controller on the true state | 1.00 | 1.00 |

Runtime was about 2.6 minutes per seed for A and 4.5 for B, on one CPU thread.

## 5. What we inferred

1. **On this world, B beat A on every model-quality metric, by large margins and consistently across seeds.** That covers velocity, rollouts, belief and calibration.
2. **The quasimetric planning head helped only when the dynamics underneath were good.** With B it was the best planner (0.87). With A, plain L1 beat it (0.53 vs 0.34). So "use a quasimetric" is not a free win: it depends on the latent and the dynamics it sits on.
3. **The test does not say which part of B won.** B differs from A in dynamics structure, belief form and state size all at once.
4. **The quasimetric's main selling point, asymmetric cost, was never really tested.** A greedy controller on the true state solves 100%, so the river is not hard enough to require planning around irreversibility.

## 6. Why we think it happened

These are hypotheses consistent with the numbers, not demonstrated.

- **B's prior matches the world exactly.**
  - The world *is* a damped, control-affine point mass, and B's dynamics are parameterised as precisely that. B carries momentum in p, so velocity is in the state by construction (R² 0.88). Its explicit damping keeps rollouts from blowing up.
  - A's free GRU prior has to *learn* integration and damping from a one-step consistency loss plus an 8-step embedding loss. That is weak pressure, so imagined states drift away from the states the probe was fit on. Hence errors of 8.2 and one diverged seed.
- **A's belief becomes overconfident with horizon (z² = 10.5 at 20 steps).**
  - The belief head is trained on one-step-ahead predictions, so its variances fit one-step error.
  - Over long horizons, A's dynamics errors compound much faster than the variances grow.
  - So A's poor calibration may come from its dynamics, not from the diagonal covariance as such. This is the confound in §5.3. The rotation-ambiguity argument against a diagonal belief (from the design review) was *not* isolated by this test.
- **The quasimetric hurt A.** The quasimetric is trained on *filtered (posterior)* states, but the planner queries it on *imagined (prior)* states. For A those two distributions differ a lot, so the quasimetric is queried off-distribution. For B, prior and posterior states are close, so it works.
- **Plain L1 hurt B.** B's state contains momentum and four over-complete position dimensions that only the 5% labels constrain. The goal state is built from a static goal observation, so its p ≈ 0. L1 therefore penalises velocity differences and unconstrained dimensions that don't matter for reaching the goal. The quasimetric learns which directions matter, and L1 cannot.

## 7. Caveats and integrity notes

- **The toy world favours B's structural prior.** On contact-rich or discontinuous dynamics, B's smooth, damped prior could be wrong, and its advantage could shrink or reverse.
- **No hyperparameter tuning was done for either model.** Only 3 seeds.
- **The probe differs across seeds.** Seeds 1 and 2 used the float32 / ridge 1e-3 probe, and seed 0 used float64 / ridge 1e-2 (see §3).
- **Parts of each design were not exercised:**
  - the sphere/cone state;
  - expected cost under the belief in planning;
  - T\*Q vs TQ, which are identical at unit mass;
  - coupling between the physics module and the recurrent state;
  - Fisher-geometry learning updates.
- **Code provenance.** `wmtest.py` in this folder was rebuilt from the session record on 2026-11-02, because the original file was no longer in the project files. One edit: results are now written to the current folder instead of the original absolute path. The verification run is described in the commit that adds the verification note.

## 8. What we would run next (proposed, never run)

The smallest test that would *separate* the ideas:
- a **2×2 ablation**: {free GRU vs structured dynamics} × {diagonal vs low-rank belief}, with both planning costs;
- on a **harder world**: a one-way gate where greedy control fails, plus a bounce or contact discontinuity that breaks B's smooth prior.

It was offered (about 30 minutes on CPU) but not run.

## 9. How to reproduce

```bash
pip install torch numpy
python3 wmtest.py 0 3000   # writes result_seed0.json (about 7 minutes on one CPU thread)
python3 wmtest.py 1 3000
python3 wmtest.py 2 3000
```
