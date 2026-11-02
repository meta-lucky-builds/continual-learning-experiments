# Toy head-to-head: Lucky's table (A) vs the earlier combination (B)

Code: `wmtest.py` (run: `python3 wmtest.py <seed> 3000`). Raw numbers: `result_seed{0,1,2}.json`. CPU, about 7 minutes per seed.

**World.** A 2D damped point mass. A one-way "river" (|x|<0.25) pushes +y harder than the maximum action. The observation is a 16-d nonlinear view of position only (velocity is hidden), plus noise, 8 pure-noise distractor channels, and random 6-step occlusions. Both models get the same data (3,000 random-action episodes), the same position labels on 5% of frames, the same SIGReg + JEPA losses and the same quasimetric (QRL) trainer.

- **A:** Euclidean GRU posterior and prior state (64-d), a diagonal Gaussian belief over the next observation, TQ heads (q, q̇) with a kinematic-consistency loss. 65k parameters.
- **B:** a phase-space state (q, p ∈ R⁴ each), a control-affine dynamics prior with explicit PSD damping and symplectic Euler steps, a learned-gain filter (precision-weighted fusion), and a low-rank + diagonal belief. 59k parameters.

## Results (mean ± std over 3 seeds)

| Metric | A (table) | B (combination) |
|---|---|---|
| Velocity linear-probe R² | 0.55 ± 0.07 | **0.88 ± 0.02** |
| Filtered position RMSE | 0.099 | **0.057** |
| Open-loop position RMSE, 5 / 10 / 20 steps | 1.38 / 2.95 / 8.18 (one seed diverged: 21.6 at 20 steps) | **0.13 / 0.24 / 0.44** |
| Belief NLL per dim, 1 / 10 / 20 steps | −0.40 / 0.64 / 2.94 | **−0.45 / −0.15 / 0.19** |
| Calibration z² (1.0 = calibrated), 1 / 10 / 20 steps | 0.93 / 3.7 / 10.5 (overconfident) | **0.97 / 1.23 / 1.47** |
| Planning success, quasimetric cost | 0.34 ± 0.16 | **0.87 ± 0.19** |
| Planning success, latent-L1 cost | 0.53 ± 0.14 | 0.18 ± 0.23 |
| Reference: greedy controller on the true state | 1.00 | 1.00 |

## Caveats (read before trusting this)

1. **The world matches B's prior.** It *is* a damped point mass, so B's structure is exactly right. On contact-rich or discontinuous worlds this advantage can shrink or reverse.
2. **The comparison is confounded.** B differs from A in dynamics structure, belief form and state size at once, so this test can't say which piece helped. For example, A's overconfident belief may come from its weak dynamics rather than from the diagonal covariance.
3. **The task is too easy for irreversibility to matter.** A greedy controller on the true state solves 100%, so the quasimetric's asymmetry was not really tested. What was tested is whether the learned latent plus cost supports planning.
4. **Hyperparameters are untuned for both models.** The probes for seeds 1 and 2 used ridge 1e-3 in float32; seed 0 used 1e-2 in float64 after a singular-matrix crash. The difference is minor.
5. **Parts of each design were not exercised.** The sphere/cone state, expected cost under belief, T\*Q vs TQ (identical at unit mass) and the coupling between the physics module and the state were not tested.

## The more accurate next test

A 2×2 ablation of {free GRU vs structured dynamics} × {diagonal vs low-rank belief}, with planning-cost conditions, on a harder world: a one-way gate where greedy fails, plus a contact or bounce discontinuity that breaks B's smooth prior.
