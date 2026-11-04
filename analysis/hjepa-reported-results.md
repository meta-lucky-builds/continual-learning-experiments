# Every H-JEPA success result reported in the paper

**Sources.** The H-JEPA paper (arXiv 2610.06805) and the repo's `docs/RESULTS.md`.

**How to read the numbers.** Values are success rates in %, given as mean ± SE over training seeds 42, 43 and 44. Each environment uses 50 fixed start/goal tasks, and the planner seed is paired with the model seed. Exceptions are noted where they apply.

## 1. Main result: model depth (Fig. 6 bottom)

LeWM uses flat planning. H-JEPA and HWM use hierarchical planning at their own depth.

| Environment | LeWM (flat) | H-JEPA 2 | H-JEPA 3 | H-JEPA 4 | HWM 2 | HWM 3 | HWM 4 |
|---|---|---|---|---|---|---|---|
| Visual AntMaze | 18.0 ± 3.5 | 39.3 ± 3.7 | **73.3 ± 3.5** | 67.3 ± 2.4 | 34.0 ± 1.2 | 38.7 ± 3.5 | 14.0 ± 2.0 |
| FourRoom Distractors | 40.7 ± 6.8 | 81.3 ± 6.7 | 96.0 ± 1.1 | 88.0 ± 2.3 | 75.3 ± 3.3 | **99.3 ± 0.7** | 92.7 ± 4.4 |
| OGBench Cube | 32.0 ± 6.1 | 47.3 ± 2.7 | 60.0 ± 5.0 | **62.7 ± 2.4** | 46.7 ± 2.9 | 53.3 ± 5.2 | 34.0 ± 3.1 |
| Push-T | 40.0 ± 2.3 | **45.3 ± 1.3** | 17.3 ± 3.5 | 0.7 ± 0.7 | 42.0 ± 6.4 | 12.0 ± 2.0 | 0.0 ± 0.0 |

**Our reproduction (2026-10-30, Colab T4):** FourRoom, H-JEPA 3, seed 42 scored **96.0%** (48 of 50). The 95% Wilson interval is 86.5 to 98.9.

- The planner uses the paper's selected planner-sample settings, chosen as the best success under 100 planner TFLOPs per episode.
- Fig. 6 (top) also sweeps success against planner TFLOPs per episode. Its exact points are only in the figure.

## 2. Cost ablation: flat level-1 planning with the goal cost in different latent spaces (Tables 1 and 9)

All rows plan at level 1. Only the space in which the goal cost is measured changes. The last column, hierarchical planning, is given for reference.

| Environment | Depth | Native L1 | L2 cost | L3 cost | L4 cost | Hierarchical |
|---|---|---|---|---|---|---|
| AntMaze | 2 | 23.3 ± 3.3 | 31.3 ± 2.7 | – | – | 39.3 ± 3.7 |
| AntMaze | 3 | 16.7 ± 1.8 | 20.7 ± 4.8 | 22.0 ± 2.3 | – | 73.3 ± 3.5 |
| AntMaze | 4 | 10.0 ± 0.0 | 20.7 ± 1.8 | 21.3 ± 4.7 | 26.7 ± 2.9 | 63.3 ± 7.7 (Table 1) / 67.3 ± 2.4 (Fig. 6, RESULTS.md) |
| FourRoom | 2 | 9.3 ± 0.7 | 38.7 ± 6.6 | – | – | 81.3 ± 6.7 |
| FourRoom | 3 | 43.3 ± 8.7 | 65.3 ± 5.5 | 71.3 ± 4.1 | – | 96.0 ± 1.1 |
| FourRoom | 4 | 68.7 ± 4.7 | 72.0 ± 3.1 | 66.0 ± 3.1 | 70.7 ± 2.7 | 88.0 ± 2.3 |
| Cube | 2 | 24.0 ± 1.2 | 13.3 ± 0.7 | – | – | 47.3 ± 2.7 |
| Cube | 3 | 24.7 ± 1.3 | 28.0 ± 4.2 | 14.0 ± 3.5 | – | 60.0 ± 5.0 |
| Cube | 4 | 26.7 ± 2.7 | 25.3 ± 2.4 | 12.0 ± 2.0 | 12.0 ± 2.3 | 62.7 ± 2.4 |
| Push-T | 2 | 36.7 ± 1.8 | 38.0 ± 3.1 | – | – | 45.3 ± 1.3 |
| Push-T | 3 | 30.7 ± 1.8 | 28.0 ± 4.0 | 24.7 ± 4.1 | – | 17.3 ± 3.5 |
| Push-T | 4 | 1.3 ± 0.7 | 1.3 ± 0.7 | 0.7 ± 0.7 | 1.3 ± 1.3 | 0.7 ± 0.7 |

The four-level AntMaze hierarchical number differs between sources. The paper's Table 1 prints 63.3 ± 7.7, while Fig. 6 and the repo's `RESULTS.md` give 67.3 ± 2.4. I copied both as printed and did not resolve which is correct.

## 3. Training data per level, two-level AntMaze (Fig. 13, App. G)

These are means over three seeds, with no SE given.

| Setup | Result |
|---|---|
| Flat level-1 planner trained on explore / stitch / union | 18 / 11 / 19 |
| Two-level end-to-end, both levels trained on explore / stitch / union | 11 / 21 / 54 |
| Two-level stagewise, explore at level 1 and stitch at level 2 | **67**, the best configuration in the figure |
| Two-level stagewise, stitch at level 1 and explore at level 2 | 2 |
| Two-level stagewise, union at both levels | 40 |
| Two-level stagewise, explore at both levels | 11 |

## 4. DROID: real-robot video, scored offline (Fig. 9b)

DROID has no simulator, so there is no success rate. Plans are scored instead by Fréchet fidelity against expert end-effector paths on 16 clips:

| Model | Fréchet fidelity |
|---|---|
| LeWM + IDM | 33.57 ± 1.52 % |
| HWM | 35.65 ± 2.22 % |
| H-JEPA (2 levels) | 38.42 ± 0.94 % |

The paper calibrates this metric on Cube and RoboCasa. On Cube, clearing a fidelity floor is necessary for a grasp but not sufficient, so a fidelity gap understates the gap in real outcomes. Fig. 9c also has a fidelity-versus-compute sweep, given only in the figure.

## What the numbers say (my reading)

- **The hierarchy helps most on long-horizon navigation.** AntMaze goes from 18 to 73 and FourRoom from 41 to 96, both with three levels. Cube gains less, from 32 to 63.
- **Push-T gets worse with depth** (45, then 17, then 0.7). The paper attributes this to short episodes: at 3 and 4 levels, deeper models see only 58% and 14% of LeWM's training clips.
- **Separate latent spaces per level matter mainly on AntMaze.** There, three-level H-JEPA reaches 73.3 against 38.7 for HWM, which shares one latent space across levels. On FourRoom, HWM 3 is slightly higher (99.3 against 96.0).
- **Most of the gain comes from hierarchical planning itself.** Swapping only the cost space under a flat planner gives at most about 31% on AntMaze, against 73% for full hierarchical planning.
- **Headroom for our JEPA-TTT and looping work:**
  - AntMaze (73%) and Cube (about 60%) still have room for a success gain.
  - FourRoom (96 to 99%) does not; there we would need the shift test or path-efficiency metrics.
