# Can H-JEPA (and the first JEPA-TTT steps) run on a Colab T4?

Short answer: **yes for the starting parts, if we start with FourRoom.**

- The models are tiny, so they fit on a T4 easily. The real limits are training time, Colab session length and dataset storage, not memory or parameter count.
- AntMaze at full paper scale is about two days of T4 time and over 100 GB of raw frames. Run it scaled down, or rent a bigger GPU for that one step.

Sources:
- Architecture and data settings: the H-JEPA paper, App. C.1 and Table 4 (architecture), and App. B and Table 3 (datasets, clips per epoch).
- Parameter counts and timings: **my estimates** from those settings. They are not reported in the paper. The paper trained on H200s.

## 1. Model size

Paper settings (App. C.1, Table 4):

| Part | Architecture | Params (my estimate) |
|---|---|---|
| Level-1 encoder | ViT-Tiny, patch 14 at 224 px (patch 4 at 64 px on AntMaze), plus an MLP projector | about 5.7M |
| Level-1 predictor | Causal transformer: 4 layers, 8 heads, width 192, FF 1024, adaLN action conditioning | about 3M |
| Each upper-level encoder | 2-layer MLP, hidden width 384 | about 0.2M |
| Each upper-level action encoder | 1-layer transformer pooling to a 4-dimensional macro-action | under 0.1M |
| Upper predictor, FourRoom | 4 layers, width 192, FF 1024 | about 3M |
| Upper predictor, AntMaze | 5 layers, 12 heads, width 224, FF 1536 | about 5M |
| Upper predictor, Push-T and Cube | 6 layers, 16 heads, width 192, FF 2048 | about 5.5M |

- **Two-level H-JEPA comes to about 10 to 15M parameters, and three-level to about 13 to 20M.** That is roughly 0.2 GB including optimizer state, so a T4's 16 GB has plenty of room for the weights.
- Activations are the memory cost. One batch is 128 clips × 7 frames, about 900 images through the ViT, which probably exceeds 16 GB at full batch (estimate). Use activation checkpointing on the ViT.
  - Do not split the batch with gradient accumulation. SIGReg is a statistic over the whole batch, and the paper's §K.3 shows that sharding changes it.

## 2. Training time on one T4 (estimates)

Assumptions:
- The ViT encoder dominates compute: about 3 GFLOPs forward per frame at 257 tokens, so about 9 GFLOPs per frame for training. The predictors are small next to it.
- A T4 sustains about 15 TFLOP/s in fp16 under real training. That may be optimistic.
- Clip counts and epochs come from the paper's Tables 3 and 4.

| Environment | Clips/epoch (2-level) | Epochs | Full-scale 2-level time | Full-scale 3-level time |
|---|---|---|---|---|
| FourRoom | 0.92M | 7 | **about 8 h** | about 13 h |
| Cube | 1.64M | 8 | about 15 h | about 20 h |
| Push-T | 1.70M | 9 | about 18 h | about 15 h (fewer valid clips) |
| AntMaze | 9.2M | 5 | **about 50+ h** | about 75 h |

- Planning evaluation is cheap by comparison. The paper's operating budget is at most 100 planner TFLOPs per episode, which is seconds to tens of seconds per episode on a T4 (estimate).
- JEPA-TTT at test time is also cheap: it updates only the small predictors and the level-1 action encoder, with the encoder frozen.

## 3. T4 and Colab gotchas

- **No bf16 on a T4.** The paper trains in bfloat16. Use fp16 autocast with a GradScaler.
  - SIGReg and the prediction loss may be less stable in fp16, so watch for NaNs.
  - fp32 is the fallback, and it is about 2 to 4 times slower.
- **Sessions are limited.** Free Colab runs disconnect when idle and have session caps, and a T4 is not guaranteed. Checkpoint to Google Drive every few hundred steps and make training resumable.
- **Kaggle is a good alternative.** It offers free T4s, with a weekly GPU quota, which is useful for the 8 to 15 h runs.
- **Dataset storage.** Raw uint8 frames are large:
  - AntMaze: 10M frames × 64×64×3 ≈ 120 GB.
  - FourRoom: 1.5M frames × 224×224×3 ≈ 225 GB.
  - Colab disk and RAM cannot hold that raw. Store frames compressed (video or JPEG shards) and decode in the loader, or generate data at lower resolution. FourRoom and AntMaze are the authors' own datasets, so we would regenerate them with their collection protocol.
- **Rendering.** AntMaze and Cube are MuJoCo. Headless rendering on Colab needs EGL or OSMesa; that is usually workable but needs setup.

## 4. What I would run first

1. **FourRoom, two-level H-JEPA, scaled down.** Use half the epochs, or 128×128 frames, to bring it to about 3 to 4 h. That reproduces the baseline within one or two sessions.
2. **The FourRoom shift test.**
   - FourRoom already randomizes which doors are closed. Add test-time shifts: agent speed, action gain, wall positions, more distractors.
   - Check whether H-JEPA's success drops. This is the go/no-go step from the plan.
3. **JEPA-TTT on FourRoom.** Adapt only the level-1 predictor first. This is cheap on a T4.
4. **Three-level FourRoom** (about 13 h at full scale, so it spans several sessions or a reduced run).
5. **AntMaze only after that.** Train on a reduced subset (for example 2M transitions), or rent a larger GPU by the hour for this single run.

## 5. Update (2026-10-30): code, checkpoints and data are all public

Checked against github.com/kevinghst/H-JEPA (MIT), huggingface.co/jepa-world-models/h-jepa (MIT) and the matching Hugging Face dataset (CC BY 4.0).

**Checkpoints: all 84 simulation models.**
- Covers FourRoom, AntMaze, Push-T and Cube.
- Each environment has LeWM, H-JEPA with 2, 3 or 4 levels, and HWM with 2, 3 or 4 levels, each at seeds 42, 43 and 44.
- Sizes: FourRoom 1.7 GB, AntMaze 2.6 GB, Push-T and Cube about 2.7 GB each, so about 9.7 GB for all 84.
- Separately, 9 DROID checkpoints.
- Each file is a pickled `*_object.ckpt`, loaded with `torch.load(..., weights_only=False)` from inside `h_jepa/`.
- **So no training is needed to start.** The three-level models we need for the JEPA-TTT and looping work already exist.

**Datasets:**

| Environment | Size | Where |
|---|---|---|
| FourRoom | about 5 GB of HDF5 | H-JEPA dataset |
| AntMaze | about 27 GB of HDF5 (22 GB is the training file) | H-JEPA dataset |
| Push-T | separate download | LeWM releases (`quentinll/lewm-pusht`) |
| Cube | separate download | LeWM releases (`quentinll/lewm-cube`) |
| DROID | 91 GB | H-JEPA dataset |

- Evaluation tasks are 50 fixed start/goal tasks per environment, about 1.45 GB in total. FourRoom's is 574 MB and AntMaze's is 155 MB.
- The images are stored compressed, so the raw-size worries in section 3 don't apply. AntMaze still needs about 30 GB of Drive or disk if we ever train it.

**Code details that matter on Colab:**
- **Python 3.11 is required** (`>=3.11,<3.12`), along with torch 2.7.1 and mujoco 3.14.0. If the Colab runtime runs a different Python, create a 3.11 environment with `uv`.
- **The configs use bf16.** On a T4, override with `trainer.precision=16-mixed`.
- **The configs set `num_workers: 12`.** Colab has about 2 CPUs, so override with `num_workers=2`. Expect data loading to be the bottleneck in training.
- **Resume checkpoints are built in** (`resume_every_n_steps: 2000`). Point `HJEPA_HOME` at Google Drive so a disconnect only loses progress since the last save.
- **Standalone planning evaluation exists:** `python eval.py --config-name fourroom_l3 policy=<ckpt>`.

**Revised first steps:**
1. Download the FourRoom and AntMaze checkpoints (about 4.3 GB) and their evaluation tasks.
2. Run the standalone planning evaluation on the T4. This reproduces the paper's success rates without any training.
3. Build the shift test on FourRoom, by changing environment parameters in `stable_worldmodel/envs/four_room/env.py`, and evaluate the pretrained models under it.
4. Add JEPA-TTT on top of the frozen checkpoints.
5. Train from scratch only later, if we need a variant the checkpoints don't cover.
