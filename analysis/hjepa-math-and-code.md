# H-JEPA: mathematics ↔ code reference

Paper: *H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning* (arXiv 2610.06805v1), full text read from `h.txt` (all 3900 lines).
Code: `github.com/kevinghst/H-JEPA` @ `97054ec`, at `/home/user/kevinghst/h-jepa`. Paths below are relative to that root; `h_jepa/` files are imported with `h_jepa/` on `sys.path`, so `import loss`, `from models.jepa import JEPA`, and so on.

Conventions in this file:
- `file:line` refers to the commit above.
- **(inferred)** marks a statement I reasoned out rather than read directly in the paper or code.
- **(verified)** marks a statement I checked by running code in `/home/user/venv311` on CPU.
- The PDF-to-text extraction scrambles sub- and superscripts. Where an equation's indices were ambiguous in the extraction, I say so and give the code's exact semantics.

---

## 1. Notation

Merged from the paper's Table 2 and the code identifiers.

| Symbol (paper) | Meaning | Code name / shape |
|---|---|---|
| ℓ, L | level index (1 = finest) and number of levels | `level`; `HJEPA.num_levels`; `cfg.num_levels` |
| t | time index at the level's own rate | time axis 1 of `(B, T, ...)` tensors |
| a:b | half-open range a,…,b−1 | Python slices |
| o_t | image observation | `pixels` / `pixels_level1`, shape `(B,T,3,H,W)`, ImageNet-normalised |
| a_t | primitive **action block** for o_t→o_{t+1} (the frameskip=5 env actions concatenated) | `action_level1`, shape `(B,T, frameskip·act_dim)`: 10 for FourRoom, 40 for Ant; z-scored per raw dimension *before* concatenation (`stable_worldmodel/data/dataset.py:176-202`) |
| z_t^{(ℓ)} | level-ℓ latent state | `embed_{ℓ}` in `encode_hierarchical` output; `embed_0` inside one `JEPA.encode` call; `(B,T,D_ℓ)` |
| a_t^{(ℓ)} | level-ℓ action embedding (a macro-action for ℓ>1) | **two tensors in code**: the *latent action* = `action_encoder` output (4-d for single-scene upper levels; this is the planner's decision variable), and `action_{ℓ}` = `action_embed(latent action)` (D_ℓ-dim; this is what the predictor and the next level up consume) |
| E^{(ℓ)} | state encoder | `JEPA.encoder` (`ProjectedEncoder` / `FusionEncoder`) |
| A^{(ℓ)} | action encoder (pools s_ℓ lower action embeddings) | `JEPA.action_encoder` (Identity at ℓ=1; `SequenceEncoder` or `SequenceMLPEncoder` above), **followed by** `JEPA.action_embed` (`Embedder` or Identity) |
| F^{(ℓ)} | latent predictor | `JEPA.predictor` (`ProjectedPredictor(ARPredictor, MLP-BN projector)` or `CausalTransformerPredictor`) |
| I^{(ℓ)} | inverse-dynamics head | `JEPA.idm_loss_fn.idm` (`InverseDynamicsModel`); only exists when `idm_coeff>0` (DROID) |
| Roll^{(ℓ)} | autoregressive rollout | `JEPA.rollout` (`h_jepa/models/jepa.py:376-418`) |
| s_ℓ, w_ℓ | stride and window from level ℓ−1 to ℓ | `levelN.stride` → `JEPA.temporal_stride`; `levelN.window_size` → `JEPA.temporal_window_size` (all paper models: s=2, w=1; DROID s=3, w=1) |
| c_ℓ | predictor context length | `levelN.wm.history_size` (3 single-scene; DROID 7 / 2); planning uses context 1 |
| λ_ℓ, γ_ℓ | SIGReg and IDM weights | `levelN.loss.embed.sigreg.weight`; `levelN.idm_coeff` |
| Z^{(ℓ)} ∈ R^{n×D} | minibatch of embeddings | SIGReg input `(T, B, D)` = `emb.flatten(2).transpose(0,1)` |
| M, u_m, h_m | number of SIGReg directions, a direction, its projection | `SIGReg.num_proj=1024`; columns of `A`; `proj @ A` |
| T_EP | Epps–Pulley statistic | `SIGReg.forward` (`h_jepa/loss.py:28-47`) |
| W | data-parallel world size | `torch.distributed.get_world_size()` |
| H_ℓ | planning horizon at level ℓ (level-ℓ steps) | `hierarchical_plan_config.levelN.horizon`, per call `planning_horizon` |
| g^{(ℓ)} | goal latent at level ℓ | `goal_embeddings[f'embed_{ℓ}']` → `goal_embed_0` |
| z̃_i^{(ℓ+1)} | level-ℓ rollout lifted to level ℓ+1 | `LevelCostModel._encode_states_up` output |
| K_{ℓ+1} | number of subgoals matched by level ℓ | `levelN+1.num_subgoals` (and `levelN.cost_last_n` ≥ K) |
| β | weight of the intermediate subgoal terms | `levelN.intermediate_cost_weight` |
| C^{(ℓ)}(z,z′) | pairwise cost ‖z−z′‖² | `F.mse_loss(..., reduction='none')` summed over D (`jepa.py:420-436`) |
| S | candidates per level | `solver.solvers.levelN.num_samples` |
| q | scored final steps | `cost_last_n` |
| R | level-1 blocks executed per replan | `hierarchical_plan_config.receding_horizon` (=2) |
| t_j, ξ_j | replanning times, commit widths | `WorldModelPolicy._action_buffer`, `receding_horizon × action_block` |
| χ(E) | within-episode variance share | not in code |
| λ_sim | temporal-repulsion weight | not in code |

Other code-only symbols: `target_length = history_size + rollout_n` (= 4 per level), `latent_action_queue` (size 4000), `var_scale`, `action_noise`, `action_clip_sigma`, `min_its=30`, `early_stop_patience=5`, `early_stop_rel_delta=1e-2`.

---

## 2. Every equation in the paper

### Eq. (1): hierarchical state and action encoding

$$
z^{(\ell)}_t = E^{(\ell)}\!\left(z^{(\ell-1)}_{t s_\ell \,:\, t s_\ell + w_\ell}\right),\qquad
a^{(\ell)}_t = A^{(\ell)}\!\left(a^{(\ell-1)}_{t s_\ell + w_\ell - 1 \,:\, (t+1)s_\ell + w_\ell - 1}\right),\qquad
z^{(1)}_t=E^{(1)}(o_t),\; a^{(1)}_t = A^{(1)}(a_t).
$$

**Meaning.** Each upper state pools a window of w_ℓ lower states that starts every s_ℓ steps. Each upper action pools the s_ℓ lower action embeddings that begin at the last state of that window, so it covers the transition to the next upper state.

**Code.**
- Window and stride indexing: `JEPA._chunk_temporal_info` (`h_jepa/models/jepa.py:190-244`). It computes `starts = arange(0, T − w + 1, s)`, then calls `_state_windows` (`jepa.py:144-158`; for w=1 this is `index_select(1, starts)`, i.e. pure subsampling) and `_action_windows` (`jepa.py:160-188`). `_action_windows` sets `action_starts = starts + w − 1` and takes `stride` consecutive entries, zero-padding past the end and returning `action_mask=True` on the pads.
- Encoding: `JEPA.encode` (`jepa.py:307-353`) runs `self.encoder.encode_info(...)`, then `latent = self.action_encoder(info['action'], action_mask)`, then `info['action_0'] = self.action_embed(latent)`.
- The whole stack: `HJEPA.encode_hierarchical` (`h_jepa/models/hjepa.py:239-294`). Level 1 goes through `_encode_level1` (`hjepa.py:174-198`). Each level ℓ≥2 calls `jepa.encode(level_info, key='embed_{ℓ-1}' (or 'pixel_embed_{ℓ-1}' for proprio-fusion levels), chunk_temporal_inputs=True)`, with `level_info['action'] = action_{ℓ-1}`.

**Paper vs code.**
1. In code, A^{(ℓ)} is a composition of two learned maps: `action_encoder` (pooling to a low-dimensional latent: 4-d single-scene, 8-d DROID) and then `action_embed` (an `Embedder` MLP, 4→4D→D; Identity on DROID). The planner's decision variable at upper levels is the **pre-`action_embed` latent** (4-d). The level-(ℓ+1) action encoder consumes the **post-`action_embed`** level-ℓ embeddings (D_ℓ-dim). On DROID, level-1 `action_embed` is Identity, so the level-2 encoder concatenates raw 7-d commands (3×7=21 inputs), as the paper's §C.2 says.
2. At level 1, `A^{(1)}` = `Identity` action encoder followed by `Embedder(input_dim=frameskip·act_dim, smoothed_dim=10, emb_dim=D, mlp_scale=4)`: a Conv1d(k=1) "patch_embed", then Linear→SiLU→Linear (`h_jepa/models/module.py:189-231`). On DROID the predictor is `CausalTransformerPredictor`, which has its own internal `action_embedder` (`h_jepa/models/predictors/predictors.py:146-148`).
3. Zero padding: the last upper action chunk can run past the lower stream. Those positions are zeroed and masked (as `src_key_padding_mask` in `SequenceEncoder`). The padded chunks are only ever used for the final upper step, which the crop and the loss never use as a context action **(inferred from the crop lengths)**.
4. Gradient flow: nothing is detached. Upper-level losses backpropagate through `embed_{ℓ-1}` into every lower encoder, and through `action_{ℓ-1}` into every lower `action_embed`/`action_encoder`. **(verified)** In a 3-level FourRoom model, `loss_level3.backward()` gives non-zero gradients in the L1 encoder, the L1 `action_embed`, the L2 encoder and the L2 action encoder, and zero gradient in the L1 predictor. At initialisation the action-path gradients are exactly zero because the AdaLN modulation is zero-initialised (`module.py:105-106`); they become non-zero once the AdaLN weights move.
5. Sparse level-1 encoding (`sparse_level1_encode=True`, the default; `hjepa.py:146-198`) encodes only the level-1 frames some kept step reads, and scatters them into a zero tensor. The architecture doc claims this is numerically identical to dense encoding. That holds only if per-frame encoding is independent. In train mode the level-1 projector's `BatchNorm1d` (and the predictor projector's) computes batch statistics over **only the encoded frames**, so sparse and dense training are not bit-identical **(inferred)**. The cited unit test `unit_tests/test_sparse_level1_encode.py` is not in this clone.
6. Random crops: `HJEPA._random_crop_level_outputs` (`hjepa.py:86-135`) crops each level independently to `target_length = history_size + rollout_n = 4`, with crop starts drawn in `_draw_crops_and_level1_frames` (`hjepa.py:146-172`). The paper says the same in §C.1 ("each lower level takes a random four-state crop of its own stream").

**Clip layout (paper §C.1, unnumbered):** n_L = 4, n_{ℓ−1} = (n_ℓ − 1)s_ℓ + w_ℓ. Code: `HDF5Dataset._setup_levels._required_level1_frames` (`stable_worldmodel/data/dataset.py:128-144`) computes the same recursion, giving 4/7/13/25 level-1 frames for 1–4 levels. `span = required_level1_frames × frameskip` env steps are loaded (e.g. 13×5 = 65 for L3, matching the paper's "loader reserves 5 n₁ environment steps").

---

### Eq. (2): teacher-forced prediction loss

As extracted:
$$
\mathcal{L}^{(\ell)}_{\text{pred}}=\frac{1}{c_\ell}\sum_{\tau=1}^{c_\ell}\left\| F^{(\ell)}\!\left(z^{(\ell)}_{<\tau},a^{(\ell)}_{<\tau}\right)-z^{(\ell)}_{\tau+1}\right\|_2^2 .
$$

Index caveat: the extraction shows `z<τ`/`a<τ` against target `z_{τ+1}`. The code semantics, with 0-based t, are
$$
\mathcal{L}^{(\ell)}_{\text{pred}}=\operatorname{mean}_{b,\;\tau\in\{0,\dots,c_\ell-1\},\;d}\Big(\big[F^{(\ell)}(z_{0:c_\ell},a_{0:c_\ell})\big]_{\tau,d}-z_{\tau+1,d}\Big)^2,
$$
where the causal mask makes output position τ depend only on z_{0..τ} and a_{0..τ}. In particular the action a_τ of the τ→τ+1 transition is included.

**Meaning.** One-step latent prediction under teacher forcing. Every position of the causal context predicts its successor, which gives c_ℓ = 3 targets per clip per level.

**Code.**
- Prediction: `hjepa_forward` (`h_jepa/hjepa_utils.py:156-159`, `rollout_n == 1` branch): `teacher_pred = jepa.predict(emb[:, :history_size], act_emb[:, :history_size])`.
- Target: `emb[:, 1:history_size+1]`.
- Loss: `F.mse_loss(...)` at `hjepa_utils.py:194-197`; the loss key is `teacher_forcing_loss_level{ℓ}` and also `pred_loss_level{ℓ}`.
- `JEPA.predict` (`jepa.py:355-360`) → `ProjectedPredictor.forward` (`jepa.py:113-116`) → `ARPredictor.forward` (`predictors.py:40-49`): adds the learned `pos_embedding[:, :T]`, then runs AdaLN-zero `ConditionalBlock`s with causal SDPA (`module.py:78-114`), then the final LayerNorm, then the BN-MLP projector.

**Paper vs code.**
1. **Normalisation constant:** `F.mse_loss` averages over B, c_ℓ **and D**. The code value is therefore (1/D)× the paper's squared-norm form, with D = 192 for FourRoom and 384 for Ant/DROID. All reported λ values are relative to the code's per-element mean **(inferred: the paper's ‖·‖² is not literally what is optimised)**.
2. No stop-gradient on the target by default: the target `emb` carries gradient into the encoder. The `level{ℓ}.wm.detach_pred_target` flag (`hjepa_utils.py:184-185`) enables the §K.1 detachment ablation; it is off in every config.
3. Code-only options not in the paper:
   - `rollout_n>1` (autoregressive multi-step loss; `hjepa_utils.py:160-169, 203-212`);
   - `nsteps` (`JEPA.parallel_unroll`, `jepa.py:362-374`, a detached multi-pass unroll).
   Neither is used by any shipped config.
4. `ARPredictor.pos_embedding` has shape `(1, history_size=3, D)` and is initialised with `torch.randn` (std 1, **not** small). It is indexed `[:, :T]`, so **the single-scene predictors cannot take a window longer than 3** (`predictors.py:26, 46`). DROID's `CausalTransformerPredictor` has `max_seq_len=16`, initialised at std 0.02.
5. Diagnostics computed under `no_grad` (`mse_loss`, `l1_loss`, `dim_*`; `hjepa_utils.py:171-177`) are logging only.

---

### Eq. (3): per-level objective

$$
\mathcal{L}^{(\ell)}=\mathcal{L}^{(\ell)}_{\text{pred}}+\lambda_\ell\,\mathrm{SIGReg}\!\left(Z^{(\ell)}\right),\qquad
\mathcal{L}=\sum_{\ell=1}^{L}\mathcal{L}^{(\ell)}\ \ (\S C.1\text{, unnumbered}).
$$

**Code.** `hjepa_forward` (`hjepa_utils.py:179-251`):
- `component_loss = 1.0·pred_loss + weight·sigreg(component_emb.flatten(2).transpose(0,1))`, with SIGReg at `hjepa_utils.py:216-225`;
- `level_loss` then adds `action_sigreg_coeff·…` (DROID; `hjepa_utils.py:232-238`) and `idm_coeff·…` (DROID; `hjepa_utils.py:240-244`);
- `total_loss = Σ_ℓ level_loss` (`hjepa_utils.py:251`).

SIGReg modules are built in `create_world_model` (`hjepa_utils.py:421-433`) and registered on the Lightning module as `sigreg_level{ℓ}`.

**Paper vs code.**
1. Z^{(ℓ)} is the **cropped** embedding stream (T=4 at every level), so the targets get SIGReg too. Its shape `(T, B, D)` is the paper's "(T, N, D) grouping".
2. Optimisation:
   - one AdamW over param groups `level1..levelL`, each with its own LR (`h_jepa/main_hjepa.py:83-108`; L1 5e-5, upper 1e-5);
   - weight decay 1e-3;
   - the schedule is `LinearWarmupCosineAnnealingLR`, which the stable_pretraining defaults set to warm-up = 1% of `estimated_stepping_batches` and cosine to 0. It is stepped every optimizer step, since spt uses manual optimization (`stable_pretraining/module.py:348-352`), even though the config says `interval: "epoch"`.
   - Gradients that upper losses send into lower-level parameters are applied with the **lower level's** LR, because parameters are grouped by owning level **(inferred from the construction)**.
3. Gradient clipping: one global norm of 1.0 over all levels (Lightning `gradient_clip_val`). DROID uses `grad_clip_per_level: true`, which clips each param group separately (`main_hjepa.py:42-50`), as in Table 5.
4. Precision is `bf16-mixed` for FourRoom and `bf16` for Ant (`config/train/base/{fourroom,ant}.yaml`).

---

### Eq. (4): top-level planning objective

$$
a^{(L),*}_{0:H_L}=\arg\min_{a^{(L)}_{0:H_L}}\ \big\|\hat z^{(L)}_{H_L}-g^{(L)}\big\|_2^2,\qquad
(\hat z^{(\ell)}_1,\dots,\hat z^{(\ell)}_{H_\ell})=\mathrm{Roll}^{(\ell)}\big(z^{(\ell)}_0,a^{(\ell)}_{0:H_\ell}\big).
$$

**Meaning.** The top level optimises macro-actions so that its final predicted latent matches the encoded goal.

**Code.**
- `HierarchicalSolver.solve` (`h_jepa/hierarchical_solver.py:142-154`): for the top level, `goal_embed_0 = goal_embeddings[f'embed_{L}']`. Then `GradientSolver.solve` (`stable_worldmodel/solver/gd.py:273-436`) calls `LevelCostModel.get_cost` (`hierarchical_solver.py:405-427`).
- With no upper models, `_project_to_goal_space` returns the rollout unchanged, and `JEPA.criterion` (`jepa.py:420-436`) is evaluated with `cost_last_n=1`.
- Roll^{(ℓ)} = `JEPA.rollout` (`jepa.py:376-418`) with `history_size=1`.

**Paper vs code.**
1. The cost is `F.mse_loss(reduction='none').sum(over time, D)`, so it is exactly a squared L2 distance (no 1/D here, unlike Eq. 2).
2. The goal is `.detach()`ed (`jepa.py:427`).
3. Upper-level action candidates live in the 4-d latent space, not in a primitive action space.

---

### Eq. (5): lifting lower-level predictions into the upper space

$$
\tilde z^{(\ell+1)}_i = E^{(\ell+1)}\!\left(\hat z^{(\ell)}_{i s_{\ell+1} : i s_{\ell+1}+w_{\ell+1}}\right),\quad i=1,\dots,K_{\ell+1},\qquad
H_\ell = K_{\ell+1}s_{\ell+1}+w_{\ell+1}-1 .
$$

**Code.**
- `LevelCostModel._project_to_goal_space` (`hierarchical_solver.py:429-457`) front-pads by `w−1` copies of z₀ (0 for w=1).
- `_encode_states_up` (`hierarchical_solver.py:459-485`) calls `upper_models[0].encode({f'embed_{ℓ}': states}, key=..., chunk_temporal_inputs=True)`. That uses the upper JEPA's own `temporal_stride` and `window_size`, so the starts are `arange(0, H+1−w+1, s)`, and index i=0 is the context state z₀.
- `JEPA.criterion` then drops index 0 (`jepa.py:422`).
- The consistency check `goal_emb.shape[-2] == (H + s)//s − 1 = ⌊H/s⌋` is at `hierarchical_solver.py:192-203`; this is the paper's K = ⌊H_ℓ/s_{ℓ+1}⌋ from Table 2.

**Paper vs code.**
1. Gradients flow through E^{(ℓ+1)} (no `no_grad`) back into ẑ^{(ℓ)} and then into the candidate actions. The upper encoder parameters receive `.grad` too (see §4.6).
2. "Dense" variant (code only, used for skipped levels / the `_project` cost-ablation configs): `temporal_stride=1` is forced, so *every* predicted step is lifted, and lifting can chain through several encoders up to `goal_level` (`hierarchical_solver.py:477-484`). This implements Table 1's "Lk projection" E^{(2)}∘…∘E^{(k)}.

---

### Eq. (6): lower-level subgoal-tracking objective

$$
a^{(\ell),*}_{0:H_\ell}=\arg\min_{a^{(\ell)}_{0:H_\ell}}\ \Big\|\tilde z^{(\ell+1)}_{K_{\ell+1}}-\hat z^{(\ell+1),*}_{K_{\ell+1}}\Big\|_2^2+\beta\sum_{i=1}^{K_{\ell+1}-1}\Big\|\tilde z^{(\ell+1)}_{i}-\hat z^{(\ell+1),*}_{i}\Big\|_2^2 .
$$

Closed loop: K=1. Open loop: K=H_{ℓ+1}, with β=0 matching only the terminal subgoal and β>0 matching intermediate ones too.

**Code.**
- Subgoal extraction (`hierarchical_solver.py:155-184`): `next_goal = subgoal_latents[:, 1:1+num_subgoals]` (index 0 is the upper context state), and `next_action = subgoal_actions[:, 0:num_subgoals]`.
- If the subgoals reach the end of the upper plan **and** the whole chain above is anchored, the last subgoal is replaced by the true goal embedding `goal_embeddings[f'embed_{ℓ+1}']`. This is Table 2's g_K = g^{(ℓ+1)}. In the closed-loop single-scene configs this never fires, because an upper level with H=1 is skipped. It does fire on DROID (K=12, H₂=12) **(inferred from the configs)**.
- Weighting: `JEPA.criterion` (`jepa.py:423-436`) keeps the last `cost_window = min(cost_last_n, #lifted)` steps, multiplies all but the last by `intermediate_cost_weight` (= β), then sums.

**Paper vs code.**
1. `cost_last_n` must be ≥ `num_subgoals`. Otherwise `goal_emb.expand_as(pred_emb)` fails (or broadcasts silently when K=1). DROID uses `cost_last_n: 999, intermediate_cost_weight: 0.5, num_subgoals: 12` (`config/eval/droid_l2.yaml`); the single-scene configs use K=1.
2. With K=1 (FourRoom/Ant): H₁ = 1·2+1−1 = 2, the lifted stream is `[E₂(z₀), E₂(ẑ₂)]`, and the cost is `‖E^{(2)}(ẑ^{(1)}_2) − ẑ^{(2),*}_1‖²`, exactly Fig. 3's C^{(1)}. The level-1 setting `cost_last_n: 2` only matters when level 1 is scored directly against a goal (dense projection), where it scores ẑ₁ and ẑ₂. That matches §E: "scores the last two predicted states at level 1".
3. **Cross-level action cost (paper §E, unnumbered; Push-T/Cube only, weight 0.2):**
   $$c_{\text{act}}=\sum_{i\in\text{last }q}\big\|A^{(\ell+1)}\big(\mathrm{embed}^{(\ell)}(a^{(\ell)}_{is:(i+1)s})\big)-a^{(\ell+1),*}_{i}\big\|_2^2 .$$
   Code: `LevelCostModel._encode_actions_up` (`hierarchical_solver.py:487-513`) and `_action_cost` (`hierarchical_solver.py:515-534`). It is added at `hierarchical_solver.py:423-426`, the target is detached, and it is disabled (weight 0.0) for FourRoom, Ant and DROID.

---

### Eq. (7): Fréchet fidelity (DROID, offline)

$$
\text{fidelity}=\frac{d_F(\mathbf 0,\tilde p)-d_F(p,\tilde p)}{d_F(\mathbf 0,\tilde p)} .
$$

Here p and p̃ are origin-prepended cumulative xyz paths of the planned and expert delta-pose sequences.

**Code.**
- `traj_metrics.frechet_skill_over_floor` (`h_jepa/traj_metrics.py:44-52`), built on `xyz_path` (`traj_metrics.py:12-16`, which prepends the origin) and `discrete_frechet` (`traj_metrics.py:19-35`, the O(nm) DP).
- It returns NaN when the floor is ≤ 1e-9.
- Aggregation is in `droid_eval.aggregate` (`h_jepa/droid_eval.py:75-88`).
- Actions are de-normalised before scoring (`droid_eval.py:172`).

**Paper vs code.** The formula matches. The paper reports the median over (train seed × planner seed) cells. The code also writes a weighted median and the ATE metrics (`droid_eval.py:52-72`).

---

### Eq. (8): inverse-dynamics loss

$$
\mathcal{L}^{(\ell)}_{\text{idm}}=\frac{1}{T-1}\sum_{t=1}^{T-1}\frac{1}{d_a}\Big\|I^{(\ell)}\big(z^{(\ell)}_t,z^{(\ell)}_{t+1}\big)-\mathrm{sg}\big[a^{(\ell)}_t\big]\Big\|_2^2 .
$$

**Code.**
- `InverseDynamicsLoss.forward` (`h_jepa/loss.py:73-79`): `pred = idm(emb[:, :-1], emb[:, 1:])` and `F.mse_loss(pred, action[:, :T-1].detach())`.
- `InverseDynamicsModel` (`loss.py:50-63`) is an MLP: Linear(2D→256)–ReLU–Linear(256→256)–ReLU–Linear(256→d_a), trunc-normal(0.02) init.
- It is wired in at `hjepa_utils.py:240-244` with `jepa.idm_loss_fn(emb, act_emb)` and created at `hjepa_utils.py:405-412` when `idm_coeff>0`.

**Paper vs code.**
1. `F.mse_loss` averages over B, T−1 and d_a, which matches 1/(T−1)·1/d_a (plus a batch mean).
2. The target is `act_emb` = `output['action_{ℓ}']`, the **post-`action_embed`** tensor. On DROID, `action_embed` is Identity at both levels, so the target is the z-scored raw 7-d command at ℓ=1 and the 8-d pooled macro-action at ℓ=2, matching the paper. If IDM were enabled in a single-scene config, the target would be the D-dim `Embedder` output, and `action_dim = levelN.predictor.action_dim` would not exist there **(inferred)**.
3. The detach matches the paper's "the gradient reaches the encoder only through the two states". The action encoder gets no IDM gradient.
4. IDM runs over all T = history+1 states (T=8 at DROID L1, 3 at L2).

**Action SIGReg (§C.2, unnumbered):** 0.005·SIGReg on level-2 macro-actions, `act_emb[:, :T−1].transpose(0,1)` → `(T−1, B, 8)` (`hjepa_utils.py:232-238`). The module is `SIGReg()` with defaults (`hjepa_utils.py:432-433`).

---

### Eq. (9): SIGReg

$$
\mathrm{SIGReg}\big(Z^{(\ell)}\big)=\frac{1}{M}\sum_{m=1}^{M}T_{EP}(h_m),\qquad h_m=Z^{(\ell)}u_m,\ u_m\in S^{D-1}.
$$

**Code.** `SIGReg.forward` (`h_jepa/loss.py:13-47`). The full derivation is in §3 below.

**Paper vs code.** The code additionally averages over the T timestep groups: `statistic.mean()` over a `(T, M)` tensor.

---

### Eq. (10): slow-feature-collapse degree

$$
\chi(E)=\frac{\mathbb E_b\big[\operatorname{tr}\operatorname{Var}_t(z_t\mid b)\big]}{\operatorname{tr}\operatorname{Var}_{b,t}(z_t)} .
$$

**Code:** not present (no within/between-episode variance analysis script).

---

### Eq. (11): SIGReg barrier against collapse when time is in the sample axis

$$\Delta=\lambda\,(T-1)\,\kappa,$$

where κ = E[T_EP] under the target, which does not depend on n.

**Code:** none. The training code uses the (T, N, D) grouping, which places time in the group axis, so Δ=0 for every shipped model, as the paper says. Numerically κ ≈ 1.05 for this implementation; see §3.4.

---

### Eq. (12): IDM floor under collapse

$$
\mathcal L_{\text{idm}}\ge\frac{1}{d_a}\frac{1}{T-1}\sum_{t=1}^{T-1}\Big(\operatorname{tr}\operatorname{Var}(a_t)-\operatorname{tr}\operatorname{Var}\big(\mathbb E[a_t\mid\mathcal C]\big)\Big).
$$

**Code:** none (theory). It equals 1 when actions are z-scored and not predictable from the episode.

---

### Unnumbered math in the paper

| Expression | Where | Code |
|---|---|---|
| K_{ℓ+1} = ⌊H_ℓ/s_{ℓ+1}⌋ ≤ H_{ℓ+1}; H_ℓ = K s + w − 1 | §2.2, Table 2 | assert at `hierarchical_solver.py:192-203` |
| C^{(1)} = ‖E^{(2)}(ẑ^{(1)}_2) − ẑ^{(2)}_1‖² | Fig. 3 | as Eq. 6 |
| Commit schedule: ξ_j = t_{j+1} − t_j ≤ H₁ | §E | `WorldModelPolicy.get_action` (`stable_worldmodel/policy.py:404-428`), ξ ≡ R·action_block = 10 env steps |
| NMSE = mean_d MSE_d / Var_d | §3, §D | `h_jepa/final_probing_decoding_eval.py` (macro NMSE, around line 262) |
| Entity-frequency gap = max centroid / min centroid | §3, §F | not in code |
| Monotonicity = −Spearman(‖z_t−g‖², t) | §I, Table 10 | not in code |
| Irreducible risk 𝓔(E) = E‖z_{t+1} − E[z_{t+1}│z_t,a_t]‖²; ν(E) = E tr Var(E(o_{t+1})│ξ_t,a_t); Theorem 1: 𝓔 ≥ ν ≥ 0 | §L.4 | theory only |
| Proposition 1 (feasibility of collapse under any marginal regulariser) | §L.3 | theory only |
| L_sim = (1/(T−1)) Σ_t ‖z_{t+1} − z_t‖², weight λ_sim < 0 | §L.6 | **not in code** |
| ϱ(f) = E(f(o_{t+1}) − f(o_t))², θ(f) = 𝓔(f)/ϱ(f); net cost ϱ(f)(θ(f) − │λ_sim│) | §L.6 | theory only |
| Effective rank | §K.1 | not in code |
| Action sensitivity | §L.4 | not in code |

---

## 3. SIGReg in full

### 3.1 Statistic

Take a univariate sample h = (h_1,…,h_n). Its empirical characteristic function is φ̂(t) = (1/n)Σ_j e^{i t h_j}, and the N(0,1) characteristic function is φ(t) = e^{−t²/2}. The Epps–Pulley statistic with Gaussian weight w(t) = e^{−t²/2} is

$$
T_{EP}(h)=n\int_{-\infty}^{\infty}\big|\hat\varphi(t)-\varphi(t)\big|^2 w(t)\,dt
= n\int_{-\infty}^{\infty}\Big[\big(\overline{\cos}(t)-e^{-t^2/2}\big)^2+\overline{\sin}(t)^2\Big]e^{-t^2/2}\,dt ,
$$

with $\overline{\cos}(t)=\frac1n\sum_j\cos(th_j)$ and $\overline{\sin}(t)=\frac1n\sum_j\sin(th_j)$.

Code (`loss.py:16-26, 34-46`):
- `phi = exp(−t²/2)` is the target characteristic function;
- `weights = trapezoid_weights × exp(−t²/2)` is the weight w(t) folded into the quadrature;
- `err = (cos − phi)² + sin²`;
- `statistic = (err @ weights) · n`.

The weight is the un-normalised e^{−t²/2} (no 1/√(2π)).

### 3.2 Quadrature

The integrand is even in t (cos̄ is even and sin̄² is even), so ∫_{−∞}^{∞} = 2∫_0^∞, truncated at 3.

Knots: `t = linspace(0, 3, 17)`, so Δt = 3/16 = 0.1875.

Weights: `[Δt, 2Δt, …, 2Δt, Δt]`. These are the trapezoid weights (Δt/2 at the ends, Δt inside) **times 2** for the half-domain folding.

**(verified)** Σ weights·e^{−t²/2} = 2.49967, against the exact ∫_{−3}^{3} e^{−t²/2} dt = 2.49986.

Cost is O(n·M·17) per timestep group. The matrix multiply `proj @ A` dominates: O(T·n·D·M).

### 3.3 Projections

`A = randn(D, M); A /= ‖A‖_col`, so M=1024 i.i.d. uniform directions on S^{D−1}. They are **redrawn on every call** (every forward, every level, every component) from the global torch RNG, and are not seeded separately.

`x_t = (proj @ A).unsqueeze(−1) * t` has shape `(T, n, M, 17)`.

The ECF means are taken over the sample axis −3 (= n): `cos, sin` have shape `(T, M, 17)`.

`statistic` has shape `(T, M)`, and `.mean()` averages over timesteps and directions. That gives Eq. (9) per timestep group, averaged over the T groups.

The input is `emb.flatten(2).transpose(0,1)`, i.e. `(T=4, B=128, D)` for single-scene and `(T=8 or 3, B=128 per rank, 384)` for DROID. Each test therefore uses n = B (128) samples of one timestep.

### 3.4 Scale and null value

Under the null (h ~ N(0,1) i.i.d.), E|φ̂(t) − φ(t)|² = (1 − e^{−t²})/n. Hence

$$\kappa=\mathbb E[T_{EP}]=\int_{-3}^{3}(1-e^{-t^2})e^{-t^2/2}dt\approx 2.4999-1.4472\approx 1.053,$$

independent of n **(inferred derivation)**.

**(verified)** With `SIGReg()(randn(4,n,192))`:

| n | Gaussian input | all-zeros (collapsed) input | 3×Gaussian (wrong scale) |
|---|---|---|---|
| 32 | 1.10 | 12.9 | 18.3 |
| 128 | 1.04 | 51.5 | 67.2 |
| 1024 | 1.05 | 412 | 524 |

The collapsed value grows like ≈ 0.402·n. The paper's §K.3 says the same: a null floor of 1.04–1.37, and linear growth in n on collapsed data.

The target is **isotropic unit-variance** N(0, I_D). So SIGReg also fixes the embedding scale to ≈1 per projected direction. This matters for interpreting latent distances and the planning-cost magnitudes **(inferred)**.

### 3.5 Distributed version (§K.3)

Code (`loss.py:37-44`), when `torch.distributed` is initialised:
- all-reduce `cos` and `sin` (the `(T,M,17)` tensors), divide `.data` by W, and set `n ← n·W`.
- So φ̂_global = (1/W)Σ_r φ̂_r, and the statistic is `n_loc·W·∫|φ̂_global − φ|²w`. This is exactly invariant to sharding with equal shards, since the loaders use `drop_last=True`.
- Gradient: the all-reduce is not tracked by autograd. Each rank backpropagates ∂T/∂φ̂_global through its own local φ̂_r, and DDP's gradient average over ranks supplies the 1/W factor. The result is the exact gradient of the global statistic (code comment, consistent with the chain rule: ∂φ̂_global/∂φ̂_r = 1/W) **(inferred check)**.
- Communication: two `T×M×17` tensors per call. The paper says "M × 17"; the code's tensor also has the T axis.
- **Caveat (inferred):** the all-reduce is only meaningful if every rank uses the **same** random directions A. The code draws A from each rank's default RNG and never broadcasts it. Correctness relies on all ranks having identical RNG streams (same seed and an identical sequence of random calls, e.g. dropout). The paper does not discuss this.

---

## 4. Hierarchical planning: algorithm and code

### 4.1 Call graph

```
planning_eval.run_planning_eval / _run_chunked_eval            (torch.enable_grad(), sdpa_kernel(MATH))  planning_eval.py:369,471
 └ _build_policy → load_model (model.eval())                    planning_eval.py:179-199, 208-229
     └ build_solver → build_hierarchical_solver                hierarchical_solver.py:21-34
          level_solvers[ℓ] = GradientSolver(model=LevelCostModel(JEPA_ℓ, upper_models=[JEPA_{ℓ+1},…]))
 └ swm.World.evaluate_from_dataset → World.step → policy.get_action(infos)   stable_worldmodel/world.py:276-282
     └ WorldModelPolicy.get_action                             stable_worldmodel/policy.py:387-439
          if action buffer empty: HierarchicalSolver.solve(info, steps_taken, eval_budget)   hierarchical_solver.py:90-216
             for ℓ = L … 1: GradientSolver.solve(level_info, planning_horizon)               gd.py:273-436
                 per GD iter: LevelCostModel.get_cost → JEPA.rollout → _project_to_goal_space → JEPA.criterion
```

### 4.2 Step by step (paper Eqs. 4–6, §E) with the matching code

**0. Commit / receding horizon.**
- `WorldModelPolicy` keeps a deque of env-step actions (`maxlen = receding_horizon × action_block` = 2×5 = 10).
- When the deque is empty, it calls the solver, keeps `actions[:, :receding_horizon]` (2 level-1 blocks), reshapes each 5·d block into 5 env actions, and enqueues them (`policy.py:404-428`).
- Each env step pops one action and `inverse_transform`s it from z-score to raw units (`policy.py:430-436`).
- Replanning is from scratch: `init_action=None`, so there is no warm start (`hierarchical_solver.py:205-210`, `gd.py:241-244`).
- So t_j = 10j and ξ_j = 10, truncated by the episode budget in `World.evaluate_from_dataset` (210 env steps for FourRoom, 350 for Ant).

**1. Encode current observation and goal at every level** (`HierarchicalSolver._encode_all_levels`, `hierarchical_solver.py:280-345`).
- Under `torch.no_grad()`: level 1 encodes `{'pixels': (B,1,3,H,W), 'proprio': ...}`, and each level ℓ>1 encodes `embed_{ℓ−1}` (one frame, repeated to `window_size` if needed) with `chunk_temporal_inputs=True`.
- It keeps `embed_ℓ[:, −1:]`, detached, shape `(B,1,D)`. The same is done with the `goal` / `goal_proprio` keys.
- No actions are passed, so the action encoder is not run and the queue is untouched.

**2. Horizon schedule** (`HierarchicalSolver._horizon_schedules`, `hierarchical_solver.py:241-278`; helpers in `stable_worldmodel/solver/solver.py:36-131`).
- `calls = ceil(eval_budget / (R·action_block))`, and `call_idx = steps_taken // (R·action_block)`.
- Top level: `decreasing_horizon_schedule(calls, H_L)`. Every value H_L…2 gets ⌊calls/H_L⌋ calls, and the remainder goes to the *smallest* values; if calls < H_L, the schedule interpolates linearly.
- Lower level ℓ: `[pre_decay_horizon] × tail_start + decreasing_horizon_schedule(calls − tail_start, H_ℓ)`, where `tail_start` is the first index at which the level above is 1. `pre_decay_horizon` defaults to the upper level's stride; the configs set it to 2.
- **(verified)** FourRoom L3 (budget 210 → 21 calls):
  - L3: `[6,6,6,5,5,5,4,4,4,3,3,3,3,2,2,2,2,1,1,1,1]`
  - L2: `[2]*19 + [1,1]`
  - L1: `[2]*20 + [1]`
- This matches §E: "Each lower level holds horizon two until the level above reaches one, then decreases from its own configured H to one".

**3. Top-down loop** (`hierarchical_solver.py:121-214`), for ℓ = L…1:
- If ℓ>1 and H_ℓ = 1: **skip** the level. With `horizon_one_goal_cost_space: upper` (FourRoom/Ant), remember `skipped_goal_level` = the highest skipped level. The next planned level then scores against the **real goal in that skipped level's space**, via dense lifting (§E: "retain the goal cost in the highest skipped level's latent space"). With `lower` (Push-T/Cube), the next planned level uses its own goal embedding.
- If no plan comes from above: `goal_embed_0 = g^{(goal_level)}` (Eq. 4, or a projected goal cost).
- Otherwise: subgoals = `pred[:, 1:1+K]` from the upper level's final rollout, and macro-action targets = `actions[:, 0:K]`. If the chain is anchored, the last subgoal is replaced by g^{(ℓ+1)} (Eq. 6).
- Run `GradientSolver.solve`, then take `subgoal_latents = result['predictions']['predicted_embed_0']` (shape `(B, H+1, D)`, index 0 = z₀) and `subgoal_actions = result['actions']` (shape `(B, H, a_dim)`).
- Return the level-1 `actions`, shape `(B, H₁, frameskip·act_dim)`.

**4. Per-level GD solve** (`GradientSolver.solve`, `gd.py:273-436`):
- **Initialisation** (`init_action`, `gd.py:217-271`):
  - prior (mean, std) = the per-dimension mean and population std (`unbiased=False`, clamped ≥ 1e-4) of `JEPA.latent_action_queue` (`gd.py:161-173`); at ℓ>1 an empty queue is a hard error;
  - at level 1 (no queue) the prior is mean 0, std 1;
  - std ×= `var_scale` (1.0; 1.5 on DROID);
  - all S candidates start at the mean; candidates 1..S−1 get `randn·std` added. **Candidate 0 stays exactly at the mean at every level, level 1 included** (paper: "one candidate at its mean" is stated for upper levels only).
- **Latent-action queue** (`JEPA._update_latent_action_queue`, `jepa.py:276-297`):
  - filled only in `self.training` mode, during training `encode` calls;
  - each batch appends `latent_actions[:, 0].detach()`, the first macro-action of each clip's **uncropped** upper stream, and keeps the last 4000 rows;
  - stored as a persistent buffer, so it is in checkpoints (`jepa.py:246-274`).
- **Clipping bounds** (`gd.py:175-196`):
  - with `action_clip_sigma=None` (single-scene): per-dimension 2%/98% `torch.quantile` of the queue; levels with no queue (level 1) are not clipped;
  - with `action_clip_sigma=σ` (DROID, σ=2): `mean ± σ·std` of the queue, or ±2 around 0/1 if there is no queue (level 1). The unscaled std is used.
  - Clipping is applied after every optimizer step (and after noise).
- **Optimiser:** a fresh `torch.optim.AdamW([batch_init], lr)` per solve. `weight_decay` is not set in the single-scene configs, so the torch default 0.01 applies (DROID sets 0.01 explicitly). Decoupled weight decay pulls upper-level candidates toward **0, not toward the queue mean** **(inferred side-effect)**.
- **Loop** (`gd.py:331-394`), at most `n_steps=90` iterations:
  - `costs = model.get_cost(info, batch_init)`, shape `(B,S)`;
  - `costs.sum().backward(); optim.step(); zero_grad()`;
  - optionally `+ action_noise·randn` (0.01 on Push-T/Cube upper levels), then clamp.
- **Early stopping:** score = mean over envs of `min_S costs` (computed *before* this step's update). It is "significant" if `best − score > 1e-2·|best|`. Stop when `step+1 ≥ min_its (30)` **and** 5 consecutive iterations are not significant. Every env is solved separately (`batch_size: 1`), so early stopping is per env.
- **Selection:** recompute `get_cost` under `no_grad` on the final candidates and pick the argmin per env (`gd.py:402-412`).
- **Final rollout:** `_rollout_final_actions` (`gd.py:135-159`, under `torch.inference_mode`) produces the predicted trajectory handed down as subgoals.

**5. Rollout** (`JEPA.rollout`, `jepa.py:376-418`), with history 1:
- `emb = z₀` expanded to `(B·S, 1, D)`.
- For k = 0…H−1: `pred = predict(emb[:, −1:], action_embed(act)[:, −1:])[:, −1:]`, then append. Each step is a **length-1 context**: the predictor sees only `pos_embedding[:, :1]`. This is the paper's "c_ℓ = 1 at planning" (§E).
- Output `predicted_embed_0` has shape `(B, S, H+1, D)` with index 0 = z₀.
- **(verified)** FourRoom L3 with S=4 at call 0:
  - L3: actions `(1,4,6,4)` → predictions `(1,4,7,192)`
  - L2: actions `(1,4,2,4)` → predictions `(1,4,3,192)`
  - L1: actions `(1,4,2,10)` → predictions `(1,4,3,192)`
  - Returned plan: `(1,2,10)`.
- Quirk: `rollout` reads the history length from `info['pixels'].size(2)`, so a `pixels` key must exist even at upper levels. It is present because `level_info = dict(info_dict)`.

**6. Cost per level** (`LevelCostModel.get_cost`, `hierarchical_solver.py:405-427`):
- Top level, or a level with no plan above it: `‖ẑ_H − g‖²`, where q = `cost_last_n` of the last steps are scored. If the goal is in a higher (skipped) space, every step is densely lifted through E^{(ℓ+1)}, … up to `goal_level` first.
- Lower level: Eq. 6 on the lifted rollout, + `action_cost_weight·c_act`.
- With the shipped closed-loop configs:
  - FourRoom/Ant L1 cost = ‖E₂(ẑ^{(1)}_2) − ẑ^{(2)}_1‖²;
  - L2 cost = ‖E₃(ẑ^{(2)}_2) − ẑ^{(3)}_1‖²;
  - L3 cost = ‖ẑ^{(3)}_{H₃} − g^{(3)}‖².

### 4.3 Settings in the shipped eval configs (match paper Table 7)

| | FourRoom L3 (`config/eval/fourroom_l3.yaml`) | Ant L3 (`config/eval/ant_l3.yaml`) |
|---|---|---|
| S (L1,L2,L3) | 50, 200, 200 | 100, 100, 400 |
| AdamW lr (L1,L2,L3) | 0.08, 0.1, 0.1 | 0.06, 0.1, 0.1 |
| H (L1,L2,L3) | 2, 2, 6 (pre_decay 2,2) | 4, 2, 5 (pre_decay 2,2) |
| receding_horizon × action_block | 2 × 5 | 2 × 5 |
| cost_last_n (L1,L2,L3) | 2,1,1 | 2,1,1 |
| horizon_one_goal_cost_space | upper | upper |
| n_steps / min_its / patience / rel_delta | 90 / 30 / 5 / 1e-2 (defaults) | same |
| eval budget | 210 | 350 |

### 4.4 Which tensors carry gradients during planning

- **Leaf being optimised:** `batch_init` (`(1,S,H,a_dim)`, `requires_grad=True`). It is the only tensor in the optimizer.
- **Gradient path:** `action_embed` → predictor (all H steps; backprop through time) → [upper encoder(s) E^{(ℓ+1)}, …] → MSE to the detached goal.
- **Detached:** z₀ and goal/subgoal embeddings (`no_grad` in `_encode_all_levels`; `.detach()` in `criterion`; `.clone()` of the inference-mode final rollout for subgoals; the target action in `_action_cost` is detached).
- **Model parameters are never frozen** (`requires_grad=True`) and **`.grad` accumulates on them across every planning iteration**, because `cost.backward()` populates the model's parameter grads and only the action optimizer's grads are zeroed. **(verified)** After one mini hierarchical solve, 209 parameter tensors had non-None `.grad`. Consequences:
  - extra backward compute for weight gradients;
  - **a TTT implementation must zero model grads (`model.zero_grad(set_to_none=True)`) before its own backward, or use `torch.autograd.grad`**;
  - `GradientSolver` is an `nn.Module` whose `.model` is the `LevelCostModel` wrapping the JEPA, so `solver.parameters()` includes the world-model weights and `solver.init`.
- Eval runs under `torch.enable_grad()` and `sdpa_kernel(SDPBackend.MATH)` (`planning_eval.py:369, 471`).

### 4.5 FLOP counting

**The repo does not count FLOPs.** It only logs per-solve records `{level, horizon, num_samples, n_iters (per env), wall_clock}` (`gd.py:427-434`), collected by `planning_eval.pop_solve_records` (`planning_eval.py:305-310`) into `planning_compute.json`. The paper says FLOPs "include forward and backward passes" and calls them "measured" (Fig. 9c). The script that turns records into TFLOPs is not in this clone.

An estimate consistent with the records **(inferred)**:

per solve at level ℓ,
$$
\text{FLOPs}\approx (n_{\text{iters}}\cdot c_{fb} + 1)\cdot S\cdot\Big[H_\ell\,(F_{\text{pred},1\text{tok}}+F_{\text{act\_embed}}) + n_{\text{lift}}\,F_{E^{(\ell+1)}}(+\dots)\Big] + H_\ell\,F_{\text{pred}}\ (\text{final rollout, }S{=}1)
$$

- c_fb ≈ 3 for forward + backward. Because parameter gradients are also computed (§4.4), the true backward is ≈ 2× forward.
- n_lift is ⌊H/s⌋+1 frames for strided lifting, or H+1 for dense lifting.
- Observation and goal encoding (2 ViT passes + upper MLPs per replan) is small by comparison.
- Sum over levels, replans and envs.

### 4.6 Training-mode hazards during planning

`load_model` calls `model.eval()` (`planning_eval.py:197`), and the in-training callback does the same (`h_jepa/utils.py:273`, then restores train mode at line 281). Planning therefore uses BN running statistics and no dropout. See §6.5.

---

## 5. Module and parameter structure (instantiated, verified)

Built with `hjepa_utils.create_world_model(compose('fourroom_hjepa_l3' | 'ant_hjepa_l3'))` on CPU. Counts are trainable parameters (BN running stats and the action queue are buffers and are not counted).

### 5.1 `fourroom_hjepa_l3`: total 21,664,134

**Level 1: 11,288,046** (stride 1, target_length 4, no queue)
- `encoder: ProjectedEncoder` 6,294,336
  - `encoder: ViTModel` (HF; tiny: 12 layers, width 192, 3 heads, MLP 768, patch 14, 224², `use_mask_token=True`, dropout 0, no pooler; output = final-LN CLS) — 5,501,568
  - `projector: MLP` Linear(192→2048)–**BatchNorm1d(2048)**–GELU–Linear(2048→192) — 792,768
- `predictor: ProjectedPredictor` 4,837,504
  - `ARPredictor` 4,044,736:
    - `pos_embedding (1,3,192)`, randn init;
    - `Transformer`: input_proj/cond_proj Identity (192=192), 4× `ConditionalBlock`. Each block: AdaLN-zero `Linear(192→1152)`; `Attention` (its own LayerNorm, qkv 192→3·8·64=1536 without bias, out 512→192, dropout 0.1, causal SDPA); `FeedForward` (LN, 192→1024, GELU, Dropout 0.1, 1024→192, Dropout 0.1);
    - final LayerNorm(192); output_proj Identity.
  - `projector: MLP` 192→2048–BN–GELU→192 — 792,768
- `action_embed: Embedder` 156,206: Conv1d(10→10, k=1) 110 + Linear(10→768)–SiLU–Linear(768→192) 156,096
- `action_encoder: Identity`

**Levels 2 and 3: 5,188,044 each** (stride 2, window 1, target_length 4, `latent_action_queue` size 4000 × 4)
- `encoder: ProjectedEncoder(ResidualLatentMLP)` 148,416: LayerNorm(192) → Linear(192→384) → GELU → Linear(384→192); final LN off; **no residual connection despite the name**; projector Identity.
- `predictor` 4,837,504: same structure as level 1 (4 layers, 8 heads × 64, width 192, FF 1024, BN-MLP projector).
- `action_embed: Embedder(patch_embed=False)` 151,488: Linear(4→768)–SiLU–Linear(768→192).
- `action_encoder: SequenceEncoder` 50,636:
  - PerStepMLP Linear(192→64);
  - learned pos/CLS table `(max_chunk+1=4, 64)`;
  - 1× `nn.TransformerEncoderLayer` (d=64, 2 heads, FF 128, dropout 0.1, pre-norm);
  - head Linear(64→64)–GELU–LN–Linear(64→4);
  - **final LayerNorm(4)** (affine).

  The macro-action is therefore layer-normalised across its 4 dims (≈ zero mean and unit variance per vector, up to the affine parameters). Per-dimension percentile clipping during planning does not enforce this constraint **(inferred)**.

### 5.2 `ant_hjepa_l3`: total 34,758,834

**Level 1: 12,349,210**
- `encoder: FusionEncoder` 6,362,368. Its output `embed = cat(pixel 256, proprio 128)`; the fusion projector is Identity.
  - `pixel_encoder: ProjectedEncoder` 6,321,792: ViT-tiny at 64², patch 4 (257 tokens), plus a BN-MLP projector 192→2048→**256**.
  - `proprio_encoder: ProjectedEncoder(MLP)` 40,576: Linear(27→256)–**LayerNorm**–GELU–Linear(256→128); projector Identity. The proprio input is z-scored qpos[2:] ⊕ qvel (27-d).
- `predictor` 5,379,328:
  - ARPredictor 4,193,152: pos (1,3,384); input_proj and cond_proj Linear(384→192); 4 blocks, width 192, 8 heads × 64, FF 1024; output_proj Identity;
  - BN-MLP projector 192→2048→384 — 1,186,176.
- `action_embed` 607,514: Conv1d(40→10) 410 + Linear(10→1536)–SiLU–Linear(1536→384).
- `action_encoder: Identity`.

**Levels 2 and 3: 11,204,812 each** (queue 4000 × 4)
- encoder `ResidualLatentMLP` 296,448: LN(384)–Linear(384→384)–GELU–Linear(384→384).
- predictor 10,247,552:
  - ARPredictor 8,668,160: input/cond proj 384→224; 5 blocks, width 224, 12 heads × 64 (inner 768), FF 1536; output_proj 224→384;
  - BN-MLP projector 384→2048→384 — 1,579,392.
- action_embed 597,888: Linear(4→1536)–SiLU–Linear(1536→384).
- action_encoder `SequenceEncoder` 62,924 (step MLP 384→64, otherwise as FourRoom).

Both models instantiated without errors. Losses created: `sigreg_level1..3` (no IDM, no action SIGReg).

**Matches the paper's Table 4 / §C.1:**
- ViT-Tiny with patch 14 or 4;
- BN projector;
- Ant 256 + 128 concatenation;
- L1 predictor 4/8/192/1024;
- upper encoder: two-layer MLP, hidden 384;
- one-layer transformer action pooling to 4-d;
- upper predictors 4/8/192/1024 (FourRoom) and 5/12/224/1536 (Ant);
- λ 0.18 / 0.09 at every level; LR 5e-5 / 1e-5; 7 / 5 epochs; batch 128.

**Details the paper does not state:**
- projector hidden width 2048;
- head dim 64 (attention inner dim 512/768 ≠ width);
- the input LayerNorm on the upper encoder;
- AdaLN-zero init;
- std-1 positional embeddings;
- the separate `action_embed` MLP.

---

## 6. Hook points for test-time training (JEPA-TTT) and looped planners

### 6.1 Predictor forward on a window of latents and actions
- `JEPA.predict(emb, act_emb)` (`jepa.py:355-360`) takes `emb (B,T,D_ℓ)` and `act_emb (B,T,D_ℓ)`, where `act_emb` comes from `JEPA.action_embed(actions)`. It returns `(B,T,D_ℓ)`, where output τ is the teacher-forced prediction of z_{τ+1}, causally masked.
- **The window must satisfy T ≤ history_size = 3** for `ARPredictor`, because of the `pos_embedding` length (`predictors.py:26,46`). DROID's predictor allows up to 16.
- For a loss identical to training, slice exactly as `hjepa_forward` does (`hjepa_utils.py:157-159, 194-197`): `pred = jepa.predict(z[:, :c], a[:, :c])` and `F.mse_loss(pred, z[:, 1:c+1])`.
- Autoregressive multi-step: `JEPA.rollout(info, actions (B,S,H,a), history_size=1)` (`jepa.py:376-418`). `info` needs `embed_0 (B,S,1,D)` (or raw obs) and a `pixels` key, whose `.size(2)` is read as the history length.
- `hjepa_forward(self_like, batch, stage, cfg, normalize_batch)` gives the complete multi-level loss. It needs an object with `.model`, `.sigreg_level{ℓ}` and `.log_dict`; a `SimpleNamespace` works **(verified)**. The batch must hold `pixels_level1 (B,T1,3,H,W)` and `action_level1 (B,T1,frameskip·a)` (plus `proprio_level1` for Ant) with T1 ≥ 4/7/13/25 for 1–4 levels. Random crops are drawn inside.

### 6.2 Encoding new observations
- Planning path (no gradient): `HierarchicalSolver._encode_all_levels(info_dict, key='pixels'|'goal')` (`hierarchical_solver.py:280-345`). It returns `{'embed_ℓ': (B,1,D_ℓ)}`, detached.
- With gradient (TTT):
  - level 1: `jepa1.encode({'pixels': x(B,T,3,H,W), 'proprio': p(B,T,27)}, key='pixels')['embed_0']`;
  - level ℓ>1: `jepa_ℓ.encode({f'embed_{ℓ-1}': z(B,T,D), 'action': a_emb(B,T,D)}, key=f'embed_{ℓ-1}', chunk_temporal_inputs=True)` gives `embed_0` and `action_0` (stride subsampling and macro-action pooling).
  - Or use the whole stack: `HJEPA.encode_hierarchical(batch)` (`hjepa.py:239-294`).
- `JEPA.encode` **mutates and returns the dict you pass**: it adds `embed_0`, `action_0`, `pixel_embed_0`, ….
- Pixel preprocessing at eval: `planning_eval.img_transform` (`planning_eval.py:27-35`; ToImage → float → ImageNet normalise → Resize). Proprio and actions are z-scored by `process` from the training `normalizer.pt` (`planning_eval.py:117-168`).

### 6.3 Action encoders per level
- Level 1: `jepa1.action_encoder` is `Identity`. `jepa1.action_embed` (`Embedder`, `module.py:189-231`) maps the z-scored action block `(…, frameskip·act_dim)` to D₁.
- Level ℓ≥2:
  - `jepa_ℓ.action_encoder` (`SequenceEncoder.forward(inputs (B,T,L=s_ℓ,D_{ℓ−1}), padding_mask (B,T,L))` → `(B,T,4)`; `models/encoders/seq_encoder.py:30-125`; DROID uses `SequenceMLPEncoder`, `seq_encoder.py:145-166`) produces the macro-action, i.e. the planner's variable;
  - `jepa_ℓ.action_embed` maps it 4 → D_ℓ for the predictor.
- Pooling a level-ℓ action sequence for level ℓ+1 the way the planner does it: `LevelCostModel._encode_actions_up` (`hierarchical_solver.py:487-513`).
- Prior queue: `JEPA.get_latent_action_queue()` (`jepa.py:299-305`).

### 6.4 Where to put an optimizer step between replans
- **Best place:** `WorldModelPolicy.get_action`, inside `if len(self._action_buffer) == 0:` and immediately before `outputs = self.solver(...)` (`stable_worldmodel/policy.py:404-415`). At that point the new observation `info_dict` is available, already transformed (`_prepare_info`, line 401).
- The policy stores **no history**. A TTT subclass must record, on every `get_action`:
  - the prepared observation (`info_dict['pixels']`, shape `(n_envs,1,3,H,W)`, plus `proprio`);
  - the action popped at line 430 **before** `inverse_transform` (line 436), which is the z-scored env action;
  - then regroup each 5 consecutive env actions into one level-1 block (`frameskip·act_dim`), matching `action_level1`.
  - With R=2 there are 2 new level-1 transitions per replan. A level-2 transition needs 2 level-1 blocks; a level-3 transition needs 4.
- **Alternative:** override `HierarchicalSolver.solve` (`hierarchical_solver.py:90-119`) and do the step before `_encode_all_levels`. `self.level_models[ℓ−1]` is the JEPA of level ℓ, and `self.level_solvers[ℓ].model.model` is the same object.
- **Before the TTT backward, call `model.zero_grad(set_to_none=True)`**, because planning has left accumulated `.grad` on every model parameter (§4.4). Planning also runs inside `torch.enable_grad()` with the MATH SDPA kernel.
- `policy.reset()` (`policy.py:375-379`) runs at episode start (via `World.reset`). Use it to reset the TTT buffers, and to restore weights if adaptation should be per episode.
- `load_model` moves the model to `"cuda"` unconditionally (`planning_eval.py:188`). Solver `device: cuda` is set in the eval YAMLs.

### 6.5 Eval mode, dropout and BatchNorm
- The model is put in eval mode by `planning_eval.load_model` (`planning_eval.py:197`) and `swm.policy._load_model_with_attribute` (`policy.py:484-485`); the in-training eval callback sets eval and restores afterwards (`utils.py:273-281`).
- **Dropout** is active in train mode in:
  - every predictor: `Attention` SDPA `dropout_p=0.1` only when `self.training` (`module.py:83`); `to_out` Dropout 0.1; `FeedForward` Dropout 0.1 ×2;
  - every upper `SequenceEncoder` action encoder (dropout 0.1);
  - DROID `CausalTransformerPredictor` (0.1).

  The ViT dropouts are 0.0, and the Embedder and latent encoders have no dropout.
- **BatchNorm1d** sits in the level-1 pixel-encoder projector and in **every predictor projector at every level** (single-scene configs). In `train()` mode, BN normalises with the statistics of the current flattened `(B·T)` rows and updates the running stats (momentum 0.1). During planning that batch is the S candidates × context, so train mode would couple candidates and drift the running stats **(inferred)**.
  - Recommendation: compute TTT losses with the model in `eval()` mode. Gradients still flow, BN uses frozen running stats, and dropout is off. Alternatively set only the BN modules to eval.
  - DROID's level-1 projector is a LayerNorm, so it has no BN at level 1, and its causal-transformer predictor has no BN.
- **`latent_action_queue`** is updated only when `self.training` and an `action` key goes through `JEPA.encode`. A TTT forward in train mode would push adaptation-time macro-actions into the queue and change later planner initialisation and clipping bounds.

---

## 7. Paper ↔ code mismatches and gaps

### In the paper, not in the code
1. FLOP computation (only `solve_records`).
2. χ(E) / latent-variance split (Eq. 10, Fig. 19), effective rank (Fig. 17), spectral centroids and frequency gap (Fig. 5, §F), Spearman monotonicity (Table 10), action sensitivity (Table 12).
3. Temporal-repulsion term L_sim (§L.6, Table 12).
4. The "time in the sample axis" SIGReg grouping used for the Table 12 ladder: the code always tests `(T,B,D)`.
5. Stagewise training (freeze level 1, then train level 2; §C.2, §G): there is no freeze option in `main_hjepa.py`.
6. The cost-geometry visualisation (Fig. 7) and decoded-plan figures, apart from the decoders in `final_probing_decoding_eval.py`.
7. The analytical DROID frame-rate and 1 fps baselines (V-JEPA 2-AC, JEPA-WM).

### In the code, not in the paper
1. `rollout_n>1` and `nsteps`/`parallel_unroll` multi-step objectives; `detach_pred_target` (this is only the ablation).
2. `window_size>1` support (prefix-repeat, `FlattenedSequenceEncoder`, dense lifting padding).
3. Goal anchoring of the last subgoal (Table 2 mentions it; the implementation details are code-only).
4. `horizon_one_goal_cost_space` and the dense multi-level projection machinery for skipped levels and `_project` configs.
5. Early-stopping constants (`min_its=30`, patience 5, relative delta 1e-2), AdamW default weight decay 0.01 on actions, and candidate 0 at the mean for level 1 too.
6. Sparse level-1 encoding.
7. Per-level grad clipping flag; L1 vs upper LR groups; the 1% warm-up comes from stable_pretraining defaults.
8. `ResidualLatentMLP` input LayerNorm and no residual; BN projectors with width 2048; std-1 `pos_embedding` in `ARPredictor`.

### Numeric or semantic differences
1. Eq. 2 is a per-element mean in code (÷D relative to ‖·‖²).
2. SIGReg is averaged over T groups as well as M directions; the distributed version relies on identical per-rank RNG for the projections.
3. Upper-level A^{(ℓ)} in code = `action_encoder` (to 4-d) ∘ then `action_embed` (to D). The planner optimises the 4-d latent.
4. DROID eval config `droid_l2.yaml` uses level-2 `num_samples: 4` and level-1 16, but Table 8 lists S = (16, 16). The `droid_eval.py` docstring (lines 12-17) gives H-JEPA level-2 lr 0.01, while the config and Table 8 say 0.3.
5. Paper §E says Level-1 candidates start from a "standard Gaussian". In code one candidate is at exactly 0, and level 1 is unclipped in single-scene configs.
6. The paper describes planning cost "C(z,z′) = ‖z−z′‖²". The code sums over the last `cost_last_n` steps with weight β on earlier ones; it equals the paper's form when K = `cost_last_n`.

### Could not verify
- Parameter counts for the DROID models (I did not instantiate them).
- The external FLOP accounting.
- The `unit_tests/test_sparse_level1_encode.py` referenced in `ARCHITECTURE.md`, which is not in this clone.
