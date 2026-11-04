"""H-JEPA FourRoom dynamics-shift benchmark ("before": frozen model, stock planner).

Run from the repo's h_jepa/ directory with the repo's Python 3.11 environment.

Subcommands
  collect      record shifted trajectories with the dataset's own collection policy (CPU)
  model-error  per-level latent prediction error of the frozen model on those trajectories (GPU)
  plan         closed-loop planning on the 50 fixed tasks under each shift, sharded over GPUs
  summarize    aggregate everything into summary.json (+ paired tests against the clean condition)

Every condition is a dynamics shift only: what the agent sees (rendering, distractors, goals,
tasks) is unchanged, only how a commanded action moves the agent. The "after" benchmark
(JEPA-TTT, looped planner) reuses this file unchanged and only swaps the model or planner, so the
two are compared on identical tasks, trajectories and metrics.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())  # h_jepa/ modules (data, planning_eval, the pickled model classes)

BENCH_VERSION = 1
ENV_ID = 'swm/FourRoomShift-v0'

# --------------------------------------------------------------------------------------
# Shifted environment
# --------------------------------------------------------------------------------------


def _rotate(a, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * a[0] - s * a[1], s * a[0] + c * a[1]], dtype=np.float32)


def make_shift_env_class():
    from stable_worldmodel.envs.four_room.env import FourRoomDistractorsEnv

    class FourRoomShiftEnv(FourRoomDistractorsEnv):
        """FourRoomDistractors with a test-time change to the action -> motion map.

        shift = {'type': 'none'}
              | {'type': 'rotate', 'deg': d}                       constant rotation of the commanded action
              | {'type': 'wave', 'amp_deg': A, 'period_px': P}     rotation A*sin(2*pi*x/P) that depends on the
                                                                   agent's x position (JEPA-TTT's Two-Room field)
        Agent speed is changed with the base env's own `agent_speed` argument.
        The commanded action is clipped to [-1, 1] first (as the base env does), then transformed;
        the base env clips again after the transform.
        """

        def __init__(self, *args, shift=None, **kwargs):
            self.shift = dict(shift or {'type': 'none'})
            assert self.shift.get('type', 'none') in ('none', 'rotate', 'wave'), self.shift
            super().__init__(*args, **kwargs)

        def transform_action(self, action):
            a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
            kind = self.shift.get('type', 'none')
            if kind == 'none':
                return a
            if kind == 'rotate':
                return _rotate(a, math.radians(float(self.shift['deg'])))
            x = float(self.agent_position[0])
            ang = math.radians(float(self.shift['amp_deg'])) * math.sin(
                2.0 * math.pi * x / float(self.shift['period_px']))
            return _rotate(a, ang)

        def step(self, action):
            return super().step(self.transform_action(action))

    return FourRoomShiftEnv


def register_shift_env():
    import gymnasium as gym
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point=make_shift_env_class())


# --------------------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------------------


def load_conditions(out):
    return json.loads((Path(out) / 'conditions.json').read_text())


def world_overrides(cond):
    over = dict(cond.get('world', {}))
    over['env_name'] = ENV_ID
    over['shift'] = cond.get('shift', {'type': 'none'})
    return over


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    raise TypeError(type(o))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, default=_json_default, indent=1))
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# collect: shifted trajectories with the dataset's own behaviour policy
# --------------------------------------------------------------------------------------


def cmd_collect(args):
    from omegaconf import OmegaConf
    import stable_worldmodel as swm
    from stable_worldmodel.envs.four_room import EpisodicPolicyMixture

    register_shift_env()
    repo = Path.cwd().parent  # run from h_jepa/
    base = OmegaConf.load(repo / 'scripts/data/config/fourroom_tp35_d1_val.yaml')
    conds = load_conditions(args.out)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for name in args.conditions.split(','):
        cond = conds[name]
        h5 = data_dir / f'{name}.h5'
        done = data_dir / f'{name}.done'
        if done.exists():
            print(f'[collect] {name}: already done')
            continue
        if h5.exists():
            h5.unlink()  # partial file from an interrupted run: start over (collection is deterministic)
        world_cfg = OmegaConf.to_container(base.world, resolve=True)
        world_cfg.update(world_overrides(cond))
        world_cfg['num_envs'] = int(args.num_envs)
        world_cfg['terminate_on_goal'] = False
        world_cfg['retarget_on_goal'] = True
        env_name = world_cfg.pop('env_name')
        world = swm.World(env_name, **world_cfg, render_mode='rgb_array')
        rng = np.random.default_rng(int(args.seed))
        policy = EpisodicPolicyMixture.from_config(
            base.ego_policy.mixture, total_episodes=int(args.episodes),
            seed=rng.integers(0, 1_000_000).item())
        world.set_policy(policy)
        t0 = time.time()
        world.record_dataset(name, episodes=int(args.episodes), seed=rng.integers(0, 1_000_000).item(),
                             cache_dir=str(data_dir))
        world.close()
        write_json(data_dir / f'{name}.collection.json', {
            'condition': cond, 'episodes': int(args.episodes), 'seed': int(args.seed),
            'max_episode_steps': int(world_cfg['max_episode_steps']), 'seconds': time.time() - t0,
            'policy_mixture': OmegaConf.to_container(base.ego_policy.mixture, resolve=True)})
        done.write_text('ok')
        print(f'[collect] {name}: {args.episodes} episodes in {time.time() - t0:.0f}s -> {h5}')


# --------------------------------------------------------------------------------------
# model-error: per-level latent prediction error of the frozen model
# --------------------------------------------------------------------------------------

HIST = 3  # predictor context (ARPredictor.pos_embedding has 3 positions)


def _load_model(policy, device):
    model = torch.load(policy, map_location='cpu', weights_only=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _level_geometry(model):
    geo = {}
    for lvl in range(1, model.num_levels + 1):
        j = model.get_level(lvl)
        geo[lvl] = (int(getattr(j, 'temporal_stride', 1)), int(getattr(j, 'temporal_window_size', 1)))
    return geo


def _level1_frames_needed(geo, top_steps):
    need = top_steps
    for lvl in range(len(geo), 1, -1):
        stride, window = geo[lvl]
        need = (need - 1) * stride + window
    return need


def _xy_for_level(out, lvl):
    v = out.get(f'xy_level{lvl}')
    if v is None:
        return None
    if v.ndim == 4:  # chunked windows (B, T, w, 2): the window's last state
        v = v[:, :, -1]
    return v


@torch.no_grad()
def _encode_batch(model, batch, device):
    from data import IMAGENET_STATS
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    px = b['pixels_level1']
    if px.dtype == torch.uint8:
        mean = torch.tensor(IMAGENET_STATS['mean'], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STATS['std'], device=device).view(1, 1, 3, 1, 1)
        b['pixels_level1'] = px.float().div(255.0).sub(mean).div(std)
    saved = [model.get_level(l).target_length for l in range(1, model.num_levels + 1)]
    try:
        for l in range(1, model.num_levels + 1):
            model.get_level(l).target_length = None  # keep full sequences (no random crop)
        out = model.encode_hierarchical(b, sparse_level1_encode=False)
    finally:
        for l, t in zip(range(1, model.num_levels + 1), saved):
            model.get_level(l).target_length = t
    return out


@torch.no_grad()
def _ar_errors(jepa, emb, act, ks, perm=None):
    """Autoregressive k-step prediction from the true context ending at index HIST-1.

    Position j of the predictor output predicts emb[j+1] from emb[..j], act[..j] (training
    alignment). Returns dict k -> per-sample MSE (mean over D), cosine and norm ratio.
    """
    s = HIST - 1
    kmax = max(ks)
    a = act if perm is None else act[perm]
    seq = emb[:, :HIST]
    preds = []
    for k in range(kmax):
        p = jepa.predict(seq[:, -HIST:], a[:, k:k + HIST])[:, -1]
        preds.append(p)
        seq = torch.cat([seq, p[:, None]], dim=1)
    res = {}
    for k in ks:
        p, t = preds[k - 1], emb[:, s + k]
        res[k] = {
            'mse': ((p - t) ** 2).mean(-1),
            'cos': torch.nn.functional.cosine_similarity(p, t, dim=-1),
            'norm_ratio': p.norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-8),
        }
    return res


def _ridge_fit(x, y, lam=1e-2):
    x1 = np.concatenate([x, np.ones((len(x), 1))], 1)
    reg = lam * np.eye(x1.shape[1])
    reg[-1, -1] = 0
    return np.linalg.solve(x1.T @ x1 + reg, x1.T @ y)


def _ridge_r2(w, x, y):
    x1 = np.concatenate([x, np.ones((len(x), 1))], 1)
    pred = x1 @ w
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-12))), float(np.sqrt(((y - pred) ** 2).sum(1).mean()))


def cmd_model_error(args):
    import stable_worldmodel as swm
    from data import Compose, get_column_normalizer_from_artifact

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    model = _load_model(args.policy, device)
    artifact = model.normalizer_artifact
    geo = _level_geometry(model)
    L = model.num_levels
    top_steps = HIST + int(args.top_k)
    n1 = _level1_frames_needed(geo, top_steps)
    level_cfgs = {'level1': {'frameskip': 5, 'num_steps': 2}}  # FourRoom level-1 frameskip (train config)
    for lvl in range(2, L + 1):
        level_cfgs[f'level{lvl}'] = {'frameskip': geo[lvl][0], 'window_size': geo[lvl][1],
                                     'num_steps': top_steps if lvl == L else 2}
    ks = {lvl: sorted({k for k in (1, 2, 4, 8) if k <= _steps_at_level(geo, lvl, n1) - HIST}) for lvl in range(1, L + 1)}
    print(f'[model-error] levels={L} geometry={geo} level-1 frames per clip={n1} horizons={ks}')

    out_dir = Path(args.out) / 'model_error' / f'seed{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_cache = {}
    for name in args.conditions.split(','):
        res_path = out_dir / f'{name}.json'
        if res_path.exists() and not args.force:
            print(f'[model-error] {name}: exists')
            continue
        ds = swm.data.HDF5Dataset(name, keys_to_load=['pixels', 'action', 'xy'], keys_to_cache=['action', 'xy'],
                                  cache_dir=args.data_dir, **level_cfgs)
        ds.transform = Compose(get_column_normalizer_from_artifact(artifact, 'action', 'action'))
        # clips start every span//5 raw steps (overlapping), evenly spread over episodes; standard
        # errors below are clustered by episode, so the overlap does not overstate precision
        span = ds.last_level['span']
        hop = max(1, span // 5)
        clips = [(ep, st) for ep, st in ds.clip_indices if st % hop == 0]
        rng = np.random.default_rng(0)
        if len(clips) > args.max_clips:
            clips = [clips[i] for i in sorted(rng.choice(len(clips), args.max_clips, replace=False))]
        ds.clip_indices = clips
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        acc = {lvl: {} for lvl in range(1, L + 1)}
        feats = {lvl: [] for lvl in range(1, L + 1)}
        xys = {lvl: [] for lvl in range(1, L + 1)}
        g = torch.Generator(device='cpu').manual_seed(0)
        t0 = time.time()
        for batch in loader:
            out = _encode_batch(model, batch, device)
            B = out['embed_1'].size(0)
            perm = torch.randperm(B, generator=g).to(device)
            for lvl in range(1, L + 1):
                jepa = model.get_level(lvl)
                emb, act = out[f'embed_{lvl}'].float(), out[f'action_{lvl}'].float()
                true = _ar_errors(jepa, emb, act, ks[lvl])
                shuf = _ar_errors(jepa, emb, act, ks[lvl], perm=perm)
                # teacher-forced one-step loss over every window (the training / JEPA-TTT loss)
                T = emb.size(1)
                tf = []
                for st in range(0, T - HIST):
                    p = jepa.predict(emb[:, st:st + HIST], act[:, st:st + HIST])
                    tf.append(((p - emb[:, st + 1:st + 1 + HIST]) ** 2).mean((-1, -2)))
                d = acc[lvl]
                d.setdefault('tf_mse', []).append(torch.stack(tf, 1).mean(1).cpu())
                d.setdefault('emb_var', []).append(emb.var(0).mean().repeat(B).cpu())
                for k in ks[lvl]:
                    d.setdefault(f'mse_k{k}', []).append(true[k]['mse'].cpu())
                    d.setdefault(f'cos_k{k}', []).append(true[k]['cos'].cpu())
                    d.setdefault(f'norm_ratio_k{k}', []).append(true[k]['norm_ratio'].cpu())
                    d.setdefault(f'shuffled_mse_k{k}', []).append(shuf[k]['mse'].cpu())
                    d.setdefault(f'copy_mse_k{k}', []).append(((emb[:, HIST - 1] - emb[:, HIST - 1 + k]) ** 2).mean(-1).cpu())
                xy = _xy_for_level(out, lvl)
                if xy is not None:
                    feats[lvl].append(emb.reshape(-1, emb.size(-1)).cpu().numpy())
                    xys[lvl].append(xy.reshape(-1, xy.size(-1)).float().cpu().numpy())
        result = {'condition': name, 'seed': args.seed, 'policy': args.policy, 'n_clips': len(clips),
                  'level1_frames_per_clip': n1, 'horizons': ks, 'seconds': time.time() - t0,
                  'bench_version': BENCH_VERSION, 'levels': {}}
        ep_of_clip = np.array([c[0] for c in clips])
        arrays = {'episode': ep_of_clip}
        for lvl in range(1, L + 1):
            lv = {}
            for key, parts in acc[lvl].items():
                v = torch.cat(parts).numpy().astype(np.float64)
                arrays[f'L{lvl}_{key}'] = v
                lv[key] = {'mean': float(v.mean()), 'se_episode': _cluster_se(v, ep_of_clip)}
            var = lv['emb_var']['mean']
            for k in ks[lvl]:
                lv[f'rel_mse_k{k}'] = lv[f'mse_k{k}']['mean'] / max(var, 1e-12)
                lv[f'action_gap_k{k}'] = lv[f'shuffled_mse_k{k}']['mean'] - lv[f'mse_k{k}']['mean']
                lv[f'beats_copy_k{k}'] = lv[f'mse_k{k}']['mean'] < lv[f'copy_mse_k{k}']['mean']
            if feats[lvl]:
                probe_cache[(name, lvl)] = (np.concatenate(feats[lvl]), np.concatenate(xys[lvl]))
            result['levels'][str(lvl)] = lv
        np.savez_compressed(out_dir / f'{name}.npz', **arrays)
        write_json(res_path, result)
        print(f'[model-error] {name}: {len(clips)} clips, {time.time() - t0:.0f}s')
        for lvl in range(1, L + 1):
            lv = result['levels'][str(lvl)]
            k = ks[lvl][-1]
            print(f'    L{lvl}: tf_mse={lv["tf_mse"]["mean"]:.4f}  mse_k{k}={lv[f"mse_k{k}"]["mean"]:.4f} '
                  f'copy_k{k}={lv[f"copy_mse_k{k}"]["mean"]:.4f}  action_gap_k{k}={lv[f"action_gap_k{k}"]:.4f}')

    # representation probe: latent -> agent xy, fit on clean (held-out episodes), tested on every condition
    ref = args.probe_reference
    if any(key[0] == ref for key in probe_cache) or (out_dir / f'{ref}.json').exists():
        _probe(args, out_dir, probe_cache, L)


def _cluster_se(v, groups):
    """Standard error of the mean with clips clustered by episode."""
    means = np.array([v[groups == g].mean() for g in np.unique(groups)])
    return float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else float('nan')


def _steps_at_level(geo, lvl, n1):
    n = n1
    for l in range(2, lvl + 1):
        stride, window = geo[l]
        n = (n - window) // stride + 1
    return n


def _probe(args, out_dir, cache, L):
    ref = args.probe_reference
    if not any(k[0] == ref for k in cache):
        print(f'[probe] reference condition {ref!r} was not encoded in this call; run model-error with it included')
        return
    probe = {}
    for lvl in range(1, L + 1):
        if (ref, lvl) not in cache:
            continue
        x, y = cache[(ref, lvl)]
        n = len(x)
        cut = int(0.7 * n)  # clips are episode-ordered, so this is an episode-wise split
        w = _ridge_fit(x[:cut], y[:cut])
        lv = {'reference': ref, 'train_frames': cut}
        lv[f'{ref}_heldout'] = dict(zip(('r2', 'rmse_px'), _ridge_r2(w, x[cut:], y[cut:])))
        for (name, l2), (xc, yc) in cache.items():
            if l2 == lvl and name != ref:
                lv[name] = dict(zip(('r2', 'rmse_px'), _ridge_r2(w, xc, yc)))
        probe[str(lvl)] = lv
        print(f'[probe] L{lvl}: ' + ', '.join(f'{k}: R2={v["r2"]:.3f}' for k, v in lv.items() if isinstance(v, dict)))
    write_json(out_dir / 'xy_probe.json', probe)


# --------------------------------------------------------------------------------------
# plan: closed-loop planning, sharded over GPUs
# --------------------------------------------------------------------------------------

_COST_STASH = {}


def _install_planner_recorders():
    """Record each solve's final best cost (per level) next to the repo's own solve_records."""
    import hierarchical_solver as hs
    from stable_worldmodel.solver import gd

    if getattr(hs.LevelCostModel, '_bench_patched', False):
        return
    orig_cost = hs.LevelCostModel.get_cost

    def get_cost(self, info_dict, action_candidates):
        costs = orig_cost(self, info_dict, action_candidates)
        if not torch.is_grad_enabled():  # the solver's final no-grad evaluation
            _COST_STASH['last'] = costs.detach().min(dim=1).values.float().cpu().tolist()
        return costs

    hs.LevelCostModel.get_cost = get_cost
    hs.LevelCostModel._bench_patched = True
    orig_solve = gd.GradientSolver.solve

    def solve(self, *a, **k):
        _COST_STASH.pop('last', None)
        out = orig_solve(self, *a, **k)
        if self.solve_records:
            self.solve_records[-1]['final_cost'] = _COST_STASH.pop('last', None)
            self.solve_records[-1]['n_steps_cap'] = int(self.n_steps)
        return out

    gd.GradientSolver.solve = solve


def _build_cfg(args, cond):
    from omegaconf import OmegaConf
    from planning_eval import load_eval_config, _resolve_existing_path

    cfg = load_eval_config(args.config_name)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.policy = args.policy
    cfg.seed = int(args.seed)
    cfg.eval.num_eval = int(cond.get('num_eval', args.num_eval))
    for k, v in world_overrides(cond).items():
        cfg.world[k] = v
    for lvl, over in cond.get('solver', {}).items():
        for k, v in over.items():
            cfg.solver.solvers[lvl][k] = v
    for item in args.override or []:  # generic key=value overrides (smoke tests), e.g. eval.eval_budget=30
        key, val = item.split('=', 1)
        OmegaConf.update(cfg, key, _parse_value(val), force_add=True)
    if not torch.cuda.is_available():
        for lvl in cfg.solver.solvers:
            cfg.solver.solvers[lvl].device = 'cpu'
    cfg.load_eval_trajs_path = str(_resolve_existing_path(cfg.load_eval_trajs_path))
    return cfg


def _parse_value(val):
    import yaml
    return yaml.safe_load(val)


def cmd_plan(args):
    from omegaconf import OmegaConf
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import stable_worldmodel as swm
    import planning_eval as pe

    register_shift_env()
    _install_planner_recorders()
    conds = load_conditions(args.out)
    torch.set_num_threads(max(1, int(args.threads)))

    for name in args.conditions.split(','):
        cond = conds[name]
        cfg = _build_cfg(args, cond)
        cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
        world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
        world_cfg = OmegaConf.to_container(world_cfg)
        transform = {'pixels': pe.img_transform(cfg), 'goal': pe.img_transform(cfg)}
        image_shape = (int(cfg.eval.img_size),) * 2
        process = pe._build_eval_process_from_policy_normalizer(cfg, model=None)
        payload = torch.load(cfg.load_eval_trajs_path, map_location='cpu', weights_only=False)
        n = int(cfg.eval.num_eval)
        if args.limit is not None:
            n = min(n, int(args.limit))
        assert len(payload['data']) >= n, f'{len(payload["data"])} tasks < num_eval={n}'
        ep_idx = payload.get('episodes_idx')
        ep_idx = list(ep_idx) if ep_idx is not None and len(ep_idx) else list(range(len(payload['data'])))
        callables = OmegaConf.to_container(cfg.eval.get('callables'), resolve=True)
        res_dir = Path(args.out) / 'plan' / name / f'seed{args.seed}'
        (res_dir / 'tasks').mkdir(parents=True, exist_ok=True)
        if args.shard == 0:
            (res_dir / 'eval_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
        my_tasks = [t for t in range(n) if t % args.num_shards == args.shard]
        todo = [t for t in my_tasks if not (res_dir / 'tasks' / f'task_{t:03d}.json').exists()]
        print(f'[plan] {name} seed{args.seed} shard {args.shard}/{args.num_shards}: '
              f'{len(my_tasks) - len(todo)} done, {len(todo)} to go', flush=True)
        if not todo:
            continue
        policy = pe._build_policy(cfg, process, transform, model=None)
        solver = getattr(policy, 'solver', None)
        gens = [(s.torch_gen, s.torch_gen.initial_seed())
                for s in getattr(solver, 'level_solvers', {1: solver}).values() if hasattr(s, 'torch_gen')]
        for t in todo:
            for gen, seed in gens:  # identical to the repo's chunked eval with chunk_size=1
                gen.manual_seed(seed + t)
            chunk = {**payload, 'data': [dict(payload['data'][t])], 'episodes_idx': ep_idx[t:t + 1]}
            world = swm.World(**{**world_cfg, 'num_envs': 1}, image_shape=image_shape)
            trace = {'dist': [], 'xy': []}
            orig_step = world.step

            def step_and_record(orig_step=orig_step, world=world, trace=trace):
                orig_step()
                info = world.infos
                d = info.get('distance_to_target')
                xy = info.get('xy')
                if d is not None:
                    trace['dist'].append(float(np.asarray(d).reshape(-1)[0]))
                if xy is not None:
                    trace['xy'].append(np.asarray(xy, dtype=np.float64).reshape(-1)[-2:].tolist())

            world.step = step_and_record
            try:
                world.set_policy(policy)
                t0 = time.time()
                with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                    m = world.evaluate_from_dataset(
                        None, episodes_idx=None, start_steps=None,
                        goal_offset_steps=cfg.eval.goal_offset_steps, eval_budget=cfg.eval.eval_budget,
                        callables=callables, load_eval_trajs_path=chunk, process=process,
                        start_state_mode=cfg.eval.get('start_state_mode', 'dataset_full'),
                        goal_state_mode=cfg.eval.get('goal_state_mode', 'dataset_full'))
                secs = time.time() - t0
            finally:
                world.close()
            success = bool(np.asarray(m['episode_successes']).reshape(-1)[0])
            steps = float(np.asarray(m['steps_to_success']).reshape(-1)[0])
            dist = trace['dist']
            rec = {
                'task': t, 'condition': name, 'seed': int(args.seed), 'success': success,
                'steps_to_success': None if not math.isfinite(steps) else steps,
                'final_distance': dist[-1] if dist else None,
                'min_distance': min(dist) if dist else None,
                'distance_trace': dist, 'xy_trace': trace['xy'],
                'solve_records': pe.pop_solve_records(solver),
                'seconds': secs, 'bench_version': BENCH_VERSION,
            }
            write_json(res_dir / 'tasks' / f'task_{t:03d}.json', rec)
            print(f'[plan] {name} task {t:02d}: success={success} steps={rec["steps_to_success"]} '
                  f'final_dist={rec["final_distance"]} ({secs:.0f}s)', flush=True)


# --------------------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------------------


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def _plan_summary(tasks, start_dist):
    n = len(tasks)
    succ = np.array([t['success'] for t in tasks], dtype=bool)
    k = int(succ.sum())
    steps = [t['steps_to_success'] for t in tasks if t['steps_to_success'] is not None]
    fin = np.array([t['final_distance'] if t['final_distance'] is not None else np.nan for t in tasks])
    mind = np.array([t['min_distance'] if t['min_distance'] is not None else np.nan for t in tasks])
    out = {'n_tasks': n, 'successes': k, 'success_rate': 100 * k / n if n else float('nan'),
           'wilson95': wilson(k, n),
           'mean_steps_success_only': float(np.mean(steps)) if steps else None,
           'final_distance_mean': float(np.nanmean(fin)) if n else None,
           'final_distance_failures_mean': float(np.nanmean(fin[~succ])) if (~succ).any() else None,
           'min_distance_mean': float(np.nanmean(mind)) if n else None,
           'seconds_per_task': float(np.mean([t['seconds'] for t in tasks])) if n else None}
    if start_dist:
        prog = [1 - (t['final_distance'] / start_dist[t['task']]) for t in tasks
                if t['final_distance'] is not None and start_dist.get(t['task'])]
        out['progress_mean'] = float(np.mean(prog)) if prog else None
    lv = {}
    for t in tasks:
        for r in t['solve_records']:
            d = lv.setdefault(str(r['level']), {'iters': [], 'cap_hits': 0, 'solves': 0, 'final_cost': [], 'wall': []})
            d['iters'].extend(r['n_iters'])
            cap = r.get('n_steps_cap')
            d['cap_hits'] += sum(int(cap is not None and it >= cap) for it in r['n_iters'])
            d['solves'] += len(r['n_iters'])
            if r.get('final_cost'):
                d['final_cost'].extend(r['final_cost'])
            d['wall'].append(r['wall_clock'])
    out['planner'] = {l: {'solves': d['solves'], 'mean_iters': float(np.mean(d['iters'])) if d['iters'] else None,
                          'cap_hit_frac': d['cap_hits'] / max(d['solves'], 1),
                          'final_cost_mean': float(np.mean(d['final_cost'])) if d['final_cost'] else None,
                          'wall_clock_total_s': float(np.sum(d['wall']))}
                      for l, d in sorted(lv.items())}
    return out


def cmd_summarize(args):
    out = Path(args.out)
    conds = load_conditions(out)
    summary = {'bench_version': BENCH_VERSION, 'conditions': conds, 'plan': {}, 'model_error': {}, 'probe': {}}
    ref = args.reference
    plan_tasks = {}
    for name in conds:
        for seed_dir in sorted((out / 'plan' / name).glob('seed*')):
            tasks = [json.loads(p.read_text()) for p in sorted((seed_dir / 'tasks').glob('task_*.json'))]
            if tasks:
                plan_tasks[(name, seed_dir.name)] = {t['task']: t for t in tasks}
    for (name, seed), tasks in plan_tasks.items():
        start = {}
        ref_tasks = plan_tasks.get((ref, seed), {})
        for tid, t in ref_tasks.items():
            if t['distance_trace']:
                start[tid] = t['distance_trace'][0]
        s = _plan_summary(list(tasks.values()), start)
        bases = [b for b in (ref, conds[name].get('paired_with')) if b and b != name]
        for base in dict.fromkeys(bases):
            base_tasks = plan_tasks.get((base, seed), {})
            if not base_tasks:
                continue
            common = sorted(set(tasks) & set(base_tasks))
            b = sum(base_tasks[i]['success'] and not tasks[i]['success'] for i in common)
            c = sum((not base_tasks[i]['success']) and tasks[i]['success'] for i in common)
            fd = [tasks[i]['final_distance'] - base_tasks[i]['final_distance'] for i in common
                  if tasks[i]['final_distance'] is not None and base_tasks[i]['final_distance'] is not None]
            s['paired_vs_' + base] = {
                'n_common': len(common),
                'success_this': 100 * sum(tasks[i]['success'] for i in common) / max(len(common), 1),
                'success_base': 100 * sum(base_tasks[i]['success'] for i in common) / max(len(common), 1),
                'delta_success_pts': 100 * (sum(tasks[i]['success'] for i in common)
                                            - sum(base_tasks[i]['success'] for i in common)) / max(len(common), 1),
                'lost': b, 'gained': c, 'mcnemar_p': mcnemar_exact(b, c),
                'delta_final_distance_mean': float(np.mean(fd)) if fd else None}
        summary['plan'].setdefault(name, {})[seed] = s
    for p in sorted((out / 'model_error').glob('seed*/*.json')):
        if p.name == 'xy_probe.json':
            summary['probe'][p.parent.name] = json.loads(p.read_text())
        else:
            r = json.loads(p.read_text())
            summary['model_error'].setdefault(r['condition'], {})[p.parent.name] = r
    write_json(out / 'summary.json', summary)
    print(f'wrote {out / "summary.json"}')


# --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--out', required=True)
    c.add_argument('--data-dir', required=True)
    c.add_argument('--conditions', required=True)
    c.add_argument('--episodes', type=int, default=100)
    c.add_argument('--num-envs', type=int, default=10)
    c.add_argument('--seed', type=int, default=3072)
    m = sub.add_parser('model-error')
    m.add_argument('--out', required=True)
    m.add_argument('--data-dir', required=True)
    m.add_argument('--conditions', required=True)
    m.add_argument('--policy', required=True)
    m.add_argument('--seed', type=int, default=42)
    m.add_argument('--top-k', type=int, default=4, help='autoregressive horizon at the top level')
    m.add_argument('--max-clips', type=int, default=1024)
    m.add_argument('--batch-size', type=int, default=16)
    m.add_argument('--num-workers', type=int, default=2)
    m.add_argument('--probe-reference', default='clean')
    m.add_argument('--force', action='store_true')
    p = sub.add_parser('plan')
    p.add_argument('--out', required=True)
    p.add_argument('--conditions', required=True)
    p.add_argument('--policy', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--config-name', default='fourroom_l3')
    p.add_argument('--num-eval', type=int, default=50)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--limit', type=int, default=None, help='plan only the first N tasks of each condition')
    p.add_argument('--override', action='append', help='extra eval-config key=value (repeatable)')
    s = sub.add_parser('summarize')
    s.add_argument('--out', required=True)
    s.add_argument('--reference', default='clean')
    args = ap.parse_args()
    {'collect': cmd_collect, 'model-error': cmd_model_error, 'plan': cmd_plan, 'summarize': cmd_summarize}[args.cmd](args)


if __name__ == '__main__':
    main()
