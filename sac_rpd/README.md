# sac_rpd: pure SAC and SAC + RPD

One script, switched by `--vla_lambda`:

| Mode | Key flags | Needs the Octo env? |
|---|---|---|
| Pure SAC baseline | `--vla_lambda 0` (default) | No |
| SAC + RPD with a fake VLA (pipeline test) | `--vla_lambda 1 --fake_vla` | No |
| SAC + RPD with the real Octo teacher | `--vla_lambda 1` | Yes (`rpd_octo`) |

## Files
- `train_sac.py`: **launcher** (same idea as RPD's `train.py`: edit the settings at the top, then run it)
- `networks.py`: actor, Q-networks, replay buffer (also stores VLA actions), distillation loss. Pure torch, testable on a laptop
- `sac_rpd.py`: training script (needs a GPU + ManiSkill)
- `test_networks.py`: unit tests for `networks.py` (CPU)

## Easiest way: `train_sac.py`
Open `train_sac.py` and edit the settings at the top:
- `mode = "sac"`: pure SAC / `"sac_rpd_fake"`: fake VLA to test the pipeline / `"sac_rpd"`: real Octo teacher
- `QUICK_TEST = True`: 50k-step quick test; `False`: 1M-step full run

Then run from the repo root:
```bash
python sac_rpd/train_sac.py
```
Results go to `sac_rpd/runs/`. You can also call `sac_rpd.py` directly with flags, as shown below.

## 1. Laptop (CPU): run the unit tests
```bash
conda activate rpd
cd ~/RPD_ADHD/sac_rpd
python test_networks.py      # expect: ALL TESTS PASSED
```

## 2. GPU machine: run in this order, move on only when each step works
Run inside `~/RPD_ADHD/sac_rpd` (`runs/` is created here):

```bash
# (a) pure SAC, small run to check that everything works
python sac_rpd.py --total_timesteps 50000 --eval_freq 10000

# (b) pure SAC, full run
python sac_rpd.py --total_timesteps 1000000

# (c) SAC + RPD with a fake VLA: checks VLA querying, the buffer and the distillation loss (success rate does not matter)
python sac_rpd.py --vla_lambda 1 --fake_vla --total_timesteps 50000 --eval_freq 10000

# (d) SAC + RPD with the real Octo teacher (run setup_octo.sh first)
python sac_rpd.py --vla_lambda 1 --vla_conda_path ~/miniconda3/envs/rpd_octo/bin/python --total_timesteps 1000000
```

View results:
```bash
tensorboard --logdir runs
```
Key curves: `eval/success_once`, `losses/distill_loss`, `charts/vla_queries`.

## Same as RPD
- Environment: `obs_mode="rgb"`, `pd_ee_delta_pose`, `normalized_dense` reward, `HumanCameraWrapper` (256x256), no state
- Teacher: `--vla_type octo` = `Juelg/octo-base-1.5-finetuned-maniskill`, queried the same way (`rgb_side` -> `info["means"]`)
- Distillation loss: `mse(mu_theta(s), a_VLA)`, matching RPD's `vlad_loss_type=mse`, `num_vla_samples=1`

## Different from RPD (mention in the report)
1. **When the VLA is queried**: RPD (PPO) queries during the update. Here the VLA is queried once at data-collection time and the action is stored in the replay buffer, because SAC re-samples old data many times and querying on every update would be too slow. In both cases the queried states are states visited by the student
2. **Student image size**: the replay buffer stores 128x128 images (`--student_image_size`) to save GPU memory; the VLA still sees the original 256x256 images
3. **Episode resets**: SAC training envs do not reset early on success (every episode runs the full 50 steps, as in the official ManiSkill SAC); RPD's PPO resets as soon as an episode ends
4. **Scale of lambda**: SAC's actor loss has a different scale from PPO's loss, so `--vla_lambda 1` is not necessarily best; sweep a few values later

## Memory
- Default replay buffer: 100k transitions of 128x128 images on the GPU, about 100k x 2 (obs + next_obs) x 49KB = 10GB
- On smaller GPUs, reduce it (`--buffer_size 50000`) or keep it in CPU RAM (`--buffer_device cpu`, a bit slower)
