"""
Launcher for sac_rpd.py (same idea as RPD's train.py): edit the settings below, then run

    python sac_rpd/train_sac.py        (from the repo root)
    python train_sac.py                (from inside sac_rpd/)

Results go to sac_rpd/runs/<exp_name>/ (view with: tensorboard --logdir sac_rpd/runs).
"""
import datetime
import os
import shlex
import subprocess
import sys

# ============================================================
# Settings
# ============================================================
mode = "sac"  # "sac" = pure SAC baseline | "sac_rpd_fake" = SAC+RPD with a fake VLA | "sac_rpd" = SAC+RPD with Octo
env = "PickCube-v1"
seed = 0
vla_lambda = 1.0  # distillation weight, only used in the sac_rpd modes
vla_type = "octo"  # "octo" (fine-tuned on ManiSkill) or "octo-base" (original)
vla_conda_path = os.path.expanduser("~/miniconda3/envs/rpd_octo/bin/python")

# Quick test first to check the pipeline, then set to False for the full run
QUICK_TEST = True

# wandb: off by default, results are always written to tensorboard
use_wandb = False
wandb_entity = None

# Replay buffer: ~10GB on the GPU for 100k transitions of 128x128 images; lower it or use "cpu" on small GPUs
buffer_size = 100_000
buffer_device = "cuda"
# ============================================================

assert mode in ("sac", "sac_rpd_fake", "sac_rpd"), f"unknown mode {mode}"

if QUICK_TEST:
    num_envs, num_eval_envs, total_timesteps, eval_freq = 16, 16, 50_000, 10_000
else:
    num_envs, num_eval_envs, total_timesteps, eval_freq = 16, 16, 1_000_000, 25_000

algo = {"sac": "SAC", "sac_rpd_fake": "SACRPD_fake", "sac_rpd": "SACRPD"}[mode]
exp_name = f"{algo}_{env}_s{seed}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

args = [
    sys.executable,
    "sac_rpd.py",
    f"--exp_name={exp_name}",
    f"--env_id={env}",
    f"--seed={seed}",
    f"--num_envs={num_envs}",
    f"--num_eval_envs={num_eval_envs}",
    f"--total_timesteps={total_timesteps}",
    f"--eval_freq={eval_freq}",
    f"--buffer_size={buffer_size}",
    f"--buffer_device={buffer_device}",
    f"--wandb_group={algo}",
]
if mode == "sac":
    args.append("--vla_lambda=0")
else:
    args.append(f"--vla_lambda={vla_lambda}")
    args.append(f"--vla_type={vla_type}")
    if mode == "sac_rpd_fake":
        args.append("--fake_vla")
    else:
        args.append(f"--vla_conda_path={vla_conda_path}")
if use_wandb:
    args.append("--track")
    if wandb_entity is not None:
        args.append(f"--wandb_entity={wandb_entity}")

print("Starting training with command:")
print(shlex.join(args))

# run inside sac_rpd/ so `from networks import ...` works and runs/ is created there
here = os.path.dirname(os.path.abspath(__file__))
sys.exit(subprocess.call(args, cwd=here))
