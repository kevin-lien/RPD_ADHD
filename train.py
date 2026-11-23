import datetime
import glob
import os
import re
import shlex

# ============================================================
# 要改的設定都在這一區
# ============================================================
use_rpd = False  # False = 純 PPO baseline（不需要 VLA）；True = RPD（需要 Octo 環境）
# env options are "LiftPegUpright-v1", "PegInsertionSide-v1", "PickCube-v1", "PlugCharger-v1", "PokeCube-v1", "PullCube-v1", "PullCubeTool-v1", "PushCube-v1", "RollBall-v1", "StackCube-v1"
env = "PickCube-v1"
loss_type = "mse"  # other options: "l1" (l1 loss), "bc" (behavior cloning loss)
vla_type = "octo"  # 我們用 Octo；other options: "octo-base", "openvla", "openvla-base"
seed = 0  # this should be varied for mass experiments
checkpoint_fmt = None  # PPO checkpoint path string with {seed} and {task} placeholders, e.g. "checkpoints/{task}/ckpt_{seed}.pt"
sparse = False
note = "first test run"

# 小規模測試：先確認整個流程跑得通，再改成 False 跑論文規模
QUICK_TEST = True

# wandb：還沒設定帳號前先關掉，只用 tensorboard（結果在 runs/ 資料夾）
use_wandb = False
wandb_entity = None  # 有 wandb 帳號後填上，例如 "rpd-adhd"

# Octo 那個 conda 環境的 python 路徑，用 `conda env list` 查
vla_conda_path = os.path.expanduser("~/miniconda3/envs/rpd_octo/bin/python")
# ============================================================


if checkpoint_fmt is not None:
    checkpoint_pattern = checkpoint_fmt.format(seed=seed, task=env)
    files = glob.glob(checkpoint_pattern)
    ckpt_files = []
    for f in files:
        base = os.path.basename(f)
        match = re.match(r"ckpt_(\d+)\.pt$", base)
        if match:
            num = int(match.group(1))
            ckpt_files.append((num, f))

    if ckpt_files:
        max_ckpt = max(ckpt_files, key=lambda x: x[0])
        checkpoint = max_ckpt[1]
    else:
        raise ValueError("No checkpoint found")
    print(checkpoint)
else:
    checkpoint = None

if QUICK_TEST:
    num_envs, num_eval_envs, total_timesteps, num_minibatches, eval_freq = 16, 16, 200_000, 8, 5
else:  # 論文原始設定
    num_envs, num_eval_envs, total_timesteps, num_minibatches, eval_freq = 128, 500, 30_000_000, 48, 10

algo = (
    f"PPO_{loss_type}_{'sparse' if sparse else 'dense'}"
    if not use_rpd
    else f"RPD_{loss_type}_{'sparse' if sparse else 'dense'}"
)
uid = f"{algo}_{env}_{seed}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
args = [
    "python",
    f"src/rpd/ppo_rgb{'' if not use_rpd else '_rpd'}.py",
    f"--exp_name={uid}",
    f"--env_id={env}",
    f"--num_envs={num_envs}",
    "--update_epochs=8",
    f"--num_minibatches={num_minibatches}",
    f"--total_timesteps={total_timesteps}",
    f"--eval_freq={eval_freq}",
    "--num-steps=50",
    "--control-mode=pd_ee_delta_pose",
    f"--wandb-group={algo}",
    f"--notes={note}",
    f"--seed={seed}",
    f"--num_eval_envs={num_eval_envs}",
    f"--gamma={0.99 if sparse else 0.8}",
    "--dense" if not sparse else "--no-dense",
]
if use_wandb:
    args.append("--track")
    if wandb_entity is not None:
        args.append(f"--wandb-entity={wandb_entity}")
if use_rpd:
    args.append("--ppd-lambda=1.0")
    args.append("--num-vla-samples=1")  # increasing this will lead to action sampling from the vla and take much more time
    args.append(f"--vlad_loss_type={loss_type}")
    args.append(f"--vla_type={vla_type}")
    args.append(f"--vla_conda_path={vla_conda_path}")
    # args.append("--max-ppd-anneal-steps=6_000_000")

if checkpoint is not None:
    args.append(f"--checkpoint={checkpoint}")

# run the command
cmd = shlex.join(args)
print("Starting training with command:")
print(cmd)

# if you run this on a cluster, just replace the line with a slurm command e.g. by
# using simple_slurm
os.system(cmd)
