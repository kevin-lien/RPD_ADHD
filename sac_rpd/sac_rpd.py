"""
SAC and SAC+RPD in one script, switched by --vla_lambda.

  --vla_lambda 0             -> pure SAC baseline (no VLA is started)
  --vla_lambda 1             -> SAC + RPD: actor loss gets an extra term  lambda * ||mu_theta(s) - a_VLA(s)||^2
  --vla_lambda 1 --fake_vla  -> fake VLA (always returns zero actions) to test the pipeline without an Octo env

Based on ManiSkill v3.0.0b15 examples/baselines/sac/sac_rgbd.py. Environment settings follow RPD
(src/rpd/ppo_rgb_rpd.py): obs_mode="rgb", control_mode="pd_ee_delta_pose", normalized_dense reward,
HumanCameraWrapper (256x256), include_state=False.

Differences from RPD (PPO version):
  - RPD queries the VLA during the network update. SAC re-samples old data many times, so querying on every
    update would be too slow. Here the VLA is queried once at data-collection time and a_VLA is stored in the
    replay buffer (one query per visited state).
  - Images in the replay buffer are downsized to --student_image_size (default 128) to save memory; the VLA
    still receives the original 256x256 images.
"""
import os
import random
import signal
import time
from collections import defaultdict
from dataclasses import dataclass
from time import sleep
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs  # noqa: F401  (registers the envs)
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from networks import (Actor, ReplayBuffer, SoftQNetwork, clamp_to_action_bounds, distill_loss,
                      to_student_obs)

# Task instructions, same as RPD
INSTRUCTIONS = {
    "LiftPegUpright-v1": "lift the peg upright",
    "PegInsertionSide-v1": "insert the peg from the side",  # not in VLA training
    "PickCube-v1": "pick up the cube",
    "PlugCharger-v1": "plug the charger in",  # not in VLA training
    "PokeCube-v1": "push the cube by using the blue tool",  # not in VLA training
    "PullCube-v1": "pull the cube towards the robot base",
    "PullCubeTool-v1": "pull the cube by using the red tool",  # not in VLA training
    "PushCube-v1": "push the cube away from the robot base",
    "PushT-v1": "align the T shape",
    "RollBall-v1": "push the ball",
    "StackCube-v1": "stack the red cube on the green cube",
}

# VLA configs, same as RPD
VLA_CONFIGS = {
    "octo": ("octodist", {"checkpoint_path": "hf://Juelg/octo-base-1.5-finetuned-maniskill",
                          "checkpoint_step": None, "horizon": 1, "unnorm_key": []}),
    "octo-base": ("octodist", {"checkpoint_path": "hf://rail-berkeley/octo-base-1.5",
                               "checkpoint_step": None, "horizon": 1, "unnorm_key": []}),
}


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    """log to wandb (run `wandb login` first)"""
    wandb_project_name: str = "RPD_ADHD"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    capture_video: bool = True
    save_model: bool = True
    log_freq: int = 1_000

    # Environment (follows RPD)
    env_id: str = "PickCube-v1"
    num_envs: int = 16
    num_eval_envs: int = 16
    num_eval_steps: int = 50
    eval_freq: int = 10_000
    """evaluate every N env steps"""
    eval_reconfiguration_freq: Optional[int] = 1
    reconfiguration_freq: Optional[int] = None
    control_mode: str = "pd_ee_delta_pose"
    dense: bool = True
    include_state: bool = False
    """RPD's student uses images only, no state"""
    use_human_camera: bool = True
    """RPD uses HumanCameraWrapper (256x256); the VLA needs this camera view"""
    student_image_size: Optional[int] = 128
    """image size for the student / replay buffer; None = original size (256 uses a lot of memory)"""

    # SAC (defaults from ManiSkill sac_rgbd.py)
    total_timesteps: int = 1_000_000
    buffer_size: int = 100_000
    """about 10GB with 128x128 images; reduce it or use --buffer_device cpu on smaller GPUs"""
    buffer_device: str = "cuda"
    gamma: float = 0.8
    tau: float = 0.01
    batch_size: int = 512
    learning_starts: int = 4_000
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 1
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    training_freq: int = 64
    utd: float = 0.25
    bootstrap_at_done: str = "always"

    # RPD distillation
    vla_lambda: float = 0.0
    """0 = pure SAC; >0 = weight of the SAC+RPD distillation loss"""
    vla_loss_type: str = "mse"
    """mse or l1 (same as RPD's vlad_loss_type)"""
    vla_type: str = "octo"
    vla_conda_path: str = os.path.expanduser("~/miniconda3/envs/rpd_octo/bin/python")
    fake_vla: bool = False
    """use a fake VLA to test the pipeline (does not start Octo)"""


class FakeVLA:
    """Fake VLA: always returns zero actions. Used to test the pipeline without an Octo env."""

    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim  # train() overwrites this with the env's real action dim

    def query(self, rgb: np.ndarray) -> np.ndarray:
        return np.zeros((rgb.shape[0], self.action_dim), dtype=np.float32)


class RemoteVLA:
    """Thin wrapper around agents.RemoteAgent, used the same way as RPD: send the rgb_side image, read info['means']."""

    def __init__(self, port: int, server_type: str, instruction: str):
        from agents.client import RemoteAgent
        from agents.policies import Obs
        self._Obs = Obs
        self.model = RemoteAgent("localhost", port, server_type, on_same_machine=True)
        while not self.model.is_initialized():
            print("Waiting for VLA to initialize ...")
            sleep(1)
        self.model.reset(obs=Obs(), instruction=instruction)

    def query(self, rgb: np.ndarray) -> np.ndarray:
        act = self.model.act(self._Obs(cameras={"rgb_side": rgb}))
        return np.asarray(act.info["means"], dtype=np.float32)


def make_envs(args: Args, run_name: str):
    env_kwargs = dict(
        obs_mode="rgb",
        sim_backend="gpu",
        control_mode=args.control_mode,
        render_mode="rgb_array",
        reward_mode="normalized_dense" if args.dense else "sparse",
        human_render_camera_configs={"width": 256, "height": 256},
    )
    envs = gym.make(args.env_id, num_envs=args.num_envs, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs,
                         reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    if args.use_human_camera:
        from agents.wrappers import HumanCameraWrapper
        envs = HumanCameraWrapper(envs)
        eval_envs = HumanCameraWrapper(eval_envs)
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_envs = RecordEpisode(eval_envs, output_dir=f"runs/{run_name}/videos", save_trajectory=False,
                                  max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=True, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=True, record_metrics=True)
    return envs, eval_envs


def train(args: Args, vla):
    use_vla = args.vla_lambda > 0
    run_name = args.exp_name
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    grad_steps_per_iteration = max(1, int(args.training_freq * args.utd))
    steps_per_env = max(1, args.training_freq // args.num_envs)

    envs, eval_envs = make_envs(args, run_name)
    act_low = envs.single_action_space.low
    act_high = envs.single_action_space.high
    action_dim = int(np.prod(envs.single_action_space.shape))
    act_low_t = torch.as_tensor(act_low, dtype=torch.float32, device=device)
    act_high_t = torch.as_tensor(act_high, dtype=torch.float32, device=device)
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    print(f"action_dim={action_dim}, max_episode_steps={max_episode_steps}, use_vla={use_vla}")

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args),
                   name=run_name, group=args.wandb_group, sync_tensorboard=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                    "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    raw_obs, _ = envs.reset(seed=args.seed)
    obs = to_student_obs(raw_obs, args.student_image_size)
    print("student obs:", {k: tuple(v.shape) for k, v in obs.items()},
          "| raw rgb:", tuple(raw_obs["rgb"].shape))

    actor = Actor(obs, act_low, act_high).to(device)
    state_dim = actor.state_dim
    qf1 = SoftQNetwork(actor.encoder, state_dim, action_dim).to(device)
    qf2 = SoftQNetwork(actor.encoder, state_dim, action_dim).to(device)
    qf1_target = SoftQNetwork(actor.encoder, state_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(actor.encoder, state_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.mlp.parameters()) + list(qf2.mlp.parameters()) +
                             list(actor.encoder.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    if args.autotune:
        target_entropy = -float(action_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        log_alpha = None
        alpha = args.alpha

    rb = ReplayBuffer({k: v[:1] for k, v in obs.items()}, action_dim, args.num_envs, args.buffer_size,
                      storage_device=args.buffer_device, sample_device=device)

    if isinstance(vla, FakeVLA):
        vla.action_dim = action_dim

    def query_vla(raw):
        a = vla.query(raw["rgb"].cpu().numpy())
        a = torch.as_tensor(a, dtype=torch.float32, device=device)
        return clamp_to_action_bounds(a, act_low_t, act_high_t)

    global_step, global_update, vla_queries = 0, 0, 0
    learning_has_started = False
    last_eval = -args.eval_freq
    times = defaultdict(float)
    l_distill = torch.tensor(0.0)
    pbar = tqdm.tqdm(total=args.total_timesteps)

    while global_step < args.total_timesteps:
        # ---------------- evaluation ----------------
        if args.eval_freq > 0 and global_step - last_eval >= args.eval_freq:
            last_eval = global_step
            actor.eval()
            raw_eval, _ = eval_envs.reset()
            metrics = defaultdict(list)
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    a = actor.get_eval_action(to_student_obs(raw_eval, args.student_image_size))
                    raw_eval, _, _, _, infos = eval_envs.step(a)
                    if "final_info" in infos:
                        for k, v in infos["final_info"]["episode"].items():
                            metrics[k].append(v)
            for k, v in metrics.items():
                writer.add_scalar(f"eval/{k}", torch.stack(v).float().mean().item(), global_step)
            if "success_once" in metrics:
                sr = torch.stack(metrics["success_once"]).float().mean().item()
                pbar.set_description(f"success_once={sr:.2f}")
            actor.train()
            if args.save_model:
                torch.save({"actor": actor.state_dict(), "qf1": qf1_target.state_dict(),
                            "qf2": qf2_target.state_dict(), "log_alpha": log_alpha},
                           f"runs/{run_name}/ckpt_{global_step}.pt")

        # ---------------- data collection ----------------
        t0 = time.perf_counter()
        for _ in range(steps_per_env):
            global_step += args.num_envs
            vla_action = None
            if use_vla:
                vla_action = query_vla(raw_obs)
                vla_queries += args.num_envs
            if not learning_has_started:
                actions = torch.as_tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                with torch.no_grad():
                    actions, _, _, _ = actor.get_action(obs)

            raw_next, rewards, terminations, truncations, infos = envs.step(actions)
            real_next = {k: v.clone() for k, v in raw_next.items()}
            if args.bootstrap_at_done == "always":
                need_final_obs = truncations | terminations
                stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool)
            else:
                need_final_obs = truncations & (~terminations)
                stop_bootstrap = terminations
            if "final_info" in infos:
                done_mask = infos["_final_info"]
                for k in real_next:
                    real_next[k][need_final_obs] = infos["final_observation"][k][need_final_obs].clone()
                for k, v in infos["final_info"]["episode"].items():
                    writer.add_scalar(f"train/{k}", v[done_mask].float().mean().item(), global_step)

            rb.add(obs, to_student_obs(real_next, args.student_image_size), actions, rewards, stop_bootstrap,
                   vla_action=vla_action)
            raw_obs = raw_next
            obs = to_student_obs(raw_obs, args.student_image_size)
        times["rollout"] += time.perf_counter() - t0
        pbar.update(args.num_envs * steps_per_env)

        if global_step < args.learning_starts:
            continue

        # ---------------- network update ----------------
        t0 = time.perf_counter()
        learning_has_started = True
        for _ in range(grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                next_a, next_logp, _, next_vf = actor.get_action(data.next_obs)
                q1_t = qf1_target(data.next_obs, next_a, next_vf)
                q2_t = qf2_target(data.next_obs, next_a, next_vf)
                min_q_t = torch.min(q1_t, q2_t) - alpha * next_logp
                next_q = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_q_t.view(-1)
            vf = actor.encoder(data.obs)
            q1_a = qf1(data.obs, data.actions, vf).view(-1)
            q2_a = qf2(data.obs, data.actions, vf).view(-1)
            qf_loss = F.mse_loss(q1_a, next_q) + F.mse_loss(q2_a, next_q)
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_update % args.policy_frequency == 0:
                pi, log_pi, det_a, vf = actor.get_action(data.obs)
                q1_pi = qf1(data.obs, pi, vf, detach_encoder=True)
                q2_pi = qf2(data.obs, pi, vf, detach_encoder=True)
                sac_loss = (alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()
                actor_loss = sac_loss
                if use_vla:
                    # RPD: pull mu_theta(s) towards the VLA action at the same state
                    l_distill = distill_loss(det_a, data.vla_actions, data.has_vla, args.vla_loss_type)
                    actor_loss = sac_loss + args.vla_lambda * l_distill
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _, _ = actor.get_action(data.obs)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            if global_update % args.target_network_frequency == 0:
                for net, tgt in ((qf1, qf1_target), (qf2, qf2_target)):
                    for p, tp in zip(net.parameters(), tgt.parameters()):
                        tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
        times["update"] += time.perf_counter() - t0

        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/qf1_values", q1_a.mean().item(), global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/sac_actor_loss", sac_loss.item(), global_step)
            writer.add_scalar("losses/distill_loss", float(l_distill), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            writer.add_scalar("charts/vla_queries", vla_queries, global_step)
            writer.add_scalar("charts/env_steps", global_step, global_step)
            for k, v in times.items():
                writer.add_scalar(f"time/total_{k}", v, global_step)

    if args.save_model:
        torch.save({"actor": actor.state_dict(), "qf1": qf1_target.state_dict(),
                    "qf2": qf2_target.state_dict(), "log_alpha": log_alpha}, f"runs/{run_name}/final_ckpt.pt")
    writer.close()
    envs.close()
    eval_envs.close()
    print(f"done. env_steps={global_step}, vla_queries={vla_queries}")


def main():
    args = tyro.cli(Args)
    algo = "SAC" if args.vla_lambda <= 0 else ("SACRPD_fake" if args.fake_vla else "SACRPD")
    if args.exp_name is None:
        args.exp_name = f"{algo}_{args.env_id}_s{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    if args.wandb_group is None:
        args.wandb_group = algo
    os.makedirs(f"runs/{args.exp_name}", exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.vla_lambda <= 0:
        train(args, vla=None)
    elif args.fake_vla:
        train(args, vla=FakeVLA())
    else:
        from agents.evaluator_envs import start_server
        server_type, kwargs = VLA_CONFIGS[args.vla_type]
        port = random.randint(10000, 40000)
        with start_server(server_type, kwargs, port, "localhost", python_path=args.vla_conda_path) as p:
            sleep(5)
            try:
                train(args, vla=RemoteVLA(port, server_type, INSTRUCTIONS[args.env_id]))
            finally:
                p.send_signal(signal.SIGINT)


if __name__ == "__main__":
    main()
