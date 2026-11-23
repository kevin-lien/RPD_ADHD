"""
Core components for SAC / SAC+RPD. Depends only on torch + numpy, so it can be unit-tested on a laptop
without a GPU or a simulator.

Adapted from ManiSkill v3.0.0b15 examples/baselines/sac/sac_rgbd.py. Main changes:
  1. Supports rgb-only observations without state (RPD's student setting: include_state=False)
  2. The CNN flatten size is computed automatically, so any image size works (the original hard-codes 64 or 128)
  3. ReplayBuffer additionally stores VLA actions (vla_actions) and whether a VLA label exists (has_vla)
  4. Distillation loss distill_loss(), matching RPD's vlad_loss_type = mse / l1
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -5


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------
def resize_rgb(rgb: torch.Tensor, size: Optional[int]) -> torch.Tensor:
    """Resize a (B, H, W, C) uint8 image to (B, size, size, C), still uint8 (saves replay buffer memory).
    Returns the input unchanged if size is None or already matches."""
    if size is None or (rgb.shape[1] == size and rgb.shape[2] == size):
        return rgb
    x = rgb.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(size, size), mode="area")
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


def to_student_obs(obs: Dict[str, torch.Tensor], size: Optional[int]) -> Dict[str, torch.Tensor]:
    """Env observation -> observation used by the student (and the replay buffer): only rgb is resized."""
    out = dict(obs)
    if "rgb" in out:
        out["rgb"] = resize_rgb(out["rgb"], size)
    return out


# ------------------------------------------------------------
# Networks
# ------------------------------------------------------------
class PlainConv(nn.Module):
    """The ManiSkill SAC CNN, with the flatten size computed automatically."""

    def __init__(self, in_channels=3, out_dim=256, image_size=(128, 128)):
        super().__init__()
        self.out_dim = out_dim
        first_pool = 4 if image_size[0] >= 128 else 2
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(first_pool, first_pool),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 1, padding=0), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, in_channels, *image_size)).flatten(1).shape[1]
        self.fc = make_mlp(n_flat, [out_dim], last_act=True)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, image):
        return self.fc(self.cnn(image).flatten(1))


class EncoderObsWrapper(nn.Module):
    def __init__(self, encoder: PlainConv):
        super().__init__()
        self.encoder = encoder

    def forward(self, obs):
        img = obs["rgb"].float() / 255.0  # (B, H, W, C)
        return self.encoder(img.permute(0, 3, 1, 2))


def _state_dim(sample_obs) -> int:
    return sample_obs["state"].shape[-1] if "state" in sample_obs else 0


class Actor(nn.Module):
    def __init__(self, sample_obs, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        action_dim = int(np.prod(action_low.shape))
        self.state_dim = _state_dim(sample_obs)
        rgb = sample_obs["rgb"]
        self.encoder = EncoderObsWrapper(
            PlainConv(in_channels=rgb.shape[-1], out_dim=256, image_size=tuple(rgb.shape[1:3]))
        )
        self.mlp = make_mlp(256 + self.state_dim, [512, 256], last_act=True)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32))

    def get_feature(self, obs, detach_encoder=False):
        visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = visual_feature if self.state_dim == 0 else torch.cat([visual_feature, obs["state"]], dim=1)
        return self.mlp(x), visual_feature

    def forward(self, obs, detach_encoder=False):
        x, visual_feature = self.get_feature(obs, detach_encoder)
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std, visual_feature

    def get_eval_action(self, obs):
        mean, _, _ = self(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_action(self, obs, detach_encoder=False):
        """Returns (sampled action, log_prob, deterministic action mu_theta(s), visual_feature).
        mu_theta(s) = tanh(mean) * scale + bias lives in the env action space; the distillation loss
        compares it with the VLA action."""
        mean, log_std, visual_feature = self(obs, detach_encoder)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        det_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, det_action, visual_feature


class SoftQNetwork(nn.Module):
    """Q(s, a). Shares the CNN encoder with the actor (as in ManiSkill SAC)."""

    def __init__(self, encoder: EncoderObsWrapper, state_dim: int, action_dim: int):
        super().__init__()
        self.encoder = encoder
        self.state_dim = state_dim
        self.mlp = make_mlp(encoder.encoder.out_dim + state_dim + action_dim, [512, 256, 1], last_act=False)

    def forward(self, obs, action, visual_feature=None, detach_encoder=False):
        if visual_feature is None:
            visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        parts = [visual_feature] + ([obs["state"]] if self.state_dim > 0 else []) + [action]
        return self.mlp(torch.cat(parts, dim=1))


# ------------------------------------------------------------
# Replay buffer (also stores VLA actions)
# ------------------------------------------------------------
@dataclass
class ReplayBufferSample:
    obs: Dict[str, torch.Tensor]
    next_obs: Dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    vla_actions: torch.Tensor  # (B, action_dim); zeros where there is no VLA label
    has_vla: torch.Tensor  # (B,) bool


class ReplayBuffer:
    """Shape (per_env_buffer_size, num_envs, ...) as in ManiSkill SAC, but the per-key shapes come from sample_obs."""

    def __init__(self, sample_obs: Dict[str, torch.Tensor], action_dim: int, num_envs: int,
                 buffer_size: int, storage_device, sample_device):
        self.num_envs = num_envs
        self.per_env_buffer_size = max(1, buffer_size // num_envs)
        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)
        self.pos = 0
        self.full = False
        shape = (self.per_env_buffer_size, num_envs)
        self.obs = {k: torch.zeros(shape + tuple(v.shape[1:]), dtype=v.dtype, device=self.storage_device)
                    for k, v in sample_obs.items()}
        self.next_obs = {k: torch.zeros_like(v) for k, v in self.obs.items()}
        self.actions = torch.zeros(shape + (action_dim,), device=self.storage_device)
        self.rewards = torch.zeros(shape, device=self.storage_device)
        self.dones = torch.zeros(shape, device=self.storage_device)
        self.vla_actions = torch.zeros(shape + (action_dim,), device=self.storage_device)
        self.has_vla = torch.zeros(shape, dtype=torch.bool, device=self.storage_device)

    def __len__(self):
        return (self.per_env_buffer_size if self.full else self.pos) * self.num_envs

    def add(self, obs, next_obs, action, reward, done, vla_action: Optional[torch.Tensor] = None):
        d = self.storage_device
        for k in self.obs:
            self.obs[k][self.pos] = obs[k].to(d)
            self.next_obs[k][self.pos] = next_obs[k].to(d)
        self.actions[self.pos] = action.to(d)
        self.rewards[self.pos] = reward.to(d).float()
        self.dones[self.pos] = done.to(d).float()
        if vla_action is not None:
            self.vla_actions[self.pos] = vla_action.to(d).float()
            self.has_vla[self.pos] = True
        else:
            self.vla_actions[self.pos] = 0.0
            self.has_vla[self.pos] = False
        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSample:
        high = self.per_env_buffer_size if self.full else self.pos
        assert high > 0, "replay buffer is empty"
        bi = torch.randint(0, high, (batch_size,))
        ei = torch.randint(0, self.num_envs, (batch_size,))
        sd = self.sample_device
        return ReplayBufferSample(
            obs={k: v[bi, ei].to(sd) for k, v in self.obs.items()},
            next_obs={k: v[bi, ei].to(sd) for k, v in self.next_obs.items()},
            actions=self.actions[bi, ei].to(sd),
            rewards=self.rewards[bi, ei].to(sd),
            dones=self.dones[bi, ei].to(sd),
            vla_actions=self.vla_actions[bi, ei].to(sd),
            has_vla=self.has_vla[bi, ei].to(sd),
        )


# ------------------------------------------------------------
# Distillation loss (RPD's l_ppd, num_vla_samples=1 version)
# ------------------------------------------------------------
def distill_loss(student_action: torch.Tensor, vla_action: torch.Tensor,
                 mask: Optional[torch.Tensor] = None, loss_type: str = "mse") -> torch.Tensor:
    """student_action: mu_theta(s) (B, A); vla_action: VLA action (B, A); mask: (B,) which samples have a VLA label.
    Returns the mean loss over labelled samples; returns 0 (still differentiable) if there are none."""
    if loss_type == "mse":
        per = ((student_action - vla_action) ** 2).mean(dim=1)
    elif loss_type == "l1":
        per = (student_action - vla_action).abs().mean(dim=1)
    else:
        raise ValueError(f"unknown loss_type {loss_type}")
    if mask is None:
        return per.mean()
    m = mask.float()
    n = m.sum()
    if n == 0:
        return student_action.sum() * 0.0
    return (per * m).sum() / n


def clamp_to_action_bounds(a: torch.Tensor, low: torch.Tensor, high: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """VLA actions may fall outside the env action bounds, while the student's tanh output never does, so clamp them into the bounds before using them as targets."""
    span = (high - low) * eps
    return torch.max(torch.min(a, high - span), low + span)
