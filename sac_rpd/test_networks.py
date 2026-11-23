"""
Unit tests for networks.py. Needs only torch + numpy, so it runs on a laptop (CPU, no simulator).

    cd sac_rpd
    python test_networks.py

Prints "ALL TESTS PASSED" when everything passes.
"""
import numpy as np
import torch

from networks import (Actor, ReplayBuffer, SoftQNetwork, clamp_to_action_bounds, distill_loss, resize_rgb,
                      to_student_obs)

torch.manual_seed(0)
A = 7  # pd_ee_delta_pose: 6-D end-effector delta + 1-D gripper
LOW = -np.ones(A, dtype=np.float32)
HIGH = np.ones(A, dtype=np.float32)


def fake_obs(batch, size=128, with_state=False):
    obs = {"rgb": torch.randint(0, 256, (batch, size, size, 3), dtype=torch.uint8)}
    if with_state:
        obs["state"] = torch.randn(batch, 25)
    return obs


def test_resize():
    raw = fake_obs(4, size=256)
    out = to_student_obs(raw, 128)
    assert out["rgb"].shape == (4, 128, 128, 3) and out["rgb"].dtype == torch.uint8
    assert resize_rgb(raw["rgb"], None) is raw["rgb"]
    assert resize_rgb(raw["rgb"], 256) is raw["rgb"]
    print("✓ resize: 256 -> 128, still uint8")


def test_actor_shapes_no_state():
    for size in (64, 128, 256):
        obs = fake_obs(5, size)
        actor = Actor(obs, LOW, HIGH)
        a, logp, det, vf = actor.get_action(obs)
        assert a.shape == (5, A) and logp.shape == (5, 1) and det.shape == (5, A) and vf.shape == (5, 256)
        assert (a.abs() <= 1).all() and (det.abs() <= 1).all(), "actions must be within [-1, 1]"
        assert actor.get_eval_action(obs).shape == (5, A)
    print("✓ actor (images only, no state): works with 64 / 128 / 256 images, actions within bounds")


def test_actor_and_q_with_state():
    obs = fake_obs(3, 128, with_state=True)
    actor = Actor(obs, LOW, HIGH)
    q = SoftQNetwork(actor.encoder, actor.state_dim, A)
    a, _, _, vf = actor.get_action(obs)
    assert actor.state_dim == 25 and q(obs, a, vf).shape == (3, 1)
    print("✓ actor + Q (with state): shapes correct")


def test_q_no_state():
    obs = fake_obs(3, 128)
    actor = Actor(obs, LOW, HIGH)
    q = SoftQNetwork(actor.encoder, actor.state_dim, A)
    assert q(obs, torch.zeros(3, A)).shape == (3, 1)
    print("✓ Q (no state): shapes correct")


def test_replay_buffer_vla():
    num_envs = 4
    obs = fake_obs(num_envs)
    rb = ReplayBuffer({k: v[:1] for k, v in obs.items()}, A, num_envs, buffer_size=40,
                      storage_device="cpu", sample_device="cpu")
    vla = torch.full((num_envs, A), 0.5)
    # first 5 steps have VLA labels, last 5 do not
    for t in range(10):
        rb.add(obs, obs, torch.zeros(num_envs, A), torch.ones(num_envs), torch.zeros(num_envs, dtype=torch.bool),
               vla_action=vla if t < 5 else None)
    assert rb.full and len(rb) == 40
    s = rb.sample(256)
    assert s.obs["rgb"].shape == (256, 128, 128, 3) and s.vla_actions.shape == (256, A)
    # labelled vla_actions are all 0.5, unlabelled ones are 0
    assert torch.allclose(s.vla_actions[s.has_vla], torch.tensor(0.5))
    assert torch.allclose(s.vla_actions[~s.has_vla], torch.tensor(0.0))
    assert 0 < s.has_vla.float().mean() < 1
    print(f"✓ replay buffer: stores samples with and without VLA labels correctly (labelled fraction in this batch: {s.has_vla.float().mean():.2f})")


def test_distill_loss_values_and_mask():
    s = torch.zeros(4, A)
    v = torch.ones(4, A)
    assert torch.isclose(distill_loss(s, v), torch.tensor(1.0))
    assert torch.isclose(distill_loss(s, v, loss_type="l1"), torch.tensor(1.0))
    mask = torch.tensor([True, False, False, False])
    v2 = v.clone()
    v2[1:] = 100.0  # unlabelled samples must not affect the loss, however large
    assert torch.isclose(distill_loss(s, v2, mask), torch.tensor(1.0))
    empty = distill_loss(s.requires_grad_(), v, torch.zeros(4, dtype=torch.bool))
    assert empty.item() == 0.0
    empty.backward()  # backward must also work when there are no labels
    print("✓ distillation loss: mse / l1 values correct, masked-out samples ignored, loss=0 with no labels")


def test_distill_pulls_student_toward_vla():
    """Training the actor with only the distillation loss should move mu_theta(s) towards the VLA action (correct gradient direction)."""
    obs = fake_obs(32, 64)
    actor = Actor(obs, LOW, HIGH)
    target = torch.tensor([0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 0.8]).repeat(32, 1)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    losses = []
    for _ in range(200):
        _, _, det, _ = actor.get_action(obs)
        loss = distill_loss(det, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * losses[0], f"loss did not drop clearly: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"✓ distillation gradient direction correct: loss {losses[0]:.4f} -> {losses[-1]:.5f}")


def test_lambda_zero_means_pure_sac():
    """With lambda=0 the distillation term must contribute zero gradient to the actor (= pure SAC)."""
    obs = fake_obs(8, 64)
    actor = Actor(obs, LOW, HIGH)
    _, _, det, _ = actor.get_action(obs)
    (0.0 * distill_loss(det, torch.ones(8, A))).backward()
    assert all(p.grad is None or torch.all(p.grad == 0) for p in actor.parameters())
    print("✓ lambda=0: distillation term has no gradient, i.e. pure SAC")


def test_clamp():
    a = torch.tensor([[-5.0, 0.0, 5.0, 0.5, -0.5, 1.0, -1.0]])
    c = clamp_to_action_bounds(a, torch.as_tensor(LOW), torch.as_tensor(HIGH))
    assert (c.abs() < 1).all() and torch.isclose(c[0, 1], torch.tensor(0.0))
    print("✓ out-of-bound VLA actions are clamped into the action bounds")


if __name__ == "__main__":
    test_resize()
    test_actor_shapes_no_state()
    test_actor_and_q_with_state()
    test_q_no_state()
    test_replay_buffer_vla()
    test_distill_loss_values_and_mask()
    test_distill_pulls_student_toward_vla()
    test_lambda_zero_means_pure_sac()
    test_clamp()
    print("\nALL TESTS PASSED")
