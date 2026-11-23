# sac_rpd：純 SAC 與 SAC + RPD

同一份程式，用 `--vla_lambda` 切換：

| 模式 | 指令重點 | 需要 Octo 環境？ |
|---|---|---|
| 純 SAC baseline | `--vla_lambda 0`（預設） | 不用 |
| SAC + RPD（假 VLA，測流程） | `--vla_lambda 1 --fake_vla` | 不用 |
| SAC + RPD（真的 Octo） | `--vla_lambda 1` | 要（`rpd_octo`） |

## 檔案
- `networks.py`：actor、Q 網路、replay buffer（多存 VLA 動作）、蒸餾 loss。只用 torch，筆電就能測
- `sac_rpd.py`：訓練主程式（需要 GPU + ManiSkill）
- `test_networks.py`：`networks.py` 的單元測試（CPU）

## 1. 筆電（CPU）：跑單元測試
```bash
conda activate rpd
cd ~/RPD_ADHD/sac_rpd
python test_networks.py      # 看到 ALL TESTS PASSED
```

## 2. GPU 電腦：照順序跑，每一步成功再往下
在 `~/RPD_ADHD/sac_rpd` 裡執行（`runs/` 會建在這裡）：

```bash
# (a) 純 SAC，小規模，確認能跑
python sac_rpd.py --total_timesteps 50000 --eval_freq 10000

# (b) 純 SAC，正式
python sac_rpd.py --total_timesteps 1000000

# (c) SAC + RPD，假 VLA：確認 VLA 查詢、buffer、蒸餾 loss 流程沒錯（成功率不重要）
python sac_rpd.py --vla_lambda 1 --fake_vla --total_timesteps 50000 --eval_freq 10000

# (d) SAC + RPD，真的 Octo（先跑 setup_octo.sh）
python sac_rpd.py --vla_lambda 1 --vla_conda_path ~/miniconda3/envs/rpd_octo/bin/python --total_timesteps 1000000
```

看結果：
```bash
tensorboard --logdir runs
```
重點看 `eval/success_once`、`losses/distill_loss`、`charts/vla_queries`。

## 和 RPD 一致的地方
- 環境：`obs_mode="rgb"`、`pd_ee_delta_pose`、`normalized_dense` reward、`HumanCameraWrapper`（256x256）、不看 state
- teacher：`--vla_type octo` = `Juelg/octo-base-1.5-finetuned-maniskill`，查詢方式一樣（`rgb_side` → `info["means"]`）
- 蒸餾 loss：`mse(μθ(s), a_VLA)`，對應 RPD 的 `vlad_loss_type=mse`、`num_vla_samples=1`

## 和 RPD 不同的地方（報告要寫）
1. **查 VLA 的時機**：RPD（PPO）在更新時查；這裡在收集資料時查一次存進 replay buffer，因為 SAC 會重複抽樣舊資料，每次更新都查太慢。查詢的狀態一樣都是 student 走到的狀態
2. **student 影像大小**：replay buffer 存 128x128（`--student_image_size`）省 GPU 記憶體；VLA 仍看原始 256x256
3. **λ 的尺度**：SAC 的 actor loss 跟 PPO 的 loss 大小不同，`--vla_lambda 1` 不一定最好，之後要掃幾個值

## 記憶體
- replay buffer 預設 10 萬筆、128x128 影像、放 GPU：約 10 萬 × 2（obs + next_obs）× 49KB ≈ 10GB
- 顯卡不夠大就調小（`--buffer_size 50000`）或放 CPU 記憶體（`--buffer_device cpu`，會慢一點）
