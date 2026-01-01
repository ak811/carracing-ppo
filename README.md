# CarRacing PPO (Gymnasium + Stable-Baselines3)

This repository implements **Proximal Policy Optimization (PPO)** for `CarRacing-v3` using Gymnasium + Stable-Baselines3 (SB3). The goal is to reach **>900 average reward**.

- Training entry: `python scripts/train.py --config configs/default.yaml`
- Evaluation: `python scripts/evaluate.py --config configs/default.yaml --checkpoint <path>`
- Video: `python scripts/record_video.py --config configs/default.yaml --checkpoint <path>`
- Sweep: `python scripts/sweep.py --sweep configs/sweep.yaml`

![Car Racing](https://gymnasium.farama.org/_images/car_racing.gif)

---

## CarRacing Environment

`CarRacing-v3` is a continuous-control Box2D environment with:

- Observation: RGB image (96×96×3) by default (optionally resized/grayscaled)
- Actions: 3D continuous vector `[steer, gas, brake]`
- Reward shaping that encourages staying on the track and making forward progress

See Gymnasium docs for full detail.

---

## PPO Notes

This implementation uses SB3’s PPO with a CNN policy (`CnnPolicy`) and the standard PPO clipped objective:

- Policy loss: `min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)`
- Value loss: MSE between predicted value and return
- Entropy bonus encourages exploration

Advantages are computed using GAE(λ). Observations are preprocessed with common image RL wrappers (optional grayscale + resize, frame stacking, channel transpose).

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Train (single run)
```bash
python scripts/train.py --config configs/default.yaml
```

Outputs go under `experiments/<run>/`:
- `models/` (best_model.zip, final_model.zip, etc.)
- `logs/` (SB3 evaluation logs and TensorBoard files if enabled)
- `plots/` (evaluation curves)
- `videos/` (if recorded)

### Evaluate a checkpoint
```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint experiments/<run>/models/best_model.zip \
  --episodes 20
```

### Record a rollout video
```bash
python scripts/record_video.py \
  --config configs/default.yaml \
  --checkpoint experiments/<run>/models/best_model.zip \
  --out_dir assets/videos
```

### Run a sweep
```bash
python scripts/sweep.py --sweep configs/sweep.yaml
```

---

## Best Performing Model

These settings were found to be stable and effective for CarRacing PPO:

| Parameter | Value |
|---|---|
| env_name | CarRacing-v3 |
| total_timesteps | 3_000_000 |
| learning_rate | 3e-4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| n_steps | 2048 |
| n_epochs | 10 |
| batch_size | 64 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| frame_stack | 4 |
| grayscale | true |
| resize | 84 |

---

## Best Model Outputs

**Policy rollout video** (best model):

<video src="https://github.com/user-attachments/assets/20a5b291-d0a4-48c0-b543-328bb94bfde1" controls width="600">
  Your browser does not support the video tag.
</video>

---

## Testing

```bash
pytest -q
```

---

## License

MIT (See `LICENSE`).
