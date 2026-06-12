# 🧟 Tiny Zombie Q Policies

<p align="center">
  <img src="arts/zombie_1_move_sample.gif" alt="zombie policy in action" width="240" />
</p>

> Small Rust models for tactical game AI in a real Godot project 🎮

Rust-only model code for the paper **"When Should Small Games Use AI?"**

This repo is the model side only 🦀 It does not ship the full Godot game source.
The paper uses a controlled 2D action-game simulation to ask a practical
question:

> when should a small game keep handcrafted tactical rules, and when do tiny
> learned policies become worth the extra data/training/runtime cost?

No fake AGI pitch. No giant checkpoint. No "trust me bro" model magic. The
current result is more useful than that: the rule expert is still the strongest
default controller in the narrow arena, while an 8k MLP gets close and the 10k
grouped-attention model increases tactical coverage without winning the primary
speed metric 🔒

## 🎥 Demo

Current gameplay demo:

[Watch the demo video](https://youtu.be/UGRb-PfT5CQ?si=FmYV-yOfVVga3x5x)

The model does not drive raw movement. Godot still owns pathfinding, collision,
animation, cooldowns, hit detection, and combat execution. The policy only
selects a high-level tactical action, then the game-side feasibility layer makes
that action executable.

## 📄 Paper

Read the current paper draft here:

[When Should Small Games Use AI?](paper-results/When%20Should%20Small%20Games%20Use%20AI.pdf)

## 📦 What Is In This Repo

- `zombie_policy_core`: small Rust policy definitions and helpers.
- `train_tiny_q`: offline CSV-to-JSON checkpoint trainer.
- `paper-results/`: paper-facing result tables copied from the current
  manuscript artifacts, plus the current PDF draft.
- `arts/`: small public demo media.

The full game source, raw telemetry, Godot caches, and local build output do not
belong here.

## 🕹️ Policy Interface

The enemy policy chooses one of five tactical actions:

```text
ATTACK
FLANK_LEFT
FLANK_RIGHT
DASH_IN
DASH_OUT
```

The player-side bot uses the same action count but a different action meaning:

```text
ATTACK
APPROACH
RETREAT
STRAFE
ROLL
```

## 🧠 Current Model Shapes

These defaults match the current paper framing.

| Model | Config | Runtime params |
| --- | --- | ---: |
| MLP 8k | 23 inputs, hidden width 275, 5 actions | 7,980 |
| Grouped attention 10k | 23 inputs, d=40, 2 layers, 4 query heads, 1 KV head, 5 actions | 9,325 |

The capacity sweep in the paper still reports 1k, 2k, 4k, 8k, and 10k budget
buckets. The selected paper comparison is:

| Controller | TTK/timeout (s) | Defeat rate | Fastest defeat (s) | Speed score | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rule expert | 5.76 +/- 1.64 | 0.98 +/- 0.03 | 3.03 +/- 0.34 | 0.90 +/- 0.03 | 0.84 +/- 0.08 |
| MLP 8k | 5.90 +/- 0.31 | 1.00 +/- 0.00 | 3.55 +/- 0.31 | 0.90 +/- 0.01 | 0.56 +/- 0.02 |
| Attention 10k | 6.18 +/- 0.35 | 1.00 +/- 0.00 | 3.77 +/- 0.16 | 0.90 +/- 0.01 | 0.66 +/- 0.10 |

Lower TTK/timeout is better. Failed player defeats count as the full 60-second
timeout.

## 📡 State Contract

Enemy policies use 23 features. The names keep the historical `zombie` and
`swarm` prefixes because existing CSV/checkpoint contracts use them.

```text
is_player_armed
is_player_stamina_less_half
is_player_health_less_half
is_player_stuck
player_stuck_normal_x
player_stuck_normal_y
nearby_zombie_count
dash_ready
attack_ready
nearest_zombie_distance
nearest_zombie_dir_x
nearest_zombie_dir_y
player_speed
swarm_centroid_dir_x
swarm_centroid_dir_y
player_facing_dot_nearest_zombie
nearest_zombie_side_sign
backstab_opportunity
swarm_left_pressure
swarm_right_pressure
swarm_front_pressure
swarm_back_pressure
swarm_spread
```

Player-side training uses a separate 15-feature duelist state:

```text
health_ratio
stamina_ratio
is_player_armed
is_player_attacking
nearest_zombie_distance
nearest_zombie_dir_x
nearest_zombie_dir_y
nearest_zombie_attacking
zombie_count
pressure_count
player_speed
player_facing_dot_nearest_zombie
swarm_centroid_dir_x
swarm_centroid_dir_y
dodge_ready
```

Do not rename these casually. The trainer expects CSV columns in the form:

```text
action_id,reward,done,prev_<feature...>,next_<feature...>
```

## ⚡ Quick Start

Run tests:

```powershell
cargo test
```

Train the selected 8k MLP enemy checkpoint:

```powershell
cargo run --bin train_tiny_q -- `
  --input ..\assets\ai\tiny_q_mlp_selfplay.csv `
  --output ..\checkpoints\zombie_policy_mlp_release.json `
  --model mlp `
  --role zombie `
  --hidden-dim 275 `
  --epochs 100
```

Train the selected 10k grouped-attention enemy checkpoint:

```powershell
cargo run --bin train_tiny_q -- `
  --input ..\assets\ai\tiny_q_attention_selfplay.csv `
  --output ..\checkpoints\zombie_policy_attention_release.json `
  --model attention `
  --role zombie `
  --d-model 40 `
  --layers 2 `
  --heads 4 `
  --kv-heads 1 `
  --epochs 100
```

Train a player-side MLP checkpoint:

```powershell
cargo run --bin train_tiny_q -- `
  --input ..\assets\ai\player_q_selfplay.csv `
  --output ..\checkpoints\player_policy_release.json `
  --model mlp `
  --role player `
  --input-dim 15 `
  --hidden-dim 275 `
  --epochs 100
```

## 🔬 Model Math

MLP:

```text
x = (s - mean) / (std + eps)
z = ReLU(W1 x + b1)
Q(s, .) = W2 z + b2
```

Grouped-attention:

```text
t_i = feature_embedding_i + x_i * value_embedding
T_0 = [t_1, ..., t_23, q_token]
T_{l+1} = T_l + GQA(RMSNorm(T_l))
Q(s, .) = Wq RMSNorm(q_token_final) + bq
```

The current offline trainer uses a Double-DQN-style target. For the attention
model, the released trainer updates the Q readout while the attention stack is
kept fixed. That matches the paper's current limitation section. Do not claim
full attention-stack backprop unless you actually implement it.

## 📊 Paper Results

The paper-facing result tables and plot inputs live in `paper-results/` and are
synced to the current manuscript:

- `eval_summary_table_latest.tex`
- `capacity_sweep_summary_table_latest.tex`
- `seed_repeat_summary_table_latest.tex`
- `default_comparison_latex_latest.csv`
- `default_action_distribution_latex_latest.csv`
- `capacity_sweep_mlp_latex_latest.csv`
- `capacity_sweep_attention_latex_latest.csv`
- `selfplay_learning_curve_mlp_latest.csv`
- `selfplay_learning_curve_attention_latest.csv`

The active paper title is **"When Should Small Games Use AI?"** The repo name is
historical because the original Godot enemy implementation used zombie agents.

## 🪪 License

MIT. See [LICENSE](LICENSE).
