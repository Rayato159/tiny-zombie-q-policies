# Tiny Zombie Q Policies

Small Rust models for tactical zombie swarm control in a real Godot game.

This repo is the model side only. It does not ship the full game source. The
goal is simple: keep the neural policy tiny enough to run inside a frame, then
check whether it can replace part of a hand-written zombie rule tree without
turning the project into a giant ML circus.

No fake AGI claim. No giant checkpoint. No benchmark number unless the command
that produced it exists.

## What This Is

The game still owns the hard real-time stuff:

- A* / navigation
- movement
- collision
- melee hit timing
- animation
- spawning

The model only chooses a high-level tactical action:

```text
ATTACK
FLANK_LEFT
FLANK_RIGHT
DASH_IN
DASH_OUT
```

That action goes through a feasibility layer in Godot/Rust. If the model asks
for nonsense, the game clamps it back into something executable.

## Current Models

### Rule Expert

The baseline is hand-written. It reads the same tactical state and chooses an
action with explicit branches. This is the thing the neural policies are trying
to compete with, not a strawman that walks into a wall.

### Tiny MLP Q-Policy

Budget target: about 1k parameters by default.

Shape:

```text
state -> linear -> ReLU -> Q(action)
```

It is boring on purpose. If attention cannot beat this under the same state and
action contract, attention does not get a trophy for looking cool.

### Grouped-Attention Q-Policy

Budget target: below 10k parameters.

The state is treated as feature tokens. A tiny grouped-attention stack lets the
policy compare player gates, relative geometry, orientation, and swarm pressure
before producing Q-values.

Grouped K/V heads keep the parameter count low. The runtime attention path in
the game uses a streamed recurrence inspired by FLASH-D style attention so it
does not need to materialize a full score buffer.

Current honest limitation: the lightweight trainer updates the Q readout for
attention checkpoints. Full attention-stack backprop is planned behind the
optional `burn` feature.

## State Contract

Zombie policies use 23 features:

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

Player-side training uses a separate 15-feature survival-fighter state:

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

These names are part of the contract. Change them casually and the checkpoint
loader should reject your nonsense.

## Quick Start

Run tests:

```powershell
cargo test
```

Train a tiny MLP zombie checkpoint from a transition CSV:

```powershell
cargo run --bin train_tiny_q -- `
  --input ..\assets\ai\tiny_q_mlp_selfplay_r1002.csv `
  --output ..\checkpoints\zombie_policy_mlp_release.json `
  --model mlp `
  --role zombie `
  --epochs 100
```

Train a grouped-attention checkpoint:

```powershell
cargo run --bin train_tiny_q -- `
  --input ..\assets\ai\tiny_q_attention_selfplay_r1205.csv `
  --output ..\checkpoints\zombie_policy_attention_release.json `
  --model attention `
  --role zombie `
  --d-model 24 `
  --layers 2 `
  --heads 4 `
  --kv-heads 1 `
  --epochs 100
```

Train a player-side policy:

```powershell
cargo run --bin train_tiny_q -- `
  --input ..\assets\ai\player_q_attention_selfplay_r1205.csv `
  --output ..\checkpoints\player_policy_r1205.json `
  --model mlp `
  --role player `
  --input-dim 15 `
  --epochs 100
```

## CSV Schema

The trainer expects transition rows:

```text
action_id,reward,done,prev_<feature...>,next_<feature...>
```

For zombie policies, `prev_` and `next_` must cover the 23 zombie features.
For player policies, they must cover the 15 player features.

Rewards are intentionally gameplay-shaped:

- hurt the player
- force stamina pressure
- finish the kill fast
- avoid getting hit or killed
- avoid wasting dash actions
- create flank and back-pressure

The point is not to solve a toy grid. The point is to make the actual game
harder in a way a player can feel.

## Parameter Budget

The paper keeps policies under 10k parameters. The crate has helpers that pin
the default sizes:

```text
Tiny MLP default:              991 parameters
Grouped-attention default:   3,677 parameters
Budget ceiling:             10,000 parameters
```

If a model crosses the ceiling, it is not part of the tiny-policy comparison.

## Runtime Shape

The Godot game loads JSON checkpoints through the Rust GDExtension runtime.
The public checkpoint format is deliberately plain JSON so the model can be
inspected, diffed, and loaded without a Python environment.

Release checkpoints used by the game:

```text
zombie_policy_mlp_release.json
zombie_policy_attention_release.json
```

The game menu can switch between:

```text
Rule Expert
Tiny MLP
Grouped Attention
```

## What To Publish

Good public repo contents:

- this `model_core` crate
- tiny release checkpoints
- small sample CSV snippets
- scripts that reproduce paper tables
- paper result summaries

Do not dump the full game, raw telemetry, Godot cache, or target directories
into the model repo. That is not reproducibility. That is a landfill.

## License

The model crate is prepared for `MIT OR Apache-2.0`.
