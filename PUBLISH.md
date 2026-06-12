# Syncopate Machine v1.0.0 Publish Notes

This folder is the model-side public repo for the current zombie-policy work.
It intentionally does not contain the full Godot game.

Before pushing:

```powershell
cargo test
cargo run --release --bin train_entity_decoder_selfplay -- --compare-rule-expert=true --resume=true --dt=0.1 --episode-seconds=60 --output-dir=checkpoints --eval-episodes=64
git status --short
```

Keep out:

- full Godot project source
- raw telemetry dumps
- local build folders
- old 5-action MLP/attention experiments

Keep in:

- current Rust model/trainer code
- selected Syncopate Machine v1.0.0 checkpoints
- paper PDF and clean CSV result tables
- small public zombie art/demo media
