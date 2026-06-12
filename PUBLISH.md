# Tiny Zombie Q Policies

This folder is ready to become its own GitHub repo.

First push:

```powershell
git init
git add .
git commit -m "initial tiny zombie q policies"
gh repo create tiny-zombie-q-policies --public --source . --remote origin --push
```

Before publishing, run:

```powershell
cargo test
git status --short
```

Keep full game source, raw telemetry, Godot cache, local build outputs, and full
training runs out of this repo.

The paper-facing result files in `paper-results/` should match the active paper
**"When Should Small Games Use AI?"**.
