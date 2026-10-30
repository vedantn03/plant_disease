# AGENTS.md – Conventions & Definition of Done

## Who Is This For?
Any human contributor or AI coding agent working on this repository.

## Definition of Done
A task is done when:
1. The code **runs** end-to-end without errors.
2. At least one **pytest test** covers the new functionality.
3. The task does **not invent facts** (no fake metrics, dosages, benchmarks, product names, or API behaviours).
4. **Raw data and model weights are NOT committed** (see .gitignore).
5. No secrets or `.env` files are committed.

## Conventions

### Code Style
- Python 3.12+ compatible (project uses 3.14; keep it importable on 3.12 for portability).
- PEP 8. Max line length 100. Use type hints where practical.
- All scripts are runnable directly (`if __name__ == "__main__"`) and importable as modules.

### Naming
- Variables and functions: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Config keys: `snake_case` in YAML.

### Configuration
- Every tunable value lives in `configs/config.yaml`. Never hardcode paths, seeds, thresholds, or sizes.
- Load config via `src/config_loader.py`. Use the `DATA_ROOT` env-var for the dataset path.

### Data Rules
- Never feed filenames, source codes, crop-identifying metadata, or UUIDs to the model.
- Never delete near-duplicate images automatically; report them, keep one exact duplicate only.

### Commit Messages (Conventional Commits)
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `test:` adding or changing tests
- `chore:` build, deps, config

### What to Avoid
- **Invented facts**: no made-up accuracy numbers, dosages, product names, or API behaviours.
- **Complexity for complexity's sake**: boring, proven, simple solutions preferred.
- **Model-generated dosages or product recommendations**.
- **Cloud/paid services**: everything runs locally and free.
- **Removing existing docstrings or comments** unrelated to your change.

## Key Decisions Log
See `docs/DECISIONS.md` for all architectural decisions and deviations from the brief.
