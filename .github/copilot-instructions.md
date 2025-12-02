# Copilot Instructions for zsc-ov2-lab

## Project map
- Primary code lives in `experiments/overcooked_v2_experiments`; `ppo/` holds Hydra configs (`config/`), training loops (`main.py`, `run.py`, `ippo.py`), and utilities (`utils/`).
- `JaxMARL/` is installed in editable mode and supplies the Overcooked-V2 environments consumed via `jaxmarl.make` (see `ppo/ippo.py`).
- Human data tooling sits under `overcooked_v2_experiments/human_rl/` (e.g., `imitation/bc_policy.py`) and plugs optional behavior-cloned partners into PPO runs.
- `experiments/sh_scripts/` contains launcher helpers (`run_user_wandb.sh`, `run_factory*.sh`, `run_visualize.sh`, `copy_fcp.sh`) that encode the teams GPU and population workflows.

## Environment & dependencies
- Follow `README.md`: create a Python 3.10 env, then `pip install -e JaxMARL` and `pip install -e experiments`; the launcher assumes the venv lives at `overcooked_v2/bin` and exports it onto `PATH`.
- `run_user_wandb.sh` wires CUDA/XLA env vars (`XLA_PYTHON_CLIENT_*`, `XLA_FLAGS`, optional `USE_SYSTEM_CUDA_LIBS`) and loads `wandb_info/wandb_api_key`; keep that file present for all runs.
- Hydra defaults live in `ppo/config/base.yaml` (wandb + CNN by default). Prefer CLI overrides (`python ... +experiment=rnn-sp +env=grounded_coord_simple`) or the launcher flags instead of editing config files in-place.

## Training workflows
- `ppo/main.py` routes to `single_run`, `state_sample_run`, or `tune` based on `TUNE`/`NUM_ITERATIONS`. `NUM_ITERATIONS>0` triggers the state-augmented iterative pipeline in `state_sample_run.py`.
- Model hyperparameters come from `ppo/config/model/*.yaml`; keep experiments aligned with `EXPERIMENT_SETTINGS.md`, which documents the expected LR/batch/time horizon per variant.
- Multi-device execution is handled explicitly in `ppo/run.py` via `get_num_devices`, `mini_batch_pmap`, and fallbacks to `vmap` when `NUM_SEEDS` does not divide device count. Match that structure when adding new parallelism paths.
- Memory-sensitive switches: `ENV_DEVICE` (place env rollout on CPU/GPU), `CAST_OBS_BF16`, and `panic.*` overrides; expose them through Hydra flags rather than hard-coding.

## Populations, BC, and evaluation
- FCP flows: copy a finished SP run into `experiments/fcp_populations/<layout>/*` using `sh_scripts/copy_fcp.sh`, then launch `rnn-fcp` with `+FCP=<population_dir>`; `ppo/run.py::load_fcp_populations` expects `fcp_*/run_i/ckpt_*` folders.
- Behavior-cloned teammates are provided by `BCPolicy` (loads from `human_rl/static/human_data`); enabling `+BC=...` forces path planning and injects the policy via `population` args.
- Training artifacts land in `experiments/runs/<timestamp>_<wandb>_<layout>_<tag>/run_<seed>/ckpt_*` plus `reward_summary_{sp,cross}.csv` for evaluation. Downstream scripts (`generate_summary.py` + `utils/summary_utils.py`) read those CSVs to compute SP/XP gaps.
- Use `sh_scripts/run_visualize.sh` (wraps `ppo/utils/visualize_ppo.py`) for GIFs and cross-play metrics; pass `--all` or `--no_viz` for bulk evaluation without rendering.

## Conventions & pitfalls
- Keep `jax.config.update("jax_debug_nans", True)` enabled (already set in `run.py`/`main.py`) when introducing new ops; silent NaNs will abort runs intentionally.
- WandB tags/names are assembled in `run_user_wandb.sh`; if you add new experiment kinds, update both Hydra configs under `ppo/config/experiment/` and the tag serialization block so dashboards stay grouped.
- State-augmentation code (`state_sample_run`) stores intermediate checkpoints via `store_checkpoint`; ensure new training modes write to the same `RUN_BASE_DIR` layout so visualization and population builders keep working.
- Any new population-like mechanism should mirror `FCPWrapperPolicy` expectations: params are stacked along axis 0, and per-policy hidden states must stay batched to avoid recompiles.

## Validation & analysis
- Preferred sanity loop: run a short `NUM_SEEDS=1 NUM_CHECKPOINTS=1` job on `grounded_coord_simple` via the launcher, then inspect `reward_summary_sp.csv` and `runs/.../.hydra/config.yaml` before scaling out.
- For aggregated reporting, run `python experiments/generate_summary.py --runs-dir experiments/runs --start-date <YYYYMMDD>`; it appends to (or creates) `runs/summary_sp_xp.csv` while deduplicating `run_name`.
- When debugging GPU issues on shared nodes, rely on the launchers `nvidia-smi` preflight and the `SUPPRESS_PTX_WARN` filter; if a run fails during `env.reset`, first try `--env-device cpu` or lower `model.NUM_ENVS` via CLI overrides.
