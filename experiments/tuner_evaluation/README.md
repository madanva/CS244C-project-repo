# Tuner Evaluation

Validation experiments comparing the profile-guided and RL bandit tuners against NCCL AUTO.

## What this measures

- **Profile-guided tuner (v2, v3)**: Static policy tables generated from profiling data, evaluated on held-out runs
- **RL bandit tuner (v2)**: Online learning of optimal configuration with deterministic round-robin exploration
- Bandit convergence speed, variance, and safety-gate behavior
- Head-to-head latency comparison against AUTO across topologies

## Key finding

The bandit achieves 25-42% latency reductions over AUTO on multi-node topologies, converging in ~40 iterations. The 5% safety gate prevents regressions on configurations where AUTO is already near-optimal.

## Contents

| Path | Description |
|------|-------------|
| `run_tuner_validation.py` | Modal runner: profile-guided tuner validation |
| `run_tuner_validation_multinode.py` | Modal runner: tuner validation on multi-node |
| `tuner_results/v1/` | Profile-guided tuner v1 results |
| `tuner_results/v2/` | Profile-guided tuner v2 results (topology-aware) |
| `tuner_results/v3/` | Profile-guided tuner v3 results (with C lookup table) |
| `bandit_results/v2/` | Bandit v2 results (deterministic round-robin) |
| `bandit_results/download/` | Downloaded bandit results from cloud runs |
| `scripts/` | Figure generation scripts |

## Key scripts

- `scripts/generate_validation_figures.py` -- Tuner validation comparison figures
- `scripts/generate_bandit_figures.py` -- Bandit convergence and comparison figures
- `scripts/generate_profile_guided_figure.py` -- Profile-guided tuner performance
- `scripts/generate_interpretability_figures.py` -- Bandit arm selection and reward visualizations
- `scripts/generate_tuner_policy.py` -- Generates policy table from profiling data
