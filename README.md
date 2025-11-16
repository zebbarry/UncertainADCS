# Uncertain ADCS

Spacecraft attitude control with reaction wheel degradation using POMDPs.

## Authors

- Emma Nicolai (enicolai@stanford.edu)
- Zeb Barry (zbarry@stanford.edu)
- Ottavia Personeni (otpers03@stanford.edu)

## Setup

Install Julia 1.6+ from [julialang.org/downloads](https://julialang.org/downloads/)

```bash
cd UncertainADCS
julia --project=.
```

```julia
using Pkg
Pkg.instantiate()
```

## Run

```bash
julia --project=. src/main.jl
```

Generates visualizations: spacecraft_attitude_control_simulation.png, health_belief_evolution.png, phase_portrait.png, and cumulative_metrics.png
