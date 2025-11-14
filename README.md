# Performance vs Longevity in Satellite ADCS

A decision-making under uncertainty project investigating optimal spacecraft attitude control when reaction wheel actuators degrade over time.

## Authors

- Emma Nicolai (enicolai@stanford.edu)
- Zeb Barry (zbarry@stanford.edu)
- Ottavia Personeni (otpers03@stanford.edu)

## Overview

This project addresses the challenge of controlling spacecraft attitude when reaction wheels experience uncertain degradation. The system balances minimizing tracking angle error while maximizing actuator health, progressing from a fully observable MDP to a POMDP with belief tracking using particle filters.

## Installation

### Prerequisites

- Julia 1.6 or higher ([download here](https://julialang.org/downloads/))

### Setup

1. Clone this repository and navigate to the project directory:

   ```bash
   cd path/to/project
   ```

2. Start Julia in the project directory:

   ```bash
   julia --project=.
   ```

3. Install dependencies:
   ```julia
   using Pkg
   Pkg.instantiate()
   ```

### Alternative: Manual Package Installation

If `Project.toml` is not yet configured, you can manually add packages:

```julia
using Pkg
Pkg.activate(".")
# Add your required packages, for example:
# Pkg.add("POMDPs")
# Pkg.add("MCTS")
# Pkg.add("ParticleFilters")
```

## Project Stages

1. **Stage 1**: Fully Observable MDP with Monte Carlo tree search
2. **Stage 2**: Observable health degradation with stochastic actuator wear
3. **Stage 3**: POMDP with particle filter belief tracking (POMCPOW-based)

## Usage

_To be added as implementation progresses_

## References

Sunberg, Z. N., & Kochenderfer, M. J. (2018). Online algorithms for POMDPs with continuous state, action, and observation spaces. _Proceedings of the International Conference on Automated Planning and Scheduling_, 28, 259-263.
