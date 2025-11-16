# Performance vs Longevity in Satellite ADCS

A decision-making under uncertainty project investigating optimal spacecraft attitude control when reaction wheel actuators degrade over time.

## Authors

- Emma Nicolai (enicolai@stanford.edu)
- Zeb Barry (zbarry@stanford.edu)
- Ottavia Personeni (otpers03@stanford.edu)

## Overview

This project addresses the challenge of controlling spacecraft attitude when reaction wheels experience uncertain degradation. The system balances minimizing attitude tracking error while managing actuator health degradation through a Partially Observable Markov Decision Process (POMDP) framework with belief tracking.

### Key Features

- **Spacecraft Dynamics**: Single-axis attitude control with reaction wheel actuator
- **Health Degradation Model**: Multi-state health progression (healthy → degraded → critical → failed) with usage-dependent degradation
- **Partial Observability**: Noisy observations of attitude and angular velocity; health state must be inferred
- **Belief Tracking**: Maintains probability distribution over health states during operation
- **Multiple Solvers**: Support for QMDP (fast approximate), SARSOP (offline optimal), and POMCP (online)
- **Comprehensive Visualization**: Attitude trajectory, health belief evolution, phase portraits, and performance metrics

## Installation

### Prerequisites

- Julia 1.6 or higher ([download here](https://julialang.org/downloads/))

### Setup

1. Clone this repository and navigate to the project directory:

   ```bash
   cd UncertainADCS
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

## Project Structure

- `src/UncertainADCS.jl`: POMDP model definition including state space, transition dynamics, observation model, and reward function
- `src/main.jl`: Solver configuration, simulation execution, and visualization generation
- `Project.toml`: Julia package dependencies

## POMDP Formulation

### State Space
- **Angle (θ)**: Spacecraft attitude error relative to target [-π, π] rad
- **Angular Velocity (ω)**: Rate of attitude change [-0.5, 0.5] rad/s
- **Health**: Discrete health states {:healthy, :degraded, :critical, :failed}

### Actions
Five discrete torque levels: {-5.0, -2.5, 0.0, 2.5, 5.0} N⋅m

### Observations
Noisy measurements of angle and angular velocity with Gaussian noise (σ_θ = 0.05 rad, σ_ω = 0.01 rad/s)

### Reward Function
Balances multiple objectives:
- Attitude error penalty: -100θ²
- Angular velocity penalty: -10ω²
- Control effort penalty: -0.01u²
- Failure penalty: -1000 (if health = failed)

### Transition Dynamics
- Spacecraft dynamics: Standard rigid body rotation with torque input
- Health degradation: Probabilistic transitions based on usage intensity
  - Degradation probability: p = k₁ + k₂|u|/u_max
  - Natural degradation: k₁ = 0.0
  - Usage-dependent: k₂ = 0.0005

## Usage

### Running a Simulation

```julia
julia --project=. src/main.jl
```

This will:
1. Create the spacecraft POMDP model
2. Solve using QMDP solver (configurable in main.jl)
3. Run a 100-step simulation
4. Generate visualization plots

### Available Solvers

Uncomment the desired solver in `src/main.jl`:

```julia
# QMDP - Fast approximate solution (default)
qmdp_solver = QMDPSolver(max_iterations=1000, verbose=true)

# SARSOP - Slower but more accurate offline solution
# sarsop_solver = SARSOPSolver(precision=1e-3, verbose=true)

# POMCP - Online Monte Carlo tree search
# pomcp_solver = POMCPSolver(tree_queries=1000, c=10.0, max_depth=50)
```

### Output Visualizations

The simulation generates four plots:

1. **spacecraft_attitude_control_simulation.png**: Time series of angle, angular velocity, control torque, and health state
2. **health_belief_evolution.png**: Stacked area chart showing probability distribution over health states
3. **phase_portrait.png**: State space trajectory with time progression and target location
4. **cumulative_metrics.png**: Cumulative reward, RMS error, and control effort over time

## References

Sunberg, Z. N., & Kochenderfer, M. J. (2018). Online algorithms for POMDPs with continuous state, action, and observation spaces. _Proceedings of the International Conference on Automated Planning and Scheduling_, 28, 259-263.
