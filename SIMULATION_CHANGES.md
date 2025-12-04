# Simulation Seed Consistency Implementation

## Summary

This update implements **multiple simulations with consistent seeding** across all solvers for rigorous statistical comparison of POMDP solver performance.

## Key Changes

### 1. Modified `src/main.jl`

#### Updated `run_simulation` function
- Added `rng` parameter to control randomness
- Passes RNG to both `HistoryRecorder` and `simulate()` for reproducible simulations

#### New Multi-Simulation Framework
- **30 simulations per solver** (configurable via `num_simulations` variable)
- **Same seed sequence (1-30)** used across all solvers for fair comparison
- **5000 steps per simulation** (configurable via `num_steps` variable)

#### Solver RNG Consistency
- All solvers now use consistent RNG seeding
- POMCP uses seed 42 for policy solving, then individual seeds for simulations
- Each simulation uses a fresh `MersenneTwister(seed)` for that specific run

### 2. Enhanced `src/plotting.jl`

#### New `save_multi_simulation_results()` function
Aggregates and analyzes results from multiple simulation runs:

**Statistical Analysis:**
- Mean, standard deviation, min, max for all metrics
- 95% confidence intervals
- Coefficient of variation (CV) for stability assessment

**Metrics Tracked:**
- Total reward
- Final RMS error
- Total control effort
- Mean angle error
- Average belief entropy

**Output Files:**
- `aggregated_statistics.txt` - Comprehensive statistical summary
- `reward_distribution.png` - Histogram of total rewards
- `error_distribution.png` - Histogram of RMS errors
- `simulation_comparison.png` - Overlay of trajectory comparisons
- `control_distribution.png` - Histogram of control efforts
- `runs/` subfolder - Individual run results (every 5th run saved)

#### New `save_individual_run()` function
Saves simplified results for individual simulation runs

## Benefits

### Fair Comparison
- Each solver faces the **exact same sequence of random events** per seed
- Eliminates luck as a confounding factor in performance comparison
- Pure policy comparison across solvers

### Statistical Robustness
- Multiple runs enable calculation of mean, std, and confidence intervals
- Understand performance variability and consistency
- Rigorous statistical conclusions about solver performance

### Reproducibility
- Fixed seeds ensure experiments can be reproduced exactly
- Easier to debug differences in behavior
- Scientific rigor for research and publication

## Usage

Simply run the modified `main.jl`:

```bash
julia src/main.jl
```

### Configuration

Edit these variables in `main.jl` to adjust the experiment:

```julia
num_simulations = 30  # Number of simulation runs per solver
num_steps = 5000      # Steps per simulation
seeds = 1:num_simulations  # Seed sequence (can customize)
```

### Output Structure

```
results/
├── QMDP/
│   ├── aggregated_statistics.txt
│   ├── reward_distribution.png
│   ├── error_distribution.png
│   ├── simulation_comparison.png
│   ├── control_distribution.png
│   └── runs/
│       ├── run_1/
│       ├── run_5/
│       └── ...
├── SARSOP/
│   └── (same structure)
└── POMCP/
    └── (same structure)
```

## Implementation Details

### Random Number Generation

1. **Policy Solving**: Each solver uses its own RNG during policy computation
   - POMCP: `MersenneTwister(42)` for tree search
   - QMDP/SARSOP: Default RNG (only affects tie-breaking)

2. **Simulation**: Each simulation run uses a fresh RNG with seed from the sequence
   ```julia
   rng = MersenneTwister(seed)
   hist = run_simulation(pomdp, policy, num_steps, rng)
   ```

3. **Cross-Solver Consistency**: All solvers use the same seed sequence (1:30)
   - Simulation 1 for QMDP uses seed 1
   - Simulation 1 for SARSOP uses seed 1
   - Simulation 1 for POMCP uses seed 1
   - This ensures each solver faces identical random events

### Statistical Metrics

The aggregated statistics include:

- **Mean & Std Dev**: Central tendency and spread
- **95% Confidence Intervals**: Statistical significance of differences
- **Coefficient of Variation (CV)**: Consistency metric (lower = more stable)
- **Min/Max**: Range of performance across runs

## Example Interpretation

If QMDP shows:
- Mean reward: 1000 ± 50 (CV: 5%)
- Mean RMS error: 5.2° ± 0.3° (CV: 6%)

And SARSOP shows:
- Mean reward: 1100 ± 200 (CV: 18%)
- Mean RMS error: 4.8° ± 1.2° (CV: 25%)

**Interpretation**: SARSOP achieves slightly better performance on average, but QMDP is more consistent and predictable (lower CV). The overlapping confidence intervals suggest the difference may not be statistically significant.

## Future Enhancements

Potential improvements:
- Add statistical hypothesis tests (t-test, Mann-Whitney U)
- Parallel simulation execution for faster runtime
- Adaptive number of simulations based on variance
- Cross-solver comparison plots and tables
