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

# References

 Sunberg, Z. N., & Kochenderfer, M. J. (2018). Online algorithms for POMDPs with continuous state, action, and observation spaces. _Proceedings of the International Conference on Automated Planning and Scheduling_, 28, 259-263.
