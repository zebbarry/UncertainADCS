module UncertainADCS

using POMDPs
using POMDPTools
using Distributions
using Parameters


export SpacecraftPOMDP, SpacecraftObs, SpacecraftState
export save_simulation_results, extract_health_beliefs

include("plotting.jl")


struct SpacecraftState
    θ::Float64          # angle
    ω::Float64          # angular velocity
    health::Symbol      # health state
    θ_target_idx::Int   # current target angle index
end

struct SpacecraftObs
    θ_error_obs::Float64  # observed angular error (θ - θ_target)
    ω_obs::Float64        # observed angular velocity
end


@with_kw struct SpacecraftPOMDP <: POMDP{SpacecraftState,Int,SpacecraftObs}
    # Physical parameters
    J_sc::Float64 = 100.0          # spacecraft inertia (kg⋅m²)
    u_max::Float64 = 4.0           # max torque (N⋅m)
    dt::Float64 = 1.0              # timestep (s)

    # State bounds
    θ_max::Float64 = π
    ω_max::Float64 = 18 * (π / 180)         # rad/s

    # Discretization (reduced for faster SARSOP/PBVI solving)
    n_θ::Int = 36                  # number of angle bins (~10° resolution)
    n_ω::Int = 20                  # number of angular velocity bins

    # Actions: discretized torque levels
    actions::Vector{Float64} = [-u_max, -u_max / 2, 0.0, u_max / 2, u_max]

    # Target angles
    target_angles::Vector{Float64} = [-π / 2, π / 2]
    target_switch_period::Float64 = 50.0   # seconds

    # Health states
    health_states::Vector{Symbol} = [:healthy, :degraded, :critical, :failed]
    health_efficiency::Dict{Symbol,Float64} = Dict(
        :healthy => 1.0,
        :degraded => 0.7,
        :critical => 0.5,
        :failed => 0.0
    )

    # Degradation parameters
    k1::Float64 = 0.0             # natural degradation rate
    k2::Float64 = 0.05            # usage-dependent degradation

    # Observation noise
    σ_θ::Float64 = 0.05            # angle measurement noise (rad)
    σ_ω::Float64 = 0.01            # angular velocity noise (rad/s)

    # Reward weights
    w_θ::Float64 = 10.0           # attitude error weight
    w_ω::Float64 = 0.1            # angular velocity weight
    w_u::Float64 = 0.0            # control effort weight
    w_fail::Float64 = 10000.0      # failure penalty
    w_success::Float64 = 500.0    # success reward

    # success bounds
    θ_success_bound::Float64 = 10
    ω_success_bound::Float64 = 5

    # Discount factor
    discount::Float64 = 0.95
end

# Helper function to discretize continuous states
function discretize_continuous(var::Float64, var_max::Float64, n_var::Int)
    idx = clamp(round(Int, (var + var_max) / (2 * var_max) * (n_var - 1)) + 1, 1, n_var)
    return idx
end

function continuous_from_discrete(var_disc::Int, var_max::Float64, n_var::Int)
    var = (var_disc - 1) / (n_var - 1) * 2 * var_max - var_max
    return var
end

function discretize_continuous(pomdp::SpacecraftPOMDP, θ::Float64, ω::Float64)
    θ_disc = discretize_continuous(θ, pomdp.θ_max, pomdp.n_θ)
    ω_disc = discretize_continuous(ω, pomdp.ω_max, pomdp.n_ω)
    return θ_disc, ω_disc
end

function continuous_from_discrete(pomdp::SpacecraftPOMDP, θ_disc::Int, ω_disc::Int)
    θ = continuous_from_discrete(θ_disc, pomdp.θ_max, pomdp.n_θ)
    ω = continuous_from_discrete(ω_disc, pomdp.ω_max, pomdp.n_ω)
    return θ, ω
end

# State space
POMDPs.states(pomdp::SpacecraftPOMDP) = [
    SpacecraftState(θ, ω, h, θ_target_idx)
    for θ_idx in 1:pomdp.n_θ
    for ω_idx in 1:pomdp.n_ω
    for h in pomdp.health_states
    for θ_target_idx in 1:length(pomdp.target_angles)
    for (θ, ω) in [continuous_from_discrete(pomdp, θ_idx, ω_idx)]
]

POMDPs.stateindex(pomdp::SpacecraftPOMDP, s::SpacecraftState) = begin
    θ_idx, ω_idx = discretize_continuous(pomdp, s.θ, s.ω)
    h_idx = findfirst(==(s.health), pomdp.health_states)

    # Linear indexing
    idx = θ_idx +
          (ω_idx - 1) * pomdp.n_θ +
          (h_idx - 1) * pomdp.n_θ * pomdp.n_ω +
          (s.θ_target_idx - 1) * pomdp.n_θ * pomdp.n_ω * length(pomdp.health_states)
    return idx
end

# Initial state
function POMDPs.initialstate(pomdp::SpacecraftPOMDP)
    # Continuous coordinates
    θ_init = 1.0
    ω_init = 0.0
    θ_target_idx = 1

    # 1. Discretize the coordinates to find the nearest grid bin index
    θ_idx, ω_idx = discretize_continuous(pomdp, θ_init, ω_init)

    # 2. Convert the discrete indices back to the center of the grid cell
    θ_disc, ω_disc = continuous_from_discrete(pomdp, θ_idx, ω_idx)

    # 3. Create the initial state using the grid center
    s0 = SpacecraftState(θ_disc, ω_disc, :healthy, θ_target_idx)

    # Return the SparseCat over this valid, discretized state
    return SparseCat([s0], [1.0])
end

# Action space
POMDPs.actions(pomdp::SpacecraftPOMDP) = 1:length(pomdp.actions)
POMDPs.actionindex(pomdp::SpacecraftPOMDP, a::Int) = a

# Observation space - discretize observations similarly to states
function POMDPs.observations(pomdp::SpacecraftPOMDP)
    obs = SpacecraftObs[]
    # Order matters - must match obsindex
    for ω_idx in 1:pomdp.n_ω
        for θ_error_idx in 1:pomdp.n_θ
            # Use same discretization for error as for angle
            θ_error, ω = continuous_from_discrete(pomdp, θ_error_idx, ω_idx)
            push!(obs, SpacecraftObs(θ_error, ω))
        end
    end
    return obs
end

function POMDPs.obsindex(pomdp::SpacecraftPOMDP, o::SpacecraftObs)
    # Discretize the observation to nearest grid point
    θ_error_idx, ω_idx = discretize_continuous(pomdp, o.θ_error_obs, o.ω_obs)

    # Linear index: observations are indexed by (θ_error_idx, ω_idx)
    idx = θ_error_idx + (ω_idx - 1) * pomdp.n_θ
    return idx
end


POMDPs.discount(pomdp::SpacecraftPOMDP) = pomdp.discount


function POMDPs.transition(pomdp::SpacecraftPOMDP, s::SpacecraftState, a::Int)
    # Get commanded torque
    u = pomdp.actions[a]

    # Get actual torque based on health
    η = pomdp.health_efficiency[s.health]
    τ_actual = η * u

    # 1. Calculate next continuous state
    θ_cont = s.θ + s.ω * pomdp.dt
    ω_cont = s.ω + (τ_actual / pomdp.J_sc) * pomdp.dt

    # Wrap angle to [-π, π] and Clamp angular velocity to bounds
    θ_cont = mod(θ_cont + π, 2π) - π
    ω_cont = clamp(ω_cont, -pomdp.ω_max, pomdp.ω_max)

    # 2. Map the continuous result to the center of the nearest discrete grid cell
    # This guarantees the resulting state is valid for the discrete state space.
    θ_idx, ω_idx = discretize_continuous(pomdp, θ_cont, ω_cont)
    θ_disc, ω_disc = continuous_from_discrete(pomdp, θ_idx, ω_idx)

    # Goal Transition Parameters
    # FIXME: Make these defined on the pomdp struct
    # If the sc is pointing near the current target angle (< 8º) and its ang velocity is small (< 8 º/sec)
    # Then allow the spacecraft to possibly switch its target index to the next one
    if abs(rad2deg(s.θ - pomdp.target_angles[s.θ_target_idx])) < pomdp.θ_success_bound && abs(rad2deg(s.ω)) < pomdp.ω_success_bound
        p_switch = 1.0 / (pomdp.target_switch_period * pomdp.dt)
    else
        p_switch = 0.0
    end
    target_count = length(pomdp.target_angles)

    # Calculate the next index in the cycle
    θ_target_idx_stay = s.θ_target_idx
    θ_target_idx_switch = mod1(s.θ_target_idx + 1, target_count) # Modulo arithmetic ensures cycling (1 -> 2 -> ... -> N -> 1)

    # Health degradation transitions
    u_normalized = abs(u) / pomdp.u_max
    p_degrade = pomdp.k1 + pomdp.k2 * u_normalized

    next_states = SpacecraftState[]
    probs = Float64[]

    # Health probabilities (P(H_new | H_old, a))
    p_stay_health = 1.0 - p_degrade
    p_down_health = p_degrade

    h_stay = s.health
    if s.health == :failed
        p_stay_health = 1.0
        p_down_health = 0.0
        h_down = :failed  # Stays failed
    else
        h_idx = findfirst(==(s.health), pomdp.health_states)
        h_down = pomdp.health_states[h_idx+1]
    end

    next_states = SpacecraftState[]
    probs = Float64[]

    # 3. Create next states using the grid-centered (θ_disc, ω_disc)
    # --- 1. Goal Stays (Probability = 1.0 - p_switch) ---
    prob_base_stay = 1.0 - p_switch

    #   Goal Stays, Health Stays
    push!(next_states, SpacecraftState(θ_disc, ω_disc, h_stay, θ_target_idx_stay))
    push!(probs, p_stay_health * prob_base_stay)

    #   Goal Stays, Health Degrades (only if possible)
    if s.health != :failed
        push!(next_states, SpacecraftState(θ_disc, ω_disc, h_down, θ_target_idx_stay))
        push!(probs, p_down_health * prob_base_stay)
    end

    # --- 2. Goal Switches (Probability = p_switch) ---
    prob_base_switch = p_switch

    #   Goal Switches, Health Stays
    push!(next_states, SpacecraftState(θ_disc, ω_disc, h_stay, θ_target_idx_switch))
    push!(probs, p_stay_health * prob_base_switch)

    #   Goal Switches, Health Degrades (only if possible)
    if s.health != :failed
        push!(next_states, SpacecraftState(θ_disc, ω_disc, h_down, θ_target_idx_switch))
        push!(probs, p_down_health * prob_base_switch)
    end

    # Ensure probabilities sum to 1.0 (they should, but a final check is safe)
    if !isapprox(sum(probs), 1.0)
        @warn "Transition probabilities do not sum to 1.0 for state $(s) and action $(a). Sum: $(sum(probs))"
    end

    return SparseCat(next_states, probs)
end


function POMDPs.observation(pomdp::SpacecraftPOMDP, a::Int, sp::SpacecraftState)
    # Calculate angular error
    θ_error = sp.θ - pomdp.target_angles[sp.θ_target_idx]
    # Calculate angular error and wrap to [-π, π]
    θ_error = sp.θ - pomdp.target_angles[sp.θ_target_idx]
    θ_error = mod(θ_error + pomdp.θ_max, 2π) - pomdp.θ_max  # Wrap to [-π, π]


    # Gaussian noise on angular error and angular velocity
    θ_error_dist = Normal(θ_error, pomdp.σ_θ)
    ω_dist = Normal(sp.ω, pomdp.σ_ω)

    # OPTIMIZATION: Only compute probabilities for nearby observations
    # Window size: ±4 bins captures >99.9% of probability mass (>4σ for both dimensions)
    window_size = 4

    # Discretization bin widths
    θ_error_width = 2 * pomdp.θ_max / pomdp.n_θ
    ω_width = 2 * pomdp.ω_max / pomdp.n_ω

    # Find the center observation index
    θ_error_idx_center, ω_idx_center = discretize_continuous(pomdp, θ_error, sp.ω)

    # Compute bounds for the window
    θ_error_idx_min = max(1, θ_error_idx_center - window_size)
    θ_error_idx_max = min(pomdp.n_θ, θ_error_idx_center + window_size)
    ω_idx_min = max(1, ω_idx_center - window_size)
    ω_idx_max = min(pomdp.n_ω, ω_idx_center + window_size)

    # Collect nearby observations and their probabilities
    obs_list = SpacecraftObs[]
    probs = Float64[]

    for ω_idx in ω_idx_min:ω_idx_max
        for θ_error_idx in θ_error_idx_min:θ_error_idx_max
            # Get the observation at this discrete index
            θ_error_val, ω_val = continuous_from_discrete(pomdp, θ_error_idx, ω_idx)
            o = SpacecraftObs(θ_error_val, ω_val)

            # Compute probability that the noisy measurement falls inside observation bin
            p_θ_error = cdf(θ_error_dist, θ_error_val + θ_error_width / 2) -
                        cdf(θ_error_dist, θ_error_val - θ_error_width / 2)
            p_ω = cdf(ω_dist, ω_val + ω_width / 2) - cdf(ω_dist, ω_val - ω_width / 2)

            p_total = p_θ_error * p_ω

            if p_total > 0.0
                push!(obs_list, o)
                push!(probs, p_total)
            end
        end
    end

    # Normalize
    total = sum(probs)
    if total > 0.0
        probs ./= total
    else
        # Fallback: uniform distribution over a single observation at the center
        @warn "Observation probabilities are <= 0 for state $(sp) and action $(a)."
        θ_error_center, ω_center = continuous_from_discrete(pomdp, θ_error_idx_center, ω_idx_center)
        obs_list = [SpacecraftObs(θ_error_center, ω_center)]
        probs = [1.0]
    end

    return SparseCat(obs_list, probs)
end

# 2-argument reward: expected reward over next states
function POMDPs.reward(pomdp::SpacecraftPOMDP, s::SpacecraftState, a::Int)
    # Compute expected reward over transition distribution
    trans = transition(pomdp, s, a)
    expected_r = 0.0

    for (sp, p) in POMDPTools.weighted_iterator(trans)
        expected_r += p * reward(pomdp, s, a, sp)
    end

    return expected_r
end

# 3-argument reward: immediate reward for s-a-sp triple
function POMDPs.reward(pomdp::SpacecraftPOMDP, s::SpacecraftState, a::Int, sp::SpacecraftState)
    u = pomdp.actions[a]

    # Calculate attitude error relative to the target
    θ_error = s.θ - pomdp.target_angles[s.θ_target_idx]

    # Quadratic costs
    r = -pomdp.w_θ * θ_error^2 -
        pomdp.w_ω * s.ω^2 -
        pomdp.w_u * u^2

    if abs(rad2deg(θ_error)) < pomdp.θ_success_bound && abs(rad2deg(s.ω)) < pomdp.ω_success_bound
        r += pomdp.w_success  # Reward for being within target zone
    end

    # Failure penalty
    if sp.health == :failed
        r -= pomdp.w_fail
    end

    return r
end

POMDPs.isterminal(pomdp::SpacecraftPOMDP, s::SpacecraftState) = s.health == :failed

end