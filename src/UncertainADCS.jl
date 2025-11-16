module UncertainADCS

using POMDPs
using POMDPTools
using Distributions
using Parameters


export SpacecraftPOMDP, SpacecraftObs, SpacecraftState


struct SpacecraftState
    θ::Float64          # angle
    ω::Float64          # angular velocity
    health::Symbol      # health state
end

struct SpacecraftObs
    θ_obs::Float64      # observed angle
    ω_obs::Float64      # observed angular velocity
end


@with_kw struct SpacecraftPOMDP <: POMDP{SpacecraftState,Int,SpacecraftObs}
    # Physical parameters
    J_sc::Float64 = 100.0          # spacecraft inertia (kg⋅m²)
    u_max::Float64 = 5.0           # max torque (N⋅m)
    dt::Float64 = 1.0              # timestep (s)

    # State bounds
    θ_max::Float64 = π
    ω_max::Float64 = 0.5           # rad/s

    # Discretization
    n_θ::Int = 21                  # number of angle bins
    n_ω::Int = 13                  # number of angular velocity bins

    # Actions: discretized torque levels
    actions::Vector{Float64} = [-u_max, -u_max / 2, 0.0, u_max / 2, u_max]

    # Health states
    health_states::Vector{Symbol} = [:healthy, :degraded, :critical, :failed]
    health_efficiency::Dict{Symbol,Float64} = Dict(
        :healthy => 1.0,
        :degraded => 0.7,
        :critical => 0.4,
        :failed => 0.0
    )

    # Degradation parameters
    k1::Float64 = 0.001            # natural degradation rate
    k2::Float64 = 0.0005           # usage-dependent degradation

    # Observation noise
    σ_θ::Float64 = 0.05            # angle measurement noise (rad)
    σ_ω::Float64 = 0.01            # angular velocity noise (rad/s)

    # Reward weights
    w_θ::Float64 = 100.0           # attitude error weight
    w_ω::Float64 = 10.0            # angular velocity weight
    w_u::Float64 = 0.0             # control effort weight
    w_fail::Float64 = 1000.0       # failure penalty

    # Discount factor
    discount::Float64 = 0.95
end

# Helper function to discretize continuous states
function discretize_continuous(pomdp::SpacecraftPOMDP, θ::Float64, ω::Float64)
    θ_disc = clamp(round(Int, (θ + pomdp.θ_max) / (2 * pomdp.θ_max) * (pomdp.n_θ - 1)) + 1, 1, pomdp.n_θ)
    ω_disc = clamp(round(Int, (ω + pomdp.ω_max) / (2 * pomdp.ω_max) * (pomdp.n_ω - 1)) + 1, 1, pomdp.n_ω)
    return θ_disc, ω_disc
end

function continuous_from_discrete(pomdp::SpacecraftPOMDP, θ_disc::Int, ω_disc::Int)
    θ = (θ_disc - 1) / (pomdp.n_θ - 1) * 2 * pomdp.θ_max - pomdp.θ_max
    ω = (ω_disc - 1) / (pomdp.n_ω - 1) * 2 * pomdp.ω_max - pomdp.ω_max
    return θ, ω
end

# State space
POMDPs.states(pomdp::SpacecraftPOMDP) = [
    SpacecraftState(θ, ω, h)
    for θ_idx in 1:pomdp.n_θ
    for ω_idx in 1:pomdp.n_ω
    for h in pomdp.health_states
    for (θ, ω) in [continuous_from_discrete(pomdp, θ_idx, ω_idx)]
]

POMDPs.stateindex(pomdp::SpacecraftPOMDP, s::SpacecraftState) = begin
    θ_idx, ω_idx = discretize_continuous(pomdp, s.θ, s.ω)
    h_idx = findfirst(==(s.health), pomdp.health_states)

    # Linear indexing (no ω_w anymore!)
    idx = θ_idx +
          (ω_idx - 1) * pomdp.n_θ +
          (h_idx - 1) * pomdp.n_θ * pomdp.n_ω
    return idx
end

# Initial state
function POMDPs.initialstate(pomdp::SpacecraftPOMDP)
    # Target continuous coordinates
    θ_target = 1.0
    ω_target = 0.5

    # 1. Discretize the target coordinates to find the nearest grid bin index
    θ_idx, ω_idx = discretize_continuous(pomdp, θ_target, ω_target)

    # 2. Convert the discrete indices back to the center of the grid cell
    θ_disc, ω_disc = continuous_from_discrete(pomdp, θ_idx, ω_idx)

    # 3. Create the initial state using the grid center
    s0 = SpacecraftState(θ_disc, ω_disc, :healthy)

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
        for θ_idx in 1:pomdp.n_θ
            θ = (θ_idx - 1) / (pomdp.n_θ - 1) * 2 * pomdp.θ_max - pomdp.θ_max
            ω = (ω_idx - 1) / (pomdp.n_ω - 1) * 2 * pomdp.ω_max - pomdp.ω_max
            push!(obs, SpacecraftObs(θ, ω))
        end
    end
    return obs
end

function POMDPs.obsindex(pomdp::SpacecraftPOMDP, o::SpacecraftObs)
    # Discretize the observation to nearest grid point
    θ_idx = clamp(round(Int, (o.θ_obs + pomdp.θ_max) / (2 * pomdp.θ_max) * (pomdp.n_θ - 1)) + 1, 1, pomdp.n_θ)
    ω_idx = clamp(round(Int, (o.ω_obs + pomdp.ω_max) / (2 * pomdp.ω_max) * (pomdp.n_ω - 1)) + 1, 1, pomdp.n_ω)

    # Linear index: observations are indexed by (θ_idx, ω_idx)
    idx = θ_idx + (ω_idx - 1) * pomdp.n_θ
    return idx
end

# Discount factor
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

    # Health degradation transitions
    u_normalized = abs(u) / pomdp.u_max
    p_degrade = pomdp.k1 + pomdp.k2 * u_normalized

    next_states = SpacecraftState[]
    probs = Float64[]

    # 3. Create next states using the grid-centered (θ_disc, ω_disc)
    if s.health == :healthy
        push!(next_states, SpacecraftState(θ_disc, ω_disc, :healthy))
        push!(probs, 1.0 - p_degrade)
        push!(next_states, SpacecraftState(θ_disc, ω_disc, :degraded))
        push!(probs, p_degrade)

    elseif s.health == :degraded
        push!(next_states, SpacecraftState(θ_disc, ω_disc, :degraded))
        push!(probs, 1.0 - p_degrade)
        push!(next_states, SpacecraftState(θ_disc, ω_disc, :critical))
        push!(probs, p_degrade)

    elseif s.health == :critical
        push!(next_states, SpacecraftState(θ_disc, ω_disc, :critical))
        push!(probs, 1.0 - p_degrade)
        push!(next_states, SpacecraftState(θ_disc, ω_disc, :failed))
        push!(probs, p_degrade)

    else # :failed
        push!(next_states, SpacecraftState(θ_disc, ω_disc, :failed))
        push!(probs, 1.0)
    end

    # Ensure probabilities sum to 1.0 (they should, but a final check is safe)
    if !isapprox(sum(probs), 1.0)
        # This branch should not be reached with the fixed logic, 
        # but it's a good practice for debugging.
        @warn "Transition probabilities do not sum to 1.0 for state $(s) and action $(a). Sum: $(sum(probs))"
        # Since this warning would cause a solver error, we must normalize:
        probs ./= sum(probs)
    end

    return SparseCat(next_states, probs)
end


function POMDPs.observation(pomdp::SpacecraftPOMDP, a::Int, sp::SpacecraftState)
    # Gaussian noise on angle and angular velocity
    θ_dist = Normal(sp.θ, pomdp.σ_θ)
    ω_dist = Normal(sp.ω, pomdp.σ_ω)

    # For discrete observations, discretize the space
    obs_list = observations(pomdp)
    probs = zeros(length(obs_list))

    for (i, o) in enumerate(obs_list)
        # Probability is product of independent Gaussians
        # Use discretization bins
        θ_width = 2 * pomdp.θ_max / pomdp.n_θ
        ω_width = 2 * pomdp.ω_max / pomdp.n_ω

        p_θ = cdf(θ_dist, o.θ_obs + θ_width / 2) - cdf(θ_dist, o.θ_obs - θ_width / 2)
        p_ω = cdf(ω_dist, o.ω_obs + ω_width / 2) - cdf(ω_dist, o.ω_obs - ω_width / 2)

        probs[i] = p_θ * p_ω
    end

    # Normalize
    probs ./= sum(probs)

    return SparseCat(obs_list, probs)
end

function POMDPs.reward(pomdp::SpacecraftPOMDP, s::SpacecraftState, a::Int)
    u = pomdp.actions[a]

    # Quadratic costs
    r = -pomdp.w_θ * s.θ^2 -
        pomdp.w_ω * s.ω^2 -
        pomdp.w_u * u^2

    # Failure penalty
    if s.health == :failed
        r -= pomdp.w_fail
    end

    return r
end

POMDPs.isterminal(pomdp::SpacecraftPOMDP, s::SpacecraftState) = false

end