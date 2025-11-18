using UncertainADCS
using QMDP
using SARSOP
using BasicPOMCP
using Random

# Create the POMDP
pomdp = SpacecraftPOMDP()

# # Test rewards at the initial state
# s_test = SpacecraftState(1.0, 0.0, :healthy)
# println("\nReward tests at θ=1.0:")
# for (i, a) in enumerate(1:5)
#     r = reward(pomdp, s_test, a)
#     println("  Action $a (u=$(pomdp.actions[a]): reward = $r")
# end

# # Also test at target
# s_target = SpacecraftState(0.0, 0.0, :healthy)
# println("\nReward tests at θ=0.0 (target):")
# for (i, a) in enumerate(1:5)
#     r = reward(pomdp, s_target, a)
#     println("  Action $a (u=$(pomdp.actions[a]): reward = $r")
# end

# Solve with QMDP (fast, approximate)
println("Solving with QMDP...")
qmdp_solver = QMDPSolver(max_iterations=1000, verbose=true)
policy = solve(qmdp_solver, pomdp)

# Solve with SARSOP (slower, more accurate)
# println("Solving with SARSOP...")
# sarsop_solver = SARSOPSolver(precision=1e-3, verbose=true)
# policy = solve(sarsop_solver, pomdp)

# # Online solver: POMCP
# println("Setting up POMCP...")
# pomcp_solver = POMCPSolver(
#     tree_queries=1000,
#     c=10.0,
#     max_depth=50,
#     rng=MersenneTwister(1)
# )
# policy = solve(pomcp_solver, pomdp)


# Run simulation
using POMDPTools
function run_simulation(pomdp, policy, num_steps=100)
    println("Running simulation for $num_steps steps...")
    sim = HistoryRecorder(max_steps=num_steps)
    hist = simulate(sim, pomdp, policy)

    return hist
end

# Evaluate policy
# Run simulation
hist = run_simulation(pomdp, policy, 2000)

# Extract data using accessor functions
states = collect(state_hist(hist))
actions = collect(action_hist(hist))
observations = collect(observation_hist(hist))
rewards = collect(reward_hist(hist))
beliefs = collect(belief_hist(hist))

# Now you can plot
θ_hist = [rad2deg(s.θ) for s in states]
ω_hist = [rad2deg(s.ω) for s in states]
health_hist = [s.health for s in states]
u_hist = [pomdp.actions[a] for a in actions]
θ_target_hist = [rad2deg(pomdp.target_angles[s.θ_target_idx]) for s in states]

# Diagnostic printout
println("\n=== DIAGNOSTICS ===")
println("Number of steps: ", length(states))
println("Initial state: θ=$(θ_hist[1]), ω=$(ω_hist[1]), health=$(health_hist[1])")
println("Final state: θ=$(θ_hist[end]), ω=$(ω_hist[end]), health=$(health_hist[end])")
println("\nActions taken: ", unique(actions))
println("Action distribution: ")
for a in unique(actions)
    count = sum(actions .== a)
    println("  Action $a (u=$(pomdp.actions[a])): $count times")
end
# println("\nFirst 10 states:")
# for i in 1:min(10, length(states))
#     println("  t=$i: θ=$(states[i].θ), ω=$(states[i].ω), u=$(pomdp.actions[actions[i]])")
# end

# Plot
using Plots

p1 = plot(θ_hist, label="Angle θ", xlabel="Time step", ylabel="Angle (°)")
plot!(p1, θ_target_hist, label="Target θ", linestyle=:dash, color=:red, linewidth=2)
p2 = plot(ω_hist, label="Angular velocity ω", xlabel="Time step", ylabel="ω (°/s)")
p3 = plot(u_hist, label="Control torque", xlabel="Time step", ylabel="Torque (N⋅m)")
p4 = plot([string(h) for h in health_hist], label="Health state", xlabel="Time step", ylabel="Health", legend=false)

p = plot(p1, p2, p3, p4, layout=(4, 1), size=(800, 800))

savefig(p, "spacecraft_attitude_control_simulation.png")

# Extract health belief evolution
using POMDPTools: weighted_iterator

function extract_health_beliefs(beliefs, pomdp)
    n_steps = length(beliefs)
    health_probs = zeros(n_steps, length(pomdp.health_states))

    for (t, b) in enumerate(beliefs)
        for (s, p) in weighted_iterator(b)
            h_idx = findfirst(==(s.health), pomdp.health_states)
            health_probs[t, h_idx] += p
        end
    end

    return health_probs
end

health_beliefs = extract_health_beliefs(beliefs, pomdp)

# Plot stacked area chart of health beliefs
p_belief = areaplot(
    1:size(health_beliefs, 1),
    health_beliefs,
    labels=["Healthy" "Degraded" "Critical" "Failed"],
    xlabel="Time step",
    ylabel="Belief Probability",
    title="Actuator Health Belief Evolution",
    legend=:right,
    fillalpha=0.7
)
savefig(p_belief, "health_belief_evolution.png")

# Phase portrait
health_colors = Dict(
    :healthy => :green,
    :degraded => :yellow,
    :critical => :orange,
    :failed => :red
)

colors = [health_colors[h] for h in health_hist]

p_phase = scatter(
    θ_hist .- θ_target_hist, ω_hist,
    color=colors,
    marker_z=1:length(θ_hist),
    xlabel="Angle Error θ (°)",
    ylabel="Angular velocity ω (°/s)",
    title="Phase Portrait (State Space Trajectory)",
    label="",
    colorbar=true,
    colorbar_title="Time step",
    markersize=3
)
# Add target point
scatter!([0.0], [0.0], color=:blue, marker=:star, markersize=10, label="Target")
savefig(p_phase, "phase_portrait.png")

# Cumulative reward
cumulative_reward = cumsum(rewards)

# RMS error over time
rms_error = sqrt.(cumsum((θ_hist .- θ_target_hist) .^ 2) ./ (1:length(θ_hist)))

# Control effort
cumulative_control = cumsum(abs.(u_hist))

p_metrics = plot(layout=(3, 1), size=(800, 600))
plot!(p_metrics[1], cumulative_reward, xlabel="Time step", ylabel="Cumulative Reward", label="", title="Performance Over Time")
plot!(p_metrics[2], rms_error, xlabel="Time step", ylabel="RMS Error (°)", label="")
plot!(p_metrics[3], cumulative_control, xlabel="Time step", ylabel="Cumulative |u|", label="")

savefig(p_metrics, "cumulative_metrics.png")

# # Compute dominant health belief at each timestep
# dominant_health = [pomdp.health_states[argmax(health_beliefs[t, :])] for t in 1:size(health_beliefs, 1)]

# # Plot action vs dominant belief
# p_action_health = scatter(
#     1:length(u_hist),
#     u_hist,
#     group=dominant_health,
#     xlabel="Time step",
#     ylabel="Control Torque (N⋅m)",
#     title="Control Actions vs. Believed Health State",
#     legend=:right,
#     markersize=4,
#     alpha=0.6
# )
# savefig(p_action_health, "./action_vs_health_belief.png")