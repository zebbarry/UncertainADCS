using UncertainADCS
using QMDP
using SARSOP
using BasicPOMCP

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
qmdp_policy = solve(qmdp_solver, pomdp)

# Solve with SARSOP (slower, more accurate)
# println("Solving with SARSOP...")
# sarsop_solver = SARSOPSolver(precision=1e-3, verbose=true)
# sarsop_policy = solve(sarsop_solver, pomdp)

# # Online solver: POMCP
# println("Setting up POMCP...")
# pomcp_solver = POMCPSolver(
#     tree_queries=1000,
#     c=10.0,
#     max_depth=50,
#     rng=MersenneTwister(1)
# )
# pomcp_policy = solve(pomcp_solver, pomdp)


# Run simulation
using POMDPTools
function run_simulation(pomdp, policy, num_steps=100)
    sim = HistoryRecorder(max_steps=num_steps)
    hist = simulate(sim, pomdp, policy)

    return hist
end

# Evaluate policy
# Run simulation
hist = run_simulation(pomdp, qmdp_policy)

# Extract data using accessor functions
states = collect(state_hist(hist))
actions = collect(action_hist(hist))
observations = collect(observation_hist(hist))
rewards = collect(reward_hist(hist))
beliefs = collect(belief_hist(hist))

# Now you can plot
θ_hist = [s.θ for s in states]
ω_hist = [s.ω for s in states]
health_hist = [s.health for s in states]
u_hist = [pomdp.actions[a] for a in actions]

# Plot
using Plots

p1 = plot(θ_hist, label="Angle θ", xlabel="Time step", ylabel="Angle (rad)")
p2 = plot(ω_hist, label="Angular velocity ω", xlabel="Time step", ylabel="ω (rad/s)")
p3 = plot(u_hist, label="Control torque", xlabel="Time step", ylabel="Torque (N⋅m)")
p4 = plot([string(h) for h in health_hist], label="Health state", xlabel="Time step", ylabel="Health", legend=false)

p = plot(p1, p2, p3, p4, layout=(4, 1), size=(800, 800))

savefig(p, "./spacecraft_attitude_control_simulation.png")

# Diagnostic printout
println("\n=== DIAGNOSTICS ===")
println("Number of steps: ", length(states))
println("Initial state: θ=$(states[1].θ), ω=$(states[1].ω), health=$(states[1].health)")
println("Final state: θ=$(states[end].θ), ω=$(states[end].ω), health=$(states[end].health)")
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