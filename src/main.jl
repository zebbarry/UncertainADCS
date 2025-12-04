using UncertainADCS
using QMDP
using SARSOP
using BasicPOMCP
using Random
using PointBasedValueIteration
using FIB
using Statistics

include("policy_utils.jl")

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
qmdp_solver = QMDPSolver(max_iterations=1000, verbose=true)
policy = get_policy("QMDP", qmdp_solver, pomdp)

# Solve with SARSOP (slower, more accurate)
# sarsop_solver = SARSOPSolver(precision=1e-3, verbose=true)
# policy = get_policy("SARSOP", sarsop_solver, pomdp)

# # Online solver: POMCP
# pomcp_solver = POMCPSolver(
#     tree_queries=1000,
#     c=10.0,
#     max_depth=50,
#     rng=MersenneTwister(1)
# )
# policy = get_policy("POMCP", pomcp_solver, pomdp)


############################################# ADDING PBVI and FIB and running solvers together: #############################################

# pbvi_solver = PBVISolver(max_iter=50, verbose=true)
# policy_pbvi = get_policy("PBVI", pbvi_solver, pomdp)

# fib_solver = FIBSolver(max_iter=50)
# policy_fib = get_policy("FIB", fib_solver, pomdp)

# TO USE ALL SOLVERS TOGETHER:

# solvers = Dict(
#     "QMDP"   => QMDPSolver(max_iterations=1000, verbose=true),
#     "SARSOP" => SARSOPSolver(precision=1e-3, verbose=true),
#     "PBVI"   => PBVISolver(max_iter=50, verbose=true),
#     "FIB"    => FIBSolver(max_iter=50),
#     "POMCP"  => POMCPSolver(tree_queries=1000, c=10.0, max_depth=50, rng=MersenneTwister(1))
# )

# policies = Dict()

# for (name, solver) in solvers
#     policies[name] = get_policy(name, solver, pomdp)
# end

############################################# END OF [ADDING PBVI and FIB and running solvers together] #############################################


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
hist = run_simulation(pomdp, policy, 200)

# Save results using the new plotting function
save_simulation_results(hist, "QMDP", pomdp)

############################################# IF WE USE/RUN ALL SOLVERS TOGETHER: #############################################

# histories = Dict()

# for (name, policy) in policies
#     println("\n=== Simulating $name policy ===")
#     histories[name] = run_simulation(pomdp, policy, 200)
# end

# COMPARE SOLVERS:
# Save results for each solver and compare metrics

# stats = Dict()
# for (name, hist) in histories
#     stats[name] = save_simulation_results(hist, name, pomdp)
# end

# # Print comparison
# println("\n=== SOLVER COMPARISON ===")
# println("\nTotal Reward:")
# for (name, stat) in stats
#     println("  $name: $(round(stat["total_reward"], digits=2))")
# end

# println("\nFinal RMS Error (°):")
# for (name, stat) in stats
#     println("  $name: $(round(stat["final_rms_error"], digits=2))")
# end

# println("\nTotal Control Effort:")
# for (name, stat) in stats
#     println("  $name: $(round(stat["total_control_effort"], digits=2))")
# end

# println("\nAverage Belief Entropy:")
# for (name, stat) in stats
#     println("  $name: $(round(stat["avg_belief_entropy"], digits=3))")
# end

############################################# END OF [IF WE USE/RUN ALL SOLVERS TOGETHER] #############################################