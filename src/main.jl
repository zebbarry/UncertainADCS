using UncertainADCS
using QMDP
using SARSOP
using BasicPOMCP
using Random
using PointBasedValueIteration
using FIB
using Statistics


using POMDPTools
function run_simulation(pomdp, policy, num_steps=100)
    println("Running simulation for $num_steps steps...")
    sim = HistoryRecorder(max_steps=num_steps)
    hist = simulate(sim, pomdp, policy)

    return hist
end

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
# println("Solving with QMDP...")
# qmdp_solver = QMDPSolver(max_iterations=1000, verbose=true)
# policy = solve(qmdp_solver, pomdp)

# Solve with SARSOP (slower, more accurate)
# println("Solving with SARSOP...")
# sarsop_solver = SARSOPSolver(precision=1e-2, verbose=true, timeout=300.0)  # Relaxed precision for faster solving
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


############################################# ADDING PBVI and FIB and running solvers together: #############################################

# println("Solving with PBVI...")
# pbvi_solver = PBVISolver(max_iter=30, verbose=true)  # Reduced iterations for faster solving
# policy_pbvi = solve(pbvi_solver, pomdp)

# println("Solving with FIB...")
# fib_solver = FIBSolver(max_iter=30)  # Reduced iterations for faster solving
# policy_fib = solve(fib_solver, pomdp)

# TO USE ALL SOLVERS TOGETHER:

solvers = Dict(
    "QMDP" => QMDPSolver(max_iterations=1000, verbose=true),
    # "PBVI" => PBVISolver(max_iterations=30, verbose=true),
    # "FIB" => FIBSolver(max_iterations=30, verbose=true),
    "SARSOP" => SARSOPSolver(precision=1e-2, verbose=true, timeout=300),
    "POMCP" => POMCPSolver(tree_queries=1000, c=10.0, max_depth=30, rng=MersenneTwister(1))
)

policies = Dict()
histories = Dict()

for (name, solver) in solvers
    println("\n=== Solving with $name ===")
    solve_time = @elapsed policies[name] = solve(solver, pomdp)
    println("\n=== Simulating $name policy ===")
    sim_time = @elapsed histories[name] = run_simulation(pomdp, policies[name], 5000)
    save_simulation_results(histories[name], name, pomdp, solve_time, sim_time)
end

############################################# END OF [ADDING PBVI and FIB and running solvers together] #############################################


# Run simulation

# Evaluate policy
# Run simulation

# Save results using the new plotting function

############################################# IF WE USE/RUN ALL SOLVERS TOGETHER: #############################################


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