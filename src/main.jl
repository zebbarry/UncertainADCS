using UncertainADCS
using QMDP
using SARSOP
using BasicPOMCP
using Random
using PointBasedValueIteration
using FIB
using Statistics


using POMDPTools
function run_simulation(pomdp, policy, num_steps=100, rng=Random.GLOBAL_RNG)
    println("Running simulation for $num_steps steps...")
    sim = HistoryRecorder(max_steps=num_steps, rng=rng)
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

# Configuration
num_simulations = 45  # Number of simulation runs per solver
num_steps = 3000      # Steps per simulation
seeds = 1:num_simulations  # Use same seed sequence for all solvers

# Create solvers with RNG support
# Note: For offline solvers (QMDP, SARSOP), the RNG only affects tie-breaking during solving
#       For online solvers (POMCP), the RNG affects tree search
solvers = Dict(
    "QMDP" => QMDPSolver(max_iterations=1000, verbose=true),
    # "PBVI" => PBVISolver(max_iterations=30, verbose=true),
    # "FIB" => FIBSolver(max_iterations=30, verbose=true),
    "SARSOP" => SARSOPSolver(precision=1e-2, verbose=true, timeout=600),
    "POMCP" => POMCPSolver(tree_queries=1000, c=10.0, max_depth=30, rng=MersenneTwister(42))
)

policies = Dict()
all_histories = Dict()  # Will store Dict(solver_name => [histories...])

for (name, solver) in solvers
    println("\n" * "="^70)
    println("=== Solving with $name ===")
    println("="^70)

    # Solve once with the policy
    solve_time = @elapsed policies[name] = solve(solver, pomdp)

    println("Solving completed in $(round(solve_time, digits=2))s")

    # Run multiple simulations with different seeds
    println("\n=== Running $num_simulations simulations with $name policy ===")
    all_histories[name] = []

    simulation_times = Float64[]
    total_rewards = Float64[]

    for (sim_idx, seed) in enumerate(seeds)
        print("  Simulation $sim_idx/$num_simulations (seed=$seed)... ")

        # Create RNG for this simulation
        rng = MersenneTwister(seed)

        # Run simulation with this seed
        sim_time = @elapsed begin
            hist = run_simulation(pomdp, policies[name], num_steps, rng)
        end

        push!(all_histories[name], hist)
        push!(simulation_times, sim_time)

        # Track total reward for this simulation
        push!(total_rewards, sum(reward_hist(hist)))

        println("done ($(round(sim_time, digits=2))s)")
    end

    total_sim_time = sum(simulation_times)
    avg_sim_time = mean(simulation_times)

    println("\nTotal simulation time: $(round(total_sim_time, digits=2))s")
    println("Average simulation time: $(round(avg_sim_time, digits=2))s")

    # Save aggregated results
    save_multi_simulation_results(all_histories[name], name, pomdp, solve_time, total_sim_time)

    # Find and save the best performing simulation
    best_idx = argmax(total_rewards)
    best_reward = total_rewards[best_idx]
    println("\n=== Saving detailed results for best run (simulation $best_idx, reward=$(round(best_reward, digits=2))) ===")
    save_simulation_results(all_histories[name][best_idx], "$(name)_best", pomdp, solve_time, simulation_times[best_idx])
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