using Plots
using POMDPTools: state_hist, action_hist, observation_hist, reward_hist, belief_hist, weighted_iterator
using Statistics

"""
    save_simulation_results(hist, solver_name, pomdp; base_dir="results")

Save simulation results including plots and text statistics to a folder based on the solver name.

# Arguments
- `hist`: Simulation history from HistoryRecorder
- `solver_name::String`: Name of the solver (used for folder name and plot titles)
- `pomdp`: The POMDP problem instance
- `base_dir::String`: Base directory for saving results (default: "results")

# Output
Creates a folder `base_dir/solver_name/` containing:
- `simulation.png`: Main simulation plots (angle, velocity, control, health)
- `health_belief_evolution.png`: Stacked area chart of health belief probabilities
- `phase_portrait.png`: Phase space trajectory
- `cumulative_metrics.png`: Cumulative reward, RMS error, and control effort
- `statistics.txt`: Text file with key metrics and statistics
"""
function save_simulation_results(hist, solver_name::String, pomdp; base_dir="results")
    # Create output directory
    output_dir = joinpath(base_dir, solver_name)
    mkpath(output_dir)

    println("\n=== Processing results for $solver_name ===")
    println("Saving to: $output_dir")

    # Extract data using accessor functions
    states = collect(state_hist(hist))
    actions = collect(action_hist(hist))
    observations = collect(observation_hist(hist))
    rewards = collect(reward_hist(hist))
    beliefs = collect(belief_hist(hist))

    # Convert to plotting-friendly formats
    θ_hist = [rad2deg(s.θ) for s in states]
    ω_hist = [rad2deg(s.ω) for s in states]
    health_hist = [s.health for s in states]
    u_hist = [pomdp.actions[a] for a in actions]
    θ_target_hist = [rad2deg(pomdp.target_angles[s.θ_target_idx]) for s in states]

    # ====== Plot 1: Main simulation plots ======
    p1 = plot(θ_hist, label="Angle θ", xlabel="Time step", ylabel="Angle (°)")
    plot!(p1, θ_target_hist, label="Target θ", linestyle=:dash, color=:red, linewidth=2)
    p2 = plot(ω_hist, label="Angular velocity ω", xlabel="Time step", ylabel="ω (°/s)")
    p3 = plot(u_hist, label="Control torque", xlabel="Time step", ylabel="Torque (N⋅m)")
    p4 = plot([string(h) for h in health_hist], label="Health state", xlabel="Time step", ylabel="Health", legend=false)

    p_main = plot(p1, p2, p3, p4, layout=(4, 1), size=(800, 800))
    plot!(p_main, plot_title="$solver_name: Spacecraft Attitude Control Simulation")
    savefig(p_main, joinpath(output_dir, "simulation.png"))
    println("  ✓ Saved simulation.png")

    # ====== Plot 2: Health belief evolution ======
    health_beliefs = extract_health_beliefs(beliefs, pomdp)

    p_belief = areaplot(
        1:size(health_beliefs, 1),
        health_beliefs,
        labels=["Healthy" "Degraded" "Critical" "Failed"],
        xlabel="Time step",
        ylabel="Belief Probability",
        title="$solver_name: Actuator Health Belief Evolution",
        legend=:right,
        fillalpha=0.7
    )
    savefig(p_belief, joinpath(output_dir, "health_belief_evolution.png"))
    println("  ✓ Saved health_belief_evolution.png")

    # ====== Plot 3: Phase portrait ======
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
        title="$solver_name: Phase Portrait (State Space Trajectory)",
        label="",
        colorbar=true,
        colorbar_title="Time step",
        markersize=3
    )
    scatter!([0.0], [0.0], color=:blue, marker=:star, markersize=10, label="Target")
    savefig(p_phase, joinpath(output_dir, "phase_portrait.png"))
    println("  ✓ Saved phase_portrait.png")

    # ====== Plot 4: Cumulative metrics ======
    cumulative_reward = cumsum(rewards)
    rms_error = sqrt.(cumsum((θ_hist .- θ_target_hist) .^ 2) ./ (1:length(θ_hist)))
    cumulative_control = cumsum(abs.(u_hist))

    p_metrics = plot(layout=(3, 1), size=(800, 600))
    plot!(p_metrics[1], cumulative_reward, xlabel="Time step", ylabel="Cumulative Reward", label="", title="$solver_name: Performance Over Time")
    plot!(p_metrics[2], rms_error, xlabel="Time step", ylabel="RMS Error (°)", label="")
    plot!(p_metrics[3], cumulative_control, xlabel="Time step", ylabel="Cumulative |u|", label="")

    savefig(p_metrics, joinpath(output_dir, "cumulative_metrics.png"))
    println("  ✓ Saved cumulative_metrics.png")

    # ====== Save statistics to text file ======
    stats_file = joinpath(output_dir, "statistics.txt")
    open(stats_file, "w") do io
        println(io, "="^60)
        println(io, "Simulation Statistics for $solver_name")
        println(io, "="^60)
        println(io)

        # Basic info
        println(io, "Simulation Duration:")
        println(io, "  Number of steps: $(length(states))")
        println(io)

        # Initial and final states
        println(io, "State Information:")
        println(io, "  Initial state: θ=$(round(θ_hist[1], digits=2))°, ω=$(round(ω_hist[1], digits=2))°/s, health=$(health_hist[1])")
        println(io, "  Final state:   θ=$(round(θ_hist[end], digits=2))°, ω=$(round(ω_hist[end], digits=2))°/s, health=$(health_hist[end])")
        println(io)

        # Action distribution
        println(io, "Action Distribution:")
        action_counts = Dict{Int, Int}()
        for a in actions
            action_counts[a] = get(action_counts, a, 0) + 1
        end
        for a in sort(collect(keys(action_counts)))
            count = action_counts[a]
            pct = round(100 * count / length(actions), digits=1)
            println(io, "  Action $a (u=$(pomdp.actions[a]) N⋅m): $count times ($pct%)")
        end
        println(io)

        # Performance metrics
        println(io, "Performance Metrics:")
        println(io, "  Total reward: $(round(sum(rewards), digits=2))")
        println(io, "  Final cumulative reward: $(round(cumulative_reward[end], digits=2))")
        println(io, "  Average reward per step: $(round(mean(rewards), digits=2))")
        println(io)

        # Tracking performance
        angle_errors = θ_hist .- θ_target_hist
        println(io, "Tracking Performance:")
        println(io, "  Mean angle error: $(round(mean(angle_errors), digits=2))°")
        println(io, "  Std angle error: $(round(std(angle_errors), digits=2))°")
        println(io, "  Max absolute error: $(round(maximum(abs.(angle_errors)), digits=2))°")
        println(io, "  Final RMS error: $(round(rms_error[end], digits=2))°")
        println(io)

        # Control effort
        println(io, "Control Effort:")
        println(io, "  Total control effort: $(round(sum(abs.(u_hist)), digits=2))")
        println(io, "  Average |u|: $(round(mean(abs.(u_hist)), digits=2))")
        println(io, "  Max |u|: $(round(maximum(abs.(u_hist)), digits=2))")
        println(io)

        # Health state distribution
        println(io, "Health State Distribution:")
        health_counts = Dict{Symbol, Int}()
        for h in health_hist
            health_counts[h] = get(health_counts, h, 0) + 1
        end
        for h in [:healthy, :degraded, :critical, :failed]
            count = get(health_counts, h, 0)
            pct = round(100 * count / length(health_hist), digits=1)
            println(io, "  $h: $count steps ($pct%)")
        end
        println(io)

        # Belief quality (entropy)
        avg_entropy = 0.0
        for t in 1:size(health_beliefs, 1)
            probs = health_beliefs[t, :]
            avg_entropy += -sum(probs .* log.(probs .+ 1e-10))
        end
        avg_entropy /= size(health_beliefs, 1)
        println(io, "Belief Quality:")
        println(io, "  Average belief entropy: $(round(avg_entropy, digits=3)) (lower = more confident)")

        println(io)
        println(io, "="^60)
    end
    println("  ✓ Saved statistics.txt")

    println("=== Results saved successfully ===\n")

    # Return summary statistics as a Dict
    return Dict(
        "total_reward" => sum(rewards),
        "final_rms_error" => rms_error[end],
        "total_control_effort" => sum(abs.(u_hist)),
        "mean_angle_error" => mean(θ_hist .- θ_target_hist),
        "avg_belief_entropy" => sum([-sum(health_beliefs[t, :] .* log.(health_beliefs[t, :] .+ 1e-10))
                                      for t in 1:size(health_beliefs, 1)]) / size(health_beliefs, 1)
    )
end

"""
    extract_health_beliefs(beliefs, pomdp)

Extract health belief probabilities from belief history.

Returns a matrix of size (n_steps, n_health_states) where each row sums to 1.
"""
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
