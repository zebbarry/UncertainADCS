using JLD2

function save_policy(policy, filepath::String)
    dir = dirname(filepath)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    jldsave(filepath; policy=policy)
    println("Policy saved to: $filepath")
end

function load_policy(filepath::String)
    if !isfile(filepath)
        error("Policy file not found: $filepath")
    end
    data = load(filepath)
    println("Policy loaded from: $filepath")
    return data["policy"]
end

function policy_exists(filepath::String)
    return isfile(filepath)
end

function get_policy(solver_name::String, solver, pomdp; force_resolve::Bool=false)
    policy_dir = "policies"
    policy_file = joinpath(policy_dir, "$(solver_name)_policy.jld2")

    if !force_resolve && policy_exists(policy_file)
        println("Loading existing $solver_name policy from cache...")
        return load_policy(policy_file)
    else
        println("Solving with $solver_name...")
        policy = solve(solver, pomdp)
        save_policy(policy, policy_file)
        return policy
    end
end
