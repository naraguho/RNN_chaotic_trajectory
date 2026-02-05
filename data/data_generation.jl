using LinearAlgebra
using Plots
using DelimitedFiles
using Random
using Printf
using ColorSchemes
using Statistics

# --------------------------------------
# Initialization
# --------------------------------------
# Initialize state with net velocity : only occupy k=\pi/2.
#function init_current!(dim, Q, P, ρ; seed=1)
#    rng = MersenneTwister(seed)
#    Q .= 1e-4 .* randn(rng, dim)
#    P .= 1e-4 .* randn(rng, dim)
#    tot_k = 2π .* (0:dim-1) ./ dim .- π
#    sum_k = filter(k -> -π/2 <= k < π/2, tot_k)
#    for i in 1:dim, j in 1:dim
#        ρ[i, j] = sum(exp(im * k * (i - j)) for k in sum_k) / dim
#    end
#end


# Initialize state without net velocity : equal superposition with two state k= \pi/2 and -\pi/2
function init_gs_deep_quench!(dim, Q, P, ρ; tp=1e-10)
    Q .= 1e-4 * randn(dim)
    P .= 1e-4 * randn(dim)
    H0 = hopping_Hamiltonian_1D(dim)
    #H1 = H0 #- Diagonal(Q)
    ρ .= compute_density_matrix_C(H0, tp, 0)
end

function compute_density_matrix_C(H, kT, mu)
    vals, vecs = eigen(H)               # vals = eigenvalues, vecs = eigenvectors
    dim = length(vals)

    occ = @. 1 / (exp((vals - mu)/kT) + 1)

    D = zeros(ComplexF64, dim, dim)

    for a in 1:dim
        for b in a:dim
            s = 0.0 + 0.0im
            for m in 1:dim
                s += occ[m] * conj(vecs[b,m]) * vecs[a,m]
            end
            D[a,b] = s
            if a != b
                D[b,a] = conj(s)
            end
        end
    end

    return D
end


# --------------------------------------
# Hamiltonians and equations of motion
# --------------------------------------
function hopping_Hamiltonian_1D(dim::Int)
    H = zeros(Float64, dim, dim)
    for i in 1:dim
        j = mod1(i + 1, dim)
        H[i, j] = -1
    end
    return H + H'
end
#

onsite_Hamiltonian_1D(dim, Q) = -Diagonal(Q)
compute_dX(Q, P, r, n) = r .* P
compute_dP(Q, P, r, n) = r .* (n .- 0.5) .- r .* Q

# --------------------------------------
# RK4 time evolution
# --------------------------------------
function evolve!(dim, X, P, ρ, dt, r, H_hop, λ)
    ons = onsite_Hamiltonian_1D(dim, X)
    n = diag(ρ)

    # RK4 step 1
    KX = dt .* compute_dX(X, P, r, n)
    KP = dt .* compute_dP(X, P, r, n)
    Kρ = -im * dt * ((H_hop * ρ - ρ * H_hop) + 4λ * (ons * ρ - ρ * ons))
    ρ2, X2, P2 = ρ + 0.5Kρ, X + 0.5KX, P + 0.5KP
    Kρ_sum, KX_sum, KP_sum = Kρ/6, KX/6, KP/6

    # RK4 step 2
    ons = onsite_Hamiltonian_1D(dim, X2)
    n = diag(ρ2)
    KX = dt .* compute_dX(X2, P2, r, n)
    KP = dt .* compute_dP(X2, P2, r, n)
    Kρ = -im * dt * ((H_hop * ρ2 - ρ2 * H_hop) + 4λ * (ons * ρ2 - ρ2 * ons))
    ρ2, X2, P2 = ρ + 0.5Kρ, X + 0.5KX, P + 0.5KP
    Kρ_sum += Kρ/3; KX_sum += KX/3; KP_sum += KP/3

    # RK4 step 3
    ons = onsite_Hamiltonian_1D(dim, X2)
    n = diag(ρ2)
    KX = dt .* compute_dX(X2, P2, r, n)
    KP = dt .* compute_dP(X2, P2, r, n)
    Kρ = -im * dt * ((H_hop * ρ2 - ρ2 * H_hop) + 4λ * (ons * ρ2 - ρ2 * ons))
    ρ2, X2, P2 = ρ + Kρ, X + KX, P + KP
    Kρ_sum += Kρ/3; KX_sum += KX/3; KP_sum += KP/3

    # RK4 step 4
    ons = onsite_Hamiltonian_1D(dim, X2)
    n = diag(ρ2)
    KX = dt .* compute_dX(X2, P2, r, n)
    KP = dt .* compute_dP(X2, P2, r, n)
    Kρ = -im * dt * ((H_hop * ρ2 - ρ2 * H_hop) + 4λ * (ons * ρ2 - ρ2 * ons))
    Kρ_sum += Kρ/6; KX_sum += KX/6; KP_sum += KP/6

    ρ .+= Kρ_sum; X .+= KX_sum; P .+= KP_sum
end

# --------------------------------------
# Order parameters δQ(t), δn(t)
# --------------------------------------
function compute_order_params(Q_traj::Matrix{Float64}, n_traj::Matrix{Float64})
    n_steps, N = size(Q_traj)
    i = collect(1:N)
    cos_term = cos.(π .* i)
    #print(cos_term)
    δQ_t = [mean(Q_traj[t, :] .* cos_term) for t in 1:n_steps]
    δn_t = [mean(n_traj[t, :] .* cos_term) for t in 1:n_steps]
    return δQ_t, δn_t
end

# --------------------------------------
# Main simulation
# --------------------------------------
function run_simulation(seed::Int, r::Float64, λ::Float64, dim::Int)
    dim, dt, nS, save_interval = dim, 0.01, 360000, 300

    X = zeros(dim)
    P = zeros(dim)
    ρ = zeros(ComplexF64, dim, dim)
    init_gs_deep_quench!(dim, X, P, ρ; tp=1e-10)
    println("ρ", ρ)

    H_hop = hopping_Hamiltonian_1D(dim)

    # number of saved frames (including t=0)
    n_saved = div(nS, save_interval) + 1

    X_traj = zeros(n_saved, dim)
    P_traj = zeros(n_saved, dim)
    n_traj = zeros(n_saved, dim)

    # save initial state (t = 0)
    X_traj[1, :] .= X
    P_traj[1, :] .= P
    n_traj[1, :] .= real(diag(ρ))

    save_idx = 1
    for step in 1:nS
        evolve!(dim, X, P, ρ, dt, r, H_hop, λ)

        if step % save_interval == 0
            save_idx += 1
            X_traj[save_idx, :] .= X
            P_traj[save_idx, :] .= P
            n_traj[save_idx, :] .= real(diag(ρ))
        end
    end

    return X_traj, P_traj, n_traj
end


# --------------------------------------
# Sweep over r and λ
# --------------------------------------
base_dir = ""
r_values = [0.3]
λ_values = [0.67]
dim_values = [32]
seeds = 1:2
save_every = 300
dt = 0.01*save_every

for r in r_values, λ in λ_values, dim in dim_values
    combo_dir = joinpath(base_dir, @sprintf("r%.3f_lambda%.3f_dim%d", r, λ, dim))
    isdir(combo_dir) || mkpath(combo_dir)
 
    for seed in seeds
        println(@sprintf("=== Running simulation: r=%.2f, λ=%.2f, seed=%d ===", r, λ, seed))
        X_traj, P_traj, n_traj = run_simulation(seed, r, λ, dim)

        save_dir = joinpath(combo_dir, "seed_$(seed)")
        isdir(save_dir) || mkpath(save_dir)


        # --- Order parameters ---
        δQ_t, δn_t = compute_order_params(X_traj, n_traj)
        writedlm(joinpath(save_dir, "order_parameter_Q.txt"), δQ_t)
        writedlm(joinpath(save_dir, "order_parameter_n.txt"), δn_t)
        time = (1:length(δQ_t)) * dt
        pδ = plot(time, δQ_t, lw=2, label="δQ(t)")
        plot!(pδ, time, δn_t, lw=2, label="δn(t)", xlabel="Time", ylabel="Order parameter",
            title=@sprintf("δQ and δn — r=%.2f, λ=%.2f, seed=%d", r, λ, seed), ylims=(-0.3, 0.3))
        savefig(pδ, joinpath(save_dir, "order_parameters.png"))

        # --- Heatmap for  configuration Q(t, site)---
        
        n_steps, n_sites = size(X_traj)
        time = (0:n_steps-1) .* dt  # continuous time axis

        pH = heatmap(
            1:n_sites,             # x-axis → site index
            time,                  # y-axis → time
            X_traj,
            aspect_ratio=:auto,
            c=cgrad(:RdBu, rev=true),
            xlabel="Site index",
            ylabel="Time",
            title=@sprintf("Q(t, site) — r=%.2f, λ=%.2f, seed=%d", r, λ, seed)
        )
        savefig(pH, joinpath(save_dir, "Q_heatmap.png"))

        println("✅ Saved results → $save_dir\n")
    end
end
