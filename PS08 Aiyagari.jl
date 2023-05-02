#PS07 RCE. The code is from Sergio and internet. Work in progress. Aiyagari OLG, T=20.  Most of assumptions are from Sergio's code

using SparseArrays
using Parameters
using Interpolations
using Distributions
using LinearAlgebra
using Statistics
using Dierckx # Pkg.add("Dierckx") 
using ForwardDiff
using Optim
using Roots

@with_kw struct Par
    α::Float64 = 1/3;
    β::Float64 = 0.96;
    δ::Float64=0.05;
    ρ::Float64 = 0.9 ; # Persistence of productivity process
    σ:: Float64=0.1;
    γ::Float64 = 2.0;
    z_bar::Float64=1;
    ϵ̄::Float64 = exp(-σ^2/(2*(1-ρ^2))); # Reference level for labor efficiency
    # Borrowing constraint
    a_min::Float64 = 0; # Borrowing constraint
    max_iter::Int64   = 800  ; 
    dist_tol::Float64 = 1E-9  ; # Tolerance for distance
    # Minimum consumption for numerical optimization
c_min::Float64    = 1E-16
end

# Put parameters into object par
par = Par()


function steady(par::Par)
    @unpack z_bar, α, β, δ, z_bar = par
    k_ss = (par.β*par.α*par.z_bar/(1-par.β*(1-par.δ)))^(1/(1-par.α))
    return k_ss
end
            
k_ss = steady(par)


function PolyRange_a(a_min::Float64, a_max::Float64, n_a::Int64, θ_a::Float64)
    return a_min .+ (a_max - par.a_min) .* (((1:n_a) .- 1) ./ (n_a - 1)).^θ_a
end

function PolyRange_k(min_value::Float64, max_value::Float64, n_k::Int, θ_k::Float64)
    return min_value .+ (max_value - min_value) .* (((1:n_k) .- 1) ./ (n_k - 1)).^θ_k
end

function Make_Grid_a(n_a, θ_a, a_min, par::Par, a_max::Float64=NaN, scale_type::String="Poly")
    a_min::Float64 = 0
    k_ss = steady(par)
        k_ss = steady(par)
        if isnan(a_max)
            a_max = 2 * k_ss
        end
        if θ_a ≠ 1
            if scale_type == "Poly"
                a_grid = PolyRange_a(a_min, a_max, n_a, θ_a)
            elseif scale_type == "Exp"
                error("scale_type must be either Poly or Exp")
            else
                error("error")
            end
        else
            a_grid = range(a_min, stop=a_max, length=n_a)
        end
        return a_grid
    end


    function Make_Grid_k(n_k::Int, θ_k::Float64, par::Par, min_value::Float64, max_value::Float64, scale_type::String = "Poly")
    k_ss=steady(par)
    min_value = 1E-5
    max_value = 2 * k_ss
    # Get k_grid
    if θ_k≠1
        if scale_type=="Poly"
        k_grid = PolyRange_k(min_value,max_value,n_k, θ_k) ; # Curved grid between 0 and 2*k_ss
        elseif scale_type=="Exp"
        error("scale_type must be either Poly or Exp")
        end
    else
    k_grid = range(1E-5,2*k_ss,length=n_k)
    end
    return k_grid
end


function Rouwenhorst95(ρ, σ, n_ϵ)
    p = (1 + ρ) / 2
    q = p
    ψ = sqrt((n_ϵ - 1) / (1 - ρ^2)) * σ

    if n_ϵ == 2
        Π_z = [p 1 - p; 1 - q q]
    else
        MP_aux = Rouwenhorst95_Aiyagari(ρ, σ, n_ϵ - 1)
        o = zeros(n_ϵ - 1)
        Π_z = p * [MP_aux.Π o; o' 0] + (1 - p) * [o MP_aux.Π; 0 o'] + (1 - q) * [o' 0; MP_aux.Π o] + q * [0 o'; o MP_aux.Π]
        Π_z = Π_z ./ repeat(sum(Π_z, dims = 2), 1, n_ϵ)
    end

    log_z_grid = range(-ψ, ψ, length = n_ϵ)
    z_grid = exp.(log_z_grid)
    return (grid=z_grid, Π=Π_z)
end

# Generate structure for markov processes 
@with_kw struct MP
        # Model Parameters
        N::Int64 # Number of states
        grid     # Grid of discrete markov process
        Π        # Transition matrix
        PDF      # Stationary distribution
        CDF      # Stationary distribution
    end

# Generate structure of model objects.  
@with_kw struct Model
    # Parameters
    par::Par = Par() # Model paramters in their own structure
    
    # Steady State Values
    k_ss = (par.β*par.α*par.z_bar/(1-par.β*(1-par.δ)))^(1/(1-par.α))
    H::Int64 = 20    ##years
    # Capital Grid
    θ_k::Float64    = 1.5                        # Curvature of k_grid
    n_k::Int64      = 500                       # Size of k_grid
    n_k_fine::Int64 = 1000                      # Size of fine grid for interpolation
    min_value = 1E-5
    max_value = 2 * k_ss
    k_grid = Make_Grid_k(n_k, θ_k, par, min_value, max_value)  # k_grid for model solution
    
    #k_grid_fine = Make_Grid_k(n_k_fine, 1, par)  # Fine grid for interpolation 
    θ_a::Float64    = 1.5                        # Curvature of a_grid
    n_a::Int64      = 500           
    n_a_fine::Int64 = 1000 
    a_min::Float64 = 0   
    a_max = 2 * k_ss 
    a_grid          = Make_Grid_a(n_a,θ_a,a_min,a_max)  # a_grid for model solution
    #a_grid_fine     = Make_Grid1_a(n_a_fine,1,par.a_min,a_max)  # Fine grid for interpolation
    # Productivity process
    n_ϵ       = 20                               # Size of ϵ_grid
    MP_ϵ      = Rouwenhorst95(par.ρ,par.σ,n_ϵ)       # Markov Process for ϵ
    ϵ_grid    = par.ϵ̄*exp.(MP_ϵ.grid)              # Grid in levels
    # State matrices
    a_mat     = repeat(a_grid',n_ϵ,1)
    a_mat_fine= repeat(a_grid_fine',n_ϵ,1)
    ϵ_mat     = p.ϵ̄*exp.(repeat(MP_ϵ.grid,1,n_a))
    # Prices and aggregates
    r::Float64 = 0.90*(1/par.β - 1)
    K::Float64 = (par.α*par.z_bar/(r+par.δ))^(1/(1-par.α)) # k_ss
    Y::Float64 = par.z_bar*K^(par.α)
    w::Float64 = (1-par.α)*par.z_bar*K^(par.α)
    V = Array{Float64}(undef, n_ϵ, n_a, H)
    G_ap = Array{Float64}(undef, n_ϵ, n_a, H)
    G_c = Array{Float64}(undef, n_ϵ, n_a, H)
    
    # Distribution
    #Γ         = 1/(n_ϵ*n_a_fine)*ones(n_ϵ,n_a_fine) # Distribution (initiliazed to uniform)
    # Error in Euler equation
    #Euler     = Array{Float64}(undef,n_ϵ,n_a_fine)  # Errors in Euler equation
      end
#error...need to fix Make_Grid_a
M = Model()

# Utility function
function utility(c,par::Par)
    if par.γ>1
    return (c).^(1-par.γ)/(1-par.γ)
    else
    return log.(c)
    end
end

function d_utility(c,par::Par)
    return (c).^(-par.γ)
end

function d_utility_inv(x,par::Par)
    return x.^(-1/par.γ)
end


# Terminal condition
@unpack H, n_ϵ, n_a, a_grid, ϵ_grid, V, G_c, G_ap = Model()
for h in 1:H
if h == H
    for i_ϵ in 1:n_ϵ
        for i_a in 1:n_a
            c = (1+r)*a_grid[i_a] + w*ϵ_grid[i_ϵ]
            V[i_ϵ, i_a, h] = utility(c, par)
            G_c[i_ϵ, i_a, h] = c
            G_ap[i_ϵ, i_a, h] = 0
        end
    end
end


# Backward induction # not complete 
for h in (H-1):-1:1
    for i_ϵ in 1:n_ϵ
        for i_a in 1:n_a
            a = a_grid[i_a]
            exp_Vnext = MP_ϵ.Π[i_ϵ, :] * V[:, :, h + 1]
            c = d_utility_inv(par.β * (1 + r) * exp_Vnext, par)
            #computing feasible consumption
            c_check = (1 + r) * a + w * ϵ_grid[i_ϵ] - a_grid
            compare = c_check .> par.c_min

            V[i_ϵ, i_a, h] = utility(c, par) + par.β * dot(MP_ϵ.Π[i_ϵ, :], V[:, :, h + 1])
            G_c[i_ϵ, i_a, h] = c

            # find optimal assets
            feasible_val = V[i_ϵ, compare, h + 1]
            opt_idx = findmax(feasible_val)[2]
            G_ap[i_ϵ, i_a, h] = a_grid[opt_idx]
        end
    end
end


#Simulation, not complete
using Random
using Statistics

agents = 100000
periods = 5000
# initialize state var
ϵ_idx = rand(MP_ϵ.CDF, agents) 
a_initial = zeros(agents) 
sim_a = zeros(agents, periods)
sim_ϵ = zeros(agents, periods)
sim_labor_income = zeros(agents, periods)
sim_total_income = zeros(agents, periods)

# simulate 
for t in 1:periods
    for i in 1:agents
        if t == 1
            a = a_initial[i]
        else
            a = sim_a[i, t - 1]
        end
        i_ϵ = ϵ_idx[i]
        ϵ = M.ϵ_grid[i_ϵ]
        if t <= H
            h = t
        else
            h = H
        end
        ap = G_a[h][i_ϵ](a)
        income_labor = ϵ * M.w
        income_total = income_labor + (1 + M.r) * a - ap
        sim_a[i, t] = ap
        sim_ϵ[i, t] = ϵ
        sim_labor_income[i, t] = income_labor
        sim_total_income[i, t] = income_total

        # update productivity 
        ϵ_idx[i] = findfirst(x -> x >= rand(), MP_ϵ.CDF[i_ϵ, :])
    end
end

labor_moments = [mean(sim_income_labor), std(sim_income_labor), skewness(sim_income_labor), kurtosis(sim_income_labor)]
assets_moments = [mean(sim_a), std(sim_a), skewness(sim_a), kurtosis(sim_a)]
top_income_share = sum(sort(sim_income_total[:, end], rev=true)[1:Int(0.01 * agents)]) / sum(sim_income_total[:, end])
top_wealth_share = sum(sort(sim_a[:, end], rev=true)[1:Int(0.01 * agents)]) / sum(sim_a[:, end])
println("labor moments: ", labor_moments)
println("assets moments: ", assets_moments)
println("Top income share: ", top_income_share)
println("Top wealth share: ", top_wealth_share)




