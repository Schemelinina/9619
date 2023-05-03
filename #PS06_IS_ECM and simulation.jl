#PS06 Modified Sergio's code and internet. This code is work in progress as there are errors i do not know how to fix.

using Parameters
using Interpolations
using Distributions
using LinearAlgebra
using Statistics
using Dierckx # Pkg.add("Dierckx") 
using ForwardDiff
using Optim
using Roots
using NLsolve

# Set parameters
@with_kw struct Par
    α::Float64 = 1/3;
    β::Float64 = 0.98;
    δ::Float64=0.05;
    η::Float64=1.0;
    sigma::Float64=2.0;
    xpar::Float64=0.0817;
    max_iter::Int64   = 800  ; 
    dist_tol::Float64 = 1E-9  ; # Tolerance for distance
    N::Int64          = 5  # grid points
    ρ::Float64 = 0.9 ; # Persistence of productivity process
    σ::Float64=0.01; 
z_bar::Float64=1;
    # Howard's Policy Iterations
H_tol::Float64    = 1E-9  ; # Tolerance for policy function iteration
N_H::Int64        = 20    ; # Maximum number of policy iterations
# Minimum consumption for numerical optimization
c_min::Float64    = 1E-16
end

par = Par()
k_ss = (1/par.β-1+par.δ)^(1/(par.α-1))/(par.α*par.z_bar)^(1/par.α-1)
y_ss = par.z_bar*k_ss^par.α
c_ss = y_ss - k_ss
l_ss = (((1-par.α)*abs.(c_ss)^(-par.σ)*par.z_bar*k_ss^par.α)/par.xpar)^(1/(par.η+par.α))

k_min=1E-5
k_max=k_ss*2

function Make_Grid(n_k::Int64, θ_k::Float64, k_min::Float64, k_max::Float64)
        k_grid = Array{Float64}(undef, n_k)
  for i in 1:n_k
        k_grid[i] = k_min + (k_max - k_min) * ((i - 1) / (n_k - 1)) ^ θ_k
    end
    return k_grid
end

# Define a markov process struct
    # Generate structure for markov processes 
    @with_kw struct MP
        # Model Parameters
        N::Int64 # Number of states
        grid     # Grid of discrete markov process
        Π        # Transition matrix
        PDF      # Stationary distribution
        CDF      # Stationary distribution
    end

# Generate structure of model objects. there is an error as Rowenhorst is nor defined. can i define it within Model struct? 
@with_kw struct Model
    # Parameters
    par::Par = Par() # Model paramters in their own structure
    
    # Steady State Values
    k_ss::Float64 = k_ss
    l_ss::Float64 = l_ss
    y_ss::Float64 = y_ss
    c_ss::Float64 = c_ss
    # Capital Grid
    θ_k::Float64    = 1.5                        # Curvature of k_grid
    n_k::Int64      = 500                       # Size of k_grid
    n_k_fine::Int64 = 1000                      # Size of fine grid for interpolation
    k_grid          = Make_Grid(n_k,θ_k,1E-5,2*k_ss)  # k_grid for model solution
    k_grid_fine     = Make_Grid(n_k_fine,1.0,1E-5,2*k_ss)  # Fine grid for interpolation
    # Productivity process
    N       = 10                               # Size of z_grid
    MP_z      = Rouwenhorst95(par.ρ,par.σ,N)      # Markov Process for z
    # State matrices
    k_mat     = repeat(k_grid',N,1)
    z_mat     = exp.(repeat(MP_z.grid,1,n_k))
        
    # Value and policy functions
    V         = Array{Float64}(undef,N,n_k)       # Value Function
    G_kp      = Array{Float64}(undef,N,n_k)       # Policy Function
    G_c       = Array{Float64}(undef,N,n_k)       # Policy Function
    V_fine    = Array{Float64}(undef,N,n_k_fine)  # Value Function on fine grid
    G_kp_fine = Array{Float64}(undef,N,n_k_fine)  # Policy Function on fine grid
    G_c_fine  = Array{Float64}(undef,N,n_k_fine)  # Policy Function on fine grid
    # Error in Euler equation
    Euler     = Array{Float64}(undef,N,n_k_fine)  # Errors in Euler equation
end

# Allocate model to object M for future calling
M = Model()

# Define the utility function
function u(c,l, par::Par)
    @unpack σ, xpar, η = par 
    if l < 0 || l > 1
        return NaN
    end
    return c^(1 - sigma) / (1 - sigma) - xpar*l^(1+η) / (1 + η)
end

function d_utility(z,k,kp,par::Par)
    @unpack z_bar, α, δ, c_min = par
    c = max.(z_bar.*z.*k.^α .* (l_policy .^ (1 - par.α)).+ (1-δ).*k  .- kp,c_min)
    return (c).^(-σ)
end

function d_utility(c,par::Par)
    return (c).^(-par.σ)
end

function d_utility_inv(x,par::Par)
    return x.^(-1/par.σ)
end

function Rouwenhorst95(ρ,σ,N)
# Define parameters for Rouwenhorst's approximation
    p = (1+ρ)/2
    q = p                   # Note: I am leaving q here for comparability with source
                ψ = sqrt((N-1)/(1-ρ^2)) * sqrt(log(1+σ^2))
            s = (1-q)/(2-(p+q))     # Note: s=0.5, I leave it for comparability with source
        # Fill in transition matrix
        if N==2
            Π_z = [p 1-p ; 1-q q]
        else
            MP_aux = Rouwenhorst95(ρ,σ,N-1)
            o = zeros(N-1)
            Π_z = p*[MP_aux.Π o ; o' 0] + (1-p)*[o MP_aux.Π ; 0 o'] + (1-q)*[o' 0 ; MP_aux.Π o] + q*[0 o' ; o MP_aux.Π]
            # Adjust scale for double counting
            Π_z = Π_z./repeat(sum(Π_z,dims=2),1,N)
        end
        # Distribution
        PDF_z = pdf.(Binomial(N-1,1-s),(0:N-1))
        CDF_z = cumsum(PDF_z)
        
        # Create z grid
        log_z_grid = range(-ψ, ψ, length = N)
        z_grid = exp.(log_z_grid)
            return MP(N=N,grid=z_grid,Π=Π_z,PDF=PDF_z,CDF=CDF_z)
    end
    
    
# G_k interpolation
function G_kp_zk(i_z::Int64,k,M::Model)
    itp = ScaledInterpolations(M.k_grid,M.G_kp[i_z,:], BSpline(Cubic(Line(OnGrid()))))
    return itp(k)
end

# VFI Fixed Point
function VFI_Fixed_Point(T::Function,M::Model,V_old=nothing)
    # Unpack model structure
    @unpack par, N, n_k, θ_k, k_grid, n_k_fine, k_grid_fine = M
    # VFI paramters
    @unpack max_iter, dist_tol = par
    # Initialize variables for loop
    if V_old==nothing
    # V_old  = zeros(N,n_k)     ; 
    V_old  = u(M.z_mat,M.k_mat,zeros(N,n_k),par)  ; # Start at utility with zero savings
        end
    G_kp_old = copy(M.k_mat)
    V_dist = 1              ; # Initialize distance
    println(" ")
    println("------------------------")
    println("VFI - N=$N, n_k=$n_k - θ_k=$θ_k")
    for iter=1:max_iter
                # Update value function
        V_new, G_kp, G_c = T(Model(M,V=copy(V_old)))
            # Update distance and iterations
        V_dist = maximum(abs.(V_new./V_old.-1))
        # V_dist = maximum(abs.(G_kp./G_kp_old.-1))
        # Update old function
        V_old  = V_new
        if mod(iter,100)==0
            println("   VFI Loop: iter=$iter, dist=",100*V_dist,"%")
        end
        # Check convergence and return results
        if V_dist<=dist_tol
            # Interpolate to fine grid
            V_fine    = zeros(N,n_k_fine)
            G_kp_fine = zeros(N,n_k_fine)
            G_c_fine  = zeros(N,n_k_fine)
            for i_z=1:N
            V_ip    = ScaledInterpolations(k_grid,V_new[i_z,:], BSpline(Cubic(Line(OnGrid()))))
                V_fine[i_z,:]   .= V_ip.(collect(k_grid_fine))
            G_kp_ip = ScaledInterpolations(k_grid,G_kp[i_z,:] , BSpline(Cubic(Line(OnGrid()))))
                G_kp_fine[i_z,:].= G_kp_ip.(collect(k_grid_fine))
            G_c_ip  = ScaledInterpolations(k_grid,G_c[i_z,:]  , BSpline(Cubic(Line(OnGrid()))))
                G_c_fine[i_z,:] .= G_c_ip.(collect(k_grid_fine))
            end
            # Update model
            M = Model(M; V=V_new,G_kp=G_kp,G_c=G_c,V_fine=V_fine,G_kp_fine=G_kp_fine,G_c_fine=G_c_fine)
            return M
        end
    end
    # If loop ends there was no convergence -> Error!
    error("Error in VFI")
end


function T_ECM(M::Model, par::Par)
    @unpack N, n_k, k_mat, z_mat, V, G_kp, G_c = M
    @unpack β, α, δ, z_bar, x, σ, η, xpar = par
       
    # Get labor numerically
    function labor_eq(l, i_z, i_k)
        y = z_bar * z_mat[i_z, i_k] * (k_mat[i_z, i_k]^α) * (l^(1 - α))
        marg_capital = α * z_bar * z_mat[i_z, i_k] * (k_mat[i_z, i_k]^(α - 1)) * (l^(1 - α))
                
        c = d_utility_inv(V[i_z, i_k] / marg_capital, par)
        k_next = y - c
        ##minimize squared difference to find the optimal labor value
        return (y - c - k_next)^2
    end
            
    labor = zeros(N, n_k)
    for i_z in 1:N
        for i_k in 1:n_k
            labor[i_z, i_k] = nlsolve((l, par) -> labor_eq(l, i_z, i_k), 0.01).zero
        end
    end

# Y_grid and Marginal Product of Capital
Y_grid = par.z_bar * z_mat .* (k_mat .^ par.α) .* (labor.^ (1 - par.α)) + (1 - par.δ) .* k_mat
MPk_mat   = par.α * par.z_bar * z_mat .* (k_mat .^ (par.α - 1)) .* (labor .^ (1 - par.α)) + (1 - par.δ)

#Get consumption analytically from envelope condition
G_c = d_utility_inv(V./ M.MPk_mat, par)

# Define savings from budget  constraint
G_kp.= Y_grid .- G_c

# Update V_k
    EV = zeros(N, n_k)
    for i_z = 1:N
        kp = G_kp[i_z, :]
        Vp = zeros(N, n_k)
        for i_zp = 1:N
            V_ip = ScaledInterpolations(k_mat, V[i_zp, :], FritschButlandMonotonicInterpolation())
            Vp[i_zp, :]= V_ip.(kp)
        end
        EV[i_z, :] = par.Π[i_z, :]' * Vp
    end

    # Update value function
    V = u(G_c, labor, par) .+ β * EV
    return V, G_kp, G_c, Y_grid, MPk_mat
end


# Execute Numerical VFI - ECM
# Initial value  , error to be fixed
s_0  = par.β*par.α
C_0  = (1-s_0)*Y_grid
V_0 = d_utility(C_0,par).* M.MPk_mat
@time M_ECM, Y_grid= VFI_Fixed_Point(T_ECM,Model(),V_0)


## Plotting
using Plots
using Random

# Plot the policy functions
plot(M_ECM.k_grid_fine, M_ECM.G_kp_fine', title="Capital policy fn (G_kp)")
plot!(M_ECM.k_grid_fine, M_ECM.G_c_fine', title="Consumption policy fn (G_c)")

## Simulation using solution

n_sim = 100
z_sim = Array{Int64}(undef, n_sim)
k_sim = Array{Float64}(undef, n_sim)
c_sim = Array{Float64}(undef, n_sim)
y_sim = Array{Float64}(undef, n_sim)
l_sim = Array{Float64}(undef, n_sim)

# Initialize the simulation
z_sim[1] = 1
k_sim[1] = M_ECM.k_ss
c_sim[1] = M_ECM.c_ss
l_sim[1] = M_ECM.l_ss
y_sim[1] = M_ECM.y_ss

Random.seed!(123) 

# Perform the simulation
for t in 2:n_sim
    
    z_sim[t] = rand(Distributions.Categorical(M_ECM.MP_z.Π[z_sim[t-1], :]))
    k_sim[t] = M_ECM.G_kp_fine[z_sim[t], findfirst(M_ECM.k_grid_fine .>= k_sim[t-1])]
    c_sim[t] = M_ECM.G_c_fine[z_sim[t], findfirst(M_ECM.k_grid_fine .>= k_sim[t-1])]
    # Compute labor and output
    l_sim[t] = ((1 - M_ECM.par.α) * M_ECM.z_mat[z_sim[t], :] * (k_sim[t] ^ M_ECM.par.α) / (M_ECM.par.xpar * c_sim[t] ^ (-M_ECM.par.σ))) ^ (1 / (M_ECM.par.η + M_ECM.par.α))
    y_sim[t] = M_ECM.z_mat[z_sim[t], :] * (k_sim[t] ^ M_ECM.par.α) * (l_sim[t] ^ (1 - M_ECM.par.α))
end

# Plot the levels of consumption, capital, labor, and output
plot(1:n_sim, [c_sim k_sim l_sim y_sim], title="Levels of Cons, Capital, Labor, and Output")

# Compute and report the second moments
println("St.dev:")
println("  Consumption: ", std(c_sim))
println("  Capital: ", std(k_sim))
println("  Labor: ", std(l_sim))
println("  Output: ", std(y_sim))

# Compute and report the first differences
println("St.dev of first diff")
println("  Consumption: ", std(diff(c_sim)))
println("  Capital: ", std(diff(k_sim)))
println("  Labor: ", std(diff(l_sim)))
println("  Output: ", std(diff(y_sim)))


#PS06 Q2
using CSV
using DataFrames
using Statistics
using LinearAlgebra

file_path = "C:/Users/ROG/Desktop/canada.csv"
data = CSV.read(file_path, DataFrame)

gdp = data[:, :gdp]

log_gdp = log.(gdp)

# Detrend 
t = collect(1:length(log_gdp))
X = hcat(ones(length(t)), t)
coeffs = X \ log_gdp
detr_log_gdp = log_gdp - X * coeffs
detr_growth_rate = diff(detr_log_gdp)

# Calculate the variance 
target_var = var(detr_growth_rate)

# Model calibration
# Compute model detrended variance 
log_y_sim = log.(y_sim)
t_sim = collect(1:length(log_y_sim))
X_sim = hcat(ones(length(t_sim)), t_sim)
coeffs_sim = X_sim \ log_y_sim
detr_log_y_sim = log_y_sim - X_sim * coeffs_sim
detr_growth_rate_sim = diff(detr_log_y_sim)
detr_var_sim = var(detr_growth_rate_sim)

function calibration(param::Vector, M::Model, target_var::Float64, data)
    ρ, σ = param
    par = Par(α=M.par.α, β=M.par.β, δ=M.par.δ, σ=σ,η=M.par.η, ρ=ρ, xpar=M.par.xpar)
    M_new = VFI_Fixed_Point(T_ECM, Model(par=par), V_0)
    y_sim_new = simulate(M_new, n_sim)
    log_y_sim_new = log.(y_sim_new)
    t_sim = collect(1:length(log_y_sim_new))
    X_sim = hcat(ones(length(t_sim)), t_sim)
    coeffs_sim_new = X_sim \ log_y_sim_new
    detr_log_y_sim_new = log_y_sim_new - X_sim * coeffs_sim_new
    detr_growth_rate_sim_new = diff(detr_log_y_sim_new)
    model_var_new = var(detr_growth_rate_sim_new)
    return (model_var_new - target_var)^2
end
   
      
# Perform calibration
using Optim
result = optimize(param -> calibration(param, M_ECM, target_var, data), [M_ECM.par.ρ, par.σ], BFGS())

calibrated_param = Optim.minimizer(result)
