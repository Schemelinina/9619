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

# Put parameters into object par
par = Par()
k_ss = (1/par.β-1+par.δ)^(1/(par.α-1))/(par.α*par.z_bar)^(1/par.α-1)
y_ss = par.z_bar*k_ss^par.α
c_ss = y_ss - k_ss
l_ss = (((1-par.α)*abs.(c_ss)^(-par.σ)*par.z_bar*k_ss^par.α)/par.xpar)^(1/(par.η+par.α))

k_min=1E-5
k_max=k_ss*2

##I added this function but it needs to be fixed.

    function Make_Grid(n_k::Int64, θ_k::Float64, k_min::Float64, k_max::Float64)
        grid = Array{Float64}(undef, n_k)
    for i in 1:n_k
        grid[i] = k_min + (k_max - k_min) * ((i - 1) / (n_k - 1)) ^ θ_k
    end
    return grid
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

# Generate structure of model objects. there is an error as MP not defined. can i define Rowenhorst within Model struct?
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
    k_grid_fine     = Make_Grid(n_k_fine,1,1E-5,2*k_ss)  # Fine grid for interpolation
    # Productivity process
    N       = 10                               # Size of z_grid
    MP_z      = Rouwenhorst95(par.ρ,par.σ,par.N)      # Markov Process for z
    # State matrices
    k_mat     = repeat(k_grid',N,1)
    z_mat     = exp.(repeat(MP_z.grid,1,n_k))
    l_policy = l_ss
    
    # Y_grid and Marginal Product of Capital
    Y_grid = par.z_bar * z_mat .* (k_mat .^ par.α) .* (l_policy .^ (1 - par.α)) + (1 - par.δ) .* k_mat
    MPk_mat   = par.α * par.z_bar * z_mat .* (k_mat .^ (par.α - 1)) .* (l_policy .^ (1 - par.α)) + (1 - par.δ)
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

function update_labor(M::Model, G_kp, G_c, par)
    # Unpack model structure
    @unpack par, N, n_k, k_mat, z_mat, l_policy = M
    @unpack α, x, σ, η, xpar = par

   new_labor = zeros(size(M.l_policy))
   for i_z in 1:N
    for j_k in 1:n_k
        new_labor[i_z, j_k] = ((1-α)*G_c[i_z, j_k]^(-σ)*z_mat[i_z]*G_kp[i_z, j_k]^α)^(1/(η+α))/xpar 
    end
end
    return new_labor
end


# EGM
function T_EGM(M::Model)
    @unpack par, N, MP_z, n_k, k_grid, Y_grid, V, G_kp, G_c = M
    @unpack β = par
    # Check Monotonicity
    if any( diff(V,dims=2).<0 )
        error("V must be monotone for EGM to work")
    end
    # Define expectation of value function for each (z,k')
    EV = β*MP_z.Π*V  # Rows are present z and columns are tomorrow's k in fixed grid
    # println("Initial EV size=$(size(EV)), and EV=$EV")
    # Check Monotonicity
    if any( diff(EV,dims=2).<0 )
        error("EV must be monotone for EGM to work")
    end
    
           # Define the derivative of EV for each value
        dEV = zeros(size(EV))
    for i_z=1:n_z
        EV_ip      = ScaledInterpolations(k_grid,EV[i_z,:], FritschButlandMonotonicInterpolation())
        dEV_ip(x)  = ForwardDiff.derivative(EV_ip,x)
        dEV[i_z,:].= dEV_ip.(k_grid)
       end
    # Check Monotonicity
    if any( dEV.<0 )
        for i_z=1:N
            println("\n i_z=$i_z, [min,max]=$([minimum(dEV[i_z,:]),maximum(dEV[i_z,:])]) \n")
        end
        error("dEV must be monotone for EGM to work")
    end
        # Define Consumption from Euler Equation
    C_endo = d_utility_inv(dEV,par)
    # Define endogenous grid on cash on hand
    Y_endo = C_endo .+ M.k_mat
        # Sort Y_endo for interpolation
        for i_z=1:N
        sort_ind = sortperm(Y_endo[i_z,:])
        Y_endo[i_z,:] .= Y_endo[i_z,:][sort_ind]
        C_endo[i_z,:] .= C_endo[i_z,:][sort_ind]
        end

    # Update labor policy function
    M.l_policy = update_labor_policy(M, G_kp, G_c)
# Update Y_grid and MPk_mat with the new labor policy
     M.Y_grid  = par.z_bar*M.z_mat.*(M.k_mat.^(par.α)) .* (M.l_policy .^ (1 - par.α)) .+ (1-par.δ).*M.k_mat
     M.MPk_mat = par.α*par.z_bar*M.z_mat.*(M.k_mat.^(par.α-1)) .* (M.l_policy .^ (1 - par.α)) .+ (1-par.δ)
    
        # Define value function on endogenous grid
    V_endo = utility(C_endo,par) .+ EV
    # Interpolate functions on exogenous grid
    for i_z=1:N
        V_ip        = Spline1D(Y_endo[i_z,:],V_endo[i_z,:])
        V[i_z,:]   .= V_ip.(Y_grid[i_z,:])
        C_ip        = Spline1D(Y_endo[i_z,:],C_endo[i_z,:])
        G_c[i_z,:] .= C_ip.(Y_grid[i_z,:])
        G_kp[i_z,:].= Y_grid[i_z,:] .- G_c[i_z,:]
    end
    return V, G_kp, G_c
end


@time M_EGM  = VFI_Fixed_Point(T_EGM,Model())