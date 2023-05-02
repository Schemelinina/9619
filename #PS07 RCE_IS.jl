#PS07 RCE. The code is from Sergio. Algorithm is from slides of Sergio. Work in progress.

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
    z_bar::Float64=1.0;
    # Howard's Policy Iterations
H_tol::Float64    = 1E-9  ; # Tolerance for policy function iteration
N_H::Int64        = 20    ; # Maximum number of policy iterations
# Minimum consumption for numerical optimization
c_min::Float64    = 1E-16
end

# Put parameters into object par
par = Par()

# Find steady state values for k, y, c, r, w, and x if l_ss=0.4
function steady(par::Par)
    @unpack z_bar, α, β, σ, δ, η, xpar, sigma = par
    k_ss = (1/β-1+δ)^(1/(α-1))/(α*z_bar)^(1/α-1)
    y_ss = z_bar*k_ss^α
    c_ss = y_ss - k_ss
    r_ss = 1/β-1+δ
    w_ss = (1-α)*y_ss
    l_ss = (((1-α)*c_ss^(-sigma)*z_bar*k_ss^α)/xpar)^(1/(η+α))
    return k_ss, y_ss, c_ss, r_ss, w_ss, l_ss
    end
            
    # Find steady states
    k_ss, y_ss, c_ss, r_ss, w_ss, l_ss = steady(par)

function PolyRange(start, stop; θ, N)
    return start .+ (0:N-1) * ((stop - start) * (θ - 1) / (θ^N - 1))
end

function Make_Grid(n_k,θ_k,par::Par,scale_type="Poly")
    k_ss,y_ss,c_ss,r_ss,w_ss = steady(par)
    # Get k_grid
    if θ_k≠1
        if scale_type=="Poly"
        k_grid = PolyRange(1E-5,2*k_ss;θ=θ_k,N=n_k) ; # Curved grid between 0 and 2*k_ss
        elseif scale_type=="Exp"
        error("scale_type must be either Poly or Exp")
        end
    else
    k_grid = range(1E-5,2*k_ss,length=n_k)
    end
    return k_grid
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
    k_ss = (1/par.β-1+par.δ)^(1/(par.α-1))/(par.α*par.z_bar)^(1/par.α-1)
    y_ss = par.z_bar*k_ss^par.α
    c_ss = y_ss - k_ss
    r_ss = 1/par.β-1+par.δ
    w_ss = (1-par.α)*y_ss
    l_ss = (((1-par.α)*c_ss^(-par.sigma)*par.z_bar*k_ss^par.α)/par.xpar)^(1/(par.η+par.α))
    # Capital Grid
    θ_k::Float64    = 1.5                        # Curvature of k_grid
    n_k::Int64      = 500                       # Size of k_grid
    n_k_fine::Int64 = 1000                      # Size of fine grid for interpolation
    k_grid = Make_Grid(n_k, θ_k, par)  # k_grid for model solution
    k_grid_fine = Make_Grid(n_k_fine, 1, par)  # Fine grid for interpolation 
    # Productivity process
    N       = 10                               # Size of z_grid
    MP_z      = Rouwenhorst95(par.ρ,par.σ,par.N)      # Markov Process for z
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
    @unpack σ, xpar, η, sigma = par 
    if l < 0 || l > 1
        return NaN
    end
    return c^(1 - sigma) / (1 - sigma) - xpar*l^(1+η) / (1 + η)
end

# Production fn
function f(k, l, par::Par)
    @unpack α = par 
    return z * (k^α) * (l^(1 - α))
end

# Wage fn
function w(k, z, par::Par)
    @unpack α = par 
    return z * (1-α)*k^α * l^(-α)
end

# Capital return fn
function r(k, z, par::Par)
    @unpack  α = par     
    return α * z * k^(α - 1) * l^(1 - α) 
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

 
# G_k interpolation
function G_kp_zk(i_z::Int64,k,M::Model)
    itp = ScaledInterpolations(M.k_grid,M.G_kp[i_z,:], BSpline(Cubic(Line(OnGrid()))))
    return itp(k)
end


# EGM
function EGM(M::Model)
@unpack par, N, MP_z, n_k, k_grid, V, G_kp, G_c= M
@unpack β, max_iter, dist_tol=par
        
V_old = copy(V)
G_kp_old = copy(G_kp)
G_c_old = copy(G_c)
if any( diff(V,dims=2).<0 )
error("V must be monotone for EGM to work")
end
   
G_c = Array{Float64}(undef, N, n_k)
G_l = Array{Float64}(undef, N, n_k) 
G_kp = Array{Float64}(undef, N, n_k)

for iter in 1:max_iter
#Loop through productivity states
    for i in 1:N
        z = MP_z.grid[i]  # productivity level from grid
        # Loop through the capital grid
            for j in 1:n_k
            # Calculate prices r and w
            r = r(k_grid[j], z, par)
            w = w(k_grid[j], z, par)
      
               # Inner loop for individual states
               for h in 1:n_k
               
                #The rest is taken from Sergio, needs to be modified
                
                # Define expectation of value function
               EV = β*MP_z.Π*V  
               # Define the derivative of EV for each value
               dEV = zeros(size(EV))
               for i_z in 1:N
               EV_ip = ScaledInterpolations(k_grid, EV[i_z, :], FritschButlandMonotonicInterpolation())
               dEV_ip(x) = ForwardDiff.derivative(EV_ip, x)
               dEV[i_z, :] = dEV_ip.(k_grid)
            end
               # Define Consumption from Euler Equation
               C_endo = d_utility_inv(dEV,par)
               #add analytical for l from foc
               # Define endogenous grid on cash on hand
               Y_endo = C_endo .+ M.k_mat
               # Sort Y_endo for interpolation
               for i_z in 1:N
                sort_ind = sortperm(Y_endo[i_z, :])
                Y_endo[i_z, :] .= Y_endo[i_z, :][sort_ind]
                C_endo[i_z, :] .= C_endo[i_z, :][sort_ind]
            end
               # Define value function on endogenous grid
               V_endo = utility(C_endo,par) .+ EV
              # Interpolate functions on exogenous grid
              for i_z in 1:N
                V_ip = Spline1D(Y_endo[i_z, :], V_endo[i_z, :])
                V[i_z, :] .= V_ip.(M.k_mat[i_z, :])
                C_ip = Spline1D(Y_endo[i_z, :], C_endo[i_z, :])
                G_c[i_z, :] .= C_ip.(M.k_mat[i_z, :])
                #Y_grid to be defined
                G_kp[i_z, :] = Y_grid[i_z, :] .- G_c[i_z, :]
                #G_l[..]=..
            end
        end
    end
        
    # Check convergence
if maximum(abs.(V .- V_old)) < dist_tol
    println("Convergence achieved")
    break
else
    V_old .= V
    G_kp_old .= G_kp
    G_c_old .= G_c
end
end

return V, G_kp, G_c, G_l
end
end

V, G_kp, G_c, G_l = EGM(M)

    

