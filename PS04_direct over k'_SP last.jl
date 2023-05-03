#PS04 Max over k' using Brent (internet source). I fixed l=l_ss as there is no closed form solution for l. No error messages, but Julia is not returning the output. Need error checks. Euler errors are to be worked out. also curvuture is assumed 1

using Optim
using Parameters
using Interpolations
using LinearAlgebra

    # Set parameters
    @with_kw struct Par
        z::Float64=1.0;
        α::Float64 = 1/3;
        β::Float64 = 0.96;
        δ::Float64=0.05;
        η::Float64=1.0;
        σ::Float64=2.0;
        xpar::Float64=0.0817;
        max_iter::Int64   = 800  ; 
        dist_tol::Float64 = 1E-9  ; # Tolerance for distance
    end
    
    # Put parameters into object par
    par = Par()
    
    # Find steady state values for k, y, c, r, w, and x if l_ss=0.4
    function steady(par::Par)
        @unpack z, α, β, σ, δ, η, xpar = par
        k_ss = (1/β-1+δ)^(1/(α-1))/(α*z)^(1/α-1)
        y_ss = z*k_ss^α
        c_ss = y_ss - k_ss
        r_ss = 1/β-1+δ
        w_ss = (1-α)*y_ss
        l_ss = (((1-α)*c_ss^(-σ)*z*k_ss^α)/xpar)^(1/(η+α))
        return k_ss, y_ss, c_ss, r_ss, w_ss, l_ss
    end
    
    # Find steady states
    k_ss, y_ss, c_ss, r_ss, w_ss, l_ss = steady(par)

    
# Define the grid of capital stock
num_k=1000
k_grid = range(0.01, stop=k_ss*2, length=num_k)
#l_grid = range(0.01, stop=1.0, length=1000)


# Define the utility function
function u(c,l, par::Par)
    @unpack xpar, η,σ = par 
        if c <= 0 || 1 - l <= 0
        return -1e10
        end
return c^(1 - σ) / (1 - σ) - xpar*l_ss^(1+η) / (1 + η)
end

# Define the consumption function
function c_analytical(k, kp, l_ss,par::Par)
    @unpack δ,α= par 
    return (k^α) * l_ss^(1 - α) + (1 - δ) * k - kp
end

# Define the objective function
function f1(par::Par)
    @unpack z, α, σ, δ,xpar, η, β, max_iter, dist_tol = par 
    V = zeros(num_k)
V_new = similar(V)
G_c=zeros(num_k)
G_l=zeros(num_k)
G_kp=zeros(num_k)
k_min, k_max = extrema(k_grid)
for iter in 1:max_iter
    for (i, k) in enumerate(k_grid)
        objective(kp) =-u(c_analytical(k, kp, l_ss, par), l_ss, par) - β * CubicSplineInterpolation(k_grid, V)(kp)
        res = optimize(objective, k_min, k_max)
        V_new[i] = -res.minimum
        kp = res.minimizer[1]
        V_new[i], G_kp[i] = -res.minimum, res.minimizer
        # Update optimal consumption policy function
        G_l = l_ss
        G_c[i] =z*k^α*l_ss - (G_kp[i] - (1 - δ) * k)
    end

    # Check for convergence
    if norm(V_new - V) < dist_tol
        break
    end

    # Update value function
    V .= V_new
    iter += 1
end
return G_c, G_kp, V, G_l
end

G_c, G_kp, V, G_l = f1(par)

println("Optimal consumption policy function: ")
println(G_c)
println("Optimal capital policy function: ")
println(G_kp)
println("Value function: ")
println(V)
println("Optimal labor policy function: ")
println(G_l)

# Plot the value function, policy functions and time
#using Plots
#gr()
# Plot value function
#plot(k_grid, V[:,1])
#plot!(k_grid, V[:,end] )
#title!("Value Function")

# Plot policy functions
#plot(k_grid, G_kp, label="Investment")
#plot!(k_grid, G_l, label="Labor")
#plot!(k_grid, G_c, label="Consumption")




