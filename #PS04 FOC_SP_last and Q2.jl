#PS04 Max over FOC (internet source). Need error checks and curvuture Euler errors are to be worked out. curvuture here is assumed 1. 

using Optim
using Parameters
using Interpolations
using LinearAlgebra
using NLsolve
    
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
    
# Define the utility function
function u(c,l, par::Par)
@unpack xpar, η,σ = par 
if c <= 0 || 1 - l <= 0
return -1e10
end
return c^(1 - σ) / (1 - σ) - xpar*l^(1+η) / (1 + η)
end

# Define the production function
function f(k, l, par::Par)
    @unpack z, α = par
    return z * k^α * l^(1 - α)
end


# FOCs//needs to be reworked 
function sy(x, par::Par)
    c, kp, l = x
    @unpack z, α, σ, δ, xpar, η, β = par
    
    y = f(k, l, par)
    r = α * z * k^(α - 1) * l^(1 - α) 
    w = z * (1-α)*k^α * l^(-α)
    c_next = (1 - δ) * kp + f(kp, l_next, par) - kp
    l_next = (c_next^(-σ) * w_next / xpar)^(1 / η)
    w_next = z * (1-α) * kp^(α) * l_next^(-α)
    eq1 = c^(-σ) - β * c_next^(-σ) * ((1 - δ) + r)
    eq2 = c + kp - (1 - δ) * k - y
    eq3 = l - (c^(-σ) * w / xpar)^(1 / η)
    return [eq1, eq2, eq3]
end

# VFI
function VFI(par::Par)
    @unpack z, α, σ, δ, xpar, η, β, max_iter, dist_tol = par
    V = zeros(num_k)
    V_new = similar(V)
    G_c = zeros(num_k)
    G_l = zeros(num_k)
    G_kp = zeros(num_k)
    k_min, k_max = extrema(k_grid)
    
    for iter in 1:max_iter
        for (i, k) in enumerate(k_grid)
            function euler_eq(x)
                c, kp, l = x
                eqs = sy([c, kp, l], par)
                return eqs
            end
            res = nlsolve(euler_eq, [c_ss, k_ss, l_ss])
            c, kp, l = res.zero
            V_new[i] = u(c, l, par) + β * CubicSplineInterpolation(k_grid, V)(kp)
            G_c[i] = c
            G_l[i] = l
            G_kp[i] = kp
        end
                
            # Check for convergence
        if norm(V_new - V) < dist_tol
            break
        end

        # Update value function
        V .= V_new
    end
    return G_c, G_kp, V, G_l
end  
           
 
G_c, G_kp, V, G_l = VFI(par)

   
println("Optimal consumption policy function: ")
println(G_c)
println("Optimal capital policy function: ")
println(G_kp)
println("Value function: ")
println(V)
println("Optimal labor policy function: ")
println(G_l)
    
# Plot
    #using Plots
    #gr()
    # Plot value function
    #plot(k_grid, V[:,1])
    #plot!(k_grid, V[:,end] )
    #title!("Value fn")
    
    # Plot policy functions
    #plot(k_grid, G_kp, label="G_kp)
    #plot!(k_grid, G_l, label="G_l")
    #plot!(k_grid, G_c, label="G_c")
    
    
#PS04_Q2_using FOCs solution from Q1
# Initial conditions
k0 = 0.8*k_ss
l0 = l_ss
# Create vectors to store the path
k_path = [k0]
l_path = [l0]
c_path = [z*l0^(1-α)*k0^α + (1-δ)*k0 - k0]
r_path = [1/β-1+δ]
w_path = [(1-α)*z*k0^α*l0^(-α)]
y_path = [z*k0^α*l0^(1-α)]

# Simulate the path
for t in 1:T
    V, G_kop, G_l, G_c = VFI(par)
    k_prime = G_kop[findfirst(x -> x >= k_path[end], k_grid)]
    l_prime = G_l[findfirst(x -> x >= k_path[end], k_grid)]

    c_prime = z * l_path[end]^(1-α) * k_path[end]^α + (1-δ) * k_path[end] - k_prime
    y_prime = z * k_prime^α * l_prime^(1-α)
    r_prime = 1/par.β - 1 + par.δ
    w_prime = (1-par.α) * y_prime

    push!(k_path, k_prime)
    push!(l_path, l_prime)
    push!(c_path, c_prime)
    push!(r_path, r_prime)
    push!(w_path, w_prime)
    push!(y_path, y_prime)
end

using Plots
gr()

# Plot the paths
plot(1:length(k_path), k_path, label="c")
plot!(1:length(k_path), l_path, label="l")
plot!(1:length(k_path), c_path, label="c")
plot!(1:length(k_path), r_path, label="r")
plot!(1:length(k_path), w_path, label="w")
plot!(1:length(k_path), y_path, label="y")
title!("Paths from old to a new ss")



    
    
    
