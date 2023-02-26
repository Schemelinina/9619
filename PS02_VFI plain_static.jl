##PS_02 VFI plain..This code needs correction, I used different sources from Internet to compile it... 
#I assumed x=0.0817 as found in a separate attached code. Optimal labour is solved for implicitly using static optimization problem for l. So we assume that at each time period the household chooses labor optimally given the current values of the state variables.

using Parameters
@with_kw struct Par
    z::Float64=1.0;
    α::Float64 = 1/3;
    β::Float64 = 0.96;
    δ::Float64=0.9;
    η::Float64=1.0;
    σ::Float64=2.0;
    x::Float64=0.0817;
    max_iter::Int64   = 500  ; 
    dist_tol::Float64 = 1E-9  ; # Tolerance for distance
end

# Put parameters into object par
par = Par()

# Find steady state values for k, y, c, r, w, and x if l_ss=0.4
function steady(par::Par)
    @unpack z, α, β, σ, δ, η, x = par
    k_ss = (1/β-1+δ)^(1/(α-1))/(α*z)^(1/α-1)
    y_ss = z*k_ss^α
    c_ss = y_ss - k_ss
    r_ss = 1/β-1+δ
    w_ss = (1-α)*y_ss
    l_ss = (((1-α)*c_ss^(-σ)*z*k_ss^α)/x)^(1/(η+α))
        return k_ss, y_ss, c_ss, r_ss, w_ss, l_ss
end

# Find steady states
k_ss, y_ss, c_ss, r_ss, w_ss, l_ss = steady(par)
println(" ")
println("------------------------")
println(" Steady State variables")
println("   k = $k_ss; y = $y_ss; c = $c_ss; l = $l_ss;")
println("   Prices: r = $r_ss; w = $w_ss;")
println("------------------------")
println(" ")

# Perform VFI plain with grid

#Define utility function
function utility(k, kop, l, par::Par)
    @unpack z, α, δ, η, σ, x = par
    c = z * l.^(1-α) * k.^α .- kop .+ k .* (1 .- δ)
    return c.^(1 - σ) ./ (1 - σ) .- x / (1 + η)
end

# Define grid for capital
function K_grid(n_k, par::Par)
    k_ss, _, _, _, _, _ = steady(par)
    k_grid = range(1E-5, 2*k_ss; length=n_k) # Equally spaced grid between 10^-5 and 2*k_ss
    return k_grid
end

function T_grid_loop(V_old, k_grid, par::Par)
    @unpack z, α, β, δ, σ, x = par
    n_k = length(k_grid)
    V = zeros(n_k)
    G_kop = zeros(n_k)
    G_c = zeros(n_k)
    for i = 1:n_k
        V_aux = zeros(n_k)
        for j = 1:n_k
            c = z * k_grid[i]^α + (1 - δ) * k_grid[i] - k_grid[j]
            if c > 0
                V_aux[j] = (c^(1 - σ) / (1 - σ)) - x / (1 + η) + β * V_old[j]
            else
                V_aux[j] = -Inf
            end
        end
        V[i], k_idx = findmax(V_aux)
        G_kop[i] = k_grid[k_idx]
        G_c[i] = z * k_grid[i]^α + (1 - δ) * k_grid[i] - G_kop[i]
    end
    return V, G_kop, G_c
end

function VFI(k_grid, par::Par)
    @unpack max_iter, dist_tol = par
    V_old = zeros(length(k_grid))
    V_dist = 1
    iter = 0
    while V_dist > dist_tol && iter < max_iter
        V_new, G_kop, G_c = T_grid_loop(V_old, k_grid, par)
        V_dist = maximum(abs.(V_new .- V_old))
        V_old = V_new
        iter += 1
    end
    if iter == max_iter
        error("Error in VFI")
    end
    return V_new, G_kop, G_c
end

# Execute numerical VFI
n_k = 20
k_grid = K_grid(n_k, par)
@time V, G_kop, G_c = VFI(k_grid, par)

# Plot results
gr()
plot(k_grid, V, title="Value Function")
plot!(k_grid, G_kop, title="Capital Policy Function")
plot!(k_grid, G_c, title="Consumption Policy Function")


