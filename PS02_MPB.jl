##PS02 using MPB (used Internet to build the code); there is still plotting error, plots are not displayed; needs to be put inside the function. in iterations l was fixed at l_ss to simplify. 

using Parameters
using Plots
gr()

@with_kw struct Par
    z::Float64 = 1.0
    α::Float64 = 1/3
    β::Float64 = 0.96
    δ::Float64 = 0.9
    η::Float64 = 1.0
    σ::Float64 = 2.0
    x::Float64 = 0.0817
    max_iter::Int64 = 2000
    dist_tol::Float64 = 1E-9
end

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

# Define MPB iteration function 
function mpb(par::Par)
    # Extract parameters
    @unpack z, α, β, δ, σ, x, η, max_iter, dist_tol = par

    # Get steady state values
    k_ss, y_ss, c_ss, r_ss, w_ss, l_ss = steady(par)

    # Define utility function
    function utility(k, kop, l)
        c = z * l^(1-α) * k^α - kop + k * (1 - δ)
        return c^(1 - σ) / (1 - σ) - x / (1 + η)
    end

    # Create grid for capital stock
    k_grid = range(10^(-5), stop=2*k_ss, length=40)
    n_k = length(k_grid)

    # Initialize value function and policy function
    V = zeros(n_k)
    G_kop = zeros(n_k)
    G_c = zeros(n_k)

    # Perform iteration 
    for i in 1:max_iter
        V_dist = 0
        V_updates = 0

        # Loop over grid points
        for (index, k) in enumerate(k_grid)
            # Initialize current value and policy functions
            V_current = V[index]
            G_kop_current = G_kop[index]
            G_c_current = G_c[index]

            # Loop over possible values of k'
            V_k = zeros(n_k)
            for (j, kp) in enumerate(k_grid)
                V_k[j] = utility(k, kp, l_ss) + β * V[j]
            end

            # Compute the MPB update for the value function and policy function
            V_mpb = maximum(V_k)
            G_kop_mpb = k_grid[argmax(V_k)]
            G_c_mpb = z * k^α * l_ss + (1 - δ) * k - G_kop_mpb
           
            if G_c_mpb < c_ss && G_kop_mpb > 0
                # Update value and policy functions 
                V_new = V_mpb
                G_kop_new = G_kop_mpb
                G_c_new = G_c_mpb
            else
                V_new, k_index = findmax(V_k)
                G_kop_new = k_grid[k_index]
                G_c_new = z * k^α * l_ss + (1 - δ) * k - G_kop_new
            end

            V_dist = max(V_dist, abs(V_new - V_current))
            V_updates += 1

            if V_new > V_current
                V[index] = V_new
                G_kop[index] = G_kop_new
                G_c[index] = G_c_new
            end
        end
    
        # Check if convergence is reached
        if V_dist < dist_tol
            break
        end
    end
    
    # Return value and policy functions
    return V, G_kop, G_c
end

par = Par()
V, G_kop, G_c = mpb(par)


plot(k_grid, V, label="Value Function")
plot!(k_grid, G_kop, label="Capital Policy Function")
