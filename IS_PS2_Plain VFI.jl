#PS2_Neoclassical growth model with labour 

using Parameters
@with_kw struct Par
    z::Float64=1.0;
    α::Float64 = 1/3;
    β::Float64 = 0.96;
    δ::Float64=0.9;
    η::Float64=1.0;
    σ::Float64=2.0;
        max_iter::Int64   = 500  ; 
        dist_tol::Float64 = 1E-9  ; # Tolerance for distance
        end
    
    # Put parameters into object par
    par = Par()

# Find χ if l_ss=0.4

    function steady(par::Par)
    @unpack z, α, β, σ, δ, η = par
    k_ss = (1/β-1+δ)^(1/(α-1))/(α*z)^(1/α-1)
    y_ss = z*k_ss^α
    c_ss = y_ss - k_ss
    r_ss = 1/β-1+δ
    w_ss = (1-α)*y_ss
    x=(1-α)*c_ss^(-σ)*z*k_ss^α*0.4^(-α-η)
    return k_ss,y_ss,c_ss,r_ss,w_ss,x
end

# Find steady states and χ
k_ss,y_ss,c_ss,r_ss,w_ss,x = steady(par)
println(" ")
println("------------------------")
println(" Steady State variables")
println("   k = $k_ss; y = $y_ss; c = $c_ss; x=$x;")
println("   Prices:     r = $r_ss; w = $w_ss;")
println("------------------------")
println(" ")


#Perform VFI plain with grid

#Define utility function
function utility(k,kop,par::Par)
    @unpack z, α, δ,η, σ = par
    c=z*0.4*k.^α  - kop +k.*(1-δ)  
return c.^(1-σ)/(1-σ)-x/(1+η)    
end


#Define grid
function K_grid(n_k,par::Par)
k_ss,y_ss,c_ss,r_ss,w_ss = steady(par)
k_grid = range(1E-5,2*k_ss;length=n_k) ; # Equally spaced grid between 0 and 2*k_ss
return k_grid
end

#Build function to solve VFI 

function VFI(T::Function,k_grid,par::Par)
@unpack max_iter, dist_tol = par
n_k = length(k_grid) ; # Number of grid nodes
V_old = zeros(n_k)     ; # Initial value, a vector of zeros
V_dist = 1              ; # Initialize distance
for iter=1:max_iter
# Update value function
V_new, G_kop, G_c = T(V_old)
# Update distance, iterations and old function
V_dist = maximum(abs.(V_new./V_old.-1))
V_old  = V_new
if mod(iter,100)==0
    println("   VFI Loop: iter=$iter, dist=",100*V_dist,"%")
end
if V_dist<=dist_tol
println("VFI - Grid Search - n_k=$n_k")
    println("Iterations = $iter and Distance = ",100*V_dist,"%")
 ("------------------------")
    println(" ")
    return V_new, G_kop, G_c
end
end
# If no convergence, print error
error("Error in VFI")
end

# Define function for Value update and policy functions
function T_grid_loop(V_old,k_grid,par::Par)
    @unpack z, α, β, δ= par
    n_k  = length(k_grid)
    V    = zeros(n_k)
    G_kop = fill(0,n_k)
    G_c  = zeros(n_k)
    for i = 1:n_k
        V_aux = zeros(n_k) ; # Empty vector for auxiliary value of V(i,j)
        for j = 1:n_k
            # Evaluate potential value function for combinations of
            # current capital k_i and future capital k_j
            V_aux[j] = utility(k_grid[i],k_grid[j],par) + β*V_old[j]
            println(V_aux[j]," ",k_grid[i]," ",k_grid[j])
        end
        # Choose maximum value given current capital k_i
        V[i], G_kop[i] = findmax(V_aux)
        G_c[i] = z*0.4*k_grid[i].^α + k_grid[i].*(1-σ) - k_grid[G_kop[i]])
    end
    return V, G_kop, G_c
end


# Solve VFI with grid search and loops
function VFI_loop(n_k,par::Par)
    k_grid = K_grid(n_k,par)
    # Solve VFI
    V, G_kop, G_c = VFI(x->T_grid_loop(x,k_grid,par),k_grid,par)
    # Return Solution
    return V,G_kop, k_grid, G_c
end


  # Execute Numerical VFI
  @time V20, G_kop20, G_c20, k_grid20 = VFI_loop(20,par)

    

using Plots
gr()
plot(k_grid20,V20,linewidth=3,title="Value Function",legend=(0.75,0.2),foreground_color_legend = nothing,background_color_legend = nothing)
plot(k_grid20,G_kop20, title="Policy function")
##plot(k_grid20, G_c20, title="Consumption")  






