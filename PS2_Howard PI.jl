#PS2_Howard's PI - function to be corrected

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
        H_tol::Float64    = 1E-9; 
    end
    
    # Put parameters into object par
    par = Par()

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

    #Define utility function
function utility(k,kop,par::Par)
    @unpack z, α, δ,η, σ = par
    c=z*0.4*k.^α  - kop +k.*(1-δ)  
return c.^(1-σ)/(1-σ)-x/(1+η)    
end

#  VFI with Grid
function K_grid(n_k,par::Par)
    # Get SS
    k_ss,y_ss,c_ss,r_ss,w_ss = steady(par)
    # Get k_grid
    k_grid = range(1E-5,2*k_ss;length=n_k) ; # Equally spaced grid between 0 and 2*k_ss
    # Return
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


#Set VFI grid search with matrices

function T_grid_mat(V_old,U_mat,k_grid,par::Par)
    @unpack z, α, β,δ = par
    n_k    = length(V_old)
    V,G_kop = findmax( U_mat .+ β*repeat(V_old',n_k,1) , dims=2 )
    G_kop   = [G_kop[i][2] for i in 1:n_k] 
    G_c    = z*k_grid.^α .- k_grid[G_kop] +k_grid.*(1-δ)
    return V, G_kop, G_c
end


# Solve VFI with grid search and loops
function Solve_VFI_mat(n_k,par::Par)
    # Get Grid
    k_grid = K_grid(n_k,par)
    # Utility matrix
    U_mat = [utility(k_grid[i],k_grid[j],par) for i in 1:n_k, j in 1:n_k]
    # Solve VFI
    V, G_kop, G_c = VFI_grid(x->T_grid_mat(x,U_mat,k_grid,par),k_grid,par)
    # Return Solution
    return V,G_kop, G_c, k_grid
end


#Set function for Value update and policy functions for Howard PI
    function HT_grid_mat(V_old,U_mat,k_grid,par::Par,n_H)
        @unpack z, α, β,δ, H_tol = par
        # Get Policy Function
        n_k    = length(V_old)
        V,G_kop = findmax( U_mat .+ β*repeat(V_old',n_k,1) , dims=2 )
        V_old  = V
        # "Optimal" U for Howard's iteration
            U_vec = U_mat[G_kop]
        # Howard's policy iteration
            for i=1:n_H
            V = U_vec .+ β*repeat(V_old',n_k,1)[G_kop]
            if maximum(abs.(V./V_old.-1))<=H_tol
                break
            end
            V_old = V
        end
        # Recover Policy Functions
        G_kop   = [G_kop[i][2] for i in 1:n_k] 
        G_c    = z*k_grid.^α  .- k_grid[G_kop]+k_grid.*(1-δ).
        # Return output
        return V, G_kop, G_c
    end


# Solve VFI with Howard's policy iteration
    function Solve_VFI_HPI(n_H,n_k,par::Par)
        # Get Grid
        k_grid = K_grid(n_k,par)
        # Utility matrix
        U_mat = [utility(k_grid[i],k_grid[j],par) for i in 1:n_k, j in 1:n_k]
        # Solve VFI
        V, G_kop, G_c = VFI(x->HT_grid_mat(x,U_mat,k_grid,par,n_H),k_grid,par)
        # Return Solution
        return V,G_kop, G_c, k_grid
    end


 # Execute Numerical VFI
 @time V, G_kop, G_c, k_grid = Solve_VFI_HPI(20,20,par)
 @time V, G_kop, G_c, k_grid = Solve_VFI_HPI(20,50,par)
 
 using Plots
 gr()
 plot(k_grid,V,linewidth=3,title="Value Function",legend=(0.75,0.2),foreground_color_legend = nothing,background_color_legend = nothing)
 plot(k_grid,G_kop, title="Policy function")
 #plot(k_grid, G_c, title="Consumption")  

