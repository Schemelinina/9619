#PS2_MPB_ errors to be corrected

using Parameters
@with_kw struct Par
    z::Float64=1.0;
    α::Float64 = 1/3;
    β::Float64 = 0.96;
    δ::Float64=0.9;
    η::Float64=1.0;
    σ::Float64=2.0;
        max_iter::Int64   = 1000  ; 
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



  #  VFI with Grid
       function K_grid(n_k,par::Par)
        # Get SS
        k_ss,y_ss,c_ss,r_ss,w_ss = steady(par)
        # Get k_grid
        k_grid = range(1E-5,2*k_ss;length=n_k) ; # Equally spaced grid between 0 and 2*k_ss
        # Return
        return k_grid
    end

 #Define utility function
 function utility(k,kop,par::Par)
    @unpack z, α, δ,η, σ = par
    c=z*0.4*k.^α  - kop +k.*(1-δ)  
return c.^(1-σ)/(1-σ)-x/(1+η)    
end

 #  Set MPB algorithm
    function VFI_grid_MPB(T::Function,k_grid,par::Par)
        @unpack β, max_iter, dist_tol = par
        # Initialize variables for loop
        n_k    = length(k_grid) ; # Number of grid nodes
        V_old  = zeros(n_k)     ; # Initial value, a vector of zeros
        iter   = 0              ; # Iteration index
        V_dist = 1              ; # Initialize distance
            for iter=1:max_iter
            # Update value function
            V_new, G_kop, G_c = T(V_old)
            # MPB and Distance
            MPB_l  = β/(1-β)*minimum(V_new-V_old)
            MPB_h  = β/(1-β)*maximum(V_new-V_old)
            V_dist = MPB_h - MPB_l
            # Update old function
            V_old  = V_new
            # Report progress
            if mod(iter,100)==0
                println("   VFI Loop: iter=$iter, dist=",100*V_dist,"%")
            end
            # Check Convergence
            if (V_dist<=dist_tol)
                # Recover value and policy functions
                V = V_old .+ (MPB_l+MPB_h)/2
                # Return
                println("VFI - Grid Search - MPB - n_k=$n_k")
                println("Iterations = $iter and Distance = ",100*V_dist)
                println("------------------------")
                println(" ")
                return V, G_kop, G_c
            end
        end
        # Report error for non-convergence
        error("Error in VFI")
    end

    # Solve VFI with MPB
function Solve_VFI_MPB(n_k,par::Par)
# Get Grid
        k_grid = K_grid(n_k,par)
        U_mat = [utility(k_grid[i],k_grid[j],par) for i in 1:n_k, j in 1:n_k]
        # Solve VFI
        V, G_kop, G_c = VFI_grid_MPB(x->T_grid_mat(x,U_mat,k_grid,par),k_grid,par)
        return V,G_kop, G_c, k_grid
    end

    # Execute Numerical VFI
    @time V_20, G_kop_20, G_20, k_grid_20 = Solve_VFI_MPB(20,par)
    @time V_50, G_kp_50, G_c_50, k_grid_50 = Solve_VFI_MPB(50,par)
    