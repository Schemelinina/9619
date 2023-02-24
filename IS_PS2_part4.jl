#PS2_Neoclassical growth model with labour
##This code is just to find x value  

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


