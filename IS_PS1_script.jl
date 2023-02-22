#PS1_Neoclassical growth model 

#Set Parameters
using Parameters
@with_kw struct Par
z::Float64=1;
α::Float64 = 1/3;
β::Float64 = 0.96;
  
    max_iter::Int64   = 200  ; 
    dist_tol::Float64 = 1E-9  ; # Tolerance for distance
end

# Put parameters into object par
par = Par()

function utility(k,kop,par::Par)
@unpack z, α = par
c = z*k.^α  - kop
if c>0
return log(c)
else
return -Inf
end
end


function SS_values(par::Par)
@unpack z, α, β = par
k_ss = (β*α*z)^(1/(1-α))
y_ss = z*k_ss^α
c_ss = y_ss - k_ss
r_ss = α*y_ss/k_ss
w_ss = (1-α)*y_ss
return k_ss,y_ss,c_ss,r_ss,w_ss
end

# Find steady states
k_ss,y_ss,c_ss,r_ss,w_ss = SS_values(par)
println(" ")
println("------------------------")
println(" Steady State variables")
println("   Quantities: k = $k_ss; y = $y_ss; c = $c_ss;")
println("   Prices:     r = $r_ss; w = $w_ss;")
println("------------------------")
println(" ")

# Find new steady states

function SS1_values(par::Par)
    @unpack z, α, β = par
    k_ss1 = 0.8*k_ss
    y_ss1 = z*1.05*k_ss1^α
    c_ss1 = y_ss1 - k_ss1
    r_ss1 = α*y_ss1/k_ss1
    w_ss1 = (1-α)*y_ss1
    return k_ss1,y_ss1,c_ss1,r_ss1,w_ss1
    end
    
# Find steady states after shocks
k_ss1,y_ss1,c_ss1,r_ss1,w_ss1 = SS1_values(par)
println(" ")
println("------------------------")
println(" Steady State variables")
println("   Quantities: k1 = $k_ss1; y1 = $y_ss1; c1 = $c_ss1;")
println("   Prices:     r1 = $r_ss1; w1 = $w_ss1;")
println("------------------------")
println(" ")


using Plots

# Set up the x-axis for the plots
k_vals = collect(range(0.9*k_ss1, stop=1.1*k_ss1, length=100))

# Plot the old steady state

c_vals = z*k_vals.^α - k_vals
plot(k_vals, c_vals, label="Old Steady State")

# Plot the new steady state
k1_vals = collect(range(0.9*k_ss1, stop=1.1*k_ss1, length=100))
c1_vals = z*1.05*k1_vals.^α - k1_vals
plot!(k1_vals, c1_vals, label="New Steady State")

# Set the plot labels
xlabel!("Capital Stock")
ylabel!("Consumption")
title!("Old and New Steady States")




