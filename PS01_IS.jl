#PS1_Neoclassical growth model 

using Parameters

#Set Parameters
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

function SS(par::Par)
@unpack z, α, β = par
k_ss = (β*α*z)^(1/(1-α))
y_ss = z*k_ss^α
c_ss = y_ss - k_ss
r_ss = α*y_ss/k_ss
w_ss = (1-α)*y_ss
return k_ss,y_ss,c_ss,r_ss,w_ss
end

# Find steady states
k_ss,y_ss,c_ss,r_ss,w_ss = SS(par)
println(" ")
println("------------------------")
println(" Steady State variables")
println("   Quantities: k = $k_ss; y = $y_ss; c = $c_ss;")
println("   Prices:     r = $r_ss; w = $w_ss;")
println("------------------------")
println(" ")

# Find new steady states

function SS1(par::Par)
    @unpack z, α, β = par
    k_ss1 = 0.8*k_ss
    y_ss1 = z*1.05*k_ss1^α
    c_ss1 = y_ss1 - k_ss1
    r_ss1 = α*y_ss1/k_ss1
    w_ss1 = (1-α)*y_ss1
    return k_ss1,y_ss1,c_ss1,r_ss1,w_ss1
    end
    
# Find steady states after shocks
k_ss1,y_ss1,c_ss1,r_ss1,w_ss1 = SS1(par)
println(" ")
println("------------------------")
println(" Steady State variables")
println("   Quantities: k1 = $k_ss1; y1 = $y_ss1; c1 = $c_ss1;")
println("   Prices:     r1 = $r_ss1; w1 = $w_ss1;")
println("------------------------")
println(" ")


using Plots

# Set up the x-axis for the plots
k_grid = collect(range(0.001, stop=2*k_ss, length=100))

# Plot the old steady state

c_vals = z*k_grid.^α - k_vals
plot(k_grid, c_vals, label="Old Steady State")


#Finding new paths
# Set the initial values
k0 = k_ss
c0 = c_ss
y0 = y_ss
r0 = r_ss
w0 = w_ss

# Define the function to compute the next period values
function SS1(par::Par, k0::Float64, c0::Float64)
    @unpack z, α, β= par
    k1 = z * 1.05*k0 ^ α + (1 - α) * k0 - c0
    y1 = z * k1 ^ α
    r1 = α * y1 / k1
    w1 = (1 - α) * y1
    c1 = β * r1 * c0
    return k1, c1, y1, r1, w1
end

# Iterate to compute the path from old to new steady state
k_path = [k0]
c_path = [c0]
y_path = [y0]
r_path = [r0]
w_path = [w0]

for i in 1:50
    @unpack z, α, β = par
      k1, c1, y1, r1, w1 = SS1(par, k_path[end], c_path[end])
    push!(k_path, k1)
    push!(c_path, c1)
    push!(y_path, y1)
    push!(r_path, r1)
    push!(w_path, w1)
end


# Plotting new path

plot(1:length(k_path), [k_path, c_path],
    label=["Capital" "Consumption"],
    title="Paths from old to new steady state")







