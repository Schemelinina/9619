#PS_04 TikTak. With errors 
#From the paper:

#Step 0. Initialization:
#1. Determine bounds for each parameter.
#2. Generate a sequence of Sobol’ points with length N .
#3. Evaluate the function value at each of these N Sobol’ points. Keep the set of N ∗
#Sobol’ points that have the lowest function values, and order them in descending
#order, as s1, . . . , sN ∗ , with f (s1) ≤ · · · ≤ f (sN ∗ ).
#4. Set the global iteration number to i = 1.

#Step 1. Global stage:
#1.Select the ith value (vector) in the Sobol’ sequence: s_i.
#2. If i > 1, read the function value (and corresponding parameter vector) of the smallest
#recorded local minimum from the “wisdom.dat” text file. Denote the lowest function
#value found so far (as of iteration i − 1) as f_low and the corresponding parameter
#vector as p_low.
#3. Generate a starting point (i.e., initial guess) S_i for the local search by using the
#convex combination of the Sobol’ point s_i and the parameter value p _low that gen-
#erated the best local minimum found so far: S_i = (1 − θi)s_i + θ p_low. The weight
#parameter θi ∈ [0, θ] with θ < 1 increases with i.

#Step 2: Local stage:
#– Select a local optimizer Nelder-Mead and implement a local search at the identified starting point Si until a local minimum
#is found.
#– Select a stopping criterion for the local search algorithm (in this paper, we use
#tolerances of either 10^(-3) as convergence criteria).
#– Open the wisdom.dat file and record the local minimum (function value and pa-
#rameters).

#Step 3. Stopping rule:
#– Repeat Steps 1 and 2 until local searches are completed from starting points that
#use each of the N ∗ Sobol’ points.
#– Return the point with the lowest function value as global minimum.

using Random
using Parameters
using Sobol
using Optim 

# Define the Griewank fn with n=2
function griewank(x)
  n = length(x)
  sum_term = sum((x[i]^2)/2000 for i in 1:n)
  prod_term = prod(cos(x[i]/sqrt(i)) for i in 1:n)
  return 1 + sum_term - prod_term
end

# Set the bounds
lb = [-100.0, -100.0]
ub = [100.0, 100.0]
N = 2000

# Generate Sobol seq
sobol_gen = SobolSeq(2)
sobol_pts = [next!(sobol_gen) for _ in 1:N]

# Evaluate fn
fn_val = zeros(N)
for i in 1:N
  x = (ub .- lb) .* sobol_pts[i] .+ lb
  fn_val[i] = griewank(x)
end

# Sort Sobol pnt and fn values
sort_ind = sortperm(fn_val)
sobol_pts_sort = sobol_pts[sort_ind,:]
fn_val_sort = fn_val[sort_ind]

# Number of local searches
n = N

# step 1: global stage
# Set initial values
i = 1
θ = 0.3
p_low = [-100.0, -100.0]
f_low = griewank(p_low)

# Matrix to store results
matrix = zeros(N, 3)

while i <= N
  s_i = sobol_pts_sort[i,:]
##Error to fix
  S_i = (1 .- θ*i) .* s_i .+ (θ*i) .* p_low'

  # step 2: local stage
  for j in 1:N
    x = (1 .- θ) .* sobol_pts[j, :] .+ θ .* p_low'
      res = optimize(x -> griewank(x), lb, ub, x, Optim.Options(iterations=100), method = NelderMead())
      f = res.minimum
      if f < f_low
          f_low = f
          best_x = res.minimizer
      end
      matrix[j, 1:3] = [f, x[1], x[2]]
  end

  i += 1
end

# Find best solution
f_best, x_best_idx = findmin(matrix[:, 1])
x_best = matrix[x_best_idx, 2:3]

println("Best sol: ", x_best)
println("Min: ", f_best)

using Plots
gr()
x = range(-100, 100, length=1000)
y = range(-100, 100, length=1000)
Z = [griewank([x[i], y[j]]) for j in eachindex(y), i in eachindex(x)]

pl = plot(x, y, Z, st=:contourf, title="Griewank fn")
contour!(x, y, Z, levels=50)
display(pl)


# Griewark for n=5

function griewank(x)
  n = length(x)
  sum_term = sum((x[i]^2)/2000 for i in 1:n)
  prod_term = prod(cos(x[i]/sqrt(i)) for i in 1:n)
  return 1 + sum_term - prod_term
end

n = 5
lb = [-100.0 for _ in 1:n]
ub = [100.0 for _ in 1:n]

sobol_gen = SobolSeq(n)
sobol_pts = [next!(sobol_gen) for _ in 1:N]

fn_val = zeros(N)
for i in 1:N
  x = (ub .- lb) .* sobol_pts[i] .+ lb
  fn_val[i] = griewank(x)
end

sort_ind = sortperm(fn_val)
sobol_pts_sort = sobol_pts[sort_ind,:]
fn_val_sort = fn_val[sort_ind]

n = N

# Set initial values
i = 1
θ = 0.3
p_low = [-100.0 for _ in 1:n]
f_low = griewank(p_low)

# Matrix to store results
matrix = zeros(N, n + 1)

while i <= N
  s_i = sobol_pts_sort[i,:]
  # the same error
  S_i = (1 .- θ*i) .* s_i .+ (θ*i) .* p_low'

  # step 2: local stage
  for j in 1:N
    x = (1 .- θ) .* sobol_pts[j, :] .+ θ .* p_low'
    res = optimize(x -> griewank(x), lb, ub, x, Optim.Options(iterations=100), method = NelderMead())
    f = res.minimum
    if f < f_low
      f_low = f
      best_x = res.minimizer
    end
    matrix[j, 1:(n+1)] = [f, best_x...]
  end

  i += 1
end

f_best, x_best_idx = findmin(matrix[:, 1])
x_best = matrix[x_best_idx, 2:(n+1)]

println("Best sol", x_best)
println("Min", f_best)



# Rastrigin fn n=2

function rastrigin(x)
  n = length(x)
  A = 10
  return A * n + sum(x[i]^2 - A * cos(2 * pi * x[i]) for i in 1:n)
end

# Set bounds
n = 2
lb = [-5.12 for _ in 1:n]
ub = [5.12 for _ in 1:n]
N = 2000

# Generate Sobol seq
sobol_gen = SobolSeq(n)
sobol_pts = [next!(sobol_gen) for _ in 1:N]

# Evaluate the fn
fn_val = zeros(N)
for i in 1:N
  x = (ub .- lb) .* sobol_pts[i] .+ lb
  fn_val[i] = rastrigin(x)
end

# Sort Sobol pnts and fn values
sort_ind = sortperm(fn_val)
sobol_pts_sort = sobol_pts[sort_ind,:]
fn_val_sort = fn_val[sort_ind]

# Number of local searches
n = N

# step 1: global stage
# Set initial values
i = 1
θ = 0.3
p_low = [-5.12, -5.12]
f_low = rastrigin(p_low)

# Matrix to store results
matrix = zeros(N, 3)

while i <= N
  s_i = sobol_pts_sort[i,:]
  #error
  S_i = (1 .- θ*i) .* s_i .+ (θ*i) .* p_low'

  # step 2: local stage
  for j in 1:N
      x = (1 .- θ) .* sobol_pts[j, :] .+ θ .* p_low'
      res = optimize(x -> rastrigin(x), lb, ub, x, Optim.Options(iterations=100), method = NelderMead())
      f = res.minimum
      if f < f_low
          f_low = f
          best_x = res.minimizer
      end
      matrix[j, 1:3] = [f, x[1], x[2]]
  end

  i += 1
end

# Find best solution
f_best, x_best_idx = findmin(matrix[:, 1])
x_best = matrix[x_best_idx, 2:3]

println("Best sol", x_best)
println("Min", f_best)

# Plotting
using Plots
gr()
x = range(-5.12, 5.12, length=1000)
y = range(-5.12, 5.12, length=1000)
Z = [rastrigin([x[i], y[j]]) for j in eachindex(y), i in eachindex(x)]

pl_rast = plot(x, y, Z, st=:contourf, title="Rastrigin fn")
contour!(x, y, Z, levels=50)
display(pl_rast)

# Rastrigin fn with n=5

function rastrigin(x)
  n = length(x)
  A = 10
  return A * n + sum(x[i]^2 - A * cos(2 * pi * x[i]) for i in 1:n)
end

n = 5
lb = [-5.12 for _ in 1:n]
ub = [5.12 for _ in 1:n]
N = 2000

# Generate Sobol seq
sobol_gen = SobolSeq(n)
sobol_pts = [next!(sobol_gen) for _ in 1:N]

# Evaluate the fn
fn_val = zeros(N)
for i in 1:N
  x = (ub .- lb) .* sobol_pts[i] .+ lb
  fn_val[i] = rastrigin(x)
end

sort_ind = sortperm(fn_val)
sobol_pts_sort = sobol_pts[sort_ind,:]
fn_val_sort = fn_val[sort_ind]

# Number of local searches
n = N

# step 1: global stage
# Set initial values
i = 1
θ = 0.3
p_low = [-5.12 for _ in 1:5]
f_low = rastrigin(p_low)

# Matrix to store results
matrix = zeros(N, 6)

while i <= N
  s_i = sobol_pts_sort[i,:]
  #error
  S_i = (1 .- θ*i) .* s_i .+ (θ*i) .* p_low'

  # Step 2: Local stage
  for j in 1:N
      x = (1 .- θ) .* sobol_pts[j, :] .+ θ .* p_low'
      res = optimize(x -> rastrigin(x), lb, ub, x, Optim.Options(iterations=100), method = NelderMead())
      f = res.minimum
      if f < f_low
          f_low = f
          best_x = res.minimizer
      end
      matrix[j, 1:6] = [f, x...]
  end

  i += 1
end

# Find best solution
f_best, x_best_idx = findmin(matrix[:, 1])
x_best = matrix[x_best_idx, 2:6]

println("Best sol", x_best)
println("Min", f_best)




