#PS05 Q1 Tauchen Code was adopted from Sergio and internet

using Parameters
using Random
using Distributions
using LinearAlgebra
using Statistics
using StatsBase

# Set random seed
Random.seed!(3486);

#-----------------------------------------------------------
# Define a markov process struct
    # Generate structure for markov processes 
    @with_kw struct MP
        # Model Parameters
        N::Int64 # Number of states
        grid     # Grid of discrete markov process
        Π        # Transition matrix
        PDF      # Stationary distribution
        CDF      # Stationary distribution
    end

#Q1a Simulate MC
# Define parameters
n = 10000  # Number of draws
ρ = 0.9
σ = 0.1

# Set initial value
z = zeros(n)
z[1] = 0.0

# Define distribution for error
η = Normal(0.0, σ)

# Generate AR1 using draws
for i in 2:n
    z[i] = ρ * z[i-1] + rand(η)
end

#Q1b; std_dev = sqrt(σ^2 / (1 - ρ^2))
#This function uses the Tauchen method to discretize an AR(1) process with parameters ρ and σ into a Markov chain with N states. 
#N=5

function tauchen86_5(ρ,σ,N,Ω::Any=3)
    # Create z grid
        z = range(-Ω*σ/sqrt(1-ρ^2),Ω*σ/sqrt(1-ρ^2),length=N)
    N=5
        # Define intermediate step length
        h = (z[2]-z[1])/2
    # Define auxiliary matrices
        z_0 = repeat(z ,1,N) # Matrix of today's z each row is a value, columns are equal
        z_1 = repeat(z',N,1) # Matrix of tomorrow's z each column is a value, rows are equal
    # Define intervals
        z_lim = zeros(N,N,2) # First matrix is lower bounds. Second matrix is uppor bounds.
        z_lim[:,1      ,1] .= -Inf
        z_lim[:,2:end  ,1] .=  ( z_1[:,2:end  ] - ρ*z_0[:,2:end  ] .- h )./σ
        z_lim[:,1:end-1,2] .=  ( z_1[:,1:end-1] - ρ*z_0[:,1:end-1] .+ h )./σ
        z_lim[:,end    ,2] .=  Inf
    # Define reference distribution
            F(x) = cdf.(Normal(),x)
    # Fill in transition matrix
        Π_z = F.(z_lim[:,:,2]) - F.(z_lim[:,:,1])
        Π_z = Π_z./repeat(sum(Π_z,dims=2),1,N)
    # Get stationary distribution of markov chain
        PDF_z = real(eigvecs(Π_z')[:,end]); PDF_z = PDF_z/sum(PDF_z) ;
        CDF_z = cumsum(PDF_z)
    # Return
        return MP(N=N,grid=z,Π=Π_z,PDF=PDF_z,CDF=CDF_z)
end


# Simulate the Markov chain
#simulates a Markov chain with Ns periods using the transition matrix Π from the MP structure. It returns a vector z_MC of simulated values of the Markov chain.

function simulation_5(Ns,MP::MP)
    # Compute conditional CDF
    Γ = cumsum(MP.Π,dims=2)
    # Allocate simulated vector
    z_ind    = zeros(Int64,Ns)
    z_MC     = zeros(Ns)
    # Starting value for simulation
    z_ind[1] = Int(ceil(length(MP.grid)/2))
    z_MC[1]  = MP.grid[z_ind[1]]
    # Simulate
    for i=2:Ns # number of simulated periods
        z_ind[i] = rand(Categorical(MP.Π[z_ind[i-1],:]))
        z_MC[i]  = MP.grid[z_ind[i]]
    end
       # Return result
    return z_MC
end

#Q1c
# Moments function for sample
function moments_5(z_MC)
    mean_MC      = mean(z_MC)
    std_MC       = std(z_MC)
    skewness_MC  = skewness(z_MC)
    kurtosis_MC  = kurtosis(z_MC)
              auto_corr_MC = cor(z_MC[1:end-1],z_MC[2:end])
              auto_corr_MC4 = zeros(4)
              for i in 2:5
                  auto_corr_MC4[i-1] = cor(z_MC[1:end-i],z_MC[1+i:end])[1]
              end
    return mean_MC, std_MC, skewness_MC, kurtosis_MC,auto_corr_MC4
end

#Report
Ns=10000
MP_T5  = tauchen86_5(ρ,σ,5)
z_MC = simulation_5(Ns, MP_T5)
mean_MC, std_MC, skewness_MC, kurtosis_MC,auto_corr_MC4 = moments_5(z_MC)


# Print the results
using DataFrames

function table_5(mean_MC, std_MC, skewness_MC, kurtosis_MC, auto_corr_MC4)
        df = DataFrame(Statistics=["Mean", "St.dev.", "Skewness", "Kurtosis", "Autocorr1", "Autocorr2", "Autocorr3", "Autocorr4"],
                    Values=[mean_MC, std_MC, skewness_MC, kurtosis_MC, auto_corr_MC4[1], auto_corr_MC4[2], auto_corr_MC4[3], auto_corr_MC4[4]])
    return df
end

df_Results_5 = table_5(mean_MC, std_MC, skewness_MC, kurtosis_MC, auto_corr_MC4)
println(df_Results_5)

using Plots
gr()

# Histograms ## There is error 
#histogram(z_MC)



#The same as above but for N=15

function tauchen86_15(ρ,σ,N,Ω::Any=3)
    # Create z grid
        z = range(-Ω*σ/sqrt(1-ρ^2),Ω*σ/sqrt(1-ρ^2),length=N)
    N=15
        # Define intermediate step length
        h = (z[2]-z[1])/2
    # Define auxiliary matrices
        z_0 = repeat(z ,1,N) # Matrix of today's z each row is a value, columns are equal
        z_1 = repeat(z',N,1) # Matrix of tomorrow's z each column is a value, rows are equal
    # Define intervals
        z_lim = zeros(N,N,2) # First matrix is lower bounds. Second matrix is uppor bounds.
        z_lim[:,1      ,1] .= -Inf
        z_lim[:,2:end  ,1] .=  ( z_1[:,2:end  ] - ρ*z_0[:,2:end  ] .- h )./σ
        z_lim[:,1:end-1,2] .=  ( z_1[:,1:end-1] - ρ*z_0[:,1:end-1] .+ h )./σ
        z_lim[:,end    ,2] .=  Inf
    # Define reference distribution
            F(x) = cdf.(Normal(),x)
    # Fill in transition matrix
        Π_z = F.(z_lim[:,:,2]) - F.(z_lim[:,:,1])
        Π_z = Π_z./repeat(sum(Π_z,dims=2),1,N)
    # Get stationary distribution of markov chain
        PDF_z = real(eigvecs(Π_z')[:,end]); PDF_z = PDF_z/sum(PDF_z) ;
        CDF_z = cumsum(PDF_z)
    # Return
        return MP(N=N,grid=z,Π=Π_z,PDF=PDF_z,CDF=CDF_z)
end

# Simulate the Markov chain
n = 10000
MP_T15  = tauchen86_15(ρ,σ,15)

# Simulate the Markov chain
#simulates a Markov chain with Ns periods using the transition matrix Π from the MP structure. It returns a vector z_MC of simulated values of the Markov chain.

function simulation_15(Ns,MP::MP)
    # Compute conditional CDF
    Γ = cumsum(MP.Π,dims=2)
    # Allocate simulated vector
    z_ind    = zeros(Int64,Ns)
    z_MC15     = zeros(Ns)
    # Starting value for simulation
    z_ind[1] = Int(ceil(length(MP.grid)/2))
    z_MC[1]  = MP.grid[z_ind[1]]
    # Simulate
    for i=2:Ns # number of simulated periods
        z_ind[i] = rand(Categorical(MP.Π[z_ind[i-1],:]))
        z_MC15[i]  = MP.grid[z_ind[i]]
    end
       # Return result
    return z_MC15
end

#Q1c
# Moments function for sample
function moments_15(z_MC)
    mean_MC      = mean(z_MC)
    std_MC       = std(z_MC)
    skewness_MC  = skewness(z_MC)
    kurtosis_MC  = kurtosis(z_MC)
              auto_corr_MC = cor(z_MC[1:end-1],z_MC[2:end])
              auto_corr_MC4 = zeros(4)
              for i in 2:5
                  auto_corr_MC4[i-1] = cor(z_MC[1:end-i],z_MC[1+i:end])[1]
              end
    return mean_MC, std_MC, skewness_MC, kurtosis_MC,auto_corr_MC4
end

#Report
Ns=10000
MP_T5  = tauchen86_15(ρ,σ,15)
z_MC = simulation_15(Ns, MP_T5)
mean_MC, std_MC, skewness_MC, kurtosis_MC,auto_corr_MC4 = moments_15(z_MC)


# Print the results
using DataFrames

function table_15(mean_MC, std_MC, skewness_MC, kurtosis_MC, auto_corr_MC4)
        df = DataFrame(Statistics=["Mean", "St.dev.", "Skewness", "Kurtosis", "Autocorr1", "Autocorr2", "Autocorr3", "Autocorr4"],
                    Values=[mean_MC, std_MC, skewness_MC, kurtosis_MC, auto_corr_MC4[1], auto_corr_MC4[2], auto_corr_MC4[3], auto_corr_MC4[4]])
    return df
end

df_Results_15 = table_15(mean_MC, std_MC, skewness_MC, kurtosis_MC, auto_corr_MC4)
println(df_Results_15)

using Plots
gr()

# Histograms ## There is error 
histogram(z_MC)



