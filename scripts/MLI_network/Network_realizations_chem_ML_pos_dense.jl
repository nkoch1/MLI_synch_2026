# ============================================================================
# Network Generation: MLI Connectivity with Chemical and Electrical Synapses
# ============================================================================
# Purpose: Generate multiple realizations of MLI (molecular layer interneuron)
#          network connectivity based on spatial positions and distribution fits.
#          Creates networks with realistic synaptic properties and connectivity
#          patterns stratified by dendritic layer (lower, mid, upper third).
#
# Output: Cell positions, connectivity matrices, and connectivity statistics
#         for multiple network realizations (100 seeds)
# ============================================================================

# ============================================================================
# SECTION 1: Load Dependencies and Plotting Setup
# ============================================================================
using DrWatson
@quickactivate  "MLI_synch_2026"
using PyPlot
include(srcdir("pyplot_fxns.jl"))
using .pyplot_fxns
using StatsBase
using SpecialFunctions
using Random
using Distributions
using DataFrames

# ============================================================================
# SECTION 2: Probability Distribution Functions
# ============================================================================
# Gamma probability density function
function gamma_pdf(x, p)
    k, θ, A = p  # k: shape, θ: scale, A: amplitude scaling
    coeff = 1 / (gamma(k) * θ^k)
    return coeff .* x.^(k - 1) .* exp.(-x ./ θ) * A
end

# Log-normal distribution 
# p = [θ (location shift), σ (shape), m (scale)]
lognormal_pdf(x, p) = 1 ./ ((x .- p[1]) .* p[2] .* sqrt(2 * π)) .* exp.(-((log.((x .- p[1]) ./ p[3])) .^ 2 ./ (2 .* p[2] .^ 2)))

# Reverse Heaviside step function (1 if x < shift, 0 otherwise)
function rev_heaviside(x, shift, scale;)
    if x - shift < 0
        return 1*scale 
    else
        return 0
    end
end
# Sample from custom log-normal distribution
function sample_custom_lognormal(p; n=1)
    θ, σ, m = p
    return θ .+ m .* exp.(σ .* randn(n))
end

# Sample from custom log-normal with specified RNG
function sample_custom_lognormal(p; n=1, rng=rng)
    θ, σ, m = p
    return θ .+ m .* exp.(σ .* randn(rng, n))
end


# ============================================================================
# SECTION 3: Load Pre-fitted Distribution Parameters
# ============================================================================
# Load distance distribution parameters (log-normal fit to MLI inter-soma distances)
param = wload(datadir("simulations", "Network", "MLI_distance_lognorm.jld2"), "param")
θ, σ, m = param
mean_lognormal = exp(log(m) + σ^2/2)  # Mean of log-normal distribution

# Load connection probability distribution parameters from experimental data
# Gamma distribution fits to synaptic and electrical coupling data
param_chem_sagittal = wload(datadir("simulations", "Network", "MLI_chem_sagittal_gamma.jld2"),  "param_chem_sagittal")
param_elec_sagittal = wload(datadir("simulations", "Network", "MLI_elec_sagittal_gamma.jld2"), "param_elec_sagittal")


# ============================================================================
# SECTION 4: Stratified Connectivity Data from Experimental Recordings
# ============================================================================
# Electrical connectivity probabilities (gap junction connection rates)
# Data from Rieubland et al. 2014 Figure S3
# Stratification: lower, mid, upper thirds of molecular layer
df_elec_thirds = DataFrame(to=["lower", "mid", "upper"], 
                          lower=[0.61, 0.46, 0.13], 
                          mid=[0.46, 0.48, 0.38], 
                          upper=[0.13, 0.46, 0.5])
# From cell type "From" to cell type "To" in different thirds
#                             From
#     |       |  lower    | mid       | upper     |
#     | lower |  0.61     |           |           |
# To  | mid   |  0.46     |  0.48     |           | 
#     | upper |  0.13     |  0.38     |  0.5      |

# Chemical connectivity probabilities (synaptic connection rates)
df_chem_thirds = DataFrame(to=["lower", "mid", "upper"],
                          lower=[0.31, 0.1, 0.], 
                          mid=[0.25, 0.20, 0.07], 
                          upper=[0.13, 0.27, 0.16])
# Chemical connectivity
#                             From
#     |       |  lower    | mid       | upper     |
#     | lower |  0.31     |  0.25     | 0.13      |
# To  | mid   |  0.10     |  0.20     | 0.27      | 
#     | upper |  0.0      |  0.07     |  0.16     |

# ============================================================================
# SECTION 5: Coordinate Transformation Utilities
# ============================================================================
# Convert polar coordinates (theta, r) to Cartesian (x, y)
function polar2cartesian(theta, r)
    x = r*cos(theta)
    y = r*sin(theta)
    return x, y
end

# Vectorized version for arrays
function polar2cartesian(theta::Vector, r::Vector)
    x = r.*cos.(theta)
    y = r.*sin.(theta)
    return x, y
end

# Vectorized version for matrices
function polar2cartesian(theta::Array, r::Array)
    x = r.*cos.(theta)
    y = r.*sin.(theta)
    return x, y
end

# Compute new coordinates given displacement
function compute_new_coord(delta_x, delta_y, x, y)
    return x + delta_x, y + delta_y
end


# ============================================================================
# SECTION 6: Distance and Neighbor Finding Utilities
# ============================================================================
# Calculate distances to n nearest neighbors for all cells
function calculate_distance_n_neighbors(x, y; n=3)
    m = size(x)[1]
    distances = []
    for i = 1:m 
        pairwise_ind = [[i, j] for j = 1:m if i != j]
        dist = ([sqrt.(diff(x[pairwise_ind[i]]).^2 .+ diff(y[pairwise_ind[i]]).^2)[1] for i in eachindex(pairwise_ind)])
        append!(distances, dist[sortperm(dist)[1:n]])
    end
    return distances
end

# Check if point is within specified bounds
function is_within_bounds(p, bounds)
    all(bounds[1] .<= p .<= bounds[2])
end

# Find indices and distances to n nearest neighbors
function dist_3_neighbors(x_i, y_i, x, y; n=3)
    dist = sqrt.((x_i .- x).^2 .+ (y_i .- y).^2)
    return sortperm(dist)[1:n], dist[sortperm(dist)[1:n]]
end

# Check if new position maintains minimum separation from existing cells
function greater_min_dist(x_i, y_i, x, y, min_sep)
    if isempty(x)
        return true
    elseif length(x) < 3
        return true
    end
    _, dist_min = dist_3_neighbors(x_i, y_i, x, y; n=1)
    return abs(dist_min[1]) > min_sep
end

# ============================================================================
# SECTION 7: Connectivity Detection Functions
# ============================================================================
# Compute Euclidean distance between two 2D points
@inline function distance(x1, y1, x2, y2)
    sqrt((x1 - x2)^2 + (y1 - y2)^2)
end

# Find pairs of points within radius (no connection info)
function find_close_points_xy(x::AbstractVector, y::AbstractVector, radius::Real)
    n = length(x)
    @assert length(y) == n "x and y must have the same length"
    results = Vector{Tuple{Int, Int, Float64}}()  # (i, j, distance)
    for i in 1:n-1
        for j in i+1:n
            d = distance(x[i], y[i], x[j], y[j])
            if d ≤ radius
                push!(results, (i, j, d))
            end
        end
    end
    return results
end

# Find pairs within radius WITH detailed connectivity info
# Includes: electrical/chemical coupling, direction, probabilities, stratification
function find_close_points_xy_connect(x::AbstractVector, y::AbstractVector, radius::Real, y_low_bound, y_mid_bound, y_upper_bound)
    n = length(x)
    @assert length(y) == n "x and y must have the same length"
    
    # Result tuples include: (i, j, distance, prob_sample_chem_up, chem_bool_up, 
    # prob_sample_chem_down, chem_bool_down, chem_direction, prob_sample_elec, 
    # elec_bool, bottom_ind, top_ind, chem_description)
    results = Vector{Tuple{Int, Int, Float64, Float64, Bool, Float64, Bool, Int, Float64, Bool, Int64, Int64, String}}()
    
    for i in 1:n-1
        for j in i+1:n
            d = distance(x[i], y[i], x[j], y[j])
            if d ≤ radius
                # Determine stratification (lower, mid, upper third) for cell i
                lower_bool_1 = is_within_bounds(y[i], y_low_bound) 
                mid_bool_1 = is_within_bounds(y[i], y_mid_bound)
                upper_bool_1 = is_within_bounds(y[i], y_upper_bound)
                if lower_bool_1
                    third_label_1 = "lower"
                elseif mid_bool_1
                    third_label_1 = "mid"
                elseif upper_bool_1
                    third_label_1 = "upper"
                end

                # Determine stratification for cell j
                lower_bool_2 = is_within_bounds(y[j], y_low_bound) 
                mid_bool_2 = is_within_bounds(y[j], y_mid_bound)
                upper_bool_2 = is_within_bounds(y[j], y_upper_bound)
                if lower_bool_2
                    third_label_2 = "lower"
                elseif mid_bool_2
                    third_label_2 = "mid"
                elseif upper_bool_2
                    third_label_2 = "upper"
                end

                # Get connection probabilities from experimental data based on stratification
                p_elec = df_elec_thirds[df_elec_thirds[!, "to"].== third_label_2, third_label_1][1]
                p_chem_1 = df_chem_thirds[df_chem_thirds[!, "to"].== third_label_2, third_label_1][1]  # i->j direction
                p_chem_2 = df_chem_thirds[df_chem_thirds[!, "to"].== third_label_1, third_label_2][1]  # j->i direction
                
                # Sample electrical coupling probability from distance-dependent gamma distribution
                prob_elec = gamma_pdf(d, param_elec_sagittal) * p_elec / 0.560
                elec_samp = rand(1)[1] 
                elec_bool = elec_samp < prob_elec
                
                # Sample chemical coupling (i->j direction) 
                prob_chem_1 = gamma_pdf(d, param_chem_sagittal) * p_chem_1 / 0.245
                chem_samp_1 = rand(1)[1] 
                chem_bool_1 = chem_samp_1 < prob_chem_1
                
                # Sample chemical coupling (j->i direction)
                prob_chem_2 = gamma_pdf(d, param_chem_sagittal) * p_chem_2 / 0.245
                chem_samp_2 = rand(1)[1] 
                chem_bool_2 = chem_samp_2 < prob_chem_2


                # Determine relative position and chemical direction
                bottom_ind = argmin(y[[i,j]])
                bottom_max = argmax(y[[i,j]])
                if y[i] < y[j]
                    lower_first = true
                else
                    lower_first = false
                end

                # Assign chemical synaptic direction based on bilateral coupling
                if chem_bool_1 && chem_bool_2
                    chem_desc = "both"  # Bidirectional synaptic coupling
                    chem_dir = 2  # Direction code for bilateral
                elseif chem_bool_1
                    if y[i] < y[j]
                        chem_desc = "up"  # i->j (upward in dendritic layer)
                        chem_dir = 1
                    else
                        chem_desc = "down"  # i->j (downward in dendritic layer)
                        chem_dir = -1
                    end
                elseif chem_bool_2
                    if y[i] < y[j]
                        chem_desc = "up"  # j->i which is downward relative to i
                        chem_dir = -1
                    else
                        chem_desc = "down"  # j->i which is upward relative to i
                        chem_dir = 1
                    end
                else 
                    chem_desc = "None"  # No chemical coupling
                    chem_dir = 0
                end
                
                # Store connectivity information
                bottom_ind = argmin(y[[i,j]])
                top_ind = argmax(y[[i,j]])
                if y[i] < y[j]
                    lower_first = true
                else
                    lower_first = false
                end

                push!(results, (i, j, d, chem_samp_1, chem_bool_1, chem_samp_2, chem_bool_2, chem_dir, elec_samp, elec_bool, bottom_ind, top_ind, chem_desc))
        end
    end

    return results
end


# ============================================================================
# SECTION 8: Set Simulation Domain and Initial Cell Generation
# ============================================================================
# Define spatial domain (micrometers)
x_bounds = (0, 3000)  # X extent (μm)
y_bounds = (0, 200)   # Y extent (μm) - depth in cerebellar molecular layer
delta_x = diff(collect(x_bounds))[1]  # Total X length
delta_y = diff(collect(y_bounds))[1]  # Total Y length

# Cell placement parameters
min_sep = 10  # Minimum separation between cell centers (μm)
MLI_dens = 28000  # MLI density (/mm³) from literature
# Calculate total number of cells based on density
nMLI = x_bounds[2] * y_bounds[2] * MLI_dens * (1e-3)^3 * 20  # cells/μm²
N = Int(nMLI)  # Total cells in network

# Define stratification boundaries (molecular layer divided into thirds)
y_lim = y_bounds[2]
y_low_bound = (0, y_lim*1/3)
y_mid_bound = (y_lim*1/3, y_lim*2/3)
y_upper_bound = (y_lim*2/3, y_lim)


# ============================================================================
# SECTION 12: Generate Multiple Network Realizations
# ============================================================================
# Initialize storage for multiple network realizations
nseeds = 100  # Number of network realizations to generate

# Storage arrays for all realizations
x_pos = Vector{Vector{Float64}}(undef, nseeds)
y_pos = Vector{Vector{Float64}}(undef, nseeds)
dist_3 = Vector{Vector{Float64}}(undef, nseeds)
num_chem_connect = Vector{Matrix{Int64}}(undef, nseeds)
num_elec_connect = Vector{Matrix{Int64}}(undef, nseeds)
num_chem_connect_total = Vector{Matrix{Int64}}(undef, nseeds)
num_elec_connect_total = Vector{Matrix{Int64}}(undef, nseeds)

# Storage for connectivity statistics per realization
mean_num_elec_connect = zeros(nseeds)
mean_num_elec_connect_lower = zeros(nseeds)
mean_num_elec_connect_mid = zeros(nseeds)
mean_num_elec_connect_upper = zeros(nseeds)
mean_num_chem_send_connect = zeros(nseeds)
mean_num_chem_send_connect_lower = zeros(nseeds)
mean_num_chem_send_connect_mid = zeros(nseeds)
mean_num_chem_send_connect_upper = zeros(nseeds)
mean_num_chem_receive_connect = zeros(nseeds)
mean_num_chem_receive_connect_lower = zeros(nseeds)
mean_num_chem_receive_connect_mid = zeros(nseeds)
mean_num_chem_receive_connect_upper = zeros(nseeds)
results_connect_array = Vector{Vector{Tuple{Float64, Float64, Vector{Int64}, Int64, Vector{Int64}, Int64, Vector{Int64}, Int64, String}}}(undef, nseeds)
chem_connections = Vector{Vector{Any}}(undef, nseeds)
elec_connections = Vector{Vector{Any}}(undef, nseeds)
mean_num_elec_connect_lower = zeros(nseeds);
mean_num_elec_connect_mid = zeros(nseeds);
mean_num_elec_connect_upper = zeros(nseeds);
mean_num_chem_send_connect = zeros(nseeds);
mean_num_chem_send_connect_lower = zeros(nseeds);
mean_num_chem_send_connect_mid = zeros(nseeds);
mean_num_chem_send_connect_upper = zeros(nseeds);
mean_num_chem_receive_connect = zeros(nseeds);
mean_num_chem_receive_connect_lower = zeros(nseeds);
mean_num_chem_receive_connect_mid = zeros(nseeds);
mean_num_chem_receive_connect_upper = zeros(nseeds);
results_connect_array = Vector{Vector{Tuple{Float64, Float64, Vector{Int64}, Int64, Vector{Int64}, Int64,  Vector{Int64}, Int64, String}}}(undef,nseeds);
chem_connections = Vector{Vector{Any}}(undef,nseeds)
elec_connections = Vector{Vector{Any}}(undef,nseeds)

# ============================================================================
# SECTION 13: Main Loop - Generate All Network Realizations
# ============================================================================
for s in range(1, nseeds)
    println(s)  # Progress indicator
    
    # Initialize random number generator with unique seed for each realization
    rng = Random.seed!(MersenneTwister(1), s)

    # Initialize first cell at domain center
    x_pos_i = [delta_x/2]
    y_pos_i = [delta_y/2] 
    i = 1
    
    # Generate cell positions using random walk with lognormal step sizes
    while length(x_pos_i) < N
        i = length(x_pos_i)
        i_rand = rand(rng, 1:size(x_pos_i, 1))
        if i > 1
            x_prev = x_pos_i[i-1]
            y_prev = y_pos_i[i-1]
        else
            x_prev = x_pos_i[i]
            y_prev = x_pos_i[i]
        end

        # Sample distance to next cell from lognormal distribution
        radius_i = sample_custom_lognormal(param; n=1, rng=rng)[1]^2

        # Sample random direction
        theta_i = rand(rng, Uniform(0, 2π), 1)[1]
        delta_x_i, _delta_y_i = polar2cartesian(theta_i, radius_i)
        x_i, y_i = compute_new_coord(delta_x_i, _delta_y_i, x_prev, y_prev)
        
        # Accept new cell if within bounds and maintains minimum separation
        if is_within_bounds((x_i), x_bounds) && is_within_bounds((y_i), y_bounds) && 
           greater_min_dist(x_i, y_i, x_pos_i, y_pos_i, min_sep) 
            x_pos_i = [x_pos_i..., x_i]
            y_pos_i = [y_pos_i..., y_i]
        end
    end

    # Calculate nearest neighbor distances
    dist_3_i = calculate_distance_n_neighbors(x_pos_i, y_pos_i; n=3)

    # Find and sample connectivity
    radius = 150.0
    results_connect_i = find_close_points_xy_connect(x_pos_i, y_pos_i, radius, y_low_bound, y_mid_bound, y_upper_bound)

    # Build connection lists
    chem_connections_i = []
    elec_connections_i = []
    for (i, j, d, chem_samp_up, chem_bool_up, chem_samp_down, chem_bool_down, chem_dir, elec_samp, elec_bool) in results_connect_i 
        if chem_bool_up || chem_bool_down
            push!(chem_connections_i, [i, j, chem_dir, d])
        end
        if elec_bool
            push!(elec_connections_i, [i, j, d])
        end
    end

    # Count connections
    num_chem_connect_i = stack([[key, value] for (key, value) in countmap(stack(chem_connections_i)[1:2, :])])
    num_chem_connect_i = num_chem_connect_i[:, sortperm(num_chem_connect_i[1, :])]
    num_elec_connect_i = stack([[key, value] for (key, value) in countmap(stack(elec_connections_i)[1:2, :])])
    num_elec_connect_i = num_elec_connect_i[:, sortperm(num_elec_connect_i[1, :])]

    # Compute per-cell connectivity statistics
    results_connect_array_dir_i = Vector{Tuple{Float64, Float64, Vector{Int64}, Int64, Vector{Int64}, Int64, Vector{Int64}, Int64, String}}(undef, N)
    for n in range(1, N)
        # Determine stratification
        lower_bool = is_within_bounds(y_pos_i[n], y_low_bound) 
        mid_bool = is_within_bounds(y_pos_i[n], y_mid_bound)
        upper_bool = is_within_bounds(y_pos_i[n], y_upper_bound)
        if lower_bool
            third_label = "lower"
        elseif mid_bool
            third_label = "mid"
        elseif upper_bool
            third_label = "upper"
        end

        # Find electrical connections
        elec_connections_s = stack(elec_connections_i)[1:2, findall(sum(stack(elec_connections_i)[1:2, :] .== n, dims=1)[1, :] .== 1)]
        elec_connections_s_ind = elec_connections_s[elec_connections_s .!= n]
        elec_connections_s_num = length(elec_connections_s_ind)

        # Find chemical connections (by direction)
        connections_s = stack(chem_connections_i)[:, findall(sum(stack(chem_connections_i)[1:2, :] .== n, dims=1)[1, :] .== 1)]
        connections_s_ind_send = connections_s[2, connections_s[3, :] .== 1 .|| connections_s[3, :] .== 2]
        connections_s_num_send = length(connections_s_ind_send)
        connections_s_ind_receive = connections_s[2, connections_s[3, :] .== -1 .|| connections_s[3, :] .== 2]
        connections_s_num_receive = length(connections_s_ind_receive)
        results_connect_array_dir_i[n] = (x_pos_i[n], y_pos_i[n], elec_connections_s_ind, elec_connections_s_num, connections_s_ind_send, connections_s_num_send, connections_s_ind_receive, connections_s_num_receive, third_label)
    end

    # Store results for this realization
    x_pos[s] = x_pos_i
    y_pos[s] = y_pos_i
    dist_3[s] = dist_3_i
    results_connect_array[s] = results_connect_array_dir_i
    chem_connections[s] = chem_connections_i
    elec_connections[s] = elec_connections_i
    
    # Compute statistics for this realization
    results_connect_array_dir_stack_i = stack(results_connect_array_dir_i)
    mean_num_elec_connect[s] = mean(results_connect_array_dir_stack_i[4, :])
    mean_num_elec_connect_lower[s] = mean(results_connect_array_dir_stack_i[4, results_connect_array_dir_stack_i[9, :] .== "lower"])
    mean_num_elec_connect_mid[s] = mean(results_connect_array_dir_stack_i[4, results_connect_array_dir_stack_i[9, :] .== "mid"])
    mean_num_elec_connect_upper[s] = mean(results_connect_array_dir_stack_i[4, results_connect_array_dir_stack_i[9, :] .== "upper"])

    mean_num_chem_send_connect[s] = mean(results_connect_array_dir_stack_i[6, :])
    mean_num_chem_send_connect_lower[s] = mean(results_connect_array_dir_stack_i[6, results_connect_array_dir_stack_i[9, :] .== "lower"])
    mean_num_chem_send_connect_mid[s] = mean(results_connect_array_dir_stack_i[6, results_connect_array_dir_stack_i[9, :] .== "mid"])
    mean_num_chem_send_connect_upper[s] = mean(results_connect_array_dir_stack_i[6, results_connect_array_dir_stack_i[9, :] .== "upper"])

    mean_num_chem_receive_connect[s] = mean(results_connect_array_dir_stack_i[8, :])
    mean_num_chem_receive_connect_lower[s] = mean(results_connect_array_dir_stack_i[8, results_connect_array_dir_stack_i[9, :] .== "lower"])
    mean_num_chem_receive_connect_mid[s] = mean(results_connect_array_dir_stack_i[8, results_connect_array_dir_stack_i[9, :] .== "mid"])
    mean_num_chem_receive_connect_upper[s] = mean(results_connect_array_dir_stack_i[8, results_connect_array_dir_stack_i[9, :] .== "upper"])
end

# ============================================================================
# SECTION 14: Save All Network Realizations
# ============================================================================
# Save all network realizations and connectivity statistics to file
wsave(
    datadir("simulations", "Network", "Network_realizations_dense_scaled_336.jld2"),
    @strdict N x_pos y_pos dist_3 num_chem_connect num_elec_connect num_chem_connect_total 
             num_elec_connect_total mean_num_elec_connect mean_num_elec_connect_lower 
             mean_num_elec_connect_mid mean_num_elec_connect_upper mean_num_chem_send_connect 
             mean_num_chem_send_connect_lower mean_num_chem_send_connect_mid mean_num_chem_send_connect_upper 
             mean_num_chem_receive_connect mean_num_chem_receive_connect_lower mean_num_chem_receive_connect_mid 
             mean_num_chem_receive_connect_upper results_connect_array chem_connections elec_connections
)