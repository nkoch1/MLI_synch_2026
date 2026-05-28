# ============================================================================
# Network Synchronization Analysis: Peak Coactivity vs Input Rate
# ============================================================================
# Purpose: Aggregate peak coactivity results across multiple  input
#          rates (lambda_n_synch = 100-1000) for a network with specified
#          gap junction and synaptic conductances. Post-processes pre-computed
#          simulation data to extract coactivity statistics.
#
# Output: Arrays of peak coactivity values as function of input rate (n_array)
# ============================================================================

# Load dependencies and project setup
using DrWatson
@quickactivate  "MLI_synch_2026"
include(srcdir("CSC1_network_poisson.jl"))
using .CSC1_network_poisson
include(srcdir("analysis.jl"))
using .analysis
using DifferentialEquations, DiffEqCallbacks,
    Optimization, SciMLSensitivity,
    Zygote,  DiffEqCallbacks, JLD2, Statistics, Peaks
using FHist
using PyCall, PyPlot
using StatsBase
using Random
using DSP
using CircStats
using LinearAlgebra


# ============================================================================
# SECTION 1: Set Fixed Network Parameters
# ============================================================================
# These parameters define the network connectivity (gap junctions and synapses)
# and are held constant across all simulations in this analysis

num_surfaces = 100  # Number of spatial network realizations to aggregate

# Network connectivity parameters (in nanoSiemens, nS)
ggap = 0.6    # Gap junction conductance (nS)
gsyn = 1.0    # GABA synaptic conductance (nS)
gAMPA = 1.2   # AMPA receptor conductance (nS)

# Simulation parameters (fixed for this analysis)
Ihold_scale = 0.2  # Scaling factor for holding current noise
synch_dur = 2.0    # Duration of input (ms)
tau_GABA = 1.9    # Time constant for GABA synaptic decay (ms)
NMDA_scale = 0.0  # NMDA receptor scaling (0.0 = no NMDA component)
syn_delay = 5     # Synaptic transmission delay (ms)


# ============================================================================
# SECTION 2: Load Network Geometry and Define File Paths
# ============================================================================
# Load pre-computed network structure (cell positions, connectivity)
fname = "Network_realizations_dense_scaled_336.jld2"

# Extract network properties
ncells = wload(datadir("simulations", "Network", fname), "N")  # Number of cells per surface
x_pos = wload(datadir("simulations", "Network", fname), "x_pos")  # Cell positions
num_surfaces = size(x_pos)[1]  # Update number of surfaces from loaded data

# File names for input and output
fname = "Network_CSC1_NMDA_CC_poisson" 
fname_save = "Network_CSC1_nsynch_array_poisson"

# ============================================================================
# SECTION 3: Define Parameter Sweep Array (Input Rates)
# ============================================================================
# Sweep input rate from 100 Hz to 1000 Hz
n_array = [100., 200., 300., 400., 500., 600., 700., 800., 900., 1000.]


# ============================================================================
# SECTION 4: Initialize Output Arrays for Results
# ============================================================================
# Create 2D array to store peak coactivity for each input rate and surface
# Dimensions: [n_array_index, surface_index]
peak_coactivity_array = zeros(length(n_array), num_surfaces)

# Create 1D arrays to store aggregated statistics across surfaces
# Dimensions: [n_array_index]
peak_coactivity_array_mean = zeros(length(n_array), 1)  # Mean coactivity per input rate
peak_coactivity_array_std = zeros(length(n_array), 1)   # Std dev of coactivity per input rate

# ============================================================================
# SECTION 5: Load and Aggregate Simulation Results
# ============================================================================
# Loop through all input rates and load pre-computed results
j = 1
for λ_rate_synch_i in n_array
    println("synch_mult: $(λ_rate_synch_i)")

    # Create parameter dictionary matching the input rate value
    # These parameters must match exactly the simulations that were run
    param = Dict(
        "gsyn" => gsyn,                 # GABA synaptic conductance (nS)
        "ggap" => ggap,                 # Gap junction conductance (nS)
        "gAMPA" => gAMPA,               # AMPA receptor conductance (nS)
        "Ihold_scale" => Ihold_scale,   # Holding current scaling
        "NMDA_scale" => NMDA_scale,     # NMDA component scaling
        "rate_synch" => λ_rate_synch_i, # input rate (Hz)
        "synch_dur" => synch_dur,       # input duration (ms)
        "tau_GABA" => tau_GABA,         # GABA decay time constant (ms)
    )

    # Load pre-computed peak coactivity array from simulation results file
    # This was computed across all num_surfaces network realizations
    peak_coactivity = wload(
        datadir("simulations", "Network", savename(fname, param, "jld2")), 
        "peak_coactivity_array"
    )
    
    # Store all per-surface values
    peak_coactivity_array[j, :] = peak_coactivity
    
    # Compute and store mean coactivity across all surfaces
    peak_coactivity_array_mean[j] = mean(peak_coactivity)
    
    # Compute and store standard deviation of coactivity across surfaces
    peak_coactivity_array_std[j] = std(peak_coactivity)
    
    j += 1
end


# ============================================================================
# SECTION 6: Save Aggregated Results to File
# ============================================================================
# Compile parameters used (excluding rate_synch which varies)
param_save = Dict(
    "I" => 0.0,                    # Current injection (none)
    "gsyn" => gsyn,                # GABA synaptic conductance (nS)
    "ggap" => ggap,                # Gap junction conductance (nS)
    "gAMPA" => gAMPA,              # AMPA receptor conductance (nS)
    "synch_dur" => synch_dur,      # input duration (ms)
    "NMDA_scale" => NMDA_scale,    # NMDA component scaling
    "tau_GABA" => tau_GABA,        # GABA decay time constant (ms)
)

# Save all aggregated coactivity data and parameter arrays to file
# Output structure: fname_save, peak_coactivity_array (all surfaces, all rates),
#                   peak_coactivity_array_mean (average across surfaces per rate),
#                   peak_coactivity_array_std (variability across surfaces per rate),
#                   n_array (input rates tested)
wsave(
    datadir("simulations", "Network", savename(fname_save, param_save, "jld2")),
    @strdict fname_save peak_coactivity_array peak_coactivity_array_mean 
             peak_coactivity_array_std n_array
)


