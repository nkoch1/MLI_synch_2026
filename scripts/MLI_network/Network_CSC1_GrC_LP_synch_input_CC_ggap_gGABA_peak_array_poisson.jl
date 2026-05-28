# ============================================================================
# Network Synchronization Analysis: CSC1-GrC-LP with Gap Junctions
# ============================================================================
# Purpose: Analyze peak coactivity across varying gap junction and synaptic 
#          conductance parameters in a network of CSC1 (cerebellar stellate cells),
#          GrC (granule cells), and LP neurons with Poisson input.
#
# Output: Arrays of peak coactivity values saved as function of ggap and gsyn
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
# SECTION 1: Load Network Geometry and Initialize Simulation Parameters
# ============================================================================

num_surfaces = 100  # Number of spatial surface locations (e.g., patches)
gAMPA = 1.2  # AMPA receptor conductance (nS)
bin_size = 2  # Bin size for histogram calculations (ms)
Ihold_scale = 0.2  # Scaling factor for holding current
syn_delay = 5.0  # Synaptic transmission delay (ms)
λ_rate_synch = 700.0  # Mean firing rate for input 
NMDA_scale = 0.0  # NMDA receptor scaling (0.0 = no NMDA component)
synch_dur = 2.0  # Duration of input (ms)
tau_GABA = 1.9  # Time constant for GABA synaptic decay (ms)

# Load pre-simulated network connectivity and cell positions
fname = "Network_realizations_dense_scaled_336.jld2"
ncells = wload(datadir("simulations", "Network", fname),  "N")  # Number of cells
x_pos = wload(datadir("simulations", "Network", fname),  "x_pos")  # Cell positions [surface, x, y, z]
num_surfaces = size(x_pos)[1]  # Update number of surfaces based on loaded data

# Base file names for loading and saving
fname = "Network_CSC1_NMDA_CC_poisson"  # Simulation results to load
fname_save = "Network_CSC1_nsynch_array_poisson"  # Output file name prefix

# Parameter sweep arrays for parameter exploration
ggap_array = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]  # Gap junction conductance values (nS)
gsyn_array = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # Synaptic conductance values (nS)



# ============================================================================
# SECTION 2: Initialize Output Arrays for Parameter Sweep Results
# ============================================================================
# Create 3D array to store peak coactivity across all parameter combinations
# Dimensions: [gsyn_index, ggap_index, surface_index]
peak_coactivity_array = zeros(length(gsyn_array), length(ggap_array), num_surfaces)

# Create 2D arrays for statistics across surfaces
# Dimensions: [gsyn_index, ggap_index]
peak_coactivity_array_mean = zeros(length(gsyn_array), length(ggap_array), 1)  # Mean coactivity
peak_coactivity_array_std = zeros(length(gsyn_array), length(ggap_array), 1)   # Std dev of coactivity



# ============================================================================
# SECTION 3: Load and Aggregate Simulation Results
# ============================================================================
# Loop through all parameter combinations to load pre-computed results
j = 1
for gsyn_i in gsyn_array
    i = 1
    for ggap_i in ggap_array
        println("ggap: $(ggap_i), gsyn: $(gsyn_i)")
        
        # Create parameter dictionary for current iteration
        # Note: conductances are specified in nS (nanoSiemens), not μA/cm²
        param = Dict(
            "gsyn" => gsyn_i,  # Synaptic conductance (nS)
            "ggap" => ggap_i,  # Gap junction conductance (nS)
            "gAMPA" => gAMPA,  # AMPA receptor conductance (nS)
            "Ihold_scale" => Ihold_scale,  # Holding current scaling
            "NMDA_scale" => NMDA_scale,  # NMDA receptor scaling
            "rate_synch" => λ_rate_synch,  # input rate 
            "synch_dur" => synch_dur,  # input duration (ms)
            "tau_GABA" => tau_GABA,  # GABA decay time constant (ms)
        )
        
        # Load pre-computed peak coactivity array from simulation results file
        peak_coactivity = wload(
            datadir("simulations", "Network", savename(fname, param, "jld2")), 
            "peak_coactivity_array"
        )
        
        # Store raw data (per-surface coactivity values)
        peak_coactivity_array[i, j, :] = peak_coactivity
        
        # Compute and store mean coactivity across all surfaces
        peak_coactivity_array_mean[i, j] = mean(peak_coactivity)
        
        # Compute and store standard deviation of coactivity across surfaces
        peak_coactivity_array_std[i, j] = std(peak_coactivity)
        
        i += 1
    end
    j += 1
end

# ============================================================================
# SECTION 4: Save Aggregated Results
# ============================================================================
# Compile parameters used (excluding gsyn and ggap which vary)
param_save = Dict(
    "gAMPA" => gAMPA,  # AMPA receptor conductance (nS)
    "NMDA_scale" => NMDA_scale,  # NMDA scaling (0.0 = disabled)
    "rate_synch" => λ_rate_synch,  # input rate (Hz)
    "synch_dur" => synch_dur,  # input duration (ms)
    "tau_GABA" => tau_GABA,  # GABA time constant (ms)
)

# Save all aggregated coactivity data and parameter arrays to file
wsave(
    datadir("simulations", "Network", savename(fname_save, param_save, "jld2")),  
    @strdict fname_save peak_coactivity_array peak_coactivity_array_mean 
             peak_coactivity_array_std gsyn_array ggap_array
)

