# ============================================================================
# Network Analysis: 2D Parameter Sweep - Gap Junction and Synaptic Conductances
# ============================================================================
# Purpose: Aggregate metrics (coactivity and phase-locking) across
#          a 2D parameter grid varying both gap junction (ggap) and synaptic (gsyn)
#          conductances. Includes NMDA synapses and analyzes pre/post-stimulus
#          phase-locking values (PLV) and secondary peaks in coactivity.
#
# Parameters tested:
#   - ggap: [0.0, 0.6, 1.2, 1.8, 2.4, 3.0] nS (gap junction conductance)
#   - gsyn: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] nS (synaptic conductance)
#
# Output: 3D arrays of coactivity, PLV, and peak statistics as functions of both parameters
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
# SECTION 1: Set Simulation Parameters and Parameter Arrays
# ============================================================================
num_surfaces = 100  # Number of network realizations to aggregate

# Fixed simulation parameters (same for all parameter combinations)
gAMPA = 1.2  # AMPA receptor conductance (nS)
bin_size = 2  # Bin size for histogram calculations (ms)
Ihold_scale = 0.2  # Scaling factor for holding current noise
syn_delay = 5  # Synaptic transmission delay (ms)
λ_rate_synch = 700.0  # input firing rate (Hz)
NMDA_scale = 1.0  # NMDA receptor scaling (1.0 = NMDA equals AMPA)
synch_dur = 2.0  # Duration of input (ms)
tau_GABA = 1.9  # Time constant for GABA synaptic decay (ms)

# Parameter sweep arrays for 2D grid search
ggap_array = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]  # Gap junction conductances (nS)
gsyn_array = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # Synaptic conductances (nS)


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
# SECTION 3: Initialize 3D Output Arrays for 2D Parameter Sweep Results
# ============================================================================
# Arrays for coactivity analysis
# Dimensions: [gsyn_index, ggap_index, surface_index] or [gsyn_index, ggap_index, 1]
peak_coactivity_array = zeros(length(gsyn_array), length(ggap_array), num_surfaces)
peak_coactivity_array_mean = zeros(length(gsyn_array), length(ggap_array), 1)  # Mean across surfaces
peak_coactivity_array_std = zeros(length(gsyn_array), length(ggap_array), 1)   # Std dev across surfaces

# Arrays for phase-locking value (PLV) analysis
# PLV before stimulus: pre-stimulus baseline control 
PLV_before_array = zeros(length(gsyn_array), length(ggap_array), num_surfaces)
PLV_mean_before_array = zeros(length(gsyn_array), length(ggap_array), 1)

# PLV in early post-stimulus window (0-50 ms)
PLV_after_50_array = zeros(length(gsyn_array), length(ggap_array), num_surfaces)
PLV_mean_after_50_array = zeros(length(gsyn_array), length(ggap_array), 1)

# Arrays for additional peak detection results
additional_peaks_array = zeros(length(gsyn_array), length(ggap_array), num_surfaces)  # Peak counts
additional_peaks_array_mean = zeros(length(gsyn_array), length(ggap_array), 1)  # Mean peak count
additional_peak_amp_array = Array{Array}(undef, length(gsyn_array), length(ggap_array))  # Peak amplitudes
additional_peak_prom_array = Array{Array}(undef, length(gsyn_array), length(ggap_array))  # Peak prominences



# ============================================================================
# SECTION 4: Load and Aggregate Results for 2D Parameter Grid
# ============================================================================
# Loop through all parameter combinations (gsyn outer loop, ggap inner loop)
i = 1
for gsyn_i in gsyn_array
    j = 1
    for ggap_i in ggap_array
        println("ggap: $(ggap_i), gsyn: $(gsyn_i)")
        
        # Create parameter dictionary for current grid point
        param = Dict(
            "gsyn" => gsyn_i,               # GABA synaptic conductance (nS)
            "ggap" => ggap_i,               # Gap junction conductance (nS)
            "gAMPA" => gAMPA,               # AMPA receptor conductance (nS)
            "Ihold_scale" => Ihold_scale,   # Holding current scaling
            "NMDA_scale" => NMDA_scale,     # NMDA receptor scaling
            "rate_synch" => λ_rate_synch,   # input rate (Hz)
            "synch_dur" => synch_dur,       # input duration (ms)
            "tau_GABA" => tau_GABA,         # input duration (ms)
        )
        
        # Load coactivity data
        peak_coactivity = wload(
            datadir("simulations", "Network", "new", savename(fname, param, "jld2")), 
            "peak_coactivity_array"
        )
        peak_coactivity_array[i, j, :] = peak_coactivity
        peak_coactivity_array_mean[i, j] = mean(peak_coactivity)
        peak_coactivity_array_std[i, j] = std(peak_coactivity)

        # Load phase-locking values (PLV) for different time windows
        PLV_before = wload(
            datadir("simulations", "Network", "new", savename(fname, param, "jld2")), 
            "PLV_mean_before_array"
        )
        PLV_after_50 = wload(
            datadir("simulations", "Network", "new", savename(fname, param, "jld2")), 
            "PLV_mean_after_50_array"
        )

        # Load secondary peak analysis
        additional_peaks = wload(
            datadir("simulations", "Network", "new", savename(fname, param, "jld2")), 
            "additional_peaks"
        )
        additional_peak_amp = wload(
            datadir("simulations", "Network", "new", savename(fname, param, "jld2")), 
            "additional_peak_amp"
        )
        additional_peak_prom = wload(
            datadir("simulations", "Network", "new", savename(fname, param, "jld2")), 
            "additional_peak_prom"
        )

        # Store PLV results
        PLV_before_array[i, j, :] = PLV_before
        PLV_after_50_array[i, j, :] = PLV_after_50

        PLV_mean_before_array[i,j] = mean(PLV_before)
        PLV_mean_after_50_array[i,j] = mean(PLV_after_50)

        additional_peaks_array_mean[i, j] = mean(additional_peaks)

        j += 1
    end
    i +=1
end

# ============================================================================
# SECTION 5: Save Aggregated 2D Parameter Grid Results
# ============================================================================
# Compile parameters used (excluding ggap and gsyn which form the grid)
param_save = Dict(
    "gAMPA" => gAMPA,              # AMPA receptor conductance (nS)
    "rate_synch" => λ_rate_synch,  # input rate (Hz)
    "NMDA_scale" => NMDA_scale,    # NMDA component scaling (equals AMPA)
    "synch_dur" => synch_dur,      # input duration (ms)
    "tau_GABA" => tau_GABA,        # GABA decay time constant (ms)
)

# Save all aggregated results to file
wsave(
    datadir("simulations", "Network", "new", savename(fname_save, param_save, "jld2")),
    @strdict fname_save peak_coactivity_array peak_coactivity_array_mean peak_coactivity_array_std 
             gsyn_array ggap_array PLV_before_array  PLV_after_50_array 
             PLV_mean_before_array  PLV_mean_after_50_array 
             additional_peaks_array additional_peaks_array_mean additional_peak_amp_array 
             additional_peak_prom_array
)




 