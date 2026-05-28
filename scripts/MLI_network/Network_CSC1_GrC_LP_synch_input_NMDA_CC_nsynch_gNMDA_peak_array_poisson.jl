# ============================================================================
# Network Analysis: 2D Parameter Sweep - Gap Junction and Synaptic Conductances
# ============================================================================
# Purpose: Aggregate metrics (coactivity and phase-locking) across
#          a 2D parameter grid varying both input
#          rates (lambda_n_synch = 200-1000) and NMDA (NMDA scale = 0-2)
#          conductances. Includes NMDA synapses and analyzes pre/post-stimulus
#          phase-locking values (PLV) and secondary peaks in coactivity.
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
using StatsBase
using Random 
using DSP
using CircStats
using LinearAlgebra

#%%
# ============================================================================
# SECTION 1: Set Fixed Network Parameters
# ============================================================================
# These parameters define the network connectivity (gap junctions and synapses)
# and are held constant across all simulations in this analysis

num_surfaces = 100  # Number of spatial network realizations to aggregate

# Network connectivity parameters (in nanoSiemens, nS)
ggap = 0.6    # Gap junction conductance (nS)
# ggap = 0.0    # Gap junction conductance (nS)
gsyn = 1.0    # GABA synaptic conductance (nS)
gAMPA = 1.2   # AMPA receptor conductance (nS)

# Simulation parameters (fixed for this analysis)
Ihold_scale = 0.2  # Scaling factor for holding current noise
synch_dur = 2.0    # Duration of input (ms)
tau_GABA = 1.9    # Time constant for GABA synaptic decay (ms)
NMDA_scale = 1.0  # NMDA receptor scaling (0.0 = no NMDA component)
syn_delay = 5     # Synaptic transmission delay (ms)


# Parameter sweep arrays for 2D grid search
n_array = [200., 400.,  600.,  800., 1000.] # AMPA  input rate 
NMDA_array = [0., 0.5, 1.0, 1.5, 2.0] # NMDA conductances scaling factor

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
fname = "Network_CSC1_NMDA_CC_poisson"  # Pre-computed simulation results to load
fname_save = "Network_CSC1_nsynch_array_poisson"  # Output file name prefix



# ============================================================================
# SECTION 3: Initialize 3D Output Arrays for 2D Parameter Sweep Results
# ============================================================================
# Arrays for coactivity analysis
# Dimensions: [n_index, NMDA_index, surface_index] or [n_index, NMDA_index, 1]
peak_coactivity_array = zeros(length(n_array), length(NMDA_array), num_surfaces)
peak_coactivity_array_mean = zeros(length(n_array),length(NMDA_array),  1)
peak_coactivity_array_std = zeros(length(n_array),length(NMDA_array), 1)

# Arrays for phase-locking value (PLV) analysis
# PLV before stimulus: pre-stimulus baseline control 
PLV_before_array = zeros(length(n_array),length(NMDA_array), num_surfaces)
PLV_mean_before_array = zeros(length(n_array),length(NMDA_array), 1)


# PLV in early post-stimulus window (0-50 ms)
PLV_after_50_array = zeros(length(n_array),length(NMDA_array), num_surfaces)
PLV_mean_after_50_array = zeros(length(n_array),length(NMDA_array), 1)

# Arrays for additional peak detection results
additional_peaks_array = zeros(length(n_array),length(NMDA_array), num_surfaces)
additional_peaks_array_mean = zeros(length(n_array),length(NMDA_array), 1)
additional_peak_amp_array = Array{Array}(undef, length(n_array),length(NMDA_array), )
additional_peak_prom_array = Array{Array}(undef, length(n_array),length(NMDA_array))

# ============================================================================
# SECTION 4: Load and Aggregate Results for 2D Parameter Grid
# ============================================================================
# Loop through all parameter combinations (gsyn outer loop, ggap inner loop)
i = 1
for λ_rate_synch_i in n_array
    j = 1
    for NMDA_scale_i in NMDA_array
        println("λ_rate_synch: $(λ_rate_synch_i), NMDA: $(NMDA_scale_i)")
        
        # Create parameter dictionary for current grid point
        param = Dict(
            "gsyn" => gsyn,                 # GABA synaptic conductance (nS)
            "ggap" => ggap,                 # Gap junction conductance (nS)
            "gAMPA" => gAMPA,               # AMPA receptor conductance (nS)
            "Ihold_scale" => Ihold_scale,   # Holding current scaling
            "NMDA_scale" =>NMDA_scale_i,    # NMDA receptor scaling
            "rate_synch" => λ_rate_synch_i, # input rate (Hz)
            "synch_dur" => synch_dur,       # input duration (ms)
            "tau_GABA" => tau_GABA,         # GABA decay time constant (ms)
        )

        # Load coactivity data
        peak_coactivity = wload(datadir("simulations", "Network", "new", savename(fname, param, "jld2")), "peak_coactivity_array");
        peak_coactivity_array[i,j, :] = peak_coactivity
        peak_coactivity_array_mean[i,j] = mean(peak_coactivity)
        peak_coactivity_array_std[i,j] = std(peak_coactivity)

        # Load phase-locking values (PLV) for different time windows
        PLV_before = wload(datadir("simulations", "Network", "new", savename(fname, param, "jld2")), "PLV_mean_before_array") 
        PLV_after_50 = wload(datadir("simulations", "Network", "new", savename(fname, param, "jld2")), "PLV_mean_after_50_array")


        # Load secondary peak analysis
        additional_peaks = wload(datadir("simulations", "Network", "new", savename(fname, param, "jld2")), "additional_peaks")
        additional_peak_amp  = wload(datadir("simulations", "Network", "new", savename(fname, param, "jld2")), "additional_peak_amp")
        additional_peak_prom = wload(datadir("simulations", "Network", "new", savename(fname, param, "jld2")), "additional_peak_prom")

        # Store PLV results
        PLV_before_array[i,j, :] = PLV_before
        PLV_after_50_array[i,j, :] = PLV_after_50

        additional_peaks_array[i,j, :] = additional_peaks
        additional_peak_amp_array[i,j] = additional_peak_amp
        additional_peak_prom_array[i,j] = additional_peak_prom

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
# Compile parameters used (excluding λ_rate_synch and NMDA_scale which form the grid


# Save all aggregated results to file
param_save = Dict(
    "I" => 0.0, 
    "gsyn" => gsyn,           # GABA synaptic conductance (nS)
    "ggap" => ggap,           # Gap junction conductance (nS)
    "gAMPA" => gAMPA,             # AMPA receptor conductance (nS)
    "synch_dur" => synch_dur,      # input duration (ms)
    "tau_GABA" => tau_GABA,        # GABA decay time constant (ms)
)
wsave(datadir("simulations", "Network", "new", savename(fname_save, param_save, "jld2")),  @strdict fname_save peak_coactivity_array peak_coactivity_array_mean peak_coactivity_array_std n_array NMDA_array  PLV_before_array PLV_after_array PLV_after_50_array PLV_mean_before_array PLV_mean_after_array PLV_mean_after_50_array additional_peaks_array additional_peaks_array_mean additional_peak_amp_array additional_peak_prom_array)


