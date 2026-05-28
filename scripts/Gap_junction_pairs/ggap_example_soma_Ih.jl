# ============================================================================
# Gap Junction Coupling: Example Simulation of Two CSCs with Soma Ih
# ============================================================================
# This script demonstrates gap junction-mediated synchronization between two
# compartmentalized cerebellar stellate cells (CSCs). It simulates electrical
# coupling via gap junctions at specified dendritic locations and analyzes
# the resulting phase synchronization using circular statistics.

using DrWatson
@quickactivate  "MLI_synch_2026"
include(srcdir("CSC_cable_net.jl"))
using .CSC_cable_net
include(srcdir("analysis.jl"))
using .analysis
using DifferentialEquations, DiffEqCallbacks,
    Optimization, SciMLSensitivity,
    Zygote,  DiffEqCallbacks, JLD2, Statistics, Peaks

using FHist
using PyCall


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
# Integration and time parameters
dt = 0.1                 # Time step (ms)
tspan = (0.0, 2.01*1000.0)  # Total simulation time (ms)
numseg = 50              # Number of dendritic compartments per cell
I = 0.                   # Background current injection (uA/cm^2)

# Membrane potential binning for phase analysis
bin_low = -70            # Lower voltage bound (mV)
bin_high = -40           # Upper voltage bound (mV)
bin_incr = 0.25          # Bin size (mV)
wlen = 25000             # Window length for analysis (samples)


# ============================================================================
# LOAD FITTED CELL PARAMETERS
# ============================================================================
# Load previously fitted CSC model parameters (cable equation with Ih)
p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_cable.jld2"), "p_fit")

# Convert to mutable array and fix integer parameter
p_fit = Array{Any}(p_fit)
p_fit[22] = Int(p_fit[22])

# Load parameter names and identify ion channel conductance indices
param_names = wload(datadir("simulations", "CSC_Ih", "fit_param_cable_names.jld2"),  "param_names")
dend_Kd_ind = findfirst(param_names .== "gKd")       # Delayed rectifier K
dend_Ad_ind = findfirst(param_names .== "gAd")       # A-type K channel
dend_SKd_ind = findfirst(param_names .== "gSKd")     # SK Ca-activated K
dend_Td_ind = findfirst(param_names .== "gTd")       # T-type Ca channel
dend_HVAd_ind = findfirst(param_names .== "gHVAd")   # HVA Ca channel




# ============================================================================
# PHASE ANALYSIS PARAMETERS
# ============================================================================
# Phase shift analysis: discretize initial phase differences into n_phase bins
n_phase = 12                                              # Number of phase bins
init_phase_diff_array = collect(0:1/n_phase:1 - 1/n_phase)  # Phase difference array

# Spatial sampling: record membrane potential at intervals along dendrite
p_int = 5                              # Interval between recorded compartments
position_array = collect(1:p_int:51)   # Compartment indices to sample

# Spike detection for Hilbert phase analysis
hilb_spike_num = 5  # Number of spikes to use for phase computation

# Gap junction conductance values to test (log-spaced range)
ggap_array = [
                0,                      # No coupling (control)
                0.01,
                0.01778279410038923,
                0.03162277660168379,
                0.05623413251903491,
                0.1,
                0.1778279410038923,
                0.31622776601683794,
                0.5623413251903491,
                1,
                1.7782794100389228,
                3.1622776601683795,
                5.62341325190349,
                10,
                17.78279410038923,
                31.622776601683793, 
                56.23413251903491, 
                100.0               # Maximum coupling strength
                ]



# ============================================================================
# GAP JUNCTION CONFIGURATION
# ============================================================================
# Initial spike timing offset (as fraction of phase)
init_phase_diff = 0.5  # Cell 2 lags Cell 1 by 50% of spike period

# Gap junction location: (segment in cell 1, segment in cell 2)
# Segment 11 is in the mid-dendrite region
gapseg = (11, 11)      # Both cells coupled at identical dendritic location

# Gap junction conductance (nS or normalized units)
gc = 2.                # Electrical coupling strength 


# ============================================================================
# INITIALIZE AND RUN SIMULATION
# ============================================================================
# Set up initial conditions with phase difference between cells
p_ic = deepcopy(p_fit)
ic = CSC_cable_net.init_phase_diff_2(init_phase_diff,  p_ic; tend=1500, samp_rate=0.1)

# Build parameter vector: [cell parameters..., gap junction location, conductance]
p = [p_fit..., gapseg, gc]

# Define ODE problem: two coupled CSC cells with gap junction
# Save soma voltage (compartment 1, 10) from both cells
prob = ODEProblem(CSC_cable_net.CSCcable_2!,  ic::Array{Float64},tspan::Tuple{Float64, Float64},tuple(p...), save_idxs=[1, 10], maxiters=1e25)

# Solve using ROCK2 (Runge-Kutta-Chebyshev) for stiff systems
sol = solve(prob, ROCK2())


#%% ========== PHASE ANALYSIS ===================================================
# Compute phase relationship and synchronization metrics from soma voltages
# sol[1,:] = soma voltage cell 1, sol[2,:] = soma voltage cell 2
hilb_spike_num = 5  # Use 5 spikes for phase computation
phaseps, t1ps, t2ps, ISI1ps, PLV, Hilbert_phase_diff, mean_peak_phase_end = analysis.phase_diff(
    sol[1,:], sol[2,:], sol.t; hilb_spike_num=hilb_spike_num)


# ============================================================================
# CIRCULAR STATISTICS: PHASE DIFFERENCE VARIABILITY
# ============================================================================
# Quantify variability in phase differences using circular distance
using CircStats
phase_diff_diff = CircStats.circ_dist(phaseps[1:end-1], phaseps[2:end])


# ============================================================================
# SAVE RESULTS WITH METADATA
# ============================================================================
# Package simulation parameters as metadata dictionary
param = Dict(
    "phase_diff" => init_phase_diff,   # Initial phase difference
    "sim_length" => tspan[2],           # Total simulation duration (ms)
    "hilb_spike_num" => hilb_spike_num, # Number of spikes for Hilbert phase
    "gc" => gc,                         # Gap junction conductance
    "gapseg1" =>  gapseg[1],            # Gap junction location in cell 1
    "gapseg2" =>  gapseg[2],            # Gap junction location in cell 2
)

# Save simulation output and analysis results with automatic naming
@tagsave(datadir("simulations", "gapjxn_charac", savename("ggap_example_CSC_soma_Ih", param, "jld2")),  
    @strdict sol phaseps t1ps t2ps ISI1ps PLV Hilbert_phase_diff mean_peak_phase_end; 
    safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))