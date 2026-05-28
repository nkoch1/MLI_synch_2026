using DrWatson
@quickactivate  "MLI_synch_2026"
# ============================================================================
# Gap-junction synchrony parameter sweep (two-cell CSC model)
#
# This script sweeps initial phase differences and coupling positions for two
# coupled CSC (cable) models with soma Ih, computes phase-locking metrics
# (PLV, Hilbert phase difference), and saves results.
# ============================================================================

# Include model and analysis modules from src/
include(srcdir("CSC_cable_2.jl"))
using .CSC_cable_2
include(srcdir("analysis.jl"))
using .analysis
using DifferentialEquations, DiffEqCallbacks, JLD2, Statistics, Peaks, FHist
using CircStats
using ProgressMeter
using FLoops

# Limit BLAS threads to keep parallelism predictable in experiments
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

# -----------------------------
# Simulation / model parameters
# -----------------------------
# Time stepping and integration window
dt = 0.1                     # Integration time step (ms)
tspan = (0.0, 30*1000.0)     # Total simulation duration (ms)

# Multicompartment cable resolution
numseg = 50                  # Number of dendritic segments per cell

# Background current (unused here but kept for compatibility)
I = 0.                        # Background current (uA/cm^2)

# Voltage binning parameters used by some analyses
bin_low = -70                # Lower bound for voltage histogram (mV)
bin_high = -40               # Upper bound for voltage histogram (mV)
bin_incr = 0.25              # Bin increment (mV)
wlen = 25000                 # Analysis window length (samples)


# ==============================================================================
# Load fitted parameters and identify conductance indices
# ==============================================================================
# Load fitted parameters from the cable model optimization
p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_cable.jld2"), "p_fit")

# Load parameter names to identify which index corresponds to each conductance
param_names = wload(datadir("simulations", "CSC_Ih", "fit_param_cable_names.jld2"),  "param_names")

# Identify indices for dendritic conductancesdend_Kd_ind = findfirst(param_names .== "gKd")
dend_Ad_ind = findfirst(param_names .== "gAd")
dend_SKd_ind = findfirst(param_names .== "gSKd")
dend_Td_ind = findfirst(param_names .== "gTd")
dend_HVAd_ind = findfirst(param_names .== "gHVAd")

# -----------------------------
# Gap-junction configuration and command-line args
# -----------------------------
# Initial relative phase between the two cells (fraction of cycle)
init_phase_diff = 0.5

# Default coupling site (segment index for each cell)
gapseg = (51, 51)  # symmetrical coupling at distal compartment by default

# Parse command-line arguments: expected two args: ggap, condition
#   ARGS[1] : gap-junction conductance (parsed to Float64)
#   ARGS[2] : condition string, e.g. "no_Kd" to zero that channel
ggap = parse(Float64, ARGS[1])
f_cond = ARGS[2]

# Build parameter vector for cable model: append gap segment and conductance
p_cable = [p_fit..., gapseg, ggap]

# Optionally zero specific dendritic conductances based on `f_cond` flag
if f_cond == "no_Kd"
    p_cable[dend_Kd_ind] = 0.
elseif f_cond == "no_Ad"
    p_cable[dend_Ad_ind] = 0.
elseif f_cond == "no_SKd"
    p_cable[dend_SKd_ind] = 0.
elseif f_cond == "no_Td"
    p_cable[dend_Td_ind] = 0.
elseif f_cond == "no_HVAd"
    p_cable[dend_HVAd_ind] = 0.
end

# -----------------------------
# Sweep setup: initial phase samples and coupling positions
# -----------------------------
# Number of discrete initial phase offsets to test (fraction of cycle)
n_phase = 12
init_phase_diff_array = collect(0:1/n_phase:1 - 1/n_phase)

# Spatial sampling along dendrite: which segment indices to test
p_int = 5
position_array = collect(1:p_int:51)


# -----------------------------
# Initial condition setup
# -----------------------------
# Create a template initial condition with the chosen `init_phase_diff`
p_ic = deepcopy(p_cable)
# Integrate single-network until steady/phase-locked initialization
ic = CSC_cable_2.init_phase_diff_2(init_phase_diff, tuple(p_ic...);
    tend=1500, samp_rate=0.001, abstol=1e-12, reltol=1e-12)


# -----------------------------
# Preallocate result containers
# -----------------------------
# PLV and Hilbert phase difference matrices (phase x position)
global PLV_array = zeros(length(init_phase_diff_array), length(position_array))
global Hilbert_phase_diff_array = zeros(length(init_phase_diff_array), length(position_array))
global mean_peak_phase_end_array = zeros(length(init_phase_diff_array), length(position_array))

# Spike times and phase difference traces stored per condition
global spike_times = Array{Array{Float64}}(undef, length(init_phase_diff_array), length(position_array), 2)
global phase_diff_array = Array{Array{Float64}}(undef, length(init_phase_diff_array), length(position_array))
global time_converg_array = Array{Array{Float64}}(undef, length(init_phase_diff_array), length(position_array))

# -----------------------------
# ODE problem template and analysis settings
# -----------------------------
# Create a template ODEProblem for the two-cell network so we can `remake`
# it with different coupling positions or strengths efficiently.
global prob = ODEProblem(CSC_cable_2.CSCcable_2!, ic, tspan, tuple(p_cable...);
    save_idxs=[1, 9], maxiters=1e25, abstol=1e-12, reltol=1e-12)

# Number of spikes to use for Hilbert-phase based phase computation
hilb_spike_num = 5


# -----------------------------
# Build list of (phase_index, position_index) pairs to iterate over
# -----------------------------
g_pos_comb = []
for i in range(1,length(init_phase_diff_array)) 
    for j in range(1, length(position_array)) 
        push!(g_pos_comb, [i,j])
    end
end


# -----------------------------
# Warm-start initial conditions for each initial-phase value
# -----------------------------
# Precompute steady / initialized states for each initial phase to avoid
# repeating long initial transients inside the main loop.
u0net_array = [
    ic_i = CSC_cable_2.init_phase_diff_2(init_ph,  tuple(p_ic...); tend=5000, samp_rate=0.001,  abstol=1e-9, reltol=1e-9) for init_ph in init_phase_diff_array]


#%%
for ii in range(1, length(g_pos_comb))
    i, j = g_pos_comb[ii]
    ic_i = u0net_array[i]

    # Remake parameter vector: set gap conductance and coupling positions
    p_loop = deepcopy(p_cable)
    p_loop[end] = ggap                     # gap conductance for this run
    p_loop[end-1] = (position_array[j], position_array[j])  # symmetric coupling

    # Remake ODE problem with new parameters and IC
    prob_i = remake(prob, p = tuple(p_loop...), u0 = ic_i, tspan = tspan)

    # Solve with a robust stiff solver (QNDF here) and reasonable dtmax
    sol_i = solve(prob_i, QNDF(), dtmax=0.01)

    # Compute phase synchrony metrics using analysis utilities
    phaseps, t1ps, t2ps, _, PLV, Hilbert_phase_diff, mean_peak_phase_end, phases_diff_diff_avg, time_converg = 
        analysis.phase_diff(sol_i[1, :], sol_i[2, :], sol_i.t;
            hilb_spike_num=hilb_spike_num, threshold = 0.01 * π / 180, n_spikes_thresh = 10)

    # Store results (store even if simulation did not converge)
    spike_times[i, j, 1] = t1ps
    spike_times[i, j, 2] = t2ps
    phase_diff_array[i, j] = phaseps
    PLV_array[i, j] = PLV
    Hilbert_phase_diff_array[i, j] = Hilbert_phase_diff
    mean_peak_phase_end_array[i, j] = mean_peak_phase_end
    time_converg_array[i, j] = [time_converg]

    # Flush and free memory between iterations
    flush(stdout)
    GC.gc()
    nothing
end

# -----------------------------
# Save results and metadata
# -----------------------------
param = Dict(
    "n_phase" => n_phase,
    "pos_int" => p_int,
    "hilb_spike_num" => hilb_spike_num,
    "tend" => tspan[2] / 1000,  # simulation time in seconds
    "ggap" => ggap,
)

# Save with an informative name that includes condition metadata
@tagsave(
    datadir("simulations", "gapjxn_charac", "new", savename("ggap_synch_2_properties_CSC_soma_Ih_time_synch_$(f_cond)", param, "jld2")),
    @strdict phase_diff_array spike_times PLV_array Hilbert_phase_diff_array mean_peak_phase_end_array init_phase_diff_array position_array time_converg_array;
    safe = DrWatson.readenv("DRWATSON_SAFESAVE", true)
)

