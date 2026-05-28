# ============================================================================
# Network Simulation: CSC1-GrC-LP with NMDA and Gap Junctions
# ============================================================================
# Purpose: Run full Hodgkin-Huxley network simulations with CSC1 stellate cells,
#          GrC (granule cell) input, and LP dendritic filtering. Includes NMDA synapses,
#          gap junctions, and analysis of spike coactivity and phase-locking.
#
# Output: Spike rasters, coactivity arrays, phase-locking values (PLV),
#         and solution trajectories for each network surface
# ============================================================================

# ============================================================================
# SECTION 1: Load Dependencies and Project Setup
# ============================================================================
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
using Distributions
using LinearAlgebra
BLAS.set_num_threads(1)
using PyCall
@pyimport scipy as scipy

# ============================================================================
# SECTION 2: Load Network Geometry and Connectivity
# ============================================================================
fname_net = "Network_realizations_dense_scaled_336.jld2"
x_pos = wload(datadir("simulations", "Network", fname_net),  "x_pos")  # X positions of cells
y_pos = wload(datadir("simulations", "Network", fname_net),  "y_pos")  # Y positions of cells
chem_connections = wload(datadir("simulations", "Network", fname_net),  "chem_connections")  # Synaptic connections
elec_connections = wload(datadir("simulations", "Network", fname_net),  "elec_connections")  # Gap junction connections
ncells = wload(datadir("simulations", "Network", fname_net),  "N")  # Number of cells in network
num_surfaces = size(x_pos)[1]  # Number of network surface patches

# ============================================================================
# SECTION 3: Simulation Timespan and Parameters
# ============================================================================
tspan = (0.0, 2 * 1000.0)  # Total simulation time: 2000 ms

pre_t  = 500.  # Pre-stimulus recording period (ms)
sim_t_start = 1000.  # Start of analysis window (ms)
synchtimes = 1500.  # Time of synchronized input onset (ms)
I = 0.  # Global current injection (nA)

# ============================================================================
# SECTION 4: Hodgkin-Huxley Ion Channel Parameters
# ============================================================================
# Maximal conductances (mS/cm²)
gNa = 3.
gK = 17.5
gleak = 0.0125
gA = 7.5
gT = 0.45045
gHVA = 0.28
gSK = 0.3
gh = 20
Eleak = -60 

# Voltage for half-activation/half-inactivation (mV)
Vm = -37
Vh = -37.5 
Vn = -26     
VnA = -27    
VhA = -82    
VmT = -54    
VhT = -74     
VmHVA = -25   

# Slope factors (steepness of activation curves, mV)
km = 3.
kmT = 3.
kh = 3.
kn = 6.
knA = 13.2
khA = 6.5
khT = 3.75 
kmHVA = 8. 

# Reversal potentials (mV)
EK = -80
ENa = 55
ECa = 22
Eh = -34.4 

# Low-pass filter time constant (dendritic filtering)
tau_d = 28.57  # Low-pass tau (ms) => 35 Hz -3dB cutoff of OU filtering

# Initial membrane voltage
V0 = -60  # Initialization voltage (mV)

# ============================================================================
# SECTION 5: Action Potential Detection and Synaptic Parameters
# ============================================================================
# AP detection settings
Vthresh = -10.  # Voltage threshold for action potential detection (mV)
refractory = 2  # Refractory period (ms)

# Synaptic parameters
syn_delay = 5  # Synaptic transmission delay (ms)
tau_GABA = parse(Float64, ARGS[6])  # GABA decay time constant (ms, from command line)
EGABA = -80.  # GABA reversal potential (mV)

# AMPA synaptic parameters
tau_AMPA = 2.0  # AMPA decay time constant (ms)
EAMPA = 0.      # AMPA reversal potential (mV)

# ============================================================================
# SECTION 6: Cable Theory Parameters (Dendritic Properties)
# ============================================================================
# Use dendritic properties from compartmental model to compute space constant λ
L = 0.015      # Dendrite length (cm)
dend_r = 0.2   # Dendrite radius (μm)
soma_r = 6.0   # Soma radius (μm)
gleakd = 0.07407  # Dendritic leak conductance (mS/cm²)
a = dend_r / 10000  # Convert dendrite radius to cm
As = soma_r / 10000  # Convert soma radius to cm
rmem = 1 / (gleakd / 1000)  # Membrane resistance Ω·cm² (convert mS->S and kΩ->Ω)
λ = sqrt(a * rmem / (2 * CSC1_network_poisson.rl)) * 10000  # Space constant (convert cm back to μm)

# ============================================================================
# SECTION 7: Spike Analysis and Histogram Binning
# ============================================================================
# Histogram bins for spike analysis
bin_size = 2  # Bin size (ms) for coarse PSTH histogram
bins = collect(sim_t_start:bin_size:tspan[end])  # Bin edges (ms)
bins_mid = [bins[i] + ((bins[i+1] - bins[i])/2) for i in eachindex(bins[1:end-1])]  # Bin midpoints

bin_size1 = 1  # Finer bin size (ms) for coactivity histogram
bins1 = collect(sim_t_start:bin_size1:tspan[end])  # Finer bin edges
bins1_mid = [bins1[i] + ((bins1[i+1] - bins1[i])/2) for i in eachindex(bins1[1:end-1])]  # Finer bin midpoints

# Single-cell parameters for initialization
p1 = [gNa, gK, gleak, gA, gT, gHVA, gSK, gh, Eleak, Eh, I, tau_d]

# ============================================================================
# SECTION 8: NMDA Receptor Kinetics
# ============================================================================
τ_NMDA_rise = 15.   # NMDA rise time constant (ms)
τ_NMDA_decay = 150. # NMDA decay time constant (ms)
ENMDA = 0.          # NMDA reversal potential (mV)
tsyn = synchtimes[1]  # Synapse onset time

# Compute NMDA peak time and normalization constant
tpeak_NMDA = tsyn + (τ_NMDA_rise*τ_NMDA_decay)/(τ_NMDA_decay - τ_NMDA_rise) * log(τ_NMDA_decay/τ_NMDA_rise)
K_NMDA = 1/((exp(-(tpeak_NMDA - tsyn)/τ_NMDA_decay)-(exp(-(tpeak_NMDA - tsyn)/τ_NMDA_rise))))

# ============================================================================
# SECTION 9: Parse Command Line Arguments and Convert Conductances
# ============================================================================
# Background Poisson GrC input rate
λ_rate = 50  # Background input rate (Hz)

# Synchronized input parameters (from command line)
λ_rate_synch = parse(Float64, ARGS[7])  # Synchronized input rate (Hz)
synch_dur = parse(Float64, ARGS[8])     # Synchronized input duration (ms)

# Gap junction and synaptic conductances (from command line)
# Convert from nS to μA/cm² using soma surface area (assuming 6 μm soma radius)
ggap = parse(Float64, ARGS[1])  * 1e-6 * 1/(4 * π * 0.0006^2)   # Gap junction conductance
gsyn = parse(Float64, ARGS[2])  * 1e-6 * 1/(4 * π * 0.0006^2)   # GABA synaptic conductance
gAMPA = parse(Float64, ARGS[3]) * 1e-6 * 1/(4 * π * 0.0006^2)   # AMPA conductance

# Other experimental parameters
Ihold_scale = parse(Float64, ARGS[4])  # Scaling for holding current noise
NMDA_scale = parse(Float64, ARGS[5])   # Scaling factor for NMDA component (0-1)
gNMDA = NMDA_scale * gAMPA  # Absolute NMDA conductance

# ============================================================================
# SECTION 10: Initialize Output Data Arrays
# ============================================================================
# Arrays to store analysis results for each surface
peak_coactivity_array = zeros(num_surfaces,)  # Peak coactivity per surface
PLV_mean_before_array = zeros(num_surfaces,)  # Mean phase-locking before stimulus
PLV_mean_after_50_array = zeros(num_surfaces,)  # Mean PLV 0-50 ms after stimulus

# Additional peak detection results
additional_peaks = zeros(num_surfaces,)  # Number of peaks in 50 ms window
additional_peak_amp = Array{Array}(undef, num_surfaces)  # Peak amplitudes
additional_peak_prom = Array{Array}(undef, num_surfaces)  # Peak prominences

# Spike histograms
psth_density_array = zeros(num_surfaces, length(bins)-1)  # Post-stimulus time histogram (2 ms bins)
coactivity_array = zeros(num_surfaces, length(bins1)-1)   # Coactivity histogram (1 ms bins)


# ============================================================================
# SECTION 11: Initialize Initial Conditions and Setup Callbacks
# ============================================================================
# Create initial conditions with random phase differences
seed_i = 1  # Random seed for reproducibility
u0_net, ph_array = init_phase_diff_rand_GrC(p1, ncells, seed_i; tend=1500, samp_rate=0.1)

# Specify which model state variables to save during ODE integration
# Variables are indexed using LinearIndices for efficient access
save_index = [
    [LinearIndices((ncells,13))[i,1] for i=1:ncells]...,  # Soma voltage (state 1)
    [LinearIndices((ncells,13))[i,11] for i=1:ncells]..., # Action potential flag (state 11)
    [LinearIndices((ncells,13))[i,12] for i=1:ncells]..., # Synaptic gating variable (state 12)
    [LinearIndices((ncells,13))[i,13] for i=1:ncells]..., # Low-pass filtered V (state 13)
    [LinearIndices((ncells,14))[i,14] for i=1:ncells]..., # GrC input gating (state 14)
]

# Initialize AP tracking and NMDA state arrays
AP_array = zeros(ncells)  # Action potential flags
NMDA_bool = ones(ncells)  # NMDA conductance active (on/off)

# ============================================================================
# SECTION 12: Setup Phase Analysis and Callbacks
# ============================================================================
# Create index pairs for pairwise phase calculations (upper triangle)
pairwise_ind = stack([[x,y] for (i,x) in enumerate(1:ncells), (j,y) in enumerate(1:ncells) if i>j])
npairs = size(pairwise_ind)[2]  # Total number of cell pairs

# Discrete callbacks for event detection
cb = DiscreteCallback(V_detect, affect!; save_positions=(false,false))  # Action potential detection
cb_GrC = DiscreteCallback(condition_GrC, affect_GrC!; save_positions=(false,false))  # GrC input events

# Preset time callbacks for scheduled events
cb_synch_init = PresetTimeCallback(synchtimes, integrator -> integrator.p[44] = synchtimes .+ rand(Exponential(1/λ_rate_synch)*1000, ncells)) 
cb_save_on = PresetTimeCallback(synchtimes - pre_t, integrator -> integrator.opts.save_on = true,)  # Turn on solution saving

# Combine all callbacks into a set
cbs = CallbackSet(cb, cb_GrC, cb_save_on, cb_synch_init)

# ============================================================================
# SECTION 13: Loop Over Network Surfaces and Run Simulations
# ============================================================================
for s in range(1,num_surfaces)
    println(s)

    # Extract connectivity for this surface
    chem_connections_i = chem_connections[s]  # Chemical (synaptic) connections
    elec_connections_i = elec_connections[s]  # Electrical (gap junction) connections
    
    # Create gap junction connection list with specified conductance
    pairs_gap_i = [(Int(elec_connections_i[i][1]), Int(elec_connections_i[i][2]), ggap, elec_connections_i[i][3]) 
                   for i=(1:length(elec_connections_i))]
    connect_ggap_i = create_gap_arrays(ncells, pairs_gap_i, λ)  # Convert to conductance matrix

    # Create synaptic connection list, accounting for directionality
    pairs_syn = [if chem_connections_i[i][3] > 0 
                    (Int(chem_connections_i[i][1]), Int(chem_connections_i[i][2]), gsyn) 
                 else 
                    (Int(chem_connections_i[i][2]), Int(chem_connections_i[i][1]), gsyn) 
                 end 
                 for i=(1:length(chem_connections_i))]
    connect_syn_i = create_syn_arrays(ncells, pairs_syn)  # Convert to conductance matrix

    # ========================================================================
    # SECTION 14: Setup ODE Problem and Run Simulation
    # ========================================================================
    # Generate next spike times for Poisson GrC input
    next_t = next_spike_time(zeros(ncells))  # for Poisson GrC input
    
    # Generate random holding current (independent noise per cell)
    Random.seed!(1)
    local I = randn(ncells) * Ihold_scale

    # Assemble complete parameter vector for ODE system
    # Includes: ion channel params, connectivity matrices, synaptic parameters, input parameters
    p_net = [gNa, gK, gleak, gA, gT, gHVA, gSK, gh, I, Eleak, Eh, Vm, Vh, Vn, VnA, VhA, VmT, VhT, VmHVA, km, kmT, kh, kn, knA, khA, khT, kmHVA, EK, ENa, ECa, ncells, Vthresh, refractory, connect_ggap_i, deepcopy(AP_array), syn_delay, tau_GABA, connect_syn_i, EGABA, tau_d, gAMPA, EAMPA, tau_AMPA, next_t, gNMDA, τ_NMDA_rise, τ_NMDA_decay, K_NMDA, tsyn, ENMDA, deepcopy(NMDA_bool), deepcopy(λ_rate)]

    # Setup and solve ODE problem
    dt = 0.01  # Time step (ms)
    prob_i = ODEProblem(CSC1_net_GrC_NMDA!, u0_net, tspan, p_net, callback=cbs, dt=dt, save_idxs=save_index, save_on=false, save_start=false) 
    sol_i = solve(prob_i, VCABM(), abstol=1e-4, reltol=1e-4)

    # Save first surface solution for later visualization
    if s == 1
        global sol_ex = deepcopy(sol_i)
    end

    # ========================================================================
    # SECTION 15: Extract and Process Spike Times
    # ========================================================================
    # Extract action potential times from solution
    AP_detected = findall(sol_i[1+ncells:1+ncells+ncells, :] .==1)  # Find AP events
    AP_times = [sol_i.t[sol_i[i+ncells, :] .== 1] for i =1:ncells]  # Spike times per cell
    AP_times_all = vcat(AP_times)  # All spike times concatenated

    # Convert spike times to array format
    all_spike_times = mapreduce(permutedims, hcat, AP_times)
    
    # Create post-stimulus time histogram (PSTH) with 2 ms bins
    h = fit(Histogram, convert(Array{Float64, 1}, vec(all_spike_times)), bins)
    psth_density_array[s, :] = h.weights ./ bin_size ./ ncells  # Normalize by bin size and ncells
    
    # Create finer coactivity histogram (1 ms bins) using rolling window sum
    h1 = fit(Histogram, convert(Array{Float64, 1}, vec(all_spike_times)), bins1)
    coactivity_array[s, :] = rolling_sum(h1.weights, 2) ./ ncells .* 100  # Percentage coactivity

    # Find peak coactivity
    peak_coactivity_ind = argmax(coactivity_array[s, :])
    peak_coactivity_array[s] = coactivity_array[s, peak_coactivity_ind]


    # ========================================================================
    # SECTION 16: Secondary Peak Detection
    # ========================================================================
    # Define time windows for peak detection
    ind_after_50_bins = (bins1_mid .>= synchtimes[1] - 10) .* (bins1_mid .<= synchtimes[1] + 50)  # -10 to +50 ms window

    # Calculate dynamic peak prominence threshold based on pre-stimulus noise
    coact_std_before = std(coactivity_array[s, bins1_mid .<= synchtimes[1]])  # Pre-stimulus std dev
    if coact_std_before * 5 < 1
        min_prom = 1  # Minimum prominence threshold
    else
        min_prom = coact_std_before * 5  # Threshold: 5x pre-stimulus std dev
    end
    
    # Detect peaks in 50 ms window
    peaks_sp, props_sp = scipy.signal.find_peaks(coactivity_array[s, ind_after_50_bins], prominence=min_prom)
    peaks_sp .+= 1  # Convert from 0-based to 1-based indexing
    additional_peaks[s] = length(peaks_sp)  # Count peaks
    additional_peak_amp[s] = coactivity_array[s, ind_after_50_bins][peaks_sp]  # Peak amplitudes
    additional_peak_prom[s] = props_sp["prominences"]  # Peak prominences


    # ========================================================================
    # SECTION 17: Phase-Locking Analysis - Define Analysis Windows
    # ========================================================================
    # Pre-stimulus control window (-250 to -50 ms relative to synchtimes)
    ind_before = (sol_i.t .>= synchtimes[1] - 250) .* (sol_i.t .>= synchtimes[1] - 50)
    
    # Post-stimulus windows for phase analysis
    ind_after_50 = (sol_i.t .>= synchtimes[1]) .* (sol_i.t .<= synchtimes[1] + 50)   # 0-50 ms


    # ========================================================================
    # SECTION 18: Phase-Locking Value Calculation
    # ========================================================================
    # Compute instantaneous phase for each cell using analytic signal (Hilbert transform)
    ph_angle = zeros(ncells, length(sol_i.t))
    for i = 1:ncells
        @views @inbounds ph_angle[i, :] = angle.(hilbert(sol_i[i, :] .- mean(sol_i[i, :])))
    end

    # Compute PLV in pre-stimulus window (baseline/control)
    ph_angle_diff = zeros(length(sol_i.t[ind_before]))
    PLV_pairwise_1 = zeros(npairs)
    for i in axes(pairwise_ind, 2)
        @views @inbounds ph_angle_diff = ph_angle[pairwise_ind[1,i], ind_before] - ph_angle[pairwise_ind[2,i], ind_before]
        @views @inbounds PLV_pairwise_1[i] = 1 - abs.(CircStats.circ_var(ph_angle_diff)[1])  # PLV = 1 - |circular variance|
    end
    PLV_mean_before_array[s] = mean(PLV_pairwise_1)  # Average across all pairs


    # Compute PLV in 0-50 ms window post-stimulus
    ph_angle_diff = zeros(length(sol_i.t[ind_after_50]))
    PLV_pairwise_1 = zeros(npairs)
    for i in axes(pairwise_ind, 2)
        @views @inbounds ph_angle_diff = ph_angle[pairwise_ind[1,i], ind_after_50] - ph_angle[pairwise_ind[2,i], ind_after_50]
        @views @inbounds PLV_pairwise_1[i] = 1 - abs.(CircStats.circ_var(ph_angle_diff)[1])
    end
    PLV_mean_after_50_array[s] = mean(PLV_pairwise_1)


    # Force garbage collection to free memory between surface iterations
    GC.gc()
end


# ============================================================================
# SECTION 19: Extract Solution Data and Save Results
# ============================================================================
# Extract solution trajectories from example surface (first surface)
solu = sol_ex.u  # Solution state vectors
solt = sol_ex.t  # Solution time points

# ============================================================================
# SECTION 20: Save All Simulation Results to File
# ============================================================================
# Create parameter dictionary for file naming (command-line arguments in nS units)
param = Dict(
    "gsyn" => ARGS[2],  # Synaptic conductance (nS, from command line)
    "ggap" => ARGS[1],  # Gap junction conductance (nS, from command line)
    "gAMPA" => ARGS[3],  # AMPA conductance (nS, from command line)
    "Ihold_scale" => Ihold_scale,  # Holding current scaling
    "NMDA_scale" => NMDA_scale,  # NMDA component scaling (0-1)
    "rate_synch" => λ_rate_synch,  # Synchronized input rate (Hz)
    "synch_dur" => synch_dur,  # Synchronized input duration (ms)
    "tau_GABA" => tau_GABA,  # GABA decay time constant (ms)
)

# Save all results to JLD2 file with parameter-dependent filename
fname_save = "Network_CSC1_NMDA_CC_poisson"
wsave(
    datadir("simulations", "Network", savename(fname_save, param, "jld2")),
    @strdict fname_net peak_coactivity_array psth_density_array coactivity_array ncells solu solt 
             PLV_mean_before_array  PLV_mean_after_50_array 
             additional_peaks additional_peak_amp additional_peak_prom  
)

















