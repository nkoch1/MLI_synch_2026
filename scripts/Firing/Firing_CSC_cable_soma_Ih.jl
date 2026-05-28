# ============================================================================
# Firing Analysis: Cerebellar Stellate Cell with Cable Properties and Ih
# ============================================================================
# This script analyzes the firing properties of a multi-compartment CSC model
# with soma Ih and dendritic ion channels under various conditions.
# It compares wild-type behavior with single ion channel knockouts.

using DrWatson
@quickactivate  "MLI_synch_2026"

using DifferentialEquations, DiffEqCallbacks, Statistics, Peaks
using FHist
using PyCall, PyPlot
using LaTeXStrings
include(srcdir("CSC_cable_Ih.jl"))
using .CSC_cable_Ih
include(srcdir("analysis.jl"))
using .analysis

#%% ========== READ IN EXPERIMENTAL DATA =========================================
# Load experimental firing data from HBP database (patch clamp recordings)
fpath = datadir("exp_pro", "HBP_firing_analysis_extract.json")

# Load data using Python/pandas
py"""
import numpy as np
import pandas as pd
df = pd.read_json($fpath)
"""
i = 2 # cell 3 (index 2, 0-based)
I_data = py"df.loc[$(i-1), 'I']"  # Injected currents (pA)
F_data = py"df.loc[$(i-1), 'F']"  # Firing rates (Hz)

# Extract data from multiple cells for averaging
F_all_ind = [1,2,5,7,8]
F_data_all = py"df.loc[$(F_all_ind .-1), 'F']"

# Compile firing rates across cells
F_plot_all = zeros(length(F_all_ind), length(I_data))
j = 1
for i in F_all_ind
    println(i)
    F_plot_all[j, :] = F_data_all[i]
    j = j+1
end

# Compute mean and std of firing rate across cells
F_plot_all_mean = mean(F_plot_all, dims= 1)
F_plot_all_std = std(F_plot_all, dims= 1)

# Extract resting state properties
t0_data = py"df.loc[$(i-1), 't0']" * 1000  # Spike times (ms)
V0_data = py"df.loc[$(i-1), 'V0']"  # Resting voltage (mV)

# Select resting state data in window [1000, 2000] ms
t0_data_plot = t0_data[1000 .<= t0_data .<= 2000] .- 1000
V0_data_plot = V0_data[1000 .<= t0_data .<= 2000]

# Extract spike properties at zero current
dV0_data = py"df.loc[$(i-1), 'dV0']"
F_data_0 = F_data[findfirst(I_data .== 0.0)]
peakamp_data = py"df.loc[$(i-1), 'spike_amp']"
peakamp_data_0 = mean(peakamp_data[findfirst(I_data .== 0.0)])

#%% ========== SETUP SIMULATION PARAMETERS ========================================
# Multi-compartment cable model with 50 dendritic segments
numseg = 50
p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_cable.jld2"), "p_fit")

# Load fitted parameter names and identify ion channel conductance indices
param_names = wload(datadir("simulations", "CSC_Ih", "fit_param_cable_names.jld2"),  "param_names")
dend_Kd_ind = findfirst(param_names .== "gKd")       # Delayed rectifier K channel
dend_Ad_ind = findfirst(param_names .== "gAd")       # A-type K channel
dend_SKd_ind = findfirst(param_names .== "gSKd")     # SK Ca-activated K channel
dend_Td_ind = findfirst(param_names .== "gTd")       # T-type Ca channel
dend_HVAd_ind = findfirst(param_names .== "gHVAd")   # HVA Ca channel
dend_hd_ind = findfirst(param_names .== "ghd")       # h-current (Ih)
V0  = -60  # Resting membrane potential (mV)
tspan_fI = (0., 1500)  # Time span for f-I curve
twindow = 1000.  # Time window for analysis (ms)

# Convert current from pA to uA/cm^2 using surface area
SA = 4* π * (p_fit[16]/10000)^2 # radius [μm] -> surface area [cm^2]
Iarray = I_data .*1e-6 ./SA  # Normalized current array


# Define timing parameters for current injection protocol
step_start = 500      # Current injection begins (ms)
step_length = 1000    # Duration of current injection (ms)
post_step = 100       # Recording time after current offset (ms)
sim_start = 0         # Simulation start time (ms)
step_end = step_length + step_start  # Current injection ends (ms)
sim_end =  step_end + post_step      # Total simulation duration (ms)
dt = 0.1              # Integration time step (ms)
tspan = (0.0, sim_end)  # Time span for ODE solver
tsteps = step_start:dt:step_end  # Time points to save
I_mag = 0.  # Placeholder for current magnitude



# ============================================================================
# Function: run_firing
# ============================================================================
# Run firing analysis with specified parameters and current protocol
# Computes firing rate, ISI, spike amplitude, and f-I curve
function run_firing(p_in, step_start, step_end, tspan, tsteps, Iarray; V0=-60, numseg=50)
    println(p_in)
    # Initialize to steady state at resting voltage
    u0_cable =  CSC_cable_Ih.steady_state_init_u0_gating(V0, numseg, p_in[24:end]);

    # Setup step current injection callback
    cbs = CSC_cable_Ih.step_I(step_start, step_end, 0.)
    prob_fit = ODEProblem(CSC_cable_Ih.CSCcable_gating!, u0_cable, tspan, p_in, save_start=false, save_end=false,
                                save_on=false, callback=cbs,  maxiters=1e25)
    # Solve ODE with ROCK2 solver (stiff integrator)
    sol = solve(prob_fit, ROCK2(), saveat = tsteps)

    # Analyze firing properties during step current injection
    F, ISI, Finst, peakamp = analysis.step_FI_analysis(sol, step_start, step_end; ind=1)
    # Extract soma voltage trace (first compartment, first state variable)
    dV = stack(sol(sol.t, Val{1}).u)[1,1,:]

    # Generate f-I curve by varying injected current
    sim, F_array = CSC_cable_Ih.fIcurve(prob_fit, Iarray, p_in, step_start, step_end)

    return sol, F, ISI, Finst, peakamp, dV, sim, F_array 
end


#%% ========== WILD-TYPE SIMULATIONS =========================================
# Run firing analysis with all ion channels intact (wild-type)
sol, F, ISI, Finst, peakamp, dV, sim, F_array = run_firing(p_fit, step_start, step_end, tspan, tsteps, Iarray; V0=-60, numseg=50)
# Save results
@tagsave(datadir("simulations", "Cable_charac", "Firing", "Firing_CSC_cable_soma_Ih_WT.jld2"),  @strdict dt sol F ISI Finst peakamp dV sim F_array Iarray;
safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))

#% ========== NO DELAYED RECTIFIER K CHANNEL (Kd) ==================================
# Knockout dendritic delayed rectifier K channel
par_f = deepcopy(p_fit)
par_f[dend_Kd_ind] = 0.  # Set gKd to zero
sol, F, ISI, Finst, peakamp, dV, sim, F_array = run_firing(par_f, step_start, step_end, tspan, tsteps, Iarray; V0=-60, numseg=50)
@tagsave(datadir("simulations", "Cable_charac", "Firing", "Firing_CSC_cable_soma_Ih_no_Kd.jld2"),  @strdict dt sol F ISI Finst peakamp dV sim F_array Iarray;
safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))

#% ========== NO A-TYPE K CHANNEL (Ad) ============================================
# Knockout dendritic A-type K channel
par_f = deepcopy(p_fit)
par_f[dend_Ad_ind] = 0.  # Set gAd to zero
sol, F, ISI, Finst, peakamp, dV, sim, F_array = run_firing(par_f, step_start, step_end, tspan, tsteps, Iarray; V0=-60, numseg=50)
@tagsave(datadir("simulations", "Cable_charac", "Firing", "Firing_CSC_cable_soma_Ih_no_Ad.jld2"),  @strdict dt sol F ISI Finst peakamp dV sim F_array Iarray;
safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))


#% ========== NO SK Ca-ACTIVATED K CHANNEL (SKd) ===================================
# Knockout dendritic SK Ca-activated K channel
par_f = deepcopy(p_fit)
par_f[dend_SKd_ind] = 0.  # Set gSKd to zero
sol, F, ISI, Finst, peakamp, dV, sim, F_array = run_firing(par_f, step_start, step_end, tspan, tsteps, Iarray; V0=-60, numseg=50)
@tagsave(datadir("simulations", "Cable_charac", "Firing", "Firing_CSC_cable_soma_Ih_no_SKd.jld2"),  @strdict dt sol F ISI Finst peakamp dV sim F_array Iarray;
safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))


#% ========== NO HIGH-VOLTAGE-ACTIVATED Ca CHANNEL (HVAd) ==========================
# Knockout dendritic HVA Ca channel
par_f = deepcopy(p_fit)
par_f[dend_HVAd_ind] = 0.  # Set gHVAd to zero
sol, F, ISI, Finst, peakamp, dV, sim, F_array = run_firing(par_f, step_start, step_end, tspan, tsteps, Iarray; V0=-60, numseg=50)
@tagsave(datadir("simulations", "Cable_charac", "Firing", "Firing_CSC_cable_soma_Ih_no_HVAd.jld2"),  @strdict dt sol F ISI Finst peakamp dV sim F_array Iarray;
safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))


#% ========== NO T-TYPE Ca CHANNEL (Td) ===========================================
# Knockout dendritic T-type Ca channel
par_f = deepcopy(p_fit)
par_f[dend_Td_ind] = 0.  # Set gTd to zero
sol, F, ISI, Finst, peakamp, dV, sim, F_array = run_firing(par_f, step_start, step_end, tspan, tsteps, Iarray; V0=-60, numseg=50)
@tagsave(datadir("simulations", "Cable_charac", "Firing", "Firing_CSC_cable_soma_Ih_no_Td.jld2"),  @strdict dt sol F ISI Finst peakamp dV sim F_array Iarray;
safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))
