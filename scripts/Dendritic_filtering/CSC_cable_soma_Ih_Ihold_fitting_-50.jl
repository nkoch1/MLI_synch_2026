# ==============================================================================
# CSC Cable Model - Soma Ih Ihold (Holding Current) Fitting at -50 mV
# ==============================================================================
# This script fits the holding current (Ihold) required to maintain the soma
# voltage at a specific level (-50 mV) for different channel knockout conditions
# in a cerebellar stellate cell (CSC) cable model with hyperpolarization-activated
# current (Ih).

using DrWatson
@quickactivate  "MLI_synch_2026"
include(srcdir("CSC_cable_Ih.jl"))
using .CSC_cable_Ih
using DifferentialEquations, DiffEqCallbacks, Statistics, Peaks	
using FFTW, DSP
using Optim
using ProgressMeter

# Simulation parameters
dt = 0.1                          # Time step (ms)
tspan = (0.0, 100*1000.0)         # Simulation time span (ms)
numseg = 50                        # Number of dendritic segments
I = 0.                             # Initial injected current (will be optimized)

# ==============================================================================
# Load fitted parameters and identify conductance indices
# ==============================================================================
# Load fitted parameters from the cable model optimization
p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_cable.jld2"), "p_fit")
p_fit = Array{Any}(p_fit)
p_fit[22] = Int(p_fit[22])

# Load parameter names to identify which index corresponds to each conductance
param_names = wload(datadir("simulations", "CSC_Ih", "fit_param_cable_names.jld2"),  "param_names")

# Identify indices for dendritic conductances
dend_Kd_ind = findfirst(param_names .== "gKd")      # Delayed rectifier K channel
dend_Ad_ind = findfirst(param_names .== "gAd")      # A-type K channel
dend_SKd_ind = findfirst(param_names .== "gSKd")    # SK (small conductance) K channel
dend_Td_ind = findfirst(param_names .== "gTd")      # T-type Ca channel
dend_HVAd_ind = findfirst(param_names .== "gHVAd")  # HVA (high voltage-activated) Ca channel

# Set somatic Na conductance to zero and retrieve resting leak current
gNa_ind = findfirst(param_names .== "gNa") 
p_fit[gNa_ind] = 0.
Eleak_ind = findfirst(param_names .== "Eleak")
Eleak = p_fit[Eleak_ind]

# ==============================================================================
# Create parameter sets for different channel knockout conditions
# ==============================================================================
# Wild-type parameters (baseline)
par_WT = deepcopy(p_fit)

# Knockout conditions: set specific conductances to zero
par_no_Kd = deepcopy(p_fit)
par_no_Kd[dend_Kd_ind] = 0.    # Remove Kd (delayed rectifier K)

par_no_Ad = deepcopy(p_fit)
par_no_Ad[dend_Ad_ind] = 0.    # Remove Ad (A-type K)

par_no_SKd = deepcopy(p_fit)
par_no_SKd[dend_SKd_ind] = 0.  # Remove SKd (SK K)

par_no_Td = deepcopy(p_fit)
par_no_Td[dend_Td_ind] = 0.    # Remove Td (T-type Ca)

par_no_HVAd = deepcopy(p_fit)
par_no_HVAd[dend_HVAd_ind] = 0. # Remove HVAd (HVA Ca)

par_no_hd = deepcopy(p_fit)
par_no_hd[dend_hd_ind] = 0.     # Remove Ih (hyperpolarization-activated)

# ==============================================================================
# Organize parameter sets and output filenames for batch processing
# ==============================================================================
# Array of parameter sets for all conditions to test
par_array =[
    par_WT,           # Wild-type (all channels)
    par_no_Kd,        # Without delayed rectifier K
    par_no_Ad,        # Without A-type K
    par_no_SKd,       # Without SK K
    par_no_Td,        # Without T-type Ca
    par_no_HVAd,      # Without HVA Ca
    ]

# Corresponding output filenames for each condition
filename_array = [
    "CSC_soma_Ih_WT",
    "CSC_soma_Ih_no_Kd",
    "CSC_soma_Ih_no_Ad",
    "CSC_soma_Ih_no_SKd",
    "CSC_soma_Ih_no_Td",
    "CSC_soma_Ih_no_HVAd",
    ]

# ==============================================================================
# Optimization functions for fitting holding current
# ==============================================================================
# Target voltage for holding current fitting
V0  = -50  # Target soma voltage (mV)

# Objective function: minimize the squared error between achieved voltage and target
# x[1] = injected current (Ihold) to be optimized
# Returns: (achieved_voltage - target_voltage)^2
function f(x, parf)
    println(x[1])
    local parf[18] = x[1]  # Set injected current parameter
    # Run steady-state simulation and get achieved voltage at soma
    mV = CSC_cable_Ih.steady_state_init(V0, parf,(0, 10000), numseg; abstol=1e-9, reltol=1e-9)[1] 
    println(mV)
    return (mV - V0)^2  # Return squared error
    end


# Fit Ihold by minimizing the objective function f
# Input: parin = parameter set for specific channel knockout condition
# Output: Ihold = optimal injected current to maintain V0
function fit_Ihold(parin)
    x0 = [-1.5]        # Initial guess for current (pA)
    lower = [-15.]      # Lower bound for search
    upper = [5.]        # Upper bound for search
    results = Optim.optimize(x -> f(x, parin), lower, upper, x0)
    minPar = Optim.minimizer(results)
    return minPar[1]
    end

# Wrapper function to run the Ihold fitting
# Input: par = parameter set
# Output: Ihold = optimal holding current
function run_Ihold_fit(par)
    println(par)
    Ihold = fit_Ihold(par) # Fit Ihold to maintain steady-state at V0
    return Ihold
end                    

# ==============================================================================
# Execute Ihold fitting for all channel conditions
# ==============================================================================
# Loop through all conditions and fit Ihold for each
for i in range(1, size(filename_array)[1])
    println(i)
    par_i = par_array[i]
    filename_i = filename_array[i]
    Ihold = run_Ihold_fit(par_i)
    # Save fitted Ihold value to disk
    wsave(datadir("simulations", "Cable_charac","new", "Ihold", "$(filename_i)_Ihold_$(V0)_new.jld2"),  @strdict Ihold)
    print(filename_i)
end
