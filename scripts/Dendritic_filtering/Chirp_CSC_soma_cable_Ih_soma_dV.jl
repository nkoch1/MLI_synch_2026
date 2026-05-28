# ============================================================================================
# Chirp stimulus response analysis for cerebellar stellate cells (CSC) with cable model
# ============================================================================================
# This script runs dendritic filtering experiments using chirp stimulus inputs
# on a compartmental CSC model with soma, dendrite, and Ih current

using DrWatson
@quickactivate  "MLI_synch_2026"
include(srcdir("CSC_cable_Ih_I.jl"))
using .CSC_cable_Ih_I
include(srcdir("analysis.jl"))
using .analysis
using DifferentialEquations, DiffEqCallbacks, Statistics, Peaks
using FFTW, DSP  # Fast Fourier Transform and Digital Signal Processing

# ============================================================================================
# MODEL SETUP: Simulation parameters and time configuration
# ============================================================================================
dt = 0.1               # Time step for saving (ms)
tspan = (0.0, 1000*1000.0)  # Total simulation time: 1 million ms (~16.7 minutes)
numseg = 50            # Number of dendritic compartments
I_type = "chirp"       # Input stimulus type: frequency sweep
input_loc = 1         # Location of current injection (soma = 1)
I = 0.                 # Initial injected current

# Load fitted parameters from previous cable model fits
p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_cable.jld2"), "p_fit")
p_fit = Array{Any}(p_fit)
p_fit[22] = Int(p_fit[22])  # Ensure index 22 is integer type

# Get indices of dendritic ion channels in parameter array
param_names = wload(datadir("simulations", "CSC_Ih", "fit_param_cable_names.jld2"),  "param_names")
dend_Kd_ind = findfirst(param_names .== "gKd")      # Delayed rectifier K channel
dend_Ad_ind = findfirst(param_names .== "gAd")      # A-type K channel
dend_SKd_ind = findfirst(param_names .== "gSKd")    # SK (small conductance) K channel
dend_Td_ind = findfirst(param_names .== "gTd")      # T-type calcium channel
dend_HVAd_ind = findfirst(param_names .== "gHVAd")  # High-voltage-activated calcium channel

# Disable sodium current (focus on voltage attenuation, not action potentials)
gNa_ind = findfirst(param_names .== "gNa") 
p_fit[gNa_ind] = 0.
# ============================================================================================
# CHIRP STIMULUS SETUP: Frequency sweep from 1 Hz to 100 Hz
# ============================================================================================
# Initial condition: holding potential
global V0  = -60  # Holding voltage (mV)

# Configure chirp stimulus parameters
CSC_cable_Ih_I.phi0 = 0.                           # Initial phase at t=0
CSC_cable_Ih_I.f0 = 0.001 /1000.                   # Starting frequency: 1 Hz (1/s)
CSC_cable_Ih_I.f1 = 100. /1000.                    # End frequency: 100 Hz
CSC_cable_Ih_I.T = tspan[2]                        # Total sweep time (ms)
CSC_cable_Ih_I.c = (CSC_cable_Ih_I.f1- CSC_cable_Ih_I.f0)/CSC_cable_Ih_I.T  # Chirp rate (Hz/ms)
CSC_cable_Ih_I.Iscale = parse(Float64, ARGS[1])    # Stimulus amplitude (pA), passed as command-line argument

I_type = "chirp"
input_loc = 1

V0 = [-50]  # Test voltage: -50 mV

# ============================================================================================
# SIMULATION 1: WT (WILD-TYPE) - baseline with all dendritic channels intact
# ============================================================================================
filename ="WT"
par_f = deepcopy(p_fit)  # Copy base parameters
Ihold = wload(datadir("simulations", "Cable_charac", "new", "Ihold", "CSC_soma_Ih_$(filename)_Ihold_$(V0)_new.jld2"), "Ihold")
par_f[18] = Ihold         # Set holding current for soma
p_fit[22] = Int(p_fit[22])
par_ic = (tuple(par_f...)..., "", input_loc)  # Parameters for initialization
par = (tuple(par_f...)..., I_type, input_loc, V0)  # Parameters for simulation

# Initialize to steady state at holding potential
ic = CSC_cable_Ih_I.steady_state_init(V0, par_ic,(0, 100000), Int(numseg))
ic[2,end] = 0.  
# Create and solve ODE problem
prob = ODEProblem(CSC_cable_Ih_I.CSCcable_soma!,ic,tspan, par)
sol = solve(prob, ROCK2(), maxiters=1e15, dt=1e-20, dtmax = 0.25, dtmin=1e-300, 
            alg_hints=[:stiff], saveat = dt,  save_idxs = [1, 9*2-8, 9*51-8])  # Save soma, 1st, and last dend compartment

# Perform frequency analysis of chirp response in three compartments
Freqs_soma, F_soma = analysis.Chirp_compart_F_analysis(1, false; sol=sol, dt=dt)  # Soma
Freqs_first, F_first = analysis.Chirp_compart_F_analysis(2, false; sol=sol, dt=dt)  # First dendritic compartment
Freqs_end, F_end = analysis.Chirp_compart_F_analysis(3, false; sol=sol, dt=dt)      # Distal dendrite

# Store chirp parameters for reproducibility
param = Dict(
    "phi0" => CSC_cable_Ih_I.phi0,
    "f0" => CSC_cable_Ih_I.f0,
    "f1" => CSC_cable_Ih_I.f1,
    "T" => CSC_cable_Ih_I.T,
    "Iscale" => CSC_cable_Ih_I.Iscale,
    "Vhold" => V0
)
# Save simulation results
@tagsave(datadir("simulations", "Cable_charac", "new", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")),
         @strdict sol dt Freqs_soma F_soma Freqs_first F_first Freqs_end F_end;
         safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))



# ============================================================================================
# SIMULATION 2: NO_Kd - Test role of dendritic delayed rectifier K channel
# ============================================================================================
filename ="no_Kd"
par_f = deepcopy(p_fit)  # Copy base parameters
Ihold = wload(datadir("simulations", "Cable_charac", "new", "Ihold", "CSC_soma_Ih_$(filename)_Ihold_$(V0)_new.jld2"), "Ihold")
par_f[18] = Ihold         # Set holding current for soma
par_f[dend_Kd_ind] = 0.   # Set dendritic Kd conductance to zero
par_ic = (tuple(par_f...)..., "", input_loc)  # Parameters for initialization
par = (tuple(par_f...)..., I_type, input_loc, V0)  # Parameters for simulation

# Initialize to steady state at holding potential
ic = CSC_cable_Ih_I.steady_state_init(V0, par_ic,(0, 100000), numseg)
ic[2,end] = 0.  
# Create and solve ODE problem
prob = ODEProblem(CSC_cable_Ih_I.CSCcable_soma!,ic,tspan, par)
sol = solve(prob, ROCK2(), maxiters=1e15, dt=1e-20, dtmax = 0.25, dtmin=1e-300,
            alg_hints=[:stiff], saveat = dt,  save_idxs = [1, 9*2-8, 9*51-8])  # Save soma, 1st, and last dend compartment

# Perform frequency analysis of chirp response in three compartments
Freqs_soma, F_soma = analysis.Chirp_compart_F_analysis(1,false; sol=sol, dt=dt)  # Soma
Freqs_first, F_first = analysis.Chirp_compart_F_analysis(2,false; sol=sol, dt=dt)  # First dendritic compartment
Freqs_end, F_end = analysis.Chirp_compart_F_analysis(3,false; sol=sol, dt=dt)      # Distal dendrite

# Store chirp parameters for reproducibility
param = Dict(
    "phi0" => CSC_cable_Ih_I.phi0,
    "f0" => CSC_cable_Ih_I.f0,
    "f1" => CSC_cable_Ih_I.f1,
    "T" => CSC_cable_Ih_I.T,
    "Iscale" => CSC_cable_Ih_I.Iscale,
    "Vhold" => V0
)
# Save simulation results
@tagsave(datadir("simulations", "Cable_charac", "new", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")),
         @strdict sol dt Freqs_soma F_soma Freqs_first F_first Freqs_end F_end;
         safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))

# ============================================================================================
# SIMULATION 3: NO_AD - Test role of dendritic A-type K channel
# ============================================================================================
filename ="no_Ad"
par_f = deepcopy(p_fit)
Ihold = wload(datadir("simulations", "Cable_charac", "new", "Ihold", "CSC_soma_Ih_$(filename)_Ihold_$(V0)_new.jld2"), "Ihold")
par_f[18] = Ihold 
par_f[dend_Ad_ind] = 0.  # Set dendritic A-type conductance to zero
par_ic = (tuple(par_f...)..., "", input_loc)
par = (tuple(par_f...)..., I_type, input_loc, V0)

ic = CSC_cable_Ih_I.steady_state_init(V0, par_ic,(0, 100000), numseg)
ic[2,end] = 0.
prob = ODEProblem(CSC_cable_Ih_I.CSCcable_soma!,ic,tspan, par)
sol = solve(prob, ROCK2(), maxiters=1e15, dt=1e-20, dtmax = 0.25, dtmin=1e-300,
            alg_hints=[:stiff], saveat = dt, save_idxs = [1, 9*2-8, 9*51-8])
Freqs_soma, F_soma = analysis.Chirp_compart_F_analysis(1, false; sol=sol, dt=dt)
Freqs_first, F_first = analysis.Chirp_compart_F_analysis(2, false; sol=sol, dt=dt)
Freqs_end, F_end = analysis.Chirp_compart_F_analysis(3, false; sol=sol, dt=dt)
param = Dict(
    "phi0" => CSC_cable_Ih_I.phi0,
    "f0" => CSC_cable_Ih_I.f0,
    "f1" => CSC_cable_Ih_I.f1,
    "T" => CSC_cable_Ih_I.T,
    "Iscale" => CSC_cable_Ih_I.Iscale,
    "Vhold" => V0
)
@tagsave(datadir("simulations", "Cable_charac", "new", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")),
         @strdict sol dt Freqs_soma F_soma Freqs_first F_first Freqs_end F_end;
         safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))

# ============================================================================================
# SIMULATION 4: NO_SKD - Test role of dendritic SK (small conductance) K channel
# ============================================================================================
filename ="no_SKd"
par_f = deepcopy(p_fit)
Ihold = wload(datadir("simulations", "Cable_charac", "new", "Ihold", "CSC_soma_Ih_$(filename)_Ihold_$(V0)_new.jld2"), "Ihold")
par_f[18] = Ihold 
par_f[dend_SKd_ind] = 0.  # Set dendritic SK conductance to zero
par_ic = (tuple(par_f...)..., "", input_loc)
par = (tuple(par_f...)..., I_type, input_loc, V0)

ic = CSC_cable_Ih_I.steady_state_init(V0, par_ic,(0, 100000), numseg)
ic[2,end] = 0.
prob = ODEProblem(CSC_cable_Ih_I.CSCcable_soma!,ic,tspan, par)
sol = solve(prob, ROCK2(), maxiters=1e15, dt=1e-20, dtmax = 0.25, dtmin=1e-300,
            alg_hints=[:stiff], saveat = dt, save_idxs = [1, 9*2-8, 9*51-8])
Freqs_soma, F_soma = analysis.Chirp_compart_F_analysis(1, false; sol=sol, dt=dt)
Freqs_first, F_first = analysis.Chirp_compart_F_analysis(2, false; sol=sol, dt=dt)
Freqs_end, F_end = analysis.Chirp_compart_F_analysis(3, false; sol=sol, dt=dt)
param = Dict(
    "phi0" => CSC_cable_Ih_I.phi0,
    "f0" => CSC_cable_Ih_I.f0,
    "f1" => CSC_cable_Ih_I.f1,
    "T" => CSC_cable_Ih_I.T,
    "Iscale" => CSC_cable_Ih_I.Iscale,
    "Vhold" => V0
)
@tagsave(datadir("simulations", "Cable_charac", "new", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")),
         @strdict sol dt Freqs_soma F_soma Freqs_first F_first Freqs_end F_end;
         safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))

# ============================================================================================
# SIMULATION 5: NO_TD - Test role of dendritic T-type calcium channel
# ============================================================================================
filename ="no_Td"
par_f = deepcopy(p_fit)
Ihold = wload(datadir("simulations", "Cable_charac", "new", "Ihold", "CSC_soma_Ih_$(filename)_Ihold_$(V0)_new.jld2"), "Ihold")
par_f[18] = Ihold 
par_f[dend_Td_ind] = 0.  # Set dendritic T-type Ca conductance to zero
par_ic = (tuple(par_f...)..., "", input_loc)
par = (tuple(par_f...)..., I_type, input_loc, V0)

ic = CSC_cable_Ih_I.steady_state_init(V0, par_ic,(0, 100000), numseg)
ic[2,end] = 0.
prob = ODEProblem(CSC_cable_Ih_I.CSCcable_soma!,ic,tspan, par)
sol = solve(prob, ROCK2(), maxiters=1e15, dt=1e-20, dtmax = 0.25, dtmin=1e-300,
            alg_hints=[:stiff], saveat = dt, save_idxs = [1, 9*2-8, 9*51-8])
Freqs_soma, F_soma = analysis.Chirp_compart_F_analysis(1, false; sol=sol, dt=dt)
Freqs_first, F_first = analysis.Chirp_compart_F_analysis(2, false; sol=sol, dt=dt)
Freqs_end, F_end = analysis.Chirp_compart_F_analysis(3, false; sol=sol, dt=dt)
param = Dict(
    "phi0" => CSC_cable_Ih_I.phi0,
    "f0" => CSC_cable_Ih_I.f0,
    "f1" => CSC_cable_Ih_I.f1,
    "T" => CSC_cable_Ih_I.T,
    "Iscale" => CSC_cable_Ih_I.Iscale,
    "Vhold" => V0
)
@tagsave(datadir("simulations", "Cable_charac", "new", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")),
         @strdict sol dt Freqs_soma F_soma Freqs_first F_first Freqs_end F_end;
         safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))

# ============================================================================================
# SIMULATION 6: NO_HVAD - Test role of dendritic HVA (high-voltage-activated) Ca channel
# ============================================================================================
filename ="no_HVAd"
par_f = deepcopy(p_fit)
Ihold = wload(datadir("simulations", "Cable_charac", "new", "Ihold", "CSC_soma_Ih_$(filename)_Ihold_$(V0)_new.jld2"), "Ihold")
par_f[18] = Ihold 
par_f[dend_HVAd_ind] = 0.  # Set dendritic HVA Ca conductance to zero
par_ic = (tuple(par_f...)..., "", input_loc)
par = (tuple(par_f...)..., I_type, input_loc, V0)

ic = CSC_cable_Ih_I.steady_state_init(V0, par_ic,(0, 100000), numseg)
ic[2,end] = 0.
prob = ODEProblem(CSC_cable_Ih_I.CSCcable_soma!,ic,tspan, par)
sol = solve(prob, ROCK2(), maxiters=1e15, dt=1e-20, dtmax = 0.25, dtmin=1e-300,
            alg_hints=[:stiff], saveat = dt, save_idxs = [1, 9*2-8, 9*51-8])\nFreqs_soma, F_soma = analysis.Chirp_compart_F_analysis(1, false; sol=sol, dt=dt)  # Soma analysis\nFreqs_first, F_first = analysis.Chirp_compart_F_analysis(2, false; sol=sol, dt=dt)  # First dend compartment\nFreqs_end, F_end = analysis.Chirp_compart_F_analysis(3, false; sol=sol, dt=dt)      # Distal dendrite\nparam = Dict(\n    \"phi0\" => CSC_cable_Ih_I.phi0,\n    \"f0\" => CSC_cable_Ih_I.f0,\n    \"f1\" => CSC_cable_Ih_I.f1,\n    \"T\" => CSC_cable_Ih_I.T,\n    \"Iscale\" => CSC_cable_Ih_I.Iscale,\n    \"Vhold\" => V0\n)\n# Save simulation results\n@tagsave(datadir(\"simulations\", \"Cable_charac\", \"new\", \"Chirp\", savename(\"Chirp_soma_dV_CSC_soma_Ih_$(filename)\", param, \"jld2\")),\n         @strdict sol dt Freqs_soma F_soma Freqs_first F_first Freqs_end F_end;\n         safe = DrWatson.readenv(\"DRWATSON_SAFESAVE\", true))"
