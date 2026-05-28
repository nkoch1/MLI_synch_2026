"""
    OU_CSC_soma_cable_Ih.jl
    
    Simulates a cerebellar stellate cell (CSC) model with cable properties and Ih current
    subject to stochastic synaptic input modeled as Ornstein-Uhlenbeck (OU) noise.
    
    This script:
    1. Loads fitted ion channel parameters for CSC model
    2. Runs stochastic simulations with OU noise at the soma
    3. Analyzes frequency filtering and amplitude responses
    4. Tests effects of knocking out different dendritic ion channels
"""
using DrWatson
@quickactivate  "MLI_synch_2026"

# Include files from the source directory
include(srcdir("CSC_cable_Ih.jl"))
using .CSC_cable_Ih
include(srcdir("analysis.jl"))
using .analysis
using DifferentialEquations, DiffEqCallbacks, Statistics, Peaks
using FFTW, DSP
using Optim
using ProgressMeter

#%%
# Model setup ########################################
# Simulation timing and parameters
dt = 0.1  # Time step for saving (ms)
tspan = (0.0, 2*1000.0)  # Total simulation time (ms)
numseg = 50  # Number of compartments in the cable model
I = 0.  # Applied current (not used in this simulation)
bin_low = -100  # Lower bound for voltage binning (mV)
bin_high = 50  # Upper bound for voltage binning (mV)
bin_incr = 0.5  # Bin size for voltage histograms (mV)

# Load fitted ion channel parameters from previous fitting
p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_cable.jld2"), "p_fit")
p_fit = Array{Any}(p_fit)
p_fit[22] = Int(p_fit[22])  # Convert parameter 22 to integer

# Load parameter names and identify indices for dendritic ion channels
param_names = wload(datadir("simulations", "CSC_Ih", "fit_param_cable_names.jld2"),  "param_names")
dend_Kd_ind = findfirst(param_names .== "gKd")  # Delayed rectifier K channel conductance
dend_Ad_ind = findfirst(param_names .== "gAd")  # A-type K channel conductance
dend_SKd_ind = findfirst(param_names .== "gSKd")  # SK channel conductance (Ca2+-dependent)
dend_Td_ind = findfirst(param_names .== "gTd")  # T-type Ca channel conductance
dend_HVAd_ind = findfirst(param_names .== "gHVAd")  # High-voltage activated Ca channel conductance

# Set somatic Na channel conductance to zero (focus on dendritic channels)
gNa_ind = findfirst(param_names .== "gNa") 
p_fit[gNa_ind] = 0.
# Store leak reversal potential
Eleak_ind = findfirst(param_names .== "Eleak")
Eleak = p_fit[Eleak_ind]
#########################################################################################

# Initial condition
V0  = -50  # Initial resting voltage (mV)
global Vhold = V0  # Holding voltage for simulations

"""
    syn_noise!(du, u, p, t)
    
    Noise function for SDE solver. Defines how stochastic noise is applied to the system.
    Only the last compartment receives noise input.
"""
function syn_noise!(du,u,p,t)
    dA = @view  du[:,:]
    dA[2:end, :] .= 0.  # Zero out noise for all state variables except voltage
    dA[1, 1:end-1] .= 0.  # Zero out noise for all compartments except the last
    dA[2,end] = 1.  # Only the last compartment receives noise (scaled by σ from OU process)
    end

# Simulation parameters (these are set but not directly used in this script)
start_t = 25  # Transient period to skip for analysis (ms)
tend = 5000. + start_t  # Total simulation time (ms)
saveat= 0.1  # Sampling interval (ms)
sde_dt = 0.001  # SDE solver time step (ms)
num_steps = Int(floor((tend / saveat)))+1
n = 50  # Number of independent simulation runs for each condition


filename ="CSC_soma_Ih_WT"

"""
    run_multi_SDE_UO(parf, n, filename, tend)
    
    Run multiple stochastic differential equation simulations with Ornstein-Uhlenbeck noise.
    
    Arguments:
    - parf: Parameter vector for the model
    - n: Number of independent simulation runs
    - filename: Base name for saved output files
    - tend: End time for simulation (ms)
    
    Outputs saved to jld2 files:
    - Raw voltage and noise time series from all compartments
    - Frequency domain analysis (power spectrum and SEM)
    - Voltage amplitude histograms
"""
function run_multi_SDE_UO(parf, n, filename, tend)
    # Load the holding current needed to maintain resting voltage
    Ihold = wload(datadir("simulations", "Cable_charac", "new", "Ihold", "$(filename)_Ihold_$(Vhold)_new.jld2"), "Ihold")

    parf[18] = Ihold  # Add fitted Ihold to parameters
    par_i = tuple(parf...)  # Convert to tuple for SDE solver
    
    # Compute initial conditions by finding steady state
    ic = CSC_cable_Ih.steady_state_init(V0, par_i,(0, 100000), numseg)
    ic[2, end] = ic[1,end]  # Set dendritic tip voltage to dendritic resting potential
    
    # SDE simulation parameters
    saveat= 0.1  # Data saved every 0.1 ms
    dt = saveat /1000.  # SDE solver time step
    num_steps = Int(floor((tend / saveat)))+1

    # Preallocate arrays for storing results from all n runs
    first = zeros(n, num_steps)  # Voltage at first dendritic compartment
    last = zeros(n, num_steps)  # Voltage at last dendritic compartment (tip)
    noise_data = zeros(n, num_steps)  # Ornstein-Uhlenbeck noise input
    soma= zeros(n, num_steps)  # Soma voltage
    time = zeros(n, num_steps)  # Time vector

    # Ornstein-Uhlenbeck noise properties
    Θ = 1. / 10.  # Speed of mean reversion (1/tau, tau = 10 ms)
    μ = ic[1,end]  # Mean of the process (resting potential)
    σ = 10.  # Volatility scaling for Wiener process (mV/√ms)
    t0 = 0.0  # Start time
    W0 = ic[1,end]  # Initial noise value
    tspan = (0., tend)  # Time span for integration
    UO_noise = OrnsteinUhlenbeckProcess(Θ,μ,σ,t0,W0)  # Create OU process object
    
    # Run simulations in parallel (multi-threaded)
    Threads.@threads for i in range(1,n)
        println(i)  # Progress indicator
        # Set up SDE problem with cable equation dynamics and noise input
        prob2 = SDEProblem(CSC_cable_Ih.CSCcable_end!,syn_noise!,  ic::Array{Float64},tspan::Tuple{Float64, Float64},par_i, noise=UO_noise)
        # Solve using Euler-Maruyama method
        sol = solve(prob2,EM(),adaptive=false, saveat=saveat, maxiters=1e15, progress = false,progress_steps=1000000, dt=dt,  save_idxs = [1,9*2-8, 9*51-8,9*51-7],save_noise=false)
        last[i,:]=sol[3,:]  # Last compartment voltage
        first[i,:]= sol[2,:]  # First dendritic compartment voltage
        noise_data[i,:]=sol[end, :]  # Noise time series
        soma[i,:]=sol[1,:]  # Soma voltage
        time[i,:] = sol.t  # Time vector
    end

    # Save raw simulation data
    param = Dict(
        "Θ" => Θ,
        "σ" => σ,
        "tspan" => tspan,
        "tend" => tend,
        "saveat" => saveat,
        "Vhold" => Vhold,
        "n"=> n,
    )
    @tagsave(datadir("simulations", "Cable_charac", "new", "OU", savename("OU_$(filename)_full", param, "jld2")),  @strdict noise_data first last soma time dt Ihold ic;
             safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))
             
    # Analyze frequency response (power spectrum)
    power_F, power, power_SEM, power_F_first, power_first, power_SEM_first  = analysis.OU_frequency_filt(; soma=soma,first=first, noise=noise_data, last=last,  dt=dt, start_t = start_t)
    param_F = Dict(
        "Θ" => Θ,
        "σ" => σ,
        "tspan" => tspan,
        "tend" => tend,
        "saveat" => saveat,
        "Vhold" => Vhold,
        "n"=> n,
    )
    # Save frequency analysis results
    @tagsave(datadir("simulations", "Cable_charac", "new", "OU", savename("OU_$(filename)_FREQ_full", param_F, "jld2")),  @strdict power_F power power_SEM power_F_first power_first power_SEM_first;
             safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))

    # Analyze voltage amplitude distributions (binned statistics)
    hsoma, hfirst, hlast, hnoise, bin_middle,n = analysis.amplitude_binned(bin_low, bin_high, bin_incr,false; soma=soma, first=first, last=last, noise=noise_data)       
    param_amp = Dict(
        "Θ" => Θ,
        "σ" => σ,
        "tspan" => tspan,
        "tend" => tend,
        "saveat" => saveat,
        "Vhold" => Vhold,
        "n"=> n,
    )
    # Save amplitude analysis results
    @tagsave(datadir("simulations", "Cable_charac", "new", "OU", savename("OU_$(filename)_AMP_full", param_amp, "jld2")),  @strdict hsoma hfirst hlast hnoise bin_middle n;
             safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))
    GC.gc()  # Force garbage collection
    nothing
    end

#%% WT (Wild-Type) #############################################################################################
# Baseline simulation with all dendritic ion channels intact
par = deepcopy(p_fit)
run_multi_SDE_UO(par, n, "CSC_soma_Ih_WT", tend)


#%% no dend Kd (Delayed Rectifier K Channel Knockout) #############################################################################################
# Test role of dendritic delayed rectifier K channel (Kd) in filtering
par = deepcopy(p_fit)
par[dend_Kd_ind] = 0.
run_multi_SDE_UO(par, n, "CSC_soma_Ih_no_Kd", tend)

#%% no dend Ad (A-type K Channel Knockout) #############################################################################################
# Test role of dendritic A-type K channel (Ad) in filtering
par = deepcopy(p_fit)
par[dend_Ad_ind] = 0.

run_multi_SDE_UO(par, n, "CSC_soma_Ih_no_Ad", tend)

#%% no dend SKd (SK Channel Knockout) #############################################################################################
# Test role of dendritic SK channel (calcium-dependent) in filtering
par = deepcopy(p_fit)
par[dend_SKd_ind] = 0.

run_multi_SDE_UO(par, n, "CSC_soma_Ih_no_SKd", tend)

#%% no dend Td (T-type Ca Channel Knockout) #############################################################################################
# Test role of dendritic T-type calcium channel (Td) in filtering
par = deepcopy(p_fit)
par[dend_Td_ind] = 0.

run_multi_SDE_UO(par, n, "CSC_soma_Ih_no_Td", tend)

#%% no dend HVAd (High-Voltage Activated Ca Channel Knockout) #############################################################################################
# Test role of dendritic high-voltage activated calcium channel (HVAd) in filtering
par = deepcopy(p_fit)
par[dend_HVAd_ind] = 0.

run_multi_SDE_UO(par, n, "CSC_soma_Ih_no_HVAd", tend)
