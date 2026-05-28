using DrWatson
@quickactivate  "MLI_synch_2026"
using DifferentialEquations, DiffEqCallbacks,
    Optimization,  SciMLSensitivity,
    Zygote,  DiffEqCallbacks, JLD2, Statistics, Peaks
using FHist
using PyCall
using PyPlot
using FFTW, DSP
include(srcdir("pyplot_fxns.jl"))
using .pyplot_fxns
include(srcdir("CSC_cable_Ih_I.jl"))
using .CSC_cable_Ih_I

#%% Load data
p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_cable.jld2"), "p_fit")
param_names = wload(datadir("simulations", "CSC_Ih", "fit_param_cable_names.jld2"),  "param_names")
Eleak_ind = findfirst(param_names .== "Eleak")
Eleak = p_fit[Eleak_ind]

# set up SDE simulation parameters
start_t = 25
tend = 5000. + start_t 
saveat= 0.1
sde_dt = 0.001 
num_steps = Int(floor((tend / saveat)))+1
n = 50 # number of runs for each

Θ = 1. / 10. # Speed of the mean reversion (scaling distance between Xt and μ) # Excitatory time constant ~2ms
μ = Eleak #mean of the process
σ = 10. # volatility that scales standard Wiener process (dWt)
t0 = 0.0
W0 = 0.0
tspan = (0., tend)
Vhold = -50


param = Dict( "Θ" => Θ, "σ" => σ, "tspan" => tspan, "tend" => tend, "saveat" => saveat,"Vhold" => Vhold, "n" => n,);
param_F = Dict("Θ" => Θ, "σ" => σ, "tspan" => tspan, "tend" => tend, "saveat" => saveat, "Vhold" => Vhold, "n" => n,);
param_amp = Dict( "Θ" => Θ, "σ" => σ,"tspan" => tspan, "tend" => tend, "saveat" => saveat, "Vhold" => Vhold, "n" => n,);

filename = "OU_CSC_soma_Ih_WT"
power_F_WT =  wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power_F");
power_WT = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power");
power_SEM_WT =  wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power_SEM");
soma_WT = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_full", param,"jld2")), "soma");
soma_t = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_full", param,"jld2")), "time") ./ 1000;
hsoma_WT = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hsoma");
hfirst_WT  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hfirst");
hlast_WT  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hlast");
hnoise_WT  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hnoise");
bin_WT = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "bin_middle");
n_WT = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "n");

filename = "OU_CSC_soma_Ih_no_Kd"
power_F_no_Kd =  wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power_F");
power_no_Kd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power");
power_SEM_no_Kd =  wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power_SEM");
soma_no_Kd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_full", param,"jld2")), "soma");
hsoma_no_Kd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hsoma");
hfirst_no_Kd  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hfirst");
hlast_no_Kd  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hlast");
hnoise_no_Kd  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hnoise");
bin_no_Kd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "bin_middle");
n_no_Kd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "n");

filename = "OU_CSC_soma_Ih_no_Ad"
power_F_no_Ad =  wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power_F");
power_no_Ad = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power");
power_SEM_no_Ad =  wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power_SEM");
soma_no_Ad = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_full", param,"jld2")), "soma");
hsoma_no_Ad = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hsoma");
hfirst_no_Ad  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hfirst");
hlast_no_Ad  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hlast");
hnoise_no_Ad  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hnoise");
bin_no_Ad = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "bin_middle");
n_no_Ad = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "n");


filename = "OU_CSC_soma_Ih_no_SKd"
power_F_no_SKd =  wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power_F");
power_no_SKd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power");
power_SEM_no_SKd =  wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power_SEM");
soma_no_SKd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_full", param,"jld2")), "soma");
hsoma_no_SKd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hsoma");
hfirst_no_SKd  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hfirst");
hlast_no_SKd  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hlast");
hnoise_no_SKd  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hnoise");
bin_no_SKd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "bin_middle");
n_no_SKd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "n");

filename = "OU_CSC_soma_Ih_no_Td"
power_F_no_Td =  wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power_F");
power_no_Td = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power");
power_SEM_no_Td =  wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power_SEM");
soma_no_Td = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_full", param,"jld2")), "soma");
hsoma_no_Td = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hsoma");
hfirst_no_Td  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hfirst");
hlast_no_Td  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hlast");
hnoise_no_Td  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hnoise");
bin_no_Td = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "bin_middle");
n_no_Td = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "n");

filename = "OU_CSC_soma_Ih_no_HVAd"
power_F_no_HVAd =  wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power_F");
power_no_HVAd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power");
power_SEM_no_HVAd =  wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_FREQ_full", param_F, "jld2")), "power_SEM");
soma_no_HVAd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_full", param,"jld2")), "soma");
hsoma_no_HVAd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hsoma");
hfirst_no_HVAd  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hfirst");
hlast_no_HVAd  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hlast");
hnoise_no_HVAd  = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "hnoise");
bin_no_HVAd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "bin_middle");
n_no_HVAd = wload(datadir("simulations", "Cable_charac", "OU", savename("$(filename)_AMP_full", param_amp, "jld2")), "n");

# CHIRP DATA setup

# SETUP CHIRP
V0 =-50
tspan = (0.0, 100*1000.0)
phi0 = 0. # initial phase at t=0
f0 = 0.001 /1000.  # starting frequency Hz = 1/s
f1 = 100. /1000.  # end frequency
T = 1e6  # msec  # time it takes to sweep from f0 to f1
c = (f1- f0)/T # chirp rate
Iscale = 5. # mV

I_type = "chirp"
input_loc = 1
param = Dict("phi0" => phi0, "f0" => f0, "f1" => f1, "T" =>T, "Iscale" => Iscale, "Vhold"=>V0
)




#%
function plot_F(ax, filename, param, color, line_alpha; p=false, Ftext=0.1, ytext =0.1)
    F_first = wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "F_first");
    F_end =wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "F_end");
    Freqs =wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "Freqs_end");
    pow =  DSP.pow2db.(abs.(F_end) ./ abs.(F_first));
    F_ind = findfirst(Freqs .>= f1*1000);
    F_ind_start = findfirst(Freqs .>= F_lim[1]);
    ax.plot(Freqs[F_ind_start:F_ind], pow[F_ind_start:F_ind],  color = color, alpha= line_alpha)
    if p
        ax.text(Ftext, ytext, "$(floor(Int, param["Iscale"])) mV", fontsize=8, ha="left", color=color)
    end
return ax
end

function plot_timeseries(ax, filename, param, color)
    sol = wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "sol");
    ax.plot(sol.t ./1000, sol[2,:], color=color, zorder=1)
return ax
end

function plot_timeseries_end(ax, filename, param, color)
    sol = wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "sol");
    ax.plot(sol.t ./1000, sol[3,:], color=color, zorder=1, "--")
return ax
end



function plot_F_norm(ax, filename, param, color, line_alpha)
    F_first_WT = wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_WT", param, "jld2")), "F_first");
    F_end_WT =wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_WT", param, "jld2")), "F_end");
    Freqs_WT =wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_WT", param, "jld2")), "Freqs_end");
    pow_WT =  DSP.pow2db.(abs.(F_end_WT) ./ abs.(F_first_WT));
    F_ind = findfirst(Freqs_WT .>= f1*1000);
    F_ind_start = findfirst(Freqs_WT .>= F_lim[1]);

    F_first = wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "F_first");
    F_end =wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "F_end");
    Freqs =wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "Freqs_end");
    pow =  DSP.pow2db.(abs.(F_end) ./ abs.(F_first));
    F_ind = findfirst(Freqs .>= f1*1000);
    F_ind_start = findfirst(Freqs .>= F_lim[1]);
    ax.plot(Freqs[F_ind_start:F_ind],pow[F_ind_start:F_ind] .- pow_WT[F_ind_start:F_ind],  color = color, alpha= line_alpha)
return ax
end

global V0 = -50

# SETUP CHIRP
V0 =-50
tspan = (0.0, 100*1000.0)
phi0 = 0. # initial phase at t=0
f0 = 0.001 /1000. #0.5 /1000. #  starting frequency Hz = 1/s
f1 = 100. /1000.  # end frequency
T = 1e6 # msec  # time it takes to sweep from f0 to f1
c = (f1- f0)/T # chirp rate
Iscale = 5. # mV

I_type = "chirp"
input_loc = 1
param = Dict("phi0" => phi0, "f0" => f0, "f1" => f1, "T" =>T, "Iscale" => Iscale, "Vhold"=>V0);



function plot_F(ax, filename, param, color, line_alpha; p=false, Ftext=0.1, ytext =0.1)
    F_first = wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "F_first");
    F_end =wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "F_end");
    Freqs =wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "Freqs_end");
    pow =  DSP.pow2db.(abs.(F_end) ./ abs.(F_first));
    F_ind = findfirst(Freqs .>= f1*1000);
    F_ind_start = findfirst(Freqs .>= F_lim[1]);
    ax.plot(Freqs[F_ind_start:F_ind], pow[F_ind_start:F_ind],  color = color, alpha= line_alpha)
    if p
        ax.text(Ftext, ytext, "$(floor(Int, param["Iscale"])) mV", fontsize=7, ha="left", color=color)
    end
return ax
end

function plot_timeseries(ax, filename, param, color)
    sol = wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "sol");
    ax.plot(sol.t ./1000, reduce(hcat,sol.u)[2,:], color=color, zorder=1)

    
return ax
end

function plot_timeseries_end(ax, filename, param, color)
    sol = wload(datadir("simulations", "Cable_charac", "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "sol");
    ax.plot(sol.t ./1000, reduce(hcat,sol.u)[3,:], color=color, zorder=1, "--")
return ax
end



function plot_F_norm(ax, filename, param, color, line_alpha)
    F_first_WT = wload(datadir("simulations", "Cable_charac",  "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_WT", param, "jld2")), "F_first");
    F_end_WT =wload(datadir("simulations", "Cable_charac",  "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_WT", param, "jld2")), "F_end");
    Freqs_WT =wload(datadir("simulations", "Cable_charac",  "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_WT", param, "jld2")), "Freqs_end");
    pow_WT =  DSP.pow2db.(abs.(F_end_WT) ./ abs.(F_first_WT));
    F_ind = findfirst(Freqs_WT .>= f1*1000);
    F_ind_start = findfirst(Freqs_WT .>= F_lim[1]);

    F_first = wload(datadir("simulations", "Cable_charac",  "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "F_first");
    F_end =wload(datadir("simulations", "Cable_charac",  "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "F_end");
    Freqs =wload(datadir("simulations", "Cable_charac",  "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_$(filename)", param, "jld2")), "Freqs_end");
    pow =  DSP.pow2db.(abs.(F_end) ./ abs.(F_first));
    F_ind = findfirst(Freqs .>= f1*1000);
    F_ind_start = findfirst(Freqs .>= F_lim[1]);
    ax.plot(Freqs[F_ind_start:F_ind],pow[F_ind_start:F_ind] .- pow_WT[F_ind_start:F_ind],  color = color, alpha= line_alpha)
return ax
end

global V0 = -50


Iscale_array = [5., 10., 15.]

param = Dict("phi0" => phi0, "f0" => f0, "f1" => f1, "T" =>T, "Iscale" => Iscale, "Vhold" => V0)
# Chirp input time series
sol_WT = wload(datadir("simulations", "Cable_charac",  "Chirp", savename("Chirp_soma_dV_CSC_soma_Ih_WT", param, "jld2")), "sol");
chirp_ex = CSC_cable_Ih_I.chirp(sol_WT.t, Iscale, phi0, c, f0) .+ V0;

#%% FIGURE ######################################################
using LaTeXStrings
@pyimport matplotlib.gridspec as gridspec
@pyimport matplotlib.ticker as ticker     

line_alpha_amp = 0.75
line_alpha_amp_SEM = 0.5
line_alpha_gain = 0.75
line_alpha_gain_SEM =0.5

t_lim = (0, 1000) ./ 1000
letter_size = 10
title_fsize = 10
title_fsize_small = 9
x_letter = -0.05
y_letter = 1.125
x_letter_sum = -0.025
y_letter_sum  = 1.075
ex_diff = 500. ./ 1000
ex_xlim = (0, 1000 ) ./ 1000
ex_ylim = (-85, -22.5)
gain_xlim = (0.1, 150.)
gain_ylim = (-7.5, 0)
Amp_xlim_1 = (-100, -0)
Amp_xlim_end = (-100, -0)
sem_alpha = 0.25
col_WT = "black"
col_passive = "tab:grey"
col_no_Kd = "tab:blue"
col_no_A = "tab:purple"
col_no_SK = "tab:green"
col_no_T = "tab:orange"
col_no_HVA = "tab:cyan"
col_no_h = "tab:red"


color_array_Wt = reverse([1., 0.6, 0.3])
color_array = reverse([1.25, 1., 0.75])

data_col = "grey"
sim_col = "black"
t_lim_chirp = (0, 10)
db_lim = (-1.5, 0.25)
F_lim = (0.1, 100)

line_alpha= 0.85

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8 

letters = collect('A':'Z')


# Fig Setup
fig = plt.figure(figsize=(6.9, 4.75))
gs_all  = fig.add_gridspec(1, 2, left=0.075,right=0.95, top=0.95, wspace=0.4, hspace=0.4, width_ratios=[0.4, 0.6])

# OU setupt
gs_all_OU  = gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.85, hspace=0.35,  subplot_spec=py"$(gs_all)[0]", height_ratios=[0.6, 0.4]) #width_ratios=[0.3, 0.5], 
gs_0_OU  =  gridspec.GridSpecFromSubplotSpec(3, 2, wspace=0.5, hspace=0.75,  subplot_spec=py"$(gs_all_OU)[0]") 
gs_1_OU  =  gridspec.GridSpecFromSubplotSpec(1, 1, wspace=0.5, hspace=0.65, subplot_spec=py"$(gs_all_OU)[1]")  # height_ratios=[1,1, 0.5]

# OU 
ax_OU_ex0 = fig.add_subplot(py"$(gs_0_OU)[0, 0]")
ax_OU_ex0 = pyplot_fxns.remove_axis_box(ax_OU_ex0; s=["top", "right", "bottom"])
ax_OU_ex0.set_ylabel("Soma (mV)")
ax_OU_ex1 = fig.add_subplot(py"$(gs_0_OU)[0, 1]")
ax_OU_ex1 = pyplot_fxns.remove_axis_box(ax_OU_ex1; s=["top", "left", "right", "bottom"])

ax_OU_ex2 = fig.add_subplot(py"$(gs_0_OU)[1, 0]")
ax_OU_ex2 = pyplot_fxns.remove_axis_box(ax_OU_ex2; s=["top", "right", "bottom"])
ax_OU_ex2.set_ylabel("Soma (mV)")
ax_OU_ex3 = fig.add_subplot(py"$(gs_0_OU)[1, 1]")
ax_OU_ex3 = pyplot_fxns.remove_axis_box(ax_OU_ex3; s=["top", "left", "right", "bottom"])
# ax_OU_ex3.set_ylabel("Soma (mV)")

ax_OU_ex4 = fig.add_subplot(py"$(gs_0_OU)[2, 0]")
ax_OU_ex4 = pyplot_fxns.remove_axis_box(ax_OU_ex4; s=["top", "right"])
ax_OU_ex4.set_xlabel("Time (s)")
ax_OU_ex4.set_ylabel("Soma (mV)")

ax_OU_ex5 = fig.add_subplot(py"$(gs_0_OU)[2, 1]")
ax_OU_ex5 = pyplot_fxns.remove_axis_box(ax_OU_ex5; s=["top", "left", "right"])
ax_OU_ex5.set_xlabel("Time (s)")


ax_OU_gain = fig.add_subplot(py"$(gs_1_OU)[0]")
ax_OU_gain = pyplot_fxns.remove_axis_box(ax_OU_gain; s=["top", "right"])
ax_OU_gain.set_xlabel("Frequency (Hz)")
ax_OU_gain.set_ylabel("Gain (dB)")
ax_OU_gain.set_xscale("log")



# CHIRP setups
gs_all_Chirp  =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.35, hspace=0.4, height_ratios=[0.35, 0.65],  subplot_spec=py"$(gs_all)[1]")
gs_0_Chirp  =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.5, hspace=0.1, height_ratios=[0.5, 0.5],  subplot_spec=py"$(gs_all_Chirp)[0]") 
gs_1_Chirp  =  gridspec.GridSpecFromSubplotSpec(2, 3, wspace=0.55, hspace=0.8, subplot_spec=py"$(gs_all_Chirp)[1]") 

ax_chirp_c = fig.add_subplot(py"$(gs_0_Chirp)[0]")
ax_chirp_c = pyplot_fxns.remove_axis_box(ax_chirp_c; s=["top", "right", "bottom"])
ax_chirp_c.set_ylabel("Soma\n(mV)")
ax_chirp_c.set_xlabel("Time (s)")
ax_chirp_ex = fig.add_subplot(py"$(gs_0_Chirp)[1]")
ax_chirp_ex = pyplot_fxns.remove_axis_box(ax_chirp_ex; s=["top", "right"])
ax_chirp_ex.set_ylabel("Dendrite\n(mV)")

ax_WT = fig.add_subplot(py"$(gs_1_Chirp)[0,0]")
ax_Kd = fig.add_subplot(py"$(gs_1_Chirp)[0,1]")
ax_A = fig.add_subplot(py"$(gs_1_Chirp)[0,2]")
ax_T = fig.add_subplot(py"$(gs_1_Chirp)[1,0]")
ax_HVA = fig.add_subplot(py"$(gs_1_Chirp)[1,1]")
ax_SK = fig.add_subplot(py"$(gs_1_Chirp)[1,2]")



ax_WT = pyplot_fxns.remove_axis_box(ax_WT; s=["top", "right"])
ax_Kd = pyplot_fxns.remove_axis_box(ax_Kd; s=["top", "right"])
ax_A = pyplot_fxns.remove_axis_box(ax_A; s=["top", "right"])
ax_T = pyplot_fxns.remove_axis_box(ax_T; s=["top", "right"])
ax_HVA = pyplot_fxns.remove_axis_box(ax_HVA; s=["top", "right"])
ax_SK = pyplot_fxns.remove_axis_box(ax_SK; s=["top", "right"])





# PLOT ORNSTEIN UHLENBECK ##################################################
# Plot example time series data 
ax_OU_ex0.plot(soma_t[1,:] .- ex_diff, soma_WT[1,:], color=col_WT)
ax_OU_ex0.set_title("WT", color=col_WT, size=title_fsize_small)

ax_OU_ex1.plot(soma_t[1,:] .- ex_diff, soma_no_Kd[1,:], color=col_no_Kd)
ax_OU_ex1.set_title(L"$-$ Kdr", color=col_no_Kd, size=title_fsize_small)

ax_OU_ex2.plot(soma_t[1,:] .- ex_diff, soma_no_Ad[1,:], color=col_no_A)
ax_OU_ex2.set_title(L"$-$ A", color=col_no_A, size=title_fsize_small)

ax_OU_ex3.plot(soma_t[1,:] .- ex_diff, soma_no_Td[1,:], color=col_no_T)
ax_OU_ex3.set_title(L"$-$ T", color=col_no_T, size=title_fsize_small)

ax_OU_ex4.plot(soma_t[1,:] .- ex_diff, soma_no_HVAd[1,:], color=col_no_HVA)
ax_OU_ex4.set_title(L"$-$ HVA", color=col_no_HVA, size=title_fsize_small)

ax_OU_ex5.plot(soma_t[1,:] .- ex_diff, soma_no_SKd[1,:], color=col_no_SK)
ax_OU_ex5.set_title(L"$-$ K(Ca)", color=col_no_SK, size=title_fsize_small)


# plot gain
ax_OU_gain.plot(power_F_WT, power_WT[1:end], color=col_WT, zorder=1, alpha=line_alpha_gain)
ax_OU_gain.fill_between(power_F_WT,power_WT[1:end] .- (power_SEM_WT[1:end]),
                        power_WT[1:end] .+ (power_SEM_WT[1:end] ),alpha=line_alpha_gain_SEM, color=col_WT, zorder=1)


ax_OU_gain.plot(power_F_no_Kd, power_no_Kd[1:end], color=col_no_Kd, alpha=line_alpha_gain)
ax_OU_gain.fill_between(power_F_no_Kd,power_no_Kd[1:end] .- (power_SEM_no_Kd[1:end]),
                        power_no_Kd[1:end] .+ (power_SEM_no_Kd[1:end] ),alpha=line_alpha_gain_SEM, color=col_no_Kd)

ax_OU_gain.plot(power_F_no_Ad, power_no_Ad[1:end], color=col_no_A, alpha=line_alpha_gain)
ax_OU_gain.fill_between(power_F_no_Ad,power_no_Ad[1:end] .- (power_SEM_no_Ad[1:end]),
                        power_no_Ad[1:end] .+ (power_SEM_no_Ad[1:end] ),alpha=line_alpha_gain_SEM, color=col_no_A)

ax_OU_gain.plot(power_F_no_SKd, power_no_SKd[1:end], color=col_no_SK, alpha=line_alpha_gain)
ax_OU_gain.fill_between(power_F_no_SKd,power_no_SKd[1:end] .- (power_SEM_no_SKd[1:end]),
                        power_no_SKd[1:end] .+ (power_SEM_no_SKd[1:end] ),alpha=line_alpha_gain_SEM, color=col_no_SK)

ax_OU_gain.plot(power_F_no_HVAd, power_no_HVAd[1:end], color=col_no_HVA, alpha=line_alpha_gain)
ax_OU_gain.fill_between(power_F_no_HVAd,power_no_HVAd[1:end] .- (power_SEM_no_HVAd[1:end]),
                        power_no_HVAd[1:end] .+ (power_SEM_no_HVAd[1:end] ),alpha=line_alpha_gain_SEM, color=col_no_HVA)

ax_OU_gain.plot(power_F_no_Td, power_no_Td[1:end], color=col_no_T, alpha=line_alpha_gain)
ax_OU_gain.fill_between(power_F_no_Td,power_no_Td[1:end] .- (power_SEM_no_Td[1:end]),
                        power_no_Td[1:end] .+ (power_SEM_no_Td[1:end] ),alpha=line_alpha_gain_SEM, color=col_no_T)


ax_OU_gain.set_xlim(gain_xlim)
ax_OU_gain.set_ylim(gain_ylim)

for ax in [ax_OU_ex0, ax_OU_ex1, ax_OU_ex2, ax_OU_ex3, ax_OU_ex4, ax_OU_ex5] 
    ax.set_xlim(ex_xlim)
    ax.set_ylim(ex_ylim)
end 


# PLOT Chirp


for ax in [ax_Kd,ax_HVA,ax_A,ax_SK,ax_T]
    ax.set_xscale("log")
    for i in [1,2,3]#,4]
        Iscale = Iscale_array[i]
        param = Dict("phi0" => phi0, "f0" => f0, "f1" => f1, "T" =>T, "Iscale" => Iscale, "Vhold" => V0)
        ax = plot_F(ax, "WT", param, pyplot_fxns.lighten_color(col_WT, color_array_Wt[i]), 0.75)
    end
end


# Chirp input time series
ax_chirp_c.plot(sol_WT.t ./1000,chirp_ex, color="black")
ax_chirp_c.set_xlim(t_lim_chirp)

# Chirp time series
ax_chirp_ex =plot_timeseries(ax_chirp_ex, "WT", param, col_WT)
ax_chirp_ex =plot_timeseries_end(ax_chirp_ex, "WT", param, "tab:grey")
ax_chirp_ex.set_xlim(t_lim_chirp)


for i in [1,2,3]#,4]
    Iscale = Iscale_array[i]
    ytext = -0.65 - (0.25*(i-1))
    Ftext = 0.15 
    param = Dict("phi0" => phi0, "f0" => f0, "f1" => f1, "T" =>T, "Iscale" => Iscale, "Vhold" => V0)
    
    # WT
    ax_WT = plot_F(ax_WT, "WT", param, pyplot_fxns.lighten_color(col_WT, color_array_Wt[i]), line_alpha; p=true, Ftext=Ftext, ytext = ytext)
    ax_WT.set_xscale("log")
    
    # Kd
    ax_Kd = plot_F(ax_Kd, "no_Kd", param, pyplot_fxns.lighten_color(col_no_Kd, color_array[i]), line_alpha; p=true, Ftext=Ftext, ytext = ytext)

    # HVA
    ax_HVA = plot_F(ax_HVA, "no_HVAd", param, pyplot_fxns.lighten_color(col_no_HVA, color_array[i]), line_alpha; p=true, Ftext=Ftext, ytext = ytext)

    # Ad
    ax_A = plot_F(ax_A, "no_Ad", param, pyplot_fxns.lighten_color(col_no_A, color_array[i]), line_alpha; p=true, Ftext=Ftext, ytext = ytext)

    # SK
    ax_SK = plot_F(ax_SK, "no_SKd", param, pyplot_fxns.lighten_color(col_no_SK, color_array[i]), line_alpha; p=true, Ftext=Ftext, ytext = ytext)

    # T
    ax_T = plot_F(ax_T, "no_Td", param, pyplot_fxns.lighten_color(col_no_T, color_array[i]), line_alpha; p=true, Ftext=Ftext, ytext = ytext)   
end


title_y = 0.95
ax_WT.set_title("WT", color=col_WT, size=title_fsize, y=title_y)
ax_Kd.set_title(L"$-$ Kdr", color=col_no_Kd, size=title_fsize, y=title_y)
ax_HVA.set_title(L"$-$ HVA", color=col_no_HVA, size=title_fsize, y=title_y)
ax_A.set_title(L"$-$ A", color=col_no_A, size=title_fsize, y=title_y)
ax_SK.set_title(L"$-$ K(Ca)", color=col_no_SK, size=title_fsize, y=title_y)
ax_T.set_title(L"$-$ T", color=col_no_T, size=title_fsize, y=title_y)


ax_WT.set_ylabel("Gain (dB)")
ax_T.set_ylabel("Gain (dB)")

for ax in [ax_WT, ax_Kd,ax_HVA,ax_A,ax_SK,ax_T]
    ax = pyplot_fxns.remove_axis_box(ax; s=["top", "right"])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(F_lim)
    ax.set_ylim(db_lim)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
end


letters = collect('A':'Z')
ax_OU_ex0.text(x_letter, y_letter, "$(letters[1])", transform=ax_OU_ex0.transAxes, size=letter_size, weight="bold")
ax_OU_ex1.text(x_letter, y_letter, "$(letters[2])", transform=ax_OU_ex1.transAxes, size=letter_size, weight="bold")
ax_OU_ex2.text(x_letter, y_letter, "$(letters[3])", transform=ax_OU_ex2.transAxes, size=letter_size, weight="bold")
ax_OU_ex3.text(x_letter, y_letter, "$(letters[4])", transform=ax_OU_ex3.transAxes, size=letter_size, weight="bold")
ax_OU_ex4.text(x_letter, y_letter, "$(letters[5])", transform=ax_OU_ex4.transAxes, size=letter_size, weight="bold")
ax_OU_ex5.text(x_letter, y_letter, "$(letters[6])", transform=ax_OU_ex5.transAxes, size=letter_size, weight="bold")
ax_OU_gain.text(x_letter, y_letter_sum, "$(letters[7])", transform=ax_OU_gain.transAxes, size=letter_size, weight="bold")
ax_chirp_c.text(x_letter, y_letter_sum, "$(letters[8])", transform=ax_chirp_c.transAxes, size=letter_size, weight="bold")
ax_WT.text(x_letter, y_letter_sum, "$(letters[9])", transform=ax_WT.transAxes, size=letter_size, weight="bold")
ax_Kd.text(x_letter, y_letter_sum, "$(letters[10])", transform=ax_Kd.transAxes, size=letter_size, weight="bold")
ax_A.text(x_letter, y_letter_sum, "$(letters[11])", transform=ax_A.transAxes, size=letter_size, weight="bold")
ax_T.text(x_letter, y_letter_sum, "$(letters[12])", transform=ax_T.transAxes, size=letter_size, weight="bold")
ax_HVA.text(x_letter, y_letter_sum, "$(letters[13])", transform=ax_HVA.transAxes, size=letter_size, weight="bold")
ax_SK.text(x_letter, y_letter_sum, "$(letters[14])", transform=ax_SK.transAxes, size=letter_size, weight="bold")





plt.savefig(plotsdir("Figure_2_Cable_filtering_OU_Chirp.png"), dpi=600)
plt.savefig(plotsdir("Figure_2_Cable_filtering_OU_Chirp.pdf"), dpi=600)
plt.savefig(plotsdir("Figure_2_Cable_filtering_OU_Chirp.eps"), dpi=600)
plt.show()
#%%