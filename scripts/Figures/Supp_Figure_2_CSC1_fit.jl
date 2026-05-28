
using DrWatson
@quickactivate  "MLI_synch_2026"

using DifferentialEquations, DiffEqCallbacks, Statistics, Peaks
using FHist
using PyCall, PyPlot
using LaTeXStrings
include(srcdir("CSC1_Ih.jl"))
using .CSC1_Ih
include(srcdir("analysis.jl"))
using .analysis
include(srcdir("pyplot_fxns.jl"))
using .pyplot_fxns
@pyimport matplotlib.gridspec as gridspec
@pyimport matplotlib.patches as patches 

#%%read in data ######################################################

fpath = datadir("exp_pro", "HBP_firing_analysis_extract.json")

py"""
import numpy as np
import pandas as pd
df = pd.read_json($fpath)
"""
i = 2 # cell 3
I_data = py"df.loc[$(i-1), 'I']"
F_data = py"df.loc[$(i-1), 'F']"

F_all_ind = [1,2,5,7,8]
F_data_all = py"df.loc[$(F_all_ind .- 1), 'F']"

F_plot_all = zeros(length(F_all_ind), length(I_data))
j = 1
for i in F_all_ind
    F_plot_all[j, :] = F_data_all[i]
    j = j+1
end

F_plot_all_mean = mean(F_plot_all, dims= 1)
F_plot_all_std = std(F_plot_all, dims= 1)

t0_data = py"df.loc[$(i-1), 't0']" * 1000
V0_data = py"df.loc[$(i-1), 'V0']"

t0_data_plot = t0_data[1000 .<= t0_data .<= 2000] .- 1000
V0_data_plot = V0_data[1000 .<= t0_data .<= 2000]

dV0_data = py"df.loc[$(i-1), 'dV0']"
F_data_0 = F_data[findfirst(I_data .== 0.0)]
peakamp_data = py"df.loc[$(i-1), 'spike_amp']"
peakamp_data_0 = mean(peakamp_data[findfirst(I_data .== 0.0)])


#%% # Simulation ######################################################
# parameters 
# initialize simulation parameters
V0 = -60;
tspan = (0.0, 2100.0)
tspan_ic = (0.0, 2000.0);
t_offset = 100.0;
step_start = 0.0 + t_offset;
step_dt = 3.0;
step_end = step_start + step_dt;
global I_mag = 0; 
thresh = 0.1;

# initialize model parameters
I = 0.0
gNa = 3. 
gK = 17.5
gleak = 0.0125
gA = 7.5 
gT = 0.45045
gHVA = 0.28 
gSK = 0.3
gh = 20
Eleak = -60

Eh   = -34.4 # (mV)
p_names = ["gNa", "gK", "gleak", "gA", "gT", "gHVA", "gSK", "gh", "Eleak", "Eh", "I"];
p_ind = findfirst(p_names .== "I")

# initialize WT parameters and find Ihold
p_WT = [gNa, gK, gleak, gA, gT, gHVA, gSK, gh, Eleak,Eh, I];

# Initial condition
V0  = -60
tspan_ic = (0, 1500.)
u0_init = CSC1_Ih.init_u0(;V0=-60)
ic =  CSC1_Ih.steady_state_init(V0, p_WT, tspan_ic);



#%%
tspan_fI = (0., 1500.)
twindow = 1000.
filename =""


# set up timing
step_start = 500.
step_length = 1000.
post_step = 100.
sim_start = 0.
step_end = step_length + step_start
sim_end =  step_end + post_step
samp_rate = 0.1
tspan = (0.0, sim_end)
tsteps = step_start:samp_rate:step_end

I_mag = 0.


cbs = CSC1_Ih.step_I(step_start, step_end, 0.)
prob_fit = ODEProblem(CSC1_Ih.CSC1_Ih_I!, ic, tspan, p_WT, save_idxs=[1],saveat=tsteps, 
                                        dtmax=samp_rate, maxiters=1e50)

p0 = [gNa, gK, gleak, gA, gT, gHVA, gSK, gh]
p_result = [p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7], p0[8],  Eleak, Eh, 0.]

ic =  CSC1_Ih.steady_state_init(V0, p_result, tspan_ic);
sol = solve(remake(prob_fit, p =p_result, u0=ic ), ROCK2());


ind_result = sol.t .>=step_start

SA = 4* π * (5/10000)^2 
Iarray = I_data .*1e-6 ./SA # convert pA to uA/cm^2


prob_fit_FI = ODEProblem(CSC1_Ih.CSC1_Ih_I!, ic, tspan, p_WT, callback=cbs, save_idxs=[1],
                                        dtmax=samp_rate,  dt=1e-10,saveat=tsteps, 
                                        maxiters=1e50)

sim, F_array = CSC1_Ih.fIcurve(prob_fit_FI, Iarray, ic,  p_result, step_start, step_end,);



#%% FIGURE ######################################################

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8 

levels = 100
vmin = -70 
vmax = 10 

data_col = "tab:red"
sim_col = "black"
letters = collect('A':'Z')
letter_size = 10
x_letter = -0.05
y_letter = 1.05
t_lim = (0, 1000)


fig = plt.figure(figsize=(6.9, 3))

gs_all  = fig.add_gridspec(1, 3, left=0.1, top=0.9, right=0.95, bottom=0.2, wspace=0.45, hspace=0.4)
ax_AP_phase = fig.add_subplot(py"$(gs_all)[1]")
ax_AP_phase = pyplot_fxns.remove_axis_box(ax_AP_phase; s=["top", "right"])

ax_FI = fig.add_subplot(py"$(gs_all)[2]")
ax_FI = pyplot_fxns.remove_axis_box(ax_FI; s=["top", "right"])

ax_spiketrain = fig.add_subplot(py"$(gs_all)[0]")
ax_spiketrain = pyplot_fxns.remove_axis_box(ax_spiketrain; s=["top", "right"])



# plotting
ax_AP_phase.plot(V0_data, dV0_data ./ 1000, color=data_col)
ax_AP_phase.plot(sol[1,ind_result], reduce(hcat, sol(sol.t[ind_result], Val{1}).u)', color=sim_col)
ax_AP_phase.set_xlabel("Somatic V (mV)")
ax_AP_phase.set_ylabel("Somatic dV/dt (mV/ms)")

ax_FI.plot(I_data, vec(F_plot_all_mean), color=data_col)
ax_FI.fill_between(I_data, vec(F_plot_all_mean) .+ vec(F_plot_all_std), vec(F_plot_all_mean) .- vec(F_plot_all_std), color=data_col, alpha=0.5)
ax_FI.plot(I_data, F_array, color=sim_col)
ax_FI.set_ylabel("Frequency (Hz)")
ax_FI.set_xlabel("Current (pA)")


ax_spiketrain.plot(t0_data_plot, V0_data_plot, color=data_col, label="Data")
ax_spiketrain.plot(sol.t .- sol.t[1], sol[1,:], color=sim_col, label="Model")
ax_spiketrain.set_ylabel("Somatic V (mV)")
ax_spiketrain.set_xlabel("Time (ms)")
ax_spiketrain.set_xlim(t0_data_plot[1], t0_data_plot[end])

ax_spiketrain.legend(loc=9, bbox_to_anchor=(1.95, -0.15), ncol=2,  frameon=false)



ax_spiketrain.text(x_letter, y_letter, "$(letters[1])", transform=ax_spiketrain.transAxes, size=letter_size, weight="bold")
ax_AP_phase.text(x_letter, y_letter, "$(letters[2])", transform=ax_AP_phase.transAxes, size=letter_size, weight="bold")
ax_FI.text(x_letter, y_letter, "$(letters[3])", transform=ax_FI.transAxes, size=letter_size, weight="bold")


plt.savefig(plotsdir("Supp_Figure_2_CSC1_fit_gating.png"), dpi=600)
plt.savefig(plotsdir("Supp_Figure_2_CSC1_fit_gating.pdf"), dpi=600)
plt.savefig(plotsdir("Supp_Figure_2_CSC1_fit_gating.eps"), dpi=600)
plt.show()

#