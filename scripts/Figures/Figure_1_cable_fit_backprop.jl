
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
include(srcdir("pyplot_fxns.jl"))
using .pyplot_fxns
@pyimport matplotlib.gridspec as gridspec
@pyimport matplotlib.patches as patches 

#read in data ######################################################

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
    global j = j+1
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
numseg = 50
p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_cable.jld2"), "p_fit")

# Initial condition
V0  = -60
u0_cable =  CSC_cable_Ih.steady_state_init_u0_gating(V0, numseg, p_fit[24:end]);

tspan_fI = (0., 1500)
twindow = 1000.

# convert pA to uA/cm^2
SA = 4* π * (p_fit[16]/10000)^2
Iarray = I_data .*1e-6 ./SA

# set up timing
step_start = 500
step_length = 1000
post_step = 100
sim_start = 0
step_end = step_length + step_start
sim_end =  step_end + post_step
samp_rate = 0.1
tspan = (0.0, sim_end)
tsteps = step_start:samp_rate:step_end
I_mag = 0.


cbs = CSC_cable_Ih.step_I(step_start, step_end, 0.)
prob_fit = ODEProblem(CSC_cable_Ih.CSCcable_gating!, u0_cable, (0.0, 17000), p_fit, 
                        maxiters=1e25)
sol = solve(prob_fit, ROCK2(), saveat = step_start:samp_rate:step_end+100)


prob_fit_FI = ODEProblem(CSC_cable_Ih.CSCcable_gating!, u0_cable, tspan, p_fit, save_start=false, save_end=false,
                            save_on=false,
                        callback=cbs, 
                        maxiters=1e25)
F, ISI, Finst, peakamp = analysis.step_FI_analysis(sol, step_start, step_end; ind=1)
dV = stack(sol(sol.t, Val{1}).u)[1,1,:]
sim, F_array = CSC_cable_Ih.fIcurve(prob_fit_FI, Iarray, p_fit, step_start, step_end)

#%% FIGURE ######################################################

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8 

dist_incr = Int(p_fit[19]*10000/50)
postion = collect(0:1:50)
levels = 100
vmin = -70 
vmax = 10 

data_col = "tab:red"
sim_col = "black"
letters = collect('A':'Z')
letter_size = 10
x_letter = -0.05
y_letter = 1.1
t_lim = (-100, 1100)
zoom_xlim = (9.5, 17.5)

fig = plt.figure(figsize=(4.56, 4.56))

gs_all  = fig.add_gridspec(2, 1, left=0.125, top=0.95, right=0.95, wspace=0.35, hspace=0.4, height_ratios=[0.55, 0.45])
gs_0  =  gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.35, hspace=0.5,  subplot_spec=py"$(gs_all)[0]", width_ratios=[0.6, 0.4]) 
gs_0_left = gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.1, hspace=0.15,  subplot_spec=py"$(gs_0)[0]", ) 


gs_1  =  gridspec.GridSpecFromSubplotSpec(1, 5, wspace=0.4, hspace=0.25, subplot_spec=py"$(gs_all)[1]", width_ratios=[0.45, 0.05, 0.45, 0.01, 0.04])
gs_1_right = gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.5, hspace=0.75,  subplot_spec=py"$(gs_1)[2]") 

ax_scheme = fig.add_subplot(py"$(gs_0_left)[0]")
ax_scheme = pyplot_fxns.remove_axis_box(ax_scheme; s=["top", "right", "bottom", "left"])


ax_AP_phase = fig.add_subplot(py"$(gs_0)[1]")
ax_AP_phase = pyplot_fxns.remove_axis_box(ax_AP_phase; s=["top", "right"])

ax_FI = fig.add_subplot(py"$(gs_1)[0]")
ax_FI = pyplot_fxns.remove_axis_box(ax_FI; s=["top", "right"])

ax_spiketrain = fig.add_subplot(py"$(gs_0_left)[1]")
ax_spiketrain = pyplot_fxns.remove_axis_box(ax_spiketrain; s=["top", "right"])

ax_backprop = fig.add_subplot(py"$(gs_1_right)[0]")
ax_backprop.set_ylabel("V (mV)")
ax_backprop.set_xlabel("Time (ms)")
ax_backprop = pyplot_fxns.remove_axis_box(ax_backprop; s=["top", "right"])

ax_backprop_zoom = fig.add_subplot(py"$(gs_1_right)[1]")
ax_backprop_zoom.set_ylabel("V (mV)")
ax_backprop_zoom.set_xlabel("Time (ms)")
ax_backprop_zoom = pyplot_fxns.remove_axis_box(ax_backprop_zoom; s=["top", "right"])

ax_backprop_cb = fig.add_subplot(py"$(gs_1)[3]")
ax_backprop_cb = pyplot_fxns.remove_axis_box(ax_backprop_cb; s=["top", "right"])


# Schematic #########################################################################################
circle1 = patches.Ellipse((0.175, 0.5), 0.35, 0.65, angle=0, edgecolor="k", facecolor="white", lw=2)
ax_scheme.add_patch(circle1)
ax_scheme.text(0.175, 0.5, "Soma", ha="center", va="center")
dist_incr1 = Int(p_fit[19]*10000/50)
num_fsize = 8

rect1 = patches.Rectangle((0.2, 0.4), 0.8, 0.2, angle=0, edgecolor="k", facecolor="white", zorder=-1, lw=2)
ax_scheme.add_patch(rect1)

ax_scheme.vlines(0.45, 0.4, 0.6,linestyle=":",  color="k", lw=2)
ax_scheme.text(0.4, 0.5, "1", ha="center", va="center", size=num_fsize)

ax_scheme.vlines(0.55, 0.4, 0.6,linestyle=":",  color="k", lw=2)
ax_scheme.text(0.5, 0.5, "2", ha="center", va="center", size=num_fsize)

ax_scheme.vlines(0.65, 0.4, 0.6,linestyle=":",  color="k", lw=2)
ax_scheme.text(0.6, 0.5, "3", ha="center", va="center", size=num_fsize)

ax_scheme.text(0.725, 0.52, "...", ha="center", va="center", fontsize=14)

ax_scheme.vlines(0.8, 0.4, 0.6,linestyle=":", color="k", lw=2)
ax_scheme.text(0.85, 0.5, "49", ha="center", va="center", size=num_fsize)

ax_scheme.vlines(0.9, 0.4, 0.6,linestyle=":", color="k", lw=2)
ax_scheme.text(0.95, 0.5, "50", ha="center", va="center", size=num_fsize)

# ax_scheme.hlines(0.3, 0.3475, 1.0025, color="k", lw=2)
ax_scheme.hlines(0.315, 0.3475, 1.0025, color="k", lw=2)
ax_scheme.vlines(0.35, 0.3, 0.35, color="k", lw=2)
ax_scheme.vlines(1, 0.3, 0.35, color="k", lw=2)
ax_scheme.text(0.7, 0.225, "$(dist_incr1*50) μm", ha="center", va="center")


ax_scheme.text(0.7, 0.775, "Dendritic\nCompartments", ha="center", va="center")


bpad = 0.1
fig.subplots_adjust(left=bpad, bottom=bpad, right=1-bpad, top=1-bpad/2)
start = 0.66
for i=1:3
    ax_scheme.vlines(start + i *0.035, 0.35, 0.65, color="w", lw=3)
end

ax_scheme.set_ylim(0,1)
ax_scheme.set_xlim(-0.05,1.05)
#####################################################################################################



# plotting
ax_AP_phase.plot(V0_data, dV0_data ./ 1000, color=data_col)
ax_AP_phase.plot(sol[1,1,:], dV[:], color=sim_col)
ax_AP_phase.set_xlabel("Somatic V (mV)")
ax_AP_phase.set_ylabel("Somatic dV/dt (mV/ms)")

ax_FI.plot(I_data, vec(F_plot_all_mean), color=data_col)
ax_FI.fill_between(I_data, vec(F_plot_all_mean) .+ vec(F_plot_all_std), vec(F_plot_all_mean) .- vec(F_plot_all_std), color=data_col, alpha=0.5)
ax_FI.plot(I_data, F_array, color=sim_col)
ax_FI.set_ylabel("Frequency (Hz)")
ax_FI.set_xlabel("Current (pA)")

ax_spiketrain.plot(t0_data_plot, V0_data_plot, color=data_col, label="Data")
ax_spiketrain.plot(sol.t .- sol.t[1] .- 6, sol[1,:], color=sim_col, label="Model")
ax_spiketrain.set_ylabel("Somatic V (mV)")
ax_spiketrain.set_xlabel("Time (ms)")
ax_spiketrain.set_xlim(t0_data_plot[1], t0_data_plot[end])

ax_spiketrain.legend(loc=9, bbox_to_anchor=(0.5, 1.225), ncol=2,  frameon=false)


cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "silver"])
norm = matplotlib.colors.Normalize(vmin=minimum(1), vmax=maximum(51))

ax_backprop.plot(sol.t .-sol.t[1] .- 20, sol[1,1,:], label="Soma", color=cmap1(norm(1)))
for i in [10, 20, 30, 40, 50]
    ax_backprop.plot(sol.t .-sol.t[1] .- 20, sol[1,i,:], label="$(dist_incr*i) μm", color=cmap1(norm(i)))
end
ax_backprop.set_xlim(0, 75)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap1)
cb1 = fig.colorbar(sm, cax = ax_backprop_cb,  ticks=1:10:51, pad=0.1, shrink=1.0,)

cb1.ax.set_yticks(1:10:51)
cb1.ax.set_yticklabels(["Soma", ["$(dist_incr*(i-1))" for i in 11:10:51]...])
cb1.set_label("Position (μm)", labelpad=-17.5, y=1.125, rotation=0)

# add rectangle for zoom in 
rect1 = patches.Rectangle((zoom_xlim[1], ax_backprop.get_ylim()[1]+1), zoom_xlim[2] - zoom_xlim[1], ax_backprop.get_ylim()[2] - ax_backprop.get_ylim()[1]-0.2,  
            linewidth=1,
            clip_on=false, edgecolor="k", facecolor="none",linestyle="--")
ax_backprop.add_patch(rect1)

ax_backprop_zoom.plot(sol.t .-sol.t[1] .- 20, sol[1,1,:], label="Soma", color=cmap1(norm(1)))
for i in [10, 20, 30, 40, 50]
    ax_backprop_zoom.plot(sol.t .-sol.t[1] .- 20, sol[1,i,:], label="$(dist_incr*i) μm", color=cmap1(norm(i)))
end
ax_backprop_zoom.set_xlim(zoom_xlim)

ax_scheme.text(x_letter, y_letter-0.1, "$(letters[1])", transform=ax_scheme.transAxes, size=letter_size, weight="bold")
ax_spiketrain.text(x_letter, y_letter-0., "$(letters[2])", transform=ax_spiketrain.transAxes, size=letter_size, weight="bold")
ax_AP_phase.text(x_letter, y_letter-0.05, "$(letters[3])", transform=ax_AP_phase.transAxes, size=letter_size, weight="bold")
ax_FI.text(x_letter, y_letter-0.05, "$(letters[4])", transform=ax_FI.transAxes, size=letter_size, weight="bold")
ax_backprop.text(x_letter, y_letter, "$(letters[5])", transform=ax_backprop.transAxes, size=letter_size, weight="bold")
ax_backprop_zoom.text(x_letter, y_letter, "$(letters[6])", transform=ax_backprop_zoom.transAxes, size=letter_size, weight="bold")



plt.savefig(plotsdir("Figure_1_cable_fit_backprop.png"), dpi=600)
plt.savefig(plotsdir("Figure_1_cable_fit_backprop.eps"), dpi=600)
plt.savefig(plotsdir("Figure_1_cable_fit_backprop.pdf"), dpi=600)
plt.show()