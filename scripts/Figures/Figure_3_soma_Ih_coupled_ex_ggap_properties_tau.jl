
using DrWatson
@quickactivate  "MLI_synch_2026"
using DifferentialEquations, DiffEqCallbacks,
    Optimization,  SciMLSensitivity,
    Zygote, DiffEqCallbacks, JLD2, Statistics, Peaks
using FHist
using PyCall
using PyPlot
include(srcdir("pyplot_fxns.jl"))
using .pyplot_fxns

#%% read in data

# example
init_phase_diff = 0.5
gapseg = (11,11)
gc = 2.

tspan_sol = (0.0, 2.01*1000.0)
hilb_spike_num = 5
param_ex = Dict(
    "phase_diff" => init_phase_diff, 
    "sim_length" => tspan_sol[2],
    "hilb_spike_num" => hilb_spike_num,
    "gc" => gc,
    "gapseg1" =>  gapseg[1],
    "gapseg2" =>  gapseg[2],
)

sol = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_example_CSC_soma_Ih", param_ex, "jld2")),  "sol")
phaseps = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_example_CSC_soma_Ih", param_ex, "jld2")),  "phaseps")
t1ps = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_example_CSC_soma_Ih", param_ex, "jld2")),  "t1ps")
t2ps = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_example_CSC_soma_Ih", param_ex, "jld2")),  "t2ps")
ISI1ps = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_example_CSC_soma_Ih", param_ex, "jld2")),  "ISI1ps")
PLV = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_example_CSC_soma_Ih", param_ex, "jld2")),  "PLV")
Hilbert_phase_diff = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_example_CSC_soma_Ih", param_ex, "jld2")),  "Hilbert_phase_diff")
mean_peak_phase_end = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_example_CSC_soma_Ih", param_ex, "jld2")),  "mean_peak_phase_end")


tspan = (0.0, 30*1000.0)
n_phase=12 
init_phase_diff_array = collect(0:1/n_phase:1 - 1/n_phase)
p_int = 5
position_array = collect(1:p_int:51) 
hilb_spike_num = 5
ggap_array = [
                0, 
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
                100.0
                ]
param = Dict(
    "n_phase" => n_phase, 
    "ggap_max" => maximum(ggap_array),
    # "ggap_n" => length(ggap_array), 
    "ggap_min" => minimum(ggap_array),
    "pos_int" => p_int,
    "hilb_spike_num" => hilb_spike_num,
    "tend" =>  tspan[2],
)

# init_phase, ggap, position
phase_diff_array = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_synch_2_properties_SUMMARY_CSC_soma_Ih_time_synch_WT", param, "jld2")),  "phase_diff_array")
spike_times = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_synch_2_properties_SUMMARY_CSC_soma_Ih_time_synch_WT", param, "jld2")),  "spike_times")
PLV_array = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_synch_2_properties_SUMMARY_CSC_soma_Ih_time_synch_WT", param, "jld2")),  "PLV_array")
Hilbert_phase_diff_array = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_synch_2_properties_SUMMARY_CSC_soma_Ih_time_synch_WT", param, "jld2")),  "Hilbert_phase_diff_array")
mean_peak_phase_end_array = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_synch_2_properties_SUMMARY_CSC_soma_Ih_time_synch_WT", param, "jld2")),  "mean_peak_phase_end_array")
tau_array = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_synch_2_properties_SUMMARY_CSC_soma_Ih_time_synch_WT", param, "jld2")),  "tau_array")

ggap_array = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_synch_2_properties_SUMMARY_CSC_soma_Ih_time_synch_WT", param, "jld2")),  "ggap_array")
init_phase_diff_array = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_synch_2_properties_SUMMARY_CSC_soma_Ih_time_synch_WT", param, "jld2")),  "init_phase_diff_array")
position_array = wload(datadir("simulations", "gapjxn_charac",  savename("ggap_synch_2_properties_SUMMARY_CSC_soma_Ih_time_synch_WT", param, "jld2")),  "position_array")


# p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_soma_Ih.jld2"), "p_fit")
p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_cable.jld2"), "p_fit")
dist_incr1 = Int(p_fit[19]*10000/50)


tau_array2 = deepcopy(tau_array)
# tau_array2[tau_array2 .> 5*30000.] .= 0.
# tau_array2[tau_array2 .> 30000.] .= NaN
# FIGURE ######################################################
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8 

using LaTeXStrings
@pyimport matplotlib.gridspec as gridspec
@pyimport matplotlib.patches as patches

sim_col_1 = "tab:brown"
sim_col_2 = "black"
letter_size = 10
x_letter = -0.075
y_letter = 1.075
x_letter_polar = -0.1
y_letter_polar = 1.3
using LaTeXStrings

PLV_cmap = "twilight_shifted"
# t_cmap = "viridis_r"
# t_cmap = "gnuplot_r"
# t_cmap = "inferno_r"
t_cmap = "magma_r"
# t_cmap = "managua"
# t_cmap = "ocean_r"
# t_cmap = "brg"

t_cmap_o = "ocean_r"
original_cmap = plt.cm.get_cmap(t_cmap_o)

t_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "trunc($(t_cmap_o), 0.05, 0.75)",
        original_cmap(range(0.05, stop=0.75, length=1000)))

# t_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
#         "trunc($(t_cmap_o), 0.25, 1.0)",
#         original_cmap(range(0.25, stop=1.0, length=1000)))

rad_labels = [L"0",L"\frac{1}{4} \pi",L"\frac{1}{2} \pi",L"\frac{3}{4} \pi",L"\pi",L"\frac{5}{4} \pi",
        L"\frac{3}{2} \pi",L"\frac{7}{4} \pi", ]
cb_lab_pad = -40
cb_lab_y = 1.1
pos_label = "Position"
pos_label2 = "Position (μm)"
pos_label_ticks = [0, 25.,50.]
pos_label_um = ["Soma", ["$(dist_incr1*(i))" for i =[25, 50]]...]
ggap_label = L"\mathrm{g}_{\mathrm{gap}}" * " (nS)"
phase_diff_label = "Initial ΔΦ"
z_min = -π
z_max = π
z_t_min = tspan[1] /1000
z_t_max = tspan[2] /1000 *2
cb_scale = 0.8

phase_col = "darkgreen"
ggap_col = "k"
pos_col = "purple"
init_col = "darkgrey"


#%

t_cmap_o = "ocean_r"
original_cmap = plt.cm.get_cmap(t_cmap_o)

t_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "trunc($(t_cmap_o), 0.05, 0.75)",
        original_cmap(range(0.05, stop=0.75, length=1000)))

# Fig Setup
fig = plt.figure(figsize=(6.9, 6.9))
# gs_all  = fig.add_gridspec(7, 2, bottom=0.1, top=0.95, right=0.9, left=0.1, wspace=0.25, hspace=0.25,  width_ratios=[0.76, 0.27], height_ratios=[0.7, 0.2, 1.2, 0.0, 0.75, 0.275, 0.75])
# gs_all  = fig.add_gridspec(7, 2, bottom=0.1, top=0.95, right=0.9, left=0.1, wspace=0.25, hspace=0.25,  width_ratios=[0.76, 0.27], height_ratios=[0.7, 0.2, 1.2, 0.0, 0.75, 0.0, 0.75])
gs_all  = fig.add_gridspec(5, 2, bottom=0.075, top=0.96, right=0.925, left=0.05, wspace=0.25, hspace=0.0,  width_ratios=[0.76, 0.27], height_ratios=[0.5, 0.025, 1.4, 0.0, 1.5])
gs_0_0  =  gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.155, hspace=0.25,  subplot_spec=py"$(gs_all)[0,0]", width_ratios=[0.0, 1.0]) 
gs_0_1  =  gridspec.GridSpecFromSubplotSpec(1, 1, wspace=0.5, hspace=0.25, subplot_spec=py"$(gs_all)[0,1]") 
gs_2 =  gridspec.GridSpecFromSubplotSpec(2, 7, wspace=0.45, hspace=0.2, width_ratios=[0.0, 0.3,0.075, 0.3,0.075, 0.3,0.01],  subplot_spec=py"$(gs_all)[4, :]") 
# gs_3 =  gridspec.GridSpecFromSubplotSpec(1, 6, wspace=0.45, hspace=0.5, width_ratios=[0.3,0.075, 0.3,0.075, 0.3,0.01],  subplot_spec=py"$(gs_all)[6, :]") 
gs_1 =  gridspec.GridSpecFromSubplotSpec(1, 3, wspace=0.5, hspace=1.0,  subplot_spec=py"$(gs_all)[2, :]") 


# example plots
ax_spikes = fig.add_subplot(py"$(gs_0_0)[1]")
ax_spikes = pyplot_fxns.remove_axis_box(ax_spikes; s=["top", "right"])#, "bottom"])
ax_phase_polar = fig.add_subplot(py"$(gs_0_1)[0]", polar=true)


# summary plots time
ax_phase_ggap_t = fig.add_subplot(py"$(gs_2)[1,1]")
ax_phase_ggap_t = pyplot_fxns.remove_axis_box(ax_phase_ggap_t; s=["top", "right"])
ax_phase_pos_t = fig.add_subplot(py"$(gs_2)[1,5]")
ax_phase_pos_t = pyplot_fxns.remove_axis_box(ax_phase_pos_t; s=["top", "right"])
ax_pos_ggap_t = fig.add_subplot(py"$(gs_2)[1,3]")
ax_pos_ggap_t = pyplot_fxns.remove_axis_box(ax_pos_ggap_t; s=["top", "right"])
cax_t = fig.add_subplot(py"$(gs_2)[1,6]")

# summary plots synch
ax_phase_ggap = fig.add_subplot(py"$(gs_2)[0,1]")
ax_phase_ggap = pyplot_fxns.remove_axis_box(ax_phase_ggap; s=["top", "right"])
ax_phase_pos = fig.add_subplot(py"$(gs_2)[0,5]")
ax_phase_pos = pyplot_fxns.remove_axis_box(ax_phase_pos; s=["top", "right"])
ax_pos_ggap = fig.add_subplot(py"$(gs_2)[0,3]")
ax_pos_ggap = pyplot_fxns.remove_axis_box(ax_pos_ggap; s=["top", "right"])
cax = fig.add_subplot(py"$(gs_2)[0,6]")

# polar plots
ax_init_phase = fig.add_subplot(py"$(gs_1)[0, 0]", polar=true)
ax_ggap = fig.add_subplot(py"$(gs_1)[0, 1]", polar=true)
ax_position = fig.add_subplot(py"$(gs_1)[0, 2]", polar=true)

# plot example spikes
ax_spikes.plot(sol.t ./ 1000, sol[1,:], linewidth=1, color=sim_col_1)
ax_spikes.plot(sol.t ./1000, sol[2,:], linewidth=1, color=sim_col_2)
ax_spikes.set_xlim(tspan_sol[1] ./1000, tspan_sol[2] ./1000)
ax_spikes.set_ylabel("Soma (mV)")

# plot example polar
ax_phase_polar.plot(phaseps, t2ps[1:end-1] ./ 1000,linewidth=2, color=sim_col_2)
ax_phase_polar.set_rgrids([0, 1, 2], labels=["","",""], angle=π/4, ha="right")
ax_phase_polar.set_xticks(0:π/4:7*π/4)
ax_phase_polar.set_xticklabels(rad_labels)
label_position=0 #ax.get_rlabel_position()
ax_phase_polar.tick_params(axis="x", which="major", pad=-2)
ax_phase_polar.text(label_position,ax_phase_polar.get_rmax()/2.,"Time (s)", rotation=label_position, va="top", ha="center",  size=6)


# example plot formatting
for ax in [ax_init_phase, ax_ggap, ax_position, ]
    ax.set_rgrids([0, 10, 20, 30], labels=["","","",""], angle=π/4, ha="right")
    ax.set_xticks(0:π/4:7*π/4)
    ax.set_xticklabels(rad_labels)
    label_position=0 #ax.get_rlabel_position()
    ax.tick_params(axis="x", which="major", pad=-2)
end
label_position =  π 
label_rot = 0
# ax_init_phase.set_ylabel("Time (10s)")
# ax_init_phase.set_rlabel_position(180)
# ax_init_phase.text(label_position,ax_init_phase.get_rmax()/2.,"Time (10s)", rotation=label_position, va="top", ha="center", size=7)
# ax_ggap.text(label_position,ax_ggap.get_rmax()/2.,"Time (10s)", rotation=label_position, va="bottom", ha="center", size=7)
# ax_position.text(label_position,ax_position.get_rmax()/2.,"Time (10s)", rotation=label_position, va="bottom", ha="center", size=7)
ax_init_phase.text(label_position, ax_init_phase.get_rmax()*0.6,"Time (10s)", rotation=label_rot, va="top", ha="center", size=6)
ax_ggap.text(label_position,ax_ggap.get_rmax()/2.,"Time (10s)", rotation=label_rot, va="bottom", ha="center", size=6)
ax_position.text(label_position,ax_position.get_rmax()/2.,"Time (10s)", rotation=label_rot, va="bottom", ha="center", size=6)

phase_ex = 5 #5 # init_ph =0.333
ggap_ex =  7 #6 # 14 # ggap=10
position_ex = 2 #5 # weg=21, 60 μm



ggap_array_log = [-2.5, log10.(ggap_array[2:end])...]
ggap_array_log_ticks = [-2.5, -2, -1, 0, 1, 2]
ggap_array_log_labels = ["$(round(10^i, digits=2))" for i in ggap_array_log_ticks] 
c = ax_phase_ggap_t.pcolormesh(ggap_array_log,init_phase_diff_array,  stack(tau_array2[:, :,position_ex]) ./1000, cmap=t_cmap, vmin=z_t_min, vmax=z_t_max, rasterized=true)#, vmin=z_min, vmax=z_max)
ax_phase_ggap_t.axis([ minimum(ggap_array_log),  maximum(ggap_array_log), minimum(init_phase_diff_array), maximum(init_phase_diff_array),])# set the limits of the plot to the limits of the data
ax_phase_ggap_t.set_ylabel(phase_diff_label)
ax_phase_ggap_t.set_xlabel(ggap_label, labelpad=0)
ax_phase_ggap_t.set_yticks([0,0.25,0.5,0.75,1])
ax_phase_ggap_t.set_yticklabels([L"0",L"\frac{1}{2} \pi",L"\pi",L"\frac{3}{2} \pi",L"2 \pi"])
ax_phase_ggap_t.set_xticks(ggap_array_log_ticks)
ax_phase_ggap_t.set_xticklabels(ggap_array_log_labels, rotation=90)


# position vs ggap
c = ax_pos_ggap_t.pcolormesh( position_array, ggap_array_log, stack(tau_array2[phase_ex, :, :]) ./1000, cmap=t_cmap, vmin=z_t_min, vmax=z_t_max, rasterized=true)#, vmin=z_min, vmax=z_max)
ax_pos_ggap_t.axis([ minimum(position_array),  maximum(position_array), minimum(ggap_array_log), maximum(ggap_array_log)]) # set the limits of the plot to the limits of the data
ax_pos_ggap_t.set_xlabel(pos_label2)
ax_pos_ggap_t.set_ylabel(ggap_label, labelpad=0)
ax_pos_ggap_t.set_xticks(pos_label_ticks)
ax_pos_ggap_t.set_xticklabels(pos_label_um)
ax_pos_ggap_t.set_yticks(ggap_array_log_ticks)
ax_pos_ggap_t.set_yticklabels(ggap_array_log_labels)


# init phase vs position
c = ax_phase_pos_t.pcolormesh( position_array,init_phase_diff_array, stack(tau_array2[:, ggap_ex, :]) ./1000, cmap=t_cmap, vmin=z_t_min, vmax=z_t_max, rasterized=true)#, vmin=z_min, vmax=z_max)
ax_phase_pos_t.axis([ minimum(position_array),  maximum(position_array), minimum(init_phase_diff_array), maximum(init_phase_diff_array)]) # set the limits of the plot to the limits of the data
ax_phase_pos_t.set_ylabel(phase_diff_label)
ax_phase_pos_t.set_xlabel(pos_label2)
ax_phase_pos_t.set_xticks(pos_label_ticks)
ax_phase_pos_t.set_xticklabels(pos_label_um)
ax_phase_pos_t.set_yticks([0,0.25,0.5,0.75,1])
ax_phase_pos_t.set_yticklabels([L"0",L"\frac{1}{2} \pi",L"\pi",L"\frac{3}{2} \pi",L"2 \pi"])


# colorbar 
cb = fig.colorbar(c, cax=cax_t, orientation="vertical", fraction=0.05,extend="max", label="Synch. " *L"$\tau$" * " (s)")
# cb.set_label("Synch Time (s)", labelpad=-10, y=1.2, rotation=0)


# summary synch plotting
# phase_ex = 5
# ggap_ex = 6 #
# position_ex = 3

ggap_array_log = [-2.5, log10.(ggap_array[2:end])...]
ggap_array_log_ticks = [-2.5, -2, -1, 0, 1, 2]
ggap_array_log_labels = ["$(round(10^i, digits=2))" for i in ggap_array_log_ticks] 
c = ax_phase_ggap.pcolormesh(ggap_array_log,init_phase_diff_array,  mean_peak_phase_end_array[:, :,position_ex], cmap= PLV_cmap, vmin=z_min, vmax=z_max, rasterized=true)
ax_phase_ggap.axis([ minimum(ggap_array_log),  maximum(ggap_array_log), minimum(init_phase_diff_array), maximum(init_phase_diff_array),])# set the limits of the plot to the limits of the data
ax_phase_ggap.set_ylabel(phase_diff_label)
# ax_phase_ggap.set_xlabel(ggap_label)
ax_phase_ggap.set_yticks([0,0.25,0.5,0.75,1])
ax_phase_ggap.set_yticklabels([L"0",L"\frac{1}{2} \pi",L"\pi",L"\frac{3}{2} \pi",L"2 \pi"])
ax_phase_ggap.set_xticks(ggap_array_log_ticks)
ax_phase_ggap.set_xticklabels([])
# ax_phase_ggap.set_xticklabels(ggap_array_log_labels, rotation=90)


# position vs ggap
c = ax_pos_ggap.pcolormesh(position_array, ggap_array_log, mean_peak_phase_end_array[phase_ex, :, :], cmap=PLV_cmap, vmin=z_min, vmax=z_max, rasterized=true)
ax_pos_ggap.axis([ minimum(position_array),  maximum(position_array), minimum(ggap_array_log), maximum(ggap_array_log)]) # set the limits of the plot to the limits of the data
# ax_pos_ggap.set_xlabel(pos_label2)
ax_pos_ggap.set_ylabel(ggap_label, labelpad=0)
ax_pos_ggap.set_xticks(pos_label_ticks)
ax_pos_ggap.set_xticklabels([])
# ax_pos_ggap.set_xticklabels(pos_label_um)
ax_pos_ggap.set_yticks(ggap_array_log_ticks)
ax_pos_ggap.set_yticklabels(ggap_array_log_labels)


# init phase vs position
c = ax_phase_pos.pcolormesh(position_array,init_phase_diff_array,  mean_peak_phase_end_array[:, ggap_ex, :], cmap=PLV_cmap, vmin=z_min, vmax=z_max, rasterized=true)
ax_phase_pos.axis([minimum(position_array),  maximum(position_array), minimum(init_phase_diff_array), maximum(init_phase_diff_array)]) # set the limits of the plot to the limits of the data
ax_phase_pos.set_ylabel(phase_diff_label)
# ax_phase_pos.set_xlabel(pos_label2)
ax_phase_pos.set_xticks(pos_label_ticks)
# ax_phase_pos.set_xticklabels(pos_label_um)
ax_phase_pos.set_yticks([0,0.25,0.5,0.75,1])
ax_phase_pos.set_xticklabels([])
ax_phase_pos.set_yticklabels([L"0",L"\frac{1}{2} \pi",L"\pi",L"\frac{3}{2} \pi",L"2 \pi"])


# colorbar 
cb2 = fig.colorbar(c, cax=cax, orientation="vertical", fraction=0.05,label= L"\Delta \Phi"*" at 30s")#, vmin=-π, vmax=π,)
cb2.ax.set_yticks([-π, -0.5*π, 0,0.5*π,π])
cb2.ax.set_yticklabels([L"-\pi",L"- \frac{1}{2} \pi", L"0",L"\frac{1}{2} \pi",L"\pi"])
# cb2.set_label(L"\Delta \Phi"*" at 30s", labelpad=-20, y=1.2, rotation=0)


# init phase diff timeseries example
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", [init_col,phase_col])
norm = matplotlib.colors.Normalize(vmin=minimum(init_phase_diff_array), vmax=1)#maximum(init_phase_diff_array))
for i =1:n_phase
    ax_init_phase.plot(phase_diff_array[i,ggap_ex,position_ex], spike_times[i,ggap_ex,position_ex, 1][1:end-1] ./ 1000, c=cmap1(norm(init_phase_diff_array[i])))
end
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap1)
init_phase_diff_array_dt = init_phase_diff_array[2] - init_phase_diff_array[1]
cb1_tick_n = 2
cb1 = fig.colorbar(sm, ax = ax_init_phase,  ticks=init_phase_diff_array[1:3:length(init_phase_diff_array)], pad=0.2, shrink=cb_scale, fraction=0.05,
 boundaries= minimum(init_phase_diff_array)-init_phase_diff_array_dt/2:init_phase_diff_array_dt:maximum(init_phase_diff_array)+ init_phase_diff_array_dt)
# init_phase_diff_array_labels = [L"0",L"1/6 \pi",L"1/3 \pi",L"1/2 \pi",L"2/3 \pi",L"5/6 \pi",
#             L"\pi",L"7/6 \pi",L"4/3 \pi",L"3/2 \pi",L"5/3 \pi",L"11/6 \pi"]#, L"2 \pi"]
init_phase_diff_array_labels = [L"0",L"\frac{1}/{6} \pi",L"1/3 \pi",L"$\frac{1}/{2}$\pi",L"2/3 \pi",L"5/6 \pi",
            L"\pi",L"7/6 \pi",L"4/3 \pi",L"$\frac{3}/{2}$ \pi",L"5/3 \pi",L"11/6 \pi"]#, L"2 \pi"]
# cb1.ax.set_yticklabels(init_phase_diff_array_labels[1:3:length(init_phase_diff_array)])
cb1.ax.set_yticks([0., 0.25, 0.5, 0.75])#0.9166666666666666])
cb1.ax.set_yticklabels([L"0",L"\frac{1}{2} \pi", L"\pi",L"\frac{3}{2} \pi"])
cb1.set_label("Initial ΔΦ", labelpad=-20, y=1.1, rotation=0)
cb1.solids.set_edgecolor("face")

# ggap timeseries example
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", [init_col,ggap_col])
norm2 = matplotlib.colors.Normalize(vmin=minimum(ggap_array), vmax=maximum(ggap_array))
ggap_log10 = [-2.5, log10.(ggap_array[2:end])...]
norm2_log = matplotlib.colors.Normalize(vmin=minimum(ggap_log10), vmax=maximum(ggap_log10))
ax_ggap.plot(phase_diff_array[phase_ex,1,position_ex], spike_times[phase_ex,1,position_ex, 1][1:end-1] ./ 1000, c=cmap2(norm2_log(-2.5)))
for i =2:size(ggap_array)[1]
    ax_ggap.plot(phase_diff_array[phase_ex,i,position_ex], spike_times[phase_ex,i,position_ex, 1][1:end-1] ./ 1000, c=cmap2(norm2_log(log10(ggap_array[i]))))
end
ggap_array_dt = log10(ggap_array[3]) - log10(ggap_array[2])
ggap_log10_labels = ["0",["10\$^{$(round(i, digits=0))}\$" for i in ggap_array_log_ticks[2:end]]...]
sm2 = plt.cm.ScalarMappable(norm=norm2_log, cmap=cmap2)
cb2_tick_n = 2
cb2 = fig.colorbar(sm2, ax = ax_ggap,  ticks=ggap_array_log_ticks, pad=0.2, shrink=cb_scale, fraction=0.05,
 boundaries= minimum(ggap_log10)-ggap_array_dt/2:ggap_array_dt:maximum(ggap_log10)+ ggap_array_dt)
cb2.ax.set_yticks(ggap_array_log_ticks)
cb2.ax.set_yticklabels(ggap_array_log_labels)
cb2.set_label(ggap_label, labelpad=-15, y=1.1, rotation=0)
cb2.solids.set_edgecolor("face")

# position timeseries example
cmap3 = matplotlib.colors.LinearSegmentedColormap.from_list("", [init_col, pos_col])
norm3 = matplotlib.colors.Normalize(vmin=minimum(position_array), vmax=maximum(position_array))
for i =1:size(position_array)[1]
    ax_position.plot(phase_diff_array[phase_ex,ggap_ex, i], spike_times[phase_ex,ggap_ex, i, 1][1:end-1] ./ 1000, c=cmap3(norm3(position_array[i])))
end
sm3 = plt.cm.ScalarMappable(norm=norm3, cmap=cmap3)
position_array_dt = position_array[2] - position_array[1]
# position_labels = ["Soma", ["$(dist_incr1*(i-1)) μm" for i =position_array[2:end]]...]
position_labels = ["Soma", ["$(dist_incr1*(i-1))" for i =position_array[2:end]]...]
cb3_tick_n = 5
cb3 = fig.colorbar(sm3, ax = ax_position,  ticks=position_array[1:cb3_tick_n:length(position_labels)], pad=0.2, shrink=cb_scale,  fraction=0.05,
 boundaries= minimum(position_array)-position_array_dt/2:position_array_dt:maximum(position_array)+ position_array_dt)
cb3.ax.set_yticklabels(position_labels[1:cb3_tick_n:length(position_labels)])
cb3.set_label("Position (μm)", labelpad=-25, y=1.1, rotation=0)
cb3.solids.set_edgecolor("face")

# subplot letters 
letters = collect('A':'Z')
ax_spikes.text(-0.01, 1.075, "$(letters[1])", transform=ax_spikes.transAxes, size=letter_size, weight="bold")
ax_phase_polar.text(x_letter-0.2, y_letter, "$(letters[2])", transform=ax_phase_polar.transAxes, size=letter_size, weight="bold")
ax_init_phase.text(x_letter_polar, y_letter_polar, "$(letters[3])", transform=ax_init_phase.transAxes, size=letter_size, weight="bold")
ax_ggap.text(x_letter_polar, y_letter_polar, "$(letters[4])", transform=ax_ggap.transAxes, size=letter_size, weight="bold")
ax_position.text(x_letter_polar, y_letter_polar, "$(letters[5])", transform=ax_position.transAxes, size=letter_size, weight="bold")
ax_phase_ggap.text(x_letter-0.275, y_letter, "$(letters[6])", transform=ax_phase_ggap.transAxes, size=letter_size, weight="bold")
ax_pos_ggap.text(x_letter-0.36, y_letter, "$(letters[7])", transform=ax_pos_ggap.transAxes, size=letter_size, weight="bold")
ax_phase_pos.text(x_letter-0.275, y_letter, "$(letters[8])", transform=ax_phase_pos.transAxes, size=letter_size, weight="bold")
# ax_phase_ggap_t.text(x_letter, y_letter, "$(letters[9])", transform=ax_phase_ggap_t.transAxes, size=letter_size, weight="bold")
# ax_pos_ggap_t.text(x_letter, y_letter, "$(letters[10])", transform=ax_pos_ggap_t.transAxes, size=letter_size, weight="bold")
# ax_phase_pos_t.text(x_letter, y_letter, "$(letters[11])", transform=ax_phase_pos_t.transAxes, size=letter_size, weight="bold")

# plt.savefig(plotsdir("figs", "Figure_3_soma_Ih_coupled_ex_ggap_properties_tau_$(tau_p).png"), dpi=600)
# plt.savefig(plotsdir("figs", "Figure_3_soma_Ih_coupled_ex_ggap_properties_tau_$(tau_p).pdf"), dpi=600)
# plt.savefig(plotsdir("figs", "Figure_3_soma_Ih_coupled_ex_ggap_properties_tau_$(tau_p).eps"), dpi=600)
plt.savefig(plotsdir("Figure_3_soma_Ih_coupled_ex_ggap_properties_tau.png"), dpi=600)
plt.savefig(plotsdir("Figure_3_soma_Ih_coupled_ex_ggap_properties_tau.pdf"), dpi=600)
plt.savefig(plotsdir("Figure_3_soma_Ih_coupled_ex_ggap_properties_tau.eps"), dpi=600)
plt.show()




# #%%
# # Fig Setup
# fig = plt.figur(=(6.9, 7.5))
# gs_all  = fig.add_gridspec(7, 2, bottom=0.1, top=0.95, right=0.9, left=0.1, wspace=0.25, hspace=0.25,  width_ratios=[0.76, 0.27], height_ratios=[0.7, 0.2, 1.2, 0.0, 0.75, 0.275, 0.75])
# gs_0_0  =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.5, hspace=0.25,height_ratios=[1, 0.7],  subplot_spec=py"$(gs_all)[0,0]") 
# gs_0_1  =  gridspec.GridSpecFromSubplotSpec(1, 1, wspace=0.5, hspace=0.25, subplot_spec=py"$(gs_all)[0,1]") 
# gs_2 =  gridspec.GridSpecFromSubplotSpec(1, 6, wspace=0.45, hspace=0.5, width_ratios=[0.3,0.05, 0.3,0.05, 0.3,0.01],  subplot_spec=py"$(gs_all)[4, :]") 
# gs_3 =  gridspec.GridSpecFromSubplotSpec(1, 6, wspace=0.45, hspace=0.5, width_ratios=[0.3,0.05, 0.3,0.05, 0.3,0.01],  subplot_spec=py"$(gs_all)[6, :]") 
# gs_1 =  gridspec.GridSpecFromSubplotSpec(1, 3, wspace=0.65, hspace=1.0,  subplot_spec=py"$(gs_all)[2, :]") 


# # example plots
# ax_spikes = fig.add_subplot(py"$(gs_0_0)[0]")
# ax_spikes = pyplot_fxns.remove_axis_box(ax_spikes; s=["top", "right", "bottom"])
# ax_phase = fig.add_subplot(py"$(gs_0_0)[1]")
# ax_phase = pyplot_fxns.remove_axis_box(ax_phase; s=["top", "right"])
# ax_phase_polar = fig.add_subplot(py"$(gs_0_1)[0]", polar=true)


# # summary plots time
# ax_phase_ggap_t = fig.add_subplot(py"$(gs_3)[0,0]")
# ax_phase_ggap_t = pyplot_fxns.remove_axis_box(ax_phase_ggap_t; s=["top", "right"])
# ax_phase_pos_t = fig.add_subplot(py"$(gs_3)[0,4]")
# ax_phase_pos_t = pyplot_fxns.remove_axis_box(ax_phase_pos_t; s=["top", "right"])
# ax_pos_ggap_t = fig.add_subplot(py"$(gs_3)[0,2]")
# ax_pos_ggap_t = pyplot_fxns.remove_axis_box(ax_pos_ggap_t; s=["top", "right"])
# cax_t = fig.add_subplot(py"$(gs_3)[0,5]")

# # summary plots synch
# ax_phase_ggap = fig.add_subplot(py"$(gs_2)[0,0]")
# ax_phase_ggap = pyplot_fxns.remove_axis_box(ax_phase_ggap; s=["top", "right"])
# ax_phase_pos = fig.add_subplot(py"$(gs_2)[0,4]")
# ax_phase_pos = pyplot_fxns.remove_axis_box(ax_phase_pos; s=["top", "right"])
# ax_pos_ggap = fig.add_subplot(py"$(gs_2)[0,2]")
# ax_pos_ggap = pyplot_fxns.remove_axis_box(ax_pos_ggap; s=["top", "right"])
# cax = fig.add_subplot(py"$(gs_2)[0,5]")

# # polar plots
# ax_init_phase = fig.add_subplot(py"$(gs_1)[0, 0]", polar=true)
# ax_ggap = fig.add_subplot(py"$(gs_1)[0, 1]", polar=true)
# ax_position = fig.add_subplot(py"$(gs_1)[0, 2]", polar=true)

# # plot example spikes
# ax_spikes.plot(sol.t ./ 1000, sol[1,:], linewidth=1, color=sim_col_1)
# ax_spikes.plot(sol.t ./1000, sol[2,:], linewidth=1, color=sim_col_2)
# ax_spikes.set_xlim(tspan_sol[1] ./1000, tspan_sol[2] ./1000)
# ax_spikes.set_ylabel("Soma (mV)")

# # plot example phase diff
# ax_phase.plot(t1ps[2:end] ./ 1000,phaseps ./ (2*pi), linewidth=2, color=sim_col_2)
# ax_phase.set_xlim(tspan_sol[1] ./1000, tspan_sol[2] ./1000)
# ax_phase.set_ylim(0, 1.05)
# ax_phase.set_yticks([0, 0.5, 1])
# ax_phase.set_yticklabels([L"0",L"\pi",L"2 \pi", ])
# ax_phase.set_ylabel("ΔΦ")
# ax_phase.set_xlabel("Time (s)")
# ax_phase.text(1.9, 0.5, "End ΔΦ = $(round(mean_peak_phase_end / π, digits=2))π")

# # plot example polar
# ax_phase_polar.plot(phaseps, t2ps[1:end-1] ./ 1000,linewidth=2, color=sim_col_2)
# ax_phase_polar.set_rgrids([0, 1, 2, 3], labels=["","","",""], angle=π/4, ha="right")
# ax_phase_polar.set_xticks(0:π/4:7*π/4)
# ax_phase_polar.set_xticklabels(rad_labels)
# label_position=0 #ax.get_rlabel_position()
# ax_phase_polar.tick_params(axis="x", which="major", pad=-2)
# ax_phase_polar.text(label_position,ax_phase_polar.get_rmax()/2.,"Time (s)", rotation=label_position, va="top", ha="center",  size=8)


# # example plot formatting
# for ax in [ax_init_phase, ax_ggap, ax_position, ]
#     ax.set_rgrids([0, 10, 20, 30], labels=["","","",""], angle=π/4, ha="right")
#     ax.set_xticks(0:π/4:7*π/4)
#     ax.set_xticklabels(rad_labels)
#     label_position=0 #ax.get_rlabel_position()
#     ax.tick_params(axis="x", which="major", pad=-2)
# end
# label_position = π
# ax_init_phase.text(label_position,ax_init_phase.get_rmax()/2.,"Time (s)", rotation=label_position, va="top", ha="center", size=8)
# ax_ggap.text(label_position,ax_ggap.get_rmax()/2.,"Time (s)", rotation=label_position, va="bottom", ha="center", size=8)
# ax_position.text(label_position,ax_position.get_rmax()/2.,"Time (s)", rotation=label_position, va="bottom", ha="center", size=8)

# phase_ex = 5 # init_ph =0.333
# ggap_ex =  7 #6 # 14 # ggap=10
# position_ex = 2 #5 # weg=21, 60 μm

# # phase_ex = 5 # init_ph =0.333
# # ggap_ex =  8 #6 # 14 # ggap=10
# # position_ex = 3 #5 # weg=21, 60 μm


# # phase_ex = 3 # init_ph =0.333
# # ggap_ex =  8 #6 # 14 # ggap=10
# # position_ex = 3 #5 # weg=21, 60 μm

# # phase_ex = 5
# # ggap_ex = 8 #6 #
# # position_ex = 3

# ggap_array_log = [-2.5, log10.(ggap_array[2:end])...]
# ggap_array_log_ticks = [-2.5, -2, -1, 0, 1, 2]
# ggap_array_log_labels = ["$(round(10^i, digits=2))" for i in ggap_array_log_ticks] 
# c = ax_phase_ggap_t.pcolormesh(ggap_array_log,init_phase_diff_array,  stack(tau_array2[:, :,position_ex]) ./1000, cmap=t_cmap, vmin=z_t_min, vmax=z_t_max, rasterized=true)#, vmin=z_min, vmax=z_max)
# ax_phase_ggap_t.axis([ minimum(ggap_array_log),  maximum(ggap_array_log), minimum(init_phase_diff_array), maximum(init_phase_diff_array),])# set the limits of the plot to the limits of the data
# ax_phase_ggap_t.set_ylabel(phase_diff_label)
# ax_phase_ggap_t.set_xlabel(ggap_label)
# ax_phase_ggap_t.set_yticks([0,0.25,0.5,0.75,1])
# ax_phase_ggap_t.set_yticklabels([L"0",L"\frac{1}{2} \pi",L"\pi",L"\frac{3}{2} \pi",L"2 \pi"])
# ax_phase_ggap_t.set_xticks(ggap_array_log_ticks)
# ax_phase_ggap_t.set_xticklabels(ggap_array_log_labels, rotation=90)


# # position vs ggap
# c = ax_pos_ggap_t.pcolormesh( position_array, ggap_array_log, stack(tau_array2[phase_ex, :, :]) ./1000, cmap=t_cmap, vmin=z_t_min, vmax=z_t_max, rasterized=true)#, vmin=z_min, vmax=z_max)
# ax_pos_ggap_t.axis([ minimum(position_array),  maximum(position_array), minimum(ggap_array_log), maximum(ggap_array_log)]) # set the limits of the plot to the limits of the data
# ax_pos_ggap_t.set_xlabel(pos_label2)
# ax_pos_ggap_t.set_ylabel(ggap_label)
# ax_pos_ggap_t.set_xticks(pos_label_ticks)
# ax_pos_ggap_t.set_xticklabels(pos_label_um)
# ax_pos_ggap_t.set_yticks(ggap_array_log_ticks)
# ax_pos_ggap_t.set_yticklabels(ggap_array_log_labels)


# # init phase vs position
# c = ax_phase_pos_t.pcolormesh( position_array,init_phase_diff_array, stack(tau_array2[:, ggap_ex, :]) ./1000, cmap=t_cmap, vmin=z_t_min, vmax=z_t_max, rasterized=true)#, vmin=z_min, vmax=z_max)
# ax_phase_pos_t.axis([ minimum(position_array),  maximum(position_array), minimum(init_phase_diff_array), maximum(init_phase_diff_array)]) # set the limits of the plot to the limits of the data
# ax_phase_pos_t.set_ylabel(phase_diff_label)
# ax_phase_pos_t.set_xlabel(pos_label2)
# ax_phase_pos_t.set_xticks(pos_label_ticks)
# ax_phase_pos_t.set_xticklabels(pos_label_um)
# ax_phase_pos_t.set_yticks([0,0.25,0.5,0.75,1])
# ax_phase_pos_t.set_yticklabels([L"0",L"\frac{1}{2} \pi",L"\pi",L"\frac{3}{2} \pi",L"2 \pi"])


# # colorbar 
# cb = fig.colorbar(c, cax=cax_t, orientation="vertical", fraction=0.05,extend="max", label="Synch. " *L"$\tau$" * " (s)")
# # cb.set_label("Synch Time (s)", labelpad=-10, y=1.2, rotation=0)


# # summary synch plotting
# # phase_ex = 5
# # ggap_ex = 6 #
# # position_ex = 3

# ggap_array_log = [-2.5, log10.(ggap_array[2:end])...]
# ggap_array_log_ticks = [-2.5, -2, -1, 0, 1, 2]
# ggap_array_log_labels = ["$(round(10^i, digits=2))" for i in ggap_array_log_ticks] 
# c = ax_phase_ggap.pcolormesh(ggap_array_log,init_phase_diff_array,  mean_peak_phase_end_array[:, :,position_ex], cmap= PLV_cmap, vmin=z_min, vmax=z_max, rasterized=true)
# ax_phase_ggap.axis([ minimum(ggap_array_log),  maximum(ggap_array_log), minimum(init_phase_diff_array), maximum(init_phase_diff_array),])# set the limits of the plot to the limits of the data
# ax_phase_ggap.set_ylabel(phase_diff_label)
# ax_phase_ggap.set_xlabel(ggap_label)
# ax_phase_ggap.set_yticks([0,0.25,0.5,0.75,1])
# ax_phase_ggap.set_yticklabels([L"0",L"\frac{1}{2} \pi",L"\pi",L"\frac{3}{2} \pi",L"2 \pi"])
# ax_phase_ggap.set_xticks(ggap_array_log_ticks)
# ax_phase_ggap.set_xticklabels(ggap_array_log_labels, rotation=90)


# # position vs ggap
# c = ax_pos_ggap.pcolormesh(position_array, ggap_array_log, mean_peak_phase_end_array[phase_ex, :, :], cmap=PLV_cmap, vmin=z_min, vmax=z_max, rasterized=true)
# ax_pos_ggap.axis([ minimum(position_array),  maximum(position_array), minimum(ggap_array_log), maximum(ggap_array_log)]) # set the limits of the plot to the limits of the data
# ax_pos_ggap.set_xlabel(pos_label2)
# ax_pos_ggap.set_ylabel(ggap_label)
# ax_pos_ggap.set_xticks(pos_label_ticks)
# ax_pos_ggap.set_xticklabels(pos_label_um)
# ax_pos_ggap.set_yticks(ggap_array_log_ticks)
# ax_pos_ggap.set_yticklabels(ggap_array_log_labels)


# # init phase vs position
# c = ax_phase_pos.pcolormesh(position_array,init_phase_diff_array,  mean_peak_phase_end_array[:, ggap_ex, :], cmap=PLV_cmap, vmin=z_min, vmax=z_max, rasterized=true)
# ax_phase_pos.axis([minimum(position_array),  maximum(position_array), minimum(init_phase_diff_array), maximum(init_phase_diff_array)]) # set the limits of the plot to the limits of the data
# ax_phase_pos.set_ylabel(phase_diff_label)
# ax_phase_pos.set_xlabel(pos_label2)
# ax_phase_pos.set_xticks(pos_label_ticks)
# ax_phase_pos.set_xticklabels(pos_label_um)
# ax_phase_pos.set_yticks([0,0.25,0.5,0.75,1])
# ax_phase_pos.set_yticklabels([L"0",L"\frac{1}{2} \pi",L"\pi",L"\frac{3}{2} \pi",L"2 \pi"])


# # colorbar 
# cb2 = fig.colorbar(c, cax=cax, orientation="vertical", fraction=0.05,label= L"\Delta \Phi"*" at 30s")#, vmin=-π, vmax=π,)
# cb2.ax.set_yticks([-π, -0.5*π, 0,0.5*π,π])
# cb2.ax.set_yticklabels([L"-\pi",L"- \frac{1}{2} \pi", L"0",L"\frac{1}{2} \pi",L"\pi"])
# # cb2.set_label(L"\Delta \Phi"*" at 30s", labelpad=-20, y=1.2, rotation=0)


# # init phase diff timeseries example
# cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", [init_col,phase_col])
# norm = matplotlib.colors.Normalize(vmin=minimum(init_phase_diff_array), vmax=maximum(init_phase_diff_array))
# for i =1:n_phase
#     ax_init_phase.plot(phase_diff_array[i,ggap_ex,position_ex], spike_times[i,ggap_ex,position_ex, 1][1:end-1] ./ 1000, c=cmap1(norm(init_phase_diff_array[i])))
# end
# sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap1)
# init_phase_diff_array_dt = init_phase_diff_array[2] - init_phase_diff_array[1]
# cb1_tick_n = 2
# cb1 = fig.colorbar(sm, ax = ax_init_phase,  ticks=init_phase_diff_array[1:3:length(init_phase_diff_array)], pad=0.2, shrink=cb_scale, fraction=0.05,
#  boundaries= minimum(init_phase_diff_array)-init_phase_diff_array_dt/2:init_phase_diff_array_dt:maximum(init_phase_diff_array)+ init_phase_diff_array_dt)
# init_phase_diff_array_labels = [L"0",L"1/6 \pi",L"1/3 \pi",L"1/2 \pi",L"2/3 \pi",L"5/6 \pi",
#             L"\pi",L"7/6 \pi",L"4/3 \pi",L"3/2 \pi",L"5/3 \pi",L"11/6 \pi"]#, L"2 \pi"]
# cb1.ax.set_yticklabels(init_phase_diff_array_labels[1:3:length(init_phase_diff_array)])
# cb1.set_label("Initial \nΔΦ", labelpad=-20, y=1.25, rotation=0)
# cb1.solids.set_edgecolor("face")

# # ggap timeseries example
# cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", [init_col,ggap_col])
# norm2 = matplotlib.colors.Normalize(vmin=minimum(ggap_array), vmax=maximum(ggap_array))
# ggap_log10 = [-2.5, log10.(ggap_array[2:end])...]
# norm2_log = matplotlib.colors.Normalize(vmin=minimum(ggap_log10), vmax=maximum(ggap_log10))
# ax_ggap.plot(phase_diff_array[phase_ex,1,position_ex], spike_times[phase_ex,1,position_ex, 1][1:end-1] ./ 1000, c=cmap2(norm2_log(-2.5)))
# for i =2:size(ggap_array)[1]
#     ax_ggap.plot(phase_diff_array[phase_ex,i,position_ex], spike_times[phase_ex,i,position_ex, 1][1:end-1] ./ 1000, c=cmap2(norm2_log(log10(ggap_array[i]))))
# end
# ggap_array_dt = log10(ggap_array[3]) - log10(ggap_array[2])
# ggap_log10_labels = ["0",["10\$^{$(round(i, digits=0))}\$" for i in ggap_array_log_ticks[2:end]]...]
# sm2 = plt.cm.ScalarMappable(norm=norm2_log, cmap=cmap2)
# cb2_tick_n = 2
# cb2 = fig.colorbar(sm2, ax = ax_ggap,  ticks=ggap_array_log_ticks, pad=0.2, shrink=cb_scale, fraction=0.05,
#  boundaries= minimum(ggap_log10)-ggap_array_dt/2:ggap_array_dt:maximum(ggap_log10)+ ggap_array_dt)
# cb2.ax.set_yticks(ggap_array_log_ticks)
# cb2.ax.set_yticklabels(ggap_array_log_labels)
# cb2.set_label(ggap_label, labelpad=-25, y=1.15, rotation=0)
# cb2.solids.set_edgecolor("face")

# # position timeseries example
# cmap3 = matplotlib.colors.LinearSegmentedColormap.from_list("", [init_col, pos_col])
# norm3 = matplotlib.colors.Normalize(vmin=minimum(position_array), vmax=maximum(position_array))
# for i =1:size(position_array)[1]
#     ax_position.plot(phase_diff_array[phase_ex,ggap_ex, i], spike_times[phase_ex,ggap_ex, i, 1][1:end-1] ./ 1000, c=cmap3(norm3(position_array[i])))
# end
# sm3 = plt.cm.ScalarMappable(norm=norm3, cmap=cmap3)
# position_array_dt = position_array[2] - position_array[1]
# position_labels = ["Soma", ["$(dist_incr1*(i-1)) μm" for i =position_array[2:end]]...]
# cb3_tick_n = 5
# cb3 = fig.colorbar(sm3, ax = ax_position,  ticks=position_array[1:cb3_tick_n:length(position_labels)], pad=0.2, shrink=cb_scale,  fraction=0.05,
#  boundaries= minimum(position_array)-position_array_dt/2:position_array_dt:maximum(position_array)+ position_array_dt)
# cb3.ax.set_yticklabels(position_labels[1:cb3_tick_n:length(position_labels)])
# cb3.set_label(pos_label, labelpad=-25, y=1.15, rotation=0)
# cb3.solids.set_edgecolor("face")

# # subplot letters 
# letters = collect('A':'Z')
# ax_spikes.text(-0.01, 1.075, "$(letters[1])", transform=ax_spikes.transAxes, size=letter_size, weight="bold")
# ax_phase_polar.text(x_letter, y_letter, "$(letters[2])", transform=ax_phase_polar.transAxes, size=letter_size, weight="bold")
# ax_init_phase.text(x_letter_polar, y_letter_polar, "$(letters[3])", transform=ax_init_phase.transAxes, size=letter_size, weight="bold")
# ax_ggap.text(x_letter_polar, y_letter_polar, "$(letters[4])", transform=ax_ggap.transAxes, size=letter_size, weight="bold")
# ax_position.text(x_letter_polar, y_letter_polar, "$(letters[5])", transform=ax_position.transAxes, size=letter_size, weight="bold")
# ax_phase_ggap.text(x_letter, y_letter, "$(letters[6])", transform=ax_phase_ggap.transAxes, size=letter_size, weight="bold")
# ax_pos_ggap.text(x_letter, y_letter, "$(letters[7])", transform=ax_pos_ggap.transAxes, size=letter_size, weight="bold")
# ax_phase_pos.text(x_letter, y_letter, "$(letters[8])", transform=ax_phase_pos.transAxes, size=letter_size, weight="bold")
# ax_phase_ggap_t.text(x_letter, y_letter, "$(letters[9])", transform=ax_phase_ggap_t.transAxes, size=letter_size, weight="bold")
# ax_pos_ggap_t.text(x_letter, y_letter, "$(letters[10])", transform=ax_pos_ggap_t.transAxes, size=letter_size, weight="bold")
# ax_phase_pos_t.text(x_letter, y_letter, "$(letters[11])", transform=ax_phase_pos_t.transAxes, size=letter_size, weight="bold")

# plt.savefig(plotsdir("figs", "Figure_3_soma_Ih_coupled_ex_ggap_properties_tau_$(tau_p).png"), dpi=600)
# plt.savefig(plotsdir("figs", "Figure_3_soma_Ih_coupled_ex_ggap_properties_tau_$(tau_p).pdf"), dpi=600)
# plt.savefig(plotsdir("figs", "Figure_3_soma_Ih_coupled_ex_ggap_properties_tau_$(tau_p).eps"), dpi=600)
# plt.show()

# #%%


# #%%
# # #%% FIGURE ######################################################
# # using LaTeXStrings
# # @pyimport matplotlib.gridspec as gridspec
# # @pyimport matplotlib.patches as patches

# # sim_col_1 = "tab:brown"
# # sim_col_2 = "black"
# # letter_size = 10
# # x_letter = -0.075
# # y_letter = 1.075
# # using LaTeXStrings
# # rad_labels = [L"0",L"\frac{1}{4} \pi",L"\frac{1}{2} \pi",L"\frac{3}{4} \pi",L"\pi",L"\frac{5}{4} \pi",
# #         L"\frac{3}{2} \pi",L"\frac{7}{4} \pi", ]
# # # t_lim = (0, 1000)


# # # Fig Setup
# # fig = plt.figure(figsize=(7.5, 8.5))
# # gs_all  = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.25,  width_ratios=[0.76, 0.3], height_ratios=[0.75, 3])#, height_ratios=[0.2, 0.0, 0.15, 0.15, 0.15, 0.15])
# # gs_0_0  =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.5, hspace=0.25,height_ratios=[1, 0.7],  subplot_spec=py"$(gs_all)[0,0]") # width_ratios=[0.15, 0.1,0.1,0.1, 0.025, 0.15],
# # gs_0_1  =  gridspec.GridSpecFromSubplotSpec(1, 1, wspace=0.5, hspace=0.25, subplot_spec=py"$(gs_all)[0,1]") #width_ratios=[0.15, 0.2, 0.1,0.1, 0.2], 
# # gs_1 =  gridspec.GridSpecFromSubplotSpec(4, 3, wspace=0.5, hspace=0.5, height_ratios=[1,1, 0.7 ,0.7], subplot_spec=py"$(gs_all)[1, :]") # width_ratios=[0.15, 0.1,0.1,0.1, 0.025, 0.15],



# # ax_spikes = fig.add_subplot(py"$(gs_0_0)[0]")
# # ax_spikes = pyplot_fxns.remove_axis_box(ax_spikes; s=["top", "right", "bottom"])
# # ax_phase = fig.add_subplot(py"$(gs_0_0)[1]")
# # ax_phase = pyplot_fxns.remove_axis_box(ax_phase; s=["top", "right"])
# # ax_phase_polar = fig.add_subplot(py"$(gs_0_1)[0]", polar=true)

# # ax_phase_ggap = fig.add_subplot(py"$(gs_1)[0,0]")
# # ax_phase_ggap = pyplot_fxns.remove_axis_box(ax_phase_ggap; s=["top", "right"])
# # ax_phase_pos = fig.add_subplot(py"$(gs_1)[0,1]")
# # ax_phase_pos = pyplot_fxns.remove_axis_box(ax_phase_pos; s=["top", "right"])
# # ax_pos_ggap = fig.add_subplot(py"$(gs_1)[0,2]")
# # ax_pos_ggap = pyplot_fxns.remove_axis_box(ax_pos_ggap; s=["top", "right"])

# # ax_phase_ggap_PLV = fig.add_subplot(py"$(gs_1)[1,0]")
# # ax_phase_ggap_PLV = pyplot_fxns.remove_axis_box(ax_phase_ggap_PLV; s=["top", "right"])
# # ax_phase_pos_PLV = fig.add_subplot(py"$(gs_1)[1,1]")
# # ax_phase_pos_PLV = pyplot_fxns.remove_axis_box(ax_phase_pos_PLV; s=["top", "right"])
# # ax_pos_ggap_PLV = fig.add_subplot(py"$(gs_1)[1,2]")
# # ax_pos_ggap_PLV = pyplot_fxns.remove_axis_box(ax_pos_ggap_PLV; s=["top", "right"])


# # ax_init_phase = fig.add_subplot(py"$(gs_1)[2, 0]", polar=true)
# # ax_ggap = fig.add_subplot(py"$(gs_1)[2, 1]", polar=true)
# # ax_position = fig.add_subplot(py"$(gs_1)[2, 2]", polar=true)

# # ax_init_phase2 = fig.add_subplot(py"$(gs_1)[3, 0]", polar=true)
# # ax_ggap2 = fig.add_subplot(py"$(gs_1)[3, 1]", polar=true)
# # ax_position2 = fig.add_subplot(py"$(gs_1)[3, 2]", polar=true)

# # bpad = 0.1
# # fig.subplots_adjust(left=bpad, bottom=bpad, right=1-bpad, top=1-bpad/2)#, wspace=0, hspace=0)




# # ax_spikes.plot(sol.t ./ 1000, sol[1,:], linewidth=1, color=sim_col_1)
# # ax_spikes.plot(sol.t ./1000, sol[2,:], linewidth=1, color=sim_col_2)
# # ax_spikes.set_xlim(tspan[1] ./1000, tspan[2] ./1000)
# # ax_spikes.set_ylabel("Soma \n(mV)")

# # ax_phase.plot(t1ps[2:end] ./ 1000,phaseps ./ (2*pi), linewidth=2, color=sim_col_2)
# # ax_phase.set_xlim(tspan[1] ./1000, tspan[2] ./1000)
# # ax_phase.set_ylim(0, 1.05)
# # ax_phase.set_yticks([0, 0.5, 1])
# # ax_phase.set_yticklabels([L"0",L"\pi",L"2 \pi", ])
# # ax_phase.set_ylabel("ΔPhase")
# # ax_phase.set_xlabel("Time (s)")
# # # ax_phase.text(2.4, 0.25, "PLV = $(round(PLV, digits=2))")
# # ax_phase.text(1.75, 0.5, "End ΔPhase = $(round(mean_peak_phase_end / π, digits=2))π \nPLV = $(round(PLV, digits=2))")


# # ax_phase_polar.plot(phaseps, t2ps[1:end-1] ./ 1000,linewidth=2, color=sim_col_2)



# # for ax in [ax_phase_polar, ax_init_phase, ax_ggap, ax_position, ax_init_phase2, ax_ggap2, ax_position2]
# #     ax.set_rgrids([0, 1, 2, 3], labels=["0","1","2","3"], angle=π/4, ha="right")
# #     ax.set_xticks(0:π/4:7*π/4)
# #     ax.set_xticklabels(rad_labels)
# #     ax.set_rlim(tspan[1] ./1000, tspan[2] ./1000)
# #     label_position=ax.get_rlabel_position()
# #     ax.text(label_position ,ax.get_rmax()/2.,"Time (s)", rotation=label_position,ha="center",va="center") # - (π/8)
# # end



# # pos_label = "position"
# # ggap_label = "ggap"
# # phase_diff_label = "initial phase diff"
# # PLV_cmap = "Greys"
# # z_min = 0
# # z_max = 1


# # position_ex = 7
# # # ax_phase_ggap_PLV.set_yscale("log")
# # c = ax_phase_ggap_PLV.pcolormesh(init_phase_diff_array, ggap_array, PLV_array[:, :,position_ex]', cmap=PLV_cmap, vmin=z_min, vmax=z_max)
# # ax_phase_ggap_PLV.axis([minimum(init_phase_diff_array), maximum(init_phase_diff_array), minimum(ggap_array),  maximum(ggap_array)])# set the limits of the plot to the limits of the data
# # fig.colorbar(c, ax=ax_phase_ggap_PLV)
# # ax_phase_ggap_PLV.set_xlabel(phase_diff_label)
# # ax_phase_ggap_PLV.set_ylabel(ggap_label)


# # phase_ggap_phase_ex = 7
# # phase_ggap_ggap_ex = 7# (x,y), width, height

# # # rect1 = patches.Rectangle((init_phase_diff_array[phase_ggap_phase_ex], -1), 0., log10(maximum(ggap_array))*1.05,  linewidth=2,clip_on=false, edgecolor="b", facecolor="none",linestyle="-")
# # # ax_phase_ggap_PLV.add_patch(rect1)

# # # rect2 = patches.Rectangle((0., log10(ggap_array[phase_ggap_ggap_ex])), maximum(init_phase_diff_array)*1.05, 0.1, linewidth=2,clip_on=false, edgecolor="g", facecolor="none",linestyle="-")
# # # ax_phase_ggap_PLV.add_patch(rect2)



# # cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightgrey","darkgrey"])
# # colors1 = cmap1(collect(1:1:n_phase)/n_phase)
# # for i =1:n_phase
# #     ax_init_phase.plot(phase_diff_array[i,phase_ggap_ggap_ex,position_ex], spike_times[i,phase_ggap_ggap_ex,phase_ggap_position, 1][1:end-1], c=colors1[i, :])
# # end


# # cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightgrey","darkgrey"])
# # colors2 = cmap2(collect(1:1:size(ggap_array)[1])/size(ggap_array)[1])
# # for i =1:size(ggap_array)[1]
# #     ax_init_phase2.plot(phase_diff_array[phase_ggap_phase_ex,i,phase_ggap_position], spike_times[phase_ggap_phase_ex,i,phase_ggap_position, 1][1:end-1], c=colors2[i, :])
# # end



# # phase_pos_ggap = 5
# # c = ax_phase_pos_PLV.pcolormesh(init_phase_diff_array, position_array, PLV_array[:, phase_pos_ggap, :]', cmap=PLV_cmap, vmin=z_min, vmax=z_max)
# # ax_phase_pos_PLV.axis([minimum(init_phase_diff_array), maximum(init_phase_diff_array), minimum(position_array),  maximum(position_array)]) # set the limits of the plot to the limits of the data
# # fig.colorbar(c, ax=ax_phase_pos_PLV)
# # ax_phase_pos_PLV.set_xlabel(phase_diff_label)
# # ax_phase_pos_PLV.set_ylabel(pos_label)

# # pos_ggap_init_phase = 6
# # # ax_pos_ggap_PLV.set_yscale("log")
# # c = ax_pos_ggap_PLV.pcolormesh( position_array, ggap_array, PLV_array[pos_ggap_init_phase, :, :], cmap=PLV_cmap, vmin=z_min, vmax=z_max)
# # ax_pos_ggap_PLV.axis([ minimum(position_array),  maximum(position_array), minimum(ggap_array), maximum(ggap_array)]) # set the limits of the plot to the limits of the data
# # fig.colorbar(c, ax=ax_pos_ggap_PLV)
# # ax_pos_ggap_PLV.set_xlabel(pos_label)
# # ax_pos_ggap_PLV.set_ylabel(ggap_label)




# # # letters 
# # letters = collect('A':'Z')
# # ax_spikes.text(-0.01, 1.075, "$(letters[1])", transform=ax_spikes.transAxes, size=letter_size, weight="bold")
# # # ax_phase.text(-0.01, 1.075, "$(letters[2])", transform=ax_phase.transAxes, size=letter_size, weight="bold")
# # ax_phase_polar.text(x_letter, y_letter, "$(letters[3])", transform=ax_phase_polar.transAxes, size=letter_size, weight="bold")
# # ax_phase_ggap_PLV.text(x_letter, y_letter, "$(letters[4])", transform=ax_init_phase.transAxes, size=letter_size, weight="bold")
# # ax_phase_pos_PLV.text(x_letter, y_letter, "$(letters[5])", transform=ax_ggap.transAxes, size=letter_size, weight="bold")
# # ax_pos_ggap_PLV.text(x_letter, y_letter, "$(letters[6])", transform=ax_position.transAxes, size=letter_size, weight="bold")


# # # plt.savefig(plotsdir("CSC_Ih", "Figure_4_soma_Ih_coupled_ex_ggap_properties.png"))
# # plt.show()