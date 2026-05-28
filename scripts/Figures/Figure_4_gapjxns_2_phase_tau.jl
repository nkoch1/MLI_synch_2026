
using DrWatson
@quickactivate  "MLI_synch_2026"
using DifferentialEquations, DiffEqCallbacks,
    Optimization, SciMLSensitivity,
    Zygote, DiffEqCallbacks, JLD2, Statistics, Peaks
using FHist
using PyCall
using PyPlot
include(srcdir("pyplot_fxns.jl"))
using .pyplot_fxns
using LaTeXStrings
@pyimport matplotlib.gridspec as gridspec

#%% read in WT data
tspan = (0.0, 30*1000.0)
n_phase=12 
init_phase_diff_array = collect(0:1/n_phase:1 - 1/n_phase)
p_int = 5
position_array = collect(1:p_int:51)
hilb_spike_num = 5

# ggap_array = [0, 0.01, 0.03162277660168379, 0.1, 0.31622776601683794, 1, 3.1622776601683795, 10, 31.622776601683793]
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
);

cond_array = ["no_Kd", "no_Ad", "no_SKd", "no_HVAd", "no_Td"]

PLV_cmap = "twilight_shifted"
# t_cmap = "hsv" #"viridis"
t_cmap = "magma_r"

pos_label = "Position"
pos_label2 = "Position (μm)"
pos_label_ticks = [0, 25.,50.]

ggap_label = L"\mathrm{g}_{\mathrm{gap}}" * " (nS)"
z_min = -π
z_max = π
cb_scale = 0.8
titlesize = 8
letter_size =  10
init_phase_diff_array_labels = [L"0",L"1/6 \pi",L"1/3 \pi",L"1/2 \pi",L"2/3 \pi",L"5/6 \pi",
                                L"\pi",L"7/6 \pi",L"4/3 \pi",L"3/2 \pi",L"5/3 \pi", L"11/6 \pi",];

ggap_array_log = [-2.5, log10.(ggap_array[2:end])...];
ggap_array_log_ticks = [-2.5, -2, -1, 0, 1];#, 2];
ggap_array_log_labels = ["$(round(10^i, digits=2))" for i in ggap_array_log_ticks] ;
ggap_array_log_labels[2] = ""
ggap_array_log_labels[4] = ""


#%%

save_name =  savename("ggap_synch_2_properties_SUMMARY_CSC_soma_Ih_time_synch_WT", param, "jld2");
phase_diff_array_WT = wload(datadir("simulations", "gapjxn_charac", save_name),  "phase_diff_array");
phase_diff_end_array_WT =  wload(datadir("simulations", "gapjxn_charac", save_name),  "phase_diff_end_array");
spike_times_WT = wload(datadir("simulations", "gapjxn_charac", save_name),  "spike_times");
PLV_array_WT = wload(datadir("simulations", "gapjxn_charac", save_name),  "PLV_array");
Hilbert_phase_diff_array_WT = wload(datadir("simulations", "gapjxn_charac", save_name),  "Hilbert_phase_diff_array");
mean_peak_phase_end_array_WT = wload(datadir("simulations", "gapjxn_charac", save_name),  "mean_peak_phase_end_array");
tau_array_WT = wload(datadir("simulations", "gapjxn_charac", save_name),  "tau_array");

tau_array_WT2 = deepcopy(tau_array_WT)
# tau_array_WT2[tau_array_WT2 .> 5*30000.] .= 0.
# tau_array_WT2[tau_array_WT2 .> 30000.] .= NaN

condition_label = Dict(
    "WT"=> "WT",
    "no_Kd"=> L"$-$ Kdr",
    "no_Ad"=> L"$-$ A", 
    "no_Td"=> L"$-$ T",
    "no_HVAd"=> L"$-$ HVA",
    "no_SKd"=> L"$-$ K(Ca)"
)

condition_col = Dict(
    "WT" => "black",
    "passive" => "tab:grey",
    "no_Kd" => "tab:blue",
    "no_Ad" => "tab:purple",
    "no_SKd" => "tab:green",
    "no_Td" => "tab:orange",
    "no_HVAd" => "tab:cyan",
)


p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_cable.jld2"), "p_fit");
#%

@pyimport matplotlib.cm as cm
@pyimport matplotlib.colors as pltcol
@pyimport colorsys  as colorsys 

cs = ["darkblue", "lightgrey", "purple"]
PLV_cmap_ch = pltcol.LinearSegmentedColormap.from_list("cmap_name", cs)

cs = ["darkgreen", "lightgrey", "darkred"]
t_cmap_ch = pltcol.LinearSegmentedColormap.from_list("cmap_name", cs)

# t_cmap = "viridis"
t_cmap = "magma_r"
t_cmap = "inferno_r"
t_cmap_o = "ocean_r"
original_cmap = plt.cm.get_cmap(t_cmap_o)

t_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "trunc($(t_cmap_o), 0.05, 0.75)",
        original_cmap(range(0.05, stop=0.75, length=1000)))

# plt.truncate_colormap(t_cmap_o, 0.25, 1.0)
PLV_cmap = "twilight_shifted"
param_names = wload(datadir("simulations", "CSC_Ih", "fit_param_cable_names.jld2"),  "param_names")
dist_incr1 = Int(p_fit[findfirst(param_names .== "L")]*10000/50)

pos_label_um = ["Soma", "", ["$(dist_incr1*(i))" for i =[50]]...]



function plot_grids(gs, ax_cb; condition_string="no_Kd", cmap="viridis", vmin= -30,vmax=30, diff=false, plot_array = "tau", cbar_label = "Change in Time to Converged (s)", cbar_label_size=8, cond_size=8)

    save_name =  savename("ggap_synch_2_properties_SUMMARY_CSC_soma_Ih_time_synch_$(condition_string)", param, "jld2")
    
    if condition_string == "WT"
        tau_array_plot_i = tau_array_WT
        phase_diff_end_array_plot_i = phase_diff_end_array_WT 
        mean_peak_phase_end_array_plot_i  = mean_peak_phase_end_array_WT
        PLV_array = PLV_array_WT
    else
        # local phase_diff_array = wload(datadir("simulations", "gapjxn_charac", save_name),  "phase_diff_array")
        local phase_diff_end_array =  wload(datadir("simulations", "gapjxn_charac", save_name),  "phase_diff_end_array")
        # local spike_times = wload(datadir("simulations", "gapjxn_charac", save_name),  "spike_times")
        local PLV_array = wload(datadir("simulations", "gapjxn_charac", save_name),  "PLV_array")
        # local Hilbert_phase_diff_array = wload(datadir("simulations", "gapjxn_charac", save_name),  "Hilbert_phase_diff_array")
        local mean_peak_phase_end_array = wload(datadir("simulations", "gapjxn_charac", save_name),  "mean_peak_phase_end_array")
        local tau_array = wload(datadir("simulations", "gapjxn_charac", save_name),  "tau_array")

        if diff
            tau_array_plot_i = tau_array .- tau_array_WT
            phase_diff_end_array_plot_i = phase_diff_end_array .- phase_diff_end_array_WT
            mean_peak_phase_end_array_plot_i  = mean_peak_phase_end_array .- mean_peak_phase_end_array_WT

        else
            tau_array_plot_i = tau_array 
            phase_diff_end_array_plot_i = phase_diff_end_array 
            mean_peak_phase_end_array_plot_i  = mean_peak_phase_end_array 
        end
    end
    # set 2pi to 0
    mean_peak_phase_end_array_plot_i[mean_peak_phase_end_array_plot_i .>= π]  = mean_peak_phase_end_array_plot_i[mean_peak_phase_end_array_plot_i .>= π] .- (2*π)
    mean_peak_phase_end_array_plot_i[mean_peak_phase_end_array_plot_i .<= -π]  = mean_peak_phase_end_array_plot_i[mean_peak_phase_end_array_plot_i .<= -π] .+ (2*π)

    phase_diff_end_array_plot_i[phase_diff_end_array_plot_i .>= 3*π]  = phase_diff_end_array_plot_i[phase_diff_end_array_plot_i .>= 3*π] .- (4*π)
    phase_diff_end_array_plot_i[phase_diff_end_array_plot_i .>= π]  = phase_diff_end_array_plot_i[phase_diff_end_array_plot_i .>= π] .- (2*π)
    phase_diff_end_array_plot_i[phase_diff_end_array_plot_i .<= -3*π]  = phase_diff_end_array_plot_i[phase_diff_end_array_plot_i .<= -3*π] .+ (4*π)
    phase_diff_end_array_plot_i[phase_diff_end_array_plot_i .<= -π]  = phase_diff_end_array_plot_i[phase_diff_end_array_plot_i .<= -π] .+ (2*π)

    position_array = wload(datadir("simulations", "gapjxn_charac", save_name),  "position_array")
    # # p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_soma_Ih_fixed.jld2"), "p_fit")
    # p_fit = wload(datadir("simulations", "CSC_Ih", "fit_param_soma_Ih_new_final_2.jld2"), "p_fit");
    # param_names = wload(datadir("simulations", "CSC_Ih", "fit_param_names.jld2"),  "param_names")
    # dist_incr1 = Int(p_fit[findfirst(param_names .== "L")]*10000/50)
    
    # pos_label_um = ["Soma", ["$(dist_incr1*(i))" for i =[25, 50]]...]
    ax_overall = fig.add_subplot(py"$(gs)[:,:]")
    ax_overall = pyplot_fxns.remove_axis_box_t(ax_overall; s=["top", "right", "bottom","left"])
    ax_overall.set_xticks([])
    ax_overall.set_yticks([])

    ax_0 = fig.add_subplot(py"$(gs)[1:3,0]")
    ax_1_top = fig.add_subplot(py"$(gs)[0:2,1]")
    ax_1_bot = fig.add_subplot(py"$(gs)[2:4,1]")
    ax_2_top = fig.add_subplot(py"$(gs)[0:2,2]")
    ax_2_bot = fig.add_subplot(py"$(gs)[2:4,2]")
    ax_3_top = fig.add_subplot(py"$(gs)[0:2,3]")
    ax_3_bot = fig.add_subplot(py"$(gs)[2:4,3]")
    ax_4_top = fig.add_subplot(py"$(gs)[0:2,4]")
    ax_4_bot = fig.add_subplot(py"$(gs)[2:4,4]")
    ax_5_top = fig.add_subplot(py"$(gs)[0:2,5]")
    ax_5_bot = fig.add_subplot(py"$(gs)[2:4,5]")
    ax_6 = fig.add_subplot(py"$(gs)[1:3,6]")

    ax_list_t = [ax_0, ax_1_top,ax_2_top,ax_3_top,ax_4_top,ax_5_top,  ax_6, ax_5_bot, ax_4_bot, ax_3_bot, ax_2_bot, ax_1_bot]

    i = 1
    # c0 = ax_0.pcolormesh( position_array, ggap_array_log,  stack(tau_array_plot_i[i, :, :])[1,:,:] ./1000, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true, )
    
    # tau_array_plot_i[tau_array_plot_i .> 5*30000.] .= 0.
    # tau_array_plot_i[tau_array_plot_i .> 30000.] .= NaN
    c0 = ax_0.pcolormesh( position_array, ggap_array_log,  stack(tau_array_plot_i[i, :, :]) ./1000, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true, )
    for ax in ax_list_t
        println(i)
        if plot_array == "tau"
            c = ax.pcolormesh( position_array, ggap_array_log,  stack(tau_array_plot_i[i, :, :]) ./1000, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true, )
        elseif  plot_array == "end_phase"
            c = ax.pcolormesh( position_array, ggap_array_log, phase_diff_end_array_plot_i[i, :, :], cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true, )
        elseif plot_array == "mean_end_phase"
            c = ax.pcolormesh( position_array, ggap_array_log, mean_peak_phase_end_array_plot_i[i, :, :], cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true, )
        elseif plot_array == "PLV"
            c = ax.pcolormesh( position_array, ggap_array_log, PLV_array[i, :, :], cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true, )

        end
        ax.axis([ minimum(position_array),  maximum(position_array), minimum(ggap_array_log), maximum(ggap_array_log)]) # set the limits of the plot to the limits of the data
        ax.set_yticks(ggap_array_log_ticks)
        ax.set_yticklabels(ggap_array_log_labels, size=6)
        ax.set_xticks(pos_label_ticks)
        if i == 1 || i >= 7
            ax.set_xticklabels([])
        else
            ax.set_xticklabels([])
        end
        ax.set_yticks(ggap_array_log_ticks)

        if i == 1
            ax.set_xticklabels(pos_label_um,rotation=90, size=6)
        end
        ax.set_title(init_phase_diff_array_labels[i], size=6, y=0.8)
        if i == 1
            println(i)
            # ax = pyplot_fxns.remove_axis_box_t(ax; s=["top", "right", "bottom","left"], left_ticklabels=true, bottom_ticklabels=true)
            ax = pyplot_fxns.remove_axis_box_t(ax; s=["top", "right",], left_ticklabels=true, bottom_ticklabels=true)
            ax.set_yticklabels(ggap_array_log_labels)
            ax.set_ylabel(ggap_label)
        else
            # ax = pyplot_fxns.remove_axis_box_t(ax; s=["top", "right", "left", "bottom"], bottom_ticklabels=true)
            ax = pyplot_fxns.remove_axis_box_t(ax; s=["top", "right", ], bottom_ticklabels=true)
            ax.set_yticklabels([])
            ax.set_ylabel("")
        end
        i += 1
    end
    if plot_array == "tau"
        cax_t = fig.colorbar(c0,location="right", fraction=1.25,aspect=20, ticks=vmin:20:vmax,label=cbar_label, orientation="vertical", ax=ax_cb, shrink=1.1,extend="max")#[ ax_0_0_t, ax_0_1_t,ax_0_2_t,ax_0_3_t,ax_0_4_t,ax_0_5_t,])#, anchor=(0, 0))
    else

        cax_t = fig.colorbar(c0,location="right", fraction=1.25,aspect=20, ticks=vmin:10:vmax,label=cbar_label, orientation="vertical", ax=ax_cb, shrink=1.1)#
        if condition_string != "WT"
            ax_overall.set_title(condition_label[condition_string], size=cond_size, color=condition_col[condition_string], y=1.1)
        end
    end
    return ax_0, ax_1_top, ax_1_bot, ax_2_top, ax_2_bot, ax_3_top, ax_3_bot, ax_4_top, ax_4_bot, ax_5_top, ax_5_bot, ax_6, cax_t, ax_cb
end

#% TIME  #############################################################################################################
fig = plt.figure(figsize=(6.9, 6.9))


rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8 

rcParams["xtick.major.size"] = 2
rcParams["xtick.major.pad"] = 1.5
rcParams["ytick.major.size"] = 2
rcParams["ytick.major.pad"] = 1.5
bpad = 0.1

fig.subplots_adjust(left=0.075, bottom=0.05, right=0.975, top=0.95)#, wspace=0, hspace=0)
gs_all  = fig.add_gridspec(3, 5, height_ratios=[1,1,1],width_ratios=[1,0.05,0.25, 1, 0.05], wspace=0.02, hspace=0.3)
gs_all_0  =  gridspec.GridSpecFromSubplotSpec(2, 2, wspace=0.25, hspace=0.5, subplot_spec=py"$(gs_all)[0,0]", width_ratios = (1, 0.1))
gs_all_1  =  gridspec.GridSpecFromSubplotSpec(2, 2, wspace=0.25, hspace=0.5, subplot_spec=py"$(gs_all)[1,0]", width_ratios = (1, 0.1))
gs_all_2  =  gridspec.GridSpecFromSubplotSpec(2, 2, wspace=0.25, hspace=0.5, subplot_spec=py"$(gs_all)[2,0]", width_ratios = (1, 0.1)) 
gs_all_3  =  gridspec.GridSpecFromSubplotSpec(2, 2, wspace=0.25, hspace=0.5, subplot_spec=py"$(gs_all)[0,3]", width_ratios = (1, 0.1)) 
gs_all_4  =  gridspec.GridSpecFromSubplotSpec(2, 2, wspace=0.25, hspace=0.5, subplot_spec=py"$(gs_all)[1,3]", width_ratios = (1, 0.1)) 
gs_all_5  =  gridspec.GridSpecFromSubplotSpec(2, 2, wspace=0.25, hspace=0.5, subplot_spec=py"$(gs_all)[2,3]", width_ratios = (1, 0.1)) 



gs_0  =  gridspec.GridSpecFromSubplotSpec(4, 7, wspace=0.75, hspace=5, subplot_spec=py"$(gs_all_0)[0,0]")
gs_1  =  gridspec.GridSpecFromSubplotSpec(4, 7, wspace=0.75, hspace=5, subplot_spec=py"$(gs_all_1)[0,0]")
gs_2  =  gridspec.GridSpecFromSubplotSpec(4, 7, wspace=0.75, hspace=5, subplot_spec=py"$(gs_all_2)[0,0]") 
gs_3  =  gridspec.GridSpecFromSubplotSpec(4, 7, wspace=0.75, hspace=5, subplot_spec=py"$(gs_all_3)[0,0]") 
gs_4  =  gridspec.GridSpecFromSubplotSpec(4, 7, wspace=0.75, hspace=5, subplot_spec=py"$(gs_all_4)[0,0]") 
gs_5  =  gridspec.GridSpecFromSubplotSpec(4, 7, wspace=0.75, hspace=5, subplot_spec=py"$(gs_all_5)[0,0]") 


gs_0_1  =  gridspec.GridSpecFromSubplotSpec(4, 7, wspace=0.75, hspace=5, subplot_spec=py"$(gs_all_0)[1,0]")
gs_1_1  =  gridspec.GridSpecFromSubplotSpec(4, 7, wspace=0.75, hspace=5, subplot_spec=py"$(gs_all_1)[1,0]")
gs_2_1  =  gridspec.GridSpecFromSubplotSpec(4, 7, wspace=0.75, hspace=5, subplot_spec=py"$(gs_all_2)[1,0]") 
gs_3_1  =  gridspec.GridSpecFromSubplotSpec(4, 7, wspace=0.75, hspace=5, subplot_spec=py"$(gs_all_3)[1,0]") 
gs_4_1  =  gridspec.GridSpecFromSubplotSpec(4, 7, wspace=0.75, hspace=5, subplot_spec=py"$(gs_all_4)[1,0]") 
gs_5_1  =  gridspec.GridSpecFromSubplotSpec(4, 7, wspace=0.75, hspace=5, subplot_spec=py"$(gs_all_5)[1,0]") 

# colorbars ##
ax_cb0 = fig.add_subplot(py"$(gs_all_0)[0,1]")
ax_cb0 = pyplot_fxns.remove_axis_box_t(ax_cb0; s=["top", "right", "left", "bottom"])
ax_cb0.set_xticks([])
ax_cb0.set_yticks([])

ax_cb1 = fig.add_subplot(py"$(gs_all_1)[0,1]")
ax_cb1 = pyplot_fxns.remove_axis_box_t(ax_cb1; s=["top", "right", "left", "bottom"])
ax_cb1.set_xticks([])
ax_cb1.set_yticks([])

ax_cb2 = fig.add_subplot(py"$(gs_all_2)[0, 1]")
ax_cb2 = pyplot_fxns.remove_axis_box_t(ax_cb2; s=["top", "right", "left", "bottom"])
ax_cb2.set_xticks([])
ax_cb2.set_yticks([])

ax_cb3 = fig.add_subplot(py"$(gs_all_3)[0, 1]")
ax_cb3 = pyplot_fxns.remove_axis_box_t(ax_cb3; s=["top", "right", "left", "bottom"])
ax_cb3.set_xticks([])
ax_cb3.set_yticks([])

ax_cb4 = fig.add_subplot(py"$(gs_all_4)[0, 1]")
ax_cb4 = pyplot_fxns.remove_axis_box_t(ax_cb4; s=["top", "right", "left", "bottom"])
ax_cb4.set_xticks([])
ax_cb4.set_yticks([])

ax_cb5 = fig.add_subplot(py"$(gs_all_5)[0, 1]")
ax_cb5 = pyplot_fxns.remove_axis_box_t(ax_cb5; s=["top", "right", "left", "bottom"])
ax_cb5.set_xticks([])
ax_cb5.set_yticks([])



# colorbars ##
ax_cb0_1 = fig.add_subplot(py"$(gs_all_0)[1, 1]")
ax_cb0_1 = pyplot_fxns.remove_axis_box_t(ax_cb0_1; s=["top", "right", "left", "bottom"])
ax_cb0_1.set_xticks([])
ax_cb0_1.set_yticks([])

ax_cb1_1 = fig.add_subplot(py"$(gs_all_1)[1,1]")
ax_cb1_1 = pyplot_fxns.remove_axis_box_t(ax_cb1_1; s=["top", "right", "left", "bottom"])
ax_cb1_1.set_xticks([])
ax_cb1_1.set_yticks([])

ax_cb2_1 = fig.add_subplot(py"$(gs_all_2)[1,1]")
ax_cb2_1 = pyplot_fxns.remove_axis_box_t(ax_cb2_1; s=["top", "right", "left", "bottom"])
ax_cb2_1.set_xticks([])
ax_cb2_1.set_yticks([])

ax_cb3_1 = fig.add_subplot(py"$(gs_all_3)[1,1]")
ax_cb3_1 = pyplot_fxns.remove_axis_box_t(ax_cb3_1; s=["top", "right", "left", "bottom"])
ax_cb3_1.set_xticks([])
ax_cb3_1.set_yticks([])

ax_cb4_1 = fig.add_subplot(py"$(gs_all_4)[1,1]")
ax_cb4_1 = pyplot_fxns.remove_axis_box_t(ax_cb4_1; s=["top", "right", "left", "bottom"])
ax_cb4_1.set_xticks([])
ax_cb4_1.set_yticks([])

ax_cb5_1 = fig.add_subplot(py"$(gs_all_5)[1,1]")
ax_cb5_1 = pyplot_fxns.remove_axis_box_t(ax_cb5_1; s=["top", "right", "left", "bottom"])
ax_cb5_1.set_xticks([])
ax_cb5_1.set_yticks([])



ax_0_0, ax_0_1_top, ax_0_1_bot, ax_0_2_top, ax_0_2_bot, ax_0_3_top, ax_0_3_bot, ax_0_4_top, ax_0_4_bot, ax_0_5_top, ax_0_5_bot, ax_0_6, ax_cb0 = plot_grids(gs_0, ax_cb0; condition_string="WT", cmap=PLV_cmap, vmin=-π, vmax=π, cbar_label = L"\Delta \Phi"*" at 30s", plot_array = "end_phase")
ax_1_0, ax_1_1_top, ax_1_1_bot, ax_1_2_top, ax_1_2_bot, ax_1_3_top, ax_1_3_bot, ax_1_4_top, ax_1_4_bot, ax_1_5_top, ax_1_5_bot, ax_1_6, ax_cb1 = plot_grids(gs_1, ax_cb1; condition_string="no_Ad", cmap=PLV_cmap, vmin=-π, vmax=π, cbar_label = L"\Delta \Phi"*" at 30s", plot_array = "end_phase")
ax_2_0, ax_2_1_top, ax_2_1_bot, ax_2_2_top, ax_2_2_bot, ax_2_3_top, ax_2_3_bot, ax_2_4_top, ax_2_4_bot, ax_2_5_top, ax_2_5_bot, ax_2_6, ax_cb2 = plot_grids(gs_2, ax_cb2; condition_string="no_HVAd", cmap=PLV_cmap, vmin=-π, vmax=π, cbar_label = L"\Delta \Phi"*" at 30s", plot_array = "end_phase")
ax_3_0, ax_3_1_top, ax_3_1_bot, ax_3_2_top, ax_3_2_bot, ax_3_3_top, ax_3_3_bot, ax_3_4_top, ax_3_4_bot, ax_3_5_top, ax_3_5_bot, ax_3_6, ax_cb3 = plot_grids(gs_3, ax_cb3; condition_string="no_Kd", cmap=PLV_cmap, vmin=-π, vmax=π, cbar_label = L"\Delta \Phi"*" at 30s", plot_array = "end_phase")
ax_4_0, ax_4_1_top, ax_4_1_bot, ax_4_2_top, ax_4_2_bot, ax_4_3_top, ax_4_3_bot, ax_4_4_top, ax_4_4_bot, ax_4_5_top, ax_4_5_bot, ax_4_6, ax_cb4 = plot_grids(gs_4, ax_cb4; condition_string="no_Td", cmap=PLV_cmap, vmin=-π, vmax=π, cbar_label = L"\Delta \Phi"*" at 30s", plot_array = "end_phase")
ax_5_0, ax_5_1_top, ax_5_1_bot, ax_5_2_top, ax_5_2_bot, ax_5_3_top, ax_5_3_bot, ax_5_4_top, ax_5_4_bot, ax_5_5_top, ax_5_5_bot, ax_5_6, ax_cb5 = plot_grids(gs_5, ax_cb5; condition_string="no_SKd", cmap=PLV_cmap, vmin=-π, vmax=π, cbar_label = L"\Delta \Phi"*" at 30s", plot_array = "end_phase")


ax_cb0.ax.set_yticks([-π, 0, π], ["-π", "0", "π"])
ax_cb1.ax.set_yticks([-π, 0, π], ["-π", "0", "π"])
ax_cb2.ax.set_yticks([-π, 0, π], ["-π", "0", "π"])
ax_cb3.ax.set_yticks([-π, 0, π], ["-π", "0", "π"])
ax_cb4.ax.set_yticks([-π, 0, π], ["-π", "0", "π"])
ax_cb5.ax.set_yticks([-π, 0, π], ["-π", "0", "π"])

tau_max = 60
ax_0_0_1, ax_0_1_top_1, ax_0_1_bot_1, ax_0_2_top_1, ax_0_2_bot_1, ax_0_3_top_1, ax_0_3_bot_1, ax_0_4_top_1, ax_0_4_bot_1, ax_0_5_top_1, ax_0_5_bot_1, ax_0_6_1, ax_cb0_1 = plot_grids(gs_0_1, ax_cb0_1; condition_string="WT", cmap=t_cmap, vmin=0, vmax=tau_max, cbar_label = "Synch. " *L"$\tau$" * " (s)")
ax_1_0_1, ax_1_1_top_1, ax_1_1_bot_1, ax_1_2_top_1, ax_1_2_bot_1, ax_1_3_top_1, ax_1_3_bot_1, ax_1_4_top_1, ax_1_4_bot_1, ax_1_5_top_1, ax_1_5_bot_1, ax_1_6_1, ax_cb1_1 = plot_grids(gs_1_1, ax_cb1_1; condition_string="no_Ad", cmap=t_cmap, vmin=0, vmax=tau_max, cbar_label = "Synch. " *L"$\tau$" * " (s)")
ax_2_0_1, ax_2_1_top_1, ax_2_1_bot_1, ax_2_2_top_1, ax_2_2_bot_1, ax_2_3_top_1, ax_2_3_bot_1, ax_2_4_top_1, ax_2_4_bot_1, ax_2_5_top_1, ax_2_5_bot_1, ax_2_6_1, ax_cb2_1 = plot_grids(gs_2_1, ax_cb2_1; condition_string="no_HVAd", cmap=t_cmap, vmin=0, vmax=tau_max, cbar_label = "Synch. " *L"$\tau$" * " (s)")
ax_3_0_1, ax_3_1_top_1, ax_3_1_bot_1, ax_3_2_top_1, ax_3_2_bot_1, ax_3_3_top_1, ax_3_3_bot_1, ax_3_4_top_1, ax_3_4_bot_1, ax_3_5_top_1, ax_3_5_bot_1, ax_3_6_1, ax_cb3_1 = plot_grids(gs_3_1, ax_cb3_1; condition_string="no_Kd", cmap=t_cmap, vmin=0, vmax=tau_max, cbar_label = "Synch. " *L"$\tau$" * " (s)")
ax_4_0_1, ax_4_1_top_1, ax_4_1_bot_1, ax_4_2_top_1, ax_4_2_bot_1, ax_4_3_top_1, ax_4_3_bot_1, ax_4_4_top_1, ax_4_4_bot_1, ax_4_5_top_1, ax_4_5_bot_1, ax_4_6_1, ax_cb4_1 = plot_grids(gs_4_1, ax_cb4_1; condition_string="no_Td", cmap=t_cmap, vmin=0, vmax=tau_max, cbar_label = "Synch. " *L"$\tau$" * " (s)")
ax_5_0_1, ax_5_1_top_1, ax_5_1_bot_1, ax_5_2_top_1, ax_5_2_bot_1, ax_5_3_top_1, ax_5_3_bot_1, ax_5_4_top_1, ax_5_4_bot_1, ax_5_5_top_1, ax_5_5_bot_1, ax_5_6_1, ax_cb5_1 = plot_grids(gs_5_1, ax_cb5_1; condition_string="no_SKd", cmap=t_cmap, vmin=0, vmax=tau_max, cbar_label = "Synch. " *L"$\tau$" * " (s)")



letters = collect('A':'Z')
ax_0_0.text(-2.5, 2.35, "$(letters[1])", transform=ax_0_0.transAxes, size=letter_size, weight="bold")
ax_3_0.text(-2.5, 2.35, "$(letters[2])", transform=ax_3_0.transAxes, size=letter_size, weight="bold")
ax_1_0.text(-2.5, 2.35, "$(letters[3])", transform=ax_1_0.transAxes, size=letter_size, weight="bold")
ax_4_0.text(-2.5, 2.35, "$(letters[4])", transform=ax_4_0.transAxes, size=letter_size, weight="bold")
ax_2_0.text(-2.5, 2.35, "$(letters[5])", transform=ax_2_0.transAxes, size=letter_size, weight="bold")
ax_5_0.text(-2.5, 2.35, "$(letters[6])", transform=ax_5_0.transAxes, size=letter_size, weight="bold")


# plt.savefig(plotsdir("figs", "Figure_4_5_gapjxns_2_phase_tau_$(tau_p).png"), dpi=600)
# plt.savefig(plotsdir("figs", "Figure_4_5_gapjxns_2_phase_tau_$(tau_p).pdf"), dpi=600)
# plt.savefig(plotsdir("figs", "Figure_4_5_gapjxns_2_phase_tau_$(tau_p).eps"), dpi=600)
plt.savefig(plotsdir("Figure_4_gapjxns_2_phase_tau.png"), dpi=600)
plt.savefig(plotsdir("Figure_4_gapjxns_2_phase_tau.pdf"), dpi=600)
plt.savefig(plotsdir("Figure_4_gapjxns_2_phase_tau.eps"), dpi=600)
plt.show()
# plt.close()

#%%