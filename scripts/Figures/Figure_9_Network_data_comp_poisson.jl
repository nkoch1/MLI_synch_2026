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
using DataFrames, CSV
#%%

df = CSV.read(datadir("exp_pro", "Brown_2025_Figure_5B_interp.csv"), DataFrame)
peak_coact_steps = [10, 20, 30, 40, 50, 60, 70, 80]

df_sum = CSV.read(datadir("exp_pro", "Brown_2025_Figure_5B_summary.csv"), DataFrame)


#%%

λ_scale_array = [100.,   250.,  400.,   550.,  700., 850., 1000. ]
NMDA_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


Ihold_scale = 0.2
ggap = 0.6
gsyn = 1.0

fname = "Network_realizations_dense_scaled_336.jld2"

ncells = wload(datadir("simulations", "Network", fname),  "N")
x_pos = wload(datadir("simulations", "Network", fname),  "x_pos");
num_surfaces = size(x_pos)[1]

tspan = (0.0, 1. * 1000.0)
bin_size = 2
bins = collect(tspan[1]:bin_size:tspan[end]);
bins_mid = [bins[i] + ((bins[i+1] - bins[i])/2) for i in eachindex(bins[1:end-1])];
bin1_size = 1
bins1 = collect(tspan[1]:bin1_size:tspan[end]);
bins1_mid = [bins1[i] + ((bins1[i+1] - bins1[i])/2) for i in eachindex(bins1[1:end-1])];
gAMPA = 1.2
# gsyn = 1.0
# ggap = 0.6 *4
syn_delay = 5
synch_dur = 2.0
tau_GABA = 1.9

fname_save = "Network_CSC1_NMDA"

param_1 = Dict(
        "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
        "ggap" => ggap, # use input value in nS not calculated uA/cm^2
        "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
        "Ihold_scale"=> Ihold_scale,
        "NMDA_scale" =>NMDA_scale_array[1],
        "rate_synch" => λ_scale_array[1],
        "synch_dur" => synch_dur,
        "tau_GABA" => tau_GABA,
    )
peak_coactivity_array_1 = wload(datadir("simulations", "Network", savename(fname_save, param_1, "jld2")), "peak_coactivity_array");
additional_peaks_array_1 = wload(datadir("simulations", "Network", savename(fname_save, param_1, "jld2")), "additional_peaks");
coactivity_array_1 = wload(datadir("simulations", "Network", savename(fname_save, param_1, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", savename(fname_save, param_1, "jld2")), "ncells")
solu_1 = wload(datadir("simulations", "Network", savename(fname_save, param_1, "jld2")), "solu");
sol_t_1 = wload(datadir("simulations", "Network", savename(fname_save, param_1, "jld2")), "solt");
sol_u_1 = reduce(hcat, solu_1)


param_2 = Dict(
        "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
        "ggap" => ggap, # use input value in nS not calculated uA/cm^2
        "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
        "Ihold_scale"=> Ihold_scale,
        "NMDA_scale" =>NMDA_scale_array[2],
        "rate_synch" => λ_scale_array[2],
        "synch_dur" => synch_dur,
        "tau_GABA" => tau_GABA,
    )
peak_coactivity_array_2 = wload(datadir("simulations", "Network", savename(fname_save, param_2, "jld2")), "peak_coactivity_array");
additional_peaks_array_2 = wload(datadir("simulations", "Network", savename(fname_save, param_2, "jld2")), "additional_peaks");
coactivity_array_2 = wload(datadir("simulations", "Network", savename(fname_save, param_2, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", savename(fname_save, param_2, "jld2")), "ncells")
solu_2 = wload(datadir("simulations", "Network", savename(fname_save, param_2, "jld2")), "solu");
sol_t_2 = wload(datadir("simulations", "Network", savename(fname_save, param_2, "jld2")), "solt");
sol_u_2 = reduce(hcat, solu_2)


param_3 = Dict(
        "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
        "ggap" => ggap, # use input value in nS not calculated uA/cm^2
        "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
        "Ihold_scale"=> Ihold_scale,
        "NMDA_scale" =>NMDA_scale_array[3],
        "rate_synch" => λ_scale_array[3],
        "synch_dur" => synch_dur,
        "tau_GABA" => tau_GABA,
    )
peak_coactivity_array_3 = wload(datadir("simulations", "Network", savename(fname_save, param_3, "jld2")), "peak_coactivity_array");
additional_peaks_array_3 = wload(datadir("simulations", "Network", savename(fname_save, param_3, "jld2")), "additional_peaks");
coactivity_array_3 = wload(datadir("simulations", "Network", savename(fname_save, param_3, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", savename(fname_save, param_3, "jld2")), "ncells")
solu_3 = wload(datadir("simulations", "Network", savename(fname_save, param_3, "jld2")), "solu");
sol_t_3 = wload(datadir("simulations", "Network", savename(fname_save, param_3, "jld2")), "solt");
sol_u_3 = reduce(hcat, solu_3)




param_4 = Dict(
        "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
        "ggap" => ggap, # use input value in nS not calculated uA/cm^2
        "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
        "Ihold_scale"=> Ihold_scale,
        "NMDA_scale" =>NMDA_scale_array[4],
        "rate_synch" => λ_scale_array[4],
        "synch_dur" => synch_dur,
        "tau_GABA" => tau_GABA,
    )
peak_coactivity_array_4 = wload(datadir("simulations", "Network", savename(fname_save, param_4, "jld2")), "peak_coactivity_array");
additional_peaks_array_4 = wload(datadir("simulations", "Network", savename(fname_save, param_4, "jld2")), "additional_peaks");
coactivity_array_4 = wload(datadir("simulations", "Network", savename(fname_save, param_4, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", savename(fname_save, param_4, "jld2")), "ncells")
solu_4 = wload(datadir("simulations", "Network", savename(fname_save, param_4, "jld2")), "solu");
sol_t_4 = wload(datadir("simulations", "Network", savename(fname_save, param_4, "jld2")), "solt");
sol_u_4 = reduce(hcat, solu_4)


param_5 = Dict(
        "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
        "ggap" => ggap, # use input value in nS not calculated uA/cm^2
        "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
        "Ihold_scale"=> Ihold_scale,
        "NMDA_scale" =>NMDA_scale_array[5],
        "rate_synch" => λ_scale_array[5],
        "synch_dur" => synch_dur,
        "tau_GABA" => tau_GABA,
    )
peak_coactivity_array_5 = wload(datadir("simulations", "Network", savename(fname_save, param_5, "jld2")), "peak_coactivity_array");
additional_peaks_array_5 = wload(datadir("simulations", "Network", savename(fname_save, param_5, "jld2")), "additional_peaks");
coactivity_array_5 = wload(datadir("simulations", "Network", savename(fname_save, param_5, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", savename(fname_save, param_5, "jld2")), "ncells")
solu_5 = wload(datadir("simulations", "Network", savename(fname_save, param_5, "jld2")), "solu");
sol_t_5 = wload(datadir("simulations", "Network", savename(fname_save, param_5, "jld2")), "solt");
sol_u_5 = reduce(hcat, solu_5)



param_6 = Dict(
        "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
        "ggap" => ggap, # use input value in nS not calculated uA/cm^2
        "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
        "Ihold_scale"=> Ihold_scale,
        "NMDA_scale" =>NMDA_scale_array[6],
        "rate_synch" => λ_scale_array[6],
        "synch_dur" => synch_dur,
        "tau_GABA" => tau_GABA,
    )
peak_coactivity_array_6 = wload(datadir("simulations", "Network", savename(fname_save, param_6, "jld2")), "peak_coactivity_array");
additional_peaks_array_6 = wload(datadir("simulations", "Network", savename(fname_save, param_6, "jld2")), "additional_peaks");
coactivity_array_6 = wload(datadir("simulations", "Network", savename(fname_save, param_6, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", savename(fname_save, param_6, "jld2")), "ncells")
solu_6 = wload(datadir("simulations", "Network", savename(fname_save, param_6, "jld2")), "solu");
sol_t_6 = wload(datadir("simulations", "Network", savename(fname_save, param_6, "jld2")), "solt");
sol_u_6 = reduce(hcat, solu_6)


param_7 = Dict(
        "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
        "ggap" => ggap, # use input value in nS not calculated uA/cm^2
        "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
        "Ihold_scale"=> Ihold_scale,
        "NMDA_scale" =>NMDA_scale_array[7],
        "rate_synch" => λ_scale_array[7],
        "synch_dur" => synch_dur,
        "tau_GABA" => tau_GABA,
    )
peak_coactivity_array_7 = wload(datadir("simulations", "Network", savename(fname_save, param_7, "jld2")), "peak_coactivity_array");
additional_peaks_array_7 = wload(datadir("simulations", "Network", savename(fname_save, param_7, "jld2")), "additional_peaks");
coactivity_array_7 = wload(datadir("simulations", "Network", savename(fname_save, param_7, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", savename(fname_save, param_7, "jld2")), "ncells")
solu_7 = wload(datadir("simulations", "Network", savename(fname_save, param_7, "jld2")), "solu");
sol_t_7 = wload(datadir("simulations", "Network", savename(fname_save, param_7, "jld2")), "solt");
sol_u_7 = reduce(hcat, solu_7)
#%% FIGURE ######################################################
using LaTeXStrings
@pyimport matplotlib.gridspec as gridspec
@pyimport matplotlib.patches as patches 
@pyimport seaborn as sns

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8 
c_size = 2.5



custom_colors_Data = ["tab:grey", "tab:red"]
data_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("nspikes", custom_colors_Data, N = 100) # Create a ListedColormap using the custom colors

cmap = PyPlot.matplotlib.colormaps.get_cmap(data_cmap)
N = length(peak_coact_steps)
gradient = LinRange(0, 1, N)
n_colors_data = [matplotlib.colors.to_hex(cmap(i)) for i in gradient]

cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#595959", "orchid"])
N = length(λ_scale_array)
gradient = LinRange(0, 1, N)
n_colors_model = [matplotlib.colors.to_hex(cmap2(i)) for i in gradient]


tlim = (-20, 100)
coact_lim  = (0, 90)

title_size = 8
letter_size = 10
x_letter = -0.05
y_letter = 1.1

m_size = 2
letters = collect('A':'Z')

# Fig Setup
fig = plt.figure(figsize=(6.9, 5))
gs_all  = fig.add_gridspec(1, 2,left=0.1, right=0.975, wspace=0.35, hspace=0.5,top=0.95, bottom=0.075)
gs_left =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.25, hspace=0.5, subplot_spec=py"$(gs_all)[0]") 
gs_right  =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.25, hspace=0.5, subplot_spec=py"$(gs_all)[1]")

ax_data_coact = fig.add_subplot(py"$(gs_left)[0]")
ax_data_coact = pyplot_fxns.remove_axis_box(ax_data_coact; s=["top", "right"])

ax_model_coact = fig.add_subplot(py"$(gs_left)[1]")
ax_model_coact = pyplot_fxns.remove_axis_box(ax_model_coact; s=["top", "right"])


ax_peaks_coact = fig.add_subplot(py"$(gs_right)[0]")
ax_peaks_coact = pyplot_fxns.remove_axis_box(ax_peaks_coact; s=["top", "right"])

ax_peaks_coact_model =  fig.add_subplot(py"$(gs_right)[1]")
ax_peaks_coact_model = pyplot_fxns.remove_axis_box(ax_peaks_coact_model; s=["top", "right"])

i = 1
for lab in peak_coact_steps
    ax_data_coact.plot(df[!, "time"] .- 4, df[!, "$(lab)"], color = n_colors_data[i])
    i += 1
end

ax_model_coact.plot(bins1_mid .- 500, mean(coactivity_array_1, dims=1)[1, :],  color=n_colors_model[1])
ax_model_coact.plot(bins1_mid .- 500, mean(coactivity_array_2, dims=1)[1, :], color=n_colors_model[2])
ax_model_coact.plot(bins1_mid .- 500, mean(coactivity_array_3, dims=1)[1, :],  color=n_colors_model[3])
ax_model_coact.plot(bins1_mid .- 500, mean(coactivity_array_4, dims=1)[1, :],  color=n_colors_model[4])
ax_model_coact.plot(bins1_mid .- 500, mean(coactivity_array_5, dims=1)[1, :],  color=n_colors_model[5])
ax_model_coact.plot(bins1_mid .- 500, mean(coactivity_array_6, dims=1)[1, :],  color=n_colors_model[6])
ax_model_coact.plot(bins1_mid .- 500, mean(coactivity_array_7, dims=1)[1, :],  color=n_colors_model[7])




ax_peaks_coact_model.errorbar(mean(peak_coactivity_array_1), mean(additional_peaks_array_1), 
                                xerr=std(peak_coactivity_array_1), yerr=std( additional_peaks_array_1), c=n_colors_model[1], capsize=c_size,marker="o", markersize=5)#, s=1)
ax_peaks_coact_model.errorbar(mean(peak_coactivity_array_2), mean(additional_peaks_array_2),
                                xerr=std(peak_coactivity_array_2), yerr=std( additional_peaks_array_2), c=n_colors_model[2], capsize=c_size,marker="o", markersize=5)#, s=1)
ax_peaks_coact_model.errorbar(mean(peak_coactivity_array_3), mean(additional_peaks_array_3), 
                                xerr=std(peak_coactivity_array_3), yerr=std( additional_peaks_array_3), c=n_colors_model[3], capsize=c_size,marker="o", markersize=5)#, s=1)
ax_peaks_coact_model.errorbar(mean(peak_coactivity_array_4), mean(additional_peaks_array_4), 
                                xerr=std(peak_coactivity_array_4), yerr=std( additional_peaks_array_4), c=n_colors_model[4], capsize=c_size,marker="o", markersize=5)#, s=1)
ax_peaks_coact_model.errorbar(mean(peak_coactivity_array_5), mean(additional_peaks_array_5), 
                                xerr=std(peak_coactivity_array_5), yerr=std( additional_peaks_array_5), c=n_colors_model[5], capsize=c_size,marker="o", markersize=5)#, s=1)
ax_peaks_coact_model.errorbar(mean(peak_coactivity_array_6), mean(additional_peaks_array_6), 
                                xerr=std(peak_coactivity_array_6), yerr=std( additional_peaks_array_6), c=n_colors_model[6], capsize=c_size,marker="o", markersize=5)#, s=1)
ax_peaks_coact_model.errorbar(mean(peak_coactivity_array_7), mean(additional_peaks_array_7), 
                                xerr=std(peak_coactivity_array_7), yerr=std( additional_peaks_array_7), c=n_colors_model[7], capsize=c_size,marker="o", markersize=5)#, s=1)


ax_peaks_coact.scatter(df_sum[!, "peak_coactivity"], df_sum[!, "num_peaks"], c=n_colors_data)


for ax in [ax_data_coact, ax_model_coact]
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Coactivity (%)")
    ax.set_xlim(tlim)
    ax.set_ylim(coact_lim)
end

for ax in [ax_peaks_coact, ax_peaks_coact_model]
    ax.set_xlabel("Peak coactivity (%)")
    ax.set_ylabel("Number of peaks")
    ax.set_xlim(0, 95)
    ax.set_ylim(0, 4)
end

ax_data_coact.text(-0.05, 1.05, "$(letters[1])", transform=ax_data_coact.transAxes, size=letter_size, weight="bold")
ax_model_coact.text(-0.05, 1.05, "$(letters[3])", transform=ax_model_coact.transAxes, size=letter_size, weight="bold")
ax_peaks_coact.text(-0.05, 1.05, "$(letters[2])", transform=ax_peaks_coact.transAxes, size=letter_size, weight="bold")
ax_peaks_coact_model.text(-0.05, 1.05, "$(letters[4])", transform=ax_peaks_coact_model.transAxes, size=letter_size, weight="bold")

plt.savefig(plotsdir("Figure_9_Network_data_comp_poisson.png"), dpi=600)
plt.savefig(plotsdir("Figure_9_Network_data_comp_poisson.pdf"), dpi=600)
plt.savefig(plotsdir("Figure_9_Network_data_comp_poisson.eps"), dpi=600)
plt.show()