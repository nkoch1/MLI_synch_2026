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

#%% LOAD 

fname = "Network_realizations_dense_scaled_336.jld2"

ncells = wload(datadir("simulations", "Network",  fname),  "N")
x_pos = wload(datadir("simulations", "Network", fname),  "x_pos");
num_surfaces = size(x_pos)[1]

tspan = (0.0, 1. * 1000.0)
bin_size = 2
bins = collect(tspan[1]:bin_size:tspan[end]);
bins_mid = [bins[i] + ((bins[i+1] - bins[i])/2) for i in eachindex(bins[1:end-1])];
bin_size1 = 1
bins1 = collect(tspan[1]:bin_size1:tspan[end]);
bins1_mid = [bins1[i] + ((bins1[i+1] - bins1[i])/2) for i in eachindex(bins1[1:end-1])];
ggap = 0.6
gsyn = 1.0
gAMPA = 1.2
Ihold_scale = 0.2
syn_delay = 5
NMDA_scale = 1.0
synch_dur = 2.0
tau_GABA = 1.9

param_save = Dict(
    "I" => 0.0, 
    "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
    "ggap" => ggap, # use input value in nS not calculated uA/cm^2
    "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
    "synch_dur" => synch_dur,
    "tau_GABA" => tau_GABA,
)


# fname_save = "Network_CSC1"

fname_save = "Network_CSC1_nsynch_array_poisson"
peak_coactivity_array = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save, "jld2")), "peak_coactivity_array");
peak_coactivity_array_mean = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save, "jld2")), "peak_coactivity_array_mean");
PLV_before_array = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save, "jld2")), "PLV_before_array");
PLV_before_array_mean = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save, "jld2")), "PLV_mean_before_array");
PLV_after_array = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save, "jld2")), "PLV_after_50_array");
PLV_after_array_mean = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save, "jld2")), "PLV_mean_after_50_array");
λ_rate_synch = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save, "jld2")), "n_array");
NMDA_array = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save, "jld2")), "NMDA_array");

delta_PLV_array = PLV_after_array .- PLV_before_array
delta_PLV_array_mean = PLV_after_array_mean .- PLV_before_array_mean

gNMDA_array = round.(NMDA_array .* gAMPA, digits=1)


# n_ind = [1,4,5]
# NMDA_ind = [1,4,5]
n_ind = [1,5]
NMDA_ind = [2,5]
n_ind_norm = findfirst(λ_rate_synch .== 600.)
NMDA_ind_norm = findfirst(NMDA_array .== 1.5)


# fname_save = "Network_CSC1_NMDA"

fname_save = "Network_CSC1_NMDA_CC_poisson"
param_0_0 = Dict(
    "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
    "ggap" => ggap, # use input value in nS not calculated uA/cm^2
    "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
    "Ihold_scale"=> Ihold_scale,
    "NMDA_scale" =>NMDA_array[NMDA_ind[1]],
    "rate_synch" => λ_rate_synch[n_ind_norm],
    "synch_dur" => synch_dur,
    "tau_GABA" => tau_GABA,
)
peak_coactivity_array_0_0 = wload(datadir("simulations", "Network", "new",  savename(fname_save, param_0_0, "jld2")), "peak_coactivity_array");
psth_density_array_0_0 = wload(datadir("simulations", "Network", "new",  savename(fname_save, param_0_0, "jld2")), "psth_density_array");
coactivity_array_0_0 = wload(datadir("simulations", "Network", "new",  savename(fname_save, param_0_0, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", "new",  savename(fname_save, param_0_0, "jld2")), "ncells")
solu_0_0 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_0_0, "jld2")), "solu");
sol_t_0_0 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_0_0, "jld2")), "solt");
sol_u_0_0 = reduce(hcat, solu_0_0)


param_0_1 = Dict(
    "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
    "ggap" => ggap, # use input value in nS not calculated uA/cm^2
    "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
    "Ihold_scale"=> Ihold_scale,
    "NMDA_scale" =>NMDA_array[NMDA_ind[2]],
    "rate_synch" => λ_rate_synch[n_ind_norm],
    "synch_dur" => synch_dur,
    "tau_GABA" => tau_GABA,
)
peak_coactivity_array_0_1 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_0_1, "jld2")), "peak_coactivity_array");
psth_density_array_0_1 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_0_1, "jld2")), "psth_density_array");
coactivity_array_0_1 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_0_1, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", "new", savename(fname_save, param_0_1, "jld2")), "ncells")
solu_0_1 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_0_1, "jld2")), "solu");
sol_t_0_1 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_0_1, "jld2")), "solt");
sol_u_0_1 = reduce(hcat, solu_0_1)




param_1_0 = Dict(
    "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
    "ggap" => ggap, # use input value in nS not calculated uA/cm^2
    "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
    "Ihold_scale"=> Ihold_scale,
    "NMDA_scale" =>NMDA_array[NMDA_ind_norm],
    "rate_synch" => λ_rate_synch[n_ind[1]],
    "synch_dur" => synch_dur,
    "tau_GABA" => tau_GABA,
)
peak_coactivity_array_1_0 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_1_0, "jld2")), "peak_coactivity_array");
psth_density_array_1_0 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_1_0, "jld2")), "psth_density_array");
coactivity_array_1_0 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_1_0, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", "new", savename(fname_save, param_1_0, "jld2")), "ncells")
solu_1_0 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_1_0, "jld2")), "solu");
sol_t_1_0 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_1_0, "jld2")), "solt");
sol_u_1_0 = reduce(hcat, solu_1_0)


param_1_1 = Dict(
    "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
    "ggap" => ggap, # use input value in nS not calculated uA/cm^2
    "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
    "Ihold_scale"=> Ihold_scale,
    "NMDA_scale" =>NMDA_array[NMDA_ind_norm],
    "rate_synch" => λ_rate_synch[n_ind[2]],
    "synch_dur" => synch_dur,
    "tau_GABA" => tau_GABA,
)
peak_coactivity_array_1_1 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_1_1, "jld2")), "peak_coactivity_array");
psth_density_array_1_1 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_1_1, "jld2")), "psth_density_array");
coactivity_array_1_1 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_1_1, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", "new", savename(fname_save, param_1_1, "jld2")), "ncells")
solu_1_1 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_1_1, "jld2")), "solu");
sol_t_1_1 = wload(datadir("simulations", "Network", "new", savename(fname_save, param_1_1, "jld2")), "solt");
sol_u_1_1 = reduce(hcat, solu_1_1)





#%% FIGURE ######################################################
using LaTeXStrings
@pyimport matplotlib.gridspec as gridspec
@pyimport matplotlib.patches as patches 
@pyimport seaborn as sns
@pyimport numpy as np

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8 


cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#595959", "orchid"])
N = length(λ_rate_synch)
gradient = LinRange(0, 1, N)
n_colors_n = [matplotlib.colors.to_hex(cmap(i)) for i in gradient]


cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkgrey","maroon"])
N = length(gNMDA_array)
gradient = LinRange(0, 1, N)
n_colors_NMDA = [matplotlib.colors.to_hex(cmap2(i)) for i in gradient]



tlim = (-50, 150)
cell_lim = (-0.5, ncells+0.5)
psth_lim = (0, 0.4)
coact_lim  = (0, 100)
PLV_lim  = (0, 0.3)

title_size = 8
letter_size = 10
x_letter = -0.05
y_letter = 1.1

m_size = 2
letters = collect('A':'Z')

n_array_labels = ["$(Int(i /50))" for i in λ_rate_synch]

gNMDA_array_labels = ["$(round(i, digits=1))" for i in gNMDA_array]



np.random.seed(0)
# Fig Setup
fig = plt.figure(figsize=(6.5, 5))
# gs_all  = fig.add_gridspec(1, 2,left=0.125, right=0.975, wspace=0.35, hspace=0.5,  width_ratios=[0.5, 0.5],top=0.95, bottom=0.075)
# gs_left =  gridspec.GridSpecFromSubplotSpec(2, 2, wspace=0.25, hspace=0.5, subplot_spec=py"$(gs_all)[0]") 
# gs_right  =  gridspec.GridSpecFromSubplotSpec(5, 2, wspace=0.5, hspace=0.25, subplot_spec=py"$(gs_all)[1]", height_ratios=[0.25, 0.065, 0.25, 0.15,  0.5]) 
gs_all  = fig.add_gridspec(1, 2,left=0.075, right=0.975, wspace=0.35, hspace=0.5,  width_ratios=[0.45, 0.55],top=0.95, bottom=0.075)
gs_left =  gridspec.GridSpecFromSubplotSpec(2, 2, wspace=0.25, hspace=0.45, subplot_spec=py"$(gs_all)[0]") 
gs_right  =  gridspec.GridSpecFromSubplotSpec(5, 2, wspace=0.5, hspace=0.25, subplot_spec=py"$(gs_all)[1]", height_ratios=[0.25, 0.065, 0.25, 0.175,  0.5]) #height_ratios=[1, 0.25, 1,1]) 


ax_ggap_sum = fig.add_subplot(py"$(gs_right)[2, 0]")
ax_ggap_sum = pyplot_fxns.remove_axis_box(ax_ggap_sum; s=["top", "right"])

ax_gGABA_sum = fig.add_subplot(py"$(gs_right)[0, 0]")
ax_gGABA_sum = pyplot_fxns.remove_axis_box(ax_gGABA_sum; s=["top", "right"])


ax_ggap_gGABA_sum = fig.add_subplot(py"$(gs_right)[4, 0]")
ax_ggap_gGABA_sum = pyplot_fxns.remove_axis_box(ax_ggap_gGABA_sum; s=["top", "right"])


ax_ggap_sum_PLV = fig.add_subplot(py"$(gs_right)[2, 1]")
ax_ggap_sum_PLV = pyplot_fxns.remove_axis_box(ax_ggap_sum_PLV; s=["top", "right"])

ax_gGABA_sum_PLV = fig.add_subplot(py"$(gs_right)[0, 1]")
ax_gGABA_sum_PLV = pyplot_fxns.remove_axis_box(ax_gGABA_sum_PLV; s=["top", "right"])

ax_ggap_gGABA_sum_PLV = fig.add_subplot(py"$(gs_right)[4, 1]")
ax_ggap_gGABA_sum_PLV = pyplot_fxns.remove_axis_box(ax_ggap_gGABA_sum_PLV; s=["top", "right"])



gs_0_0 =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.5, hspace=0.25, subplot_spec=py"$(gs_left)[1, 0]", height_ratios=[0.5, 0.5])
ax_0_0_raster = fig.add_subplot(py"$(gs_0_0)[0]")
ax_0_0_raster = pyplot_fxns.remove_axis_box(ax_0_0_raster; s=["top", "right", "bottom"])
ax_0_0_coactivity = fig.add_subplot(py"$(gs_0_0)[1]")
ax_0_0_coactivity = pyplot_fxns.remove_axis_box(ax_0_0_coactivity; s=["top", "right",])


gs_0_1 =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.5, hspace=0.25, subplot_spec=py"$(gs_left)[1, 1]", height_ratios=[0.5, 0.5])
ax_0_1_raster = fig.add_subplot(py"$(gs_0_1)[0]")
ax_0_1_raster = pyplot_fxns.remove_axis_box(ax_0_1_raster; s=["top", "right", "bottom", "left"])
ax_0_1_coactivity = fig.add_subplot(py"$(gs_0_1)[1]")
ax_0_1_coactivity = pyplot_fxns.remove_axis_box(ax_0_1_coactivity; s=["top", "right", "left"])



gs_1_0 =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.5, hspace=0.25, subplot_spec=py"$(gs_left)[0, 0]", height_ratios=[0.5, 0.5])
ax_1_0_raster = fig.add_subplot(py"$(gs_1_0)[0]")
ax_1_0_raster = pyplot_fxns.remove_axis_box(ax_1_0_raster; s=["top", "right", "bottom"])
ax_1_0_coactivity = fig.add_subplot(py"$(gs_1_0)[1]")
ax_1_0_coactivity = pyplot_fxns.remove_axis_box(ax_1_0_coactivity; s=["top", "right",])


gs_1_1 =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.5, hspace=0.25, subplot_spec=py"$(gs_left)[0, 1]", height_ratios=[0.5, 0.5])
ax_1_1_raster = fig.add_subplot(py"$(gs_1_1)[0]")
ax_1_1_raster = pyplot_fxns.remove_axis_box(ax_1_1_raster; s=["top", "right", "bottom", "left"])
ax_1_1_coactivity = fig.add_subplot(py"$(gs_1_1)[1]")
ax_1_1_coactivity = pyplot_fxns.remove_axis_box(ax_1_1_coactivity; s=["top", "right", "left"])




# for ax_0_0
ax_0_0_raster.eventplot([sol_t_0_0[sol_u_0_0[i+ncells, :] .== 1] .- 1500 for i =1:ncells], color=n_colors_NMDA[NMDA_ind[1]])
ax_0_0_raster.set_title(L"g_{NMDA}" * "=$(gNMDA_array[NMDA_ind[1]]) nS", color=n_colors_NMDA[NMDA_ind[1]], size=title_size)

sat_0 = collect(range(0.1, stop=3, length=ncells))
for i=1:num_surfaces
    ax_0_0_coactivity.plot(bins1_mid .- 500, coactivity_array_0_0[i,:],  color="lightgrey")
end
ax_0_0_coactivity.plot(bins1_mid .- 500, mean(coactivity_array_0_0, dims=1)[1, :],  color=n_colors_NMDA[NMDA_ind[1]])



# for ax_0_1
ax_0_1_raster.eventplot([sol_t_0_1[sol_u_0_1[i+ncells, :] .== 1] .- 1500 for i =1:ncells], color=n_colors_NMDA[NMDA_ind[2]])
ax_0_1_raster.set_title(L"g_{NMDA}" * "=$(gNMDA_array[NMDA_ind[2]]) nS", color=n_colors_NMDA[NMDA_ind[2]], size=title_size)

sat_0 = collect(range(0.1, stop=3, length=ncells))
for i=1:num_surfaces
    ax_0_1_coactivity.plot(bins1_mid .- 500, coactivity_array_0_1[i,:],  color="lightgrey")
end
ax_0_1_coactivity.plot(bins1_mid .- 500, mean(coactivity_array_0_1, dims=1)[1, :],  color=n_colors_NMDA[NMDA_ind[2]])#







# for ax_1_0
ax_1_0_raster.eventplot([sol_t_1_0[sol_u_1_0[i+ncells, :] .== 1] .- 1500 for i =1:ncells], color=n_colors_n[n_ind[1]])
ax_1_0_raster.set_title("$(n_array_labels[n_ind[1]])"  * L"\, x \, \lambda_{P, basal}", color=n_colors_n[n_ind[1]], size=title_size)

sat_0 = collect(range(0.1, stop=3, length=ncells))
for i=1:num_surfaces
    ax_1_0_coactivity.plot(bins1_mid .- 500, coactivity_array_1_0[i,:],  color="lightgrey")
end
ax_1_0_coactivity.plot(bins1_mid .- 500, mean(coactivity_array_1_0, dims=1)[1, :],  color=n_colors_n[n_ind[1]])



# for ax_1_1
ax_1_1_raster.eventplot([sol_t_1_1[sol_u_1_1[i+ncells, :] .== 1] .- 1500 for i =1:ncells], color=n_colors_n[n_ind[2]])
ax_1_1_raster.set_title("$(n_array_labels[n_ind[2]])" * L"\, x \, \lambda_{P, basal}", color=n_colors_n[n_ind[2]], size=title_size)

sat_0 = collect(range(0.1, stop=3, length=ncells))
for i=1:num_surfaces
    ax_1_1_coactivity.plot(bins1_mid .- 500, coactivity_array_1_1[i,:],  color="lightgrey")
end
ax_1_1_coactivity.plot(bins1_mid .- 500, mean(coactivity_array_1_1, dims=1)[1, :],  color=n_colors_n[n_ind[2]])#




# plot summary
sns.boxplot(peak_coactivity_array[n_ind_norm,:,:]', ax=ax_ggap_sum, saturation=0.5, showfliers=false, palette=n_colors_NMDA)
sns.stripplot(peak_coactivity_array[n_ind_norm,:,:]', ax=ax_ggap_sum, palette=n_colors_NMDA, size=m_size, alpha=0.75)
ax_ggap_sum.set_xticks(collect(0:length(NMDA_array)-1))
ax_ggap_sum.set_xticklabels(gNMDA_array_labels, rotation=45)
ax_ggap_sum.set_xlabel(L"g_{NMDA}" * " (nS)")
ax_ggap_sum.set_ylim(coact_lim)
ax_ggap_sum.set_ylabel("Peak Coactivity (%)")

sns.boxplot(peak_coactivity_array[:,NMDA_ind_norm,:]', ax=ax_gGABA_sum, saturation=0.5, showfliers=false, palette=n_colors_n)
sns.stripplot(peak_coactivity_array[:,NMDA_ind_norm,:]', ax=ax_gGABA_sum, palette=n_colors_n, size=m_size, alpha=0.75)
ax_gGABA_sum.set_xticks(collect(0:length(λ_rate_synch)-1))
ax_gGABA_sum.set_xticklabels(n_array_labels)
ax_gGABA_sum.set_xlabel(L"x \, \lambda_{P, basal}")
ax_gGABA_sum.set_ylim(coact_lim)
ax_gGABA_sum.set_ylabel("Peak Coactivity (%)")


sns.heatmap(peak_coactivity_array_mean[:,:, 1]', ax=ax_ggap_gGABA_sum, cmap="viridis", annot=false,  vmin=coact_lim[1], vmax=coact_lim[2],
            linewidths=0, square=true, rasterized=true, cbar_kws=Dict("label"=> "Mean Peak\nCoactivity (%)", "shrink"=>0.95, 
            "location"=> "top", "orientation"=> "horizontal"))

ax_ggap_gGABA_sum.set_xticks(collect(0:length(λ_rate_synch)-1) .+ 0.5)
ax_ggap_gGABA_sum.set_xticklabels(n_array_labels)
ax_ggap_gGABA_sum.set_xlabel(L"x \, \lambda_{P, basal}")

ax_ggap_gGABA_sum.set_yticks(collect(0:length(NMDA_array)-1) .+ 0.5)
ax_ggap_gGABA_sum.set_yticklabels(gNMDA_array_labels)
ax_ggap_gGABA_sum.set_ylabel(L"g_{NMDA}" * " (nS)")

ax_ggap_gGABA_sum.set_yticklabels(ax_ggap_gGABA_sum.get_yticklabels(), rotation=0, ha="right") 
ax_ggap_gGABA_sum.invert_yaxis()


# plot summary
sns.boxplot(delta_PLV_array[n_ind_norm,:,:]', ax=ax_ggap_sum_PLV, saturation=0.5, showfliers=false, palette=n_colors_NMDA)
sns.stripplot(delta_PLV_array[n_ind_norm,:,:]', ax=ax_ggap_sum_PLV, palette=n_colors_NMDA, size=m_size, alpha=0.75)
ax_ggap_sum_PLV.set_xticks(collect(0:length(NMDA_array)-1))
ax_ggap_sum_PLV.set_xticklabels(gNMDA_array_labels, rotation=45)
ax_ggap_sum_PLV.set_xlabel(L"g_{NMDA}" * " (nS)")
ax_ggap_sum_PLV.set_ylim(PLV_lim)
ax_ggap_sum_PLV.set_ylabel("ΔPLV")

sns.boxplot(delta_PLV_array[:,NMDA_ind_norm,:]', ax=ax_gGABA_sum_PLV, saturation=0.5, showfliers=false, palette=n_colors_n)
sns.stripplot(delta_PLV_array[:,NMDA_ind_norm,:]', ax=ax_gGABA_sum_PLV, palette=n_colors_n, size=m_size, alpha=0.75)
ax_gGABA_sum_PLV.set_xticks(collect(0:length(λ_rate_synch)-1))
ax_gGABA_sum_PLV.set_xticklabels(n_array_labels)
ax_gGABA_sum_PLV.set_xlabel(L"x \, \lambda_{P, basal}")
ax_gGABA_sum_PLV.set_ylim(PLV_lim)
ax_gGABA_sum_PLV.set_ylabel("ΔPLV")


sns.heatmap(delta_PLV_array_mean[:,:, 1]', ax=ax_ggap_gGABA_sum_PLV, cmap="plasma", annot=false, vmin=PLV_lim[1], vmax=PLV_lim[2],
            linewidths=0, square=true, rasterized=true, cbar_kws=Dict("label"=> "Mean ΔPLV", "shrink"=>0.95, 
            "location"=> "top", "orientation"=> "horizontal"))

ax_ggap_gGABA_sum_PLV.set_xticks(collect(0:length(λ_rate_synch)-1) .+ 0.5)
ax_ggap_gGABA_sum_PLV.set_xticklabels(n_array_labels)
ax_ggap_gGABA_sum_PLV.set_xlabel(L"x \, \lambda_{P, basal}")

ax_ggap_gGABA_sum_PLV.set_yticks(collect(0:length(NMDA_array)-1) .+ 0.5)
ax_ggap_gGABA_sum_PLV.set_yticklabels(gNMDA_array_labels)
ax_ggap_gGABA_sum_PLV.set_ylabel(L"g_{NMDA}" * " (nS)")

ax_ggap_gGABA_sum_PLV.set_yticklabels(ax_ggap_gGABA_sum_PLV.get_yticklabels(), rotation=0, ha="right") 
ax_ggap_gGABA_sum_PLV.invert_yaxis()




for ax in [ax_0_0_raster, ax_1_0_raster]
    ax.set_ylabel("Cell")
end


for ax in [ax_0_0_coactivity, ax_1_0_coactivity]
    ax.set_ylabel("Coactivity (%)")
end


for ax in [ax_0_0_raster, ax_0_1_raster, ax_1_0_raster,ax_1_1_raster,
    ax_0_0_coactivity, ax_0_1_coactivity, ax_1_0_coactivity, ax_1_1_coactivity,
    ]
    ax.set_xlim(tlim)
end


for ax in [ax_0_0_raster, ax_0_1_raster, ax_1_0_raster,ax_1_1_raster,
    ]
    ax.set_ylim(cell_lim)
end

for ax in [
    ax_0_0_coactivity, ax_0_1_coactivity, ax_1_0_coactivity, ax_1_1_coactivity,
    ]
    ax.set_ylim(coact_lim)
    ax.set_xlabel("Time (ms)")
end



ax_1_0_raster.text(-0.05, 1.1, "$(letters[1])", transform=ax_1_0_raster.transAxes, size=letter_size, weight="bold")
ax_1_1_raster.text(-0.05, 1.1, "$(letters[2])", transform=ax_1_1_raster.transAxes, size=letter_size, weight="bold")
ax_gGABA_sum.text(-0.05, 1.075, "$(letters[3])", transform=ax_gGABA_sum.transAxes, size=letter_size, weight="bold")
ax_gGABA_sum_PLV.text(-0.05, 1.075, "$(letters[4])", transform=ax_gGABA_sum_PLV.transAxes, size=letter_size, weight="bold")

ax_0_0_raster.text(-0.05, 1.1, "$(letters[5])", transform=ax_0_0_raster.transAxes, size=letter_size, weight="bold")
ax_0_1_raster.text(-0.05, 1.1, "$(letters[6])", transform=ax_0_1_raster.transAxes, size=letter_size, weight="bold")
ax_ggap_sum.text(-0.05, 1.075, "$(letters[7])", transform=ax_ggap_sum.transAxes, size=letter_size, weight="bold")
ax_ggap_sum_PLV.text(-0.05, 1.075, "$(letters[8])", transform=ax_ggap_sum_PLV.transAxes, size=letter_size, weight="bold")


ax_ggap_gGABA_sum.text(-0.2, 1.4, "$(letters[9])", transform=ax_ggap_gGABA_sum.transAxes, size=letter_size, weight="bold")
ax_ggap_gGABA_sum_PLV.text(-0.2, 1.4, "$(letters[10])", transform=ax_ggap_gGABA_sum_PLV.transAxes, size=letter_size, weight="bold")

plt.savefig(plotsdir("Figure_7_Network_synch_input_nsynch_NMDA_poisson.png"), dpi=600)
plt.savefig(plotsdir("Figure_7_Network_synch_input_nsynch_NMDA_poisson.pdf"), dpi=600)
plt.savefig(plotsdir("Figure_7_Network_synch_input_nsynch_NMDA_poisson.eps"), dpi=600)
plt.show()

#%%

@pyimport scipy as sp

# lambda peak
x = [collect((λ_rate_synch * ones(1, 100))')...]
y = [collect(peak_coactivity_array[:,NMDA_ind_norm,:]')...]

rho, p = sp.stats.spearmanr(x,y)

# lambda PLV
x = [collect((λ_rate_synch * ones(1, 100))')...]
y = [collect(delta_PLV_array[:,NMDA_ind_norm,:]')...]

rho, p = sp.stats.spearmanr(x,y)
@pyimport scipy as sp

# NMDA peak
x = [collect((λ_rate_synch * ones(1, 100))')...]
y = [collect(peak_coactivity_array[n_ind_norm,:,:]')...]

rho, p = sp.stats.spearmanr(x,y)

# NMDA PLV
x = [collect((λ_rate_synch * ones(1, 100))')...]
y = [collect(delta_PLV_array[n_ind_norm,:,:]')...]

rho, p = sp.stats.spearmanr(x,y)