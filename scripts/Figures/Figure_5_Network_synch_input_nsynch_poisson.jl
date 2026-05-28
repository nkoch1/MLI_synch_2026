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

#%% LOAD 

fname = "Network_realizations_dense_scaled_336.jld2"

ncells = wload(datadir("simulations", "Network", fname),  "N")
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

NMDA_scale = 0.0
syn_delay = 5
synch_dur = 2.0
tau_GABA = 1.9

param_save = Dict(
    "I" => 0.0, 
    "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
    "ggap" => ggap, # use input value in nS not calculated uA/cm^2
    "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
    "synch_dur" => synch_dur,
    "NMDA_scale" => NMDA_scale, 
    "tau_GABA" => tau_GABA,
)

fname_save = "Network_CSC1"
peak_coactivity_array = wload(datadir("simulations", "Network", savename(fname_save, param_save, "jld2")), "peak_coactivity_array");
peak_coactivity_array_mean = wload(datadir("simulations", "Network", savename(fname_save, param_save, "jld2")), "peak_coactivity_array_mean");
peak_coactivity_array_std = wload(datadir("simulations", "Network", savename(fname_save, param_save, "jld2")), "peak_coactivity_array_std");
λ_rate_synch = wload(datadir("simulations", "Network", savename(fname_save, param_save, "jld2")), "n_array")

#%%
# n_array,ncells
n_ind = [2,5,8]

fname_save = "Network_CSC1_NMDA"

param_0_0 = Dict(
        "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
        "ggap" => ggap, # use input value in nS not calculated uA/cm^2
        "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
        "Ihold_scale"=> Ihold_scale,
        "NMDA_scale" =>NMDA_scale,
        "rate_synch" => λ_rate_synch[n_ind[1]],
        "synch_dur" => synch_dur,
        "tau_GABA" => tau_GABA,
    )

peak_coactivity_array_0_0 = wload(datadir("simulations", "Network",  savename(fname_save, param_0_0, "jld2")), "peak_coactivity_array");
psth_density_array_0_0 = wload(datadir("simulations", "Network",  savename(fname_save, param_0_0, "jld2")), "psth_density_array");
coactivity_array_0_0 = wload(datadir("simulations", "Network",  savename(fname_save, param_0_0, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network",  savename(fname_save, param_0_0, "jld2")), "ncells")
solu_0_0 = wload(datadir("simulations", "Network", savename(fname_save, param_0_0, "jld2")), "solu");
sol_t_0_0 = wload(datadir("simulations", "Network", savename(fname_save, param_0_0, "jld2")), "solt");
sol_u_0_0 = reduce(hcat, solu_0_0)


param_0_1 = Dict(
        "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
        "ggap" => ggap, # use input value in nS not calculated uA/cm^2
        "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
        "Ihold_scale"=> Ihold_scale,
        "NMDA_scale" =>NMDA_scale,
        "rate_synch" => λ_rate_synch[n_ind[2]],
        "synch_dur" => synch_dur,
        "tau_GABA" => tau_GABA,
    )
peak_coactivity_array_0_1 = wload(datadir("simulations", "Network", savename(fname_save, param_0_1, "jld2")), "peak_coactivity_array");
psth_density_array_0_1 = wload(datadir("simulations", "Network", savename(fname_save, param_0_1, "jld2")), "psth_density_array");
coactivity_array_0_1 = wload(datadir("simulations", "Network", savename(fname_save, param_0_1, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", savename(fname_save, param_0_1, "jld2")), "ncells")
solu_0_1 = wload(datadir("simulations", "Network", savename(fname_save, param_0_1, "jld2")), "solu");
sol_t_0_1 = wload(datadir("simulations", "Network", savename(fname_save, param_0_1, "jld2")), "solt");
sol_u_0_1 = reduce(hcat, solu_0_1)


param_0_2 = Dict(
        "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
        "ggap" => ggap, # use input value in nS not calculated uA/cm^2
        "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
        "Ihold_scale"=> Ihold_scale,
        "NMDA_scale" =>NMDA_scale,
        "rate_synch" => λ_rate_synch[n_ind[3]],
        "synch_dur" => synch_dur,
        "tau_GABA" => tau_GABA,
    )
peak_coactivity_array_0_2 = wload(datadir("simulations", "Network", savename(fname_save, param_0_2, "jld2")), "peak_coactivity_array");
psth_density_array_0_2 = wload(datadir("simulations", "Network", savename(fname_save, param_0_2, "jld2")), "psth_density_array");
coactivity_array_0_2 = wload(datadir("simulations", "Network", savename(fname_save, param_0_2, "jld2")), "coactivity_array");
ncells = wload(datadir("simulations", "Network", savename(fname_save, param_0_2, "jld2")), "ncells")
solu_0_2 = wload(datadir("simulations", "Network", savename(fname_save, param_0_2, "jld2")), "solu");
sol_t_0_2 = wload(datadir("simulations", "Network", savename(fname_save, param_0_2, "jld2")), "solt");
sol_u_0_2 = reduce(hcat, solu_0_2)




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

custom_colors = ["#595959", "orchid"]
num_spikes_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("nspikes", custom_colors, N = 100) # Create a ListedColormap using the custom colors


cmap = PyPlot.matplotlib.colormaps.get_cmap(num_spikes_cmap)
N = length(λ_rate_synch)
gradient = LinRange(0, 1, N)
n_colors = [matplotlib.colors.to_hex(cmap(i)) for i in gradient]


tlim = (-150, 150)
cell_lim = (-0.5, ncells+0.5)
psth_lim = (0, 0.5)
coact_lim  = (0, 100)
coact_lim2  = (0, 90)

title_size = 8
letter_size = 10
x_letter = -0.05
y_letter = 1.1

m_size = 2
letters = collect('A':'Z')

n_array_labels = ["$(Int(i /50))" for i in λ_rate_synch] 

# Fig Setup
fig = plt.figure(figsize=(6.9, 3))
gs_all  = fig.add_gridspec(1, 2,left=0.1, right=0.975, wspace=0.35, hspace=0.5,  width_ratios=[0.7, 0.3],top=0.9, bottom=0.125)
gs_left =  gridspec.GridSpecFromSubplotSpec(1, 3, wspace=0.25, hspace=0.5, subplot_spec=py"$(gs_all)[0]") 
gs_right  =  gridspec.GridSpecFromSubplotSpec(1, 1, wspace=0., hspace=0., subplot_spec=py"$(gs_all)[1]")

ax_ggap_sum = fig.add_subplot(py"$(gs_right)[0]")
ax_ggap_sum = pyplot_fxns.remove_axis_box(ax_ggap_sum; s=["top", "right"])



gs_0_0 =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.5, hspace=0.25, subplot_spec=py"$(gs_left)[0, 0]", height_ratios=[0.5, 0.5]) 
ax_0_0_raster = fig.add_subplot(py"$(gs_0_0)[0]")
ax_0_0_raster = pyplot_fxns.remove_axis_box(ax_0_0_raster; s=["top", "right", "bottom"])
ax_0_0_coactivity = fig.add_subplot(py"$(gs_0_0)[1]")
ax_0_0_coactivity = pyplot_fxns.remove_axis_box(ax_0_0_coactivity; s=["top", "right",])


gs_0_1 =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.5, hspace=0.25, subplot_spec=py"$(gs_left)[0, 1]", height_ratios=[0.5, 0.5]) 
ax_0_1_raster = fig.add_subplot(py"$(gs_0_1)[0]")
ax_0_1_raster = pyplot_fxns.remove_axis_box(ax_0_1_raster; s=["top", "right", "bottom", "left"])
ax_0_1_coactivity = fig.add_subplot(py"$(gs_0_1)[1]")
ax_0_1_coactivity = pyplot_fxns.remove_axis_box(ax_0_1_coactivity; s=["top", "right", "left"])


gs_0_2 =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.5, hspace=0.25, subplot_spec=py"$(gs_left)[0, 2]", height_ratios=[0.5, 0.5]) 
ax_0_2_raster = fig.add_subplot(py"$(gs_0_2)[0]")
ax_0_2_raster = pyplot_fxns.remove_axis_box(ax_0_2_raster; s=["top", "right", "bottom", "left"])
ax_0_2_coactivity = fig.add_subplot(py"$(gs_0_2)[1]")
ax_0_2_coactivity = pyplot_fxns.remove_axis_box(ax_0_2_coactivity; s=["top", "right", "left"])





# for ax_0_0
ax_0_0_raster.eventplot([sol_t_0_0[sol_u_0_0[i+ncells, :] .== 1] .- 1500 for i =1:ncells], color=n_colors[n_ind[1]])
ax_0_0_raster.set_title("$(n_array_labels[n_ind[1]])" * L"\, x \, \lambda_{P, basal}", color=n_colors[n_ind[1]], size=title_size)

sat_0 = collect(range(0.1, stop=3, length=ncells))
for i=1:num_surfaces
    ax_0_0_coactivity.plot(bins1_mid .- 500, coactivity_array_0_0[i,:],  color="lightgrey")
end
ax_0_0_coactivity.plot(bins1_mid .- 500, mean(coactivity_array_0_0, dims=1)[1, :],  color=n_colors[n_ind[1]])



# for ax_0_1
ax_0_1_raster.eventplot([sol_t_0_1[sol_u_0_1[i+ncells, :] .== 1] .- 1500 for i =1:ncells], color=n_colors[n_ind[2]])
ax_0_1_raster.set_title("$(n_array_labels[n_ind[2]])" * L"\, x \, \lambda_{P, basal}",  color=n_colors[n_ind[2]], size=title_size)

sat_0 = collect(range(0.1, stop=3, length=ncells))
for i=1:num_surfaces
    ax_0_1_coactivity.plot(bins1_mid .- 500, coactivity_array_0_1[i,:],  color="lightgrey")
end
ax_0_1_coactivity.plot(bins1_mid .- 500, mean(coactivity_array_0_1, dims=1)[1, :],  color=n_colors[n_ind[2]])#



# for ax_0_2
ax_0_2_raster.eventplot([sol_t_0_2[sol_u_0_2[i+ncells, :] .== 1] .- 1500 for i =1:ncells], color=n_colors[n_ind[3]])
ax_0_2_raster.set_title( "$(n_array_labels[n_ind[3]])" * L"\, x \, \lambda_{P, basal}", color=n_colors[n_ind[3]], size=title_size)

sat_0 = collect(range(0.1, stop=3, length=ncells))
for i=1:num_surfaces
    ax_0_2_coactivity.plot(bins1_mid .- 500, coactivity_array_0_2[i,:],  color="lightgrey")
end
ax_0_2_coactivity.plot(bins1_mid .- 500, mean(coactivity_array_0_2, dims=1)[1, :],  color=n_colors[n_ind[3]])


# plot summary
sns.boxplot(peak_coactivity_array[:,:]', ax=ax_ggap_sum, saturation=0.5, showfliers=false, palette=n_colors)
sns.stripplot(peak_coactivity_array[:,:]', ax=ax_ggap_sum, palette=n_colors, size=m_size, alpha=0.75)
ax_ggap_sum.set_ylim(0, 100)
ax_ggap_sum.set_xticks(collect(0:length(λ_rate_synch)-1))
ax_ggap_sum.set_xticklabels(n_array_labels)
ax_ggap_sum.set_xlabel(L"x \, \lambda_{P, basal}")
ax_ggap_sum.set_ylim(coact_lim)
ax_ggap_sum.set_ylabel("Peak Coactivity (%)")


for ax in [ax_0_0_raster,]
    ax.set_ylabel("Cell")
end

for ax in [ax_0_0_coactivity,]
    ax.set_ylabel("Coactivity (%)")
end


for ax in [
    ax_0_0_raster, ax_0_1_raster, ax_0_2_raster, 
    ax_0_0_coactivity, ax_0_1_coactivity, ax_0_2_coactivity,
    ]
    ax.set_xlim(tlim)
end


for ax in [ax_0_0_raster, ax_0_1_raster, ax_0_2_raster, 
    ]
    ax.set_ylim(cell_lim)
end

for ax in [
    ax_0_0_coactivity, ax_0_1_coactivity, ax_0_2_coactivity,
    ]
    ax.set_ylim(coact_lim2)
    ax.set_xlabel("Time (ms)")
end


ax_0_0_raster.text(-0.05, 1.1, "$(letters[1])", transform=ax_0_0_raster.transAxes, size=letter_size, weight="bold")
ax_0_1_raster.text(-0.05, 1.1, "$(letters[2])", transform=ax_0_1_raster.transAxes, size=letter_size, weight="bold")
ax_0_2_raster.text(-0.05, 1.1, "$(letters[3])", transform=ax_0_2_raster.transAxes, size=letter_size, weight="bold")
ax_ggap_sum.text(-0.05, 1.05, "$(letters[4])", transform=ax_ggap_sum.transAxes, size=letter_size, weight="bold")


plt.savefig(plotsdir("Figure_5_Network_synch_input_nsynch_poisson.png"), dpi=600)
plt.savefig(plotsdir("Figure_5_Network_synch_input_nsynch_poisson.pdf"), dpi=600)
plt.savefig(plotsdir("Figure_5_Network_synch_input_nsynch_poisson.eps"), dpi=600)
plt.show()


