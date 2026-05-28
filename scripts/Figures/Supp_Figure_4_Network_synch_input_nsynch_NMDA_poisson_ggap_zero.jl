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
syn_delay = 5
# NMDA_scale = 1.5
NMDA_scale = 1.0
# NMDA_scale = 0.5
synch_dur = 2.0
tau_GABA = 1.9

param_save_ggap = Dict(
    "I" => 0.0, 
    "gsyn" => gsyn, # use input value in nS not calculated uA/cm^2
    "ggap" => 0.0, # use input value in nS not calculated uA/cm^2
    "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
    # "rate_synch" => λ_rate_synch_i,
    "synch_dur" => synch_dur,
    # "NMDA_scale" => NMDA_scale, 
    "tau_GABA" => tau_GABA,
)

param_save_GABA = Dict(
    "I" => 0.0, 
    "gsyn" => 0.0, # use input value in nS not calculated uA/cm^2
    "ggap" => ggap, # use input value in nS not calculated uA/cm^2
    "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
    # "rate_synch" => λ_rate_synch_i,
    "synch_dur" => synch_dur,
    # "NMDA_scale" => NMDA_scale, 
    "tau_GABA" => tau_GABA,
)


param_save_ggap_GABA = Dict(
    "I" => 0.0, 
    "gsyn" => 0.0, # use input value in nS not calculated uA/cm^2
    "ggap" => 0.0, # use input value in nS not calculated uA/cm^2
    "gAMPA" => gAMPA, # use input value in nS not calculated uA/cm^2
    # "rate_synch" => λ_rate_synch_i,
    "synch_dur" => synch_dur,
    # "NMDA_scale" => NMDA_scale, 
    "tau_GABA" => tau_GABA,
)


# wsave(datadir("simulations", "Network", savename(fname_save, param, "jld2")),  @strdict fname_net peak_coactivity_array peak_coactivity_array_mean gsyn_array ggap_array)

fname_save = "Network_CSC1_nsynch_array_poisson"

peak_coactivity_array_ggap = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save_ggap, "jld2")), "peak_coactivity_array");
peak_coactivity_array_mean_ggap = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save_ggap, "jld2")), "peak_coactivity_array_mean");
PLV_before_array_ggap = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save_ggap, "jld2")), "PLV_before_array");
PLV_before_array_mean_ggap = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save_ggap, "jld2")), "PLV_mean_before_array");
PLV_after_array_ggap = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save_ggap, "jld2")), "PLV_after_50_array");
PLV_after_array_mean_ggap = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save_ggap, "jld2")), "PLV_mean_after_50_array");
λ_rate_synch_ggap = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save_ggap, "jld2")), "n_array");
NMDA_array_ggap = wload(datadir("simulations", "Network", "new", savename(fname_save, param_save_ggap, "jld2")), "NMDA_array");

delta_PLV_array_ggap = PLV_after_array_ggap .- PLV_before_array_ggap
delta_PLV_array_mean_ggap = PLV_after_array_mean_ggap .- PLV_before_array_mean_ggap

gNMDA_array_ggap = round.(NMDA_array_ggap .* gAMPA, digits=1)

NMDA_ind_ggap = [2,5]
n_ind_norm_ggap = findfirst(λ_rate_synch_ggap .== 600.)
# NMDA_ind_norm = findfirst(NMDA_array .== 1.5)
NMDA_ind_norm_ggap = findfirst(NMDA_array_ggap .== 1.0)






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



cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkgrey","maroon"])

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#595959", "orchid"])
N = length(λ_rate_synch_ggap)
gradient = LinRange(0, 1, N)
n_colors_n = [matplotlib.colors.to_hex(cmap(i)) for i in gradient]


cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkgrey","maroon"])
N = length(gNMDA_array_ggap)
gradient = LinRange(0, 1, N)
n_colors_NMDA = [matplotlib.colors.to_hex(cmap2(i)) for i in gradient]



# tlim = (-200, 200)
tlim = (-50, 150)
cell_lim = (-0.5, ncells+0.5)
psth_lim = (0, 0.4)
coact_lim  = (0, 100)
# PLV_lim  = (0, 0.375)#0.275)
# PLV_lim  = (0, 0.275)
PLV_lim  = (0, 0.3)

title_size = 8
letter_size = 10
x_letter = -0.05
y_letter = 1.1

m_size = 2
letters = collect('A':'Z')

n_array_labels_ggap = ["$(Int(i /50))" for i in λ_rate_synch_ggap]#"$(i)" for i in n_array]
gNMDA_array_labels_ggap = ["$(round(i, digits=1))" for i in gNMDA_array_ggap]



#%%


# Fig Setup
fig = plt.figure(figsize=(6.5, 3.5))
gs_all  = fig.add_gridspec(2, 2,left=0.075, right=0.975, bottom=0.15, top=0.9, wspace=0.25, hspace=0.5,)


ax_NMDA_sum = fig.add_subplot(py"$(gs_all)[1, 0]")
ax_NMDA_sum = pyplot_fxns.remove_axis_box(ax_NMDA_sum; s=["top", "right"])

ax_n_sum = fig.add_subplot(py"$(gs_all)[0, 0]")
ax_n_sum = pyplot_fxns.remove_axis_box(ax_n_sum; s=["top", "right"])



ax_n_NMDA_sum = fig.add_subplot(py"$(gs_all)[:, 1]")
ax_n_NMDA_sum = pyplot_fxns.remove_axis_box(ax_n_NMDA_sum; s=["top", "right"])





# plot summary

sns.boxplot(delta_PLV_array_ggap[:,NMDA_ind_norm_ggap,:]', ax=ax_n_sum, saturation=0.5, showfliers=false, palette=n_colors_n)
sns.stripplot(delta_PLV_array_ggap[:,NMDA_ind_norm_ggap,:]', ax=ax_n_sum, palette=n_colors_n, size=m_size, alpha=0.75)
ax_n_sum.set_xticks(collect(0:length(λ_rate_synch_ggap)-1))
ax_n_sum.set_xticklabels(n_array_labels_ggap)
# ax_n_sum.set_xlabel(L"g_{gap}" * " (nS)")#, labelpad=0)
ax_n_sum.set_ylim(PLV_lim)
ax_n_sum.set_ylabel("ΔPLV")
ax_n_sum.set_xticklabels(ax_n_sum.get_xticklabels(), rotation=45, ha="center", va="top") 
ax_n_sum.set_xlabel(L"x \, \lambda_{P, basal}")


sns.boxplot(delta_PLV_array_ggap[n_ind_norm_ggap,:,:]', ax=ax_NMDA_sum, saturation=0.5, showfliers=false, palette=n_colors_NMDA)
sns.stripplot(delta_PLV_array_ggap[n_ind_norm_ggap,:,:]', ax=ax_NMDA_sum, palette=n_colors_NMDA, size=m_size, alpha=0.75)
ax_NMDA_sum.set_xticks(collect(0:length(NMDA_array_ggap)-1))
ax_NMDA_sum.set_xticklabels(gNMDA_array_labels_ggap )
# ax_NMDA_sum.set_xlabel(L"g_{GABA}" * " (nS)")#, labelpad=0)
ax_NMDA_sum.set_ylim(PLV_lim)
ax_NMDA_sum.set_ylabel("ΔPLV")
ax_NMDA_sum.set_xticklabels(ax_NMDA_sum.get_xticklabels(), rotation=45, ha="center", va="top") 

ax_NMDA_sum.set_xlabel(L"g_{NMDA}" * " (nS)")#, labelpad=0)


sns.heatmap(delta_PLV_array_mean_ggap[:,:, 1]', ax=ax_n_NMDA_sum, cmap="plasma", annot=false, vmin=PLV_lim[1], vmax=PLV_lim[2],
            linewidths=0, square=true, rasterized=true, cbar_kws=Dict("label"=> "Mean ΔPLV", "shrink"=>0.8, 
            "location"=> "top", "orientation"=> "horizontal"))
ax_n_NMDA_sum.set_xticks(collect(0:length(λ_rate_synch_ggap)-1) .+ 0.5)
ax_n_NMDA_sum.set_xticklabels(n_array_labels_ggap)
ax_n_NMDA_sum.set_xlabel(L"x \, \lambda_{P, basal}")

ax_n_NMDA_sum.set_yticks(collect(0:length(NMDA_array_ggap)-1) .+ 0.5)
ax_n_NMDA_sum.set_yticklabels(gNMDA_array_labels_ggap)
ax_n_NMDA_sum.set_ylabel(L"g_{NMDA}" * " (nS)")#, labelpad=0)

ax_n_NMDA_sum.set_yticklabels(ax_n_NMDA_sum.get_yticklabels(), rotation=0, ha="right") 
ax_n_NMDA_sum.invert_yaxis()



ax_n_sum.text(-0.15, 1.1, "$(letters[1])", transform=ax_n_sum.transAxes, size=letter_size, weight="bold")
ax_NMDA_sum.text(-0.15, 1.1, "$(letters[2])", transform=ax_NMDA_sum.transAxes, size=letter_size, weight="bold")
ax_n_NMDA_sum.text(-0.2, 1.3, "$(letters[3])", transform=ax_n_NMDA_sum.transAxes, size=letter_size, weight="bold")


plt.savefig(plotsdir( "Supp_Figure_4_Network_synch_input_nsynch_NMDA_poisson_ggap_zero.png"), dpi=600)
plt.savefig(plotsdir( "Supp_Figure_4_Network_synch_input_nsynch_NMDA_poisson_ggap_zero.pdf"), dpi=600)
plt.savefig(plotsdir( "Supp_Figure_4_Network_synch_input_nsynch_NMDA_poisson_ggap_zero.eps"), dpi=600)
plt.show()

#%%

@pyimport scipy as sp

x = [collect((NMDA_array_ggap * ones(1, 100))')...]
y = [collect(delta_PLV_array_ggap[n_ind_norm_ggap,:,:]')...]

rho, p = sp.stats.spearmanr(x,y)


x = [collect((λ_rate_synch_ggap * ones(1, 100))')...]
y = [collect(delta_PLV_array_ggap[:,NMDA_ind_norm_ggap,:]')...]

rho, p = sp.stats.spearmanr(x,y)
