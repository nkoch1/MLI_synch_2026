using DrWatson
@quickactivate  "MLI_synch_2026"

using PyPlot
using LsqFit
using CSV
using DataFrames

include(srcdir("pyplot_fxns.jl"))
using .pyplot_fxns

round_step(x, step) = round.(x / step) .* step # round to nearest multiple of step

# Rieubland et al 2014 https://doi.org/10.1016/j.neuron.2013.12.029
# data digitized by https://web.eecs.utk.edu/~dcostine/personal/PowerDeviceLib/DigiTest/index.html
#%% Rieubland_2014 Figure 2 A - Sagittal probability
fpath = datadir("exp_raw", "Rieubland_2014", "Rieubland_2014_Fig2_a_chem.csv")
df_chem_prob_sagittal = CSV.read(fpath, DataFrame; header=false)
rename!(df_chem_prob_sagittal, ["distance", "prob_chem"])
df_chem_prob_sagittal[!, "distance rounded"] = round_step(df_chem_prob_sagittal[!, "distance"], 5)

fpath = datadir("exp_raw", "Rieubland_2014", "Rieubland_2014_Fig2_a_elec.csv")
df_elec_prob_sagittal = CSV.read(fpath, DataFrame; header=false)
rename!(df_elec_prob_sagittal, ["distance", "prob_elec"])
df_elec_prob_sagittal[!, "distance rounded"] = round_step(df_chem_prob_sagittal[!, "distance"], 5)




#%% Rieubland_2014 Figure 2 B - Transverse probability
fpath = datadir("exp_raw", "Rieubland_2014", "Rieubland_2014_Fig2_b_chem.csv")
df_chem_prob_transverse = CSV.read(fpath, DataFrame; header=false)
rename!(df_chem_prob_transverse, ["distance", "prob_chem"])
df_chem_prob_transverse[!, "distance rounded"] = round_step(df_chem_prob_transverse[!, "distance"], 5)

fpath = datadir("exp_raw", "Rieubland_2014", "Rieubland_2014_Fig2_b_elec.csv")
df_elec_prob_transverse = CSV.read(fpath, DataFrame; header=false)
rename!(df_elec_prob_transverse, ["distance", "prob_elec"])
df_elec_prob_transverse[!, "distance rounded"] = round_step(df_elec_prob_transverse[!, "distance"], 5)



#%% Rieubland_2014 Figure S2 B - Sagittal distance
fpath = datadir("exp_raw", "Rieubland_2014", "Rieubland_2024_FigS2_b_sagittal.csv")
df_dist = CSV.read(fpath, DataFrame; header=false)
rename!(df_dist, ["distance", "pairs"])
df_dist[!, "distance rounded"] = round_step(df_dist[!, "distance"], 5)

df_dist[!, "pairs integers"] = round_step(df_dist[!, "pairs"], 1)
df_dist[!, "pairs normalized"] = df_dist[!, "pairs integers"] / sum(df_dist[!, "pairs integers"])
df_dist[!, "pairs density"] = df_dist[!, "pairs integers"] / sum(df_dist[!, "pairs integers"]) / 10



#%%
using SpecialFunctions
function gamma_pdf(x, p)
    k, θ, A = p
    # k is the shape parameter
    # θ is the scale parameter
    # A is the scaling from pdf to the probability in the data
    coeff = 1 / (gamma(k) * θ^k)
    return coeff .* x.^(k - 1) .* exp.(-x ./ θ) *A
end

fit_chem_sagittal = curve_fit(gamma_pdf, df_chem_prob_sagittal[!, "distance rounded"], df_chem_prob_sagittal[!, "prob_chem"], [5, 1, 0.5], lower=[0.00001, 0.01, 0.1])  #[5, 10, 0.5]
fit_elec_sagittal = curve_fit(gamma_pdf, df_elec_prob_sagittal[!, "distance rounded"], df_elec_prob_sagittal[!, "prob_elec"], [5, 1, 0.5], lower=[0.00001, 0.01, 0.1])  #[5, 10, 0.5]


param_chem_sagittal= fit_chem_sagittal.param
wsave(datadir("simulations", "Network", "MLI_chem_sagittal_gamma.jld2"),  @strdict fit_chem_sagittal param_chem_sagittal)
param_elec_sagittal= fit_elec_sagittal.param
wsave(datadir("simulations", "Network", "MLI_elec_sagittal_gamma.jld2"),  @strdict fit_elec_sagittal param_elec_sagittal)


x_sagittal = collect(5:0.1:150)
fit_chem_gamma_sagittal = gamma_pdf(x_sagittal,fit_chem_sagittal.param)
fit_elec_gamma_sagittal = gamma_pdf(x_sagittal,fit_elec_sagittal.param)



#%% FITTING - Rieubland_2014 Figure 2 B - Transverse probability

fit_chem_transverse = curve_fit(gamma_pdf, df_chem_prob_transverse[!, "distance rounded"], df_chem_prob_transverse[!, "prob_chem"], [1.5, 1, 50], lower=[0.00001, 1, 0.1])  #[5, 10, 0.5]
fit_elec_transverse = curve_fit(gamma_pdf, df_elec_prob_transverse[!, "distance rounded"], df_elec_prob_transverse[!, "prob_elec"], [1.5, 1, 50], lower=[0.00001, 1, 0.1])  #[5, 10, 0.5]


x_transverse = collect(5:0.1:50)
fit_chem_gamma_transverse = gamma_pdf(x_transverse,fit_chem_transverse.param)
fit_elec_gamma_transverse = gamma_pdf(x_transverse,fit_elec_transverse.param)


param_chem_transverse= fit_chem_transverse.param
wsave(datadir("simulations", "Network", "MLI_chem_transverse_gamma.jld2"),  @strdict fit_chem_transverse param_chem_transverse)
param_elec_transverse= fit_elec_transverse.param
wsave(datadir("simulations", "Network", "MLI_elec_transverse_gamma.jld2"),  @strdict fit_elec_transverse param_elec_transverse)




#%% FITTING - Rieubland_2014 Figure S2 B - Sagittal distance
using LsqFit

# Log-normal distribution
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3669.htm
lognormal_pdf(x, p) = 1 ./ ((x .- p[1]) .* p[2] .* sqrt(2 * π)) .* exp.(-((log.((x .- p[1]) ./ p[3])) .^ 2 ./ (2 .* p[2] .^ 2))) # p = θ, σ, m

fit = curve_fit(lognormal_pdf, df_dist[!, "distance rounded"], df_dist[!, "pairs density"], [2., 0.5, 50.])
μ, σ, m = fit.param
fit_pdf = lognormal_pdf(df_dist[!, "distance rounded"], fit.param)

x = collect(0:0.1:200)
fit_x = lognormal_pdf(x, fit.param)


fpath = datadir("simulations", "Network", "MLI_distance_lognorm.jld2")
fpath_param = datadir("simulations", "Network", "MLI_distance_lognorm_param.jld2")

param = fit.param
wsave(datadir("simulations", "Network", "MLI_distance_lognorm.jld2"),  @strdict fit param)

μ_L = log(m) 
σ_R = μ_L /(sqrt(π/2)) 



#%%

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8 
x_letter = -0.25
y_letter = 1.025
letter_size = 10

fig, axs = plt.subplots(1, 2, figsize=(6.9, 3))
plt.subplots_adjust(wspace=0.5, hspace=0.5, left=0.1, bottom=0.15, right=0.95)

axs[1] = pyplot_fxns.remove_axis_box(axs[1]; s=["top", "right"])
axs[1].bar(round_step(df_dist[!, "distance rounded"], 5), df_dist[!, "pairs density"], color="grey", width=10, label="Rieubland (2014)")
axs[1].plot(x, fit_x, color="k", label="LogNormal fit")
axs[1].set_xlabel("Distance in sagittal plane  (μm)")
axs[1].set_ylabel("Probability density")
axs[1].set_xlim(0, 200)
axs[1].legend(frameon=false, fontsize=6, loc=1)

handles, labels = axs[1].get_legend_handles_labels()
order = [2, 1]
axs[1].legend([handles[idx] for idx in order],[labels[idx] for idx in order], frameon=false, fontsize=6, loc=1)


axs[2] = pyplot_fxns.remove_axis_box(axs[2]; s=["top", "right"])
axs[2].plot(df_chem_prob_sagittal[!, "distance rounded"], df_chem_prob_sagittal[!, "prob_chem"], color="b", label="Rieubland (2014) - Chemical", marker="o")
axs[2].plot(x_sagittal, fit_chem_gamma_sagittal,"--", color="cyan", label="Chemical gamma fit")
axs[2].plot(df_elec_prob_sagittal[!, "distance rounded"], df_elec_prob_sagittal[!, "prob_elec"], color="r", label="Rieubland (2014) - Electrical", marker="o")
axs[2].plot(x_sagittal, fit_elec_gamma_sagittal,"--", color="tab:orange", label="Electrical gamma fit")
axs[2].set_xlabel("Distance in sagittal plane (μm)")
axs[2].set_ylabel("Connection probability")
axs[2].set_xlim(0, 200)
axs[2].set_ylim((0, 0.67))
axs[2].legend(frameon=false, fontsize=6, loc=1)

letters = collect('A':'Z')
axs[1].text(x_letter, y_letter, "$(letters[1])", transform=axs[1].transAxes, size=letter_size, weight="bold")
axs[2].text(x_letter+0.05, y_letter, "$(letters[2])", transform=axs[2].transAxes, size=letter_size, weight="bold")
plt.savefig(plotsdir("Supp_Figure_1_Rieubland_2014_fitting.png"), dpi=600)
plt.savefig(plotsdir("Supp_Figure_1_Rieubland_2014_fitting.pdf"), dpi=600)
plt.savefig(plotsdir("Supp_Figure_1_Rieubland_2014_fitting.eps"), dpi=600)
plt.show()