
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

#%%
fpath = datadir("exp_pro", "HBP_firing_analysis_extract.json")

py"""
import numpy as np
import pandas as pd
df = pd.read_json($fpath)
"""
i = 2 # cell 3
I_data = py"df.loc[$(i-1), 'I']"

function plot_ts(ax, filename, col; title="")
    sol =  wload(datadir("simulations", "Cable_charac", "Firing", "Firing_CSC_cable_soma_Ih_$(filename).jld2"), "sol")
    ax.plot(sol.t .- sol.t[1], sol[1,:], color=col)
    ax.set_title(title, color=col)
    return ax
end

function plot_AP(ax, filename, col; alpha=0.5, ls="--")
    sol =  wload(datadir("simulations", "Cable_charac", "Firing", "Firing_CSC_cable_soma_Ih_$(filename).jld2"), "sol")
    dV =  wload(datadir("simulations", "Cable_charac", "Firing", "Firing_CSC_cable_soma_Ih_$(filename).jld2"), "dV")
    ax.plot(sol[1,:], dV, color=col, ls, alpha=alpha)
    return ax
end


function plot_fI(ax, filename, col; alpha=0.5, ls="-")
    F_array =  wload(datadir("simulations", "Cable_charac", "Firing", "Firing_CSC_cable_soma_Ih_$(filename).jld2"), "F_array")
    # Iarray =  wload(datadir("simulations", "Cable_charac", "Firing", "Firing_CSC_cable_soma_Ih_$(filename).jld2"), "Iarray")
    ax.plot(I_data, F_array, color=col, ls, alpha=alpha)
    return ax
end


#%% FIGURE ######################################################
using LaTeXStrings
@pyimport matplotlib.gridspec as gridspec

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8 

letter_size = 10
x_letter = -0.025
y_letter = 1.125
x_letter2 = -0.035
y_letter2  = 1.075

col_WT = "black"
col_passive = "tab:grey"
col_no_Kd = "tab:blue"
col_no_A = "tab:purple"
col_no_SK = "tab:green"
col_no_T = "tab:orange"
col_no_HVA = "tab:cyan"
col_no_h = "tab:red"

# Fig Setup
fig = plt.figure(figsize=(7.5, 4))
gs_all  = fig.add_gridspec(1, 2, left=0.075, right=0.95, wspace=0.5, hspace=0.5, width_ratios=[0.7, 0.3])#, height_ratios=[0.2, 0.0, 0.15, 0.15, 0.15, 0.15])
gs_left  =  gridspec.GridSpecFromSubplotSpec(3, 2, wspace=0.3, hspace=0.75,  subplot_spec=py"$(gs_all)[0,0]") # width_ratios=[0.15, 0.1,0.1,0.1, 0.025, 0.15],
gs_right  =  gridspec.GridSpecFromSubplotSpec(2, 1, wspace=0.5, hspace=0.6, subplot_spec=py"$(gs_all)[0,1]") #width_ratios=[0.15, 0.2, 0.1,0.1, 0.2], 

# time series 
ax_ts_WT = fig.add_subplot(py"$(gs_left)[0,0]")
ax_ts_Kd = fig.add_subplot(py"$(gs_left)[0,1]")
ax_ts_Ad = fig.add_subplot(py"$(gs_left)[1,0]")
ax_ts_SKd = fig.add_subplot(py"$(gs_left)[2,1]")
ax_ts_HVAd = fig.add_subplot(py"$(gs_left)[2,0]")
ax_ts_Td = fig.add_subplot(py"$(gs_left)[1,1]")


for ax in [ax_ts_WT, ax_ts_Ad]
    ax = pyplot_fxns.remove_axis_box(ax; s=["top", "right", "bottom"])
    ax.set_ylabel("V (mV)")
end

for ax in [ ax_ts_Kd, ax_ts_Td]
    ax = pyplot_fxns.remove_axis_box(ax; s=["top", "right", "bottom", "left"])
end

for ax in [ax_ts_HVAd]
    ax = pyplot_fxns.remove_axis_box(ax; s=["top", "right"])
    ax.set_ylabel("V (mV)")
    ax.set_xlabel("Time (ms)")
end


for ax in [ax_ts_SKd]
    ax = pyplot_fxns.remove_axis_box(ax; s=["top", "right", "left"])
    ax.set_xlabel("Time (ms)")
end

# Phase plot
ax_AP_phase = fig.add_subplot(py"$(gs_right)[0]")
ax_AP_phase = pyplot_fxns.remove_axis_box(ax_AP_phase; s=["top", "right", ])
ax_AP_phase.set_xlabel("V (mV)")
ax_AP_phase.set_ylabel("dV/dt (mV/ms)")

# fI plot
ax_FI = fig.add_subplot(py"$(gs_right)[1]")
ax_FI = pyplot_fxns.remove_axis_box(ax_FI; s=["top", "right", ])
ax_FI.set_ylabel("Frequency (Hz)")
ax_FI.set_xlabel("Current (pA)")

# plot time series sim
ax_ts_WT = plot_ts(ax_ts_WT, "WT", col_WT; title = "WT")
ax_ts_Kd = plot_ts(ax_ts_Kd, "no_Kd", col_no_Kd; title = L"$-$ Kdr")
ax_ts_Ad = plot_ts(ax_ts_Ad, "no_Ad", col_no_A; title = L"$-$ A")
ax_ts_SKd = plot_ts(ax_ts_SKd, "no_SKd", col_no_SK; title = L"$-$ K(Ca)")
ax_ts_HVAd = plot_ts(ax_ts_HVAd, "no_HVAd", col_no_HVA; title = L"$-$ HVA")
ax_ts_Td = plot_ts(ax_ts_Td, "no_Td", col_no_T; title = L"$-$ T")

# plot AP phase plot sim
ax_AP_phase = plot_AP(ax_AP_phase, "WT", col_WT; alpha=1, ls="-")
ax_AP_phase = plot_AP(ax_AP_phase, "no_Kd", col_no_Kd; alpha=1, ls="-")
ax_AP_phase = plot_AP(ax_AP_phase, "no_Ad", col_no_A; alpha=1, ls="-")
ax_AP_phase = plot_AP(ax_AP_phase, "no_SKd", col_no_SK; alpha=1, ls="-")
ax_AP_phase = plot_AP(ax_AP_phase, "no_HVAd", col_no_HVA; alpha=1, ls="-")
ax_AP_phase = plot_AP(ax_AP_phase, "no_Td", col_no_T; alpha=1, ls="-")


# plot fI plot sim
ax_FI = plot_fI(ax_FI, "WT", col_WT; alpha=1, ls="-")
ax_FI = plot_fI(ax_FI, "no_Kd", col_no_Kd; alpha=1, ls="-")
ax_FI = plot_fI(ax_FI, "no_Ad", col_no_A; alpha=1, ls="-")
ax_FI = plot_fI(ax_FI, "no_SKd", col_no_SK; alpha=1, ls="-")
ax_FI = plot_fI(ax_FI, "no_HVAd", col_no_HVA; alpha=1, ls="-")
ax_FI = plot_fI(ax_FI, "no_Td", col_no_T; alpha=1, ls="-")
ax_FI.set_xticks([-10, 0, 10, 20])


# ax_FI_ins = ax_FI.inset_axes([0.75, 0.2, 0.25, 0.25])#,xtick=[-35, -25] )# xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
ax_FI_ins = ax_FI.inset_axes([0.175, 0.6, 0.275, 0.35])#,xtick=[-35, -25] )# xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
ax_FI_ins = pyplot_fxns.remove_axis_box(ax_FI_ins; s=["top", "right", ])
ax_FI_ins = plot_fI(ax_FI_ins, "WT", col_WT; alpha=1, ls="-")
ax_FI_ins = plot_fI(ax_FI_ins, "no_Kd", col_no_Kd; alpha=1)
ax_FI_ins = plot_fI(ax_FI_ins, "no_Ad", col_no_A; alpha=1)
ax_FI_ins = plot_fI(ax_FI_ins, "no_SKd", col_no_SK; alpha=1)
ax_FI_ins = plot_fI(ax_FI_ins, "no_HVAd", col_no_HVA; alpha=1)
ax_FI_ins = plot_fI(ax_FI_ins, "no_Td", col_no_T; alpha=1)
# ax_FI_ins.set_xticks([-10, 0, 10, 20])
# ax_FI_ins.set_xlim(-1, 1)
# ax_FI_ins.set_ylim(25, 35)
ax_FI_ins.set_xlim(-0.5, 0.5)
# ax_FI_ins.set_ylim(27.5, 35)
ax_FI_ins.set_ylim(17.5, 30)


letters = collect('A':'Z')
ax_ts_WT.text(x_letter, y_letter, "$(letters[1])", transform=ax_ts_WT.transAxes, size=letter_size, weight="bold")
ax_ts_Kd.text(x_letter, y_letter, "$(letters[2])", transform=ax_ts_Kd.transAxes, size=letter_size, weight="bold")
ax_ts_Ad.text(x_letter, y_letter, "$(letters[3])", transform=ax_ts_Ad.transAxes, size=letter_size, weight="bold")
ax_ts_Td.text(x_letter, y_letter, "$(letters[4])", transform=ax_ts_Td.transAxes, size=letter_size, weight="bold")
ax_ts_HVAd.text(x_letter, y_letter, "$(letters[5])", transform=ax_ts_HVAd.transAxes, size=letter_size, weight="bold")
ax_ts_SKd.text(x_letter, y_letter, "$(letters[6])", transform=ax_ts_SKd.transAxes, size=letter_size, weight="bold")
ax_AP_phase.text(x_letter2, y_letter2, "$(letters[7])", transform=ax_AP_phase.transAxes, size=letter_size, weight="bold")
ax_FI.text(x_letter2, y_letter2, "$(letters[8])", transform=ax_FI.transAxes, size=letter_size, weight="bold")

for ax in [ax_ts_WT, ax_ts_Kd, ax_ts_Ad, ax_ts_SKd, ax_ts_HVAd, ax_ts_Td]
    ax.set_xlim(0, 1000)
    ax.set_ylim(-62.5, 5)
end

plt.savefig(plotsdir("Supp_Figure_3_firing_soma_Ih_dend.png"), dpi=600)
plt.savefig(plotsdir("Supp_Figure_3_firing_soma_Ih_dend.pdf"), dpi=600)
plt.savefig(plotsdir("Supp_Figure_3_firing_soma_Ih_dend.eps"), dpi=600)
plt.show()

