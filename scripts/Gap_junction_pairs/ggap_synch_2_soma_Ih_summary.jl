# ============================================================================
# Gap Junction Synchronization Analysis - Summary Script
# ============================================================================
# Purpose: Aggregate and summarize gap junction conductance (ggap) 
# synchronization properties across different mutation conditions
# Results are collected from individual simulation files and saved as summary
# ============================================================================

using DrWatson
@quickactivate  "MLI_synch_2026"

# Include files from the source directory
include(srcdir("CSC_cable_2.jl"))
using .CSC_cable_2
include(srcdir("analysis.jl"))
using .analysis
using DifferentialEquations, DiffEqCallbacks, JLD2, Statistics, Peaks, FHist
using CircStats
using ProgressMeter


# ============================================================================
# Analysis Conditions and Parameters
# ============================================================================
# Define genetic conditions to analyze (wildtype + channel deletions)
cond_string_array = ["WT", "no_Kd", "no_Ad", "no_SKd", "no_HVAd", "no_Td"]

# ============================================================================
# Simulation Parameters
# ============================================================================
# Simulation time window (in milliseconds)
tspan = (0.0, 30*1000.0)

# Phase difference parameters
n_phase = 12  # Number of initial phase differences to sample
init_phase_diff_array = collect(0:1/n_phase:1 - 1/n_phase)  # Initial phase differences (0 to ~1, normalized)

# Spatial sampling
p_int = 5  # Position interval for subsampling (sample every 5th position along dendrite)
position_array = collect(1:p_int:51)  # Positions along the dendrite (1 to 51, step 5)

# Spike detection parameter
hilb_spike_num = 5  # Number of spikes for Hilbert transform analysis

# ============================================================================
# Gap Junction Conductance Values (logarithmic spacing)
# ============================================================================
# Test range of gap junction conductances on a log scale


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

# ============================================================================
# Main Analysis Loop: Process Each Condition
# ============================================================================
# Iterate through each genetic condition, aggregating results from all ggap 
# conductance simulations

for condition_string in cond_string_array
    # ========================================================================
    # Pre-allocate storage arrays for all results
    # Dimensions: [initial_phase_diff, ggap_conductance, dendrite_position]
    # ========================================================================
    PLV_array = zeros(length(init_phase_diff_array),length(ggap_array),length(position_array))
    phase_diff_end_array = zeros(length(init_phase_diff_array),length(ggap_array),length(position_array))
    Hilbert_phase_diff_array = zeros(length(init_phase_diff_array),length(ggap_array),length(position_array))
    mean_peak_phase_end_array = zeros(length(init_phase_diff_array),length(ggap_array),length(position_array))
    spike_times = Array{Array{Float64}}(undef,length(init_phase_diff_array),length(ggap_array),length(position_array),2)
    phase_diff_array = Array{Array{Float64}}(undef,length(init_phase_diff_array),length(ggap_array),length(position_array))
    time_converg_array = Array{Array{Float64}}(undef,length(init_phase_diff_array),length(ggap_array),length(position_array))
    #%
    for ggap_ind in 1:length(ggap_array)
        # Track progress through ggap values
        println(ggap_ind)
        
        # ====================================================================
        # Define simulation parameters for this ggap value
        # ====================================================================
        param_i = Dict(
        "n_phase" => n_phase, 
        "pos_int" => p_int,
        "hilb_spike_num" => hilb_spike_num,
        "tend" => tspan[2]/1000, 
        "ggap" => ggap_array[ggap_ind],
            )

        # ====================================================================
        # Load simulation data for this ggap value from file
        # File contains results from all phase differences and positions
        # ====================================================================
        phase_diff_array_i = wload(datadir("simulations", "gapjxn_charac",folder, savename("ggap_synch_2_properties_CSC_soma_Ih_time_synch_$(condition_string)", param_i, "jld2")), "phase_diff_array")
        spike_times_i = wload(datadir("simulations", "gapjxn_charac",folder, savename("ggap_synch_2_properties_CSC_soma_Ih_time_synch_$(condition_string)", param_i, "jld2")), "spike_times")
        PLV_array_i = wload(datadir("simulations", "gapjxn_charac",folder, savename("ggap_synch_2_properties_CSC_soma_Ih_time_synch_$(condition_string)", param_i, "jld2")), "PLV_array")
        Hilbert_phase_diff_array_i = wload(datadir("simulations", "gapjxn_charac",folder, savename("ggap_synch_2_properties_CSC_soma_Ih_time_synch_$(condition_string)", param_i, "jld2")), "Hilbert_phase_diff_array")
        mean_peak_phase_end_array_i = wload(datadir("simulations", "gapjxn_charac",folder, savename("ggap_synch_2_properties_CSC_soma_Ih_time_synch_$(condition_string)", param_i, "jld2")), "mean_peak_phase_end_array")
        time_converg_array_i = wload(datadir("simulations", "gapjxn_charac",folder, savename("ggap_synch_2_properties_CSC_soma_Ih_time_synch_$(condition_string)", param_i, "jld2")), "time_converg_array")

        # ====================================================================
        # Aggregate loaded data into summary arrays
        # Extract final phase difference values and store all metrics
        # ====================================================================
        # Extract final phase difference at end of simulation for each condition
        phase_diff_end_array[:, ggap_ind, :] =  [v[end] for (i,v) in pairs(phase_diff_array_i)]
        phase_diff_array[:, ggap_ind, :] = phase_diff_array_i
        spike_times[:, ggap_ind, :, :] = spike_times_i
        PLV_array[:, ggap_ind, :] = PLV_array_i
        Hilbert_phase_diff_array[:, ggap_ind, :] = Hilbert_phase_diff_array_i
        mean_peak_phase_end_array[:, ggap_ind, :] = mean_peak_phase_end_array_i
        time_converg_array[:, ggap_ind, :] = time_converg_array_i

    end

    # ========================================================================
    # Save Summary Results for This Condition
    # ========================================================================
    # Aggregate all results for this mutation condition into a single file
    param = Dict(
        "n_phase" => n_phase, 
        "ggap_max" => maximum(ggap_array),
        "ggap_min" => minimum(ggap_array),
        "pos_int" => p_int,
        "hilb_spike_num" => hilb_spike_num,
        "tend" =>  tspan[2],
    )
    @tagsave(datadir("simulations", "gapjxn_charac", savename("ggap_synch_2_properties_SUMMARY_CSC_soma_Ih_time_synch_$(condition_string)", param, "jld2")),  @strdict phase_diff_array spike_times PLV_array Hilbert_phase_diff_array mean_peak_phase_end_array init_phase_diff_array position_array ggap_array time_converg_array phase_diff_end_array; safe = DrWatson.readenv("DRWATSON_SAFESAVE", true))
end