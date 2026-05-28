# ============================================================================
# HBP Firing Analysis Data Extraction
# ============================================================================
# Script to extract firing properties from HBP (Human Brain Project) patch-clamp
# recordings of cerebellar stellate cells. Analyzes current-clamp step protocols
# to compute firing rates, spike times, and voltage dynamics.
#
# Reference:
# Locatelli, F., Gagliano, G., & Prestori, F. (2020). Whole cell patch-clamp
# recordings of cerebellar stellate cells [Data set]. EBRAINS.
# https://doi.org/10.25493/M1AQ-3AC
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pyabf
import scipy


# ============================================================================
# FILE LOADING
# ============================================================================
# HBP dataset identifier and experimental protocol (current-clamp step)
folder = 'hbp-d000020_PatchClamp-StellateCells_pub'
protocol = 'ccstep'

# Build file list from directory
file_list = []
dir_path  ='../../data/exp_raw/{}/{}'.format(folder, protocol)
# Iterate through all files in protocol directory
for file_path in os.listdir(dir_path):
    # Check if entry is a file (not directory)
    if os.path.isfile(os.path.join(dir_path, file_path)):
        # Add filename to processing list
        file_list.append(os.path.join(dir_path, file_path))

# Remove problematic file
file_list.remove('../../data/exp_raw/hbp-d000020_PatchClamp-StellateCells_pub/ccstep/161017-1610008-ccstep.abf')
print(file_list)




#%% ========== UTILITY FUNCTION FOR SPIKE DETECTION ============================

def ranges(nums):
    """
    Extract index ranges of consecutive sequences in array.
    
    Finds contiguous regions in an array of indices, useful for identifying
    distinct current injection periods in electrophysiology recordings.
    
    Source: https://stackoverflow.com/a/48106843
    
    Parameters
    ----------
    nums : array-like
        Array of indices to analyze

    Returns
    -------
    list of tuples
        Each tuple (start, end) defines a contiguous index range
    """
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

# ============================================================================
# SIGNAL PROCESSING AND VISUALIZATION
# ============================================================================
import scipy.signal
import matplotlib.pyplot as plt

#%% ========== BATCH PROCESSING ALL CELLS ========================================
# Extract firing data from all cells in the dataset
# Build list of cell identifiers from file paths
cell_list = [i.split('/')[-1].split('.abf')[0] for i in file_list]
# Initialize DataFrame to store results from all cells
summary_df = pd.DataFrame(columns=[ 'cellID', 'filepath', 'I','F','t0', 'V0', 'dV0', 't', 'V', 'dV', ], dtype='object')

# Loop through each cell file and extract firing analysis
for ii in range(len(file_list)):
    print('Processing cell {} ...'.format(ii))
    
    # Load cell data
    f = pyabf.ABF(file_list[ii])
    sweep_count = f.sweepCount
    
    # Extract current magnitude for each sweep
    I_sweeps = []
    for j in range(sweep_count):
        f.setSweep(j)
        if np.abs(f.sweepC.min()) > np.abs(f.sweepC.max()):
            I_sweeps.append(f.sweepC.min())
        else:
            I_sweeps.append(f.sweepC.max())
    I_sweeps = np.array(I_sweeps)
    numsteps = I_sweeps.shape[0]

    
    # Initialize spike analysis arrays
    spike_times = np.zeros((numsteps), dtype=object)
    ISI = np.zeros((numsteps), dtype=object)
    F = np.zeros((numsteps), dtype=object)
    Fmean = np.zeros((numsteps))

    spike_amp = np.zeros((numsteps), dtype=object)

    # Spike detection parameters
    min_spike_height = -40
    prominence = 15
    
    # Analyze each current level
    for stim in range(0, numsteps):
        f.setSweep(stim)  # Load sweep data
        
        # Detect spikes in voltage trace
        peaks, peak_prop = scipy.signal.find_peaks(f.sweepY, height=min_spike_height, prominence=prominence)
        spike_times_all = f.sweepX[peaks]
        # Determine current step timing (use reference sweep if at zero current)
        if I_sweeps[stim] == 0.:  # if not current input i.e. step current = 0, use next step to determine where step would be
            f.setSweep(stim+1)
            nonzeroI = np.where(f.sweepC != 0)[0]
            I_ran = ranges(nonzeroI)
            ind_length = [(i[1] - i[0]) for i in I_ran]
            tlength = [i * (1 / f.sampleRate) for i in ind_length]
            t_thresh = 0.2  # threshold of step length
            I_start = f.sweepX[I_ran[np.where(np.array(tlength) > t_thresh)[0][0]][0]]
            I_end = f.sweepX[I_ran[np.where(np.array(tlength) > t_thresh)[0][0]][1]]
            I_start_ind = I_ran[np.where(np.array(tlength) > t_thresh)[0][0]][0]
            I_end_ind = I_ran[np.where(np.array(tlength) > t_thresh)[0][0]][1]

            f.setSweep(stim)
        else:
            nonzeroI = np.where(f.sweepC  != 0)[0] 
            ind_length = [(i[1] - i[0]) for i in I_ran]
            tlength = [i * (1 / f.sampleRate) for i in ind_length]
            t_thresh = 0.2  # threshold of step length
            I_start = f.sweepX[I_ran[np.where(np.array(tlength) > t_thresh)[0][0]][0]]
            I_end = f.sweepX[I_ran[np.where(np.array(tlength) > t_thresh)[0][0]][1]]
            I_start_ind = I_ran[np.where(np.array(tlength) > t_thresh)[0][0]][0]
            I_end_ind = I_ran[np.where(np.array(tlength) > t_thresh)[0][0]][1]


        spike_I_ind = np.where(np.logical_and(spike_times_all >= I_start, spike_times_all <= I_end))  # find spikes within current step
        if spike_I_ind[0].shape[0] > 1:  # if more than 1 spike can have ISI
            spike_times[stim] = spike_times_all[spike_I_ind]
            ISI[stim] = np.array([x - spike_times[stim][i] if i else None for i, x in enumerate(spike_times[stim][1:])])  # msec
            F[stim] = 1 / (ISI[stim][1:].astype(float))
            Fmean[stim] = np.mean(F[stim][1:])
            spike_amp[stim] = f.sweepY[peaks[spike_I_ind]]
        else:
            Fmean[stim] = 0.

    V_out = np.zeros((numsteps, (I_end_ind - I_start_ind)))
    dV_out = np.zeros((numsteps, (I_end_ind - I_start_ind)))
    t_out = np.zeros((numsteps, (I_end_ind - I_start_ind)))
    
    # Extract voltage traces and their derivatives during current step window
    for stim in range(numsteps):
        if I_sweeps[stim] == 0.:  # if not current input i.e. step current = 0, use next step to determine where step would be
            f.setSweep(stim)
        else:
            f.setSweep(stim)
        V_out[stim] = f.sweepY[I_start_ind:I_end_ind]
        dV_out[stim] = np.gradient(V_out[stim], f.sampleRate)
        t_out[stim] = f.sweepX[I_start_ind:I_end_ind]

    # Extract resting state properties (zero current sweep if available)
    if np.any(I_sweeps == 0.):    
        I0_ind = np.argwhere(I_sweeps == 0)[0][0]
        f.setSweep(I0_ind)
        V_0  = f.sweepY
        dV_0 = np.gradient(f.sweepY,  1 / f.sampleRate)
        t0 = f.sweepX
    print(file_list[ii].split('/')[-1].split('.abf')[0])

    # Append cell data to summary DataFrame
    summary_df = pd.concat([summary_df, pd.DataFrame.from_records([{'cellID': file_list[ii].split('/')[-1].split('.abf')[0],
                                'filepath': file_list[ii], 'I': I_sweeps,'F': Fmean,'t': t_out - (t_out[:, 0] * np.ones(t_out.shape).T).T,
                                'V': V_out, 'dV': dV_out, 't0': t0, 'V0': V_0, 'dV0': dV_0, 'spike_amp': spike_amp}])])


# save
summary_df.reset_index(drop=True, inplace=True)
summary_df.to_json('../../data/exp_pro/HBP_firing_analysis_extract.json')
print('\nAnalysis complete. Data saved to HBP_firing_analysis_extract.json')
