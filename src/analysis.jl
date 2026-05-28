
# ============================================================================
# Analysis Module
# ============================================================================
# Purpose: Comprehensive signal analysis and spike statistics for neural simulations
# Functions for: phase synchronization, spike detection, frequency analysis,
# F-I curve analysis, and signal filtering
# ============================================================================

module analysis
    using StaticArrays
    using Peaks
    using LinearAlgebra
    using DSP
    using DirectionalStatistics
    using Statistics, StatsBase
    using FFTW, DSP
    using CircStats
    using Statistics: mean
    using StatsBase, StatsAPI, Distributions
    export phase_diff,time_converge_reanalysis, firing_F_analysis, step_FI_analysis, sol_F_analysis, chirp_timeseries, Chirp_compart_F_analysis, OU_frequency_filt, OU_frequency_filt_win, OU_time_series, amplitude_binned, rolling_sum, calc_χ
 

    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    # Moving average with adaptive window size
    # Starts with window size 1 and increases to n as it progresses
    # Useful for smoothing signals at the beginning with limited data
    movingaverage(g, n) = [i-n < 1 ? mean(g[1:i]) : mean(g[i-n:i]) for i in 1:length(g)]


    # Moving Spearman correlation with adaptive window and tie handling
    # Computes rolling Spearman correlation, expanding window at start and handling tied values
    # Args: g = signal vector, n = window size
    # Returns: Spearman correlation coefficients for each window position
    function movingspearman_ties(g, n) 
        y = [i-n < 1  ? corspearman(collect(1:1:i) , g[1:i]) : corspearman(collect(1:1:n), g[(i-n+1):i]) for i in 1:length(g)]
        for i in 1:length(g)
            if i-n < 1 
                if length(unique(g[1:i])) == 1
                    y[i] = 0.
                end
            else
                if length(unique(g[(i-n+1):i])) == 1
                    y[i] = 0.
                end
            end
        end
        return y
    end


    # Run-Length Encoding with Range Information
    # Identifies runs of identical values and their index ranges
    # Args: v = input vector
    # Returns: (values, lengths, start_indices, end_indices, ranges)
    function rle_ranges(v::AbstractVector{T}) where T
        # Based on StatsBase.rle: https://juliastats.org/StatsBase.jl/v0.32/misc/#StatsBase.rle
        n = length(v)
        vals = T[]
        lens = Int[]
    
        n>0 || return (vals,lens)
    
        cv = v[1]
        cl = 1
        istart = [1]
        iend = Int[]
    
    
        i = 2
        @inbounds while i <= n
            vi = v[i]
            if isequal(vi, cv)
                cl += 1
            else
                push!(vals, cv)
                push!(lens, cl)
                push!(iend, i-1)
                push!(istart, sum(lens[1:end])+1)
                cv = vi
                cl = 1
            end
            i += 1
        end
    
        # the last section
        push!(vals, cv)
        push!(lens, cl)
        push!(iend, i-1)
    
        # get ranges
        ranges = []
        for i in 1:length(istart)
            push!(ranges, istart[i]:iend[i])
        end
        return (vals, lens, istart, iend, ranges)
    end
    

    # ========================================================================
    # SPIKE SYNCHRONIZATION AND PHASE ANALYSIS
    # ========================================================================
    
    # Phase difference analysis between two spike trains
    # Computes spike-triggered phase differences and synchronization metrics
    # Args:
    #   s1, s2: signal vectors from two neurons
    #   t: time vector
    #   hilb_spike_num: number of spikes for Hilbert analysis (default=2)
    #   threshold: phase convergence threshold in radians (default=0.1°)
    #   rho_threshold: Spearman correlation threshold (default=0.9)
    #   n_spikes_thresh: window size for moving statistics (default=10)
    # Returns: (phaseps, t1ps, t2ps, ISI1ps, PLV, Hilbert_phase_diff, 
    #           mean_peak_phase_hilb_spikes, phases_diff_diff_avg, time_converg)
    function phase_diff(s1, s2, t; hilb_spike_num=2, threshold = 0.1*π /180, rho_threshold = 0.9, n_spikes_thresh = 10);
        xpks1ps = @views argmaxima(s1)
        (peaks1ps, _) = @views peakproms(xpks1ps, s1, strict=true, minprom=25, maxprom=nothing)
        tpeaks1ps = @views t[peaks1ps]
        xpks2ps = @views argmaxima(s2)
        (peaks2ps, _) = @views peakproms(xpks2ps, s2, strict=true, minprom=25, maxprom=nothing)
        tpeaks2ps = @views t[peaks2ps]
        if @views length(peaks1ps) <= length(peaks2ps) && length(peaks1ps) > 1 && length(peaks2ps) >1
            ISI1ps = zeros(length(peaks1ps)-1)
            for i in range(2,length(peaks1ps)); @views ISI1ps[i-1] = (tpeaks1ps[i]- tpeaks1ps[i-1])  end
            t1ps = @views tpeaks1ps[1:length(peaks1ps)]
            t2ps = @views tpeaks2ps[1:length(peaks1ps)]
            phaseps = @views (t2ps[2:end].-t1ps[2:end])./ISI1ps .*2*pi
            # hilb_ind = peaks1ps[end-hilb_spike_num]
            if @views length(peaks1ps) >= hilb_spike_num
                hilb_ind = @views peaks1ps[end-hilb_spike_num]
            else
                println("Number of spikes < hilb_spike_num")
                hilb_ind = @views peaks1ps
            end
            # Hilbert analysis
            ph1 = @views hilbert(s1[hilb_ind:end] .- mean(s1[hilb_ind:end]))
            ph2 = @views hilbert(s2[hilb_ind:end] .- mean(s2[hilb_ind:end]))
            PLV  = @views 1 - abs.(CircStats.circ_var(angle.(ph1) - angle.(ph2))[1]) # PLV
            Hilbert_phase_diff  = @views angle.(complex(CircStats.circ_mean(real.((ph1).-(ph2)))[1], CircStats.circ_mean(imag.((ph1).-(ph2)))[1]))# angle
            mean_peak_phase_hilb_spikes = CircStats.circ_mean(phaseps[end-5:end])[1]
            phases_diff_diff_avg = movingaverage(CircStats.circ_dist(phaseps[1:end-1], phaseps[2:end]), n_spikes_thresh)
            below_thresh = phases_diff_diff_avg .<= threshold
            (vals, lens, istart, iend, r) = rle_ranges(below_thresh)
            

            below_rho_thresh = abs.(analysis.movingspearman_ties(phaseps, n_spikes_thresh)) 
            rho = abs.(analysis.movingspearman_ties(phaseps, n_spikes_thresh)) #25))
            cond = (phases_diff_diff_avg .<= threshold) .* (rho .<= rho_threshold)[2:end]
            (vals, lens, istart, iend, r) = analysis.rle_ranges(cond)

            if vals[end] # if cond is true
                time_converg = t1ps[r[end][1]]
            else # if cond is false
                time_converg = NaN
            end

        elseif @views length(peaks1ps) > 1 && length(peaks2ps) > 1
            ISI1ps = @views zeros(length(peaks2ps)-1)
            for i in range(2,length(peaks2ps)); @views ISI1ps[i-1] = (tpeaks1ps[i]- tpeaks1ps[i-1])  end
            t1ps = @view tpeaks1ps[1:length(peaks2ps)]
            t2ps = @view tpeaks2ps[1:length(peaks2ps)]
            phaseps = @views (t2ps[2:end] .- t1ps[2:end]) ./ ISI1ps .*2*pi
            if @views length(peaks2ps) >= hilb_spike_num
                hilb_ind = @views peaks2ps[end-hilb_spike_num]
            else
                println("Number of spikes < hilb_spike_num")
                hilb_ind = @views peaks2ps
            end
            # Hilbert analysis
            ph1 = @views hilbert(s1[hilb_ind:end] .- mean(s1[hilb_ind:end]))
            ph2 = @views hilbert(s2[hilb_ind:end] .- mean(s2[hilb_ind:end]))
            PLV  = @views 1 - abs.(CircStats.circ_var(angle.(ph1) - angle.(ph2))[1]) # PLV
            Hilbert_phase_diff  = @views angle.(complex(CircStats.circ_mean(real.((ph1).-(ph2)))[1], CircStats.circ_mean(imag.((ph1).-(ph2)))[1]))# angle
            mean_peak_phase_hilb_spikes = CircStats.circ_mean(phaseps[end-5:end])[1]
            phases_diff_diff_avg = movingaverage(CircStats.circ_dist(phaseps[1:end-1], phaseps[2:end]), n_spikes_thresh)
            below_thresh = phases_diff_diff_avg .<= threshold
            (vals, lens, istart, iend, r) = rle_ranges(below_thresh)
            below_rho_thresh = abs.(analysis.movingspearman_ties(phaseps, n_spikes_thresh)) 
            rho = abs.(analysis.movingspearman_ties(phaseps, n_spikes_thresh))
            cond = (phases_diff_diff_avg .<= threshold) .* (rho .<= rho_threshold)[2:end]
            (vals, lens, istart, iend, r) = analysis.rle_ranges(cond)

            if vals[end] # if cond is true
                time_converg = t1ps[r[end][1]]
            else # if cond is false
                time_converg = NaN
            end

        else
            @warn "peaks <= 1, all outputs set to NaN"
            phaseps = [NaN]
            t1ps = [NaN]
            t2ps = [NaN]
            ISI1ps = [NaN]
            PLV = NaN
            Hilbert_phase_diff = NaN
            mean_peak_phase_hilb_spikes = NaN
            phases_diff_diff_avg = [NaN]
            time_converg = NaN
        end
        GC.gc()
        return  phaseps, t1ps, t2ps, ISI1ps, PLV, Hilbert_phase_diff, mean_peak_phase_hilb_spikes, phases_diff_diff_avg, time_converg
    end


    # Phase difference analysis with start time constraint
    # Same as phase_diff but only analyzes spikes after tstart
    # Args: (same as phase_diff) + tstart = start time for analysis (default=0)
    # Returns: (same as phase_diff)
    function phase_diff_tstart(s1, s2, t; hilb_spike_num=2, threshold = 0.1*π /180, rho_threshold = 0.9, n_spikes_thresh = 10, tstart=0.);
        t_ind = t .>= tstart


        xpks1ps = @views argmaxima(s1[t_ind])
        (peaks1ps, _) = @views peakproms(xpks1ps, s1[t_ind], strict=true, minprom=25, maxprom=nothing)
        tpeaks1ps = @views t[t_ind][peaks1ps]
        xpks2ps = @views argmaxima(s2[t_ind])
        (peaks2ps, _) = @views peakproms(xpks2ps, s2[t_ind], strict=true, minprom=25, maxprom=nothing)
        tpeaks2ps = @views t[t_ind][peaks2ps]
        if @views length(peaks1ps) <= length(peaks2ps) && length(peaks1ps) > 2
            ISI1ps = zeros(length(peaks1ps)-1)
            for i in range(2,length(peaks1ps)); @views ISI1ps[i-1] = (tpeaks1ps[i]- tpeaks1ps[i-1])  end
            t1ps = @views tpeaks1ps[1:length(peaks1ps)]
            t2ps = @views tpeaks2ps[1:length(peaks1ps)]
            phaseps = @views (t2ps[2:end].-t1ps[2:end])./ISI1ps .*2*pi
            # hilb_ind = peaks1ps[end-hilb_spike_num]
            if @views length(peaks1ps) >= hilb_spike_num
                hilb_ind = @views peaks1ps[end-hilb_spike_num]
            else
                println("Number of spikes < hilb_spike_num")
                hilb_ind = @views peaks1ps
            end
            # Hilbert analysis
            ph1 = @views hilbert(s1[t_ind][hilb_ind:end] .- mean(s1[t_ind][hilb_ind:end]))
            ph2 = @views hilbert(s2[t_ind][hilb_ind:end] .- mean(s2[t_ind][hilb_ind:end]))
            PLV  = @views 1 - abs.(CircStats.circ_var(angle.(ph1) - angle.(ph2))[1]) # PLV
            Hilbert_phase_diff  = @views angle.(complex(CircStats.circ_mean(real.((ph1).-(ph2)))[1], CircStats.circ_mean(imag.((ph1).-(ph2)))[1]))# angle
            mean_peak_phase_hilb_spikes = CircStats.circ_mean(phaseps[end-5:end])[1]
            phases_diff_diff_avg = movingaverage(CircStats.circ_dist(phaseps[1:end-1], phaseps[2:end]), n_spikes_thresh)
            below_thresh = phases_diff_diff_avg .<= threshold
            (vals, lens, istart, iend, r) = rle_ranges(below_thresh)
            

            below_rho_thresh = abs.(analysis.movingspearman_ties(phaseps, n_spikes_thresh))
            rho = abs.(analysis.movingspearman_ties(phaseps, n_spikes_thresh))
            cond = (phases_diff_diff_avg .<= threshold) .* (rho .<= rho_threshold)[2:end]
            (vals, lens, istart, iend, r) = analysis.rle_ranges(cond)

            if vals[end] # if cond is true
                time_converg = t1ps[r[end][1]]
            else # if cond is false
                time_converg = NaN
            end
        elseif @views length(peaks1ps) >= 2 && length(peaks2ps) >=2
            ISI1ps = @views zeros(length(peaks2ps)-1)
            for i in range(2,length(peaks2ps)); @views ISI1ps[i-1] = (tpeaks1ps[i]- tpeaks1ps[i-1])  end
            t1ps = @view tpeaks1ps[1:length(peaks2ps)]
            t2ps = @view tpeaks2ps[1:length(peaks2ps)]
            phaseps = @views (t2ps[2:end] .- t1ps[2:end]) ./ ISI1ps .*2*pi
            if @views length(peaks2ps) >= hilb_spike_num
                hilb_ind = @views peaks2ps[end-hilb_spike_num]
            else
                println("Number of spikes < hilb_spike_num")
                hilb_ind = @views peaks2ps
            end
            # Hilbert analysis
            ph1 = @views hilbert(s1[t_ind][hilb_ind:end] .- mean(s1[t_ind][hilb_ind:end]))
            ph2 = @views hilbert(s2[t_ind][hilb_ind:end] .- mean(s2[t_ind][hilb_ind:end]))
            PLV  = @views 1 - abs.(CircStats.circ_var(angle.(ph1) - angle.(ph2))[1]) # PLV
            Hilbert_phase_diff  = @views angle.(complex(CircStats.circ_mean(real.((ph1).-(ph2)))[1], CircStats.circ_mean(imag.((ph1).-(ph2)))[1]))# angle
            mean_peak_phase_hilb_spikes = CircStats.circ_mean(phaseps[end-5:end])[1]
            phases_diff_diff_avg = movingaverage(CircStats.circ_dist(phaseps[1:end-1], phaseps[2:end]), n_spikes_thresh)
            below_thresh = phases_diff_diff_avg .<= threshold
            (vals, lens, istart, iend, r) = rle_ranges(below_thresh)
            below_rho_thresh = abs.(analysis.movingspearman_ties(phaseps, n_spikes_thresh))
            rho = abs.(analysis.movingspearman_ties(phaseps, n_spikes_thresh)) #25))
            cond = (phases_diff_diff_avg .<= threshold) .* (rho .<= rho_threshold)[2:end]
            (vals, lens, istart, iend, r) = analysis.rle_ranges(cond)

            if vals[end] # if cond is true
                time_converg = t1ps[r[end][1]]
            else # if cond is false
                time_converg = NaN
            end
        else
            @warn "peaks <= 1, all outputs set to NaN"
            phaseps = [NaN]
            t1ps = [NaN]
            t2ps = [NaN]
            ISI1ps = [NaN]
            PLV = NaN
            Hilbert_phase_diff = NaN
            mean_peak_phase_hilb_spikes = NaN
            phases_diff_diff_avg = [NaN]
            time_converg = NaN
        end
        GC.gc()
        return  phaseps, t1ps, t2ps, ISI1ps, PLV, Hilbert_phase_diff, mean_peak_phase_hilb_spikes, phases_diff_diff_avg, time_converg
    end

    # Find action potential nearest to a specified time
    # Args: sol = ODE solution object, tsyn = target time, ind = compartment index (default=1)
    # Returns: (tsyn_AP, ISI_after) - AP time and ISI following that AP
    function find_AP_near_tsyn(sol, tsyn; ind=1)
        xpks = argmaxima(sol[ind,:])
        (peaks, proms) =peakproms(xpks, sol[ind,:], strict=true, minprom=25, maxprom=nothing)
        spiket = sol.t[peaks]
        closest_AP_ind = findmin(abs.(spiket .- tsyn))[2] # Find AP closest to tsyn
        ISI_after = spiket[closest_AP_ind+1] - spiket[closest_AP_ind] # ISI after that AP
        tsyn_AP = spiket[closest_AP_ind] # Time of closest AP
        return tsyn_AP, ISI_after
    end

    # ========================================================================
    # FIRING RATE AND F-I CURVE ANALYSIS
    # ========================================================================
    
    # Analyze firing frequency over a time window
    # Args: sol = ODE solution, t_start/t_end = time window, ind = compartment index
    # Returns: (F, ISI, Finst, peakamp, peak_proms)
    #   F = mean firing frequency, ISI = interspike intervals,
    #   Finst = instantaneous frequency, peakamp = spike amplitudes
    function firing_F_analysis(sol, t_start, t_end; ind=1)
        t_ind = t_start .<= sol.t .<= t_end
        xpks = argmaxima(sol[ind,t_ind])
        (peaks, peak_proms) =peakproms(xpks, sol[ind,t_ind], strict=true, minprom=25, maxprom=nothing)
        if length(peaks) >= 2 # If more than 2 peaks detected
            peakamp = sol[1,t_ind][peaks]
            spiket = sol.t[t_ind][peaks]
            ISI = zeros(length(peaks)-1)
            for i in range(2,length(peaks)); ISI[i-1] = (spiket[i]- spiket[i-1])  end
            Finst = 1 ./ (ISI/1000)
            F = mean(Finst)
        else
            F = 0.
            peakamp = 0.
            Finst = 0.
            ISI =[0]
            peak_proms = 0.
        end
        return F, ISI, Finst, peakamp, peak_proms
    end

    # Analyze firing frequency during a current step stimulus
    # Args: sol = ODE solution, step_start/step_end = stimulus window, ind = compartment index
    # Returns: (F, ISI, Finst, peakamp)
    function step_FI_analysis(sol, step_start, step_end; ind=1)
        t_ind = step_start .<= sol.t .<= step_end
        xpks = argmaxima(sol[ind,t_ind])
        (peaks, _) =peakproms(xpks, sol[ind,t_ind], strict=true, minprom=25, maxprom=nothing)
        if length(peaks) >= 2 # If more than 2 peaks detected
            peakamp = sol[1,t_ind][peaks]
            spiket = sol.t[t_ind][peaks]
            ISI = zeros(length(peaks)-1)
            for i in range(2,length(peaks)); ISI[i-1] = (spiket[i]- spiket[i-1])  end
            Finst = 1 ./ (ISI/1000)
            F = mean(Finst)
        else
            F = 0.
            peakamp = 0.
            ISI = [0.]
            Finst = [0.]
        end
        return F, ISI, Finst, peakamp
    end


    # ========================================================================
    # FREQUENCY DOMAIN ANALYSIS
    # ========================================================================
    
    # FFT-based frequency analysis of solution
    # Args: j = trial index, sol = ODE solution, dt = time step (ms), 
    #       i = compartment index, size_ind = dimension for size extraction
    # Returns: (frequencies, power) - positive frequencies and FFT magnitudes
    function sol_F_analysis(j, sol, dt; i = 1, size_ind = 3)
        N = size(sol)[size_ind] # Number of time points
        Ts = dt /1000 # seconds # Sample period
        t0 = 0 # Start time
        tmax = t0 + N * Ts -Ts
        t = t0:Ts:tmax # time coordinate
        Freqs = fftfreq(length(t), 1.0/Ts) |> fftshift
    
        F_ind = Int(floor(size(Freqs)[1]/2))+1
        F = (fft(sol[i, j,:] .- mean(sol[i, j,:])) |> fftshift)[F_ind:end]
        return Freqs[F_ind:end], F
    end


    # Load chirp stimulus response timeseries from file
    # Args: filename = path to JLD2 file containing solution
    # Returns: (t, soma, chirp) - time vector, soma voltage, dendritic response
    function chirp_timeseries(filename)
        jldopen(filename, "r") do file
                sol = file["sol"]
                t = sol.t/1000 # Convert time to seconds
                soma = sol[1,1,:] # Soma voltage (compartment 1)
                chirp = sol[2,end,:] # Dendrite response (last position)
                return t, soma, chirp
        end
    end


    # Frequency analysis of compartmental response to chirp stimulus
    # Args: i = compartment index, f = flag (true=load from file, false=use provided sol),
    #       filename = file path if f=true, sol = solution object if f=false, dt = time step
    # Returns: (frequencies, power) - positive frequencies and FFT magnitudes
    function Chirp_compart_F_analysis(i,f;filename= "", sol=sol, dt=0.1)
        if f
            jldopen(filename, "r") do file
                sol = file["sol"]
                N = size(sol)[3] # Number of time points
                Ts = dt /1000 # seconds # Sample period
                t0 = 0 # Start time
                tmax = t0 + N * Ts -Ts
                t = t0:Ts:tmax # time coordinate
                Freqs = fftfreq(length(t), 1.0/Ts) |> fftshift

                F_ind = Int(floor(size(Freqs)[1]/2))+1
                F = (fft(sol[1,i,:] .- mean(sol[1,i,:])) |> fftshift)[F_ind+1:end]
                return Freqs[F_ind+1:end], F

            end
        else
            N = size(sol)[2]# Number of points
            Ts = dt /1000 # seconds # Sample period
            t0 = 0 # Start time
            tmax = t0 + N * Ts -Ts
            t = t0:Ts:tmax # time coordinate
            Freqs = fftfreq(length(t), 1.0/Ts) |> fftshift

            F_ind = Int(floor(size(Freqs)[1]/2))+1
            F = (fft(sol[i,:] .- mean(sol[i,:])) |> fftshift)[F_ind+1:end]
            return Freqs[F_ind+1:end], F
        end
    end


    # Load Ornstein-Uhlenbeck process simulation data
    # Args: filename = JLD2 file path, n_ind = trial index, 
    #       start_ind/end_ind = data range indices
    # Returns: (soma, last, noise, time) - voltage and noise signals with time vector
    function OU_time_series(filename,n_ind, start_ind, end_ind)
        jldopen(filename, "r") do file
            soma = file["soma"][n_ind, start_ind:end_ind] # Soma voltage
            Last = file["last"][n_ind, start_ind:end_ind] # Last compartment voltage
            noise = file["noise"][n_ind, start_ind:end_ind] # Noise input
            time = (file["time"][n_ind, start_ind:end_ind] .- file["time"][n_ind, start_ind]) ./ 1000 # Time (seconds)
            return  soma, Last, noise, time
        end
    end


    # Create amplitude histograms binned by voltage value
    # Compute probability distributions for soma, dendrite, and noise signals
    # Args: bin_low/high/incr = histogram bin range and spacing,
    #       f = flag (true=load from file, false=use provided data)
    # Returns: (hsoma, hlast, hnoise, bin_middle, n) - normalized histograms and bin centers
    function amplitude_binned(bin_low, bin_high, bin_incr,f;filename= "", soma=soma, first=first, last=last, noise=noise)
        if f
            jldopen(filename, "r") do file
                soma = file["soma"]
                last = file["last"]
                noise = file["noise"]
                bins = collect(bin_low:bin_incr:bin_high) # Create histogram bins
        
                n = size(soma)[1]
                hsoma = zeros(n, size(bins)[1]-1)
                hlast = zeros(n, size(bins)[1]-1)
                hnoise = zeros(n, size(bins)[1]-1)
        
                for i in range(1, n)
                    hsoma_fit= fit(Histogram,soma[i,:],bins, closed=:left) #
                    hlast_fit= fit(Histogram,last[i,:],bins, closed=:left) #
                    hnoise_fit= fit(Histogram,noise[i,:],bins, closed=:left) #
                    hsoma[i,:] = normalize(hsoma_fit, mode=:density).weights
                    hlast[i,:] = normalize(hlast_fit, mode=:density).weights
                    hnoise[i,:] = normalize(hnoise_fit, mode=:density).weights
                end
                bin_middle = collect(bin_low+bin_incr:bin_incr:bin_high);
                return hsoma, hlast, hnoise, bin_middle,n
            end
        else
            bins = collect(bin_low:bin_incr:bin_high);
            n = size(soma)[1]
            nsamp = size(soma)[2]
            w = weights(ones(nsamp, 1)/nsamp)
            hsoma = zeros(n, size(bins)[1]-1)
            hfirst = zeros(n, size(bins)[1]-1)
            hlast = zeros(n, size(bins)[1]-1)
            hnoise = zeros(n, size(bins)[1]-1)

            for i in range(1, n)
                hsoma_fit= fit(Histogram,soma[i,:], w, bins, closed=:left) #
                hfirst_fit= fit(Histogram,first[i,:], w, bins, closed=:left) #
                hlast_fit= fit(Histogram,last[i,:], w, bins, closed=:left) #
                hnoise_fit= fit(Histogram,noise[i,:], w, bins, closed=:left) #
                hsoma[i,:] = hsoma_fit.weights
                hfirst[i,:] = hfirst_fit.weights
                hlast[i,:] = hlast_fit.weights
                hnoise[i,:] = hnoise_fit.weights
            end
            bin_middle = collect(bin_low+bin_incr:bin_incr:bin_high);
            return hsoma, hfirst, hlast, hnoise, bin_middle,n
        end
    end
        
        
    # Frequency domain filtering and power analysis for OU process
    # Computes power spectral density ratios (soma/dendrite) across frequency range
    # Args: soma/first/last/noise = signal arrays, dt = time step (ms), start_t = analysis start time (ms)
    # Returns: (power_F, power, power_SEM, power_F_first, power_first, power_SEM_first)
    #   Power ratios and standard error of the mean at each frequency
    function OU_frequency_filt(; soma=soma,first=first, noise=noise,last=last, dt=0.0001, start_t = 25) 
        n = size(soma)[1] # Number of trials
        start_ind = Int(start_t/(dt*1000)) # Start index (convert ms to sample index)

        soma_i = soma[:, start_ind:end] # Truncate signals to start from start_ind
        N = size(soma_i)[2] # Number of time points
        Ts = dt # Sample period (seconds)
        t0 = 0 # Start time
        tmax = t0 + N * Ts - Ts
        t = t0:Ts:tmax # Time coordinate
        freqs = fftfreq(length(t), 1.0/Ts) |> fftshift # Compute frequency vector

        # Compute FFT for all trials and signals
        Fsoma_ind = zeros(n, size(freqs)[1])
        Fsoma_ind = complex(Fsoma_ind)
        Ffirst_ind = zeros(n, size(freqs)[1])
        Ffirst_ind = complex(Ffirst_ind)
        Flast_ind = zeros(n, size(freqs)[1])
        Flast_ind = complex(Flast_ind)
        Fnoise_ind = zeros(n, size(freqs)[1])
        Fnoise_ind = complex(Fnoise_ind)
        
        for i in range(1, n)
            # Compute FFT with DC component removed (mean subtracted)
            Fsoma_ind[i,:]  = fft(soma[i,start_ind:end] .- mean(soma[i, start_ind:end])) |> fftshift
            Fnoise_ind[i,:]  = fft(noise[i,start_ind:end] .- mean(noise[i, start_ind:end])) |> fftshift
            Flast_ind[i,:]  = fft(last[i,start_ind:end] .- mean(last[i, start_ind:end])) |> fftshift
            Ffirst_ind[i,:]  = fft(first[i,start_ind:end] .- mean(first[i, start_ind:end])) |> fftshift
        end
        positive_F_ind = Int(floor(size(abs.(Fsoma_ind) ./ abs.(Fnoise_ind))[2]/2))+1 # Index for positive frequencies
        # Soma/dendrite (last) power ratio in dB
        power = mean(DSP.pow2db.(abs.(Fsoma_ind[:, positive_F_ind:end]) ./ abs.(Flast_ind[:, positive_F_ind:end])), dims= 1)
        power_SEM = std(DSP.pow2db.(abs.(Fsoma_ind[:, positive_F_ind:end]) ./ abs.(Flast_ind[:, positive_F_ind:end])), dims=1) / sqrt(n)
        power_F = freqs[positive_F_ind:end]

        # First compartment/dendrite (last) power ratio in dB
        power_first = mean(DSP.pow2db.(abs.(Ffirst_ind[:, positive_F_ind:end]) ./ abs.(Flast_ind[:, positive_F_ind:end])), dims= 1)
        power_SEM_first = std(DSP.pow2db.(abs.(Ffirst_ind[:, positive_F_ind:end]) ./ abs.(Flast_ind[:, positive_F_ind:end])), dims=1) / sqrt(n)
        power_F_first = freqs[positive_F_ind:end]



        return  power_F, power, power_SEM, power_F_first, power_first, power_SEM_first 
    end  
       
    
    # Rolling sum with adaptive window
    # Computes sum over ±n elements, adapting window at boundaries
    # Args: a = input vector, n = half-window size
    # Returns: vector of rolling sums with size matching input
    function rolling_sum(a, n::Int)
        @assert 1<=n<=length(a)
        out = similar(a, length(a))
        out[1] = sum(a[1:n])
        for i in eachindex(out)[2:end]
            # Adjust start index to stay within bounds
            if i-n < 1
                ind_start = 1
            else
                ind_start = i-n
            end
            # Adjust end index to stay within bounds
            if i+n > length(a)
                ind_end = length(a)
            else
                ind_end = i+n
            end
            out[i] = sum(a[ind_start:ind_end])
        end
        return out
    end

end
