module CSC1_Ih
using DiffEqCallbacks
using DifferentialEquations
using Parameters
using Statistics, Peaks
using LoopVectorization


###########################################################################
#I-h HCN1 channel from Kamilla Angelo, Michael London,Soren R. Christensen, and Michael Hausser 2007 J. of Neurosci.
# used in Rizza et al 2021
const ratetau = 1              # (ms)
const q10 = 3
# : We set the recording temperature here to room temperature as in Angelo et al.,
# : they forgot to mention the recording temperature.
const rec_temp = 23            # (deg)
const ljp = 9.3                # (mV) : liquid_junction_potential
const v_inf_half_noljp = -90.3 # (mV)
const v_inf_k = 9.67           # (mV)
const v_tau_const = 0.0018     # (1)
const v_tau_half1_noljp = -68  # (mV)
const v_tau_half2_noljp = -68  # (mV)
const v_tau_k1 = -22           # (mv)
const v_tau_k2 = 7.14          # (mv)
const qt = q10^((rec_temp - 37) / 10) #(degC)
const v_inf_half = (v_inf_half_noljp - ljp)
const v_tau_half1 = (v_tau_half1_noljp - ljp)
const v_tau_half2 = (v_tau_half2_noljp - ljp)

hinf(v) = 1 / (1 + exp((v - v_inf_half) / v_inf_k))
tauh(v) = (ratetau / (v_tau_const * (exp((v - v_tau_half1) / v_tau_k1) + exp((v - v_tau_half2) / v_tau_k2)))) / qt

const rl = 110 # Ω*cm  -  specific intracellular resisitivity Rizza et al 2021
const capm = 1.50148 # μF/cm^2  -  specific membrane capacitance
const D = 5.3e-9 # μm^2/ms-> cm^2/ms
const A = 2 * 1.602176634e−19 / (1.380649e−20 * 309.15)


################################################################################
# Main ODE Functions for Cable Equation Simulations
################################################################################

"""One compartment cerebellar stellate cell 

State vector r (9 × numseg+1):
  r[1] = voltage (mV) at soma (col 1) and each dendritic compartment
  r[2] = h_Na gating (Na inactivation)
  r[3] = n_K gating (K activation)  
  r[4] = m_A gating (A-current activation)
  r[5] = h_A gating (A-current inactivation)
  r[6] = h_T gating (T-type Ca inactivation)
  r[7] = m_HVA gating (HVA Ca activation)
  r[8] = [Ca2+] intracellular (μM)
  r[9] = h_Ih gating (HCN inactivation)
"""
function CSC1_Ih_I!(dr, r, p, t)
    u = @view r[:, :]
    du = @view dr[:, :]
    gNa, gK, gleak, gA, gT, gHVA, gSK, gh, Eleak,Eh, I = p
    capm = 1.50148 # μF/cm^2  -  specific membrane capacitance

    # steady state gating variables
    mSS = @inbounds @views 1 / (1 + exp((-u[1] - 37) / 5.25));
    mTSS = @inbounds @views 1 ./ (1 .+ exp.(-(u[1] .+ 54) ./ 3));
    @inbounds @views du[1] = (I
                - gK  * (u[3]^4) * (u[1] + 80)
                - gNa  * (mSS^3) * u[2] * (u[1]-55)
                - gleak   * (u[1] - Eleak)
                - gA * u[4] * u[5] * (u[1] + 80)
                - gT  * mTSS * u[6] * (u[1] - 22)
                - gSK  * (u[8]^5 / ((0.45^5) + (u[8]^5))) * (u[1] + 80) 
                - gHVA  * u[7] * (u[1] - 22)
                - gh * u[9] *(u[1] - Eh)
                ) / capm;
    @inbounds @views du[2] = ((1 / (1 + exp((u[1] + 40) / 4))) - u[2]) / ( 0.1 + (2 * 322 * 46) / (4 * pi * (u[1] + 74) ^ 2 + 46 ^ 2));
    @inbounds @views du[3] = ((1 / (1 + exp(-(u[1] + 21) / 6))) - u[3]) / (6 / (1 + exp((u[1] + 23) / 15)));
    @inbounds @views du[4] = ((1 / (1 + exp((-u[1] - 24.5) / 13.2))) - u[4]) / 5;
    @inbounds @views du[5] = ((1 / (1 + exp((u[1] + 79.5) / 6.5))) - u[5]) / 10;
    @inbounds @views du[6] = ((1 / (1 + exp((u[1] + 74) / 3.75))) - u[6]) / 15;
    @inbounds @views du[7] = ((1 / (1 + exp(-(u[1] + 25) / 8))) - u[7]) / 4;
    @inbounds @views du[9] = (hinf(u[1]) - u[9]) / tauh(u[1]) 
    @inbounds @views du[8] = - 0.015 * (0.018 * (gT  * mTSS * u[6] * (u[1] - 22) + gHVA  * u[7] * (u[1] - 22)) + 0.1 * u[8]); 

end


function init_u0(;V0=-60)
    hSS = 1 / (1 + exp((V0 + 37.5) / 3));
    nSS = 1 / (1 + exp(-(V0 + 21) / 6));
    nASS = 1 / (1 + exp((-V0 - 24.5) / 13.2));
    hASS = 1 / (1 + exp((V0 + 79.5) / 6.5));
    hTSS = 1 / (1 + exp((V0 + 74) / 3.75));
    sSS = 1 / (1 + exp(-(V0 + 25) / 8));
    Ca = 0.0001
    return [V0, hSS, nSS, nASS, hASS, hTSS,sSS, Ca, hinf(V0)]

end


function CSC1_Ih_I_LP!(dr, r, p, t)
    u = @view r[:, :]
    du = @view dr[:, :]
    gNa, gK, gleak, gA, gT, gHVA, gSK, gh, Eleak,Eh, I, tau_d = p
    capm = 1.50148 # μF/cm^2  -  specific membrane capacitance

    # steady state gating variables
    mSS = @inbounds @views 1 / (1 + exp((-u[1] - 37) / 5.25));
    mTSS = @inbounds @views 1 ./ (1 .+ exp.(-(u[1] .+ 54) ./ 3));
    @inbounds @views du[1] = (I
                - gK  * (u[3]^4) * (u[1] + 80)
                - gNa  * (mSS^3) * u[2] * (u[1]-55)
                - gleak   * (u[1] - Eleak)
                - gA * u[4] * u[5] * (u[1] + 80)
                - gT  * mTSS * u[6] * (u[1] - 22)
                - gSK  * (u[8]^5 / ((0.45^5) + (u[8]^5))) * (u[1] + 80)
                - gHVA  * u[7] * (u[1] - 22)
                - gh * u[9] *(u[1] - Eh)
                ) / capm;
    @inbounds @views du[2] = ((1 / (1 + exp((u[1] + 37.5) / 3))) - u[2]) / ( 0.1 + (2 * 322 * 46) / (4 * pi * (u[1] + 74) ^ 2 + 46 ^ 2));
    @inbounds @views du[3] = ((1 / (1 + exp(-(u[1] + 21) / 6))) - u[3]) / (6 / (1 + exp((u[1] + 23) / 15)));
    @inbounds @views du[4] = ((1 / (1 + exp((-u[1] - 24.5) / 13.2))) - u[4]) / 5;
    @inbounds @views du[5] = ((1 / (1 + exp((u[1] + 79.5) / 6.5))) - u[5]) / 10;
    @inbounds @views du[6] = ((1 / (1 + exp((u[1] + 74) / 3.75))) - u[6]) / 15;
    @inbounds @views du[7] = ((1 / (1 + exp(-(u[1] + 25) / 8))) - u[7]) / 4;
    @inbounds @views du[9] = (hinf(u[1]) - u[9]) / tauh(u[1]) #hcn

    @inbounds @views du[8] = - 0.015 * (0.018 * (gT  * mTSS * u[6] * (u[1] - 22) + gHVA  * u[7] * (u[1] - 22)) + 0.1 * u[8]); 
    @inbounds @views du[10] = (u[1] - u[10]) / tau_d  #lowpass

end


################################################################################
# Initialization Functions
################################################################################

"""Initialize gating variables to steady state at resting potential V0.
   Used to set initial conditions for ODE simulations.
"""
function init_u0_LP(;V0=-60)
    hSS = 1 / (1 + exp((V0 + 37.5) / 3));
    nSS = 1 / (1 + exp(-(V0 + 21) / 6));
    nASS = 1 / (1 + exp((-V0 - 24.5) / 13.2));
    hASS = 1 / (1 + exp((V0 + 79.5) / 6.5));
    hTSS = 1 / (1 + exp((V0 + 74) / 3.75));
    sSS = 1 / (1 + exp(-(V0 + 25) / 8));
    Ca = 0.0001
    return [V0, hSS, nSS, nASS, hASS, hTSS,sSS, Ca, hinf(V0), V0]

end

"""Evolve cell to steady state at constant holding potential V0.
   Uses callback to terminate integration when state reaches steady state.
   Returns converged gating variable state for use as initial condition.
"""
function steady_state_init(V0, p,tspan)
    prob_ic = ODEProblem(CSC1_Ih_I!, init_u0(;V0=V0), tspan, p, callback=DiffEqCallbacks.TerminateSteadyState(), save_everystep=false, save_end=true, save_start=false)
    sol_ic = solve(prob_ic, ROCK2(), abstol = 1e-12, reltol = 1e-10, maxiters=1e25);

    return sol_ic.u[1]
    nothing
end


"""Steady-state initialization with low pass filtered V.
"""
function steady_state_init_LP(V0, p,tspan)
    prob_ic = ODEProblem(CSC1_Ih_I_LP!, init_u0_LP(;V0=V0), tspan, p, callback=DiffEqCallbacks.TerminateSteadyState(), save_everystep=false, save_end=true, save_start=false)
    sol_ic = solve(prob_ic, ROCK2(), abstol = 1e-12, reltol = 1e-10, maxiters=1e25);

    return sol_ic.u[1]
    nothing
end




"""Steady-state initialization of 2 models with an initial phase difference.
"""
function init_phase_diff_2(phase_diff,  p_ic; tend=1500, samp_rate=0.1)
    p_ic = tuple(p_ic...)
    V0  = -60.
    ic_net_ic = steady_state_init(V0, p,tspan)
    prob_ic = ODEProblem(CSC1_Ih_I!, ic_net_ic, (0, tend), p_ic, saveat=samp_rate, maxiters=1e25)
    sol_ic = solve(prob_ic, ROCK2(), abstol = 1e-12, reltol = 1e-10, maxiters=1e25)
    xpks = argmaxima(sol_ic[1,:])
    (peaks, proms) =peakproms(xpks, sol_ic[1,:], strict=true, minprom=25, maxprom=nothing)
    spiket = sol_ic.t[peaks]
    ISI_2nd_last = sol_ic.t[peaks[end-1]] - sol_ic.t[peaks[end-2]]
    
    phase_i_ind = findfirst(sol_ic.t .>= (spiket[end-1] + ISI_2nd_last * -0.5))
    ic_i = sol_ic[:,:, phase_i_ind]

    phase_ii_ind = findfirst(sol_ic.t .>= (spiket[end-1] + ISI_2nd_last * (phase_diff-0.5)))
    ic_ii = sol_ic[:,:, phase_ii_ind]
    ic = [ic_i; ic_ii]
    return ic
    nothing
end


################################################################################
# Current Injection Utilities
################################################################################

"""Callback to turn off current injection."""
function step_I(step_start, step_end, I_mag)
    current_step= PresetTimeCallback(step_start,integrator -> integrator.p[11] = I_mag)
    current_step_off = PresetTimeCallback(step_end, integrator -> integrator.p[11] = 0.)
    cbs = CallbackSet(current_step, current_step_off)
    return cbs
    nothing
end

################################################################################
# f-I Curve Analysis
################################################################################

"""Compute firing rate (frequency) vs. current amplitude (f-I curve).
   Performs ensemble of simulations over Iarray, detects action potentials,
   and computes firing frequency from interspike intervals during step_start:step_end window.
   
   Args:
     pro: Base ODEProblem
     Iarray: Array of current amplitudes to test (nA)
     par: Full parameter vector
     step_start, step_end: Time window for analysis (ms)
     somaidx: Which state to save (usually 1 for soma voltage)
   
   Returns:
     sim: Solution ensemble
     F: Firing frequency for each current step (Hz)
"""
function fIcurve(pro, Iarray,ic, par, step_start, step_end; somaidx=1,
    dt=1e-20,dtmin=1e-300, dtmax = 0.1, maxiters=1e15, alg_hints=[:stiff])
    proo = remake(pro, p=par, u0=ic)

    # Function to modify problem for each current amplitude
    function prob_func(pro,i,repeat) 
        par_new = par
        par_new[11] = 0.
        remake(pro,p=par_new, callback=step_I(step_start, step_end, Iarray[i])) # for each current step
    end
    
    # Solve ensemble
    ensemble_prob = EnsembleProblem(proo,prob_func=prob_func);
    sim = solve(ensemble_prob, ROCK2(),
    EnsembleSerial(),trajectories=length(Iarray), save_idxs= [somaidx],
            dt=dt,dtmin=dtmin, dtmax = dtmax, maxiters=maxiters,
            alg_hints=alg_hints,
            progress=false);


    # Extract firing frequency from spike times
    F = zeros(length(sim))
    for i in range(1, length(sim));# for each sim/current step
        if SciMLBase.successful_retcode(sim[i].retcode)
            t_ind = step_start .<= sim[i].t .<= step_end
            # Find local maxima (spike peaks) in soma voltage
            xpks = argmaxima(sim[i][1,t_ind])
            (peaks, proms) =peakproms(xpks, sim[i][1,t_ind], strict=true, minprom=25, maxprom=nothing)
            if length(peaks) >= 2 #  need ≥ 2 spikes to estimate frequency
                spiket = sim[i].t[t_ind][peaks]
                ISI = zeros(length(peaks)-1)  # interspike intervals (ms)
                for i in range(2,length(peaks)); ISI[i-1] = (spiket[i]- spiket[i-1])  end
                Finst = 1 ./ (ISI/1000)# convert to frequency (Hz)
                F[i] = mean(Finst)
            else
                F[i] = 0. # no spiking
            end
        else
            F[i] = 0.
        end
    end
    return sim, F
    nothing
end


function fIcurve_distributed(pro, Iarray,ic, par, step_start, step_end; somaidx=1,
    dt=1e-20,dtmin=1e-300, dtmax = 0.1, maxiters=1e15, alg_hints=[:stiff])
    proo = remake(pro, p=par, u0=ic)
    
    # Function to modify problem for each current amplitude
    function prob_func(pro,i,repeat)
        par_new = par
        par_new[11] = 0. 
        remake(pro,p=par_new, callback=step_I(step_start, step_end, Iarray[i])) # for each current step
    end
    # Solve ensemble
    ensemble_prob = EnsembleProblem(proo,prob_func=prob_func);
    sim = solve(ensemble_prob, ROCK2(),
    EnsembleThreads(),trajectories=length(Iarray), save_idxs= [somaidx],
            dt=dt,dtmin=dtmin, dtmax = dtmax, maxiters=maxiters,
            alg_hints=alg_hints,
            progress=false);
            
    # Extract firing frequency from spike times
    F = zeros(length(sim))
    Threads.@threads for i in range(1, length(sim));# for each sim/current step
        if SciMLBase.successful_retcode(sim[i].retcode)
            t_ind = step_start .<= sim[i].t .<= step_end
            # Find local maxima (spike peaks) in soma voltage
            xpks = argmaxima(sim[i][1,t_ind])
            (peaks, proms) =peakproms(xpks, sim[i][1,t_ind], strict=true, minprom=25, maxprom=nothing)
            if length(peaks) >= 2 # need ≥ 2 spikes to estimate frequency
                spiket = sim[i].t[t_ind][peaks]
                ISI = zeros(length(peaks)-1) # interspike intervals (ms)
                for i in range(2,length(peaks)); ISI[i-1] = (spiket[i]- spiket[i-1])  end
                Finst = 1 ./ (ISI/1000)  # convert to frequency (Hz)
                F[i] = mean(Finst)
            else
                F[i] = 0. # no spiking
            end
        else
            F[i] = 0.
        end
    end
    return sim, F
    nothing
end


end