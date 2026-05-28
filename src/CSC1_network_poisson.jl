module CSC1_network_poisson
export check_gating_bounds_net,check_gating_bounds_net!, CSC1_net!, CSC1_net_GrC!, steady_state_init_u0, steady_state_init_u0_GrC, 
init_phase_diff_rand, init_phase_diff_rand_GrC, create_gap_arrays, create_syn_arrays,
V_detect, affect!, next_spike_time, condition_GrC, affect_GrC!, CSC1_net_GrC_NMDA!, CSC1_net_GrC_new!

using DrWatson
@quickactivate  "MLI_synch_2026"
include(srcdir("CSC1_Ih.jl"))
using .CSC1_Ih
using DifferentialEquations, DiffEqCallbacks,Statistics, Peaks
using FHist
using Random
using Distributions



#%%
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
const D = 5.3e-9  # μm^2/ms-> cm^2/ms
const A = 2 * 1.602176634e−19 / (1.380649e−20 * 309.15)
const  my_lock = ReentrantLock()


function check_gating_bounds_net(x, ncells)
    for c in Int.(collect(1:ncells))
        # Clamp all gating variables and Ca to non-negative
        for i in 2:8 
            if x[c, i] < 0
                x[c, i] = 0
            end
        end

        # Clamp conventional gating variables to [0,1]
        for i in 2:7  
            if x[c, i] > 1
                x[c, i] = 1
            end
        end

        # Clamp Ih gating variable to [0,1]
        if x[c, 9] < 0
            x[c, 9] = 0
        end
        if x[c, 9] > 1
            x[c, 9] = 1
        end
    end

    return x
end


function check_gating_bounds_net!(x, ncells)
    # 8 → ≥ 0
    @inbounds @views x[:, 8] = max.(x[:, 8], 0)

    # 2:7 → [0, 1]
    @inbounds @views x[:, 2:7] = clamp.(x[:, 2:7], 0, 1)

    # 9 → [0, 1]
    @inbounds @views x[:, 9] = clamp.(x[:, 9], 0, 1)
    return nothing
end

################################################################################
# Main ODE Functions for Cable Equation Simulations
################################################################################
"""Cerebellar stellate cell network with n cells

State vector r (ncells, 9):
  r[:, 1] = voltage (mV) at soma (col 1) and each dendritic compartment
  r[:, 2] = h_Na gating (Na inactivation)
  r[:, 3] = n_K gating (K activation)  
  r[:, 4] = m_A gating (A-current activation)
  r[:, 5] = h_A gating (A-current inactivation)
  r[:, 6] = h_T gating (T-type Ca inactivation)
  r[:, 7] = m_HVA gating (HVA Ca activation)
  r[:, 8] = [Ca2+] intracellular (μM)
  r[:, 9] = h_Ih gating (HCN inactivation)

Parameters include conductances, reversal potentials, and gating kinetics.
"""
function CSC1_net!(dr, r, p, t)
    gNa, gK, gleak, gA, gT, gHVA, gSK, gh, I,  Eleak, Eh, Vm, Vh, Vn, VnA, VhA, VmT, VhT, VmHVA, km, kmT, kh, kn, knA, khA, khT, kmHVA, EK, ENa, ECa, ncells, Vthresh, refractory, connect_ggap, AP_array, syn_delay, tau_GABA, connect_syn, EGABA, tau_d = p
    check_gating_bounds_net!(r, ncells)
    u = @view r[:, :, :]
    du = @view dr[:, :, :]
    @inbounds @views begin
        mSS = @inbounds @views 1 ./ (1 .+ exp.((u[:, 1] .- Vm) ./ -km))
        mTSS = @inbounds @views @. 1 ./ (1 .+ exp.(-(u[:, 1] .- VmT) ./ kmT))
        du[:, 1] =  (I
                           .- gK .* (u[:, 3].^4) .* (u[:, 1] .- EK)
                           .- gNa .* (mSS[:].^3) .* u[:, 2] .* (u[:, 1] .- ENa)
                           - gleak .* (u[:, 1] .- Eleak)
                           .- gA .* u[:, 4] .* u[:, 5] .* (u[:, 1] .- EK)
                           .- gT .* mTSS[:, 1] .* u[:, 6] .* (u[:, 1] .- ECa)
                           .- gSK .* (u[:, 8].^5 ./ ((0.45.^5) .+ (u[:, 8].^5))) .* (u[:, 1] .- EK) #1.0
                           .- gHVA .* u[:, 7] .* (u[:, 1] .- ECa)
                           .- gh .* u[:, 9] .*(u[:, 1] .- Eh)
                           .- sum(connect_ggap .* (u[:, 13] .- u[:, 13]'), dims=2)
                            .- (connect_syn' * u[:, 12]) .* (u[:, 1] .- EGABA)
            ) ./ capm

        # add Ca diffusion from cable
        du[:, 8] = @. -0.015 * (0.018 * (gT * mTSS[:, 1] * u[:, 6] * (u[:, 1] - ECa) + gHVA * u[:, 7] * (u[:, 1] - ECa)) + 0.1 * u[:, 8])

        ## Cable ##
        # gating in each section
        du[:, 2] = @. ((1 / (1 + exp((u[:, 1] - Vh) / kh))) - u[:, 2]) / (0.1 + (2 * 322 * 46) / (4 * pi * (u[:, 1] + 74)^2 + 46^2)) #h
        du[:, 3] = @. ((1 / (1 + exp(-(u[:, 1] - Vn) / kn))) - u[:, 3]) / (6 / (1 + exp((u[:, 1] + 23) / 15))) #n
        du[:, 4] = @. ((1 / (1 + exp((u[:, 1] - VnA) / -knA))) - u[:, 4]) / 5 #nA
        du[:, 5] = @. ((1 / (1 + exp((u[:, 1] - VhA) / khA))) - u[:, 5]) / 10 #hA
        du[:, 6] = @. ((1 / (1 + exp((u[:, 1] - VhT) / khT))) - u[:, 6]) / 15 #hT
        du[:, 7] = @. ((1 / (1 + exp(-(u[:, 1] - VmHVA) / kmHVA))) - u[:, 7]) / 4 #mHV
        du[:, 9] = @. (hinf(u[:, 1]) - u[:, 9]) / tauh(u[:, 1]) #hcn

        du[:, 12] = @. -(u[:, 12]) / tau_GABA # postsynaptic effect of presynaptic cell c
        du[:, 13] = @. (u[:, 1] - u[:, 13]) / tau_d # postsynaptic effect of presynaptic cell c
    end
end

"""Network with GrC AMPA input
"""
function CSC1_net_GrC!(dr, r, p, t)
    gNa, gK, gleak, gA, gT, gHVA, gSK, gh, I,  Eleak, Eh, Vm, Vh, Vn, VnA, VhA, VmT, VhT, VmHVA, km, kmT, kh, kn, knA, khA, khT, kmHVA, EK, ENa, ECa, ncells, Vthresh, refractory, connect_ggap, AP_array, syn_delay, tau_GABA, connect_syn, EGABA, tau_d, gAMPA, EAMPA, tau_AMPA, next_t, λ = p
    r = check_gating_bounds_net(r, ncells)
    u = @view r[:, :, :]
    du = @view dr[:, :, :]
    @inbounds @views begin
        mSS =  1 ./ (1 .+ exp.((u[:, 1] .- Vm) ./ -km))
        mTSS =  @. 1 ./ (1 .+ exp.(-(u[:, 1] .- VmT) ./ kmT))

        ggap_array = connect_syn .* (u[:, 13] .- u[:, 13]')
        s_vec_ggap = Vector{eltype(ggap_array)}(undef, size(ggap_array, 1))
        du[:, 1] = (I
                           .- gK .* (u[:, 3].^4) .* (u[:, 1] .- EK)
                           .- gNa .* (mSS[:].^3) .* u[:, 2] .* (u[:, 1] .- ENa)
                           - gleak .* (u[:, 1] .- Eleak)
                           .- gA .* u[:, 4] .* u[:, 5] .* (u[:, 1] .- EK)
                           .- gT .* mTSS[:, 1] .* u[:, 6] .* (u[:, 1] .- ECa)
                           .- gSK .* (u[:, 8].^5 ./ ((0.45.^5) .+ (u[:, 8].^5))) .* (u[:, 1] .- EK) #1.0
                           .- gHVA .* u[:, 7] .* (u[:, 1] .- ECa)
                           .- gh .* u[:, 9] .*(u[:, 1] .- Eh)
                        .-   sum!(s_vec_ggap, ggap_array)
                        .- (connect_syn' * u[:, 12]) .* (u[:, 1] .- EGABA)
                        .-  gAMPA .* u[:, 14] .* (u[:, 1] .- EAMPA)
                            ) ./ capm

                            
        # add Ca diffusion from cable
        du[:, 8] = @. -0.015 * (0.018 * (gT * mTSS[:, 1] * u[:, 6] * (u[:, 1] - ECa) + gHVA * u[:, 7] * (u[:, 1] - ECa)) + 0.1 * u[:, 8])

        ## Cable ##
        # gating in each section
        du[:, 2] = @. ((1 / (1 + exp((u[:, 1] - Vh) / kh))) - u[:, 2]) / (0.1 + (2 * 322 * 46) / (4 * pi * (u[:, 1] + 74)^2 + 46^2)) #h
        du[:, 3] = @. ((1 / (1 + exp(-(u[:, 1] - Vn) / kn))) - u[:, 3]) / (6 / (1 + exp((u[:, 1] + 23) / 15))) #n
        du[:, 4] = @. ((1 / (1 + exp((u[:, 1] - VnA) / -knA))) - u[:, 4]) / 5 #nA
        du[:, 5] = @. ((1 / (1 + exp((u[:, 1] - VhA) / khA))) - u[:, 5]) / 10 #hA
        du[:, 6] = @. ((1 / (1 + exp((u[:, 1] - VhT) / khT))) - u[:, 6]) / 15 #hT
        du[:, 7] = @. ((1 / (1 + exp(-(u[:, 1] - VmHVA) / kmHVA))) - u[:, 7]) / 4 #mHV
        du[:, 9] = @. (hinf(u[:, 1]) - u[:, 9]) / tauh(u[:, 1]) #hcn

        du[:, 12] = @. -(u[:, 12]) / tau_GABA # postsynaptic effect of presynaptic cell c
        du[:, 13] = @. (u[:, 1] - u[:, 13]) / tau_d # postsynaptic effect of presynaptic cell c

        # GrC AMPA 
        du[:, 14] = @. -(u[:, 14]) / tau_AMPA
    end
end


"""Network with GrC AMPA input and NMDA input
"""
function CSC1_net_GrC_NMDA!(dr, r, p, t)
    gNa, gK, gleak, gA, gT, gHVA, gSK, gh, I,  Eleak, Eh, Vm, Vh, Vn, VnA, VhA, VmT, VhT, VmHVA, km, kmT, kh, kn, knA, khA, khT, kmHVA, EK, ENa, ECa, ncells, Vthresh, refractory, connect_ggap, AP_array, syn_delay, tau_GABA, connect_syn, EGABA, tau_d, gAMPA, EAMPA, tau_AMPA, next_t, gNMDA, τ_NMDA_rise, τ_NMDA_decay, K_NMDA, tsyn, ENMDA, NMDA_bool, λ = p
    check_gating_bounds_net!(r, ncells)
    u = @view r[:, :, :]
    du = @view dr[:, :, :]# steady state gating variables
    @inbounds @views begin
        mSS = @inbounds @views 1 ./ (1 .+ exp.((u[:, 1] .- Vm) ./ -km))
        mTSS = @inbounds @views @. 1 ./ (1 .+ exp.(-(u[:, 1] .- VmT) ./ kmT))
        du[:, 1] =  (I
                           .- gK .* (u[:, 3].^4) .* (u[:, 1] .- EK)
                           .- gNa .* (mSS[:].^3) .* u[:, 2] .* (u[:, 1] .- ENa)
                           - gleak .* (u[:, 1] .- Eleak)
                           .- gA .* u[:, 4] .* u[:, 5] .* (u[:, 1] .- EK)
                           .- gT .* mTSS[:, 1] .* u[:, 6] .* (u[:, 1] .- ECa)
                           .- gSK .* (u[:, 8].^5 ./ ((0.45.^5) .+ (u[:, 8].^5))) .* (u[:, 1] .- EK) #1.0
                           .- gHVA .* u[:, 7] .* (u[:, 1] .- ECa)
                           .- gh .* u[:, 9] .*(u[:, 1] .- Eh)
                           .- sum(connect_ggap .* (u[:, 13] .- u[:, 13]'), dims=2)
                            .- (connect_syn' * u[:, 12]) .* (u[:, 1] .- EGABA)
                            .-  gAMPA .* u[:, 14] .* (u[:, 1] .- EAMPA)
                            .+ (gNMDA .* K_NMDA .* (exp.(-(t .- tsyn) ./τ_NMDA_rise) .- exp.(-(t .- tsyn) ./τ_NMDA_decay)) .* (u[:, 1] .- ENMDA)  .* (t .>= tsyn) .*  (NMDA_bool) .* (1 ./ (1 .+ 1 .* exp.(-0.062 .* u[:, 1] ) / 3.57)))
                                ) ./ capm

                            
        # add Ca diffusion from cable
        du[:, 8] = @. -0.015 * (0.018 * (gT * mTSS[:, 1] * u[:, 6] * (u[:, 1] - ECa) + gHVA * u[:, 7] * (u[:, 1] - ECa)) + 0.1 * u[:, 8])

        ## Cable ##
        # gating in each section
        du[:, 2] = @. ((1 / (1 + exp((u[:, 1] - Vh) / kh))) - u[:, 2]) / (0.1 + (2 * 322 * 46) / (4 * pi * (u[:, 1] + 74)^2 + 46^2)) #h
        du[:, 3] = @. ((1 / (1 + exp(-(u[:, 1] - Vn) / kn))) - u[:, 3]) / (6 / (1 + exp((u[:, 1] + 23) / 15))) #n
        du[:, 4] = @. ((1 / (1 + exp((u[:, 1] - VnA) / -knA))) - u[:, 4]) / 5 #nA
        du[:, 5] = @. ((1 / (1 + exp((u[:, 1] - VhA) / khA))) - u[:, 5]) / 10 #hA
        du[:, 6] = @. ((1 / (1 + exp((u[:, 1] - VhT) / khT))) - u[:, 6]) / 15 #hT
        du[:, 7] = @. ((1 / (1 + exp(-(u[:, 1] - VmHVA) / kmHVA))) - u[:, 7]) / 4 #mHV
        du[:, 9] = @. (hinf(u[:, 1]) - u[:, 9]) / tauh(u[:, 1]) #hcn

        du[:, 12] = @. -(u[:, 12]) / tau_GABA # postsynaptic effect of presynaptic cell c
        du[:, 13] = @. (u[:, 1] - u[:, 13]) / tau_d # postsynaptic effect of presynaptic cell c

        # GrC AMPA 
        du[:, 14] = @. -(u[:, 14]) / tau_AMPA
    end
end


################################################################################
# Initialization Functions
################################################################################

"""Initialize gating variables to steady state at resting potential V0.
   Used to set initial conditions for ODE simulations.
"""
function steady_state_init_u0(V0, pgating)
    Vm, Vh, Vn, VnA, VhA, VmT, VhT, VmHVA, km, kmT, kh, kn, knA, khA, khT, kmHVA = pgating
    hSS = 1 / (1 + exp((V0 - Vh) / kh))
    nSS = 1 / (1 + exp(-(V0 - Vn) / kn))
    nASS = 1 / (1 + exp((-(V0 - VnA) / knA)))
    hASS = 1 / (1 + exp((V0 - VhA) / khA))
    hTSS = 1 / (1 + exp((V0 - VhT) / khT))
    sSS = 1 / (1 + exp(-(V0 - VmHVA) / kmHVA))
    Ca = 0.001
    Y0 = [V0, hSS, nSS, nASS, hASS, hTSS, sSS, Ca, hinf(V0)]
    Y = zeros(13)
    Y[1] = V0
    for i in range(2, 9)
        Y[i] = Y0[i]
    end
    Y[13] = V0
    return Y
    nothing
end;



"""Steady-state initialization for network with GrC AMPA input
"""
function steady_state_init_u0_GrC(V0, pgating)
    Vm, Vh, Vn, VnA, VhA, VmT, VhT, VmHVA, km, kmT, kh, kn, knA, khA, khT, kmHVA = pgating
    hSS = 1 / (1 + exp((V0 - Vh) / kh))
    nSS = 1 / (1 + exp(-(V0 - Vn) / kn))
    nASS = 1 / (1 + exp((-(V0 - VnA) / knA)))
    hASS = 1 / (1 + exp((V0 - VhA) / khA))
    hTSS = 1 / (1 + exp((V0 - VhT) / khT))
    sSS = 1 / (1 + exp(-(V0 - VmHVA) / kmHVA))
    Ca = 0.001
    Y0 = [V0, hSS, nSS, nASS, hASS, hTSS, sSS, Ca, hinf(V0)]
    Y = zeros(14)
    Y[1] = V0
    for i in range(2, 9)
        Y[i] = Y0[i]
    end
    Y[13] = V0
    return Y
    nothing
end;


"""Steady-state initialization for network with random phase difference
"""
function init_phase_diff_rand(p_ic, n_rand, seed_i; tend=1500, samp_rate=0.1)
    p_ic = tuple(p_ic...)
    V0  = -60.
    ic_net_ic = CSC1_Ih.steady_state_init_LP(V0, p_ic,(0, tend))
    prob_ic = ODEProblem(CSC1_Ih.CSC1_Ih_I_LP!, ic_net_ic, (0, tend), p_ic, saveat=samp_rate, maxiters=1e25)
    sol_ic = solve(prob_ic, ROCK2(), abstol = 1e-12, reltol = 1e-10, maxiters=1e25)
    xpks = argmaxima(sol_ic[1,:])
    (peaks, proms) =peakproms(xpks, sol_ic[1,:], strict=true, minprom=25, maxprom=nothing)
    spiket = sol_ic.t[peaks]
    ISI_2nd_last = sol_ic.t[peaks[end-1]] - sol_ic.t[peaks[end-2]]
    
    Random.seed!(seed_i)
    ph_array = Random.rand(n_rand)
    ic = zeros( n_rand, 13)

    for i in eachindex(ph_array)
        phase_ii_ind = findfirst(sol_ic.t .>= (spiket[end-1] + ISI_2nd_last * (ph_array[i])))
        ic_ii = sol_ic[:, phase_ii_ind]
        ic[i, 1:9] = ic_ii[1:9]
        ic[i, 13] =  ic_ii[10]
    end
    return ic, ph_array
    nothing
end


"""Steady-state initialization for network with GrC AMPA input with random phase difference
"""
function init_phase_diff_rand_GrC(p_ic, n_rand, seed_i; tend=1500, samp_rate=0.1)
    p_ic = tuple(p_ic...)
    V0  = -60.
    ic_net_ic = CSC1_Ih.steady_state_init_LP(V0, p_ic,(0, tend))
    prob_ic = ODEProblem(CSC1_Ih.CSC1_Ih_I_LP!, ic_net_ic, (0, tend), p_ic, saveat=samp_rate, maxiters=1e25)
    sol_ic = solve(prob_ic, ROCK2(), abstol = 1e-6, reltol = 1e-6, maxiters=1e25)
    xpks = argmaxima(sol_ic[1,:])
    (peaks, proms) =peakproms(xpks, sol_ic[1,:], strict=true, minprom=25, maxprom=nothing)
    spiket = sol_ic.t[peaks]
    ISI_2nd_last = sol_ic.t[peaks[end-1]] - sol_ic.t[peaks[end-2]]
    

    Random.seed!(seed_i)
    ph_array = Random.rand(n_rand)
    ic = zeros( n_rand, 14)

    for i in eachindex(ph_array)
        phase_ii_ind = findfirst(sol_ic.t .>= (spiket[end-1] + ISI_2nd_last * (ph_array[i])))
        ic_ii = sol_ic[:, phase_ii_ind]
        ic[i, 1:9] = ic_ii[1:9]
        ic[i, 13] =  ic_ii[10]
    end
    return ic, ph_array
    nothing
end

################################################################################
# Network connectivitn Functions
################################################################################

"""Initialize gap junction connectivity array
"""
function create_gap_arrays(ncells, pairs, λ)
    ggap = zeros(ncells, ncells)
    connect = zeros(ncells, ncells)
    if length(pairs) > 0
        for con in pairs
            lock(my_lock) do
            connect[con[1], con[2]] = 1
            connect[con[2], con[1]] = 1
            ggap[con[1], con[2]] = con[3] * exp(- con[4] / λ)
            ggap[con[2], con[1]] = con[3] * exp(- con[4] / λ)
            end
        end
    end
    return ggap .* connect
    nothing
end


"""Initialize synaptic connectivity array
"""
function create_syn_arrays(ncells, pairs)
    gsyn = zeros(ncells, ncells)
    connect = zeros(ncells, ncells) # presyn, postsyn, postsyn_segement
    if length(pairs) > 0
        for con in pairs
            lock(my_lock) do
            connect[con[1], con[2]] = 1
            gsyn[con[1], con[2]] = con[3]
            end
        end
    end
    return gsyn .* connect
    nothing
end

################################################################################
# Utility functions and callbacks for network
################################################################################

"""AP dectection callback function
"""
function V_detect(u, t, integrator)
    @inbounds @views begin
    for c in Int.(collect(1:integrator.p[31])) # for each cell 
        # Access the current value (u) and time (t)
        u_curr = u[c, 1, 1]
        # Access the previous value (integrator.uprev) and time (integrator.tprev)
        u_prev = integrator.uprev[c, 1, 1]

        # if larger than Vthresh, V_current < V_previous, and time since last AP > refrractory
        # refractory time is need to isolate AP to peak and not continuously during downstroke until Vthresh is 
        lock(my_lock) do
        if (u[c, 1, 1] >= integrator.p[32] && u_curr < u_prev && (abs((integrator.t .- integrator.p[35][c])[1]) > integrator.p[33]))
            # detect spike bool
            integrator.u[c, 10] = 1
            # update last spike time
            integrator.p[35][c] = integrator.t
        else
            # no spike 
            integrator.u[c, 10] = 0
        end
        end
        # if time from last spike greater than synaptic delay, but difference between time from last spike and synaptic delay is not larger than time step
        # also not with syn_delay from tstart
        # then add presyn AP with delay 
        lock(my_lock) do
        if (abs((integrator.t.- integrator.p[35][c])[1]) > integrator.p[36]) && abs((abs((integrator.t.-integrator.p[35][c])[1]) - integrator.p[36])) <= integrator.dt && integrator.t > integrator.p[36] && integrator.p[35][c] > 0
            # presyn AP after delay
            integrator.u[c, 11] = 1
        else
            integrator.u[c, 11] = 0
        end
        end

    end
    end
    return true
end


"""AP dectection time affect function
"""
function affect!(integrator)
    # add latest spike time to AP_array in parameters
    @inbounds @views integrator.u[:, 12] += integrator.u[:, 11]
end


"""Select Poissionian ISI
"""
function next_spike_time(t; λ=50)
    return t + rand(Exponential(1/λ)*1000)
end


"""Select Poissionian ISI
"""
function next_spike_time(t::Array; λ=50)
    for i in eachindex(t)
        lock(my_lock) do
        t[i] += rand(Exponential(1/λ)*1000)
        end
    end
    return t
end


""" AMPA input callback function
"""
function condition_GrC(u, t, integrator)
    if synchtimes <= t  && t <= synchtimes + synch_dur
        @inbounds @views begin
        for c in Int.(collect(1:integrator.p[31])) # for each cell 
        lock(CSC1_network_poisson.my_lock) do
            if  (t - integrator.p[44][c] .>= 0.) && (abs((integrator.t .- integrator.p[35][c])[1]) > integrator.p[33])
                    integrator.u[c, 14] += 1
                    integrator.p[44][c] += rand(Exponential(1/λ_rate_synch)*1000) 
                end
            end
        end
        end
        return true
    else
        for c in Int.(collect(1:integrator.p[31])) # for each cell 
        lock(CSC1_network_poisson.my_lock) do
            if  (t - integrator.p[44][c] .>= 0.) && (abs((integrator.t .- integrator.p[35][c])[1]) > integrator.p[33])
                    integrator.u[c, 14] += 1
                    integrator.p[44][c] += rand(Exponential(1/λ_rate)*1000)
                end
            end
        end
        return true
    end

end


""" AMPA input affect function
"""
function affect_GrC!(integrator)
    # do nothing
end

end