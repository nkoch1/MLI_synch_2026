module CSC_cable_Ih

using DifferentialEquations
using Parameters
using Statistics, Peaks
using LoopVectorization

###########################################################################
#I-h HCN1 channel from Kamilla Angelo, Michael London,Soren R. Christensen, and Michael Hausser 2007 J. of Neurosci.
# used in Rizza et al 2021
const ratetau = 1              # (ms)
const q10=3
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
const qt = q10^((rec_temp-37)/10 ) #(degC)
const v_inf_half  = (v_inf_half_noljp  - ljp)
const v_tau_half1 = (v_tau_half1_noljp - ljp)
const v_tau_half2 = (v_tau_half2_noljp - ljp)

hinf(v) = 1 / (1+exp( (v-v_inf_half) / v_inf_k) )
tauh(v) = (ratetau / (v_tau_const * ( exp( (v-v_tau_half1) / v_tau_k1) + exp( (v-v_tau_half2) / v_tau_k2) )))/qt


const rl = 110 # Ω*cm  -  specific intracellular resisitivity Rizza et al 2021
const capm = 1.50148 # μF/cm^2  -  specific membrane capacitance
const D = 5.3e-9 # μm^2/ms-> cm^2/ms
const A = 2 * 1.602176634e−19/(1.380649e−20 *309.15)


function check_gating_bounds(x, numseg)
    # Clamp all gating variables and Ca to non-negative
    for i in 2:9 
        for j in 1:Int(numseg)+1
            if x[i,j] < 0
                x[i,j] = 0.
            end
        end
    end

    # Clamp conventional gating variables to [0,1]
    for i in 2:7  # gating variables
        for j in 1:Int(numseg)+1
            if x[i, j] > 1
                x[i, j] = 1.
            end
        end
    end

    # Clamp Ih gating variable to [0,1]
    for j in 1:Int(numseg)+1
        if x[9, j] > 1
            x[9,j] = 1.
        end
    end

    return x
end


function check_gating_bounds_end(x, numseg)
    # Clamp all gating variables and Ca to non-negative
    for i in 2:9 # gating variables and Ca
        if i > 2
            for j in 1:Int(numseg)+1 
                if x[i,j] < 0
                    x[i,j] = 0.
                end
            end
        else
            for j in 1:Int(numseg) 
                if x[i,j] < 0
                    x[i,j] = 0.
                end
            end
        end
    end

    # Clamp conventional gating variables to [0,1]
    for i in 2:7  # gating variables
        if i > 2
            for j in 1:Int(numseg)+1
                if x[i, j] > 1
                    x[i, j] = 1.
                end
            end
        else
            for j in 1:Int(numseg)
                if x[i, j] > 1
                    x[i, j] = 1.
                end
            end
        end
    end

    # Clamp Ih gating variable to [0,1]
    for j in 1:Int(numseg)+1
        if x[9, j] > 1
            x[9,j] = 1.
        end
    end

    return x
end

################################################################################
# Main ODE Functions for Cable Equation Simulations
################################################################################

"""Cerebellar stellate cell with soma, dendrite, and current injection.

State vector r (9 × numseg+1):
  r[1,:] = voltage (mV) at soma (col 1) and each dendritic compartment
  r[2,:] = h_Na gating (Na inactivation)
  r[3,:] = n_K gating (K activation)  
  r[4,:] = m_A gating (A-current activation)
  r[5,:] = h_A gating (A-current inactivation)
  r[6,:] = h_T gating (T-type Ca inactivation)
  r[7,:] = m_HVA gating (HVA Ca activation)
  r[8,:] = [Ca2+] intracellular (μM)
  r[9,:] = h_Ih gating (HCN inactivation)

Parameters include conductances, reversal potentials, and gating kinetics.
"""
function CSCcable_gating!(dr, r, p, t)
    gNa::Float64, gK::Float64, gleak::Float64, gA::Float64, gT::Float64, gHVA::Float64, gSK::Float64, gh::Float64, gKd::Float64, gleakd::Float64, gAd::Float64, gTd::Float64, gHVAd::Float64, gSKd::Float64,ghd::Float64, soma_r::Float64, dend_r::Float64, I::Float64, L::Float64, Eleak::Float64, Eleakd::Float64, numseg::Float64, Eh::Float64, Vm::Float64, Vh::Float64, Vn::Float64, VnA::Float64, VhA::Float64, VmT::Float64, VhT::Float64, VmHVA::Float64, km::Float64,kmT::Float64, kh::Float64, kn::Float64, knA::Float64, khA::Float64,khT::Float64, kmHVA::Float64,  EK::Float64, ENa::Float64, ECa::Float64 = p
    r = check_gating_bounds(r, numseg)
    u = @view r[:, :]
    du = @view dr[:, :]

    Δx = L / numseg; # cm
    a = dend_r/10000 #  μm => cm
    As = soma_r /10000
    rmem = 1/(gleakd/1000) # Ω/cm^2 (divide by 1000 to get S-> Ω instead of mS-> kΩ)  -
    τm = rmem * capm /1000  # μF/cm^2 * Ω*cm^2 = uF*Ω = μsec /1000 to get ms
    λ = sqrt(a*rmem/(2*rl))
    κ = 1. /(rl/1000) * (a/(2*As))^2
    γleak = 1/(gleakd)


    # Pre-compute voltage-dependent steady-state variables
    # These are fast (instantaneous) gating variables
    mSS =  @inbounds @views 1 / (1 + exp((u[1,1] - Vm) / -km));
    mTSS = @inbounds @views @. 1 / (1 + exp(-(u[1, :] - VmT) / kmT));
    @inbounds @views  begin
        # SOMA (compartment 1) voltage equation
        # dV/dt = (I_input + I_axial - I_Na - I_K - I_leak - I_A - I_T - I_SK - I_HVA - I_Ih) / Cm
            
        du[1,1] =  (I
                    - gK  * (u[3,1]^4) * (u[1,1] - EK)
                    - gNa  * (mSS^3) * u[2,1] * (u[1,1]-ENa)
                    - gleak   * (u[1,1] - Eleak)
                    - gA * u[4,1] * u[5,1] * (u[1,1] - EK)
                    - gT  * mTSS[1] * u[6,1] * (u[1,1] - ECa)
                    - gSK  * (u[8,1]^5 / ((0.45^5) + (u[8,1]^5))) * (u[1,1] - EK) 
                    - gHVA  * u[7,1] * (u[1,1] - ECa)
                    - u[9,1]*gh*(u[1,1] - Eh)
                    +  κ  * (u[1,2] - u[1,1]) / Δx # uA/cm^2
                    ) / capm;

        # SOMA calcium dynamics (accumulation from Ca channels + diffusion from dendrite)
        # dCa/dt = -diffusion - extrusion + channel current
        du[8,1] =  - D/(Δx)^2 *(( u[8, 1]  - u[8, 2]) + A *(u[8, 2] * (u[1, 2] - u[1, 1]) )) 
                    - 0.015 * (0.018 * (gT  * mTSS[1] * u[6,1] * (u[1,1] - 22) + gHVA  * u[7,1] * (u[1,1] - 22)) + 0.1 * u[8,1]); 
        
        # Update all gating variables throughout soma and cable (instantaneous kinetics)
        for ii in 1:Int(numseg+1)::Int64
            du[2, ii] = ((1 / (1 + exp((u[1, ii] - Vh) / kh))) - u[2, ii]) / ( 0.1 + (2 * 322 * 46) / (4 * pi * (u[1, ii] + 74) ^ 2 + 46 ^ 2)); #k
            du[3, ii] = ((1 / (1 + exp(-(u[1, ii] - Vn) / kn))) - u[3, ii]) / (6 / (1 + exp((u[1, ii] + 23) / 15))); #n
            du[4, ii] = ((1 / (1 + exp((u[1, ii] - VnA) / -knA))) - u[4, ii]) / 5; #nA
            du[5, ii] = ((1 / (1 + exp((u[1, ii] - VhA) / khA))) - u[5, ii]) / 10; #hA
            du[6, ii] = ((1 / (1 + exp((u[1, ii] - VhT) / khT))) - u[6, ii]) / 15; #hT
            du[7, ii] = ((1 / (1 + exp(-(u[1, ii] - VmHVA) / kmHVA))) - u[7,ii]) / 4; #mHV
            du[9, ii] = (hinf(u[1, ii]) - u[9, ii]) / tauh(u[1, ii]) #hcn
        end

        # Ca diffusion along cable
        du[8, 2] = -D/(Δx)^2 *(( 2 * u[8, 2] - u[8, 1] - u[8, 3]) + A *(u[8, 3] * (u[1, 3] - u[1, 2]) - u[8, 2] * (u[1, 2] - u[1, 1])))
                                -0.015 .* (0.018  .* (gTd .* mTSS[2] .* u[6, 2] .* (u[1, 2] .- 22) + gHVAd  .* u[7, 2] .* (u[1, 2] .- 22)) .+ 0.1 .* u[8, 2]) ; 
                                
            
        for ii in 3:Int(numseg)::Int64              
            du[8, ii] =  -D/(Δx)^2 * (( 2 * u[8, ii] - u[8, ii-1] - u[8, ii+1]) + A * (u[8, ii+1] * (u[1, ii+1] - u[1, ii]) - u[8, ii] * (u[1, ii] - u[1, ii-1])))
                                - 0.015 * (0.018  * (gTd  * mTSS[ii] * u[6, ii] * (u[1, ii] - 22)
                                + gHVAd * u[7, ii] * (u[1, ii] - 22)) + 0.1 * u[8, ii]); 
            end
        du[8, end] = -D/(Δx)^2 *(( u[8, end] - u[8, end-1]) + A *( - u[8, end] * (u[1, end] - u[1, end - 1])))  
                - 0.015 .* (0.018 .* (gTd  .* mTSS[end] .* u[6, end] .* (u[1, end] .- 22)
                .+  gHVAd  .* u[7, end] .* (u[1, end] .- 22)) .+ 0.1 .* u[8, end]); 

        # Formulism from Goldberg et al. 2007 https://doi.org/10.1152/jn.00810.2006
        # active cable (V)     
            du[1,2] = ((λ^2)/(Δx^2) * (u[1,3]- 2*u[1,2] + u[1,1])
            - gleakd * γleak * (u[1,2] - Eleakd)
            - gKd * γleak * (u[3,2]^4) * (u[1,2]  - EK)
            - gAd * γleak * u[4,2] * u[5,2] * (u[1,2]  - EK)
            - gTd * γleak * mTSS[2] * u[6,2] * (u[1,2]-ECa)
            - gSKd * γleak * (u[8,2]^5/((0.45^5)+(u[8,2]^5))) * (u[1,2]  - EK)
            - gHVAd * γleak * u[7,2] * (u[1,2] - ECa)
            - u[9,2] * ghd * γleak * (u[1,2] - Eh)
            ) / τm
        # Interior dendritic compartments (intermediate cable sections)
        for ii in 3:Int(numseg)::Int64
                du[1,ii] = ((λ^2) / (Δx^2) * (u[1,ii+1] - 2 *u[1,ii] + u[1,ii-1])
                - gleakd * γleak * (u[1,ii] - Eleakd)
                - gKd * γleak * (u[3,ii]^4) * (u[1,ii]  - EK)
                - gAd * γleak * (u[4,ii]) * (u[5,ii]) * (u[1,ii]  - EK)
                - gTd * γleak * (mTSS[ii]) * (u[6,ii]) * (u[1,ii] - ECa)
                - gSKd * γleak * (u[8,ii]^5 / ((0.45^5) + (u[8,ii]^5))) * (u[1,ii]  - EK)
                - gHVAd * γleak * (u[7,ii]) * (u[1,ii] - ECa)
                - u[9,ii] * ghd * γleak * (u[1,ii] - Eh)
                ) / τm
        end

        # Terminal dendrite (open-ended: last compartment)
        du[1,end] = ((λ^2) /(Δx^2) * (-u[1,end] + u[1,end-1])
                - gleakd * γleak *(u[1,end] - Eleakd)
                - gKd * γleak * (u[3,end]^4) * (u[1,end]  - EK)
                - gAd * γleak * (u[4,end]) * (u[5,end]) * (u[1,end]  - EK)
                - gTd * γleak * (mTSS[end]) * (u[6,end]) * (u[1,end] - ECa)
                - gSKd * γleak * (u[8,end]^5/((0.45^5) + (u[8,end]^5))) * (u[1,end]  - EK)
                - gHVAd * γleak * (u[7,end]) * (u[1,end] - ECa)
                - u[9,end] * ghd * γleak  * (u[1,end] - Eh)
                ) / τm
    end
end


function CSCcable_end!(dr, r, p, t)
    gNa::Float64, gK::Float64, gleak::Float64, gA::Float64, gT::Float64, gHVA::Float64, gSK::Float64, gh::Float64, gKd::Float64, gleakd::Float64, gAd::Float64, gTd::Float64, gHVAd::Float64, gSKd::Float64,ghd::Float64, soma_r::Float64, dend_r::Float64, I::Float64, L::Float64, Eleak::Float64, Eleakd::Float64, numseg::Float64, Eh::Float64, Vm::Float64, Vh::Float64, Vn::Float64, VnA::Float64, VhA::Float64, VmT::Float64, VhT::Float64, VmHVA::Float64, km::Float64,kmT::Float64, kh::Float64, kn::Float64, knA::Float64, khA::Float64,khT::Float64, kmHVA::Float64,  EK::Float64, ENa::Float64, ECa::Float64 = p
    r = check_gating_bounds_end(r, numseg)
    u = @view r[:, :]
    du = @view dr[:, :]

    Δx = L / numseg; # cm
    a = dend_r/10000 #  μm => cm
    As = soma_r /10000
    rmem = 1/(gleakd/1000) # Ω/cm^2 (divide by 1000 to get S-> Ω instead of mS-> kΩ)  -
    τm = rmem * capm /1000  # μF/cm^2 * Ω*cm^2 = uF*Ω = μsec /1000 to get ms
    λ = sqrt(a*rmem/(2*rl))
    κ = 1. /(rl/1000) * (a/(2*As))^2
    γleak = 1/(gleakd)


    # steady state gating variables
    mSS =  @inbounds @views 1 / (1 + exp((u[1,1] - Vm) / -km));
    mTSS = @inbounds @views @. 1 / (1 + exp(-(u[1, :] - VmT) / kmT));
    @inbounds @views  begin
    du[1,1] =  (I
                - gK  * (u[3,1]^4) * (u[1,1] - EK)
                - gNa  * (mSS^3) * u[2,1] * (u[1,1]-ENa)
                - gleak   * (u[1,1] - Eleak)
                - gA * u[4,1] * u[5,1] * (u[1,1] - EK)
                - gT  * mTSS[1] * u[6,1] * (u[1,1] - ECa)
                - gSK  * (u[8,1]^5 / ((0.45^5) + (u[8,1]^5))) * (u[1,1] - EK) #1.0
                - gHVA  * u[7,1] * (u[1,1] - ECa)
                - u[9,1]*gh*(u[1,1] - Eh)
                +  κ  * (u[1,2] - u[1,1]) / Δx # uA/cm^2
                ) / capm;

    # add Ca diffusion from cable
    du[8,1] =  - D/(Δx)^2 *(( u[8, 1]  - u[8, 2]) + A *(u[8, 2] * (u[1, 2] - u[1, 1]) )) 
                - 0.015 * (0.018 * (gT  * mTSS[1] * u[6,1] * (u[1,1] - 22) + gHVA  * u[7,1] * (u[1,1] - 22)) + 0.1 * u[8,1]); 
    ## Cable ##
    # gating in each section
    for ii in 1:Int(numseg+1)::Int64
        if ii != Int(numseg+1) 
            du[2, ii] = ((1 / (1 + exp((u[1, ii] - Vh) / kh))) - u[2, ii]) / ( 0.1 + (2 * 322 * 46) / (4 * pi * (u[1, ii] + 74) ^ 2 + 46 ^ 2)); #k
        end
        du[3, ii] = ((1 / (1 + exp(-(u[1, ii] - Vn) / kn))) - u[3, ii]) / (6 / (1 + exp((u[1, ii] + 23) / 15))); #n
        du[4, ii] = ((1 / (1 + exp((u[1, ii] - VnA) / -knA))) - u[4, ii]) / 5; #nA
        du[5, ii] = ((1 / (1 + exp((u[1, ii] - VhA) / khA))) - u[5, ii]) / 10; #hA
        du[6, ii] = ((1 / (1 + exp((u[1, ii] - VhT) / khT))) - u[6, ii]) / 15; #hT
        du[7, ii] = ((1 / (1 + exp(-(u[1, ii] - VmHVA) / kmHVA))) - u[7,ii]) / 4; #mHV
        du[9, ii] = (hinf(u[1, ii]) - u[9, ii]) / tauh(u[1, ii]) #hcn
    end

    # Ca diffusion along cable
    du[8, 2] = -D/(Δx)^2 *(( 2 * u[8, 2] - u[8, 1] - u[8, 3]) + A *(u[8, 3] * (u[1, 3] - u[1, 2]) - u[8, 2] * (u[1, 2] - u[1, 1])))
                            -0.015 .* (0.018  .* (gTd .* mTSS[2] .* u[6, 2] .* (u[1, 2] .- 22) + gHVAd  .* u[7, 2] .* (u[1, 2] .- 22)) .+ 0.1 .* u[8, 2]) ; 
                            
        
    for ii in 3:Int(numseg)::Int64       
        du[8, ii] =  -D/(Δx)^2 * (( 2 * u[8, ii] - u[8, ii-1] - u[8, ii+1]) + A * (u[8, ii+1] * (u[1, ii+1] - u[1, ii]) - u[8, ii] * (u[1, ii] - u[1, ii-1])))
                            - 0.015 * (0.018  * (gTd  * mTSS[ii] * u[6, ii] * (u[1, ii] - 22)
                            + gHVAd * u[7, ii] * (u[1, ii] - 22)) + 0.1 * u[8, ii]); 
        end
    du[8, end] = -D/(Δx)^2 *(( u[8, end] - u[8, end-1]) + A *( - u[8, end] * (u[1, end] - u[1, end - 1])))  
            - 0.015 .* (0.018 .* (gTd  .* mTSS[end] .* u[6, end] .* (u[1, end] .- 22)
            .+  gHVAd  .* u[7, end] .* (u[1, end] .- 22)) .+ 0.1 .* u[8, end]); 

    # Formulism from Goldberg et al. 2007 https://doi.org/10.1152/jn.00810.2006
    # active cable (V)     
    du[1,2] = ((λ^2)/(Δx^2) * (u[1,3]- 2*u[1,2] + u[1,1])
        - gleakd * γleak * (u[1,2] - Eleakd)
        - gKd * γleak * (u[3,2]^4) * (u[1,2]  - EK)
        - gAd * γleak * u[4,2] * u[5,2] * (u[1,2]  - EK)
        - gTd * γleak * mTSS[2] * u[6,2] * (u[1,2]-ECa)
        - gSKd * γleak * (u[8,2]^5/((0.45^5)+(u[8,2]^5))) * (u[1,2]  - EK)
        - gHVAd * γleak * u[7,2] * (u[1,2] - ECa)
        - u[9,2] * ghd * γleak * (u[1,2] - Eh)
        ) / τm
    for ii in 3:Int(numseg)::Int64  
            du[1,ii] = ((λ^2) / (Δx^2) * (u[1,ii+1] - 2 *u[1,ii] + u[1,ii-1])
            - gleakd * γleak * (u[1,ii] - Eleakd)
            - gKd * γleak * (u[3,ii]^4) * (u[1,ii]  - EK)
            - gAd * γleak * (u[4,ii]) * (u[5,ii]) * (u[1,ii]  - EK)
            - gTd * γleak * (mTSS[ii]) * (u[6,ii]) * (u[1,ii] - ECa)
            - gSKd * γleak * (u[8,ii]^5 / ((0.45^5) + (u[8,ii]^5))) * (u[1,ii]  - EK)
            - gHVAd * γleak * (u[7,ii]) * (u[1,ii] - ECa)
            - u[9,ii] * ghd * γleak * (u[1,ii] - Eh)
            ) / τm
    end

    du[1,end] = ((λ^2) /(Δx^2) * (u[2,end] - 2*u[1,end] + u[1,end-1])
            - gleakd * γleak *(u[1,end] - Eleakd)
            - gKd * γleak * (u[3,end]^4) * (u[1,end]  - EK)
            - gAd * γleak * (u[4,end]) * (u[5,end]) * (u[1,end]  - EK)
            - gTd * γleak * (mTSS[end]) * (u[6,end]) * (u[1,end] - ECa)
            - gSKd * γleak * (u[8,end]^5/((0.45^5) + (u[8,end]^5))) * (u[1,end]  - EK)
            - gHVAd * γleak * (u[7,end]) * (u[1,end] - ECa)
            - u[9,end] * ghd * γleak  * (u[1,end] - Eh)
            ) / τm
    end
end


function CSC_cable_syn_I_OU!(dr, r, p, t)
    gNa::Float64, gK::Float64, gleak::Float64, gA::Float64, gT::Float64, gHVA::Float64, gSK::Float64, gh::Float64, gKd::Float64, gleakd::Float64, gAd::Float64, gTd::Float64, gHVAd::Float64, gSKd::Float64,ghd::Float64, soma_r::Float64, dend_r::Float64, I::Float64, L::Float64, Eleak::Float64, Eleakd::Float64, numseg::Float64, Eh::Float64, Vm::Float64, Vh::Float64, Vn::Float64, VnA::Float64, VhA::Float64, VmT::Float64, VhT::Float64, VmHVA::Float64, km::Float64,kmT::Float64, kh::Float64, kn::Float64, knA::Float64, khA::Float64,khT::Float64, kmHVA::Float64,  EK::Float64, ENa::Float64, ECa::Float64,gsyn, θ, μ, σ, τ = p
    r = check_gating_bounds_end(r, numseg)
    u = @view r[:, :]
    du = @view dr[:, :]

    Δx = L / numseg; # cm
    a = dend_r/10000 #  μm => cm
    As = soma_r /10000
    rmem = 1/(gleakd/1000) # Ω/cm^2 (divide by 1000 to get S-> Ω instead of mS-> kΩ)  -
    τm = rmem * capm /1000  # μF/cm^2 * Ω*cm^2 = uF*Ω = μsec /1000 to get ms
    λ = sqrt(a*rmem/(2*rl))
    κ = 1. /(rl/1000) * (a/(2*As))^2
    γleak = 1/(gleakd)


    # steady state gating variables
    mSS =  @inbounds @views 1 / (1 + exp((u[1,1] - Vm) / -km));
    mTSS = @inbounds @views @. 1 / (1 + exp(-(u[1, :] - VmT) / kmT));
    @inbounds @views  begin
    du[1,1] =  (I
                - gK  * (u[3,1]^4) * (u[1,1] - EK)
                - gNa  * (mSS^3) * u[2,1] * (u[1,1]-ENa)
                - gleak   * (u[1,1] - Eleak)
                - gA * u[4,1] * u[5,1] * (u[1,1] - EK)
                - gT  * mTSS[1] * u[6,1] * (u[1,1] - ECa)
                - gSK  * (u[8,1]^5 / ((0.45^5) + (u[8,1]^5))) * (u[1,1] - EK) #1.0
                - gHVA  * u[7,1] * (u[1,1] - ECa)
                - u[9,1]*gh*(u[1,1] - Eh)
                +  κ  * (u[1,2] - u[1,1]) / Δx # uA/cm^2
                ) / capm;

    # add Ca diffusion from cable
    du[8,1] =  - D/(Δx)^2 *(( u[8, 1]  - u[8, 2]) + A *(u[8, 2] * (u[1, 2] - u[1, 1]) )) 
                - 0.015 * (0.018 * (gT  * mTSS[1] * u[6,1] * (u[1,1] - 22) + gHVA  * u[7,1] * (u[1,1] - 22)) + 0.1 * u[8,1]); 
    ## Cable ##
    # gating in each section
    for ii in 1:Int(numseg+1)::Int64
        if ii >= Int(numseg) 
            du[2, ii] = ((1 / (1 + exp((u[1, ii] - Vh) / kh))) - u[2, ii]) / ( 0.1 + (2 * 322 * 46) / (4 * pi * (u[1, ii] + 74) ^ 2 + 46 ^ 2)); #k
        end
        du[3, ii] = ((1 / (1 + exp(-(u[1, ii] - Vn) / kn))) - u[3, ii]) / (6 / (1 + exp((u[1, ii] + 23) / 15))); #n
        du[4, ii] = ((1 / (1 + exp((u[1, ii] - VnA) / -knA))) - u[4, ii]) / 5; #nA
        du[5, ii] = ((1 / (1 + exp((u[1, ii] - VhA) / khA))) - u[5, ii]) / 10; #hA
        du[6, ii] = ((1 / (1 + exp((u[1, ii] - VhT) / khT))) - u[6, ii]) / 15; #hT
        du[7, ii] = ((1 / (1 + exp(-(u[1, ii] - VmHVA) / kmHVA))) - u[7,ii]) / 4; #mHV
        du[9, ii] = (hinf(u[1, ii]) - u[9, ii]) / tauh(u[1, ii]) #hcn
    end

    # Ca diffusion along cable
    du[8, 2] = -D/(Δx)^2 *(( 2 * u[8, 2] - u[8, 1] - u[8, 3]) + A *(u[8, 3] * (u[1, 3] - u[1, 2]) - u[8, 2] * (u[1, 2] - u[1, 1])))
                            -0.015 .* (0.018  .* (gTd .* mTSS[2] .* u[6, 2] .* (u[1, 2] .- 22) + gHVAd  .* u[7, 2] .* (u[1, 2] .- 22)) .+ 0.1 .* u[8, 2]) ; 
    for ii in 3:Int(numseg)::Int64       
        du[8, ii] =  -D/(Δx)^2 * (( 2 * u[8, ii] - u[8, ii-1] - u[8, ii+1]) + A * (u[8, ii+1] * (u[1, ii+1] - u[1, ii]) - u[8, ii] * (u[1, ii] - u[1, ii-1])))
                            - 0.015 * (0.018  * (gTd  * mTSS[ii] * u[6, ii] * (u[1, ii] - 22)
                            + gHVAd * u[7, ii] * (u[1, ii] - 22)) + 0.1 * u[8, ii]); 
        end
    du[8, end] = -D/(Δx)^2 *(( u[8, end] - u[8, end-1]) + A *( - u[8, end] * (u[1, end] - u[1, end - 1])))  
            - 0.015 .* (0.018 .* (gTd  .* mTSS[end] .* u[6, end] .* (u[1, end] .- 22)
            .+  gHVAd  .* u[7, end] .* (u[1, end] .- 22)) .+ 0.1 .* u[8, end]); 

    # Formulism from Goldberg et al. 2007 https://doi.org/10.1152/jn.00810.2006
    # active cable (V)     
    du[1,2] = ((λ^2)/(Δx^2) * (u[1,3]- 2*u[1,2] + u[1,1])
        - gleakd * γleak * (u[1,2] - Eleakd)
        - gKd * γleak * (u[3,2]^4) * (u[1,2]  - EK)
        - gAd * γleak * u[4,2] * u[5,2] * (u[1,2]  - EK)
        - gTd * γleak * mTSS[2] * u[6,2] * (u[1,2]-ECa)
        - gSKd * γleak * (u[8,2]^5/((0.45^5)+(u[8,2]^5))) * (u[1,2]  - EK)
        - gHVAd * γleak * u[7,2] * (u[1,2] - ECa)
        ) / τm
    for ii in 3:Int(numseg)::Int64  
            du[1,ii] = ((λ^2) / (Δx^2) * (u[1,ii+1] - 2 *u[1,ii] + u[1,ii-1])
            - gleakd * γleak * (u[1,ii] - Eleakd)
            - gKd * γleak * (u[3,ii]^4) * (u[1,ii]  - EK)
            - gAd * γleak * (u[4,ii]) * (u[5,ii]) * (u[1,ii]  - EK)
            - gTd * γleak * (mTSS[ii]) * (u[6,ii]) * (u[1,ii] - ECa)
            - gSKd * γleak * (u[8,ii]^5 / ((0.45^5) + (u[8,ii]^5))) * (u[1,ii]  - EK)
            - gHVAd * γleak * (u[7,ii]) * (u[1,ii] - ECa)
            ) / τm
    end

    du[1,end] = ((λ^2) /(Δx^2) * (-u[1,end] + u[1,end-1])
            - gleakd * γleak *(u[1,end] - Eleakd)
            - gKd * γleak * (u[3,end]^4) * (u[1,end]  - EK)
            - gAd * γleak * (u[4,end]) * (u[5,end]) * (u[1,end]  - EK)
            - gTd * γleak * (mTSS[end]) * (u[6,end]) * (u[1,end] - ECa)
            - gSKd * γleak * (u[8,end]^5/((0.45^5) + (u[8,end]^5))) * (u[1,end]  - EK)
            - gHVAd * γleak * (u[7,end]) * (u[1,end] - ECa)
            + gsyn * γleak * u[2, end] # inhib input
            ) / τm
    end
    du[2, end] =  θ * (μ - u[2, end]) # OU term 
end


function syn_OU_noise_I!(du,u,p,t)
    """
    add synaptic input as OU process
    """
    gNa::Float64, gK::Float64, gleak::Float64, gA::Float64, gT::Float64, gHVA::Float64, gSK::Float64, gh::Float64, gKd::Float64, gleakd::Float64, gAd::Float64, gTd::Float64, gHVAd::Float64, gSKd::Float64,ghd::Float64, soma_r::Float64, dend_r::Float64, I::Float64, L::Float64, Eleak::Float64, Eleakd::Float64, numseg::Float64, Eh::Float64, Vm::Float64, Vh::Float64, Vn::Float64, VnA::Float64, VhA::Float64, VmT::Float64, VhT::Float64, VmHVA::Float64, km::Float64,kmT::Float64, kh::Float64, kn::Float64, knA::Float64, khA::Float64,khT::Float64, kmHVA::Float64,  EK::Float64, ENa::Float64, ECa::Float64,gsyn, θ, μ, σ, τ  = p
    
    dA = @view  du[:,:]
    dA[1:end, :] .= 0.
    dA[2,end] = sqrt((2.0*σ^2.0/τ))
end

################################################################################
# Initialization Functions
################################################################################

"""Initialize gating variables to steady state at resting potential V0.
   Used to set initial conditions for ODE simulations.
"""
function steady_state_init_u0_gating(V0, numseg, pgating);
    Vm::Float64, Vh::Float64, Vn::Float64, VnA::Float64, VhA::Float64, VmT::Float64, VhT::Float64, VmHVA::Float64, km::Float64,kmT::Float64, kh::Float64, kn::Float64, knA::Float64, khA::Float64,khT::Float64, kmHVA::Float64 = pgating
    hSS = 1 / (1 + exp((V0 - Vh) / kh));
    nSS = 1 / (1 + exp(-(V0 - Vn) / kn));
    nASS = 1 / (1 + exp((-(V0 - VnA) / knA)));
    hASS = 1 / (1 + exp((V0 - VhA) / khA));
    hTSS = 1 / (1 + exp((V0 - VhT) / khT));
    sSS = 1 / (1 + exp(-(V0 - VmHVA) / kmHVA));
    Ca = 0.001
    Y0 = [V0, hSS, nSS, nASS, hASS, hTSS,sSS, Ca, hinf(V0)]
    Y = zeros(9,numseg+1)
    Y[1,1] = V0
    Y[1, 2:end] .= V0
    for i in range(2,9); 
        Y[i,:] .= Y0[i]  
    end
    return Y;
    nothing
end;


"""Evolve cell to steady state at constant holding potential V0.
   Uses callback to terminate integration when state reaches steady state.
   Returns converged gating variable state for use as initial condition.
"""
function steady_state_init(V0, p,tspan, numseg; abstol=1e-12, reltol=1e-10,  alg=QNDF())
    p = tuple(p...)
    u0 = steady_state_init_u0_gating(V0, numseg, p[24:end])
    ss_cb = TerminateSteadyState( min_t = 100)
    prob_ic = ODEProblem(CSCcable_gating!, u0, tspan, p, callback=ss_cb, save_everystep=false, save_end=true, save_start=false)
    sol_ic = solve(prob_ic,alg, dtmin=1e-60, dt=1e-20, abstol = abstol, reltol = reltol, maxiters=10000000, alg_hints=[:stiff]); 
    return sol_ic.u[end]
    nothing
end
    

"""Steady-state initialization with AP.
"""
function steady_state_init_AP(V0, p,tspan, numseg; abstol=1e-12, reltol=1e-10, alg=ROCK2())
    p = tuple(p...)
    u0 = steady_state_init_u0_gating(V0, numseg, p[24:end])
    ss_cb = TerminateSteadyState( min_t = 100)
    prob_ic = ODEProblem(CSCcable_gating!, u0, tspan, p, callback=ss_cb, save_everystep=false, save_end=true, save_start=false)
    sol_ic = solve(prob_ic, alg, abstol = abstol, reltol = reltol, maxiters=10000000, alg_hints=[:stiff]); 
    return sol_ic.u[end]
    nothing
end
    

"""Steady-state initialization for OU input protocols.
"""
function steady_state_init_syn_OU(V0, p,tspan, numseg; abstol=1e-12, reltol=1e-10)
    p = tuple(p...)
    u0 = steady_state_init_u0_gating(V0, numseg, p[24:end])
    prob_ic = SDEProblem(CSC_cable_syn_I_OU!,syn_OU_noise_I!, u0, tspan, p, save_everystep=false, save_end=true, save_start=false)
    sol_ic = solve(prob_ic); 
    return sol_ic.u[end]
    nothing
end
    


################################################################################
# Current Injection Utilities
################################################################################

"""Callback to turn off current injection."""
function step_off!(integrator)
    integrator.p[18] = 0.
    integrator.opts.save_on = false
end


"""Create callback set for step current injection.
   Turns current on at step_start, off at step_end. Amplitude = I_mag (into parameter p[18]).
"""
function step_I(step_start, step_end, I_mag; saveall=false)
    if saveall
        current_step= PresetTimeCallback(step_start,integrator -> integrator.p[18] = I_mag)
        current_step_off = PresetTimeCallback(step_end, integrator -> integrator.p[18] = 0.)
        cbs = CallbackSet(current_step, current_step_off)
    else
        function step_on!(integrator)
            integrator.p[18] = I_mag
            integrator.opts.save_on = true
        end
        
        sav_on = PresetTimeCallback(step_start,step_on!)
        sav_off = PresetTimeCallback(step_end,step_off!)
        cbs = CallbackSet(sav_on, sav_off)   
    end
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
function fIcurve(pro, Iarray, par, step_start, step_end; somaidx=1, # tspan, tsteps
    dt=1e-20,dtmin=1e-300, dtmax = 0.1, maxiters=1e15, alg_hints=[:stiff])

    proo = remake(pro, p=par)

    # Function to modify problem for each current amplitude
    function prob_func(pro,i,repeat) 
        par_new = par
        par_new[18] = 0. # reset before application
        remake(proo, p=par_new, callback=step_I(step_start, step_end,  Iarray[i])) # for each current step
    end

    # Solve ensemble
    ensemble_prob = EnsembleProblem(proo,prob_func=prob_func);
    sim = solve(ensemble_prob, ROCK2(),
                EnsembleSerial(),trajectories=length(Iarray), save_idxs= [somaidx],
                dt=dt,dtmin=dtmin, dtmax = dtmax, maxiters=maxiters,
                alg_hints=alg_hints, progress=false);

    
    # Extract firing frequency from spike times
    F = zeros(length(sim))
    for i in range(1, length(sim));# for each sim/current step
        if SciMLBase.successful_retcode(sim[i].retcode)
            t_ind = step_start .<= sim[i].t .<= step_end
            # Find local maxima (spike peaks) in soma voltage
            xpks = argmaxima(sim[i][1,t_ind])
            (peaks, proms) =peakproms(xpks, sim[i][1,t_ind], strict=true, minprom=25, maxprom=nothing)
            if length(peaks) >= 2 # need ≥ 2 spikes to estimate frequency
                spiket = sim[i].t[t_ind][peaks]
                ISI = zeros(length(peaks)-1) # interspike intervals (ms)
                for i in range(2,length(peaks)); ISI[i-1] = (spiket[i]- spiket[i-1])  end
                Finst = 1 ./ (ISI/1000)# convert to frequency (Hz)
                F[i] = mean(Finst)
                # end
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

