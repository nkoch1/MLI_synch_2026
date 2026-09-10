

module CSC_cable_2
using DrWatson
@quickactivate "gapjxns"
using DifferentialEquations
using Parameters
using Statistics, Peaks
# using LoopVectorization


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

# ih = h*gbarh*(v- Eh)
# dh/dt =  (hinf(v) - h)/tauh(v)
hinf(v) = 1 / (1+exp( (v-v_inf_half) / v_inf_k) )
tauh(v) = (ratetau / (v_tau_const * ( exp( (v-v_tau_half1) / v_tau_k1) + exp( (v-v_tau_half2) / v_tau_k2) )))/qt

const rl = 110 # Ω*cm  -  specific intracellular resisitivity Rizza et al 2021
const capm = 1.50148 # μF/cm^2  -  specific membrane capacitance
const D = 5.3e-9 #0.53 *1e-4*1e-4 # μm^2/ms-> cm^2/ms
const A = 2 * 1.602176634e−19/(1.380649e−20 *309.15)
samp_rate = 0.1


function check_gating_bounds(x, numseg)

    # CELL 1 #####
    for i in 2:8 # gating variables and Ca
        for j in 1:Int(numseg)+1 # soma and all of cable
            if x[i,j] < 0
                x[i,j] = 0.
            end
        end
    end

    for i in 2:7  # gating variables
        for j in 1:Int(numseg)+1# soma and all of cable
            if x[i, j] > 1
                x[i, j] = 1.
            end
        end
    end

    # CELL 2 ######
    for i in 10:16 # gating variables and Ca
        for j in 1:Int(numseg)+1 # soma and all of cable
            if x[i,j] < 0
                x[i,j] = 0.
            end
        end
    end

    for i in 10:15  # gating variables
        for j in 1:Int(numseg)+1# soma and all of cable
            if x[i, j] > 1
                x[i, j] = 1.
            end
        end
    end
    return x
end

function check_gating_bounds!(x, numseg)

    # CELL 1 #####
    # 8 → ≥ 0
    @inbounds @views x[8, :] = max.(x[8, :], 0)

    # 2:7 → [0, 1]
    @inbounds @views x[2:7, :] = clamp.(x[2:7, :], 0, 1)

    # CELL 2 #####
    # 8 → ≥ 0
    @inbounds @views x[16, :] = max.(x[8, :], 0)

    # 2:7 → [0, 1]
    @inbounds @views x[10:15, :] = clamp.(x[2:7, :], 0, 1)



    return nothing
end

function check_gating_bounds_1(x, numseg)
    for i in 2:8 # gating variables and Ca
        for j in 1:Int(numseg)+1 # soma and all of cable
            if x[i,j] < 0
                x[i,j] = 0.
            end
        end
    end

    for i in 2:7  # gating variables
        for j in 1:Int(numseg)+1# soma and all of cable
            if x[i, j] > 1
                x[i, j] = 1.
            end
        end
    end


    return x
end


function check_gating_bounds_1!(x, numseg)
    # 8 → ≥ 0
    @inbounds @views x[8, :] = max.(x[8, :], 0)

    # 2:7 → [0, 1]
    @inbounds @views x[2:7, :] = clamp.(x[2:7, :], 0, 1)



    return nothing
end

mSS_f(v, Vm, km) =  1 / (1 + exp((v - Vm) / -km));
mTSS_f(v, VmT, kmT) = @. 1 / (1 + exp(-(v - VmT) / kmT));

# MTSS_v = similar(V_array)
function mTSS_f!(y, v, VmT, kmT)
    for (i, vl) in pairs(v)
        y[i] = 1 / (1 + exp(-(vl - VmT) / kmT))
    end
end


n_inf(v, Vn, kn)= 1 / (1 + exp(-(v - Vn) / kn))
n_tau(v) = 6 / (1 + exp((v + 23) / 15))
nA_inf(v, VnA, knA) = 1 / (1 + exp((v - VnA) / -knA))
hA_inf(v, VhA, khA) = 1 / (1 + exp((v - VhA) / khA))
hT_inf(v, VhT, khT) = 1 / (1 + exp((v - VhT) / khT))
mHVA_inf(v, VmHVA, kmHVA) = 1 / (1 + exp(-(v - VmHVA) / kmHVA))

function CSCcable_2!(dr, r, p, t)
    gNa::Float64, gK::Float64, gleak::Float64, gA::Float64, gT::Float64, gHVA::Float64, gSK::Float64, gh::Float64, gKd::Float64, gleakd::Float64, gAd::Float64, gTd::Float64, gHVAd::Float64, gSKd::Float64,ghd::Float64, soma_r::Float64, dend_r::Float64, I::Float64, L::Float64, Eleak::Float64, Eleakd::Float64, numseg::Float64, Eh::Float64, Vm::Float64, Vh::Float64, Vn::Float64, VnA::Float64, VhA::Float64, VmT::Float64, VhT::Float64, VmHVA::Float64, km::Float64,kmT::Float64, kh::Float64, kn::Float64, knA::Float64, khA::Float64,khT::Float64, kmHVA::Float64,  EK::Float64, ENa::Float64, ECa::Float64, gapseg::Tuple{Int64, Int64}, gc::Float64 = p
    r = check_gating_bounds(r, numseg)
    # check_gating_bounds!(r, numseg)
    u = @view r[:, :]
    du = @view dr[:, :]

    Δx = L / numseg # cm
    a = dend_r/10000 #  μm => cm
    As = soma_r /10000
    rmem = 1/(gleakd/1000) # Ω/cm^2 (divide by 1000 to get S-> Ω instead of mS-> kΩ)  -
    τm = rmem * capm /1000  # μF/cm^2 * Ω*cm^2 = uF*Ω = μsec /1000 to get ms
    λ = sqrt(a*rmem/(2*rl))
    κ = 1. /(rl/1000) * (a/(2*As))^2
    γs = 1/(gleak)


    # CELL 1 ############################
    # steady state gating variables
    mSS =  @inbounds @views 1 / (1 + exp((u[1,1] - Vm) / -km));
    mTSS = @inbounds @views @. 1 / (1 + exp(-(u[1, :] - VmT) / kmT));
    @inbounds @views  begin
        du[1,1] =  (I
                    - gK  * (u[3,1]^4) * (u[1,1]  - EK)
                    - gNa  * (mSS^3) * u[2,1] * (u[1,1] - ENa)
                    - gleak   * (u[1,1] - Eleak)
                    - gA * u[4,1] * u[5,1] * (u[1,1] - EK)
                    - gT  * mTSS[1] * u[6,1] * (u[1,1] - ECa)
                    - gSK  * (u[8,1]^5 / ((0.45^5) + (u[8,1]^5))) * (u[1,1]  - EK) #1.0
                    - gHVA  * u[7,1] * (u[1,1] - ECa)
                    - u[2,2]*gh*(u[1,1] - Eh)
                    +  κ  * (u[1,2] - u[1,1]) / Δx # uA/cm^2
                          ) / capm;

        # add Ca diffusion from cable
        du[8,1] = -D/(Δx)^2 *(( u[8, 1]  - u[8, 2]) + A *(u[8, 2] * (u[1, 2] - u[1, 1]) )) 
                - 0.015 * (0.018 * (gT  * mTSS[1] * u[6,1] * (u[1,1] - ECa) + gHVA  * u[7,1] * (u[1,1] - ECa)) + 0.1 * u[8,1]); 
        ## Cable ##

        # gating in each section
        for ii in 1:Int(numseg+1)::Int64 
            du[2, ii] = 0.
            du[3, ii] = ((1 / (1 + exp(-(u[1, ii] - Vn) / kn))) - u[3, ii]) / (6 / (1 + exp((u[1, ii] + 23) / 15))); #n
            du[4, ii] = ((1 / (1 + exp((u[1, ii] - VnA) / -knA))) - u[4, ii]) / 5; #nA
            du[5, ii] = ((1 / (1 + exp((u[1, ii] - VhA) / khA))) - u[5, ii]) / 10; #hA
            du[6, ii] = ((1 / (1 + exp((u[1, ii] - VhT) / khT))) - u[6, ii]) / 15; #hT
            du[7, ii] = ((1 / (1 + exp(-(u[1, ii] - VmHVA) / kmHVA))) - u[7,ii]) / 4; #mHV
        end
        du[2, 1] = ((1 / (1 + exp((u[1, 1] - Vh) / kh))) - u[2, 1]) / ( 0.1 + (2 * 322 * 46) / (4 * pi * (u[1, 1] + 74) ^ 2 + 46 ^ 2)); #k
        du[2, 2] = (hinf(u[1, 1]) - u[2, 2]) / tauh(u[1, 1]) #hcn

        # Ca diffusion along cable 
        du[8, 2] = -D/(Δx)^2 *(( 2 * u[8, 2] - u[8, 1] - u[8, 3]) + A *(u[8, 3] * (u[1, 3] - u[1, 2]) - u[8, 2] * (u[1, 2] - u[1, 1])))
        -0.015 .* (0.018  .* (gTd .* mTSS[2] .* u[6, 2] .* (u[1, 2] .- 22) + gHVAd  .* u[7, 2] .* (u[1, 2] .- 22)) .+ 0.1 .* u[8, 2]) ; 
                   
        
        for ii in 3:Int(numseg)::Int64
            du[8, ii] = -D/(Δx)^2 * (( 2 * u[8, ii] - u[8, ii-1] - u[8, ii+1]) + A * (u[8, ii+1] * (u[1, ii+1] - u[1, ii]) - u[8, ii] * (u[1, ii] - u[1, ii-1])))
                            - 0.015 * (0.018  * (gTd  * mTSS[ii] * u[6, ii] * (u[1, ii] - ECa)
                            + gHVAd * u[7, ii] * (u[1, ii] - ECa)) + 0.1 * u[8, ii]); 

            end
        du[8, end] = -D/(Δx)^2 *(( u[8, end] - u[8, end-1]) + A *( - u[8, end] * (u[1, end] - u[1, end - 1])))  
                - 0.015 .* (0.018 .* (gTd  .* mTSS[end] .* u[6, end] .* (u[1, end] .- 22)
                .+  gHVAd .* u[7, end] .* (u[1, end] .- 22)) .+ 0.1 .* u[8, end]); 

        # Formulism from Goldberg et al. 2007 https://doi.org/10.1152/jn.00810.2006
        # active cable (V)     
        du[1,2] = ((λ^2) / (Δx^2) * (u[1,3]- 2*u[1,2] + u[1,1])
            - gleakd * γs * (u[1,2] - Eleakd)
            - gKd * γs * (u[3,2]^4) * (u[1,2]  - EK)
            - gAd * γs * u[4,2] * u[5,2] * (u[1,2]  - EK)
            - gTd * γs * mTSS[2] * u[6,2] * (u[1,2]-22)
            - gSKd * γs * (u[8,2]^5/((0.45^5)+(u[8,2]^5))) * (u[1,2] - EK)
            - gHVAd * γs * u[7,2] * (u[1,2] - ECa)
            ) / τm
        for ii in 3:Int(numseg)::Int64
            du[1,ii] = ((λ^2) / (Δx^2) * (u[1,ii+1] - 2 *u[1,ii] + u[1,ii-1])
                            - gleakd * γs * (u[1,ii] - Eleakd)
                            - gKd * γs * (u[3,ii]^4) * (u[1,ii] - EK)
                            - gAd * γs * (u[4,ii]) * (u[5,ii]) * (u[1,ii] - EK)
                            - gTd * γs * (mTSS[ii]) * (u[6,ii]) * (u[1,ii] - ECa)
                            - gSKd * γs * (u[8,ii]^5 / ((0.45^5) + (u[8,ii]^5))) * (u[1,ii] - EK)
                            - gHVAd * γs * (u[7,ii]) * (u[1,ii] - ECa)
                            ) / τm
        end
        du[1,end] = ((λ^2) / (Δx^2)  * (-u[1,end] + u[1,end-1])
                - gleakd * γs *(u[1,end] - Eleakd)
                - gKd * γs * (u[3,end]^4) * (u[1,end] - EK)
                - gAd * γs * (u[4,end]) * (u[5,end]) * (u[1,end] - EK)
                - gTd * γs * (mTSS[end]) * (u[6,end]) * (u[1,end] - ECa)
                - gSKd * γs * (u[8,end]^5/((0.45^5) + (u[8,end]^5))) * (u[1,end] - EK)
                - gHVAd * γs * (u[7,end]) * (u[1,end] - ECa)
                ) / τm

        if gapseg[1] == 1 # if cell one is connected at soma
            du[1, gapseg[1]] = du[1, gapseg[1]] - (gc * (u[1, gapseg[1]] - u[9,gapseg[2]]))/capm 
        else
            du[1, gapseg[1]] = du[1, gapseg[1]] - (gc * γs* (u[1, gapseg[1]] - u[9,gapseg[2]]))/ τm
        end

    end

    # CELL 2 ############################
    # steady state gating variables
    mSS2 =  @inbounds @views 1 / (1 + exp((u[9,1] - Vm) / -km));
    mTSS2 = @inbounds @views @. 1 / (1 + exp(-(u[9, :] - VmT) / kmT));
    @inbounds @views  begin
        du[9,1] =  (I
                    - gK  * (u[11,1]^4) * (u[9,1] - EK)
                    - gNa  * (mSS2^3) * u[10,1] * (u[9,1] - ENa)
                    - gleak   * (u[9,1] - Eleak)
                    - gA * u[12,1] * u[13,1] * (u[9,1] - EK)
                    - gT  * mTSS2[1] * u[14,1] * (u[9,1] - ECa)
                    - gSK  * (u[16,1]^5 / ((0.45^5) + (u[16,1]^5))) * (u[9,1] - EK) #1.0
                    - gHVA  * u[15,1] * (u[9,1] - ECa)
                    - u[10,2]*gh*(u[9,1] - Eh)
                    +  κ  * (u[9,2] - u[9,1]) / Δx  # uA/cm^2
                    ) / capm;

        # add Ca diffusion from cable
        du[16,1] = -D/(Δx)^2 *(( u[16, 1]  - u[16, 2]) + A *(u[16, 2] * (u[9, 2] - u[9, 1]) )) 
                - 0.015 * (0.018 * (gT  * mTSS2[1] * u[14,1] * (u[9,1] - ECa) + gHVA  * u[15,1] * (u[9,1] - ECa)) + 0.1 * u[16,1]); 
        ## Cable ##

        # gating in each section
        for ii in 1:Int(numseg+1)::Int64  
            du[10, ii] = 0.
            du[11, ii] = ((1 / (1 + exp(-(u[9, ii] - Vn) / kn))) - u[11, ii]) / (6 / (1 + exp((u[9, ii] + 23) / 15))); #n
            du[12, ii] = ((1 / (1 + exp((u[9, ii] - VnA) / -knA))) - u[12, ii]) / 5; #nA
            du[13, ii] = ((1 / (1 + exp((u[9, ii] - VhA) / khA))) - u[13, ii]) / 10; #hA
            du[14, ii] = ((1 / (1 + exp((u[9, ii] - VhT) / khT))) - u[14, ii]) / 15; #hT
            du[15, ii] = ((1 / (1 + exp(-(u[9, ii] - VmHVA) / kmHVA))) - u[15,ii]) / 4; #mHV
        end
        du[10, 1] = ((1 / (1 + exp((u[9, 1] - Vh) / kh))) - u[10, 1]) / ( 0.1 + (2 * 322 * 46) / (4 * pi * (u[9, 1] + 74) ^ 2 + 46 ^ 2)); #k 
        du[10, 2] = (hinf(u[9, 1]) - u[10, 2]) / tauh(u[9, 1]) #hcn

        # Ca diffusion along cable
        du[16, 2] = -D/(Δx)^2 *(( 2 * u[16, 2] - u[16, 1] - u[16, 3]) + A *(u[16, 3] * (u[9, 3] - u[9, 2]) - u[16, 2] * (u[9, 2] - u[9, 1])))
                            -0.015 .* (0.018  .* (gTd .* mTSS2[2] .* u[14, 2] .* (u[9, 2] .- 22) + gHVAd  .* u[15, 2] .* (u[9, 2] .- 22)) .+ 0.1 .* u[16, 2]) ; 
        
        
        for ii in 3:Int(numseg)::Int64                   
            du[16, ii] = -D/(Δx)^2 * (( 2 * u[16, ii] - u[16, ii-1] - u[16, ii+1]) + A * (u[16, ii+1] * (u[9, ii+1] - u[9, ii]) - u[16, ii] * (u[9, ii] - u[9, ii-1])))
                            - 0.015 * (0.018  * (gTd  * mTSS2[ii] * u[14, ii] * (u[9, ii] - ECa)
                            + gHVAd * u[15, ii] * (u[9, ii] - ECa)) + 0.1 * u[16, ii]); 
            end
        du[16, end] = -D/(Δx)^2 *(( u[16, end] - u[16, end-1]) + A *( - u[16, end] * (u[9, end] - u[9, end - 1])))  
                - 0.015 .* (0.018 .* (gTd  .* mTSS2[end] .* u[14, end] .* (u[9, end] .- 22)
                .+  gHVAd  .* u[15, end] .* (u[9, end] .- 22)) .+ 0.1 .* u[16, end]); 

        # Formulism from Goldberg et al. 2007 https://doi.org/10.1152/jn.00810.2006
        # active cable (V)     
        du[9,2] = ((λ^2) / (Δx^2) * (u[9,3]- 2*u[9,2] + u[9,1])
            - gleakd * γs * (u[9,2] - Eleakd)
            - gKd * γs * (u[11,2]^4) * (u[9,2] - EK)
            - gAd * γs * u[12,2] * u[13,2] * (u[9,2] - EK)
            - gTd * γs * mTSS2[2] * u[14,2] * (u[9,2]-22)
            - gSKd * γs * (u[16,2]^5/((0.45^5)+(u[16,2]^5))) * (u[9,2] - EK)
            - gHVAd * γs * u[15,2] * (u[9,2] - ECa)
            ) / τm
        for ii in 3:Int(numseg)::Int64
            du[9,ii] = ((λ^2) / (Δx^2)  * (u[9,ii+1] - 2 *u[9,ii] + u[9,ii-1])
                            - gleakd * γs * (u[9,ii] - Eleakd)
                            - gKd * γs * (u[11,ii]^4) * (u[9,ii] - EK)
                            - gAd * γs * (u[12,ii]) * (u[13,ii]) * (u[9,ii] - EK)
                            - gTd * γs * (mTSS2[ii]) * (u[14,ii]) * (u[9,ii] - ECa)
                            - gSKd * γs * (u[16,ii]^5 / ((0.45^5) + (u[16,ii]^5))) * (u[9,ii] - EK)
                            - gHVAd * γs * (u[15,ii]) * (u[9,ii] - ECa)
                            ) / τm
        end

        du[9,end] = ((λ^2) / (Δx^2) * (-u[9,end] + u[9,end-1])
            - gleakd * γs *(u[9,end] - Eleakd)
            - gKd * γs * (u[11,end]^4) * (u[9,end] - EK)
            - gAd * γs * (u[12,end]) * (u[13,end]) * (u[9,end] - EK)
            - gTd * γs * (mTSS2[end]) * (u[14,end]) * (u[9,end] - ECa)
            - gSKd * γs * (u[16,end]^5/((0.45^5) + (u[16,end]^5))) * (u[9,end] - EK)
            - gHVAd * γs * (u[15,end]) * (u[9,end] - ECa)
            ) / τm

        if gapseg[2] == 1 # if cell two is connected at soma
            du[9, gapseg[2]] = du[9, gapseg[2]] - (gc * (u[9, gapseg[2]] - u[1,gapseg[1]])) /capm 
        else
            du[9, gapseg[2]] = du[9, gapseg[2]] - (gc * γs * (u[9, gapseg[2]] - u[1,gapseg[1]]))/ τm
        end
    end
end


function CSCcable_2f!(dr, r, p, t)
    # gNa::Float64, gK::Float64, gleak::Float64, gA::Float64, gT::Float64, gHVA::Float64, gSK::Float64, gh::Float64, gKd::Float64, gleakd::Float64, gAd::Float64, gTd::Float64, gHVAd::Float64, gSKd::Float64,ghd::Float64, soma_r::Float64, dend_r::Float64, I::Float64, L::Float64, Eleak::Float64, Eleakd::Float64, numseg::Float64, Eh::Float64, Vm::Float64, Vh::Float64, Vn::Float64, VnA::Float64, VhA::Float64, VmT::Float64, VhT::Float64, VmHVA::Float64, km::Float64,kmT::Float64, kh::Float64, kn::Float64, knA::Float64, khA::Float64,khT::Float64, kmHVA::Float64,  EK::Float64, ENa::Float64, ECa::Float64, gapseg::Tuple{Int64, Int64}, gc::Float64 = p
    gNa::Float64, gK::Float64, gleak::Float64, gA::Float64, gT::Float64, gHVA::Float64, gSK::Float64, gh::Float64, gKd::Float64, gleakd::Float64, gAd::Float64, gTd::Float64, gHVAd::Float64, gSKd::Float64,ghd::Float64, As::Float64, a::Float64, I::Float64, Δx::Float64, Eleak::Float64, Eleakd::Float64, numseg::Float64, Eh::Float64, Vm::Float64, Vh::Float64, Vn::Float64, VnA::Float64, VhA::Float64, VmT::Float64, VhT::Float64, VmHVA::Float64, km::Float64,kmT::Float64, kh::Float64, kn::Float64, knA::Float64, khA::Float64,khT::Float64, kmHVA::Float64,  EK::Float64, ENa::Float64, ECa::Float64,γs::Float64, τm::Float64, κ::Float64, Ddx::Float64, λdx::Float64, gapseg::Tuple{Int64, Int64}, gc::Float64 = p
    # gNa, gK, gleak, gA, gT, gHVA, gSK, gh, gKd, gleakd, gAd, gTd, gHVAd, gSKd,ghd, soma_r, dend_r, I, L, Eleak, Eleakd, numseg, Eh, Vm, Vh, Vn, VnA, VhA, VmT, VhT, VmHVA, km,kmT, kh, kn, knA, khA,khT, kmHVA,  EK, ENa, ECa, gapseg, gc = p

    # r = check_gating_bounds(r, numseg)
    u = @view r[:, :]
    du = @view dr[:, :]

    # Δx = L / numseg # cm
    # a = dend_r/10000 #  μm => cm
    # As = soma_r /10000
    # rmem = 1/(gleakd/1000) # Ω/cm^2 (divide by 1000 to get S-> Ω instead of mS-> kΩ)  -
    # τm = rmem * capm /1000  # μF/cm^2 * Ω*cm^2 = uF*Ω = μsec /1000 to get ms
    # λ = sqrt(a*rmem/(2*rl))
    # κ = 1. /(rl/1000) * (a/(2*As))^2
    # γs = 1/(gleak)


    # CELL 1 ############################
    # steady state gating variables
    mSS =  @inbounds @views mSS_f(u[1,1], Vm,km);
    mTSS = @inbounds @views mTSS_f(u[1, :], VmT, kmT);
    # mSS =  @inbounds @views 1 / (1 + exp((u[1,1] - Vm) / -km));
    # mTSS = @inbounds @views @. 1 / (1 + exp(-(u[1, :] - VmT) / kmT));
    @inbounds @views  begin
        du[1,1] =  (I
                    - gK  * (u[3,1]^4) * (u[1,1]  - EK)
                    - gNa  * (mSS^3) * u[2,1] * (u[1,1] - ENa)
                    - gleak   * (u[1,1] - Eleak)
                    - gA * u[4,1] * u[5,1] * (u[1,1] - EK)
                    - gT  * mTSS[1] * u[6,1] * (u[1,1] - ECa)
                    - gSK  * (u[8,1]^5 / ((0.45^5) + (u[8,1]^5))) * (u[1,1]  - EK) #1.0
                    - gHVA  * u[7,1] * (u[1,1] - ECa)
                    - u[2,2]*gh*(u[1,1] - Eh)
                    # +  κ  * (u[1,2] - u[1,1]) / Δx # uA/cm^2
                    +  κ  * (u[1,2] - u[1,1]) # uA/cm^2
                          ) / capm;

        # add Ca diffusion from cable
        du[8,1] = -Ddx *(( u[8, 1]  - u[8, 2]) + A *(u[8, 2] * (u[1, 2] - u[1, 1]) )) 
                - 0.015 * (0.018 * (gT  * mTSS[1] * u[6,1] * (u[1,1] - ECa) + gHVA  * u[7,1] * (u[1,1] - ECa)) + 0.1 * u[8,1]); 
        # du[8,1] = -D/(Δx)^2 *(( u[8, 1]  - u[8, 2]) + A *(u[8, 2] * (u[1, 2] - u[1, 1]) )) 
        #         - 0.015 * (0.018 * (gT  * mTSS[1] * u[6,1] * (u[1,1] - ECa) + gHVA  * u[7,1] * (u[1,1] - ECa)) + 0.1 * u[8,1]); 
        ## Cable ##

        # gating in each section
        for ii in 1:Int(numseg+1)::Int64 
            du[2, ii] = 0.
            du[3, ii] = (n_inf(u[1, ii], Vn, kn) - u[3, ii]) / n_tau(u[1,ii]); #n
            du[4, ii] = (nA_inf(u[1, ii], VnA, knA) - u[4, ii]) / 5; #nA
            du[5, ii] = (hA_inf(u[1, ii], VhA, khA) - u[5, ii]) / 10; #hA
            du[6, ii] = (hT_inf(u[1, ii], VhT, khT)- u[6, ii]) / 15; #hT
            du[7, ii] = (mHVA_inf(u[1, ii], VmHVA, kmHVA)  - u[7,ii]) / 4; #mHV
            # du[3, ii] = ((1 / (1 + exp(-(u[1, ii] - Vn) / kn))) - u[3, ii]) / (6 / (1 + exp((u[1, ii] + 23) / 15))); #n
            # du[4, ii] = ((1 / (1 + exp((u[1, ii] - VnA) / -knA))) - u[4, ii]) / 5; #nA
            # du[5, ii] = ((1 / (1 + exp((u[1, ii] - VhA) / khA))) - u[5, ii]) / 10; #hA
            # du[6, ii] = ((1 / (1 + exp((u[1, ii] - VhT) / khT))) - u[6, ii]) / 15; #hT
            # du[7, ii] = ((1 / (1 + exp(-(u[1, ii] - VmHVA) / kmHVA))) - u[7,ii]) / 4; #mHV
        end

        # du[2, 1] = ((1 / (1 + exp((u[1, 1] - Vh) / kh))) - u[2, 1]) / ( 0.1 + (29624) / (4 * pi * (u[1, 1] + 74) ^ 2 + 2116)); #k
        du[2, 1] = ((1 / (1 + exp((u[1, 1] - Vh) / kh))) - u[2, 1]) / ( 0.1 + (2 * 322 * 46) / (4 * pi * (u[1, 1] + 74) ^ 2 + 46 ^ 2)); #k
        du[2, 2] = (hinf(u[1, 1]) - u[2, 2]) / tauh(u[1, 1]) #hcn
        # Ca diffusion along cable
        du[8, 2] = -Ddx *(( 2 * u[8, 2] - u[8, 1] - u[8, 3]) + A *(u[8, 3] * (u[1, 3] - u[1, 2]) - u[8, 2] * (u[1, 2] - u[1, 1])))
                            -0.015 .* (0.018  .* (gTd/γs .* mTSS[2] .* u[6, 2] .* (u[1, 2] .- 22) + gHVAd/γs  .* u[7, 2] .* (u[1, 2] .- 22)) .+ 0.1 .* u[8, 2]) ; 
        # du[8, 2] = -D/(Δx)^2 *(( 2 * u[8, 2] - u[8, 1] - u[8, 3]) + A *(u[8, 3] * (u[1, 3] - u[1, 2]) - u[8, 2] * (u[1, 2] - u[1, 1])))
        # -0.015 .* (0.018  .* (gTd .* mTSS[2] .* u[6, 2] .* (u[1, 2] .- 22) + gHVAd  .* u[7, 2] .* (u[1, 2] .- 22)) .+ 0.1 .* u[8, 2]) ; 
                   
        
        for ii in 3:Int(numseg)::Int64
            du[8, ii] = -Ddx * (( 2 * u[8, ii] - u[8, ii-1] - u[8, ii+1]) + A * (u[8, ii+1] * (u[1, ii+1] - u[1, ii]) - u[8, ii] * (u[1, ii] - u[1, ii-1])))
                            - 0.015 * (0.018  * (gTd/γs  * mTSS[ii] * u[6, ii] * (u[1, ii] - ECa)
                            + gHVAd/γs * u[7, ii] * (u[1, ii] - ECa)) + 0.1 * u[8, ii]); 
            # du[8, ii] = -D/(Δx)^2 * (( 2 * u[8, ii] - u[8, ii-1] - u[8, ii+1]) + A * (u[8, ii+1] * (u[1, ii+1] - u[1, ii]) - u[8, ii] * (u[1, ii] - u[1, ii-1])))
            #                 - 0.015 * (0.018  * (gTd  * mTSS[ii] * u[6, ii] * (u[1, ii] - ECa)
            #                 + gHVAd * u[7, ii] * (u[1, ii] - ECa)) + 0.1 * u[8, ii]); 
            du[1,ii] = (λdx * (u[1,ii+1] - 2 *u[1,ii] + u[1,ii-1])
                            - gleakd * (u[1,ii] - Eleakd)
                            - gKd * (u[3,ii]^4) * (u[1,ii] - EK)
                            - gAd * (u[4,ii]) * (u[5,ii]) * (u[1,ii] - EK)
                            - gTd * (mTSS[ii]) * (u[6,ii]) * (u[1,ii] - ECa)
                            - gSKd * (u[8,ii]^5 / ((0.45^5) + (u[8,ii]^5))) * (u[1,ii] - EK)
                            - gHVAd * (u[7,ii]) * (u[1,ii] - ECa)
                            ) / τm
            end
        du[8, end] = -Ddx *(( u[8, end] - u[8, end-1]) + A *( - u[8, end] * (u[1, end] - u[1, end - 1])))  
                - 0.015 .* (0.018 .* (gTd/γs  .* mTSS[end] .* u[6, end] .* (u[1, end] .- 22)
                .+  gHVAd/γs .* u[7, end] .* (u[1, end] .- 22)) .+ 0.1 .* u[8, end]); 
        # du[8, end] = -D/(Δx)^2 *(( u[8, end] - u[8, end-1]) + A *( - u[8, end] * (u[1, end] - u[1, end - 1])))  
        #         - 0.015 .* (0.018 .* (gTd  .* mTSS[end] .* u[6, end] .* (u[1, end] .- 22)
        #         .+  gHVAd .* u[7, end] .* (u[1, end] .- 22)) .+ 0.1 .* u[8, end]); 

        # Formulism from Goldberg et al. 2007 https://doi.org/10.1152/jn.00810.2006
        # active cable (V)     
        du[1,2] = (λdx * (u[1,3]- 2*u[1,2] + u[1,1])
            - gleakd * (u[1,2] - Eleakd)
            - gKd * (u[3,2]^4) * (u[1,2]  - EK)
            - gAd * u[4,2] * u[5,2] * (u[1,2]  - EK)
            - gTd * mTSS[2] * u[6,2] * (u[1,2]-22)
            - gSKd * (u[8,2]^5/((0.45^5)+(u[8,2]^5))) * (u[1,2] - EK)
            - gHVAd * u[7,2] * (u[1,2] - ECa)
            ) / τm
        # du[1,2] = ((λ^2) / (Δx^2) * (u[1,3]- 2*u[1,2] + u[1,1])
        #     - gleakd * γs * (u[1,2] - Eleakd)
        #     - gKd * γs * (u[3,2]^4) * (u[1,2]  - EK)
        #     - gAd * γs * u[4,2] * u[5,2] * (u[1,2]  - EK)
        #     - gTd * γs * mTSS[2] * u[6,2] * (u[1,2]-22)
        #     - gSKd * γs * (u[8,2]^5/((0.45^5)+(u[8,2]^5))) * (u[1,2] - EK)
        #     - gHVAd * γs * u[7,2] * (u[1,2] - ECa)
        #     ) / τm
        # for ii in 3:Int(numseg)::Int64
        #     du[1,ii] = ((λ^2) / (Δx^2) * (u[1,ii+1] - 2 *u[1,ii] + u[1,ii-1])
        #                     - gleakd * γs * (u[1,ii] - Eleakd)
        #                     - gKd * γs * (u[3,ii]^4) * (u[1,ii] - EK)
        #                     - gAd * γs * (u[4,ii]) * (u[5,ii]) * (u[1,ii] - EK)
        #                     - gTd * γs * (mTSS[ii]) * (u[6,ii]) * (u[1,ii] - ECa)
        #                     - gSKd * γs * (u[8,ii]^5 / ((0.45^5) + (u[8,ii]^5))) * (u[1,ii] - EK)
        #                     - gHVAd * γs * (u[7,ii]) * (u[1,ii] - ECa)
        #                     ) / τm
        # end

        du[1,end] = (λdx * (-u[1,end] + u[1,end-1])
                - gleakd *(u[1,end] - Eleakd)
                - gKd * (u[3,end]^4) * (u[1,end] - EK)
                - gAd * (u[4,end]) * (u[5,end]) * (u[1,end] - EK)
                - gTd * (mTSS[end]) * (u[6,end]) * (u[1,end] - ECa)
                - gSKd * (u[8,end]^5/((0.45^5) + (u[8,end]^5))) * (u[1,end] - EK)
                - gHVAd * (u[7,end]) * (u[1,end] - ECa)
                ) / τm
        # du[1,end] = ((λ^2) / (Δx^2)  * (-u[1,end] + u[1,end-1])
        #         - gleakd * γs *(u[1,end] - Eleakd)
        #         - gKd * γs * (u[3,end]^4) * (u[1,end] - EK)
        #         - gAd * γs * (u[4,end]) * (u[5,end]) * (u[1,end] - EK)
        #         - gTd * γs * (mTSS[end]) * (u[6,end]) * (u[1,end] - ECa)
        #         - gSKd * γs * (u[8,end]^5/((0.45^5) + (u[8,end]^5))) * (u[1,end] - EK)
        #         - gHVAd * γs * (u[7,end]) * (u[1,end] - ECa)
        #         ) / τm

        if gapseg[1] == 1 # if cell one is connected at soma
            du[1, gapseg[1]] = du[1, gapseg[1]] - (gc * (u[1, gapseg[1]] - u[9,gapseg[2]]))/capm 
        else
            du[1, gapseg[1]] = du[1, gapseg[1]] - (gc * γs* (u[1, gapseg[1]] - u[9,gapseg[2]]))/ τm
        end

    end

    # CELL 2 ############################
    # steady state gating variables
    mSS2 =  @inbounds @views 1 / (1 + exp((u[9,1] - Vm) / -km));
    mTSS2 = @inbounds @views mTSS_f(u[9, :], VmT, kmT);
    # mSS2 =  @inbounds @views 1 / (1 + exp((u[9,1] - Vm) / -km));
    # mTSS2 = @inbounds @views @. 1 / (1 + exp(-(u[9, :] - VmT) / kmT));
    @inbounds @views  begin
        du[9,1] =  (I
                    - gK  * (u[11,1]^4) * (u[9,1] - EK)
                    - gNa  * (mSS2^3) * u[10,1] * (u[9,1] - ENa)
                    - gleak   * (u[9,1] - Eleak)
                    - gA * u[12,1] * u[13,1] * (u[9,1] - EK)
                    - gT  * mTSS2[1] * u[14,1] * (u[9,1] - ECa)
                    - gSK  * (u[16,1]^5 / ((0.45^5) + (u[16,1]^5))) * (u[9,1] - EK) #1.0
                    - gHVA  * u[15,1] * (u[9,1] - ECa)
                    - u[10,2]*gh*(u[9,1] - Eh)
                    # +  κ  * (u[9,2] - u[9,1]) / Δx  # uA/cm^2
                    +  κ  * (u[9,2] - u[9,1])   # uA/cm^2
                    ) / capm;

        # add Ca diffusion from cable
        du[16,1] = -Ddx *(( u[16, 1]  - u[16, 2]) + A *(u[16, 2] * (u[9, 2] - u[9, 1]) )) 
                - 0.015 * (0.018 * (gT  * mTSS2[1] * u[14,1] * (u[9,1] - ECa) + gHVA  * u[15,1] * (u[9,1] - ECa)) + 0.1 * u[16,1]); 
        # du[16,1] = -D/(Δx)^2 *(( u[16, 1]  - u[16, 2]) + A *(u[16, 2] * (u[9, 2] - u[9, 1]) )) 
        #         - 0.015 * (0.018 * (gT  * mTSS2[1] * u[14,1] * (u[9,1] - ECa) + gHVA  * u[15,1] * (u[9,1] - ECa)) + 0.1 * u[16,1]); 
        ## Cable ##
        # gating in each section
        for ii in 1:Int(numseg+1)::Int64  
            du[10, ii] = 0.
            du[11, ii] = (n_inf(u[9, ii], Vn, kn) - u[11, ii]) / n_tau(u[9,ii]); #n
            du[12, ii] = (nA_inf(u[9, ii], VnA, knA) - u[12, ii]) / 5; #nA
            du[13, ii] = (hA_inf(u[9, ii], VhA, khA) - u[13, ii]) / 10; #hA
            du[14, ii] = (hT_inf(u[9, ii], VhT, khT)- u[14, ii]) / 15; #hT
            du[15, ii] = (mHVA_inf(u[9, ii], VmHVA, kmHVA)  - u[15,ii]) / 4; #mHV
            # du[11, ii] = ((1 / (1 + exp(-(u[9, ii] - Vn) / kn))) - u[11, ii]) / (6 / (1 + exp((u[9, ii] + 23) / 15))); #n
            # du[12, ii] = ((1 / (1 + exp((u[9, ii] - VnA) / -knA))) - u[12, ii]) / 5; #nA
            # du[13, ii] = ((1 / (1 + exp((u[9, ii] - VhA) / khA))) - u[13, ii]) / 10; #hA
            # du[14, ii] = ((1 / (1 + exp((u[9, ii] - VhT) / khT))) - u[14, ii]) / 15; #hT
            # du[15, ii] = ((1 / (1 + exp(-(u[9, ii] - VmHVA) / kmHVA))) - u[15,ii]) / 4; #mHV
        end
        # du[10, 1] = ((1 / (1 + exp((u[9, 1] - Vh) / kh))) - u[10, 1]) / ( 0.1 + (29624) / (4 * pi * (u[9, 1] + 74) ^ 2 + 2116)); #k
        du[10, 1] = ((1 / (1 + exp((u[9, 1] - Vh) / kh))) - u[10, 1]) / ( 0.1 + (2 * 322 * 46) / (4 * pi * (u[9, 1] + 74) ^ 2 + 46 ^ 2)); #k 
        du[10, 2] = (hinf(u[9, 1]) - u[10, 2]) / tauh(u[9, 1]) #hcn

        # Ca diffusion along cable
        du[16, 2] = -Ddx *(( 2 * u[16, 2] - u[16, 1] - u[16, 3]) + A *(u[16, 3] * (u[9, 3] - u[9, 2]) - u[16, 2] * (u[9, 2] - u[9, 1])))
                            -0.015 .* (0.018  .* (gTd/γs .* mTSS2[2] .* u[14, 2] .* (u[9, 2] .- 22) + gHVAd/γs  .* u[15, 2] .* (u[9, 2] .- 22)) .+ 0.1 .* u[16, 2]) ; 
        # du[16, 2] = -D/(Δx)^2 *(( 2 * u[16, 2] - u[16, 1] - u[16, 3]) + A *(u[16, 3] * (u[9, 3] - u[9, 2]) - u[16, 2] * (u[9, 2] - u[9, 1])))
        #                     -0.015 .* (0.018  .* (gTd .* mTSS2[2] .* u[14, 2] .* (u[9, 2] .- 22) + gHVAd  .* u[15, 2] .* (u[9, 2] .- 22)) .+ 0.1 .* u[16, 2]) ; 
        
        
        for ii in 3:Int(numseg)::Int64                   
            du[16, ii] = -Ddx * (( 2 * u[16, ii] - u[16, ii-1] - u[16, ii+1]) + A * (u[16, ii+1] * (u[9, ii+1] - u[9, ii]) - u[16, ii] * (u[9, ii] - u[9, ii-1])))
                            - 0.015 * (0.018  * (gTd/γs  * mTSS2[ii] * u[14, ii] * (u[9, ii] - ECa)
                            + gHVAd * u[15, ii] * (u[9, ii] - ECa)) + 0.1 * u[16, ii]); 
            # du[16, ii] = -D/(Δx)^2 * (( 2 * u[16, ii] - u[16, ii-1] - u[16, ii+1]) + A * (u[16, ii+1] * (u[9, ii+1] - u[9, ii]) - u[16, ii] * (u[9, ii] - u[9, ii-1])))
            #                 - 0.015 * (0.018  * (gTd  * mTSS2[ii] * u[14, ii] * (u[9, ii] - ECa)
            #                 + gHVAd * u[15, ii] * (u[9, ii] - ECa)) + 0.1 * u[16, ii]); 
            du[9,ii] = (λdx * (u[9,ii+1] - 2 *u[9,ii] + u[9,ii-1])
                            - gleakd * (u[9,ii] - Eleakd)
                            - gKd * (u[11,ii]^4) * (u[9,ii] - EK)
                            - gAd * (u[12,ii]) * (u[13,ii]) * (u[9,ii] - EK)
                            - gTd * (mTSS2[ii]) * (u[14,ii]) * (u[9,ii] - ECa)
                            - gSKd * (u[16,ii]^5 / ((0.45^5) + (u[16,ii]^5))) * (u[9,ii] - EK)
                            - gHVAd * (u[15,ii]) * (u[9,ii] - ECa)
                            ) / τm
            end
        du[16, end] = -Ddx *(( u[16, end] - u[16, end-1]) + A *( - u[16, end] * (u[9, end] - u[9, end - 1])))  
                - 0.015 .* (0.018 .* (gTd  .* mTSS2[end] .* u[14, end] .* (u[9, end] .- 22)
                .+  gHVAd  .* u[15, end] .* (u[9, end] .- 22)) .+ 0.1 .* u[16, end]); 
        # du[16, end] = -D/(Δx)^2 *(( u[16, end] - u[16, end-1]) + A *( - u[16, end] * (u[9, end] - u[9, end - 1])))  
        #         - 0.015 .* (0.018 .* (gTd  .* mTSS2[end] .* u[14, end] .* (u[9, end] .- 22)
        #         .+  gHVAd  .* u[15, end] .* (u[9, end] .- 22)) .+ 0.1 .* u[16, end]); 

        # Formulism from Goldberg et al. 2007 https://doi.org/10.1152/jn.00810.2006
        # active cable (V)     
        du[9,2] = (λdx * (u[9,3]- 2*u[9,2] + u[9,1])
            - gleakd * (u[9,2] - Eleakd)
            - gKd * (u[11,2]^4) * (u[9,2] - EK)
            - gAd * u[12,2] * u[13,2] * (u[9,2] - EK)
            - gTd * mTSS2[2] * u[14,2] * (u[9,2]-22)
            - gSKd * (u[16,2]^5/((0.45^5)+(u[16,2]^5))) * (u[9,2] - EK)
            - gHVAd * u[15,2] * (u[9,2] - ECa)
            ) / τm
        # du[9,2] = ((λ^2) / (Δx^2) * (u[9,3]- 2*u[9,2] + u[9,1])
        #     - gleakd * γs * (u[9,2] - Eleakd)
        #     - gKd * γs * (u[11,2]^4) * (u[9,2] - EK)
        #     - gAd * γs * u[12,2] * u[13,2] * (u[9,2] - EK)
        #     - gTd * γs * mTSS2[2] * u[14,2] * (u[9,2]-22)
        #     - gSKd * γs * (u[16,2]^5/((0.45^5)+(u[16,2]^5))) * (u[9,2] - EK)
        #     - gHVAd * γs * u[15,2] * (u[9,2] - ECa)
        #     ) / τm
        # for ii in 3:Int(numseg)::Int64
        #     du[9,ii] = ((λ^2) / (Δx^2)  * (u[9,ii+1] - 2 *u[9,ii] + u[9,ii-1])
        #                     - gleakd * γs * (u[9,ii] - Eleakd)
        #                     - gKd * γs * (u[11,ii]^4) * (u[9,ii] - EK)
        #                     - gAd * γs * (u[12,ii]) * (u[13,ii]) * (u[9,ii] - EK)
        #                     - gTd * γs * (mTSS2[ii]) * (u[14,ii]) * (u[9,ii] - ECa)
        #                     - gSKd * γs * (u[16,ii]^5 / ((0.45^5) + (u[16,ii]^5))) * (u[9,ii] - EK)
        #                     - gHVAd * γs * (u[15,ii]) * (u[9,ii] - ECa)
        #                     ) / τm
        # end

        du[9,end] = (λdx * (-u[9,end] + u[9,end-1])
                - gleakd *(u[9,end] - Eleakd)
                - gKd * (u[11,end]^4) * (u[9,end] - EK)
                - gAd * (u[12,end]) * (u[13,end]) * (u[9,end] - EK)
                - gTd * (mTSS2[end]) * (u[14,end]) * (u[9,end] - ECa)
                - gSKd * (u[16,end]^5/((0.45^5) + (u[16,end]^5))) * (u[9,end] - EK)
                - gHVAd * (u[15,end]) * (u[9,end] - ECa)
                ) / τm

        # du[9,end] = ((λ^2) / (Δx^2) * (-u[9,end] + u[9,end-1])
        #     - gleakd * γs *(u[9,end] - Eleakd)
        #     - gKd * γs * (u[11,end]^4) * (u[9,end] - EK)
        #     - gAd * γs * (u[12,end]) * (u[13,end]) * (u[9,end] - EK)
        #     - gTd * γs * (mTSS2[end]) * (u[14,end]) * (u[9,end] - ECa)
        #     - gSKd * γs * (u[16,end]^5/((0.45^5) + (u[16,end]^5))) * (u[9,end] - EK)
        #     - gHVAd * γs * (u[15,end]) * (u[9,end] - ECa)
        #     ) / τm

        if gapseg[2] == 1 # if cell two is connected at soma
            du[9, gapseg[2]] = du[9, gapseg[2]] - (gc * (u[9, gapseg[2]] - u[1,gapseg[1]])) /capm 
        else
            du[9, gapseg[2]] = du[9, gapseg[2]] - (gc * γs * (u[9, gapseg[2]] - u[1,gapseg[1]]))/ τm
        end
    end
end





function steady_state_init_u0(V0, numseg, pgating);
    Vm, Vh, Vn, VnA, VhA, VmT, VhT, VmHVA, km,kmT, kh, kn, knA, khA,khT, kmHVA = pgating
    hSS = 1 / (1 + exp((V0 - Vh) / kh));
    Ca = 0.001
    Y0 = [V0, hSS, n_inf(V0, Vn, kn), nA_inf(V0, VnA, knA), hA_inf(V0, VhA, khA), hT_inf(V0, VhT, khT),mHVA_inf(V0, VmHVA, kmHVA), Ca]
    Y = zeros(8,numseg+1)
    Y[1,1] = V0
    Y[1, 2:end] .= V0
    Y[2,1] = Y0[2]
    Y[2,2] = hinf(V0)
    for i in range(3,8); 
        Y[i,:] .= Y0[i]  
    end
    return Y;
    nothing
    end;


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
function CSCcable_gating!(dr, r, p, t)
    gNa::Float64, gK::Float64, gleak::Float64, gA::Float64, gT::Float64, gHVA::Float64, gSK::Float64, gh::Float64, gKd::Float64, gleakd::Float64, gAd::Float64, gTd::Float64, gHVAd::Float64, gSKd::Float64,ghd::Float64, soma_r::Float64, dend_r::Float64, I::Float64, L::Float64, Eleak::Float64, Eleakd::Float64, numseg::Float64, Eh::Float64, Vm::Float64, Vh::Float64, Vn::Float64, VnA::Float64, VhA::Float64, VmT::Float64, VhT::Float64, VmHVA::Float64, km::Float64,kmT::Float64, kh::Float64, kn::Float64, knA::Float64, khA::Float64,khT::Float64, kmHVA::Float64,  EK::Float64, ENa::Float64, ECa::Float64 = p
    # gNa::Float64, gK::Float64, gleak::Float64, gA::Float64, gT::Float64, gHVA::Float64, gSK::Float64, gh::Float64, gKd::Float64, gleakd::Float64, gAd::Float64, gTd::Float64, gHVAd::Float64, gSKd::Float64,ghd::Float64, As::Float64, a::Float64, I::Float64, Δx::Float64, Eleak::Float64, Eleakd::Float64, numseg::Float64, Eh::Float64, Vm::Float64, Vh::Float64, Vn::Float64, VnA::Float64, VhA::Float64, VmT::Float64, VhT::Float64, VmHVA::Float64, km::Float64,kmT::Float64, kh::Float64, kn::Float64, knA::Float64, khA::Float64,khT::Float64, kmHVA::Float64,  EK::Float64, ENa::Float64, ECa::Float64,γs::Float64, τm::Float64, κ::Float64, Ddx::Float64, λdx::Float64, gapseg::Tuple{Int64, Int64}, gc::Float64 = p
    
    # r = check_gating_bounds_1(r, numseg)
    check_gating_bounds_1!(r, numseg)
    u = @view r[:, :]
    du = @view dr[:, :]

    Δx = L / numseg # cm
    a = dend_r/10000 #  μm => cm
    As = soma_r /10000
    rmem = 1/(gleakd/1000) # Ω/cm^2 (divide by 1000 to get S-> Ω instead of mS-> kΩ)  -
    τm = rmem * capm /1000  # μF/cm^2 * Ω*cm^2 = uF*Ω = μsec /1000 to get ms
    λ = sqrt(a*rmem/(2*rl))
    κ = 1. /(rl/1000) * (a/(2*As))^2
    γs = 1/(gleak)

    # steady state gating variables
    # mSS =  @inbounds @views mSS_f(u[1,1], Vm,km);
    # mTSS = @inbounds @views mTSS_f(u[1, :], VmT, kmT);
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
                - u[2,2]*gh*(u[1,1] - Eh)
                +  κ  * (u[1,2] - u[1,1]) / Δx # uA/cm^2
                ) / capm;

    # add Ca diffusion from cable
    # du[8,1] =  -Ddx *(( u[8, 1]  - u[8, 2]) + A *(u[8, 2] * (u[1, 2] - u[1, 1]) )) 
    #             - 0.015 * (0.018 * (gT  * mTSS[1] * u[6,1] * (u[1,1] - 22) + gHVA  * u[7,1] * (u[1,1] - 22)) + 0.1 * u[8,1]); 
    du[8,1] =  -D/(Δx)^2 *(( u[8, 1]  - u[8, 2]) + A *(u[8, 2] * (u[1, 2] - u[1, 1]) )) 
                - 0.015 * (0.018 * (gT  * mTSS[1] * u[6,1] * (u[1,1] - 22) + gHVA  * u[7,1] * (u[1,1] - 22)) + 0.1 * u[8,1]); 
    ## Cable ##
    # gating in each section

    
    # du[2, 1] = ((1 / (1 + exp((u[1, 1] - Vh) / kh))) - u[2, 1]) / ( 0.1 + (29624) / (4 * pi * (u[1, 1] + 74) ^ 2 + 2116)); #k
    # 
    # gating in each section
    for ii in 1:Int(numseg+1)::Int64 
        # du[3, ii] = (n_inf(u[1, ii], Vn, kn) - u[3, ii]) / n_tau(u[1,ii]); #n
        # du[4, ii] = (nA_inf(u[1, ii], VnA, knA) - u[4, ii]) / 5; #nA
        # du[5, ii] = (hA_inf(u[1, ii], VhA, khA) - u[5, ii]) / 10; #hA
        # du[6, ii] = (hT_inf(u[1, ii], VhT, khT)- u[6, ii]) / 15; #hT
        # du[7, ii] = (mHVA_inf(u[1, ii], VmHVA, kmHVA)  - u[7,ii]) / 4; #mHV
        du[2, ii] = 0.
        du[3, ii] = ((1 / (1 + exp(-(u[1, ii] - Vn) / kn))) - u[3, ii]) / (6 / (1 + exp((u[1, ii] + 23) / 15))); #n
        du[4, ii] = ((1 / (1 + exp((u[1, ii] - VnA) / -knA))) - u[4, ii]) / 5; #nA
        du[5, ii] = ((1 / (1 + exp((u[1, ii] - VhA) / khA))) - u[5, ii]) / 10; #hA
        du[6, ii] = ((1 / (1 + exp((u[1, ii] - VhT) / khT))) - u[6, ii]) / 15; #hT
        du[7, ii] = ((1 / (1 + exp(-(u[1, ii] - VmHVA) / kmHVA))) - u[7,ii]) / 4; #mHV
        
    end
    du[2, 1] = ((1 / (1 + exp((u[1, 1] - Vh) / kh))) - u[2, 1]) / ( 0.1 + (2 * 322 * 46) / (4 * pi * (u[1, 1] + 74) ^ 2 + 46 ^ 2)); #k
    du[2, 2] = (hinf(u[1, 1]) - u[2, 2]) / tauh(u[1, 1]) #hcn
    # Ca diffusion along cable
    # du[8, 2] = -Ddx *(( 2 * u[8, 2] - u[8, 1] - u[8, 3]) + A *(u[8, 3] * (u[1, 3] - u[1, 2]) - u[8, 2] * (u[1, 2] - u[1, 1])))
                            # -0.015 .* (0.018  .* (gTd/γs .* mTSS[2] .* u[6, 2] .* (u[1, 2] .- 22) + gHVAd/γs  .* u[7, 2] .* (u[1, 2] .- 22)) .+ 0.1 .* u[8, 2]) ; 
    du[8, 2] = -D/(Δx)^2 *(( 2 * u[8, 2] - u[8, 1] - u[8, 3]) + A *(u[8, 3] * (u[1, 3] - u[1, 2]) - u[8, 2] * (u[1, 2] - u[1, 1])))
                            -0.015 .* (0.018  .* (gTd .* mTSS[2] .* u[6, 2] .* (u[1, 2] .- 22) + gHVAd  .* u[7, 2] .* (u[1, 2] .- 22)) .+ 0.1 .* u[8, 2]) ; 
                                                   
        
    for ii in 3:Int(numseg)::Int64              
        # du[8, ii] =  -Ddx * (( 2 * u[8, ii] - u[8, ii-1] - u[8, ii+1]) + A * (u[8, ii+1] * (u[1, ii+1] - u[1, ii]) - u[8, ii] * (u[1, ii] - u[1, ii-1])))
        #                     - 0.015 * (0.018  * (gTd/γs  * mTSS[ii] * u[6, ii] * (u[1, ii] - 22)
        #                     + gHVAd/γs * u[7, ii] * (u[1, ii] - 22)) + 0.1 * u[8, ii]); 
        du[8, ii] =  -D/(Δx)^2 * (( 2 * u[8, ii] - u[8, ii-1] - u[8, ii+1]) + A * (u[8, ii+1] * (u[1, ii+1] - u[1, ii]) - u[8, ii] * (u[1, ii] - u[1, ii-1])))
                            - 0.015 * (0.018  * (gTd  * mTSS[ii] * u[6, ii] * (u[1, ii] - 22)
                            + gHVAd * u[7, ii] * (u[1, ii] - 22)) + 0.1 * u[8, ii]); 

        # du[1,ii] = (λdx * (u[1,ii+1] - 2 *u[1,ii] + u[1,ii-1])
        #     - gleakd * (u[1,ii] - Eleakd)
        #     - gKd * (u[3,ii]^4) * (u[1,ii]  - EK)
        #     - gAd * (u[4,ii]) * (u[5,ii]) * (u[1,ii]  - EK)
        #     - gTd * (mTSS[ii]) * (u[6,ii]) * (u[1,ii] - ECa)
        #     - gSKd * (u[8,ii]^5 / ((0.45^5) + (u[8,ii]^5))) * (u[1,ii]  - EK)
        #     - gHVAd * (u[7,ii]) * (u[1,ii] - ECa)
        #     ) / τm

        
        end
    # du[8, end] = -Ddx *(( u[8, end] - u[8, end-1]) + A *( - u[8, end] * (u[1, end] - u[1, end - 1])))  
    #         - 0.015 .* (0.018 .* (gTd/γs  .* mTSS[end] .* u[6, end] .* (u[1, end] .- 22)
    #         .+  gHVAd/γs  .* u[7, end] .* (u[1, end] .- 22)) .+ 0.1 .* u[8, end]); 
    du[8, end] = -D/(Δx)^2 *(( u[8, end] - u[8, end-1]) + A *( - u[8, end] * (u[1, end] - u[1, end - 1])))  
            - 0.015 .* (0.018 .* (gTd  .* mTSS[end] .* u[6, end] .* (u[1, end] .- 22)
            .+  gHVAd  .* u[7, end] .* (u[1, end] .- 22)) .+ 0.1 .* u[8, end]); 

    # Formulism from Goldberg et al. 2007 https://doi.org/10.1152/jn.00810.2006
    # active cable (V)     
    # du[1,2] = (λdx * (u[1,3]- 2*u[1,2] + u[1,1])
    #     - gleakd * (u[1,2] - Eleakd)
    #     - gKd * (u[3,2]^4) * (u[1,2]  - EK)
    #     - gAd * u[4,2] * u[5,2] * (u[1,2]  - EK)
    #     - gTd * mTSS[2] * u[6,2] * (u[1,2]-ECa)
    #     - gSKd * (u[8,2]^5/((0.45^5)+(u[8,2]^5))) * (u[1,2]  - EK)
    #     - gHVAd * u[7,2] * (u[1,2] - ECa)
    #     ) / τm
    du[1,2] = ((λ^2) / (Δx^2) * (u[1,3]- 2*u[1,2] + u[1,1])
        - gleakd * γs * (u[1,2] - Eleakd)
        - gKd * γs * (u[3,2]^4) * (u[1,2]  - EK)
        - gAd * γs * u[4,2] * u[5,2] * (u[1,2]  - EK)
        - gTd * γs * mTSS[2] * u[6,2] * (u[1,2]-ECa)
        - gSKd * γs * (u[8,2]^5/((0.45^5)+(u[8,2]^5))) * (u[1,2]  - EK)
        - gHVAd * γs * u[7,2] * (u[1,2] - ECa)
        ) / τm
    for ii in 3:Int(numseg)::Int64
        du[1,ii] = ((λ^2) / (Δx^2)  * (u[1,ii+1] - 2 *u[1,ii] + u[1,ii-1])
            - gleakd * γs * (u[1,ii] - Eleakd)
            - gKd * γs * (u[3,ii]^4) * (u[1,ii]  - EK)
            - gAd * γs * (u[4,ii]) * (u[5,ii]) * (u[1,ii]  - EK)
            - gTd * γs * (mTSS[ii]) * (u[6,ii]) * (u[1,ii] - ECa)
            - gSKd * γs * (u[8,ii]^5 / ((0.45^5) + (u[8,ii]^5))) * (u[1,ii]  - EK)
            - gHVAd * γs * (u[7,ii]) * (u[1,ii] - ECa)
            ) / τm
    end

    # du[1,end] = (λdx * (-u[1,end] + u[1,end-1])
    #         - gleakd *(u[1,end] - Eleakd)
    #         - gKd * (u[3,end]^4) * (u[1,end]  - EK)
    #         - gAd * (u[4,end]) * (u[5,end]) * (u[1,end]  - EK)
    #         - gTd * (mTSS[end]) * (u[6,end]) * (u[1,end] - ECa)
    #         - gSKd * (u[8,end]^5/((0.45^5) + (u[8,end]^5))) * (u[1,end]  - EK)
    #         - gHVAd * (u[7,end]) * (u[1,end] - ECa)
    #         ) / τm
    du[1,end] = ((λ^2) / (Δx^2) * (-u[1,end] + u[1,end-1])
            - gleakd * γs *(u[1,end] - Eleakd)
            - gKd * γs * (u[3,end]^4) * (u[1,end]  - EK)
            - gAd * γs * (u[4,end]) * (u[5,end]) * (u[1,end]  - EK)
            - gTd * γs * (mTSS[end]) * (u[6,end]) * (u[1,end] - ECa)
            - gSKd * γs * (u[8,end]^5/((0.45^5) + (u[8,end]^5))) * (u[1,end]  - EK)
            - gHVAd * γs * (u[7,end]) * (u[1,end] - ECa)
            ) / τm
    end
end
function CSCcable_gating_prc!(dr, r, p, t)
    gNa::Float64, gK::Float64, gleak::Float64, gA::Float64, gT::Float64, gHVA::Float64, gSK::Float64, gh::Float64, gKd::Float64, gleakd::Float64, gAd::Float64, gTd::Float64, gHVAd::Float64, gSKd::Float64,ghd::Float64, soma_r::Float64, dend_r::Float64, I::Float64, L::Float64, Eleak::Float64, Eleakd::Float64, numseg::Float64, Eh::Float64, Vm::Float64, Vh::Float64, Vn::Float64, VnA::Float64, VhA::Float64, VmT::Float64, VhT::Float64, VmHVA::Float64, km::Float64,kmT::Float64, kh::Float64, kn::Float64, knA::Float64, khA::Float64,khT::Float64, kmHVA::Float64,  EK::Float64, ENa::Float64, ECa::Float64,stim_ind = p
    # gNa::Float64, gK::Float64, gleak::Float64, gA::Float64, gT::Float64, gHVA::Float64, gSK::Float64, gh::Float64, gKd::Float64, gleakd::Float64, gAd::Float64, gTd::Float64, gHVAd::Float64, gSKd::Float64,ghd::Float64, As::Float64, a::Float64, I::Float64, Δx::Float64, Eleak::Float64, Eleakd::Float64, numseg::Float64, Eh::Float64, Vm::Float64, Vh::Float64, Vn::Float64, VnA::Float64, VhA::Float64, VmT::Float64, VhT::Float64, VmHVA::Float64, km::Float64,kmT::Float64, kh::Float64, kn::Float64, knA::Float64, khA::Float64,khT::Float64, kmHVA::Float64,  EK::Float64, ENa::Float64, ECa::Float64,γs::Float64, τm::Float64, κ::Float64, Ddx::Float64, λdx::Float64, gapseg::Tuple{Int64, Int64}, gc::Float64 = p
    
    # r = check_gating_bounds_1(r, numseg)
    check_gating_bounds_1!(r, numseg)
    u = @view r[:, :]
    du = @view dr[:, :]

    Δx = L / numseg # cm
    a = dend_r/10000 #  μm => cm
    As = soma_r /10000
    rmem = 1/(gleakd/1000) # Ω/cm^2 (divide by 1000 to get S-> Ω instead of mS-> kΩ)  -
    τm = rmem * capm /1000  # μF/cm^2 * Ω*cm^2 = uF*Ω = μsec /1000 to get ms
    λ = sqrt(a*rmem/(2*rl))
    κ = 1. /(rl/1000) * (a/(2*As))^2
    γs = 1/(gleak)

    # steady state gating variables
    mSS =  @inbounds @views 1 / (1 + exp((u[1,1] - Vm) / -km));
    mTSS = @inbounds @views @. 1 / (1 + exp(-(u[1, :] - VmT) / kmT));
    

    @inbounds @views  begin
    du[1,1] =  (
                - gK  * (u[3,1]^4) * (u[1,1] - EK)
                - gNa  * (mSS^3) * u[2,1] * (u[1,1]-ENa)
                - gleak   * (u[1,1] - Eleak)
                - gA * u[4,1] * u[5,1] * (u[1,1] - EK)
                - gT  * mTSS[1] * u[6,1] * (u[1,1] - ECa)
                - gSK  * (u[8,1]^5 / ((0.45^5) + (u[8,1]^5))) * (u[1,1] - EK) #1.0
                - gHVA  * u[7,1] * (u[1,1] - ECa)
                - u[2,2]*gh*(u[1,1] - Eh)
                +  κ  * (u[1,2] - u[1,1]) / Δx # uA/cm^2
                ) / capm;

    # add Ca diffusion from cable
    du[8,1] =  -D/(Δx)^2 *(( u[8, 1]  - u[8, 2]) + A *(u[8, 2] * (u[1, 2] - u[1, 1]) )) 
                - 0.015 * (0.018 * (gT  * mTSS[1] * u[6,1] * (u[1,1] - 22) + gHVA  * u[7,1] * (u[1,1] - 22)) + 0.1 * u[8,1]); 
    ## Cable ##
    # gating in each section

    # gating in each section
    for ii in 1:Int(numseg+1)::Int64 
        du[2, ii] = 0.
        du[3, ii] = ((1 / (1 + exp(-(u[1, ii] - Vn) / kn))) - u[3, ii]) / (6 / (1 + exp((u[1, ii] + 23) / 15))); #n
        du[4, ii] = ((1 / (1 + exp((u[1, ii] - VnA) / -knA))) - u[4, ii]) / 5; #nA
        du[5, ii] = ((1 / (1 + exp((u[1, ii] - VhA) / khA))) - u[5, ii]) / 10; #hA
        du[6, ii] = ((1 / (1 + exp((u[1, ii] - VhT) / khT))) - u[6, ii]) / 15; #hT
        du[7, ii] = ((1 / (1 + exp(-(u[1, ii] - VmHVA) / kmHVA))) - u[7,ii]) / 4; #mHV
        
    end
    du[2, 1] = ((1 / (1 + exp((u[1, 1] - Vh) / kh))) - u[2, 1]) / ( 0.1 + (2 * 322 * 46) / (4 * pi * (u[1, 1] + 74) ^ 2 + 46 ^ 2)); #k
    du[2, 2] = (hinf(u[1, 1]) - u[2, 2]) / tauh(u[1, 1]) #hcn
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

    du[1,2] = ((λ^2) / (Δx^2) * (u[1,3]- 2*u[1,2] + u[1,1])
        - gleakd * γs * (u[1,2] - Eleakd)
        - gKd * γs * (u[3,2]^4) * (u[1,2]  - EK)
        - gAd * γs * u[4,2] * u[5,2] * (u[1,2]  - EK)
        - gTd * γs * mTSS[2] * u[6,2] * (u[1,2]-ECa)
        - gSKd * γs * (u[8,2]^5/((0.45^5)+(u[8,2]^5))) * (u[1,2]  - EK)
        - gHVAd * γs * u[7,2] * (u[1,2] - ECa)
        ) / τm
    for ii in 3:Int(numseg)::Int64
        du[1,ii] = ((λ^2) / (Δx^2)  * (u[1,ii+1] - 2 *u[1,ii] + u[1,ii-1])
            - gleakd * γs * (u[1,ii] - Eleakd)
            - gKd * γs * (u[3,ii]^4) * (u[1,ii]  - EK)
            - gAd * γs * (u[4,ii]) * (u[5,ii]) * (u[1,ii]  - EK)
            - gTd * γs * (mTSS[ii]) * (u[6,ii]) * (u[1,ii] - ECa)
            - gSKd * γs * (u[8,ii]^5 / ((0.45^5) + (u[8,ii]^5))) * (u[1,ii]  - EK)
            - gHVAd * γs * (u[7,ii]) * (u[1,ii] - ECa)
            ) / τm
    end

    du[1,end] = ((λ^2) / (Δx^2) * (-u[1,end] + u[1,end-1])
            - gleakd * γs *(u[1,end] - Eleakd)
            - gKd * γs * (u[3,end]^4) * (u[1,end]  - EK)
            - gAd * γs * (u[4,end]) * (u[5,end]) * (u[1,end]  - EK)
            - gTd * γs * (mTSS[end]) * (u[6,end]) * (u[1,end] - ECa)
            - gSKd * γs * (u[8,end]^5/((0.45^5) + (u[8,end]^5))) * (u[1,end]  - EK)
            - gHVAd * γs * (u[7,end]) * (u[1,end] - ECa)
            ) / τm
    end

    if Int(stim_ind) == 1 # if cell one is connected at soma
        du[1, Int(stim_ind)] = du[1, Int(stim_ind)] - (I)/capm 
    else
        du[1, Int(stim_ind)] = du[1,Int(stim_ind)] - (I)/ τm
    end
    
end


function init_phase_diff_2(phase_diff,  p_ic; tend=1500, samp_rate=0.1, abstol=1e-15, reltol=1e-15)
    p_ic = tuple(p_ic...)
    V0  = -60.
    numseg = 50
    ic_net_ic = steady_state_init_u0(V0, numseg,  p_ic[24:end])
    prob_ic = ODEProblem(CSCcable_gating!, ic_net_ic, (0, tend), p_ic, saveat=samp_rate, maxiters=1e25, abstol=abstol, reltol=reltol)
    sol_ic = solve(prob_ic, QNDF())
    xpks = argmaxima(sol_ic[1,1,:])
    (peaks, proms) =peakproms(xpks, sol_ic[1,1,:], strict=true, minprom=25, maxprom=nothing)
    spiket = sol_ic.t[peaks]
    ISI_2nd_last = sol_ic.t[peaks[end-1]] - sol_ic.t[peaks[end-2]]
    
    # ic= sol_ic[:,:, peaks[end]]
    phase_i_ind = findfirst(sol_ic.t .>= (spiket[end-1] + ISI_2nd_last * -0.5))
    ic_i = sol_ic[:,:, phase_i_ind]

    phase_ii_ind = findfirst(sol_ic.t .>= (spiket[end-1] + ISI_2nd_last * (phase_diff-0.5)))
    ic_ii = sol_ic[:,:, phase_ii_ind]
    ic = [ic_i; ic_ii]
    return ic
    nothing
    end

function step_I(step_start, step_end, I_mag; Imag_base=0., p_ind=1)
    current_step= PresetTimeCallback(step_start,integrator -> integrator.p[p_ind] = (Imag_base - I_mag))
    current_step_off = PresetTimeCallback(step_end, integrator -> integrator.p[p_ind] = Imag_base)
    cbs = CallbackSet(current_step, current_step_off)
    return cbs
    nothing
end
    
end