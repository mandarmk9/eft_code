#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from functions import spectral_calc, read_hier, smoothing, read_sim_data
from scipy.optimize import curve_fit

def EFT_solve(j, Lambda, path, kind, folder_name=''):
    a, dx, M0_nbody, M1_nbody, M2_nbody, C0_nbody, C1_nbody, C2_nbody = read_hier(path, j, folder_name)
    x = np.arange(0, 1.0, dx)

    # moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    # moments_file = np.genfromtxt(path + moments_filename)
    # a = moments_file[:,-1][0]
    # x = moments_file[:,0]
    # M0_nbody = moments_file[:,2]
    # M1_nbody = moments_file[:,4]
    # M2_nbody = moments_file[:,6]


    L = 1.0
    Nx = x.size
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    rho_0 = 27.755 #this is the comoving background density
    rho_b = rho_0 / (a**3) #this is the physical background density
    H0 = 100
    H = H0 * (a**(-3/2))

    M0 = M0_nbody * rho_b #this makes M0 a physical density ρ, which is the same as defined in Eq. (8) of Hertzberg (2014)
    M1 = M1_nbody * rho_b / a #this makes M1 a velocity density ρv, which the same as π defined in Eq. (9) of Hertzberg (2014)
    M2 = M2_nbody * rho_b / a**2 #this makes MH_2 into the form ρv^2 + κ, which this the same as σ as defiend in Eq. (10) of Hertzberg (2014)

    #now all long-wavelength moments
    rho = M0
    rho_l = smoothing(M0, k, Lambda, kind) #this is ρ_{l}
    rho_s = rho - rho_l
    pi_l = smoothing(M1, k, Lambda, kind) #this is π_{l}
    sigma_l = smoothing(M2, k, Lambda, kind) #this is σ_{l}
    dc = M0_nbody - 1 #this is the overdensity δ from the hierarchy
    dc_l = (rho_l / rho_b) - 1
    dc_s = dc - dc_l

    #now we calculate the kinetic part of the (smoothed) stress tensor in EFT (they call it κ_ij)
    #in 1D, κ_{l} = σ_{l} - ([π_{l}]^{2} / ρ_{l})
    kappa_l = (sigma_l - (pi_l**2 / rho_l))
    v_l = pi_l / rho_l
    dv_l = spectral_calc(v_l, L, o=1, d=0) #the derivative of v_{l}

    #next, we build the gravitational part of the smoothed stress tensor (this is a consequence of the smoothing)
    rhs = (3 * H0**2 / (2 * a)) * dc #using the hierarchy δ here
    phi = spectral_calc(rhs, L, o=2, d=1)
    grad_phi = spectral_calc(phi, L, o=1, d=0) #this is the gradient of the unsmoothed potential ∇ϕ

    rhs_l = (3 * H0**2 / (2 * a)) * dc_l
    phi_l = spectral_calc(rhs_l, L, o=2, d=1)
    grad_phi_l = spectral_calc(phi_l, L, o=1, d=0) #this is the gradient of the smoothed potential ∇(ϕ_l)

    grad_phi_l2 = grad_phi_l**2 #this is [(∇(ϕ_{l})]**2
    grad_phi2_l = smoothing(grad_phi**2, k, Lambda, kind) #this is [(∇ϕ)^2]_l

    #finally, the gravitational part of the smoothed stress tensor
    Phi_l = (rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi_l2 - grad_phi2_l)

    #here is the full stress tensor; this is the object to be fitted for the EFT paramters
    tau_l = (kappa_l - Phi_l)
    dtau_l = spectral_calc(tau_l, L, o=1, d=0) #this is the gradient of tau_l

    rhs_cont_int = (1+dc_l)*v_l
    con_rhs = spectral_calc(rhs_cont_int, 1, o=1, d=0) / a

    return a, x, dc_l, con_rhs, dtau_l, v_l, dv_l, H, grad_phi_l, rho_l


path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda_int = 6
Lambda = Lambda_int * (2 * np.pi)
mode = 1
kind = 'sharp'
kind_txt = 'sharp cutoff'

Nfiles = 50
n_runs = 8
n_use = 10
A = []
per = 46.6
plots_folder = '/sim_1_11/'
folder_name = 'hierarchy'

# j= 10
a_list, ratio = [], []


for j in range(1):
    j = 0
    a0, x, dc0, con_rhs, dtau, v0, dv, H, dphi, rho_l = EFT_solve(j, Lambda, path, kind, folder_name)
    a1, _, dc1, _, _, v1, _, _, _, _ = EFT_solve(j+1, Lambda, path, kind, folder_name)


    a_dot = H * a0
    da_v = (v1 - v0) / (a1 - a0)
    dt_v = a_dot * da_v
    LHS = dt_v + (H*v0) + (v0*(dv) / a0) + (dphi / a0) # + (dphi / 1000)# / a0)
    RHS = -dtau / (a0 * rho_l)

    rat = LHS / RHS
    # da_dc = (dc1 - dc0) / (a1-a0)
    # dt_dc = -a_dot * da_dc
    #
    # # print(a0, LHS.max() / RHS.max())
    # a_list.append(a0)
    # ratio.append(LHS.max() / RHS.max())


a_list = np.array(a_list)

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
# ax.plot(a_list, ratio, c='b', lw=1.5)
# ax.plot(a_list, a_list**(-2), c='k', lw=1.5, ls='dashed')

ax.set_title(rf'$a = {np.round(a0, 3)}$', fontsize=14)
ax.set_xlabel(r'$x/L$', fontsize=12)
# ax.set_ylabel()
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.yaxis.set_ticks_position('both')
# ax.plot(x, LHS, lw=1.5, ls='dashed', c='k', label='LHS')
# ax.plot(x, RHS, lw=1.5, c='b', label='RHS')
ax.plot(x, rat, lw=1.5, c='b', label='RHS')

# ax.plot(x, dt_dc, lw=1.5, ls='dashed', c='k', label='LHS')
# ax.plot(x, con_rhs, lw=1.5, c='b', label='RHS')


ax.minorticks_on()
# plt.savefig(f'../plots/{plots_folder}/euler.png', bbox_inches='tight', dpi=150)
# plt.close()
plt.show()
