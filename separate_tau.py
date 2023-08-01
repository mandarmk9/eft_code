#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from functions import spectral_calc, smoothing, EFT_sm_kern, read_density, SPT_tr, read_sim_data, read_hier
from tqdm import tqdm



def EFT_solve(j, Lambda, path, kind, folder_name=''):
    a, dx, M0_nbody, M1_nbody, M2_nbody, C0_nbody, C1_nbody, C2_nbody = read_hier(path, j, folder_name)
    x = np.arange(0, 1.0, dx)

    L = 1.0
    Nx = x.size
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    rho_0 = 27.755 #this is the comoving background density
    rho_b = rho_0 / (a**3) #this is the physical background density
    H0 = 100

    M0_bar = smoothing(M0_nbody, k, Lambda, kind)
    M0_k = np.fft.fft(M0_bar) / M0_bar.size
    P_nb_a = np.real(M0_k * np.conj(M0_k))

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
    p_l = (pi_l**2 / rho_l)
    kappa_l = (sigma_l - p_l)

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

    phi_s = phi - phi_l

    dx_phi = spectral_calc(phi, L, o=1, d=0)
    dx_phi_l = spectral_calc(phi_l, L, o=1, d=0)
    dx_phi_s = spectral_calc(phi_s, L, o=1, d=0)

    #finally, the gravitational part of the smoothed stress tensor
    Phi_l = -(rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi_l2 - grad_phi2_l)

    # #here is the full stress tensor; this is the object to be fitted for the EFT paramters
    tau_l = (kappa_l + Phi_l)
    return a, x, sigma_l*a**2, p_l*a**2, kappa_l*a**2, Phi_l*a**2, tau_l*a**2


path = 'cosmo_sim_1d/sim_k_1_11/run1/'
folder_name = 'shell_crossed_hier/'
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'


Lambda_int = 3
Lambda = Lambda_int * (2*np.pi)
# j = 15


for j in tqdm(range(50)):
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})

    fig, ax = plt.subplots()

    a, x, sigma_l, p_l, kappa_l, Phi_l, tau_l = EFT_solve(j, Lambda, path, kind)
    a, x, sigma_l_nsc, _, _, _, _ = EFT_solve(j, Lambda, path, kind, folder_name=folder_name)

    ax.set_title(rf'$a = {a}, \Lambda = {Lambda_int} \,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=18)
    ax.plot(x, tau_l, c='b', lw=1.5, ls='solid', label=r'$\tau$')
    ax.plot(x, sigma_l - sigma_l_nsc, c='seagreen', lw=1.5, ls='dashed', label=r'$\Xi^{\mathrm{m}}_{\ell}$')
    ax.plot(x, sigma_l_nsc, c='cyan', lw=1.5, ls='dashed', label=r'$\Xi^{\mathrm{s}}_{\ell}$')
    ax.plot(x, -p_l, c='r', lw=1.5, ls='dashed', label=r'$-\rho_{\ell}U^{2}$')
    ax.plot(x, kappa_l, c='orange', lw=1.5, ls='solid', label=r'$\tau_{\mathrm{k}}$')
    ax.plot(x, Phi_l, c='k', lw=1.5, ls='solid', label=r'$\tau_{\mathrm{g}}$')

    ax.set_xlabel(r'$x/L$', fontsize=16)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=18)
    ax.yaxis.set_ticks_position('both')
    
    ax.set_ylabel(r'$a^{2}\tau\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=16)
    plt.legend(fontsize=14, bbox_to_anchor=(1, 1.025))
    ax.set_ylim(-20, 27)
    # plt.show()
    # plt.tight_layout()
    plt.savefig(f'../plots/paper_plots_final/tau_decomp_L{Lambda_int}/tau_{j:02}.png', bbox_inches='tight', dpi=300)
    plt.close()