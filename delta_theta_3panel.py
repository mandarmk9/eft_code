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
    rho_l = smoothing(M0, k, Lambda, kind) #this is ρ_{\ell}
    rho_s = rho - rho_l
    pi_l = smoothing(M1, k, Lambda, kind) #this is π_{\ell}
    sigma_l = smoothing(M2, k, Lambda, kind) #this is σ_{\ell}
    dc = M0_nbody - 1 #this is the overdensity δ from the hierarchy
    dc_l = (rho_l / rho_b) - 1
    dc_s = dc - dc_l

    #now we calculate the kinetic part of the (smoothed) stress tensor in EFT (they call it κ_ij)
    #in 1D, κ_{\ell} = σ_{\ell} - ([π_{\ell}]^{2} / ρ_{\ell})
    p_l = (pi_l**2 / rho_l)
    kappa_l = (sigma_l - p_l)

    v_l = pi_l / rho_l
    dv_l = spectral_calc(v_l, L, o=1, d=0) #the derivative of v_{\ell}

    #next, we build the gravitational part of the smoothed stress tensor (this is a consequence of the smoothing)
    rhs = (3 * H0**2 / (2 * a)) * dc #using the hierarchy δ here
    phi = spectral_calc(rhs, L, o=2, d=1)
    grad_phi = spectral_calc(phi, L, o=1, d=0) #this is the gradient of the unsmoothed potential ∇ϕ

    rhs_l = (3 * H0**2 / (2 * a)) * dc_l
    phi_l = spectral_calc(rhs_l, L, o=2, d=1)
    grad_phi_l = spectral_calc(phi_l, L, o=1, d=0) #this is the gradient of the smoothed potential ∇(ϕ_l)

    grad_phi_l2 = grad_phi_l**2 #this is [(∇(ϕ_{\ell})]**2
    grad_phi2_l = smoothing(grad_phi**2, k, Lambda, kind) #this is [(∇ϕ)^2]_l

    phi_s = phi - phi_l

    dx_phi = spectral_calc(phi, L, o=1, d=0)
    dx_phi_l = spectral_calc(phi_l, L, o=1, d=0)
    dx_phi_s = spectral_calc(phi_s, L, o=1, d=0)

    #finally, the gravitational part of the smoothed stress tensor
    Phi_l = -(rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi_l2 - grad_phi2_l)

    # #here is the full stress tensor; this is the object to be fitted for the EFT paramters
    tau_l = (kappa_l + Phi_l)

    H = a**(-1/2)*100
    dv_l = dv_l / (H)

    # corr = np.corrcoef(dc_l + dv_l, dc_l - dv_l, rowvar=False)[0, 1]
    corr = np.corrcoef(dc_l, dv_l, rowvar=False)[0, 1]
    # print(1-corr**2)
    print(1+corr)
    return a, x, sigma_l*a**2, p_l*a**2, kappa_l*a**2, Phi_l*a**2, tau_l*a**2, dc_l/a, dv_l/a, 1+corr #/(a**(-1/2)*100)


path = 'cosmo_sim_1d/sim_k_1_11/run1/'
folder_name = 'shell_crossed_hier/'
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'


Lambda_int = 3
Lambda = Lambda_int * (2*np.pi)
# j = 15


plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
props = dict(boxstyle='round', facecolor='white', alpha=0.5)

fig, ax = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1]})
fig.suptitle(rf'$\Lambda = {Lambda_int} \,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=22, y=0.975)

j = 11
a, x, sigma_l, p_l, kappa_l, Phi_l, tau_l, dc_l, dv_l, corr = EFT_solve(j, Lambda, path, kind)
# a, x, sigma_l_nsc, _, _, _, _, _, _, _ = EFT_solve(j, Lambda, path, kind, folder_name=folder_name)

ax[0].set_title(rf'$a = {a}$', fontsize=20, x=0.15, y=0.9)
# ax[0].plot(x, tau_l, c='b', lw=1.5, ls='solid', label=r'$\tau$')
# ax[0].plot(x, sigma_l - sigma_l_nsc, c='seagreen', lw=1.5, ls='dashed', label=r'$\Xi^{\mathrm{m}}_{\ell}$')
# ax[0].plot(x, sigma_l_nsc, c='cyan', lw=1.5, ls='dashed', label=r'$\Xi^{\mathrm{s}}_{\ell}$')
# ax[0].plot(x, -p_l, c='r', lw=1.5, ls='dashed', label=r'$-\rho_{\ell}U^{2}$')
# ax[0].plot(x, kappa_l, c='orange', lw=1.5, ls='solid', label=r'$\tau_{\mathrm{k}}$')
# ax[0].plot(x, Phi_l, c='k', lw=1.5, ls='solid', label=r'$\tau_{\mathrm{g}}$')
ax[0].plot(x, dc_l+dv_l, c='k', lw=1.5, label=r'$\delta_{\ell}$')
ax[0].plot(x, dc_l-dv_l, c='b', ls='dashed', lw=1.5, label=r'$\theta_{\ell}$')

# ax[0].plot(x, dc_l, c='k', lw=1.5, label=r'$\delta_{\ell}$')
# ax[0].plot(x, 1*dc_l+0.99*dv_l, c='b', ls='dashed', lw=1.5, label=r'$\theta_{\ell}$')

corr_str = f'{corr:.1e}'
corr_str = rf'{corr_str[0:3]} \times 10^{{-{corr_str[-1]}}}'
ax[0].text(0.25, 0.075, rf'$1+r = {corr_str}$', transform=ax[0].transAxes, fontsize=18, bbox=props)
# ax[0].plot(x, 193*dc_l+1.3*dv_l, c='k', lw=1.5, label=r'$\delta_{\ell}$')

j = 23
a, x, sigma_l, p_l, kappa_l, Phi_l, tau_l, dc_l, dv_l, corr  = EFT_solve(j, Lambda, path, kind)
# a, x, sigma_l_nsc, _, _, _, _, _, _, _ = EFT_solve(j, Lambda, path, kind, folder_name=folder_name)

ax[1].set_title(rf'$a = {a}$', fontsize=20, x=0.15, y=0.9)
# ax[1].plot(x, tau_l, c='b', lw=1.5, ls='solid', label=r'$\tau$')
# ax[1].plot(x, sigma_l - sigma_l_nsc, c='seagreen', lw=1.5, ls='dashed', label=r'$\Xi^{\mathrm{m}}_{\ell}$')
# ax[1].plot(x, sigma_l_nsc, c='cyan', lw=1.5, ls='dashed', label=r'$\Xi^{\mathrm{s}}_{\ell}$')
# ax[1].plot(x, -p_l, c='r', lw=1.5, ls='dashed', label=r'$-\rho_{\ell}U^{2}$')
# ax[1].plot(x, kappa_l, c='orange', lw=1.5, ls='solid', label=r'$\tau_{\mathrm{k}}$')
# ax[1].plot(x, Phi_l, c='k', lw=1.5, ls='solid', label=r'$\tau_{\mathrm{g}}$')
ax[1].plot(x, dc_l+dv_l, c='k', lw=1.5, label=r'$\delta_{\ell}$')
ax[1].plot(x, dc_l-dv_l, c='b', ls='dashed', lw=1.5, label=r'$\theta_{\ell}$')
corr_str = f'{corr:.1e}'
corr_str = rf'{corr_str[0:3]} \times 10^{{-{corr_str[-1]}}}'
ax[1].text(0.25, 0.075, rf'$1+r = {corr_str}$', transform=ax[1].transAxes, fontsize=18, bbox=props)

j = 50
a, x, sigma_l, p_l, kappa_l, Phi_l, tau_l, dc_l, dv_l, corr  = EFT_solve(j, Lambda, path, kind)
# a, x, sigma_l_nsc, _, _, _, _, _, _, _ = EFT_solve(j, Lambda, path, kind, folder_name=folder_name)

ax[2].set_title(rf'$a = {a}$', fontsize=20, x=0.15, y=0.9)
# ax[2].plot(x, tau_l, c='b', lw=1.5, ls='solid', label=r'$\tau$')
# ax[2].plot(x, sigma_l - sigma_l_nsc, c='seagreen', lw=1.5, ls='dashed', label=r'$\Xi^{\mathrm{m}}_{\ell}$')
# ax[2].plot(x, sigma_l_nsc, c='cyan', lw=1.5, ls='dashed', label=r'$\Xi^{\mathrm{s}}_{\ell}$')
# ax[2].plot(x, -p_l, c='r', lw=1.5, ls='dashed', label=r'$-\rho_{\ell}U^{2}$')
# ax[2].plot(x, kappa_l, c='orange', lw=1.5, ls='solid', label=r'$\tau_{\mathrm{k}}$')
# ax[2].plot(x, Phi_l, c='k', lw=1.5, ls='solid', label=r'$\tau_{\mathrm{g}}$')

ax[2].plot(x, dc_l+dv_l, c='k', lw=1.5, label=r'$\delta_{\ell}+\theta_{\ell}$')
ax[2].plot(x, dc_l-dv_l, c='b', ls='dashed', lw=1.5, label=r'$\delta_{\ell}-\theta_{\ell}$')
# ax[2].plot(x, 0.25*dc_l+193*dv_l, c='k', lw=1.5, label=r'$\delta_{\ell}$')
corr_str = f'{corr:.1e}'
corr_str = rf'{corr_str[0:3]} \times 10^{{-{corr_str[-1]}}}'
ax[2].text(0.25, 0.075, rf'$1+r = {corr_str}$', transform=ax[2].transAxes, fontsize=18, bbox=props)

for j in range(3):
    ax[j].set_xlabel(r'$x/L$', fontsize=20)
    ax[j].minorticks_on()
    ax[j].tick_params(axis='both', which='both', direction='in', labelsize=18)
    ax[j].yaxis.set_ticks_position('both')
    # ax[j].set_ylim(-20, 30)
   
# ax[0].set_ylabel(r'$a^{2}\tau\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=20)
# ax[2].set_ylabel(r'$a^{2}\tau\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=20)

ax[0].set_ylabel(r'$a^{-1}\left[\delta_{\ell} \pm \theta_{\ell}\right]$', fontsize=22)
# ax[2].set_ylabel(r'$a^{-1}\left[\delta_{\ell} \pm \theta_{\ell}\right]$', fontsize=22)

# ax[2].yaxis.set_label_position('right')
# ax[2].yaxis.set_tick_params(labelright=True)
plt.legend(fontsize=18, ncol=3, bbox_to_anchor=(0.975, 1.15))
plt.subplots_adjust(wspace=0.05)
fig.align_labels()

# plt.show()
# # # plt.tight_layout()
# # # # # plt.savefig(f'../plots/paper_plots_final/tau_decomp_L{Lambda_int}/tau_{j:02}.png', bbox_inches='tight', dpi=300)
# # # # plt.savefig(f'../plots/paper_plots_final/tau_decomp_{kind}.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'../plots/paper_plots_final/delta_theta_{kind}.pdf', bbox_inches='tight', dpi=300)
plt.close()