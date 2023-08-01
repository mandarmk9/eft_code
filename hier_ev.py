#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from functions import write_hier, read_hier, smoothing, spectral_calc

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# j = 0
Lambda = 3 * (2*np.pi)
kind = 'sharp'

j0, j1 = 0, 1
def calc_new_hier(path, j, Lambda, kind, folder_name):
    a, dx, M0_nbody, M1_nbody, M2_nbody, C0_nbody, C1_nbody, C2_nbody = read_hier(path, j, folder_name)
    x = np.arange(0, 1.0, dx)
    L = 1.0
    Nx = x.size
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    rho_0 = 27.755 #this is the comoving background density
    rho_b = rho_0 / (a**3) #this is the physical background density
    H0 = 100

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

    phi_s = phi - phi_l

    dx_phi = spectral_calc(phi, L, o=1, d=0)
    dx_phi_l = spectral_calc(phi_l, L, o=1, d=0)
    dx_phi_s = spectral_calc(phi_s, L, o=1, d=0)

    #finally, the gravitational part of the smoothed stress tensor
    Phi_l = (rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi_l2 - grad_phi2_l)

    # #here is the full stress tensor; this is the object to be fitted for the EFT paramters
    tau_l = (kappa_l + Phi_l)
    
    # Phi_l = smoothing(dx_phi_s * (rho - rho_l), k, Lambda, kind)

    return a, x, k, sigma_l, pi_l, kappa_l, (grad_phi_l2 - grad_phi2_l), Phi_l, tau_l, dc_l, dv_l, C1_nbody, M1_nbody, M0_nbody, M2_nbody, C2_nbody

a0, x, k, sigma_l_0, pi_l, kappa_l, phi_l_0, Phi_l_0, tau_l_0, dc_l_0, dv_l_0, C1_nbody_0, M1_nbody, M0_nbody, M2_nbody, C2_nbody = calc_new_hier(path, j0, Lambda, kind, folder_name='/hierarchy/')
a1, x, k, sigma_l_1, pi_l, kappa_l, phi_l_1, Phi_l_1, tau_l_1, dc_l_1, dv_l_1, C1_nbody_1, M1_nbody, M0_nbody, M2_nbody, C2_nbody = calc_new_hier(path, j1, Lambda, kind, folder_name='/hierarchy/')


plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
# ax.set_title(r'$a = {}$'.format(a), fontsize=20)
ax.set_xlabel(r'$x/L$', fontsize=18)
# ax.set_ylabel(r'$[\tau]_{\Lambda}$', fontsize=18)

# ax.plot(x, dc_l_0/a0, c='b', label=rf'a = {a0}')
# ax.plot(x, dc_l_1/a1, c='k', ls='dashed', label=rf'a = {a1}')


# ax.plot(x, dv_l_0/np.sqrt(a0), c='b', label=rf'a = {a0}')
# ax.plot(x, dv_l_1/np.sqrt(a1), c='k', ls='dashed', label=rf'a = {a1}')

# C1_bar_0 = smoothing(C1_nbody_0, k, Lambda, kind)
# C1_bar_1 = smoothing(C1_nbody_1, k, Lambda, kind)

# ax.plot(x, C1_bar_0 / a0**(3/2), c='b', label=rf'a = {a0}')
# ax.plot(x, C1_bar_1 / a1**(3/2), c='k', ls='dashed', label=rf'a = {a1}')

# ax.plot(x, sigma_l_0 * a0**(2), c='b', label=rf'a = {a0}')
# ax.plot(x, sigma_l_1 * a1**(2), c='k', ls='dashed', label=rf'a = {a1}')

ax.plot(x, Phi_l_0 * a0**2, c='b', label=rf'a = {a0}')
ax.plot(x, Phi_l_1 * a1**2, c='k', ls='dashed', label=rf'a = {a1}')


# ax.plot(x, phi_l_0 , c='b', label=rf'a = {a0}')
# ax.plot(x, phi_l_1 , c='k', ls='dashed', label=rf'a = {a1}')


# ax.plot(x, tau_l_0 * a0**2, c='b', label=rf'a = {a0}')
# ax.plot(x, tau_l_1 * a1**2, c='k', ls='dashed', label=rf'a = {a1}')



ax.legend(fontsize=12, bbox_to_anchor=(1,1))
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.ticklabel_format(scilimits=(-2, 3))
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('both')
ax.minorticks_on()

plt.show()
# plt.savefig('../plots/test/new_paper_plots/tau_l_{}.png'.format(j), bbox_inches='tight', dpi=150)
# plt.close()
