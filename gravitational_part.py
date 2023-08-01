#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

from functions import spectral_calc, smoothing, read_density, SPT_tr, read_hier

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

j = 0
path = 'cosmo_sim_1d/sim_k_1_11/run1/'
kind = 'sharp'
kind_txt = 'sharp cutoff'
Lambda_int = 3
Lambda = Lambda_int * (2 * np.pi)

a, dx, M0_nbody, M1_nbody, M2_nbody, C0_nbody, C1_nbody, C2_nbody = read_hier(path, j, folder_name='')
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

#next, we build the gravitational part of the smoothed stress tensor (this is a consequence of the smoothing)
rhs = (3 * H0**2 / (2 * a)) * dc #using the hierarchy δ here
phi = spectral_calc(rhs, L, o=2, d=1)
grad_phi = spectral_calc(phi, L, o=1, d=0) #this is the gradient of the unsmoothed potential ∇ϕ

rhs_l = (3 * H0**2 / (2 * a)) * dc_l
phi_l = spectral_calc(rhs_l, L, o=2, d=1)
grad_phi_l = spectral_calc(phi_l, L, o=1, d=0) #this is the gradient of the smoothed potential ∇(ϕ_l)

grad_phi_l2 = grad_phi_l**2 #this is [(∇(ϕ_{l})]**2
grad_phi2_l = smoothing(grad_phi**2, k, Lambda, kind) #this is [(∇ϕ)^2]_l

dx_phi = spectral_calc(phi, L, o=1, d=0)
phi_s = phi - phi_l
dx_phi_s = spectral_calc(phi_s, L, o=1, d=0)


#finally, the gravitational part of the smoothed stress tensor
Phi_l = -(rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi_l2 - grad_phi2_l)

# Phi_l_true = smoothing(dx_phi * rho, k, Lambda, kind) - (rho_l * grad_phi_l)
Phi_l_true = smoothing(dx_phi_s * rho_s, k, Lambda, kind)


# Phi_l_cgpt = (dx_phi_l * rho_l) + smoothing(dx_phi_s * rho_s, k, Lambda, kind)
# Phi_l_bau = (dx_phi_l * rho_l) + spectral_calc(smoothing(dx_phi_s**2, k, Lambda, kind), L, o=1, d=0) / (3 * H0**2 * a**2 / rho_0)


plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})

fig, ax = plt.subplots()

ax.set_title(r'$a = {}, \Lambda = {}\,k_{{\mathrm{{f}}}}$ ({})'.format(a, Lambda_int, kind_txt), fontsize=20)

Phi_l = spectral_calc(Phi_l, L, o=1, d=0)

ax.set_ylabel(r'$\Phi_{l}\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=18)
ax.set_xlabel(r'$x/L$', fontsize=18)
ax.plot(x, Phi_l, c='b', lw=1.5, label=r'approximate')
ax.plot(x, Phi_l_true, ls='dashed', c='r', lw=1.5, label=r'exact')

ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=18)
ax.yaxis.set_ticks_position('both')

plt.legend(fontsize=18)#, bbox_to_anchor=(1, 1.325))
plt.show()
# plt.savefig('../plots/paper_plots_final/tau_fits_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# plt.close()
