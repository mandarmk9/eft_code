#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from functions import smoothing, read_hier

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda_int = 3
Lambda = Lambda_int * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'



def calc_new_hier(path, j, Lambda, kind, folder_name):
    a, dx, M0_nbody, M1_nbody, M2_nbody, C0_nbody, C1_nbody, C2_nbody = read_hier(path, j, folder_name)
    x = np.arange(0, 1.0, dx)
    L = 1.0
    Nx = x.size
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    rho_0 = 27.755 #this is the comoving background density
    rho_b = rho_0 / (a**3) #this is the physical background density
    H0 = 100

    M0 = M0_nbody #* rho_b #this makes M0 a physical density ρ, which is the same as defined in Eq. (8) of Hertzberg (2014)
    M1 = M1_nbody #* rho_b / a #this makes M1 a velocity density ρv, which the same as π defined in Eq. (9) of Hertzberg (2014)
    M2 = M2_nbody #* rho_b / a**2 #this makes MH_2 into the form ρv^2 + κ, which this the same as σ as defiend in Eq. (10) of Hertzberg (2014)

    #now all long-wavelength moments
    M0_bar = smoothing(M0, k, Lambda, kind) #this is ρ_{l}
    M1_bar = smoothing(M1, k, Lambda, kind) #this is π_{l}
    M2_bar = smoothing(M2, k, Lambda, kind) #this is σ_{l}
    return a, x, k, M0, M1, M2, M0_bar, M1_bar, M2_bar

def calc_old_hier(path, j, Lambda, kind):
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x = moments_file[:,0]
    M0_nbody = moments_file[:,2]
    M1_nbody = moments_file[:,4]
    M2_nbody = moments_file[:,6]
    C2_nbody = moments_file[:,7]

    L = 1.0
    Nx = x.size
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    rho_0 = 27.755 #this is the comoving background density
    rho_b = rho_0 / (a**3) #this is the physical background density
    H0 = 100

    M0 = M0_nbody #* rho_b #this makes M0 a physical density ρ, which is the same as defined in Eq. (8) of Hertzberg (2014)
    M1 = M1_nbody #* rho_b / a #this makes M1 a velocity density ρv, which the same as π defined in Eq. (9) of Hertzberg (2014)
    M2 = M2_nbody #* rho_b / a**2 #this makes MH_2 into the form ρv^2 + κ, which this the same as σ as defiend in Eq. (10) of Hertzberg (2014)

    #now all long-wavelength moments
    M0_bar = smoothing(M0, k, Lambda, kind) #this is ρ_{l}
    M1_bar = smoothing(M1, k, Lambda, kind) #this is π_{l}
    M2_bar = smoothing(M2, k, Lambda, kind) #this is σ_{l}
    return a, x, k, M0, M1, M2, M0_bar, M1_bar, M2_bar


j = 0

a, x_new, k_new, M0_new, M1_new, M2_new, M0_bar_new, M1_bar_new, M2_bar_new = calc_new_hier(path, j, Lambda, kind, folder_name='hierarchy/')
a, x_old, k_old, M0_old, M1_old, M2_old, M0_bar_old, M1_bar_old, M2_bar_old = calc_old_hier(path, j, Lambda, kind)

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})

fig, ax = plt.subplots()

ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.yaxis.set_ticks_position('both')
ax.set_title(rf'$a = {a}, \Lambda = {Lambda_int} \,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=16)

C2_old = M2_bar_old - (M1_bar_old**2 / M0_bar_old)
C2_new = smoothing(M2_old - (M1_old**2 / M0_old), k_old, Lambda, kind)

ax.plot(x_old, C2_old, c='b', ls='dashed', lw=1.5, label=r'$M_{0}$; old')
ax.plot(x_old, C2_new, c='r', lw=1.5, label=r'$M_{0}$; new')

plt.legend(fontsize=14, bbox_to_anchor=(1, 1.025))
ax.set_xlabel(r'$x/L$', fontsize=14)
ax.set_ylabel(r'$\langle[\tau]_{\Lambda}\rangle\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=14)

plt.show()
# plt.tight_layout()
# plt.savefig(f'../plots/paper_plots_final/tau_decomp_L{Lambda_int}/tau_{j:02}.png', bbox_inches='tight', dpi=300)
# plt.close()



