#!/usr/bin/env python3
import h5py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
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

    M0 = M0_nbody * rho_b #this makes M0 a physical density ρ, which is the same as defined in Eq. (8) of Hertzberg (2014)
    M1 = M1_nbody * rho_b / a #this makes M1 a velocity density ρv, which the same as π defined in Eq. (9) of Hertzberg (2014)
    M2 = M2_nbody * rho_b / a**2 #this makes MH_2 into the form ρv^2 + κ, which this the same as σ as defiend in Eq. (10) of Hertzberg (2014)

    def dc_in_finder(path, j, x):
        moments_filename = 'output_hierarchy_{0:04d}.txt'.format(0)
        moments_file = np.genfromtxt(path + moments_filename)
        a0 = moments_file[:,-1][0]

        initial_file = np.genfromtxt(path + 'output_initial.txt')
        q = initial_file[:,0]
        Psi = initial_file[:,1]

        nbody_file = np.genfromtxt(path + 'output_{0:04d}.txt'.format(j))
        x_in = nbody_file[:,-1]

        Nx = x_in.size
        L = np.max(x_in)
        k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
        dc_in = -spectral_calc(Psi, L, o=1, d=0) / a0

        f = interp1d(q, dc_in, kind='cubic', fill_value='extrapolate')
        dc_in = f(x)

        return dc_in

    dc_in = dc_in_finder(path, j, x)
    d1k, d2k, P_1l_a_tr, P_2l_a_tr = SPT_tr(dc_in, k, L, Lambda, kind, a)

    #now all long-wavelength moments
    rho = M0
    rho_l = smoothing(M0, k, Lambda, kind) #this is ρ_{l}
    rho_s = rho - rho_l
    pi_l = smoothing(M1, k, Lambda, kind) #this is π_{l}
    sigma_l = smoothing(M2, k, Lambda, kind) #this is σ_{l}
    dc = M0_nbody - 1 #this is the overdensity δ from the hierarchy
    dc_l = (rho_l / rho_b) - 1
    dc_s = dc - dc_l

    v_l = pi_l / rho_l
    dv_l = spectral_calc(v_l, L, o=1, d=0) #the derivative of v_{l}

    #now we calculate the kinetic part of the (smoothed) stress tensor in EFT (they call it κ_ij)
    #in 1D, κ_{l} = σ_{l} - ([π_{l}]^{2} / ρ_{l})
    # kappa_l = (sigma_l - (pi_l**2 / rho_l))
    kappa_l = sigma_l - rho_l*v_l**2


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

    # #finally, the gravitational part of the smoothed stress tensor
    Phi_l = -(rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi_l2 - grad_phi2_l)

    # field = spectral_calc(phi, 1, o=1, d=0) * rho
    # dx_Phi = smoothing(field, k, Lambda, kind)
    # Phi_l = spectral_calc(dx_Phi, 1, o=1, d=1)

    # A = rho_b * smoothing(dx_phi, k, Lambda, kind)
    # B = rho_b * smoothing(dx_phi*dc, k, Lambda, kind)
    # Phi_l = spectral_calc(A+B, L=1, o=1, d=1)

    Phi_l_true = smoothing(dx_phi * rho, k, Lambda, kind)
    Phi_l_cgpt = (dx_phi_l * rho_l) + smoothing(dx_phi_s * rho_s, k, Lambda, kind)
    Phi_l_bau = (dx_phi_l * rho_l) + spectral_calc(smoothing(dx_phi_s**2, k, Lambda, kind), L, o=1, d=0) / (3 * H0**2 * a**2 / rho_0)

    # Phi_l_true = spectral_calc(Phi_l_true, L, o=1, d=1)
    # Phi_l_cgpt = spectral_calc(Phi_l_bau, L, o=1, d=1)
    # Phi_l_bau = spectral_calc(Phi_l_bau, L, o=1, d=1)

    # Phi_l = spectral_calc(Phi_l_true, 1, o=1, d=1)

    # #here is the full stress tensor; this is the object to be fitted for the EFT paramters
    tau_l = (kappa_l + Phi_l)
    return a, x, d1k, dc_l, dv_l, kappa_l, Phi_l, tau_l, Phi_l_true, Phi_l_cgpt, Phi_l_bau, rho_l, sigma_l, -rho_l*v_l**2

Lambda_int = 3
Lambda = Lambda_int * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Nfiles = 23


def ctot2_from_tau(a, x, tau_l, Lambda_int):
    tau_l_k = np.fft.fft(tau_l) / x.size
    num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))
    denom = ((d1k * np.conj(d1k)) * (a**2))
    ntrunc = int(num.size-Lambda_int)
    num[Lambda_int+1:ntrunc] = 0
    denom[Lambda_int+1:ntrunc] = 0
    ctot2 = np.real(sum(num) / sum(denom)) / 27.755 * a**3
    return ctot2

# # j = 1
a_list, ctot2_kappa, ctot2_phi, ctot2, ctot2_sigma, ctot2_g = [], [], [], [], [], []
for j in tqdm(range(Nfiles)):
    a, x, d1k, dc_l, dv_l, kappa_l, Phi_l, tau_l, Phi_l_true, Phi_l_cgpt, Phi_l_bau, rho_l, sigma_l, g_l = EFT_solve(j, Lambda, path, kind)
    a_list.append(a)
    # ctot2_kappa.append(ctot2_from_tau(a, x, kappa_l, Lambda_int))
    ctot2_sigma.append(ctot2_from_tau(a, x, sigma_l, Lambda_int))
    ctot2_g.append(ctot2_from_tau(a, x, g_l, Lambda_int))

    ctot2_phi.append(ctot2_from_tau(a, x, Phi_l, Lambda_int))
    ctot2.append(ctot2_from_tau(a, x, tau_l, Lambda_int))

# ctot2_kappa = np.array(ctot2_kappa)
# ctot2_phi = np.array(ctot2_phi)

# df = pd.DataFrame(data=[a_list, ctot2_kappa, ctot2_phi])
# file = open("./{}/ctot2_decomp_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), "wb")
# pickle.dump(df, file)
# file.close()


# file = open("./{}/ctot2_decomp_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), "rb")
# a_list, ctot2_kappa, ctot2_phi = np.array(pickle.load(file))
# file.close()


plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
# fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))), fontsize=16)
# ax.plot(x, tau_l, c='b', lw=2, label=r'$\left<[\tau]_{\Lambda}\right>$')
# ax.plot(x, kappa_l, c='r', ls='dashed', lw=2, label=r'$\kappa_{l}$')
# ax.plot(x, Phi_l, c='k', ls='dashed', lw=2, label=r'$\Phi_{l}$')


# Phi_l = spectral_calc(Phi_l, 1, o=1, d=0) * 2400
# # # Phi_l_appr = spectral_calc(Phi_l, 1, o=1, d=0) #1000 / a**2

# # ax.plot(x, Phi_l_true, c='b', lw=2, label=r'Exact')
# # # ax.plot(x, Phi_l_appr, c='r', lw=2, ls='dashed', label=r'Approximate')

# ax.plot(x, Phi_l_cgpt, c='r', ls='dashdot', lw=2, label=r'Pietroni+')
# ax.plot(x, Phi_l_bau, c='k', ls='dashed', lw=2, label=r'Baumann+')
# ax.plot(x, Phi_l, c='brown', ls='dotted', lw=2, label=r'1D')

# # ax.set_xlabel(r'$x$', fontsize=16)
# # ax.set_ylabel(r'$\partial_{x}[\Phi_l]_{\Lambda}$', fontsize=16)

ax.set_title(rf'$\Lambda = {Lambda_int} \,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=18, y=1.01)
ax.set_xlabel(r'$a$', fontsize=20)
ax.set_ylabel('$c_{\mathrm{tot}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=20)
# ax.plot(a_list, ctot2_kappa, c='b', lw=1.5, ls='dashed',  label=r'from $\kappa_{l}$')
ax.plot(a_list, ctot2_sigma, c='b', lw=1.5, ls='dashed',  label=r'from $\sigma_{l}$')
ax.plot(a_list, ctot2_g, c='r', lw=1.5, ls='dashed', label=r'from $-\rho_{l}v_{l}^{2}$')
ax.plot(a_list, ctot2_phi, c='k', lw=1.5, ls='dashed', label=r'from $\Phi_{l}$')

ax.plot(a_list, np.array(ctot2_sigma) + np.array(ctot2_g) + np.array(ctot2_phi), c='k', lw=1.5, label=r'from $\tau_{l}$', marker='o')
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.legend(fontsize=14, bbox_to_anchor=(1,1))
ax.yaxis.set_ticks_position('both')



flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')
for j in range(Nfiles):
    if flags[j] == 1:
        sc_line = ax.axvline(a_list[j], c='teal', lw=1, zorder=1)
    else:
        pass

# plt.show()

plt.savefig(f'../plots/paper_plots_final/ctot2_decomp_{kind}_{Lambda_int}.png', bbox_inches='tight', dpi=300)
plt.close()

