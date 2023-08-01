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

    P_lin_a = np.real(d1k * np.conj(d1k)) * (a**2)

    # dk_par, a, dx = read_density(path, j)
    #
    # x_grid = np.arange(0, 1.0, dx)
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
    Phi_l = -(rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi_l2 - grad_phi2_l)

    # #here is the full stress tensor; this is the object to be fitted for the EFT paramters
    tau_l = (kappa_l + Phi_l)
    kappa_l = (smoothing(M2, k, Lambda, kind) - (smoothing(M1, k, Lambda, kind)**2 / smoothing(M0, k, Lambda, kind)))
    return a, x, d1k, kappa_l, Phi_l, sigma_l #tau_l/a**5


def ctot2_from_tau(a, x, d1k, tau_l, Lambda_int):
    tau_l_k = np.fft.fft(tau_l) / x.size
    num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))
    denom = ((d1k * np.conj(d1k)) * (a**2))
    ntrunc = int(num.size-Lambda_int)
    num[Lambda_int+1:ntrunc] = 0
    denom[Lambda_int+1:ntrunc] = 0
    ctot2 = np.real(sum(num) / sum(denom)) / 27.755 * a**3
    return ctot2


folder_name = 'shell_crossed_hier/'
num = 11
# j = 20
kind = 'sharp'
kind_txt = 'sharp cutoff'
kind = 'gaussian'
kind_txt = 'Gaussian smoothing'


Lambda_int = 3
Lambda = Lambda_int * (2*np.pi)

paths = [f'cosmo_sim_1d/sim_k_1_7/run1/', f'cosmo_sim_1d/sim_k_1_11/run1/', f'cosmo_sim_1d/sim_k_1_15/run1/']
file_keys = [7, 11, 15]
for j in range(1):
    j = 1
    path = paths[j]
    file_key = file_keys[j]

    a_list, ctot2, ctot2_all = [], [], []
    # for j in tqdm(range(50)):
    # for j in range(14, 24):
        # j = 18

    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=False, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1]})

    fig.suptitle(rf'$\Lambda = {Lambda_int} \,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=22, y=0.995)
    j = 0
    a, x, d1k, kappa_l, Phi_l, tau_l = EFT_solve(j, Lambda, path, kind, folder_name)
    a, x, d1k, kappa_all, Phi_all, tau_all = EFT_solve(j, Lambda, path, kind, folder_name='')
    ax[0].set_title(rf'$a={a}$', fontsize=20, x=0.125, y=0.9)
    ax[0].plot(x, tau_l, c='r', lw=1.5, label=r'Single-stream particles')
    ax[0].plot(x, (tau_all - tau_l), c='b', lw=1.5, label=r'Multistreaming particles')
    ax[0].plot(x, tau_all, ls='dashed', c='k', lw=1.5, label='Total')
    ax[0].plot(x, kappa_l, ls='dashdot', c='seagreen', lw=1.5, label='$\rho_{l}v^{2}_{l}$')


    j = 15
    a, x, d1k, kappa_l, Phi_l, tau_l = EFT_solve(j, Lambda, path, kind, folder_name)
    a, x, d1k, kappa_all, Phi_all, tau_all = EFT_solve(j, Lambda, path, kind, folder_name='')
    ax[1].set_title(rf'$a={a}$', fontsize=20, x=0.125, y=0.9)
    ax[1].plot(x, tau_l, c='r', lw=1.5)
    ax[1].plot(x, (tau_all - tau_l), c='b', lw=1.5)
    ax[1].plot(x, tau_all, ls='dashed', c='k', lw=1.5)
    ax[1].plot(x, kappa_l, ls='dashdot', c='seagreen', lw=1.5)


    j = 22
    a, x, d1k, kappa_l, Phi_l, tau_l = EFT_solve(j, Lambda, path, kind, folder_name)
    a, x, d1k, kappa_all, Phi_all, tau_all = EFT_solve(j, Lambda, path, kind, folder_name='')
    ax[2].set_title(rf'$a={a}$', fontsize=20, x=0.125, y=0.9)
    ax[2].plot(x, tau_l, c='r', lw=1.5)
    ax[2].plot(x, (tau_all - tau_l), c='b', lw=1.5)
    ax[2].plot(x, tau_all, ls='dashed', c='k', lw=1.5)
    ax[2].plot(x, kappa_l, ls='dashdot', c='seagreen', lw=1.5)

    for j in range(3):
        ax[j].set_xlabel(r'$x/L$', fontsize=20)
        ax[j].minorticks_on()
        ax[j].tick_params(axis='both', which='both', direction='in', labelsize=18)
        ax[j].yaxis.set_ticks_position('both')
    
    ax[0].set_ylabel(r'$\langle[\tau]_{\Lambda}\rangle\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=20)
    ax[2].set_ylabel(r'$\langle[\tau]_{\Lambda}\rangle\;[\mathrm{M}_\mathrm{p}H_{0}^{2}L^{-1}]$', fontsize=20)
    ax[2].yaxis.set_label_position('right')
    ax[0].legend(fontsize=18)
    # ax[2].tick_params(labelleft=False, labelright=True)


    # plt.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.375, 1))
    plt.subplots_adjust(wspace=0.1)
    plt.show()
    # ax.set_ylim(-0.5, 4.5)

    # plt.savefig(f'../plots/paper_plots_final/tau_decomp.pdf', bbox_inches='tight', dpi=300)
    # plt.close()
