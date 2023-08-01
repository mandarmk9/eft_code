#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from functions import spectral_calc, smoothing, EFT_sm_kern, read_density, read_hier, param_calc_ens

def field_calc(j, Lambda, path, A, kind):
    nbody_filename = 'output_{0:04d}.txt'.format(j)
    nbody_file = np.genfromtxt(path + nbody_filename)
    x_nbody = nbody_file[:,-1]
    v_nbody = nbody_file[:,2]

    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x_cell = moments_file[:,0]
    M0_nbody = moments_file[:,2]
    M1_nbody = moments_file[:,4]
    M2_nbody = moments_file[:,6]
    C1_nbody = moments_file[:,5]
    C2_nbody = moments_file[:,7]

    M0_hier = M0_nbody

    # a, dx, M0_nbody, M1_nbody, M2_nbody, C0_nbody, C1_nbody, C2_nbody = read_hier(path, j)
    # x = np.arange(0, 1.0, dx)

    x = x_cell
    L = 1.0 #x[-1]
    Nx = x.size
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    rho_0 = 27.755 #this is the comoving background density
    rho_b = rho_0 / (a**3) #this is the physical background density
    H0 = 100

    # M0_nbody /= np.mean(M0_nbody)

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
    tau_l = (kappa_l + Phi_l)
    y_l = v_l**2 / rho_l#(pi_l**2 / rho_l)

    # plt.rcParams.update({"text.usetex": True})
    # plt.rcParams.update({"font.family": "serif"})
    # fig, ax = plt.subplots()
    # ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(a, int(Lambda/(2*np.pi)), kind_txt), fontsize=12)
    # ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=12)
    # # ax.plot(x, tau_l, c='b', lw=2)
    # # ax.plot(x, grad_phi_l2, c='b', lw=2)
    # # ax.plot(x, grad_phi2_l, c='r', lw=2, ls='dashed')
    # # ax.plot(x, grad_phi_l2 - grad_phi2_l, c='k', lw=2, ls='dotted')
    # # ax.plot(x, sigma_l, c='b', lw=2)
    # # ax.plot(x, y_l, c='r', lw=2)
    # ax.plot(x, kappa_l, c='k', lw=2)
    #
    #
    #
    # ax.minorticks_on()
    # ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
    # ax.ticklabel_format(scilimits=(-2, 3))
    # ax.yaxis.set_ticks_position('both')
    # plt.show()
    # sigma_l = sigma_l - y_l

    Lambda = int(Lambda / (2 * np.pi))
    M2_k = np.fft.fft(M2)
    n_trunc = M2.size-Lambda
    # print(Lambda+1, n_trunc)
    M2_k[Lambda+1:n_trunc] = 0+0j
    # print(M2_k[:5])
    sigma_l = np.real(np.fft.ifft(M2_k))

    print('L', Lambda, 'sigma_l', (np.fft.fft(Phi_l)/kappa_l.size)[:10])

    return a, k, x, dc_l, dv_l, kappa_l, Phi_l, sigma_l, y_l


# path = 'cosmo_sim_1d/new_sim_k_1_11/run15/'
path = 'cosmo_sim_1d/sim_k_1_11/run1/'

A = []
# j = 20
Lambda = (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
fields = ['den', 'vel', 'kappa', 'Phi', 'tau']
field = fields[3]
# plots_folder = 'sim_k_1_11/sm/{}'.format(field)
plots_folder = 'sim_k_1/tau_lam/'.format(field)
mode = 1

L1, L2, L3 = 2, 3, 4
# a, x, ctot2, ctot2_2, ctot2_3, err0, err1, err2, cs2, cv2, red_chi, yerr, tau_l, fit, terr, P_nb, P_1l, d1k
# for l in range(2, 100):
#     Lambda = l * (2*np.pi)
for j in range(1):
    j = 0
    sol_1 = field_calc(j, L1*Lambda, path, A, kind)
    sol_2 = field_calc(j, L2*Lambda, path, A, kind)
    sol_3 = field_calc(j, L3*Lambda, path, A, kind)

    a, k, x = sol_1[:3]

    kappa_1 = sol_1[5]
    kappa_2 = sol_2[5]
    kappa_3 = sol_3[5]

    sigma_1 = sol_1[-2]
    sigma_2 = sol_2[-2]
    sigma_3 = sol_3[-2]

    y_1 = sol_1[-1]
    y_2 = sol_2[-1]
    y_3 = sol_3[-1]

    # Lambda = L1*Lambda
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots()
    ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(a, int(Lambda/(2*np.pi)), kind_txt), fontsize=12)
    ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=12)
    k/= (2*np.pi)

    ax.scatter(k, np.real(np.fft.fft(kappa_1)/kappa_1.size), c='b', s=40, label=r'${}\Lambda$'.format(L1))
    ax.scatter(k, np.real(np.fft.fft(kappa_2)/kappa_2.size), c='k', s=30, label=r'${}\Lambda$'.format(L2))
    ax.scatter(k, np.real(np.fft.fft(kappa_3)/kappa_3.size), c='r', s=20, label=r'${}\Lambda$'.format(L3))

    ax.set_xlim(-0.5, 14)
    # ax.plot(x, kappa_1, c='b', lw=2, label=r'${}\Lambda$'.format(L1))
    # ax.plot(x, kappa_2, c='k', lw=2, label=r'${}\Lambda$'.format(L2), ls='dashdot')
    # ax.plot(x, kappa_3, c='r', lw=2, label=r'${}\Lambda$'.format(L3), ls='dashed')

    # ax.plot(x, sigma_1, c='b', lw=2, label=r'${}\Lambda$'.format(L1))
    # ax.plot(x, sigma_2, c='k', lw=2, label=r'${}\Lambda$'.format(L2), ls='dashdot')
    # ax.plot(x, sigma_3, c='r', lw=2, label=r'${}\Lambda$'.format(L3), ls='dashed')

    # ax.plot(x, y_1, c='b', lw=1, label=r'${}\Lambda$'.format(L1))
    # ax.plot(x, y_2, c='k', lw=1, label=r'${}\Lambda$'.format(L2), ls='dashdot')
    # ax.plot(x, y_3, c='r', lw=1, label=r'${}\Lambda$'.format(L3), ls='dashed')


    plt.legend()
    # ax.plot(x, tau_l, c='b', lw=2)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
    ax.ticklabel_format(scilimits=(-2, 3))

    ax.yaxis.set_ticks_position('both')

    plt.show()
    # plt.savefig('../plots/test/new_paper_plots/kappa_L{}.png'.format(Lambda), bbox_inches='tight', dpi=150)
    # plt.close()

    # a, k, x, dc_l, dv_l, kappa_l, Phi_l = field_calc(j, Lambda, path, A, kind)
    # tau_l = kappa_l + Phi_l
    # # sol = param_calc_ens(j, Lambda, path, A, mode, kind, n_runs=24, n_use=10)
    # # x = sol[1]
    # # a = sol[0]
    # # tau_l = sol[-6]
    #
    # plt.rcParams.update({"text.usetex": True})
    # plt.rcParams.update({"font.family": "serif"})
    # fig, ax = plt.subplots()
    # ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(a, int(Lambda/(2*np.pi)), kind_txt), fontsize=12)
    # ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=12)
    # if field == 'tau':
    #     ax.plot(x, tau_l, c='b', lw=2)#, label=r'$[\tau]_{\Lambda}$')
    #     ax.set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=12)
    #
    # elif field == 'Phi':
    #     ax.plot(x, Phi_l, c='b', lw=2)#, label=r'$[\tau]_{\Lambda}$')
    #     ax.set_ylabel(r'$[\Phi]_{l}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=12)
    #
    # elif field == 'kappa':
    #     ax.plot(x, kappa_l, c='b', lw=2)#, label=r'$[\tau]_{\Lambda}$')
    #     ax.set_ylabel(r'$[\kappa]_{l}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=12)
    #
    # elif field == 'den':
    #     ax.plot(x, dc_l, c='b', lw=2)
    #     ax.set_ylabel(r'$\delta_{l}$', fontsize=12)
    #
    # elif field == 'vel':
    #     ax.plot(x, dv_l, c='b', lw=2)
    #     ax.set_ylabel(r'$v_{l}\;\;[\mathrm{km\;s}^{1}}]$', fontsize=12)
    #
    # else:
    #     raise Exception('Field not specified correctly.')
    # # ax.set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=12)
    #
    # ax.minorticks_on()
    # ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
    # ax.ticklabel_format(scilimits=(-2, 3))
    # # ax.grid(lw=0.2, ls='dashed', color='grey')
    # # ax.legend(fontsize=12, bbox_to_anchor=(1,1))
    # ax.yaxis.set_ticks_position('both')
    # plt.show()
    # # plt.savefig('../plots/{}/{}_{}.png'.format(plots_folder, field, j), bbox_inches='tight', dpi=150)
    # # plt.savefig('../plots/{}/tau_{}.png'.format(plots_folder, l), bbox_inches='tight', dpi=150)
    # #
    # # plt.close()
