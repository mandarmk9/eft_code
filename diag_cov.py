#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pickle
from functions import spectral_calc, smoothing, EFT_sm_kern, read_density, SPT_tr, read_sim_data, read_hier
from tqdm import tqdm
import sympy
from sympy import *

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


    cov_mat = np.cov(dc_l, dv_l)
    tD = np.mean(tau_l*dc_l) / rho_b
    tT = np.mean(tau_l*dv_l) / rho_b
    DT = np.mean(dc_l*dv_l)
    TT = np.mean(dv_l*dv_l)
    DD = np.mean(dc_l*dc_l)
    rhs = (tD / DT) - (tT / TT)
    lhs = (DD / DT) - (DT / TT)
    cs2 = rhs / lhs
    cv2 = (DD*cs2 - tD) / DT

    print(cs2, cv2, cs2+cv2)
    # print(cov_mat[0][0], cov_mat[0][1], cov_mat[1][0], cov_mat[1][1])
    print(tD / DD)
    return a, x, sigma_l, p_l, kappa_l, Phi_l, tau_l, dc_l, dv_l, cov_mat, cs2, cv2, tD / DD#+cv2, tD/DD 

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
folder_name = 'shell_crossed_hier/'
kind = 'sharp'
kind_txt = 'sharp cutoff'
kind = 'gaussian'
kind_txt = 'Gaussian smoothing'


Lambda_int = 3
Lambda = Lambda_int * (2*np.pi)
# j = 15


# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})

# fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharex=True, sharey=True, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1]})
# fig.suptitle(rf'$\Lambda = {Lambda_int} \,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=22, y=0.975)

# j = 19
a_list, c1_list, c2_list, cD_list = [], [], [], []
for j in (range(51)):
    a, x, sigma_l, p_l, kappa_l, Phi_l, tau_l, dc_l, dv_l, cov, c1, c2, cD = EFT_solve(j, Lambda, path, kind)
    a_list.append(a)
    c1_list.append(c1)
    c2_list.append(c2)
    cD_list.append(cD)


import pandas
df = pandas.DataFrame(data=[a_list, c1_list, c2_list, cD_list])
file = open("./{}/cross_corr_cs_L{}_{}.p".format(path, int(Lambda/(2*np.pi)), kind), 'wb')
pickle.dump(df, file)
file.close()



# # # # plt.plot(a_list, c1_list)
# # # # plt.plot(a_list, c2_list)
# # # # plt.show()

# file = open("./{}/ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'rb')
# read_file = pickle.load(file)
# a_list, ctot2_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, err4_list = np.array(read_file)
# print(a_list.size)
# file.close()
# N = 50
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_title(rf'$\Lambda = {Lambda_int} \,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=18, y=1.01)


# ax.set_xlabel(r'$a$', fontsize=20)
# ax.set_ylabel('$c_{\mathrm{tot}}^{2}\;[H_{0}^{2}L^{2}]$', fontsize=20)



# # ax.fill_between(a_list, ctot2_list-t_err, ctot2_list+t_err, color='darkslategray', alpha=0.55, zorder=2)
# ctot2_line, = ax.plot(a_list[:N], ctot2_list[:N], c='k', lw=1.5, zorder=4, marker='o') #from tau fit
# # ctot2_line, = ax.plot(a_3list[:N], ctot2_list[:N], c='k', lw=1.5, zorder=4, marker='o') #from tau fit

# ctot2_2_line, = ax.plot(a_list[:N], ctot2_2_list[:N], c='cyan', lw=1.5, marker='*', zorder=2) #M&W
# ctot2_3_line, = ax.plot(a_list[:N], ctot2_3_list[:N], c='orange', lw=1.5, marker='v', zorder=3) #B+12

# # ctot2_4_line, = ax.plot(a_list[:N], ctot2_4_list[:N], c='lightseagreen', lw=1.5, marker='+', zorder=1) #DDE
# ctot2_5_line, = ax.plot(a_list[:N], np.array(c1_list[:N])+np.array(c2_list[:N]), c='magenta', lw=1.5, zorder=1) #DDE

# plt.legend(handles=[ctot2_line, ctot2_2_line, ctot2_3_line, ctot2_5_line], labels=[r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$', r'spatial corr'], fontsize=14, framealpha=0.75)

# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
# ax.yaxis.set_ticks_position('both')
# plt.show()
# # # # print(ctot2_list[N-1], ctot2_2_list[N-1])
# # plt.savefig(f'../plots/paper_plots_final/ctot2_ev_{kind}.png', bbox_inches='tight', dpi=300)
# # plt.close()

# # from decimal import Decimal, getcontext

# # # Set the desired precision (e.g. 50 decimal places)
# # getcontext().prec = 10
# # print(np.linalg.det(cov))
# # print(cov[0][0])
# # # sym_cov = sympy.Matrix(cov)
# # # diag_cov = sym_cov.diagonalize()
# # # # diag_cov = np.array(diag_cov) / np.sqrt(diag_cov[0,0]**2 + diag_cov[1, 1]**2)
# # # # print(diag_cov)

# # # eigenvalues, eigenvectors = np.linalg.eig(cov)
# # # diagonal = np.diag(eigenvalues)
# # # inverse_eigenvalues = np.linalg.inv(eigenvectors)
# # # diag_cov = eigenvectors @ diagonal @ inverse_eigenvalues
# # # print(cov, diag_cov)