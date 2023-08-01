#!/usr/bin/env python3

#import libraries
# import h5py
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pickle
from functions import read_density, dc_in_finder, dn, smoothing, read_sim_data
from tqdm import tqdm


def SPT(dc_in, L, a):
   """Returns the SPT PS upto 2-loop order"""
   Nx = dc_in.size
   F = dn(5, L, dc_in)
   d1k = (np.fft.fft(F[0]) / Nx)
   d2k = (np.fft.fft(F[1]) / Nx)
   d3k = (np.fft.fft(F[2]) / Nx)
   d4k = (np.fft.fft(F[3]) / Nx)
   d5k = (np.fft.fft(F[4]) / Nx)

   P11 = np.real((d1k * np.conj(d1k)) * (a**2))
   P12 = np.real(((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3))
   P22 = np.real((d2k * np.conj(d2k)) * (a**4))
   P13 = np.real(((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4))
   P14 = np.real(((d1k * np.conj(d4k)) + (d4k * np.conj(d1k))) * (a**5))
   P23 = np.real(((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a**5))
   P33 = np.real((d3k * np.conj(d3k)) * (a**6))
   P15 = np.real(((d1k * np.conj(d5k)) + (d5k * np.conj(d1k))) * (a**6))
   P24 = np.real(((d2k * np.conj(d4k)) + (d4k * np.conj(d2k))) * (a**6))
   return P11, P12, P22, P13, P14, P23, P33, P15, P24

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

Lambda_int = 3
j = 50
mode = 1
kf = (2*np.pi)
x = np.arange(0, 1.0, 0.001)
dc_in, k = dc_in_finder(path, x, interp=True)
Lambdas = np.array([3, 4, 5, 6, 7, 8])

def P13_calc(path, kind, j, dc_in):
    P13_ren_Lambda, P13_Lambda, Pnb = [], [], []
    for Lambda_int in Lambdas:
        Lambda = Lambda_int*(2*np.pi)
        folder_name = '/new_hier/data_{}/L{}/'.format(kind, Lambda_int)
        dc_in = smoothing(dc_in, k, Lambda, kind)


        a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
        P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)[-2:]
        P11, P12, P22, P13, _, _, _, _, _ = SPT(dc_in, 1.0, a)
        file = open(f"./{path}/alphas_{kind}_{Lambda_int}_{mode}.p", "rb")
        read_file = pickle.load(file)
        a_list, alpha_c_F3P, alpha_c_SC = np.array(read_file)
        file.close()

        alpha_c_F3P /= kf**2
        alpha_c_SC /= kf**2

        alpha_ren = (P_nb[mode] - P11[mode]) / (2 * kf**2 * P11[mode]) # this is a number (fixed k, \Lambda, a)
        alpha_ctr = alpha_c_F3P[j] - alpha_ren # again, a number
        P13_ren_Lambda.append(-2*alpha_ctr*(kf**2)*P11[mode])
        P13_Lambda.append(P13[mode])
        # Pnb.append(P_nb[mode])
        Pnb.append(alpha_ren)

    return a, np.array(P13_Lambda), np.array(P13_ren_Lambda), np.array(Pnb)

# # ylabel = r'$a^{-2}L^{-1}P(k, a) \times 10^{4}$'
ylabel = r'$\mathrm{log}_{10}\left|L^{-1}P_{13}(k_{\mathrm{ren}}, \Lambda)\right|$'

# ylabel = r'$\left|\frac{\Delta P_{13}}{P_{13}}\right|$'


plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
# title = rf'$a={a}, \,k_{{\mathrm{{ren}}}} = k_{{\mathrm{{f}}}}$'
title = rf'$k_{{\mathrm{{ren}}}} = k_{{\mathrm{{f}}}}$'

ax.set_xlabel(r'$\Lambda/k_{\mathrm{f}}$', fontsize=18)
ax.set_ylabel(ylabel, fontsize=18)

# ax.plot(Lambdas, np.log10(np.abs(P13_Lambda)), c='b', lw=2, marker='o', label=r'$P_{13}$')
# ax.plot(Lambdas, np.log10(np.abs(P13_ren_Lambda)), c='k', lw=2, marker='^', label=r'$P_{13,\,\mathrm{ren}}$')

# a, P13_Lambda, P13_ren_Lambda = P13_calc(path, kind, 0, dc_in)
# ax.plot(Lambdas, np.log10(np.abs(P13_Lambda)), c='b', lw=1.5, marker='o', label=rf'$a={a}$')
# ax.plot(Lambdas, np.log10(np.abs(P13_ren_Lambda)), c='k', lw=1.5, marker='^')#, label=rf'$a={a}$')

a, P13_Lambda, P13_ren_Lambda, P_nb = P13_calc(path, kind, 11, dc_in)
# ax.plot(Lambdas, np.log10(np.abs(P13_Lambda)), ls='solid', c='b', lw=1.5, marker='o')#, label=rf'$a={a}$')
# ax.plot(Lambdas, np.log10(np.abs(P13_ren_Lambda)), c='k', ls='solid', lw=1.5, marker='^')#, label=rf'$a={a}$')
# P13 = (np.abs((P13_Lambda - P13_ren_Lambda) / (P13_Lambda)))
# ax.plot(Lambdas, P13, ls='solid', c='b', lw=1.5, marker='o')#, label=rf'$a={a}$')
ax.plot(Lambdas, P_nb, ls='solid', c='b', lw=1.5, marker='o')#, label=rf'$a={a}$')
a0_line = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=1.5, ls='solid', label=rf'$a={a}$')

a, P13_Lambda, P13_ren_Lambda, P_nb = P13_calc(path, kind, 23, dc_in)
# ax.plot(Lambdas, np.log10(np.abs(P13_Lambda)), ls='dashed', c='b', lw=1.5, marker='o')#, label=rf'$a={a}$')
# ax.plot(Lambdas, np.log10(np.abs(P13_ren_Lambda)), c='k', ls='dashed', lw=1.5, marker='^')#, label=rf'$a={a}$')
# P13 = (np.abs((P13_Lambda - P13_ren_Lambda) / (P13_Lambda)))
# ax.plot(Lambdas, P13, ls='dashed', c='b', lw=1.5, marker='o')#, label=rf'$a={a}$')
ax.plot(Lambdas, P_nb, ls='dashed', c='b', lw=1.5, marker='o')#, label=rf'$a={a}$')
a1_line = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=1.5, ls='dashed', label=rf'$a={a}$')


a, P13_Lambda, P13_ren_Lambda, P_nb = P13_calc(path, kind, 50, dc_in)
# ax.plot(Lambdas, np.log10(np.abs(P13_Lambda)), ls='dotted', c='b', lw=1.5, marker='o')#, label=rf'$a={a}$')
# ax.plot(Lambdas, np.log10(np.abs(P13_ren_Lambda)), c='k', ls='dotted', lw=1.5, marker='^')#, label=rf'$a={a}$')
# P13 = (np.abs((P13_Lambda - P13_ren_Lambda) / (P13_Lambda)))
# ax.plot(Lambdas, P13, ls='dotted', c='b', lw=1.5, marker='o')#, label=rf'$a={a}$')
ax.plot(Lambdas, P_nb, ls='dotted', c='b', lw=1.5, marker='o')#, label=rf'$a={a}$')
a2_line = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=1.5, ls='dotted', label=rf'$a={a}$')

handles = [a0_line, a1_line, a2_line]
plt.legend(handles=handles, fontsize=13, bbox_to_anchor=(0.85, 1))


ax.tick_params(axis='both', which='both', direction='in', labelsize=14)
ax.set_title(f'{title}', fontsize=18)
ax.minorticks_on()
ax.yaxis.set_ticks_position('both')
# ax.set_ylim(-3.35, -3.25)
# plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig(f'../plots/paper_plots_final/renorm_lambda_{kind}.png', dpi=300)
plt.close()



