#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from functions import read_sim_data
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def Power_fou(f1, f2, Lambda_int=3):
    f1_k = np.fft.fft(f1)
    f2_k = np.fft.fft(f2)
    corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
    n_trunc = corr.size-Lambda_int
    corr[(Lambda_int+1):n_trunc] = 0
    return sum(corr)

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'

# kind = 'gaussian'

mode = 1

j = 25
a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
T = -dv_l / (100 / (a**(1/2)))
rho_b = 27.755 / (a**3)
tau_l -= np.mean(tau_l)
P_AD = Power_fou(tau_l/rho_b, dc_l)
P_AT = Power_fou(tau_l/rho_b, T)

P_AD2 = Power_fou(tau_l/rho_b, dc_l**2)
P_AT2 = Power_fou(tau_l/rho_b, T**2)
P_ADT = Power_fou(tau_l/rho_b, dc_l*T)


P_DD2 = Power_fou(dc_l, dc_l**2)
P_DD = Power_fou(dc_l, dc_l)
P_D2D2 = Power_fou(dc_l**2, dc_l**2)

P_DT = Power_fou(dc_l, T)
P_DTD = Power_fou(dc_l*T, dc_l)
P_DTT = Power_fou(dc_l*T, T)
P_DTD2 = Power_fou(dc_l*T, dc_l**2)
P_DTT2 = Power_fou(dc_l*T, T**2)
P_DTDT = Power_fou(dc_l*T, dc_l*T)

P_TT = Power_fou(T, T)
P_DT2 = Power_fou(dc_l, T**2)
P_TT2 = Power_fou(T, T**2)
P_TD2 = Power_fou(T, dc_l**2)
P_DDT = Power_fou(dc_l, dc_l*T)
P_DT2 = Power_fou(dc_l, T**2)
P_TDT = Power_fou(T, dc_l*T)
P_T2DT = Power_fou(T**2, dc_l*T)
P_D2T2 = Power_fou(T**2, dc_l**2)
P_D2DT = Power_fou(dc_l**2, dc_l*T)
P_T2T2 = Power_fou(T**2, T**2)


R1 = [P_DD, P_DT, P_DD2, P_DDT, P_DT2]
R2 = [P_DT, P_TT, P_TD2, P_TDT, P_TT2]
R3 = [P_DD2, P_TD2, P_D2D2, P_D2DT, P_D2T2]
R4 = [P_DT2, P_TT2, P_D2T2, P_T2DT, P_T2T2]
R5 = [P_DTD, P_DTT, P_DTD2, P_DTDT, P_DTT2]

B = np.matrix([P_AD, P_AT, P_AD2, P_AT2, P_ADT])
A = np.matrix([R1, R2, R3, R4, R5])
# print(B.shape, A.shape)
# X = np.dot(B, np.linalg.inv(A))
# print(X)

print(np.linalg.det(np.linalg.inv(A)))

ctot2 = np.real(P_AD / P_DD) #original B^{+12}
c2 = np.real((P_AD2*P_DD - P_AD*P_DD2) / (P_D2D2*P_DD - P_DD2)) #second-order coeff from B^{+12}
ctot2_second = np.real((P_AD - c2*P_DD2) / P_DD)


print('a = {}, first ctot2 = {}, second ctot2 = {}, c2 = {}'.format(a, ctot2, ctot2_second, c2))
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(r'$a = {}, \Lambda = {} \,k_{{\mathrm{{f}}}}$ ({})'.format(np.round(a, 3), int(Lambda/(2*np.pi)), kind_txt), fontsize=18, y=1.01)

ax.set_xlabel(r'$x/L$', fontsize=20)
ax.set_ylabel(r'$\rho_{b}^{-1}[\tau]_{\Lambda} \; [H_{0}^{2}L^{2}]$', fontsize=20)
ax.plot(x, tau_l/rho_b, c='b', lw=2, label='measured')
ax.plot(x, ctot2*dc_l, c='k', ls='dashdot', lw=2, label='M\&W: first-order')
ax.plot(x, ctot2_second*dc_l + c2*dc_l**2, c='r', ls='dashed', lw=2, label='M\&W: second-order')
plt.legend(fontsize=12)

ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.yaxis.set_ticks_position('both')

plt.savefig('../plots/test/new_paper_plots/taus_mw_{}.png'.format(kind), bbox_inches='tight', dpi=150)
plt.close()
