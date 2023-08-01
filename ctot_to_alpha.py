#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
Nfiles = 24

file = open("./{}/ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'rb')
read_file = pickle.load(file)
a_list, ctot2_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, err_list = np.array(read_file)
file.close()

ctot2_2_list = ctot2_4_list[:Nfiles]
err_list = err_list[:Nfiles]

file = open("./{}/new_binning_ctot2_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'rb')
read_file = pickle.load(file)
a_list, ctot2_list, err = np.array(read_file)
file.close()

#An and Bn for the integral over the Green's function
An = np.zeros(Nfiles)
Bn = np.zeros(Nfiles)
An2 = np.zeros(Nfiles)
Bn2 = np.zeros(Nfiles)

err_I1 = np.zeros(Nfiles)
err_I2 = np.zeros(Nfiles)

#for α_c using c^2 from fit to τ_l
Pn = ctot2_list * (a_list**(5/2)) #for calculation of alpha_c
Qn = ctot2_list
Pn2 = ctot2_2_list * (a_list**(5/2)) #for calculation of alpha_c
Qn2 = ctot2_2_list


da_list = np.roll(a_list, 1) - a_list
err_A = (err_list * (a_list**(5/2)) * da_list)**2
err_B = (err_list * da_list)**2

#A second loop for the integration
for j in range(1, Nfiles):
    An[j] = np.trapz(Pn[:j], a_list[:j])
    Bn[j] = np.trapz(Qn[:j], a_list[:j])

    An2[j] = np.trapz(Pn2[:j], a_list[:j])
    Bn2[j] = np.trapz(Qn2[:j], a_list[:j])

    err_I1[j] = sum(err_A[:j])
    err_I2[j] = sum(err_B[:j])

#calculation of the Green's function integral
C = 2 / (5 * 100**2)
An /= (a_list**(5/2))
An2 /= (a_list**(5/2))

err_I1 = np.sqrt(err_I1) / a_list**(5/2)
err_I2 = np.sqrt(err_I2)

err_Int = C * np.sqrt(err_I1**2 + err_I2**2) * 1e4
alpha_c = C * (An - Bn) * 1e4
alpha_c_2 = C * (An2 - Bn2) * 1e4

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(9,6))
ax.set_title(r'$k = k_{{f}}, \Lambda = {}\,k_{{f}}$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=20, y=1.01)
ax.set_xlabel(r'$a$', fontsize=20)
ax.set_ylabel(r'$\alpha_{c} \;[10^{-4}L^{2}]$', fontsize=20)
ax.plot(a_list, alpha_c, c='k', lw=2)
ax.plot(a_list, alpha_c_2, c='r', lw=2)
ax.fill_between(a_list, alpha_c-err_Int, alpha_c+err_Int, rasterized=True, alpha=0.55, color='darkslategray')
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.yaxis.set_ticks_position('both')
plt.savefig('../plots/test/new_paper_plots/test.png'.format(kind), bbox_inches='tight', dpi=150)
plt.close()
