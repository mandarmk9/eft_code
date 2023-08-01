#!/usr/bin/env python3

#import libraries
import pandas
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from functions import read_sim_data, param_calc_ens
from tqdm import tqdm
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def tau_ext(file_num, Lambda, path, mode, kind, folder_name, n_use=10):
    a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, file_num, folder_name)
    taus = []
    taus.append(tau_l_0)
    for run in range(1, n_runs+1):
        path = path[:-2] + '{}/'.format(run)
        sol = read_sim_data(path, Lambda, kind, file_num, folder_name)
        taus.append(sol[-3])

    Nt = len(taus)
    tau_l = sum(np.array(taus)) / Nt

    rho_0 = 27.755
    rho_b = rho_0 / a**3
    H0 = 100

    diff = np.array([(taus[i] - tau_l)**2 for i in range(1, Nt)])
    yerr = np.sqrt(sum(diff) / (Nt*(Nt-1)))
    n_ev = x.size // n_use
    dc_l_sp = dc_l[0::n_ev]
    dv_l_sp = dv_l[0::n_ev]
    tau_l_sp = tau_l[0::n_ev]
    yerr_sp = yerr[0::n_ev]

    guesses = 1, 1, 1
    def fitting_function(X, a0, a1, a2):
        x1, x2 = X
        return a0 + a1*x1 + a2*x2
    C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    cs2 = np.real(C[1] / rho_b)
    cv2 = -np.real(C[2] * H0 / (rho_b * np.sqrt(a)))
    ctot2 = (cs2 + cv2)
    P_lin = np.abs(d1k**2) * a**2
    return a, x, tau_l_0, tau_l, dc_l, dv_l, P_lin, ctot2


path = 'cosmo_sim_1d/sim_k_1_11/run1/'

# path = 'cosmo_sim_1d/multi_sim_3_15_33/run1/'
Lambda_int = 3
Lambda = Lambda_int * (2 * np.pi)
mode = 1
kind = 'sharp'
kind_txt = 'sharp cutoff'
kind = 'gaussian'
kind_txt = 'Gaussian smoothing'

Nfiles = 23
n_runs = 8
n_use = 10
folder_name = '/new_hier/data_{}/L{}/'.format(kind, Lambda_int)
A = []
plots_folder = '/paper_plots_final/'


#An and Bn for the integral over the Green's function
An = np.zeros(Nfiles)
Bn = np.zeros(Nfiles)
Pn = np.zeros(Nfiles)
Qn = np.zeros(Nfiles)

An2 = np.zeros(Nfiles)
Bn2 = np.zeros(Nfiles)
Pn2 = np.zeros(Nfiles)
Qn2 = np.zeros(Nfiles)

An3 = np.zeros(Nfiles)
Bn3 = np.zeros(Nfiles)
Pn3 = np.zeros(Nfiles)
Qn3 = np.zeros(Nfiles)

An4 = np.zeros(Nfiles)
Bn4 = np.zeros(Nfiles)
Pn4 = np.zeros(Nfiles)
Qn4 = np.zeros(Nfiles)


AJ = np.zeros(Nfiles)
BJ = np.zeros(Nfiles)
PJ = np.zeros(Nfiles)
QJ = np.zeros(Nfiles)

P_nb = np.zeros(Nfiles)
P_1l = np.zeros(Nfiles)
P_lin = np.zeros(Nfiles)
dk_lin = np.zeros(Nfiles, dtype='complex')

sol = tau_ext(4, Lambda, path, mode, kind, folder_name)
a1, ctot2_1 = sol[0], sol[-1]

sol = tau_ext(5, Lambda, path, mode, kind, folder_name)
a2, ctot2_2 = sol[0], sol[-1]
slope = (ctot2_2-ctot2_1) / (a2-a1)

a_list = np.array([np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j)) for j in range(Nfiles)])
C_P = 4*slope*(a_list[0]**(9/2)) / (45 * (100**2) * a_list**(5/2))
C_Q = (slope*a_list[0]**2 / (5*100**2)) * np.ones(a_list.size)
alpha_c_0 = C_P - C_Q #this is the part of alpha_c integrated from 0 to a0. Add this to the integral from a0 to a

flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

a0 = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(0))
q = np.genfromtxt(path + 'output_{0:04d}.txt'.format(0))[:,0]

for file_num in tqdm(range(Nfiles)):
   #the function 'EFT_solve' return solutions of all modes + the EFT parameters
   ##the following line is to keep track of 'a' for the numerical integration
   if file_num > 0:
      a0 = a

   a, x, ctot2, ctot2_2, ctot2_3, err0, err1, err2, cs2, cv2, red_chi, yerr, tau_l, fit, terr, P_nb_a, P_1l_a_tr, \
    d1k, taus, x_binned, chisq, ctot2_4, err_4 = param_calc_ens(file_num, Lambda, path, A, mode, kind, folder_name=folder_name, n_runs=n_runs, n_use=n_use, ens=True)

   P_nb[file_num] = P_nb_a[mode]
   P_1l[file_num] = P_1l_a_tr[mode]
   P_lin[file_num] = (np.abs(d1k**2)*a**2)[mode]
   dk_lin[file_num] = d1k[mode] * a
   Nx = x.size
   k = np.fft.ifftshift(2.0 * np.pi * np.arange(-Nx/2, Nx/2))
   J = np.real(np.fft.fft(taus[0] - tau_l)[mode]) / tau_l.size

   ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
   if file_num > 0:
      da = a - a0
      #for α_c using c^2 from fit to τ_l
      Pn[file_num] = ctot2 * (a**(5/2)) #for calculation of alpha_c
      Qn[file_num] = ctot2

      #for α_c using τ_l directly (M&W)
      Pn2[file_num] = ctot2_2 * (a**(5/2)) #for calculation of alpha_c
      Qn2[file_num] = ctot2_2

      #for α_c using correlations (Baumann)
      Pn3[file_num] = ctot2_3 * (a**(5/2)) #for calculation of alpha_c
      Qn3[file_num] = ctot2_3

      #for α_c using DDE
      Pn4[file_num] = ctot2_4 * (a**(5/2)) #for calculation of alpha_c
      Qn4[file_num] = ctot2_4

      #for the stochastic contribution
      PJ[file_num] = J * (a**(3/2)) #for calculation of alpha_c
      QJ[file_num] = J / a


#A second loop for the integration
for j in range(1, Nfiles):
    An[j] = np.trapz(Pn[:j], a_list[:j])
    Bn[j] = np.trapz(Qn[:j], a_list[:j])

    An2[j] = np.trapz(Pn2[:j], a_list[:j])
    Bn2[j] = np.trapz(Qn2[:j], a_list[:j])

    An3[j] = np.trapz(Pn3[:j], a_list[:j])
    Bn3[j] = np.trapz(Qn3[:j], a_list[:j])

    An4[j] = np.trapz(Pn4[:j], a_list[:j])
    Bn4[j] = np.trapz(Qn4[:j], a_list[:j])

    AJ[j] = np.trapz(PJ[:j], a_list[:j])
    BJ[j] = np.trapz(QJ[:j], a_list[:j])

C = 2 / (5 * 100**2)
An /= (a_list**(5/2))
An2 /= (a_list**(5/2))
An3 /= (a_list**(5/2))
An4 /= (a_list**(5/2))


AJ /= (a_list**(3/2))
BJ *= a_list
del_J = k[mode]**2 * C*(AJ - BJ) / (27.755 / a_list**3)
 

alpha_c_true = k[mode]**2 * ((P_nb - P_1l) / (2 * P_lin * k[mode]**2))
alpha_c_naive = k[mode]**2 * ((C * (An - Bn)) + alpha_c_0)
alpha_c_naive2 = k[mode]**2 * ((C * (An2 - Bn2)) + alpha_c_0)
alpha_c_naive3 = k[mode]**2 * ((C * (An3 - Bn3)) + alpha_c_0)
alpha_c_naive4 = k[mode]**2 * ((C * (An4 - Bn4)) + alpha_c_0)
alpha_c_pred = -slope*a_list**2*k[mode]**2 / (9 * 100**2)

df = pandas.DataFrame(data=[a_list, alpha_c_true, alpha_c_naive, alpha_c_naive2, alpha_c_naive3, alpha_c_naive4, alpha_c_pred])
file = open(f"./{path}/alpha_c_{kind}_{Lambda_int}.p", "wb")
pickle.dump(df, file)
file.close()

N = 23
file = open(f"./{path}/alpha_c_{kind}_{Lambda_int}.p", "rb")
read_file = pickle.load(file)
a_list, alpha_c_true, alpha_c_naive, alpha_c_naive2, alpha_c_naive3, alpha_c_naive4, alpha_c_pred = np.array(read_file)
file.close()

xaxis = a_list[:N]

if kind == 'sharp':
    yaxes = [alpha_c_true[:N], alpha_c_naive[:N], alpha_c_naive2[:N], alpha_c_naive3[:N], alpha_c_naive4[:N]]
    colors = ['g', 'k', 'cyan', 'orange', 'lightseagreen']
    linestyles = ['solid', 'dashed', 'dashed', 'dashed', 'dashed']
    labels=[r'from matching $P_{N-\mathrm{body}}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$', r'DDE']

elif kind == 'gaussian':
    yaxes = [alpha_c_true[:N], alpha_c_naive[:N], alpha_c_naive2[:N], alpha_c_naive3[:N]]#, alpha_c_naive4[:N]]
    colors = ['g', 'k', 'cyan', 'orange']#, 'lightseagreen']
    linestyles = ['solid', 'dashed', 'dashed', 'dashed']#, 'dashed']
    labels=[r'from matching $P_{N-\mathrm{body}}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$', r'DDE']

else:
    pass

savename = 'alpha_c_{}'.format(kind)

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(9,6))
ax.set_title(rf'$k = k_{{\mathrm{{f}}}}, \Lambda = {Lambda_int}\,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=20, y=1.01)
ax.set_xlabel(r'$a$', fontsize=20)
ax.set_ylabel(r'$k^{2}\alpha_{c}$', fontsize=20)
errors = [(100 * (yaxes[j] - yaxes[0]) / yaxes[0]) for j in range(len(yaxes))]

handles = []
for i in range(len(yaxes)):
    line, = ax.plot(xaxis, yaxes[i], c=colors[i], ls=linestyles[i], lw=2)
    handles.append(line)

pred_line, = ax.plot(a_list[:N], alpha_c_pred[:N], ls='dashdot', c='magenta', lw=1.5, zorder=0)
handles.append(pred_line)
labels.append(r'$\alpha_{c} \propto a^{2}$')

for j in range(Nfiles):
    if flags[j] == 1:
        sc_line = ax.axvline(a_list[j], c='teal', lw=0.5, zorder=1)
    else:
        pass

labels.append(r'$a_{\mathrm{sc}}$')
handles.append(sc_line)


ax.legend(handles, labels, fontsize=14, framealpha=1, loc=3) #, bbox_to_anchor=(1,1))
for j in range(2):
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax.yaxis.set_ticks_position('both')


plt.savefig(f'../plots/{plots_folder}/alpha_c_{kind}.pdf', bbox_inches='tight', dpi=150)
plt.close()
# plt.show()
