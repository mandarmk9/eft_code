#!/usr/bin/env python3
import tqdm as tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize, shgo
from scipy.optimize import Bounds
# from functions import

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda_int = 3
Lambda = Lambda_int * (2 * np.pi)
mode = 1
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'


file = open(f"./{path}/new_trunc_alpha_c_{kind}_{Lambda_int}.p", "rb")
read_file = pickle.load(file)
a_list, alpha_c_true, alpha_c_naive, alpha_c_naive2, alpha_c_naive3, alpha_c_naive4 = np.array(read_file)
file.close()

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda_int = 3
Lambda = Lambda_int * (2 * np.pi)
mode = 1
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

Nfiles = 50
n_runs = 8
n_use = 10
plots_folder = 'test/c_from_alpha/'

# def alpha_c_machine(ctot2_list, a_list):
#     slope = (ctot2_list[1]-ctot2_list[0]) / (a[1]-a[0])
#     C_P = 4*slope*(a_list[0]**(9/2)) / (45 * (100**2) * a_list**(5/2))
#     C_Q = (slope*a_list[0]**2 / (5*100**2)) * np.ones(a_list.size)
#     alpha_c_0 = C_P - C_Q
#     Pn = ctot2_list * a_list**(5/2)
#     Qn = ctot2_list * a_list
#     for j in range(1, a_list.size):
#         An[j] = np.trapz(Pn[:j], a_list[:j])
#         Bn[j] = np.trapz(Qn[:j], a_list[:j])
#
#     An /= a_list**(5/2)
#     alpha_c_num = (An - Bn) * (2 / (5*100**2))
#     return alpha_c_num + alpha_c_0


def f(x, y):
    An = np.zeros_like(y)
    Bn = np.zeros_like(y)
    slope = (x[1]-x[0]) / (y[1]-y[0])
    C_P = 4*slope*(y[0]**(9/2)) / (45 * (100**2) * y**(5/2))
    C_Q = (slope*y[0]**2 / (5*100**2)) * np.ones(y.size)
    z_0 = (C_P - C_Q)
    Pn = x * y**(5/2)
    Qn = x * y
    for j in range(1, y.size):
        An[j] = np.trapz(Pn[:j], a_list[:j])
        Bn[j] = np.trapz(Qn[:j], a_list[:j])

    An /= y**(5/2)
    z_guess = ((An - Bn) * (2 / (5*100**2))) + z_0
    return z_guess


def objective(x, y, z):
    return np.abs(sum((f(x, y) - z)))**(1/64)

# z = alpha_c_true[:23]
# y = a_list[:23]
# x0 = y * 100
# print(objective(x0, y, z))

def g(y, z):
    x0 = np.ones(z.size)
    result = minimize(objective, x0, args=(y,z))

    return result

z = alpha_c_true[:23]
y = a_list[:23]

output = g(y, z)
print(output)
ctot2_list_approx = output.x

file = open(rf"./{path}/new_ctot2_plot_{kind}_L{Lambda_int}.p", 'rb')
read_file = pickle.load(file)
a_list, ctot2_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, err4_list = np.array(read_file)
file.close()

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
ax.set_title(rf'$k = {mode}\,k_{{\mathrm{{f}}}}, \Lambda = {Lambda_int}\,k_{{\mathrm{{f}}}}$ ({kind_txt})', fontsize=20, y=1.01)
ax.set_xlabel(r'$a$', fontsize=20)
ax.set_ylabel(r'$\alpha_{c}\,[L^{-2}]$', fontsize=20)
ax.plot(y[2:22], ctot2_list_approx[2:22], c='b', lw=1.5, marker='o')
ax.plot(a_list[2:22], ctot2_list[2:22], c='k', lw=1.5, marker='*')
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.yaxis.set_ticks_position('both')

plt.show()
