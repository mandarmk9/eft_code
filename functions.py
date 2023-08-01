#!/usr/bin/env python3
import numpy as np
import h5py
import time
import os
# import warnings
# warnings.filterwarnings("error")
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit, minimize
from scipy.interpolate import interp1d, interp2d, griddata
from lmfit import Minimizer, Parameters, report_fit
from tqdm import tqdm
# from scipy.integrate import simpson

def poisson_solver(rhs, k):
    V = np.fft.fft(rhs)
    V[0] = 0
    V[1:] /= -k[1:] ** 2
    return np.fft.ifft(V)

def initial_density(q, A, L):
    N_waves = len(A) // 2
    den = 0
    for j in range(0, N_waves):
        den += A[2*j] * np.cos(2 * np.pi * q * A[2*j+1] / L)
    return den

def nabla_Psi(q, A, a, L):
    return - a * initial_density(q, A, L)

def Psi_q_finder(q, A, L):
    N = q.size
    del_Psi = initial_density(q, A, L)
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-N/2, N/2))
    return spectral_calc(del_Psi, L, o=1, d=1)

def Psi(q, A, a, L):
    N = q.size
    del_Psi = nabla_Psi(q, A, a, L)
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-N/2, N/2))
    return spectral_calc(del_Psi, L, o=1, d=1)

def eul_pos(q, A, a, L):
    disp = Psi(q, A, a, L)
    return q + disp

def q_new(q, A, a, L):
    guess = 0
    def f(point):
        return point - a * (((A[0] * L / (2 * np.pi * A[1])) * np.sin(2 * np.pi * point * A[1] / L)) + ((A[2] * L / (2 * np.pi * A[3])) * np.sin(2 * np.pi * point * A[3] / L))) - c
    q_traj = np.empty(q.size)

    for i in range(q.size):
        c = q[i]
        q_traj[i] = fsolve(f, guess)
    return q_traj

def eul_vel(H0, q, A, a, L):
    return H0 *  Psi(q, A, a, L) * (a ** (3/2))

def sub_find(num, N):
    seq = np.arange(0, N, dtype=int)
    start_ = np.random.choice(seq)
    n_ev = N // num
    sub = []
    for j in range(num):
        ind_next = start_ + int(j*n_ev)
        if ind_next >= N:
            ind_next = ind_next - N
        sub.append(ind_next)
    sub = list(np.sort(sub))
    return sub


def spectral_calc(f, L, o, d):
    """a function to calculate a 1D spectral derivative/integral. o is the order,
    and d == 0(1) implies derivative(integral). L is the size of the box."""

    N = int(len(f))
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-N/2, N/2))
    w = np.zeros(N, dtype=complex)
    f_k = np.fft.fft(f)

    if d == 0:
        if o%2 == 1:
            w[:int(N/2)] = ((1j*k[:int(N/2)])**o)*(f_k[:int(N/2)])
            w[int(N/2)] = 0
            w[int(N/2):] = ((1j*k[int(N/2):])**o)*(f_k[int(N/2):])
        elif o%2 == 0:
            w = ((1j*k)**o)*(f_k)

        w = np.real(np.fft.ifft(w))

    elif d == 1:
        if o%2 == 1:
            w[0] = 0
            w[1:int(N/2)] = (f_k[1:int(N/2)])/((1j*k[1:int(N/2)])**o)
            w[int(N/2)] = 0
            w[int(N/2):] = (f_k[int(N/2):])/((1j*k[int(N/2):])**o)
        elif o%2 == 0:
            w[0] = 0
            w[1:] = (f_k[1:])/((1j*k[1:])**o)

        w = np.real(np.fft.ifft(w))

    return w

def husimi(psi, X, P, sigma_x, h, L):
    """A more efficient calculation of the Husimi distribution from ψ"""
    #we make the X-grid periodic
    X_per = X - 0
    X_per[X_per < 0] += L
    X_per[X_per > L/2] = - L + X_per[X_per > L/2]

    #now, calculate Husimi on the 2d grid
    A = ((2 * np.pi * h) ** (-1/2)) * ((2 * np.pi * sigma_x **2) ** (-1/4)) #the normalisation of Husimi

    g = np.exp(-((X_per**2) / (4 * (sigma_x**2))) + (1j * P * X_per / (h)))
    hu =  A * np.exp(-1j * X_per * P / (2 * h)) * np.fft.ifft(np.fft.fft(psi) * np.fft.fft(g, axis=1))
    F = np.abs(hu)**2
    # F /= np.mean(np.sum(F, axis=0), axis=0) #/ X.size
    return F


def AIC(k, chisq, n=1):
    """Calculates the Akaike Information from the number of parameters k
    and the chi-squared of the fit chisq. If n > 1, it modifies the formula to
    account for a small sample size (specified by n).
    """
    if n > 1:
        aic = ((2*(k**2) + 2*k) / (n - k - 1)) + 2*k + chisq
    else:
        aic = 2*k + chisq
    return aic

def BIC(k, n, chisq):
    """Calculates the Bayesian Information from the number of parameters k,
    the sample size n, and the chi-squared of the fit chisq.
    """
    bic = k*np.log(n) + chisq
    return bic


def binning(j, path, Lambda, kind, nbins_x, nbins_y, npars, folder_name=''):
    a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)
    d_dcl = spectral_calc(dc_l, 1.0, o=1, d=0)
    d_dvl = spectral_calc(dv_l, 1.0, o=1, d=0)
    lin_dc = np.fft.ifft(d1k * d1k.size)

    # def find_nearest(array, value):
    #     array = np.asarray(array)
    #     idx = (np.abs(array - value)).argmin()
    #     return idx, array[idx]
    # print(find_nearest(dc_l, 0))
    # print(find_nearest(dv_l, 0))
    # dc_bins =
    # print(dc_l.max())

    # bin_size = 3.5e-3
    # dc_bins = np.arange(dc_l.min(), dc_l.max()+bin_size, bin_size)
    # bin_size = 0.5
    # dv_bins = np.arange(dv_l.min(), dv_l.max()+bin_size, bin_size)
    #
    # print(dc_bins.size, dv_bins.size)

    # print(dc_bins.size)
    # dc_binned = np.digitize(dc_l, dc_bins)
    # print(dc_binned[:10])
    # # print(dc_binned.size)
    # plt.plot(x, dc_binned, c='b')
    # plt.plot(x, dc_l, ls='dashed', c='k')
    #
    # # plt.hist(dc_l, dc_bins)
    # plt.show()

    # nvlines, nhlines = dc_bins.size, dv_bins.size

    nvlines, nhlines = nbins_x, nbins_y
    min_dc, max_dc = dc_l.min(), dc_l.max()
    dc_bins = np.linspace(min_dc, max_dc, nvlines)

    min_dv, max_dv = dv_l.min(), dv_l.max()
    dv_bins = np.linspace(min_dv, max_dv, nhlines)

    mns, meds, counts, inds, yerr, taus, dels, thes, delsq, thesq, delthe, dx_del, dx_the, delcu, thecu, delsqthe, thesqdel, x_binned, lin_dels  = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    count = 0
    for i in range(nvlines-1):
        for j in range(nhlines-1):
            count += 1
            start_coos = (i,j)
            m,n = start_coos
            mns.append([m,n])

            # indices = []
            # for l in range(x.size):
            #     if dc_bins[m] <= dc_l[l] <= dc_bins[m+1] and dv_bins[n] <= dv_l[l] <= dv_bins[n+1]:
            #         indices.append(l)

            indices = [l for l in range(x.size) if dc_bins[m] <= dc_l[l] <= dc_bins[m+1] and dv_bins[n] <= dv_l[l] <= dv_bins[n+1]]

            try:
                left = indices[0]
                right = len(indices)
                inds_ = list(np.arange(left, left+right+1, 1))
                # print(len(inds_))
                # print(len(indices))
                # inds_ = np.sort(indices)
                tau_mean = sum(tau_l[inds_]) / len(inds_)
                delta_mean = sum(dc_l[inds_]) / len(inds_)
                theta_mean = sum(dv_l[inds_]) / len(inds_)
                del_sq_mean = sum((dc_l**2)[inds_]) / len(inds_)
                the_sq_mean = sum((dv_l**2)[inds_]) / len(inds_)
                del_the_mean = sum((dc_l*dv_l)[inds_]) / len(inds_)
                d_dcl_mean = sum((d_dcl)[inds_]) / len(inds_)
                d_dvl_mean = sum((d_dvl)[inds_]) / len(inds_)
                delcu_mean = sum((dc_l**3)[inds_]) / len(inds_)
                thecu_mean = sum((dv_l**3)[inds_]) / len(inds_)
                delsqthe_mean = sum((dv_l*dc_l**2)[inds_]) / len(inds_)
                thesqdel_mean = sum((dc_l*dv_l**2)[inds_]) / len(inds_)
                x_bin = sum(x[inds_]) / len(inds_)
                # print(tau_mean, delta_mean, theta_mean, x_bin)
                taus.append(tau_mean)
                dels.append(delta_mean)
                thes.append(theta_mean)
                delsq.append(del_sq_mean)
                thesq.append(the_sq_mean)
                delthe.append(del_the_mean)
                dx_del.append(d_dcl_mean)
                dx_the.append(d_dvl_mean)
                delcu.append(delcu_mean)
                thecu.append(thecu_mean)
                delsqthe.append(delsqthe_mean)
                thesqdel.append(thesqdel_mean)
                lin_dels.append(sum(lin_dc[inds_]) / len(inds_))

                x_binned.append(x_bin)
                # yerr_ = np.sqrt(sum((tau_l[inds_] - tau_mean)**2) / ((len(inds_)-1) * len(inds_)))
                yerr_ = np.sqrt(sum((tau_l[inds_] - tau_mean)**2) / (len(inds_) - 1))

                # print('poisson err = ', yerr_ / np.sqrt(indices.size))
                yerr.append(yerr_)

                medians = np.mean(inds_)
                meds.append(medians)
                counts.append(count)
            except:
                left = None

    meds, counts = (list(t) for t in zip(*sorted(zip(meds, counts))))
    meds, x_binned = (list(t) for t in zip(*sorted(zip(meds, x_binned))))
    meds, taus = (list(t) for t in zip(*sorted(zip(meds, taus))))
    meds, dels = (list(t) for t in zip(*sorted(zip(meds, dels))))
    meds, thes = (list(t) for t in zip(*sorted(zip(meds, thes))))
    meds, delsq = (list(t) for t in zip(*sorted(zip(meds, delsq))))
    meds, thesq = (list(t) for t in zip(*sorted(zip(meds, thesq))))
    meds, delthe = (list(t) for t in zip(*sorted(zip(meds, delthe))))
    meds, dx_del = (list(t) for t in zip(*sorted(zip(meds, dx_del))))
    meds, dx_the = (list(t) for t in zip(*sorted(zip(meds, dx_the))))
    meds, dx_the = (list(t) for t in zip(*sorted(zip(meds, dx_the))))
    meds, thecu = (list(t) for t in zip(*sorted(zip(meds, dx_the))))
    meds, delsqthe = (list(t) for t in zip(*sorted(zip(meds, delsqthe))))
    meds, thesqdel = (list(t) for t in zip(*sorted(zip(meds, thesqdel))))
    meds, lin_dels = (list(t) for t in zip(*sorted(zip(meds, lin_dels))))


    # # # print(dels, thes)
    # # # taus_f = interp2d(dels, thes, taus, kind='linear')
    # # Nx = 1000
    # # grids = np.arange(0, 0.4, 1/Nx)
    # # grid_x, grid_y = np.meshgrid(grids, grids)
    # # points = (dels, thes)
    # # taus_f = griddata(points, taus, (grid_x, grid_y), method='nearest')
    # # print(taus_f)
    # # # print(, np.where(np.abs(thes) == np.min(np.abs(thes)))[0])
    # indx = int(np.where(np.abs(dels) == min(np.abs(dels)))[0][0])
    # indy = int(np.where(np.abs(thes) == min(np.abs(thes)))[0][0])
    #
    # # print(dels[indx-1:indx+2])
    # # print(thes[indy-1:indy+2])
    # # print(taus[indy-1:indy+2])
    #
    # # # D, T = np.meshgrid(dels, thes)
    # # #
    # # # import seaborn as sns
    # # # hm = sns.heatmap(T,
    # # #              cbar=True,
    # # #              annot=True,
    # # #              square=True,
    # # #              fmt='.3f',
    # # #              annot_kws={'size': 8})
    # # # plt.title('delta')
    # # # plt.tight_layout()
    # # # plt.show()
    # # # print(dels[12:15], thes[12:15])
    # hd = (dels[indx] - dels[indx-1]) #+ (dels[indx] - dels[indx-1])
    # ddtau = (taus[indx] - taus[indx-1]) / (hd)
    #
    # ht = (thes[indy] - thes[indy-1]) #+ (dels[indx] - dels[indx-1])
    # dttau = (taus[indy] - taus[indy-1]) / (ht)
    #
    # print(taus[indx], ddtau, dttau)

    if npars == 1: #constant
        def fitting_function(X, a0):
            return a0 * X[0]
        X = (np.ones(len(dels)))
        X_ = (np.ones(len(x)))
        guesses = 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0 = C
        fit_sp = C0 * X
        fit = C0 * X_

    elif npars == 3: #first-order
        def fitting_function(X, a0, a1, a2):
            return a0*X[0] + a1*X[1] + a2*X[2]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes))
        X_ = (np.ones(len(x)), dc_l, dv_l)
        guesses = 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2 = C
        fit_sp = fitting_function(X, C0, C1, C2)
        fit = fitting_function(X_, C0, C1, C2)

    elif npars == 5: #derivative terms
        def fitting_function(X, a0, a1, a2, a3, a4):
            return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes), np.array(dx_del), np.array(dx_the))
        X_ = (np.ones(len(x)), dc_l, dv_l, d_dcl, d_dvl)
        guesses = 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4 = C
        fit_sp = fitting_function(X, C0, C1, C2, C3, C4)
        fit = fitting_function(X_, C0, C1, C2, C3, C4)


    elif npars == 6: #second-order
        def fitting_function(X, a0, a1, a2, a3, a4, a5):
            return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4] + a5*X[5]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes), np.array(delsq), np.array(thesq), np.array(delthe))
        X_ = (np.ones(len(x)), dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l)
        guesses = 1, 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4, C5 = C
        fit_sp = fitting_function(X, C0, C1, C2, C3, C4, C5)
        fit = fitting_function(X_, C0, C1, C2, C3, C4, C5)

    elif npars == 8: #second-order + derivative terms
        def fitting_function(X, a0, a1, a2, a3, a4, a5, a6, a7):
            return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4] + a5*X[5] + a6*X[6] + a7*X[7]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes), np.array(dx_del), np.array(dx_the), np.array(delsq), np.array(thesq), np.array(delthe))
        X_ = (np.ones(len(x)), dc_l, dv_l, d_dcl, d_dvl, dc_l**2, dv_l**2, dc_l*dv_l)
        guesses = 1, 1, 1, 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4, C5, C6, C7 = C
        fit_sp = fitting_function(X, C0, C1, C2, C3, C4, C5, C6, C7)
        fit = fitting_function(X_, C0, C1, C2, C3, C4, C5, C6, C7)

    elif npars == 10: #second-order + derivative terms
        def fitting_function(X, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
            return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4] + a5*X[5] + a6*X[6] + a7*X[7] + a8*X[8] + a9*X[9]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes), np.array(delsq), np.array(thesq), np.array(delthe), np.array(delcu), np.array(thecu), np.array(delsqthe), np.array(thesqdel))
        X_ = (np.ones(len(x)), dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l, dc_l**3, dv_l**3, dv_l*(dc_l**2), dc_l*(dv_l**2))

        guesses = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4, C5, C6, C7, C8, C9 = C
        fit_sp = fitting_function(X, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9)
        fit = fitting_function(X_, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9)

    elif npars == 12: #second-order + derivative terms
        def fitting_function(X, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11):
            return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4] + a5*X[5] + a6*X[6] + a7*X[7] + a8*X[8] + a9*X[9] + a10*X[10] + a11*X[11]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes), np.array(dx_del), np.array(dx_the), np.array(dx_del)*np.array(dels), np.array(dx_the)*np.array(thes), np.array(dx_del)*np.array(thes), np.array(dx_the)*np.array(dels), np.array(delsq), np.array(thesq), np.array(delthe))
        X_ = (np.ones(len(x)), dc_l, dv_l, d_dcl, d_dvl, d_dcl*dc_l, d_dvl*dv_l, d_dcl*dv_l, d_dvl*dc_l, dc_l**2, dv_l**2, dc_l*dv_l)

        guesses = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11 = C
        fit_sp = fitting_function(X, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11)
        fit = fitting_function(X_, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11)


    else:
        pass

    # print(C[0], C[1], C[2])

    resid = fit_sp - taus
    chisq = sum((resid / yerr)**2)
    red_chi = chisq / (len(dels) - npars)
    # aic = AIC(npars, red_chi, n=1)
    aic = AIC(npars, chisq, n=1)
    bic = BIC(npars, len(taus), chisq)
    # print(a, chisq, red_chi)
    return a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, fit, cov, C, x_binned



def kde_gaussian(x, pos, a, L):
    N = len(x)
    gauss = np.zeros(N)
    for j in range(0, N):
        dist = x - pos[j]
        dist[dist < 0] += L
        dist[dist > L/2] = - L + dist[dist > L/2]
        gauss += np.exp(-a * (dist)**2)
    over = (gauss - np.mean(gauss)) / np.mean(gauss)
    return over

def kde_gaussian_moments(n, x, x_nbody, v_nbody, sm, L):
    N = len(x)
    gauss = np.zeros(N)
    for j in range(0, N):
        dist = x - x_nbody[j]
        dist[dist < 0] += L
        dist[dist > L/2] = - L + dist[dist > L/2]
        gauss += (v_nbody[j]**n) * np.exp(-sm * (dist)**2)
    # over = (gauss - np.mean(gauss)) / np.mean(gauss)
    return gauss

def EFT_sm_kern(k, Lambda):
    #technically, the kernel W is supposed to be normalised such that W(k=0) = 1;
    #however, with the following form, this is already guaranteed.
    kernel = np.exp(- (k ** 2) / (2 * Lambda**2))
    assert kernel[0] == 1, "The mean of W must be 1, please check the code."
    return kernel

def smoothing(field, k, Lambda, kind='gaussian'):
    if kind == 'gaussian':
        kernel = EFT_sm_kern(k, Lambda)
        return np.real(np.fft.ifft(np.fft.fft(field) * kernel))
    elif kind == 'sharp':
        if type(Lambda) != int:
            Lambda = int(Lambda / (2 * np.pi))
        else:
            print('Warning: the given Lambda is an integer, please review the code!')
        field_k = np.fft.fft(field)
        n_trunc = field.size-Lambda
        field_k[Lambda+1:n_trunc] = 0
        return np.real(np.fft.ifft(field_k))
    else:
        raise Exception('kind must be either gaussian (default) or sharp')

def det_is_bet(x_nbody, cell_left, cell_right):
    # left_ind = np.where((x_nbody - cell_left) > 0, x_nbody - cell_left, np.inf).argmin()
    # right_ind = np.where((x_nbody - cell_right) > 0, x_nbody - cell_right, np.inf).argmin()
    ind1 = np.where((x_nbody - cell_left) > 0, x_nbody - cell_left, np.inf).argmin()
    ind2 = np.where((x_nbody - cell_right) > 0, x_nbody - cell_right, np.inf).argmin()
    left_ind = min(ind1, ind2)
    right_ind = max(ind1, ind2)
    inds, vals = is_between(x_nbody[left_ind:right_ind+1], cell_left, cell_right)
    inds += left_ind
    return inds, vals

def is_between(x, x1, x2):
    """returns a subset of x that lies between x1 and x2"""
    indices = []
    values = []
    for j in range(x.size):
        if x1 <= x[j] <= x2:
            indices.append(j)
            values.append(x[j])
    values = np.array(values)
    indices = np.array(indices)
    return indices, values


def dn(n, L, d0):
    den = np.zeros(shape=(n, len(d0)))
    den[0] = d0
    for j in range(0, n-1):
        for m in range(0, j+1):
            den[j+1] += (((j+2) + 1/2) * (spectral_calc(den[m], L, d=0, o=1) * spectral_calc(den[j - m], L, d=1, o=1))) + (((j+2) + 3/2) * (den[j - m] * den[m])) + ((spectral_calc(den[m], L, d=1, o=1) * spectral_calc(den[j - m], L, d=0, o=1)))
        den[j+1] *= (2 / ((2*(j+2) + 3) * (j+1)))
    return den

# def dn(n, L, d0):
#     x = np.arange(0, L, L/d0.size)
#     den = np.zeros(shape=(n, len(d0)))
#     den[0] = d0
#     for j in range(0, n-1):
#         for m in range(0, j+1):
#             den[j+1] += (((j+2) + 1/2) * (np.gradient(den[m], x, edge_order=2) * np.trapz(den[j - m], x))) + (((j+2) + 3/2) * (den[j - m] * den[m])) + ((np.trapz(den[m], x) * np.gradient(den[j - m], x, edge_order=2)))
#         den[j+1] *= (2 / ((2*(j+2) + 3) * (j+1)))
#     return den

# def dn(n, L, d0):
#     x = np.arange(0, L, L/d0.size)
#     den = np.zeros(shape=(n, len(d0)))
#     den[0] = d0
#     for j in range(0, n-1):
#         for m in range(0, j+1):
#             den[j+1] += (((j+2) + 1/2) * (spectral_calc(den[m], L, d=0, o=1) * simpson(den[j - m], x))) + (((j+2) + 3/2) * (den[j - m] * den[m])) + ((simpson(den[m], x) * spectral_calc(den[j-m], L, d=0, o=1)))
#         den[j+1] *= (2 / ((2*(j+2) + 3) * (j+1)))
#     return den

def write_to_hdf5(filename, d, a_list, k, n, sm):
    with h5py.File(filename, mode='w') as hdf:
        hdf.create_dataset('a, a_sch', data=a_list)
        hdf.create_dataset('den_k2_{}spt'.format(n), data=d[0])
        hdf.create_dataset('den_k2_sch', data=d[1])
        # hdf.create_dataset('den_k2_nbody', data=d[1])
        # hdf.create_dataset('den_k2_zel', data=d[3])
        hdf.create_dataset('k', data=k)
        hdf.create_dataset('smoothing', data=sm)
        print('writing done!')

def phase(nd, L, H0, a0):
    V = spectral_calc(nd, L, o=2, d=1)
    return V * H0 * np.sqrt(a0)

def moment(f, p, dp, n):
    if type(n) != int:
        raise Exception('n must be an integer')
    else:
        f1 = f * (p ** n)
        m = np.trapz(f1, dx=dp, axis=0)
        return m

# def interp_green(Pn, Qn, a_list, da_new, C, simpsons=True, final_interp=True):
#     a_new = np.arange(a_list[0], a_list[-1], da_new)
#
#     An_func = interp1d(x=a_list, y=Pn, kind='linear')
#     Bn_func = interp1d(x=a_list, y=Qn, kind='linear')
#
#     An_interp = An_func(a_new)
#     Bn_interp = Bn_func(a_new)
#
#     if simpsons == True:
#         An = np.zeros(a_new.size)
#         Bn = np.zeros(a_new.size)
#         for j in range(1, a_new.size):
#             An[j] = simpson(An_interp[:j], a_new[:j])
#             Bn[j] = simpson(Bn_interp[:j], a_new[:j])
#         An /= (a_new**(5/2))
#         alpha_c = C * (An - Bn)
#
#     else:
#         for j in range(1, a_new.size):
#             An_interp[j] += An_interp[j-1]
#             Bn_interp[j] += Bn_interp[j-1]
#
#         An_interp /= (a_new**(5/2))
#         alpha_c = C * (An_interp - Bn_interp) * (da_new)
#
#     if final_interp == True:
#         alpha_c_interp = interp1d(x=a_new, y=alpha_c, kind='linear')
#         alpha_c = alpha_c_interp(a_list[1:-1])
#         a_new = a_list[1:-1]
#
#     return a_new, alpha_c

def write_vlasov_ic(f, x, p, H0, m, a0, da, an, loc, N_out, A):
    print('Writing ICs to file...\n')
    filename = str(loc) + 'ICs.hdf5'
    params = [H0, m, a0, da, an, N_out]
    with h5py.File(filename, 'w') as hdf:
        hdf.create_dataset('A', data=A)
        hdf.create_dataset('f', data=f)
        hdf.create_dataset('x', data=x)
        hdf.create_dataset('p', data=p)
        hdf.create_dataset('params', data=params)
        hdf.create_dataset('loc', data=loc)

def vla_ic_plotter(f, x, v, v_zel, a, run):
    fig, ax = plt.subplots()
    ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=12)
    ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=12)
    title = ax.text(0.05, 0.9, 'a = {}'.format(str(np.round(a, 3))),  bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
    plot2d_2 = ax.pcolormesh(x, v, f, shading='auto', cmap='inferno')
    # ax.scatter(pos, vel, color='w', alpha=0.7, s=5, label='N-body')
    ax.plot(x, v_zel, color='r', ls='dashed', label='Zel')

    ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
    c = fig.colorbar(plot2d_2, fraction=0.15)
    c.set_label(r'$f_{H}$', fontsize=20)

    ax.set_ylim(-100, 100)
    ax.legend(loc='upper right')
    legend = ax.legend(frameon = 1, loc='upper right', fontsize=12)
    frame = legend.get_frame()
    plt.tight_layout()
    frame.set_edgecolor('white')
    frame.set_facecolor('black')
    for text in legend.get_texts():
        plt.setp(text, color = 'w')
    plt.savefig('/vol/aibn31/data1/mandar/plots/' + str(run) + 'IC.png')
    plt.close()

def read_vlasov_ic(loc):
    with h5py.File(loc + 'ICs.hdf5', 'r') as hdf:
        ls = list(hdf.keys())
        A = np.array(hdf.get(str(ls[0])))
        f = np.array(hdf.get(str(ls[1])))
        p = np.array(hdf.get(str(ls[3])))
        H0, m, a0, da, an, N_out = np.array(hdf.get(str(ls[4])))
        x = np.array(hdf.get(str(ls[5])))
    return f, x, p, H0, m, a0, da, an, N_out, A


def assign_weight(pos, x_grid):
  assert pos >= 0
  dx_grid = x_grid[1] - x_grid[0]
  diff = np.abs(pos - x_grid)
  ngp_i = int(np.where(diff == np.min(diff))[0])
  ngp = float(x_grid[ngp_i])
  W1 = 1 - (diff[ngp_i] / dx_grid)
  W2 = 1 - W1
  return ngp_i, W1, W2


def write_density(path, file_num, Nfiles, dx_grid):
    filepath = path + '/moments/'
    try:
        os.makedirs(filepath, 0o755)
        print('Path doesn\'t exist, new directory created.')

    except:
        print('Path exists, writing files...')
        pass

    for i in range(file_num, Nfiles):
        print("Reading file {} of {}".format(i+1, Nfiles))
        # t0 = time.time()
        nbody_filename = 'output_{0:04d}.txt'.format(i)
        nbody_file = np.genfromtxt(path + nbody_filename)

        moments_filename = 'output_hierarchy_{0:04d}.txt'.format(i)
        moments_file = np.genfromtxt(path + moments_filename)
        a = moments_file[:,-1][0]
        print('a = ', a)
        x_nbody = nbody_file[:,-1]
        v_nbody = nbody_file[:,2]

        par_num = x_nbody.size
        L = 1.0
        x_grid = np.arange(0, L, dx_grid)
        N = x_grid.size
        k_grid = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-N/2, N/2))

        if dx_grid > 0.1:
            prod = np.outer(1j * k_grid, x_nbody)
            par_ = np.exp(prod)
            dk_par = np.sum(par_, axis=1)

        else:
            dk_par = np.zeros(x_grid.size, dtype=complex)
            for j in range(par_num):
                dk_par += np.exp(1j * k_grid * x_nbody[j]) #there is a minus sign for nbody_phase_inv


        filename = 'M0_{0:04d}.hdf5'.format(i)
        file = h5py.File(filepath+filename, 'w')
        file.create_group('Header')
        header = file['Header']
        header.attrs.create('a', a, dtype=float)
        header.attrs.create('dx', dx_grid, dtype=float)

        moments = file.create_group('Moments')
        moments.create_dataset('dk_nbody', data=dk_par)

        file.close()
        print("Done!\n")
        # t1 = time.time()
        # del_t = t1-t0
        # print('Time taken = {}s'.format(np.round(del_t, 6)))

    return None

def read_density(path, file_num):
   filename = path + '/moments/M0_{0:04d}.hdf5'.format(file_num)
   file = h5py.File(filename, mode='r')
   header = file['/Header']
   a = header.attrs.get('a')
   dx = header.attrs.get('dx')

   moments = file['/Moments']
   M0 = np.array(moments['dk_nbody'])

   file.close()

   return M0, a, dx

def SPT(dc_in, L, a):
   """Returns the SPT PS upto 2-loop order"""
   Nx = dc_in.size
   F = dn(5, L, dc_in)
   d1k = (np.fft.fft(F[0]) / Nx)
   d2k = (np.fft.fft(F[1]) / Nx)
   d3k = (np.fft.fft(F[2]) / Nx)
   d4k = (np.fft.fft(F[3]) / Nx)
   d5k = (np.fft.fft(F[4]) / Nx)

   P11 = (d1k * np.conj(d1k)) * (a**2)
   P12 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3)
   P22 = (d2k * np.conj(d2k)) * (a**4)
   P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
   P14 = ((d1k * np.conj(d4k)) + (d4k * np.conj(d1k))) * (a**5)
   P23 = ((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a**5)
   P33 = (d3k * np.conj(d3k)) * (a**6)
   P15 = ((d1k * np.conj(d5k)) + (d5k * np.conj(d1k))) * (a**6)
   P24 = ((d2k * np.conj(d4k)) + (d4k * np.conj(d2k))) * (a**6)

   P_lin = P11
   P_1l = P_lin + P12 + P13 + P22
   P_2l = P_1l + P14 + P15 + P23 + P24 + P33
   return d1k, d2k, np.real(P_1l), np.real(P_2l)

def SPT_sm(dc_in, k, L, Lambda, a):
  """Returns the smoothed SPT PS upto 1-loop order"""
  Nx = k.size
  F = dn(5, L, dc_in)
  W_EFT = EFT_sm_kern(k, Lambda)
  #smoothing the overdensity solutions
  d1k = (np.fft.fft(F[0]) / Nx) * W_EFT
  d2k = (np.fft.fft(F[1]) / Nx) * W_EFT
  d3k = (np.fft.fft(F[2]) / Nx) * W_EFT
  d4k = (np.fft.fft(F[3]) / Nx) * W_EFT
  d5k = (np.fft.fft(F[4]) / Nx) * W_EFT

  P11 = (d1k * np.conj(d1k)) * (a**2)
  P12 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3)
  P22 = (d2k * np.conj(d2k)) * (a**4)
  P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
  P14 = ((d1k * np.conj(d4k)) + (d4k * np.conj(d1k))) * (a**5)
  P23 = ((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a**5)
  P33 = (d3k * np.conj(d3k)) * (a**6)
  P15 = ((d1k * np.conj(d5k)) + (d5k * np.conj(d1k))) * (a**6)
  P24 = ((d2k * np.conj(d4k)) + (d4k * np.conj(d2k))) * (a**6)

  P_lin = P11
  P_1l = P_lin + P12 + P13 + P22
  P_2l = P_1l + P14 + P15 + P23 + P24 + P33
  return d1k, np.real(P_1l), np.real(P_2l)

def SPT_real_sm(dc_in, k, L, Lambda, a, kind):
    """Returns the real-space smoothed SPT density upto third order"""
    F = dn(3, L, dc_in)

    if kind == 'sharp':
        W_EFT = np.ones(k.size)
        Lambda = np.int(Lambda/(2*np.pi))
        W_EFT[Lambda+1:] = 0

    elif kind == 'gaussian':
        W_EFT = EFT_sm_kern(k, Lambda)

    else:
        raise Exception('kind must be either gaussian (default) or sharp')

    d1k = (np.fft.fft(F[0])) * W_EFT
    d2k = (np.fft.fft(F[1])) * W_EFT
    d3k = (np.fft.fft(F[2])) * W_EFT

    dk_spt = a*d1k + (a**2)*d2k + (a**3)*d3k
    dc_spt = np.real(np.fft.ifft(dk_spt))

    return dc_spt

def SPT_real_tr(dc_in, k, L, Lambda, a, kind):
    """Returns the real-space truncated SPT density upto third order"""
    dc_in = smoothing(dc_in, k, Lambda, kind) #truncating the initial overdensity
    F = dn(3, L, dc_in)
    d1k = (np.fft.fft(F[0]))
    d2k = (np.fft.fft(F[1]))
    d3k = (np.fft.fft(F[2]))

    dk_spt = a*d1k + (a**2)*d2k + (a**3)*d3k
    dc_spt = np.real(np.fft.ifft(dk_spt))

    return dc_spt


def SPT_tr(dc_in, k, L, Lambda, kind, a):
  """Returns the truncated SPT PS upto 1-loop order"""
  dc_in = smoothing(dc_in, k, Lambda, kind) #truncating the initial overdensity
  Nx = k.size
  F = dn(5, L, dc_in)
  d1k = (np.fft.fft(F[0]) / Nx)
  d2k = (np.fft.fft(F[1]) / Nx)
  d3k = (np.fft.fft(F[2]) / Nx)
  d4k = (np.fft.fft(F[3]) / Nx)
  d5k = (np.fft.fft(F[4]) / Nx)

  P11 = (d1k * np.conj(d1k)) * (a**2)
  P12 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3)
  P22 = (d2k * np.conj(d2k)) * (a**4)
  P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
  P14 = ((d1k * np.conj(d4k)) + (d4k * np.conj(d1k))) * (a**5)
  P23 = ((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a**5)
  P33 = (d3k * np.conj(d3k)) * (a**6)
  P15 = ((d1k * np.conj(d5k)) + (d5k * np.conj(d1k))) * (a**6)
  P24 = ((d2k * np.conj(d4k)) + (d4k * np.conj(d2k))) * (a**6)

  P_lin = P11
  P_1l = P_lin + P12 + P13 + P22
  P_2l = P_1l + P14 + P15 + P23 + P24 + P33
  return d1k, d2k, np.real(P_1l), np.real(P_2l)


def alpha_to_corr(alpha_c, a, x, k, L, dc_in, Lambda, kind):#, err_Int):
    dc_in = smoothing(dc_in, k, Lambda, kind) #truncating the initial overdensity
    F = dn(3, L, dc_in)
    d1k = (np.fft.fft(F[0]))
    d2k = (np.fft.fft(F[1]))
    d3k = (np.fft.fft(F[2]))
    d3k_corr = alpha_c * (k**2) * d1k * a
    dc_k3 = a*d1k + (a**2)*d2k + (a**3)*(d3k)
    den_k = dc_k3 + d3k_corr

    dc_eft = np.real(np.fft.ifft(den_k))
    # err_eft = np.fft.ifft(err_Int * (k**2 * d1k * a))
    return dc_eft #, err_eft

def plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc=0, which='', title_str='', error_plotting=True, terr=[], zel=False, save=False, leg=True, texts=[], flags=[], dashes=[]):
    if error_plotting == False:
        plt.rcParams.update({"text.usetex": True})
        plt.rcParams.update({"font.family": "serif"})
        fig, ax = plt.subplots()

        ax.set_title(title_str, fontsize=16)

        if xlabel == '':
            ax.set_xlabel(r'$a$', fontsize=16)
        else:
            ax.set_xlabel(xlabel, fontsize=16)

        ax.set_ylabel(ylabel, fontsize=16)

        for i in range(len(yaxes)):
            ax.plot(xaxis, yaxes[i], c=colours[i], ls=linestyles[i], lw=2.5, label=labels[i])
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', direction='in', labelleft=False, labelright=True)
        ax.ticklabel_format(scilimits=(-2, 3))
        # ax.grid(lw=0.2, ls='dashed', color='grey')
        ax.yaxis.set_ticks_position('both')
        ax.axvline(a_sc, c='g', lw=1, label=r'$a_{\mathrm{sc}}$')
        ax.legend(fontsize=11)#, loc=2, bbox_to_anchor=(1,1))
        # plt.savefig('../plots/{}/{}.png'.format(plots_folder, savename), bbox_inches='tight', dpi=150)
        plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
        plt.close()
        # plt.show()

    else:
        if zel == True:
            errors = [(yaxis - yaxes[0]) * 100 / yaxes[0] for yaxis in yaxes[:-2]]
            err_zel = (yaxes[-1] - yaxes[0][:int(yaxes[-1].size)]) * 100 / yaxes[0][:int(yaxes[-1].size)]
            errors.append(err_zel)
        else:
            errors = [(yaxis - yaxes[0]) * 100 / yaxes[0] for yaxis in yaxes]

        plt.rcParams.update({"text.usetex": True})
        plt.rcParams.update({"font.family": "serif"})
        handles = []
        fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [3, 1]})
        ax[0].set_title(title_str, fontsize=20)

        ax[1].set_xlabel(xlabel, fontsize=20)
        ax[0].set_ylabel(ylabel, fontsize=20)
        ax[1].set_ylabel(r'$\%$ err', fontsize=20)

        for i in range(len(yaxes)-int(zel)):
            if dashes[i] is not None:
                line, = ax[0].plot(xaxis, yaxes[i], c=colours[i], ls=linestyles[i], dashes=dashes[i], lw=2.5)#, label=labels[i])
            else:
                line, = ax[0].plot(xaxis, yaxes[i], c=colours[i], ls=linestyles[i], lw=2.5)#, label=labels[i])
            handles.append(line)
            if i == 0:
                ax[1].axhline(0, c=colours[0])
            elif i > 0:
                if dashes[i] is not None:
                    ax[1].plot(xaxis, errors[i-int(zel)], ls=linestyles[i-int(zel)], lw=2.5, c=colours[i-int(zel)], dashes=dashes[i])
                else:
                    ax[1].plot(xaxis, errors[i-int(zel)], ls=linestyles[i-int(zel)], lw=2.5, c=colours[i-int(zel)])

        if zel == True:
            ax[0].plot(xaxis[:int(yaxes[-1].size)], yaxes[-1], ls=linestyles[-1], lw=2.5, c=colours[-1], label=labels[-1])
            ax[1].plot(xaxis[:int(errors[-1].size)], errors[-1], ls=linestyles[-1], lw=2.5, c=colours[-1])


        # if len(terr) != 0:
        #     fill_line = ax[0].fill_between(xaxis, yaxes[5]-terr, yaxes[5]+terr, color='darkslategray', alpha=0.5)
        #     terr_err = terr * 100 / yaxes[0]
        #     ax[1].fill_between(xaxis, errors[5]-terr_err, errors[5]+terr_err, color='darkslategray', alpha=0.5)


        for i in range(2):
            ax[i].minorticks_on()
            ax[i].tick_params(axis='both', which='both', direction='in', labelsize=15)
            if a_sc != 0:
                ax[i].axvline(a_sc, c='teal', lw=0.5, ls='dashed', label=r'$a_{\mathrm{shell}}$')
            # ax[i].ticklabel_format(scilimits=(-2, 3))
            # ax[i].grid(lw=0.2, ls='dashed', color='grey')
            ax[i].yaxis.set_ticks_position('both')
        # ax[1].set_ylim(-0.5, 0.5)

        if len(flags) != 0:
            Nfiles = 51
            labels.append(r'$a_{\mathrm{shell}}$')
            for j in range(Nfiles):
                if flags[j] == 1:
                    sc_line = ax[0].axvline(xaxis[j], c='teal', ls='dashed', lw=0.5, zorder=1)
                    ax[1].axvline(xaxis[j], c='teal', ls='dashed', lw=0.5, zorder=1)

                else:
                    pass
            handles.append(sc_line)

        # sc_line = ax[0].axvline(1.81818, c='teal', ls='dashed', lw=1, zorder=1)#, label=r'$a_{\mathrm{shell}}$')
        # ax[1].axvline(1.8181, c='teal', ls='dashed', lw=1, zorder=1)
        # handles.append(sc_line)
        # labels.append(r'$a_{\mathrm{shell}}$')

        # knl_line = ax[0].axvline(2.26, c='magenta', ls='dashed', lw=1, zorder=1)#, label=r'$k_{\mathrm{NL}} / k_{\mathrm{f}} = 11$')
        # ax[1].axvline(2.26, c='magenta', ls='dashed', lw=1, zorder=1)

        # handles.append(knl_line)
        # labels.append(r'$k_{\mathrm{NL}} / k_{\mathrm{f}} = 11$')




        if leg == True:# & (len(terr) != 0):
            # handles[-2] = (handles[-2], fill_line)
            # ax[0].legend(fontsize=13)#, loc=2, bbox_to_anchor=(1,1))
            ax[0].legend(handles, labels, fontsize=12, framealpha=1, loc=3)#, ncol=2)
        else:
            pass
        # ax[1].set_ylim(-10, 10)

        if len(texts) != 0:
            text_str = texts[0]
            x,y = texts[1]
            ax[0].text(x, y, text_str, bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12, transform=ax[0].transAxes)

        plt.subplots_adjust(hspace=0)

        if save == False:
            plt.show()
        else:
            plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
            # plt.savefig('../plots/{}/{}.png'.format(plots_folder, savename), bbox_inches='tight', dpi=150)

            plt.close()

def plotter2(mode, Lambda, xaxes, yaxes, ylabel, colours, labels, linestyles, plots_folder, savename, errors, a_sc, which='', xlabel='', title_str='', handles=[], handles2=[], sm=False, save=False):
    err_x, err_y, err_c, err_ls = errors
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})

    fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [3, 1]})
    # ax[0].set_title(r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(mode, int(Lambda/(2*np.pi))), fontsize=12) #16 for pdf
    ax[1].set_xlabel(r'$a$', fontsize=22) #16 for pdf
    ax[0].set_ylabel(ylabel, fontsize=22) #16 for pdf
    ax[0].set_title(title_str, fontsize=22)
    ax[1].axhline(0, c='cyan', lw=2.5)
    ax[1].set_ylabel(r'$\%$ err', fontsize=22) #16 for pdf
    for i in range(len(yaxes)):
        ax[0].plot(xaxes[i], yaxes[i], c=colours[i], ls=linestyles[i], lw=2.5, zorder=i)#, label=labels[i])
        # if i == 1:
        # ax[1].plot(xaxes[1], errors[0], c=colours[1], ls=linestyles[1], lw=2.5) #if varying \Lambda, then it's i>2 and i-3, if varying simulations, then it's i>3 and i-4
        # #error[i-3] and if i > 2: for all sims
        if i < len(err_x):
            ax[1].plot(err_x[i], err_y[i], c=err_c[i], ls=err_ls[i], lw=2.5)
        else:
            pass

    if handles == []:
        ax[0].legend(fontsize=16, loc=3)#, bbox_to_anchor=(1,1))
    else:
        ax[0].legend(handles=handles2, fontsize=16, loc=3)
        # leg1 = plt.legend(handles=handles, fontsize=16, loc=3, bbox_to_anchor=(1,3))
        # plt.gca().add_artist(leg1)

        # plt.gca().add_artist(leg2)

    for i in range(2):
        ax[i].minorticks_on()
        # ax[i].axvline(a_sc, c='g', lw=1, label=r'$a_{\mathrm{sc}}$')
        ax[i].tick_params(axis='both', which='both', direction='in', labelsize=16)
        # ax[i].ticklabel_format(scilimits=(-2, 3))
        # ax[i].grid(lw=0.1, ls='dashed', color='grey')
        ax[i].yaxis.set_ticks_position('both')

    plt.subplots_adjust(hspace=0)
    fig.align_labels()
    # ax[0].set_ylim(5.45, 6.29)
    if save == False:
        plt.show()
    else:
        plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
        # plt.savefig('../plots/{}/{}.png'.format(plots_folder, savename), bbox_inches='tight', dpi=150)
        plt.close()


def weighted_ls_fit(d, x, Np, d_cov):
    """Fits a multivariate linear model to a data vector.
    d: the data vector
    x: a tuple of size Np containing the fields used for the fit
    Np: number of parameters to be used for the fit
    d_cov: the covariance matrix of the data
    Returns the parameters of the fit, and the covariance and correlation matrices.
    """
    assert len(x) == Np, "x must be a tuple of size Np"

    W = d_cov
    y = np.array([sum(d * field) for field in x])
    X = np.empty(shape=(Np, Np))

    for j in range(Np):
        X[j] = np.array([sum(field*x[j]) for field in x])

    A = np.dot(np.linalg.inv(W), X)
    cov = np.linalg.inv(np.dot(X.T, A))
    B = np.dot(np.linalg.inv(W), y)
    params = np.dot(cov, np.dot(X.T, B))

    corr = np.zeros(cov.shape)
    for i in range(Np):
        corr[i,:] = [cov[i,j] / np.sqrt(cov[i,i]*cov[j,j]) for j in range(Np)]

    return params, cov, corr

def lmfit_est(d, X, err, d_cov):
    # create data to be fitted
    def model(x0, x1, x2, params):
        a0 = params['C0']
        a1 = params['C1']
        a2 = params['C2']

        return a0*x0 + a1*x1 + a2*x2

    def sq_residuals(params, d, X, err, d_cov):
        """Returns the sum of square residuals from the data, input, and errors."""
        weights = 1/err**2
        x0, x1, x2 = X
        resid = np.array([weights[i] * (d[i] - model(x0[i], x1[i], x2[i], params)) for i in range(x1.size)])
        # # print(d_cov.shape, resid.shape)
        # A = np.dot(d_cov, resid)
        # inv_resid = np.linalg.inv(resid)
        # B = np.dot(inv_resid, A)
        # print(B)
        # print(A.shape, B.shape)
        return resid

    # create a set of Parameters
    params = Parameters()
    bst = 1
    params.add('C0', value=1)
    params.add('C1', value=1)
    params.add('C2', value=1)

    # do the fit, here with the default leastsq algorithm
    minner = Minimizer(sq_residuals, params, fcn_args=(d, X, err, d_cov))
    result = minner.minimize('leastsq')
    # print(result.errorbars)

    cov = result.covar
    par = result.params

    # print(par["C0"].stderr)
    # print(par["C1"].stderr)
    # print(par["C2"].stderr)
    red_chi = result.redchi
    corr = np.zeros(cov.shape)
    for i in range(3):
        corr[i,:] = [cov[i,j] / np.sqrt(cov[i,i]*cov[j,j]) for j in range(3)]

    p = [par['C0'].value, par['C1'].value, par['C2'].value]
    return p, cov, corr, red_chi


def dc_in_finder(path, x, interp=False):
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(0)
    moments_file = np.genfromtxt(path + moments_filename)
    a0 = moments_file[:,-1][0]

    initial_file = np.genfromtxt(path + 'output_initial.txt')
    q = initial_file[:,0]
    Psi = initial_file[:,1]

    k_in = np.fft.ifftshift(2.0 * np.pi / q[-1] * np.arange(-q.size/2, q.size/2))
    dc_in = -spectral_calc(Psi, 1.0, o=1, d=0) / a0

    if interp == True:
        f = interp1d(q, dc_in, kind='cubic', fill_value='extrapolate')
        dc_in = f(x)
        k_in = np.fft.ifftshift(2.0 * np.pi / x[-1] * np.arange(-x.size/2, x.size/2))
        return dc_in, k_in

    else:
        return dc_in, k_in

def write_sim_data(j, Lambda, path, A, kind, mode, read_folder_name='', write_folder_name=''):
    # t0 = time.time()
    # from EFT_ens_solver import EFT_solve
    # a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = EFT_solve(j, Lambda, path, A, kind)

    from EFT_hier_solver import EFT_solve
    a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = EFT_solve(j, Lambda, path, kind, read_folder_name)

    if write_folder_name == '':
        filepath = path + '/data_{}/L{}/'.format(kind, int(Lambda/(2*np.pi)))
    else:
        filepath = path + write_folder_name

    try:
        os.makedirs(filepath, 0o755)
        print('Path doesn\'t exist, new directory created.')

    except:
        # print('Path exists, writing file...')
        pass

    filename = filepath + '/sol_{0:04d}.hdf5'.format(j)


    file = h5py.File(filename, 'w')
    header = file.create_group('Header')
    fields = file.create_group('Fields')
    header.attrs.create('a', a, dtype=float)
    fields.create_dataset('tau_l', data=tau_l)

    # if phase == False:
    header.attrs.create('mode', mode, dtype=float)
    header.attrs.create('Lambda', Lambda, dtype=float)
    header.attrs.create('smoothing', kind)
    fields.create_dataset('d1k', data=d1k)
    fields.create_dataset('dc_l', data=dc_l)
    fields.create_dataset('dv_l', data=dv_l)
    fields.create_dataset('P_nb', data=P_nb)
    fields.create_dataset('P_1l', data=P_1l)

    fields.create_dataset('x', data=x)
    # else:
    #     pass
    #
    # # print('Done!')
    # # t1 = time.time()
    # # print('This took {}s\n'.format(np.round(t1-t0, 6)))
    return None


def read_sim_data(path, Lambda, kind, j, folder_name=''):
    try:
        run = int(path[-3:-1])
    except:
        run = int(path[-2:-1])

    if folder_name == '':
        filename = path + '/data_{}/L{}/'.format(kind, int(Lambda/(2*np.pi))) + '/sol_{0:04d}.hdf5'.format(j)
    else:
        filename = path + folder_name + '/sol_{0:04d}.hdf5'.format(j)

    file = h5py.File(filename, mode='r')
    header = file['/Header']
    fields = file['/Fields']

    a = header.attrs.get('a')
    tau_l = np.array(fields['tau_l'])

    # if run == 1:
    mode = header.attrs.get('mode')
    Lambda = header.attrs.get('Lambda')
    kind = header.attrs.get('smoothing')
    d1k = np.array(fields['d1k'])
    dc_l = np.array(fields['dc_l'])
    dv_l = np.array(fields['dv_l'])
    x = np.array(fields['x'])
    P_nb = np.array(fields['P_nb'])
    P_1l = np.array(fields['P_1l'])

    file.close()
    return a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l
    # else:
    #     file.close()
    #     return a, tau_l

# def param_calc_ens(j, Lambda, path, A, mode, kind, n_runs, n_use, folder_name='', fitting_method='curve_fit', nbins_x=10, nbins_y=10, npars=3, fde_method='percentile', per=43, ens=False):
#     a, x, d1k, dc_l_0, dv_l_0, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)

#     # if ens == True:
#     #     taus = []
#     #     taus.append(tau_l_0)
#     #     for run in range(1, n_runs+1):
#     #         path = path[:-2] + '{}/'.format(run)
#     #         sol = read_sim_data(path, Lambda, kind, j, folder_name)
#     #         taus.append(sol[-3])

#     #     Nt = len(taus)

#     #     tau_l = sum(np.array(taus)) / Nt

#     # else:
#     #     tau_l = tau_l_0
#     #     taus = []

#     H0 = 100
#     rho_0 = 27.755
#     rho_b = rho_0 / a**3
#     H = a**(-1/2)*H0
#     dv_l_0 = -dv_l_0 / (H)

#     taus, dels, thes = [], [], []
#     taus.append(tau_l_0)
#     dels.append(dc_l_0)
#     thes.append(dv_l_0)

#     for run in range(1, n_runs+1):
#         path = path[:-2] + '{}/'.format(run)
#         sol = read_sim_data(path, Lambda, kind, j, folder_name)
#         dels.append(sol[3])
#         thes.append(-sol[4] / H)
#         taus.append(sol[5])

#     Nt = len(taus)

#     tau_l = sum(np.array(taus)) / Nt
#     dc_l = sum(np.array(dels)) / Nt
#     dv_l = sum(np.array(thes)) / Nt


#     if fitting_method == 'curve_fit':
#         n_use = 10
#         n_ev = x.size // n_use
#         if ens == True:
#             diff = np.array([(taus[i] - tau_l)**2 for i in range(1, Nt)])
#             yerr = np.sqrt(sum(diff) / (Nt*(Nt-1)))
#             yerr_sp = yerr[0::n_ev]

#         else:
#             yerr_sp = None

#         dc_l_sp = dc_l[0::n_ev]
#         dv_l_sp = dv_l[0::n_ev]
#         tau_l_sp = tau_l[0::n_ev]
#         x_sp = x[0::n_ev]

#         def fitting_function(X, a0, a1, a2):
#             x1, x2 = X
#             return a0 + a1*x1 + a2*x2

#         guesses = 1, 1, 1 #, 1, 1, 1
#         C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)

#         fit = fitting_function((dc_l, dv_l), C[0], C[1], C[2])#, C[3], C[4], C[5])
#         fit_sp = fit[0::n_ev]

#         resid = fit_sp - tau_l_sp
#         chisq = 1#sum((resid / yerr_sp)**2)
#         red_chi = 1#chisq / (n_use - 3)
#         yerr = 0

#         # print('C1: ', C[1]+C[2])
#         cs2 = np.real(C[1] / rho_b)
#         cv2 = -np.real(C[2] * H0 / (rho_b * np.sqrt(a)))
#         ctot2 = (cs2 + cv2)

#         f1 = (1 / rho_b)
#         f2 = (-H0 / (rho_b * np.sqrt(a)))

#         # cov[0,1] *= f1
#         # cov[1,0] *= f1
#         # cov[0,2] *= f2
#         # cov[2,0] *= f2
#         # cov[1,1] *= f1**2
#         # cov[2,2] *= f2**2
#         # cov[2,1] *= f2*f1
#         # cov[1,2] *= f1*f2
#         #
#         # corr = np.zeros(cov.shape)
#         #
#         # for i in range(3):
#         #     for j in range(3):
#         #         corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
#         #
#         # err0, err1, err2 = np.sqrt(np.diag(cov))[:3]
#         # terr = np.sqrt(err1**2 + err2**2 + corr[1,2]*err1*err2 + corr[2,1]*err2*err1)
#         err0, err1, err2 = 0, 0, 0
#         terr = 0
#         x_binned = None

#     else:
#         a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, fit, cov, C, x_binned = binning(j, path, Lambda, kind, nbins_x, nbins_y, npars, folder_name=folder_name)

#         resid = fit_sp - taus
#         chisq = sum((resid / yerr)**2)
#         red_chi = chisq / (len(dels) - npars)

#         C0, C1, C2 = C[:3]
#         cs2 = np.real(C1 / rho_b)
#         cv2 = -np.real(C2 * H0 / (rho_b * np.sqrt(a)))
#         ctot2 = (cs2 + cv2)

#         f1 = (1 / rho_b)
#         f2 = (-H0 / (rho_b * np.sqrt(a)))

#         cov[0,1] *= f1
#         cov[1,0] *= f1
#         cov[0,2] *= f2
#         cov[2,0] *= f2
#         cov[1,1] *= f1**2
#         cov[2,2] *= f2**2
#         cov[2,1] *= f2*f1
#         cov[1,2] *= f1*f2

#         corr = np.zeros(cov.shape)

#         # for i in range(cov.shape[0]):
#         #     for j in range(cov.shape[1]):
#         #         corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
#         #         # corr[i,j] = cov[i,j] / np.sqrt(np.abs(cov[i,i]*cov[j,j]))

#         # err0, err1, err2, err3, err4, err5 = np.sqrt(np.abs(np.diag(cov)))
#         # err0, err1, err2 = (np.sqrt(np.diag(cov)))[:3]

#         err0, err1, err2 = np.sqrt(np.diag(cov[:3,:3]))
#         corr[1,2] = cov[1,2] / np.sqrt(cov[1,1]*cov[2,2])
#         corr[2,1] = cov[2,1] / np.sqrt(cov[1,1]*cov[2,2])

#         ctot2 = (cs2 + cv2)
#         terr = np.sqrt(err1**2 + err2**2 + corr[1,2]*err1*err2 + corr[2,1]*err2*err1)
#         # print(ctot2, terr)

#     # M&W Estimator
#     Lambda_int = int(Lambda / (2*np.pi))
#     tau_l_k = np.fft.fft(tau_l_0) / x.size
#     num = (np.conj(a * d1k) * ((np.fft.fft(tau_l_0)) / x.size))
#     denom = ((d1k * np.conj(d1k)) * (a**2))
#     ntrunc = int(num.size-Lambda_int)
#     num[Lambda_int+1:ntrunc] = 0
#     denom[Lambda_int+1:ntrunc] = 0

#     ctot2_2 = np.real(sum(num) / sum(denom)) / rho_b

#     T = -dv_l / (H0 / (a**(1/2)))

#     def Power_fou(f1, f2):
#         f1_k = np.fft.fft(f1)
#         f2_k = np.fft.fft(f2)
#         corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
#         return corr[1]

#     ctot2_3 = np.real(Power_fou(tau_l_0/rho_b, dc_l) / Power_fou(dc_l, T))

#     # if fde_method == 'algorithm':
#     #     sol_deriv = deriv_param_calc(dc_l, dv_l, tau_l, a)
#     #     ctot2_4 = sol_deriv[0][1] / rho_b
#     #     err_4 = sol_deriv[1][1] / rho_b

#     # elif fde_method == 'percentile':
#     #     dv_l_normed = -dv_l * np.sqrt(a) / 100
#     #     tau_l_normed = tau_l - np.mean(tau_l)
#     #     # pers = np.arange(25, 75, 2.5)
#     #     # ctot2_4_list = []
#     #     # for per in pers:
#     #     #     sol_deriv = percentile_fde(dc_l, dv_l_normed, tau_l_normed, per)
#     #     #     ctot2_4_list.append(sol_deriv[0][1] / rho_b)
#     #     #
#     #     # err_4 = 0
#     #     # ctot2_4 = sum(ctot2_4_list) / len(ctot2_4_list)

#     #     sol_deriv = percentile_fde(dc_l, dv_l_normed, tau_l_normed, per)
#     #     ctot2_4 = sol_deriv[0][1] / rho_b
#     #     err_4 = sol_deriv[1][1] / rho_b

#     # else:
#     #     raise Exception('fde_method must be \'algorithm\' or \'percentile\'')

#     def fitting_function(X, a0, a1, a2, a3, a4, a5):
#         x1, x2 = X
#         return a0 + a1*x1 + a2*x2 + a3*x1**2 + a4*x2**2 + a5*x1*x2

#     guesses = 1, 1, 1, 1, 1, 1
#     C_6par, _ = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
#     ctot2_4 = np.real(C_6par[1] / rho_b) - np.real(C_6par[2] * H0 / (rho_b * np.sqrt(a)))
#     err_4 = 0

#     # spatial correlations
#     H = a**(-1/2)*100
#     dv_l = dv_l / (H)
#     tD = np.mean(tau_l*dc_l) / rho_b
#     tT = np.mean(tau_l*dv_l) / rho_b
#     DT = np.mean(dc_l*dv_l)
#     TT = np.mean(dv_l*dv_l)
#     DD = np.mean(dc_l*dc_l)
#     rhs = (tD / DT) - (tT / TT)
#     lhs = (DD / DT) - (DT / TT)
#     cs2 = rhs / lhs
#     cv2 = (DD*cs2 - tD) / DT
#     ctot2_5 = (cs2+cv2)
#     ctot2_6 = (tD/ DD)
#     fit_corr = (np.mean(tau_l_0) + cs2*rho_b*dc_l_0 + cv2*rho_b*dv_l_0)
#     return a, x, ctot2, ctot2_2, ctot2_3, err0, err1, err2, cs2, cv2, red_chi, yerr, tau_l, fit, dv_l, P_nb, P_1l, d1k, taus, x_binned, chisq, ctot2_4, err_4, dc_l, ctot2_5, ctot2_6, tau_l_0, fit_corr
#     # return a, pos, ctot2, ctot2_2, ctot2_3, err0, err1, err2, cs2, cv2, red_chi, yerr, taus, fit_sp, terr, P_nb, P_1l, d1k


def param_calc_ens(j, Lambda, path, mode, kind, n_runs=8, n_use=8, folder_name=''):
    a, x, d1k, dc_l_0, dv_l_0, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)

    H0 = 100
    H = a**(-3/2)*H0
    dv_l_0 = -dv_l_0 / (a*H)

    taus, dels, thes = [], [], []
    taus.append(tau_l_0)
    dels.append(dc_l_0)
    thes.append(dv_l_0)

    for run in range(1, n_runs+1):
        path = path[:-2] + '{}/'.format(run)
        sol = read_sim_data(path, Lambda, kind, j, folder_name)
        dels.append(sol[3])
        thes.append(-sol[4] / (a*H))
        taus.append(sol[5])


    Nt = len(taus)

    tau_l = sum(np.array(taus)) / Nt
    dc_l = sum(np.array(dels)) / Nt
    dv_l = sum(np.array(thes)) / Nt


    rho_0 = 27.755
    rho_b = rho_0 / a**3

    diff = np.array([(taus[i] - tau_l)**2 for i in range(1, Nt)])
    yerr = np.sqrt(sum(diff) / (Nt*(Nt-1)))

    n_ev = x.size // n_use
    dc_l_sp = dc_l[0::n_ev]
    dv_l_sp = dv_l[0::n_ev]
    tau_l_sp = tau_l[0::n_ev]
    x_sp = x[0::n_ev]
    yerr_sp = yerr[0::n_ev]

    # F3P
    def fitting_function(X, a0, a1, a2):
        x1, x2 = X
        return a0 + a1*x1 + a2*x2
    guesses = 1, 1, 1
    C_F3P, cov_F3P = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    fit_F3P = fitting_function((dc_l, dv_l), C_F3P[0], C_F3P[1], C_F3P[2])
    cs2_F3P = C_F3P[1] / rho_b
    cv2_F3P = C_F3P[2] / rho_b
    ctot2_F3P = cs2_F3P + cv2_F3P

    # F6P
    def fitting_function(X, a0, a1, a2, a3, a4, a5):
        x1, x2 = X
        return a0 + a1*x1 + a2*x2 + a3*x1**2 + a4*x2**2 + a5*x1*x2
    guesses = 1, 1, 1, 1, 1, 1
    C_F6P, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    fit_F6P = fitting_function((dc_l, dv_l), C_F6P[0], C_F6P[1], C_F6P[2], C_F6P[3], C_F6P[4], C_F6P[5])
    cs2_F6P = C_F6P[1] / rho_b
    cv2_F6P = C_F6P[2] / rho_b
    ctot2_F6P = cs2_F6P + cv2_F6P

    # M&W
    Lambda_int = int(Lambda / (2*np.pi))
    tau_l_k = np.fft.fft(tau_l_0) / x.size
    num = (np.conj(a * d1k) * ((np.fft.fft(tau_l_0)) / x.size))
    denom = ((d1k * np.conj(d1k)) * (a**2))
    ntrunc = int(num.size-Lambda_int)
    num[Lambda_int+1:ntrunc] = 0
    denom[Lambda_int+1:ntrunc] = 0
    ctot2_MW = np.real(sum(num) / sum(denom)) / rho_b

    # SC, SC\delta
    tD = np.mean(tau_l*dc_l) / rho_b
    tT = np.mean(tau_l*dv_l) / rho_b
    DT = np.mean(dc_l*dv_l)
    TT = np.mean(dv_l*dv_l)
    DD = np.mean(dc_l*dc_l)
    rhs = (tD / DT) - (tT / TT)
    lhs = (DD / DT) - (DT / TT)
    cs2 = rhs / lhs
    cv2 = -(DD*cs2 - tD) / DT
    ctot2_SC = (cs2+cv2)
    ctot2_SCD = (tD/ DD)
    fit_SC = (np.mean(tau_l_0) + cs2*rho_b*dc_l_0 + cv2*rho_b*dv_l_0)

    return a, x, d1k, P_nb, P_1l, cs2_F3P, cv2_F3P, cs2_F6P, cv2_F6P, ctot2_F3P, ctot2_F6P, ctot2_MW, ctot2_SC, ctot2_SCD, dc_l, dv_l, tau_l_0, tau_l, fit_F3P, fit_F6P, fit_SC


# path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# A = [-0.05, 1, -0.5, 11]
# file_num = 23
# kind = 'sharp'
# n_runs = 8
# n_use = n_runs - 1
# n_fits = 100
# Lambda = 3 * (2*np.pi)
# mode = 1
# param_calc_ens(path, file_num, Lambda, kind, mode, n_runs, n_use, n_fits)

def tau_ext(file_num, Lambda, path, mode, kind, folder_name, n_use=10, n_runs=8):
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


def spec_from_ens(Nfiles, Lambda, path, mode, kind, n_runs=8, n_use=10, H0=100, folder_name=''):
    print('\npath = {}'.format(path))
    #define lists to store the data
    a_list = np.zeros(Nfiles)
    ctot2_list = np.zeros(Nfiles)
    ctot2_list2 = np.zeros(Nfiles)
    ctot2_list3 = np.zeros(Nfiles)
    ctot2_list4 = np.zeros(Nfiles)

    cs2_list = np.zeros(Nfiles)
    cv2_list = np.zeros(Nfiles)
    fit_list = np.zeros(Nfiles)
    tau_list = np.zeros(Nfiles)

    #An and Bn for the integral over the Green's function
    An_F3P = np.zeros(Nfiles)
    Bn_F3P = np.zeros(Nfiles)
    Pn_F3P = np.zeros(Nfiles)
    Qn_F3P = np.zeros(Nfiles)

    An_F6P = np.zeros(Nfiles)
    Bn_F6P = np.zeros(Nfiles)
    Pn_F6P = np.zeros(Nfiles)
    Qn_F6P = np.zeros(Nfiles)

    An_MW = np.zeros(Nfiles)
    Bn_MW = np.zeros(Nfiles)
    Pn_MW = np.zeros(Nfiles)
    Qn_MW = np.zeros(Nfiles)

    An_SC = np.zeros(Nfiles)
    Bn_SC = np.zeros(Nfiles)
    Pn_SC = np.zeros(Nfiles)
    Qn_SC = np.zeros(Nfiles)

    An_SCD = np.zeros(Nfiles)
    Bn_SCD = np.zeros(Nfiles)
    Pn_SCD = np.zeros(Nfiles)
    Qn_SCD = np.zeros(Nfiles)

    AJ_F3P = np.zeros(Nfiles)
    BJ_F3P = np.zeros(Nfiles)
    PJ_F3P = np.zeros(Nfiles)
    QJ_F3P = np.zeros(Nfiles)

    AJ_F6P = np.zeros(Nfiles)
    BJ_F6P = np.zeros(Nfiles)
    PJ_F6P = np.zeros(Nfiles)
    QJ_F6P = np.zeros(Nfiles)

    AJ_SC = np.zeros(Nfiles)
    BJ_SC = np.zeros(Nfiles)
    PJ_SC = np.zeros(Nfiles)
    QJ_SC = np.zeros(Nfiles)


    #the densitites
    P_nb = np.zeros(Nfiles)
    P_lin = np.zeros(Nfiles)
    P_1l_tr = np.zeros(Nfiles)
    dk_lin = np.zeros(Nfiles, dtype=complex)

    sol = tau_ext(4, Lambda, path, mode, kind, folder_name)
    a1, ctot2_1 = sol[0], sol[-1]

    sol = tau_ext(5, Lambda, path, mode, kind, folder_name)
    a2, ctot2_2 = sol[0], sol[-1]
    slope = (ctot2_2-ctot2_1) / (a2-a1)

    a_list = np.array([np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j)) for j in range(Nfiles)])
    C_P = 4*slope*(a_list[0]**(9/2)) / (45 * (100**2) * a_list**(5/2))
    C_Q = (slope*a_list[0]**2 / (5*100**2)) * np.ones(a_list.size)
    alpha_c_0 = C_P - C_Q #this is the part of alpha_c integrated from 0 to a0. Add this to the integral from a0 to a

    a_list = np.array([np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j)) for j in range(Nfiles)])


    #initial scalefactor
    a0 = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(0))
    q = np.genfromtxt(path + 'output_{0:04d}.txt'.format(0))[:,0]

    for file_num in tqdm(range(Nfiles)):
        # filename = '/output_hierarchy_{0:03d}.txt'.format(file_num)
        #the function 'EFT_solve' return solutions of all modes + the EFT parameters
        ##the following line is to keep track of 'a' for the numerical integration
        if file_num > 0:
            a0 = a

        # a, x, ctot2, ctot2_2, ctot2_3, err0, err1, err2, cs2, cv2, red_chi, yerr, tau_l, fit, terr, P_nb_a, P_1l_a_tr, \
        #     d1k, taus, x_binned, chisq, ctot2_4, err_4, dc_l, ctot2_5, ctot2_6, tau_l_0, fit_corr = param_calc_ens(file_num, Lambda, path, A, mode, kind, n_runs, n_use, folder_name, fde_method, ens=True)

        a, x, d1k, P_nb_a, P_1l_a_tr, cs2_F3P, cv2_F3P, cs2_F6P, cv2_F6P, ctot2_F3P, ctot2_F6P, ctot2_MW, ctot2_SC, ctot2_SCD,\
             dc_l, dv_l, tau_l_0, tau_l, fit_F3P, fit_F6P, fit_SC = param_calc_ens(file_num, Lambda, path, mode, kind, n_runs, n_use, folder_name=folder_name)


        Nx = x.size
        k = np.fft.ifftshift(2.0 * np.pi * np.arange(-Nx/2, Nx/2))
        P_nb[file_num] = P_nb_a[mode]
        P_1l_tr[file_num] = P_1l_a_tr[mode]
        P_lin[file_num] = (np.abs(d1k**2)*a**2)[mode]
        dk_lin[file_num] = d1k[mode] * a

        J_F3P = np.real(np.fft.fft(tau_l - fit_F3P)[mode]) / tau_l.size
        J_F6P = np.real(np.fft.fft(tau_l - fit_F6P)[mode]) / tau_l.size
        J_SC = np.real(np.fft.fft(tau_l_0 - fit_SC)[mode]) / tau_l.size


        ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
        if file_num > 0:
            da = a - a0

            #F3P
            Pn_F3P[file_num] = ctot2_F3P * (a**(5/2)) #for calculation of alpha_c
            Qn_F3P[file_num] = ctot2_F3P

            #F6P
            Pn_F6P[file_num] = ctot2_F6P * (a**(5/2)) #for calculation of alpha_c
            Qn_F6P[file_num] = ctot2_F6P

            #M&W
            Pn_MW[file_num] = ctot2_MW * (a**(5/2)) #for calculation of alpha_c
            Qn_MW[file_num] = ctot2_MW

            #SC
            Pn_SC[file_num] = ctot2_SC * (a**(5/2)) #for calculation of alpha_c
            Qn_SC[file_num] = ctot2_SC

            #SCD
            Pn_SCD[file_num] = ctot2_SCD * (a**(5/2)) #for calculation of alpha_c
            Qn_SCD[file_num] = ctot2_SCD

            #Stoch F3P
            PJ_F3P[file_num] = J_F3P * (a**(3/2)) #for calculation of alpha_c
            QJ_F3P[file_num] = J_F3P / a

            #Stoch F6P
            PJ_F6P[file_num] = J_F6P * (a**(3/2)) #for calculation of alpha_c
            QJ_F6P[file_num] = J_F6P / a

            #Stoch SC
            PJ_SC[file_num] = J_SC * (a**(3/2)) #for calculation of alpha_c
            QJ_SC[file_num] = J_SC / a

    #A second loop for the integration
    for j in range(1, Nfiles):
        An_F3P[j] = np.trapz(Pn_F3P[:j], a_list[:j])
        Bn_F3P[j] = np.trapz(Qn_F3P[:j], a_list[:j])

        An_F6P[j] = np.trapz(Pn_F6P[:j], a_list[:j])
        Bn_F6P[j] = np.trapz(Qn_F6P[:j], a_list[:j])

        An_MW[j] = np.trapz(Pn_MW[:j], a_list[:j])
        Bn_MW[j] = np.trapz(Qn_MW[:j], a_list[:j])

        An_SC[j] = np.trapz(Pn_SC[:j], a_list[:j])
        Bn_SC[j] = np.trapz(Qn_SC[:j], a_list[:j])

        An_SCD[j] = np.trapz(Pn_SCD[:j], a_list[:j])
        Bn_SCD[j] = np.trapz(Qn_SCD[:j], a_list[:j])

        AJ_F3P[j] = np.trapz(PJ_F3P[:j], a_list[:j])
        BJ_F3P[j] = np.trapz(QJ_F3P[:j], a_list[:j])

        AJ_F6P[j] = np.trapz(PJ_F6P[:j], a_list[:j])
        BJ_F6P[j] = np.trapz(QJ_F6P[:j], a_list[:j])

        AJ_SC[j] = np.trapz(PJ_SC[:j], a_list[:j])
        BJ_SC[j] = np.trapz(QJ_SC[:j], a_list[:j])

    #calculation of the Green's function integral
    C = 2 / (5 * H0**2)

    An_F3P /= (a_list**(5/2))
    An_F6P /= (a_list**(5/2))
    An_MW /= (a_list**(5/2))
    An_SC /= (a_list**(5/2))
    An_SCD /= (a_list**(5/2))

    AJ_F3P /= (a_list**(3/2))
    AJ_F6P /= (a_list**(3/2))
    AJ_SC /= (a_list**(3/2))

    BJ_F3P *= a_list
    BJ_F6P *= a_list
    BJ_SC *= a_list

    del_J_F3P = C * (AJ_F3P - BJ_F3P)
    del_J_F6P = C * (AJ_F6P - BJ_F6P)
    del_J_SC = C * (AJ_SC - BJ_SC)
    
    alpha_c_true = k[mode]**2 * ((P_nb - P_1l_tr) / (2 * P_lin * k[mode]**2))
    alpha_c_F3P = k[mode]**2 * ((C * (An_F3P - Bn_F3P)) + alpha_c_0)
    alpha_c_F6P = k[mode]**2 * ((C * (An_F6P - Bn_F6P)) + alpha_c_0)
    alpha_c_MW = k[mode]**2 * ((C * (An_MW - Bn_MW)) + alpha_c_0)
    alpha_c_SC = k[mode]**2 * ((C * (An_SC - Bn_SC)) + alpha_c_0)
    alpha_c_SCD = k[mode]**2 * ((C * (An_SCD - Bn_SCD)) + alpha_c_0)

    PJ_F3P = np.real(del_J_F3P * np.conj(del_J_F3P)) * a**4
    PJ_F6P = np.real(del_J_F6P * np.conj(del_J_F6P)) * a**4
    PJ_SC = np.real(del_J_SC * np.conj(del_J_SC)) * a**4

    P_eft_true = P_1l_tr + ((2 * alpha_c_true) * P_lin)
    P_eft_F3P = P_1l_tr + ((2 * alpha_c_F3P) * P_lin) + PJ_F3P
    P_eft_F6P = P_1l_tr + ((2 * alpha_c_F6P) * P_lin) + PJ_F6P
    P_eft_MW = P_1l_tr + ((2 * alpha_c_MW) * P_lin)
    P_eft_SC = P_1l_tr + ((2 * alpha_c_SC) * P_lin) + PJ_SC
    P_eft_SCD = P_1l_tr + ((2 * alpha_c_SCD) * P_lin)

    return a_list, x, P_nb, P_1l_tr, P_eft_F3P, P_eft_F6P, P_eft_MW, P_eft_SC, P_eft_SCD, P_eft_true


def alpha_c_finder(Nfiles, Lambda, path, A, mode, kind, n_runs=8, n_use=10, H0=100, fm='curve_fit', nbins_x=10, nbins_y=10, npars=3, fde_method='algorithm', folder_name=''):
    print('\npath = {}'.format(path))
    #define lists to store the data
    a_list = np.zeros(Nfiles)
    ctot2_list = np.zeros(Nfiles)
    ctot2_list2 = np.zeros(Nfiles)
    ctot2_list3 = np.zeros(Nfiles)
    ctot2_list4 = np.zeros(Nfiles)

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

    err_A = np.zeros(Nfiles)
    err_B = np.zeros(Nfiles)
    err_I1 = np.zeros(Nfiles)
    err_I2 = np.zeros(Nfiles)

    P_nb = np.zeros(Nfiles)
    P_1l = np.zeros(Nfiles)
    P_lin = np.zeros(Nfiles)

    terr_list = np.zeros(Nfiles)

    #initial scalefactor
    a0 = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(0))
    q = np.genfromtxt(path + 'output_{0:04d}.txt'.format(0))[:,0]

    for file_num in range(Nfiles):
       # filename = '/output_hierarchy_{0:03d}.txt'.format(file_num)
       #the function 'EFT_solve' return solutions of all modes + the EFT parameters
       ##the following line is to keep track of 'a' for the numerical integration
       if file_num > 0:
          a0 = a


       a, x, ctot2, ctot2_2, ctot2_3, err0, err1, err2, cs2, cv2, red_chi, yerr, tau_l, fit, terr, P_nb_a, P_1l_a_tr, d1k, taus, x_binned, chisq, ctot2_4, err_4, dc_l, ctot2_5, ctot2_6 = param_calc_ens(file_num, Lambda, path, A, mode, kind, n_runs, n_use, folder_name=folder_name, fitting_method=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars, fde_method=fde_method, ens=True)
       print('a = ', a)
       a_list[file_num] = a
       ctot2_list[file_num] = ctot2
       ctot2_list2[file_num] = ctot2_2
       ctot2_list3[file_num] = ctot2_3
       ctot2_list4[file_num] = ctot2_4

       terr_list[file_num] = err_4 #terr
       P_nb[file_num] = P_nb_a[mode]
       P_1l[file_num] = P_1l_a_tr[mode]
       P_lin[file_num] = a**2 * np.real(d1k * np.conj(d1k))[mode]

       Nx = x.size
       k = np.fft.ifftshift(2.0 * np.pi * np.arange(-Nx/2, Nx/2))

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

          # err_A[file_num] = (terr * (a**(5/2)) * da)**2
          # err_B[file_num] = (terr * da)**2

          err_A[file_num] = (err_4 * (a**(5/2)) * da)**2
          err_B[file_num] = (err_4 * da)**2

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

        # err_A[j] = np.trapz((a_list**(5/2))[:j], a_list[:j])
        # err_B[j] = np.trapz((np.ones(a_list.size))[:j], a_list[:j])

        err_I1[j] = sum(err_A[:j])
        err_I2[j] = sum(err_B[:j])

    #calculation of the Green's function integral
    C = 2 / (5 * H0**2)
    An /= (a_list**(5/2))
    An2 /= (a_list**(5/2))
    An3 /= (a_list**(5/2))
    An4 /= (a_list**(5/2))

    # err_A /= a_list**(5/2)
    # err_Int = C * terr_list * (err_A - err_B)**2
    err_I1 = np.sqrt(err_I1) / a_list**(5/2)
    err_I2 = np.sqrt(err_I2)

    # print(P_nb[21:26], P_1l[21:26], P_lin[21:26])
    alpha_c_true = (P_nb - P_1l) / (2 * P_lin * k[mode]**2)
    # print(alpha_c_true[21:26])
    err_Int = C * np.sqrt(err_I1**2 + err_I2**2)
    alpha_c_naive = C * (An - Bn)
    alpha_c_naive2 = C * (An2 - Bn2)
    alpha_c_naive3 = C * (An3 - Bn3)
    alpha_c_naive4 = C * (An4 - Bn4)

    return a_list, x, alpha_c_true, alpha_c_naive, alpha_c_naive2, alpha_c_naive3, alpha_c_naive4, err_Int

def spec_nbody(path, Nfiles, mode, sm=False, kind='sharp', Lambda=1, folder_name=''):
    a_list = np.zeros(Nfiles)
    P_nb = np.zeros(Nfiles)
    print('\npath = {}'.format(path))
    for j in range(0, Nfiles):
        a, dx, M0_par = read_hier(path, j, folder_name)[:3]
        L = 1.0
        x = np.arange(0, L, dx)
        Nx = x.size
        k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
        dc_in, k_in = dc_in_finder(path, x)
        M0_par = (M0_par - 1) / np.mean(M0_par)
        if sm == True:
            M0_par = smoothing(M0_par, k, Lambda, kind)
        M0_k = np.fft.fft(M0_par) / M0_par.size
        P_nb[j] = (np.real(M0_k * np.conj(M0_k)))[mode]
        a_list[j] = a

    return a_list, P_nb



def is_between(x, x1, x2):
   """returns a subset of x that lies between x1 and x2"""
   indices = []
   values = []
   for j in range(x.size):
      if x1 <= x[j] <= x2:
         indices.append(j)
         values.append(x[j])
   values = np.array(values)
   indices = np.array(indices)
   return indices, values


def hier_calc(j, path, dx_grid):
    nbody_filename = 'output_{0:04d}.txt'.format(j)
    nbody_file = np.genfromtxt(path + nbody_filename)
    x_nbody = nbody_file[:,-1]
    v_nbody = nbody_file[:,2]

    x_grid = np.arange(0, 1, dx_grid)
    M0 = [[] for j in range(x_grid.size)]
    C1 = [[] for j in range(x_grid.size)]

    for m in range(x_nbody.size):
        p = int(round(x_nbody[m]/dx_grid))
        if p == x_grid.size:
            p = 0
        else:
            pass
        M0[p].append(m)
        C1[p].append(v_nbody[m])


    M0 = [len(M0[j]) for j in range(len(M0))]
    C2 = [sum(np.array(C1[j])**2)/len(C1[j]) for j in range(len(C1))]
    C1 = [sum(C1[j])/len(C1[j]) for j in range(len(C1))]

    M0 /= np.mean(M0)
    M1 = M0 * C1
    M2 = C2 * M0
    C0 = M0
    return x_grid, M0, M1, M2, C0, C1, C2

def write_hier(zero, Nfiles, path, dx_grid, folder_name=''):
    if folder_name == '':
        filepath = path + '/hierarchy/'
    else:
        filepath = path + '/{}/'.format(folder_name)

    try:
        os.makedirs(filepath, 0o755)
        print('Path doesn\'t exist, new directory created.')

    except:
        print('Path exists, writing files...')
        pass

    print('Writing moments...\n')
    for j in range(zero, Nfiles):
        x_grid, M0, M1, M2, C0, C1, C2 = hier_calc(j, path, dx_grid)
        filename = 'hier_{0:04d}.hdf5'.format(j)
        file = h5py.File(filepath+filename, 'w')
        file.create_group('Header')
        header = file['Header']
        a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
        print('a = ', a)
        header.attrs.create('a', a, dtype=float)
        header.attrs.create('dx', dx_grid, dtype=float)

        moments = file.create_group('Moments')
        moments.create_dataset('M0', data=M0)
        moments.create_dataset('M1', data=M1)
        moments.create_dataset('M2', data=M2)
        moments.create_dataset('C0', data=C0)
        moments.create_dataset('C1', data=C1)
        moments.create_dataset('C2', data=C2)

        file.close()
    print("Done!\n")

def read_hier(path, j, folder_name=''):
    if folder_name == '':
        filepath = path + '/hierarchy/'
    else:
        filepath = path + '/{}/'.format(folder_name)

    filename = 'hier_{0:04d}.hdf5'.format(j)
    file = h5py.File(filepath+filename, mode='r')
    header = file['/Header']
    a = header.attrs.get('a')
    dx = header.attrs.get('dx')

    moments = file['/Moments']
    mom_keys = list(moments.keys())
    C0 = np.array(moments[mom_keys[0]])
    C1 = np.array(moments[mom_keys[1]])
    C2 = np.array(moments[mom_keys[2]])
    M0 = np.array(moments[mom_keys[3]])
    M1 = np.array(moments[mom_keys[4]])
    M2 = np.array(moments[mom_keys[5]])
    file.close()
    return a, dx, M0, M1, M2, C0, C1, C2


def best_ind_par(dc_l, dv_l, tau_l, dist):
    def new_param_calc(dc_l, dv_l, tau_l, dist, ind):
        def dir_der_o1(X, tau_l, ind):
            """Calculates the first-order directional derivative of tau_l along the vector X."""
            x1 = np.array([X[0][ind], X[1][ind]])
            x2 = np.array([X[0][ind+1], X[1][ind+1]])
            v = (x2 - x1)
            D_v_tau = (tau_l[ind+1] - tau_l[ind]) / v[0]
            # print(D_v_tau)
            return v, D_v_tau

        def dir_der_o2(X, tau_l, ind):
            """Calculates the second-order directional derivative of tau_l along the vector X."""
            v0, D_v_tau0 = dir_der_o1(X, tau_l, ind-2)
            v1, D_v_tau1 = dir_der_o1(X, tau_l, ind)
            v2, D_v_tau2 = dir_der_o1(X, tau_l, ind+2)
            x0 = np.array([X[0][ind-2], X[1][ind-2]])
            x1 = np.array([X[0][ind], X[1][ind]])
            x2 = np.array([X[0][ind+2], X[1][ind+2]])
            v = (x2 - x1)
            D2_v_tau = (D_v_tau2 - D_v_tau1) / v[0]
            return v, D2_v_tau


        X = np.array([dc_l, dv_l])
        params_list = []
        for j in range(-dist//2, dist//2 + 1):
            v1, dtau1 = dir_der_o1(X, tau_l, ind+j)
            v1_o2, dtau1_o2 = dir_der_o2(X, tau_l, ind+j)
            dc_0, dv_0 = dc_l[ind], dv_l[ind]
            C_ = [((tau_l[ind])-(dtau1*dc_0)+((dtau1_o2*dc_0**2)/2)), dtau1-(dtau1_o2*dc_0), dtau1_o2/2]
            params_list.append(C_)

        params_list = np.array(params_list)
        dist = params_list.shape[0]
        if dist != 0:
            C0_ = np.mean(np.array([params_list[j][0] for j in range(dist)]))
            C1_ = np.mean(np.array([params_list[j][1] for j in range(dist)]))
            C2_ = np.mean(np.array([params_list[j][2] for j in range(dist)]))
            C_ = [C0_, C1_, C2_]
        else:
            C_ = [0, 0, 0]
        return C_

    def opt_fun(opt_params, dc_l, dv_l, tau_l, dist):
        ind = int(opt_params)
        C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
        est = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2
        resid = sum((tau_l - est)**2)
        return resid


    x0 = (np.argmin(dc_l**2 + dv_l**2))
    # bounds = [(0, 62500)]
    bounds = [(0, dc_l.size)]

    sol = minimize(opt_fun, x0, args=(dc_l, dv_l, tau_l, dist), bounds=bounds)
    sol_params = int(sol.x)
    if sol.success:
        pass
    else:
        print('Warning: the optimisation did not converge!')


    def calc_fit(params, dc_l, dv_l, tau_l, dist):
        ind = int(params)
        C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
        return C_, ind

    return calc_fit(sol_params, dc_l, dv_l, tau_l, dist)


def deriv_param_calc(dc_l, dv_l, tau_l, a, bounds=None, x0=None):
    def new_param_calc(dc_l, dv_l, tau_l, dist, ind, a):
        def dir_der_o1(X, tau_l, ind):
            """Calculates the first-order directional derivative of tau_l along the vector X."""
            ind_right = ind + 2
            if ind_right >= tau_l.size:
                ind_right = ind_right - tau_l.size
                print(ind, ind_right)

            x1 = np.array([X[0][ind], X[1][ind]])
            x2 = np.array([X[0][ind_right], X[1][ind_right]])
            v = (x2 - x1)
            D_v_tau = (tau_l[ind_right] - tau_l[ind]) / v[0]
            return v, D_v_tau

        def dir_der_o2(X, tau_l, ind):
            """Calculates the second-order directional derivative of tau_l along the vector X."""
            ind_right = ind + 4
            if ind_right >= tau_l.size:
                ind_right = ind_right - tau_l.size
                print(ind, ind_right)
            x1 = np.array([X[0][ind], X[1][ind]])
            x2 = np.array([X[0][ind_right], X[1][ind_right]])
            v1, D_v_tau1 = dir_der_o1(X, tau_l, ind)
            v2, D_v_tau2 = dir_der_o1(X, tau_l, ind_right)
            v = (x2 - x1)
            D2_v_tau = (D_v_tau2 - D_v_tau1) / v[0]
            return v, D2_v_tau


        X = np.array([dc_l, dv_l])
        params_list = []
        for j in range(-dist//2, dist//2 + 1):
            v1, dtau1 = dir_der_o1(X, tau_l, ind+j)
            v1_o2, dtau1_o2 = dir_der_o2(X, tau_l, ind+j)
            if dtau1_o2 != None:
                dc_0, dv_0 = dc_l[ind], dv_l[ind]
                # C_ = [((tau_l[ind])-(dtau1*dc_0)+((dtau1_o2*dc_0**2)/2)), dtau1-(dtau1_o2*dc_0), dtau1_o2/2]
                C_ = [((tau_l[ind])-(dtau1*dc_0)+((dtau1_o2*dc_0**2)/2)), dtau1-(dtau1_o2*dc_0), dtau1_o2/2]

                params_list.append(C_)

        params_list = np.array(params_list)
        dist = params_list.shape[0]
        if dist != 0:
            C0_ = np.mean(np.array([params_list[j][0] for j in range(dist)]))
            C1_ = np.mean(np.array([params_list[j][1] for j in range(dist)]))
            C2_ = np.mean(np.array([params_list[j][2] for j in range(dist)]))

            C_ = [C0_, C1_, C2_]
        else:
            C_ = [0, 0, 0]
        return C_


    def minimise_deriv(params, dc_l, dv_l, tau_l, dist):
        start, thresh, n_sub = params
        n_sub = int(np.abs(n_sub))
        N = dc_l.size
        if start < 0:
            start = N + start
        C_list = []
        sub = np.linspace(start, N-start+1, n_sub, dtype=int)
        del_ind = np.argmax(tau_l)
        for point in sub:
            if del_ind-thresh < point < del_ind+thresh:
                sub = np.delete(sub, np.where(sub==point)[0][0])
            else:
                pass
        n_sub = sub.size
        for j in range(n_sub):
            if sub[j] < 0:
                sub[j] += N
            elif sub[j] > N:
                sub[j] = N - sub[j]
            tau_val = tau_l[sub[j]]
            tau_diff = np.abs(tau_l - tau_val)
            ind_tau = np.argmin(tau_diff)
            dc_0, dv_0 = dc_l[ind_tau], dv_l[ind_tau]
            ind = np.argmin((dc_l-dc_0)**2 + (dv_l-dv_0)**2)
            C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind, a)
            C_list.append(C_)

        try:
            C0_ = np.mean([C_list[l][0] for l in range(len(C_list))])
            C1_ = np.mean([C_list[l][1] for l in range(len(C_list))])
            C2_ = np.mean([C_list[l][2] for l in range(len(C_list))])
        except:
            C0_ = 0
            C1_ = 0
            C2_ = 0

        C_ = [C0_, C1_, C2_]
        fit = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2

        resid = sum((tau_l - fit)**2)
        return resid

    # if a == 2.48 or 2.7:
    #     x0 = (5000, 8000, 50)
    # else:
    #     x0 = (8000, 8000, 25)
    # # if x0 == None:
    # #     x0 = (5000, 7000, 35)
    # # else:
    # #     pass
    #
    # if bounds == None:
    #     bounds = [(1000, 20000), (1000, 21000), (1, 250)]
    # else:
    #     pass
    #
    # dist = 1
    # sol = minimize(minimise_deriv, x0, args=(dc_l, dv_l, tau_l, dist), bounds=bounds, method='Powell')#, tol=10)
    # sol_params = [int(np.abs(par)) for par in sol.x]


    # x0 = (8000, 8000, 25)
    # bounds = [(1000, 20000), (1000, 21000), (1, 250)]
    x0 = (100, 500, 5)
    bounds = [(50, 900), (450, 650), (1, 100)]
    dist = 1
    sol = minimize(minimise_deriv, x0, args=(dc_l, dv_l, tau_l, dist), bounds=bounds, method='Powell')#, tol=10)
    sol_params = [int(np.abs(par)) for par in sol.x]

    def calc_fit(params, dc_l, dv_l, tau_l, dist, a):
        start, thresh, n_sub = params
        n_sub = int(np.abs(n_sub))
        C_list = []
        N = dc_l.size
        sub = np.linspace(start, N-start+1, n_sub, dtype=int)
        del_ind = np.argmax(tau_l)

        for point in sub:
            if del_ind-thresh < point < del_ind+thresh:
                sub = np.delete(sub, np.where(sub==point)[0][0])
            else:
                pass
            if sub.size < 5:
                break
            else:
                pass

        # C_best, ind_best = best_ind_par(dc_l, dv_l, tau_l, dist)
        # distance = np.sqrt((dc_l-dc_l[ind_best])**2 + (dv_l-dv_l[ind_best])**2)
        # per = np.percentile(distance, 1)
        # indices = np.where(distance < per)[0]
        # params_list = []
        # for ind in indices:
        #     C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
        #     C_list.append(C_)

        # n_sub = len(C_list)
        # C0_ = np.mean([C_list[l][0] for l in range(len(C_list))])
        # C1_ = np.mean([C_list[l][1] for l in range(len(C_list))])
        # C2_ = np.mean([C_list[l][2] for l in range(len(C_list))])
        # err0_ = np.sqrt(sum([((C_list[l][0] - C0_)**2/(n_sub*(n_sub-1))) for l in range(len(C_list))]))
        # err1_ = np.sqrt(sum([((C_list[l][1] - C0_)**2/(n_sub*(n_sub-1))) for l in range(len(C_list))]))
        # err2_ = np.sqrt(sum([((C_list[l][2] - C0_)**2/(n_sub*(n_sub-1))) for l in range(len(C_list))]))
        # err_best = [err0_, err1_, err2_]
        # C_best = [C0_, C1_, C2_]

        try:
            n_sub = sub.size
            for j in range(n_sub):
                tau_val = tau_l[sub[j]]
                tau_diff = np.abs(tau_l - tau_val)
                ind_tau = np.argmin(tau_diff)
                dc_0, dv_0 = dc_l[ind_tau], dv_l[ind_tau]
                ind = np.argmin((dc_l-dc_0)**2 + (dv_l-dv_0)**2)
                C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind, a)
                C_list.append(C_)
            C0_ = np.mean([C_list[l][0] for l in range(len(C_list))])
            C1_ = np.mean([C_list[l][1] for l in range(len(C_list))])
            C2_ = np.mean([C_list[l][2] for l in range(len(C_list))])
            err0_ = np.sqrt(sum([((C_list[l][0] - C0_)**2/(n_sub*(n_sub-1))) for l in range(len(C_list))]))
            err1_ = np.sqrt(sum([((C_list[l][1] - C0_)**2/(n_sub*(n_sub-1))) for l in range(len(C_list))]))
            err2_ = np.sqrt(sum([((C_list[l][2] - C0_)**2/(n_sub*(n_sub-1))) for l in range(len(C_list))]))
            err_ = [err0_, err1_, err2_]
            C_ = [C0_, C1_, C2_]
            param_1 = C1_ / 27.755 * a**3
            # if np.abs(param_1) > 2 or C1_ < 0:
            #     raise Exception('Finding better method...')

        except Exception as e:
            print(e)
            print('Warning: trying best points estimate!')
            # err_, C_ = err_best, C_best
            # err_, C_ = [0, 0, 0], [0, 0, 0]
            C_best, ind_best = best_ind_par(dc_l, dv_l, tau_l, dist)
            distance = np.sqrt((dc_l-dc_l[ind_best])**2 + (dv_l-dv_l[ind_best])**2)
            per = np.percentile(distance, 1)
            indices = np.where(distance < per)[0]
            params_list = []
            C_list = []
            for ind in indices:
                C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind, a)
                C_list.append(C_)

            n_sub = len(C_list)
            C0_ = np.mean([C_list[l][0] for l in range(len(C_list))])
            C1_ = np.mean([C_list[l][1] for l in range(len(C_list))])
            C2_ = np.mean([C_list[l][2] for l in range(len(C_list))])
            err0_ = np.sqrt(sum([((C_list[l][0] - C0_)**2/(n_sub*(n_sub-1))) for l in range(len(C_list))]))
            err1_ = np.sqrt(sum([((C_list[l][1] - C0_)**2/(n_sub*(n_sub-1))) for l in range(len(C_list))]))
            err2_ = np.sqrt(sum([((C_list[l][2] - C0_)**2/(n_sub*(n_sub-1))) for l in range(len(C_list))]))
            err_best = [err0_, err1_, err2_]
            C_best = [C0_, C1_, C2_]
            err_, C_ = err_best, C_best
            sub = indices

        # fit = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2
        # resid = sum((tau_l-fit)**2)
        # fit_best = C_best[0] + C_best[1]*dc_l + C_best[2]*dc_l**2
        # resid_best = sum((tau_l-fit_best)**2)
        #
        # # return C_, err_, sub
        #
        # if resid < resid_best:
        #     return C_, err_, sub
        # else:
        #     return C_best, err_best, sub
        return C_, err_, sub

    return calc_fit(sol_params, dc_l, dv_l, tau_l, dist, a)


def percentile_fde(dc_l, dv_l, tau_l, per=43):
    ind_ord = np.argsort(dv_l)
    dc_l_sorted = dc_l[ind_ord]
    dv_l_sorted = dv_l[ind_ord]
    tau_l_sorted = tau_l[ind_ord]

    def new_param_calc(dc_l, dv_l, tau_l, dist, ind):
        # print('ind in func = ', dc_l[ind], dv_l[ind], tau_l[ind])
        def dir_der_o1(X, tau_l, ind):
            """Calculates the first-order directional derivative of tau_l along the vector X."""
            ind_right = ind + 2
            if ind_right >= tau_l.size:
                ind_right = ind_right - tau_l.size
                print(ind, ind_right)

            x1 = np.array([X[0][ind], X[1][ind]])
            x2 = np.array([X[0][ind_right], X[1][ind_right]])
            v = (x2 - x1)
            D_v_tau = (tau_l[ind_right] - tau_l[ind]) / v[1]
            return v, D_v_tau

        def dir_der_o2(X, tau_l, ind):
            """Calculates the second-order directional derivative of tau_l along the vector X."""
            ind_right = ind + 4
            if ind_right >= tau_l.size:
                ind_right = ind_right - tau_l.size
                print(ind, ind_right)
            x1 = np.array([X[0][ind], X[1][ind]])
            x2 = np.array([X[0][ind_right], X[1][ind_right]])
            v1, D_v_tau1 = dir_der_o1(X, tau_l, ind)
            v2, D_v_tau2 = dir_der_o1(X, tau_l, ind_right)
            v = (x2 - x1)
            D2_v_tau = (D_v_tau2 - D_v_tau1) / v[1]
            return v, D2_v_tau

        X = np.array([dc_l, dv_l])
        # params_list = []
        # params_list_first = []

        # for j in range(1):#-dist//2, dist//2 + 1):
        v1, dtau1 = dir_der_o1(X, tau_l, ind)
        v1_o2, dtau1_o2 = dir_der_o2(X, tau_l, ind)
        dc_0, dv_0 = dc_l[ind], dv_l[ind]
        # C_ = [((tau_l[ind])-(dtau1*dc_0)+((dtau1_o2*dc_0**2)/2)), dtau1-(dtau1_o2*dc_0), dtau1_o2/2]
        C_ = [((tau_l[ind])-(dtau1*dv_0)+((dtau1_o2*dv_0**2)/2)), dtau1-(dtau1_o2*dv_0), dtau1_o2/2]
        C_first = [(tau_l[ind]-dtau1*dv_0), dtau1, dtau1_o2/2]


        return C_, C_first

    dist = 1
    # ind = np.argmin(dc_l_sorted**2 + dv_l_sorted**2)
    # distance = np.sqrt(dc_l_sorted**2 + dv_l_sorted**2)
    ind_0 = np.argmin(dc_l**2 + dv_l**2)
    distance = np.sqrt(dc_l**2 + dv_l**2)
    per_dist = np.percentile(distance, per)
    indices = np.where(distance < per_dist)[0]
    params_list = []
    for ind in indices:
        # C_, C_first = new_param_calc(dc_l_sorted, dv_l_sorted, tau_l_sorted, dist, ind)
        C_, C_first = new_param_calc(dc_l, dv_l, tau_l, dist, ind)

        params_list.append(C_)
        # params_list_first = np.array(C_first)

    # C0_ = np.median([params_list[j][0] for j in range(len(params_list))])
    # C1_ = np.median([params_list[j][1] for j in range(len(params_list))])
    # C2_ = np.median([params_list[j][2] for j in range(len(params_list))])

    # C0_ = np.median([params_list_first[j][0] for j in range(len(params_list_first))])
    # C1_ = np.median([params_list_first[j][1] for j in range(len(params_list_first))])
    # C2_ = np.median([params_list_first[j][2] for j in range(len(params_list_first))])
    # C_first = [C0_, C1_, C2_]


    weights = None#(distance[indices])**(-1)
    # C0_ = np.mean([params_list[j][0] for j in range(len(params_list))])
    # C1_ = np.mean([params_list[j][1] for j in range(len(params_list))])
    # C2_ = np.mean([params_list[j][2] for j in range(len(params_list))])

    C0_ = np.average([params_list[j][0] for j in range(len(params_list))], weights=weights)
    C1_ = np.average([params_list[j][1] for j in range(len(params_list))], weights=weights)
    C2_ = np.average([params_list[j][2] for j in range(len(params_list))], weights=weights)

    err0_ = np.sqrt(sum([((params_list[l][1] - C0_)**2/(indices.size*(indices.size-1))) for l in range(len(params_list))]))
    err1_ = np.sqrt(sum([((params_list[l][1] - C1_)**2/(indices.size*(indices.size-1))) for l in range(len(params_list))]))
    err2_ = np.sqrt(sum([((params_list[l][1] - C2_)**2/(indices.size*(indices.size-1))) for l in range(len(params_list))]))

    C_ = [C0_, C1_, C2_]
    # print(C1_)
    # err_ = [0, 0, 0]
    err_ = [err0_, err1_, err2_]


    return C_, err_, C_first, indices, ind_0


# def percentile_fde(dc_l, dv_l, tau_l, per):
#     def new_param_calc(dc_l, dv_l, tau_l, dist, ind):
#         def dir_der_o1(X, tau_l, ind):
#             """Calculates the first-order directional derivative of tau_l along the vector X."""
#             ind_right = ind + 2
#             if ind_right >= tau_l.size:
#                 ind_right = ind_right - tau_l.size
#                 print(ind, ind_right)
#
#             x1 = np.array([X[0][ind], X[1][ind]])
#             x2 = np.array([X[0][ind_right], X[1][ind_right]])
#             v = (x2 - x1)
#             D_v_tau = (tau_l[ind_right] - tau_l[ind]) / v[0]
#             return v, D_v_tau
#
#         def dir_der_o2(X, tau_l, ind):
#             """Calculates the second-order directional derivative of tau_l along the vector X."""
#             ind_right = ind + 4
#             if ind_right >= tau_l.size:
#                 ind_right = ind_right - tau_l.size
#                 print(ind, ind_right)
#             x1 = np.array([X[0][ind], X[1][ind]])
#             x2 = np.array([X[0][ind_right], X[1][ind_right]])
#             v1, D_v_tau1 = dir_der_o1(X, tau_l, ind)
#             v2, D_v_tau2 = dir_der_o1(X, tau_l, ind_right)
#             v = (x2 - x1)
#             D2_v_tau = (D_v_tau2 - D_v_tau1) / v[0]
#             return v, D2_v_tau
#
#         X = np.array([dc_l, dv_l])
#         params_list = []
#         for j in range(-dist//2, dist//2 + 1):
#             v1, dtau1 = dir_der_o1(X, tau_l, ind+j)
#             v1_o2, dtau1_o2 = dir_der_o2(X, tau_l, ind+j)
#             dc_0, dv_0 = dc_l[ind], dv_l[ind]
#             # C_ = [((tau_l[ind])-(dtau1*dc_0)+((dtau1_o2*dc_0**2)/2)), dtau1-(dtau1_o2*dc_0), dtau1_o2/2]
#             C_ = [((tau_l[ind])-(dtau1*dc_0)+((dtau1_o2*dc_0**2)/2)), dtau1, dtau1_o2/2]
#
#             params_list.append(C_)
#
#         params_list = np.array(params_list)
#         dist = params_list.shape[0]
#         if dist != 0:
#             C0_ = np.mean(np.array([params_list[j][0] for j in range(dist)]))
#             C1_ = np.mean(np.array([params_list[j][1] for j in range(dist)]))
#             C2_ = np.mean(np.array([params_list[j][2] for j in range(dist)]))
#             C_ = [C0_, C1_, C2_]
#         else:
#             C_ = [0, 0, 0]
#         return C_
#
#     dist = 1
#     ind = np.argmin(dc_l**2 + dv_l**2)
#     distance = np.sqrt(dc_l**2 + dv_l**2)
#     per_dist = np.percentile(distance, per)
#     indices = np.where(distance < per_dist)[0]
#
#     params_list = []
#     for ind in indices:
#         C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
#         params_list.append(C_)
#
#     C0_ = np.median([params_list[j][0] for j in range(len(params_list))])
#     C1_ = np.median([params_list[j][1] for j in range(len(params_list))])
#     C2_ = np.median([params_list[j][2] for j in range(len(params_list))])
#
#     # C0_ = np.mean([params_list[j][0] for j in range(len(params_list))])
#     # C1_ = np.mean([params_list[j][1] for j in range(len(params_list))])
#     # C2_ = np.mean([params_list[j][2] for j in range(len(params_list))])
#     C_ = [C0_, C1_, C2_]
#
#     err_ = [0, 0, 0]
#     return C_, err_
