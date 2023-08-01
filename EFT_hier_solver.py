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

from functions import spectral_calc, smoothing, EFT_sm_kern, read_density, SPT_sm, SPT_tr, write_sim_data, read_sim_data, read_hier, interp1d
from zel import initial_density


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
    M2 = M2_nbody * rho_b / a**2 #this makes MH_2 into the form ρv^2 + κ, which this the same as σ as defined in Eq. (10) of Hertzberg (2014)

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

    # Phi_l_true = smoothing(dx_phi * rho, k, Lambda, kind)
    # Phi_l_cgpt = (dx_phi_l * rho_l) + smoothing(dx_phi_s * rho_s, k, Lambda, kind)
    # Phi_l_bau = (dx_phi_l * rho_l) + spectral_calc(smoothing(dx_phi_s**2, k, Lambda, kind), L, o=1, d=0) / (3 * H0**2 * a**2 / rho_0)
    
    # Phi_l = smoothing(dx_phi_s * (rho - rho_l), k, Lambda, kind)


    # #here is the full stress tensor; this is the object to be fitted for the EFT paramters
    tau_l = (kappa_l + Phi_l)
    return a, x, d1k, dc_l, dv_l, tau_l, P_nb_a, P_1l_a_tr
