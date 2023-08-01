#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from functions import write_hier, read_hier, smoothing, spectral_calc

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
j = 2
Lambda = 3 * (2*np.pi)
kind = 'sharp'

def calc_new_hier(path, j, Lambda, kind, folder_name):
    a, dx, M0_nbody, M1_nbody, M2_nbody, C0_nbody, C1_nbody, C2_nbody = read_hier(path, j, folder_name)
    x = np.arange(0, 1.0, dx)
    L = 1.0
    Nx = x.size
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    rho_0 = 27.755 #this is the comoving background density
    rho_b = rho_0 / (a**3) #this is the physical background density
    H0 = 100

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
    Phi_l = (rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi_l2 - grad_phi2_l)

    # #here is the full stress tensor; this is the object to be fitted for the EFT paramters
    tau_l = (kappa_l + Phi_l)
    return a, x, k, sigma_l, pi_l, kappa_l, phi_l, Phi_l, tau_l, dc_l, dv_l, C1_nbody, M1_nbody, M0_nbody, M2_nbody, C2_nbody

def calc_old_hier(path, j, Lambda, kind):
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x = moments_file[:,0]
    M0_nbody = moments_file[:,2]
    M1_nbody = moments_file[:,4]
    M2_nbody = moments_file[:,6]
    C2_nbody = moments_file[:,7]


    L = 1.0
    Nx = x.size
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    rho_0 = 27.755 #this is the comoving background density
    rho_b = rho_0 / (a**3) #this is the physical background density
    H0 = 100

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
    Phi_l = (rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi_l2 - grad_phi2_l)

    # #here is the full stress tensor; this is the object to be fitted for the EFT paramters
    tau_l = (kappa_l + Phi_l)
    return a, x, k, sigma_l, pi_l, kappa_l, phi_l, Phi_l, tau_l, dc_l, dv_l, M1_nbody/M0_nbody, M1_nbody, M0_nbody, M2_nbody, C2_nbody


# a, x, sigma_l, pi_l, kappa_l, phi_l, Phi_l, tau_l, dc_l, dv_l = calc_new_hier(path, j, Lambda, kind, folder_name='/test/')
a, x, k, sigma_l, pi_l, kappa_l, phi_l, Phi_l, tau_l, dc_l, dv_l, C1_nbody, M1_nbody, M0_nbody, M2_nbody, C2_nbody = calc_new_hier(path, j, Lambda, kind, folder_name='/hierarchy/')
a, x2, k2, sigma_l2, pi_l2, kappa_l2, phi_l2, Phi_l2, tau_l2, dc_l2, dv_l2, C1_nbody2, M1_nbody2, M0_nbody2, M2_nbody2, C2_nbody2 = calc_new_hier(path, j, Lambda, kind, folder_name='/hierarchy_coarse/')
a, x3, k3, sigma_l3, pi_l3, kappa_l3, phi_l3, Phi_l3, tau_l3, dc_l3, dv_l3, C1_nbody3, M1_nbody3, M0_nbody3, M2_nbody3, C2_nbody3 = calc_new_hier(path, j, Lambda, kind, folder_name='/hierarchy_even_coarser/')
a, x4, k4, sigma_l4, pi_l4, kappa_l4, phi_l4, Phi_l4, tau_l4, dc_l4, dv_l4, C1_nbody4, M1_nbody4, M0_nbody4, M2_nbody4, C2_nbody4 = calc_new_hier(path, j, Lambda, kind, folder_name='/hierarchy_old/')


a, x_old, k_old, sigma_l_old, pi_l_old, kappa_l_old, phi_l_old, Phi_l_old, tau_l_old, dc_l_old, dv_l_old, C1_nbody_old, M1_nbody_old, M0_nbody_old, M2_nbody_old, C2_nbody_old = calc_old_hier(path, 0, Lambda, kind)
kappa_l = M0_nbody #- M1_nbody**2/M0_nbody
kappa_l_old = M0_nbody_old * a #-spectral_calc(M0_nbody_old, L=1, o=1, d=1) * (100 * a**(1/2)) #- M1_nbody_old**2/M0_nbody_old 


plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
ax.set_title(r'$a = {}$'.format(a), fontsize=20)
ax.set_xlabel(r'$x/L$', fontsize=18)
ax.set_ylabel(r'$[\tau]_{\Lambda}$', fontsize=18)

# ax.plot(x, M1_nbody, c='r', lw=2, label=r'New, $N={}$'.format(x.size))
# ax.plot(x, C1_nbody, c='b', lw=2, ls='dashdot', label=r'New, $N={}$'.format(x.size))
# ax.plot(x, M0_nbody, c='k', lw=2, ls='dashed', label=r'New, $N={}$'.format(x.size))

# ax.plot(x_old, dc_l_old, c='k', lw=2, label='Old')
# ax.plot(x, dc_l, c='r', lw=2, ls='dashdot', label=r'New, $N={}$'.format(x.size))
# ax.plot(x2, dc_l2, c='seagreen', lw=2, ls='dashed', label=r'New, $N={}$'.format(x2.size))
# ax.plot(x3, dc_l3, c='b', lw=2, ls='dotted', label=r'New, $N={}$'.format(x3.size))

# ax.plot(x_old, M0_nbody_old, c='k', lw=2, label='Old')
# ax.plot(x, M0_nbody, c='r', lw=2, ls='dashdot', label=r'New, $N={}$'.format(x.size))
# ax.plot(x2, M0_nbody2, c='seagreen', lw=2, ls='dashed', label=r'New, $N={}$'.format(x2.size))
# ax.plot(x3, M0_nbody3, c='b', lw=2, ls='dotted', label=r'New, $N={}$'.format(x3.size))
# ax.plot(x4, M0_nbody4, c='cyan', lw=2, ls='dotted', label=r'New, $N={}$'.format(x4.size))

# ax.plot(x_old, tau_l_old, c='k', lw=2, label='Old')
# ax.plot(x, tau_l, c='r', lw=2, ls='dashdot', label=r'New, $N_{{\mathrm{{bins}}}}={}$'.format(x.size))
# ax.plot(x3, tau_l3, c='b', lw=2, ls='dashed', label=r'New, $N_{{\mathrm{{bins}}}}={}$'.format(x3.size))
# ax.plot(x2, tau_l2, c='seagreen', lw=2, ls='dotted', label=r'New, $N_{{\mathrm{{bins}}}}={}$'.format(x2.size))
# # ax.plot(x4, tau_l4, c='cyan', lw=2, ls='dotted', label=r'New, $N={}$'.format(x4.size))



# ax.plot(x2, kappa_l2, c='b', ls='dashed', lw=2, label=r'New, $N={}$'.format(x2.size))
# ax.plot(x3, kappa_l3, c='seagreen', ls='dotted', lw=2, label=r'New, $N={}$'.format(x3.size))


ax.plot(x, kappa_l, c='b', lw=1.5, label='new')
ax.plot(x_old, kappa_l_old, c='k', lw=1.5, ls='dashed', label='old')



# ax.plot(x3, tau_l3, c='r', lw=2, ls='dotted', label=r'New, $N={}$'.format(x3.size))

# ax.plot(x_old, dv_l_old, c='k', lw=2, label='Old')
# ax.plot(x, dv_l, c='b', lw=2, ls='dashdot', label=r'New, $N={}$'.format(x.size))
# ax.plot(x2, dv_l2, c='seagreen', lw=2, ls='dashed', label=r'New, $N={}$'.format(x2.size))
# ax.plot(x3, dv_l3, c='r', lw=2, ls='dotted', label=r'New, $N={}$'.format(x3.size))


ax.legend(fontsize=12, bbox_to_anchor=(1,1))
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.ticklabel_format(scilimits=(-2, 3))
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('both')
ax.minorticks_on()

plt.show()
# plt.savefig('../plots/test/new_paper_plots/tau_l_{}.png'.format(j), bbox_inches='tight', dpi=150)
# plt.close()
