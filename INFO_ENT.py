#!/usr/bin/env python3
#Libraries
import pyfftw
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm

def gauss_2D(l_box: float = 2*np.pi, size: int = 32, alpha: float = 1, norm: bool = True,
            visualise: bool = False, save_name: str = 'gauss_2D', seed:int=12345564631) -> np.array:

    """Returns a 2D Gaussian field of specified size from a power spectrum
        with scale alpha. If norm is True, the resulting field will have zero
        mean and unit standard deviation.
        Changed normalizations of the code.
        """
    np.random.seed(seed)
    k_fund = 2*np.pi/l_box

    # Set up the k-space grid
    k_ind = np.mgrid[:size, :size]- int((size+1)/2)
    k_idx = np.fft.fftshift(k_ind)*k_fund

    # Define the amplitude as a power law |k|^(alpha)
    amplitude = np.sqrt((size**6/l_box**3) * np.power(np.sqrt(k_idx[0]**2 + k_idx[1]**2), alpha))
    amplitude[0,0] = 0

    # Generate the field and transford to real space
    noise = np.random.normal(size = (size, size)) + 1j * np.random.normal(size = (size, size))
    gfield = np.fft.ifft2(noise * amplitude).real

    if visualise == True:
        # plt.rcParams.update({"text.usetex": True})
        # plt.rcParams.update({"font.family": "serif"})
        fig, ax = plt.subplots()
        ax.minorticks_on()
        plt.imshow(gfield, cmap='rainbow', extent=(0, 0.5, 0, 0.5))
        ax.set_xlabel(r'$x/L$', fontsize=18)
        ax.set_ylabel(r'$y/L$', fontsize=18)
        cbar = plt.colorbar()
        cbar.set_label(r'$\delta(\mathbf{x})$', fontsize=20)
        plt.savefig('./%s.png'%save_name,dpi=300)
        plt.close()

    else:
        pass

    return gfield


def ft(pix, threads=1):
    """This function performs the 2D FFT of a field in single precision using pyfftw.

        Parameters
    ----------
        pix : ndarray
            3D field to transform
        threads : int
            Number of threads to use (default=4)

    Returns
    -------
        pix_ft : ndarray
            Fourier transform of pix
    """

    # align arrays
    grid_2d = pix.shape
    a_in  = pyfftw.empty_aligned(grid_2d,dtype='complex64')
    a_out = pyfftw.empty_aligned(grid_2d,dtype='complex64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD', threads=threads,normalise_idft=False)

    # put input array into delta_r and perform FFTW
    a_in [:] = pix;  fftw_plan(a_in,a_out);
    return a_out

def power_spectrum(field:np.array, l_box:float, ngrid:int)-> tuple:
    """This Computes the power spectrum with the correct normalizations.
        TODO: Push to vectorise the loops

        Parameters
    ----------
        field : ndarray
            2D field to transform
        threads : int
            Number of threads to use (default=4)
        l_box:
            size of box
        ngrid:
            number_of_grid_points

    Returns
    -------
        pix_ft : ndarray
            Fourier transform of pix
    """
    grid_size = l_box/ngrid
    k_nyquist= np.pi/grid_size
    k_fund = 2*np.pi/l_box
    wind_const = 2*np.pi/(2*k_nyquist)
    Ngrid_K = int(ngrid/2)+1
    grid_volume=grid_size**3
    fft_norm=l_box**3/ngrid**6
    k_bins = np.zeros(Ngrid_K)
    power_k = np.zeros(Ngrid_K)
    nu_of_cells = np.zeros(Ngrid_K)
    print(field)
    field_k = ft(field)
    for a in range(ngrid):
        for b in range(ngrid):
            if (a>(ngrid-1)/2):
                k_x=(-ngrid+a)*k_fund
            else:
                k_x=a*k_fund
            if (b>(ngrid-1)/2):
                k_y=(-ngrid+b)*k_fund
            else:
                k_y=b*k_fund
            k_modulas=np.sqrt(k_x**2+k_y**2)
            if ((k_modulas >0.0)&(k_modulas<k_nyquist)):
                func_fft_norm = field_k[a,b].real**2 + field_k[a,b].imag**2
                idx_k=int( k_modulas/k_fund+0.5)
                k_bins[idx_k]+=k_modulas
                power_k[idx_k]+=func_fft_norm
                nu_of_cells[idx_k]+=1.0

    mask =  nu_of_cells>0
    nu_of_cells = nu_of_cells[mask]
    k_bins = k_bins[mask]/nu_of_cells
    power_k = power_k[mask]*fft_norm/nu_of_cells
    return k_bins, power_k, nu_of_cells


def rebin_PL(P_k,dk,number_of_modes):
    """Rebins the power spectra
            Parameters
    ----------
        P_k : 1darray
            1D field to rebin
        dk : int
            number of k-bins to put in one bin
        number_of_modes:
            number of modes in each power spectrum input bin, must be the same size as P_k
    Returns
    -------
        Rebinned spectrum
    """
    if(len(number_of_modes)!=len(P_k)):
        print("Something is off",len(number_of_modes),len(P_k))
        return 0
    if dk<2:
        return P_k
    nu_of_bins=int(len(P_k)//dk)
    P_k_av=np.ones(nu_of_bins-1)
    for idx in range(nu_of_bins-1):
        P_k_av[idx]=np.average(P_k[idx*dk:(idx+1)*dk],weights=number_of_modes[idx*dk:(idx+1)*dk])
    return  P_k_av


def rebin_all_P(k_bins , all_p,theory,dk,number_of_models):
    """wrapper around rebin_Pl to rebin multiple spectra
            Parameters
    ----------
        k_bins : 1darray
            1D field to rebin
        all_p : list of P-array
            will rebin each element of the list
         theory: thoery model,
         will be rebinned similar to the power spectra
        number_of_modes:
            number of modes in each power spectrum input bin, must be the same size as P_k
    Returns
    -------
        Rebinned spectrum
        """
    resukt = []
    for idx in tqdm(range(len(all_p))):
        resukt.append( rebin_PL(all_p[idx],dk,number_of_models))
    k_binned = rebin_PL(k_bins,dk,number_of_models)
    th_binned = rebin_PL(theory,dk,number_of_models)

    return k_binned, th_binned, resukt

def lamb_mik(alpha,k_min,k_max):
    xi_dd = np.abs(-(np.power(k_max,alpha + 4) - np.power(k_min,alpha + 4))/(alpha + 4))
    xi = (np.power(k_max,alpha + 2) - np.power(k_min,alpha + 2))/(alpha + 2)
    return np.sqrt(xi_dd/(2*np.pi*xi))

power_cont = []
l_box = 100
alpha = -0.5
N = 256 # size of field
# for seed in tqdm(np.random.randint(2**32,size=10)):
    # print(l_box, N, alpha, seed)
field = gauss_2D(l_box = l_box, size=N, alpha=alpha, seed = 123456)
k, power, nu_modes = power_spectrum(field,l_box,N)

    # k,power,nu_modes = power_spectrum(field,l_box,N)
    # power_cont.append(power)

# print(power_cont)

# k_binned, th_binned, resukt = rebin_all_P(k, power_cont,k**(-alpha), alpha, nu_modes)

# arr_binned = np.stack(resukt)
# arr_mean_bin = np.mean(arr_binned,axis=0)
# arr_std_bin = np.std(arr_binned,axis=0,ddof=1)
# arr_unbinned = np.stack(power_cont)
# arr_mean_un = np.mean(arr_unbinned,axis=0)
# arr_std_un = np.std(arr_unbinned,axis=0,ddof=1)
#
# ### Plot power spectra
# fig, ax = plt.subplots(figsize=(10,10))
# plt.errorbar(k ,arr_mean_un,arr_std_un,c="b",fmt="")
# plt.plot(k, k**(alpha), ls="--", c="k" )
# plt.yscale("log")
# plt.xlabel(r'$k$',fontsize=15)
# plt.ylabel(r'$P(k)$',fontsize=15)
# plt.show()



# def Get_lamb_dist(k_bins, power_all_arr, idx_real):
#     power_all_arr = np.stack(power_all_arr)
#     nu_of_realizations = power_all_arr.shape[0]
#     data_vec = power_all_arr - np.mean(power_all_arr,axis=0)
#     print(data_vec)
#     def model(lam):
#         return k_bins**(alpha_func(lam))
#
#     COV =  np.cov(data_vec.T)
#     def func_find_tails(one_posterior , paramater_arr):
#         moment_array = one_posterior*paramater_arr
#         mean = np.sum(moment_array)/np.sum(one_posterior)
#         std_prop = np.sqrt((np.sum(one_posterior*(paramater_arr - mean)**2))/np.sum(one_posterior))
#         max2= paramater_arr[one_posterior.argmax()]
#
#         if (np.abs(max2/mean-1)>.05):
#             cumulative_sum = np.cumsum(one_posterior)
#             # left = paramater_arr[cumulative_sum<0.18][-1]
#             # print(cum)
#             left_arr_cum_sum = paramater_arr[cumulative_sum<0.18]
#             if len(left_arr_cum_sum)==0:
#                 left = paramater_arr[0]
#             else:
#                 left = left_arr_cum_sum[-1]
#             right =paramater_arr[cumulative_sum<0.84][-1]
#             errm = mean - left
#             errp = right - mean
#         else:
#             errm = std_prop
#             errp = std_prop
#         return mean ,errm , errp
#
#     def compute_one_likeli(data, model, cov, nu_of_reals):
#         nu_of_bins = len(model)
#         correction= 1 - (nu_of_bins+1)/(nu_of_reals-1)
#         cov_det = np.log(np.linalg.det(cov))
#         try:
#             percision = correction * np.linalg.inv(cov)
#         except:
#             print("Singular matrix",cov_det)
#         vector=data-model
#         B=percision.dot(vector)
#         chisq=vector.dot(B)
#         loglikeli = -chisq/2
#         return loglikeli
#
#     lamb = np.linspace(0,1,1000)
#     lamb_posterior = np.zeros(len(lamb))
#     for idx_al in range(len(lamb)):
#         lamb_posterior [idx_al ] = compute_one_likeli(power_all_arr[idx_real,:],model(lamb[idx_al]), COV, nu_of_realizations)
#
#     lamb_posterior  = np.exp( lamb_posterior )
#     lamb_posterior = lamb_posterior/np.sum(lamb_posterior)
#     mean , left , right = func_find_tails(lamb_posterior , lamb)
#     return lamb, lamb_posterior, mean, left, right, COV
#


# alpha_interp = np.linspace(-1.99,2,1000)
# lamb_interp = np.zeros(len(alpha_interp))
# for idx in range(len(lamb_interp)):
#     lamb_interp[idx] = lamb_mik(alpha_interp[idx], k_binned[0], k_binned[-1])
#
# alpha_func = interp1d(lamb_interp, alpha_interp, fill_value="extrapolate", bounds_error=False)
#
# # alpha_arr, alpha_posterior, mean, left, right, COV = Get_lamb_dist(k_binned, resukt, 0)


# fig, ax = plt.subplots(figsize=(10,10))
# for idx_real in range(0,100,10):
#     alpha_arr,alpha_posterior,mean , left , right, COV = Get_lamb_dist(k_binned,resukt,idx_real)
#     plt.plot(alpha_arr ,alpha_posterior)
# lamb_true = lamb_mik(alpha,k_binned[0],k_binned[-1])
# plt.xlim(lamb_true-lamb_true/2,lamb_true+lamb_true/2)
# plt.axvline(x=lamb_true,c="k")
# plt.xlabel(r'$\alpha$',fontsize=15)
# plt.ylabel(r'$\mathcal{P}( \alpha )$',fontsize=15)
# plt.show()
