# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:11:30 2023

@author: Rasjied
"""
import numpy as np
import os
import glob
from astropy.io import fits
from scipy.signal import find_peaks_cwt
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy import signal
from pathlib import Path
from tqdm import tqdm
# %%


# The following code can be used to load the flux data from an order:
# formatting of exported datasets: Thar,Tungsten,Bias,Dark,Object, SNR_A, darkflat_A
    
    # For example if you would like the flux of the object in order 3:


main_folder_A = r'C:\Users\post\OneDrive\Documenten\UvA-VU\Jaar 2\Practicum zonnefysica\NSP2\Flux_raw_sunLimbA\Flux_raw_sunLimbA'
main_folder_B = r'C:\Users\post\OneDrive\Documenten\UvA-VU\Jaar 2\Practicum zonnefysica\NSP2\Flux_raw_sunLimbB\Flux_raw_sunLimbB'

N_order = 7
data_order_N_A = np.loadtxt(os.path.join(main_folder_A, "data_raw_order_{}.csv").format(N_order),  delimiter=',')
data_order_N_B = np.loadtxt(os.path.join(main_folder_B, "data_raw_order_{}.csv").format(N_order),  delimiter=',')



x_pixelvalues_A = np.arange(len(data_order_N_A[0]))
thar_A = data_order_N_A[0]
tungstenflat_A = data_order_N_A[1]
bias_A = data_order_N_A[2]
dark_A = data_order_N_A[3]
flux_object_A = data_order_N_A[4]
SNR_A = data_order_N_A[5]
darkflat_A = data_order_N_A[6]


x_pixelvalues_B = np.arange(len(data_order_N_B[0]))
thar_B = data_order_N_B[0]
tungstenflat_B = data_order_N_B[1]
bias_B = data_order_N_B[2]
dark_B = data_order_N_B[3]
flux_object_B = data_order_N_B[4]
SNR_B = data_order_N_B[5]
darkflat_B = data_order_N_B[6]
"""
# different absorption spectra with A
plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(x_pixelvalues_A,thar_A, label = 'ThAr')
plt.plot(x_pixelvalues_A,tungstenflat_A, label = 'Tungsten')
plt.plot(x_pixelvalues_A,bias_A, label = 'Bias')
plt.plot(x_pixelvalues_A,dark_A, label = 'Dark_A')
plt.plot(x_pixelvalues_A,flux_object_A, label = 'Object')
plt.plot(x_pixelvalues_A,SNR_A, label = 'SNR')
plt.plot(x_pixelvalues_A,darkflat_A, label = 'darkflat')
plt.legend()
plt.show()


# different absorption spectra with B
plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(x_pixelvalues_B,thar_B, label = 'ThAr')
plt.plot(x_pixelvalues_B,tungstenflat_B, label = 'Tungsten')
plt.plot(x_pixelvalues_B,bias_B, label = 'Bias')
plt.plot(x_pixelvalues_B,dark_B, label = 'Dark')
plt.plot(x_pixelvalues_B,flux_object_B, label = 'Object')
plt.plot(x_pixelvalues_B,SNR_B, label = 'SNR')
plt.plot(x_pixelvalues_B,darkflat_B, label = 'darkflat')
plt.legend()
plt.show()
"""

# %% Golflengte Kalibratie met polynoomfit

wavelength_list =   [5912.0853,
                     5914.7139,
                     5916.5992,
                     5928.8130,
                     5888.584,
                     5882.6242,
                     5885.7016,
                     5860.3102,
                     5834.2633,
                     5938.8252,
                     5908.9257,
                     5891.451,
                     ]

x_list =            [3271,
                     3211,
                     3165,
                     2878,
                     3808,
                     3942,
                     3874,
                     4437,
                     5001,
                     2546,
                     3344,
                     3744,
                     ]

uncertainty_x =     [0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5,]

#calibration points
plt.plot(x_pixelvalues_A,thar_A)
plt.scatter(x_list,thar_A[x_list], c='red', label = 'calibration points' )
for index in range(len(x_list)):
    plt.text(x_list[index]+20, thar_A[x_list][index]+20, wavelength_list[index], size=8)
plt.legend()
plt.show()

# %% Polynomial fit for wavelength calibration

fit_order = 2 #3 or higher goes wrong
fit_1 = np.polynomial.polynomial.polyfit(x_list,wavelength_list,fit_order,w=uncertainty_x)
print(fit_1)

# x & y coordinaten van de fit
wavelength_object = []
for x in x_pixelvalues_A:
    y = 0
    # Calculate y_coordinate
    for n in range(len(fit_1)):
        y += fit_1[n] * (x)**n       
    # Save coordinates
    wavelength_object.append(y)   

#  Residuals berekenen
residuals = []
for i, x_value in enumerate(x_list):
    # Bereken de voorspelde waarde met de fit-coëfficiënten
    predicted_wavelength = sum(fit_1[n] * (x_value)**n for n in range(len(fit_1)))
    
    # Bereken het residual door het verschil te nemen tussen de werkelijke en voorspelde waarde
    residual = wavelength_list[i] - predicted_wavelength
    residuals.append(residual)
    
# lekker plotten:
#calibration fit
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [7, 2]})
fig.subplots_adjust(hspace=0)

ax1.set_title("Wavelength calibration fit (x-pixels vs wavelength)")
ax1.plot(x_pixelvalues_A, wavelength_object)
ax1.set_ylabel("Wavelength [Angstrom]")
ax1.errorbar(x_list, wavelength_list, yerr=np.abs(uncertainty_x*np.array(fit_1[1])), fmt='o', ecolor='red', capsize=3, label='Residuals with error bars')
ax1.scatter(x_list,wavelength_list, c='blue')

ax2.errorbar(x_list, residuals, yerr=np.abs(uncertainty_x*np.array(fit_1[1])), fmt='o', ecolor='red', capsize=3, label='Residuals with error bars')
ax2.scatter(x_list,residuals)
ax2.set_xlabel("Pixels")
ax2.set_ylabel("Residuals [Angstrom]")
ax2.axhline(0, color='black', linestyle='--', linewidth=1, label = 'model')
ax2.axhline(fit_1[1], color='gray', linestyle='--', linewidth=1, label = '1 pixel difference')
ax2.axhline(-1*fit_1[1], color='gray', linestyle='--', linewidth=1)
for index in range(len(x_list)):
    ax2.text(x_list[index], residuals[index], wavelength_list[index], size=8)
plt.legend()
plt.show()



# %% first order flux correction- not normalized:

plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(wavelength_object,(flux_object_A-dark_A)/(tungstenflat_A-darkflat_A))
plt.ylim(0,)
plt.show()

plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(wavelength_object,(flux_object_B-dark_B)/(tungstenflat_B-darkflat_B))
plt.ylim(0,)
plt.show()

# %% Nu aan jullie om lekker te normaliseren:
#first order flux correction- normalized:

fit_order_norm = 10
fit_2_A = np.polynomial.polynomial.polyfit(wavelength_object,(flux_object_A-dark_A)/(tungstenflat_A-darkflat_A),fit_order_norm)

# x & y coordinaten van de fit
normalisation_fit_A= []
for x in wavelength_object:
    y = 0
    # Calculate y_coordinate
    for n in range(len(fit_2_A)):
        y += (fit_2_A[n] * (x)**n) + 0.1
    # Save coordinates
    normalisation_fit_A.append(y)   

fit_2_B = np.polynomial.polynomial.polyfit(wavelength_object,(flux_object_B-dark_B)/(tungstenflat_B-darkflat_B),fit_order_norm)

# x & y coordinaten van de fit
normalisation_fit_B= []
for x in wavelength_object:
    y = 0
    # Calculate y_coordinate
    for n in range(len(fit_2_B)):
        y += (fit_2_B[n] * (x)**n) + 0.1
    # Save coordinates
    normalisation_fit_B.append(y)   

flux_object_norm_A = (flux_object_A-dark_A)/((tungstenflat_A-darkflat_A)*normalisation_fit_A)
flux_object_norm_B = (flux_object_B-dark_B)/((tungstenflat_B-darkflat_B)*normalisation_fit_B)
#%%

Na_D1_A_wavelength = []
Na_D1_A_intensity = []
Na_D1_A_error = []

Na_D1_B_wavelength = []
Na_D1_B_intensity = []
Na_D1_B_error = []

# calculate rotation period with Mg-b1
for i in range(len(wavelength_object)):
    if 5889.75 < wavelength_object[i] < 5890.75:
        Na_D1_A_wavelength.append(wavelength_object[i])
        Na_D1_A_intensity.append(flux_object_norm_A[i])
        Na_D1_A_error.append(flux_object_norm_A[i]/SNR_A[i])

for i in range(len(wavelength_object)):
        if 5889.75 < wavelength_object[i] < 5890.75:
            Na_D1_B_wavelength.append(wavelength_object[i])
            Na_D1_B_intensity.append(flux_object_norm_B[i])
            Na_D1_B_error.append(flux_object_norm_B[i]/SNR_B[i])


def normal_distribution(x, std, avg, c):
    return -(np.e**(-(((x-avg)/std)**2)/2))/(std*np.sqrt(2*np.pi))+c

popt_n_A, pcov_n_A = curve_fit(normal_distribution, Na_D1_A_wavelength, Na_D1_A_intensity, p0=[1, 5890.20, 1], sigma=Na_D1_A_error)
std_opt_A , avg_opt_A, c_opt_A= popt_n_A
error_std_cov_A, error_avg_cov_A, error_c_cov_A = pcov_n_A
print(f'minimum gaussische functie {avg_opt_A}')
print(f'sqrt variantie en sigma std{(pcov_n_A[0][0]**(1/2)), std_opt_A}')

popt_n_B, pcov_n_B = curve_fit(normal_distribution, Na_D1_B_wavelength, Na_D1_B_intensity, p0=[1, 5890.30, 1], sigma=Na_D1_B_error)
std_opt_B , avg_opt_B, c_opt_B= popt_n_B
error_std_cov_B, error_avg_cov_B, error_c_cov_B = pcov_n_B
print(f'minimum gaussische functie {avg_opt_B}')
print(f'sqrt variantie en sigma std{(pcov_n_B[0][0]**(1/2)), std_opt_B}')

"""
fit_Na_D1_A= np.polynomial.polynomial.polyfit(Na_D1_A_wavelength,Na_D1_A_intensity, 5)
Na_D1_A = []
for x in Na_D1_A_wavelength:
    y = 0
    # Calculate y_coordinate
    for n in range(len(fit_Na_D1_A)):
        y += (fit_Na_D1_A[n] * (x)**n)
    # Save coordinates
    Na_D1_A.append(y) 

fit_Na_D1_B =  np.polynomial.polynomial.polyfit(Na_D1_B_wavelength,Na_D1_B_intensity, 5)
Na_D1_B = []
for x in Na_D1_B_wavelength:
    y = 0
    # Calculate y_coordinate
    for n in range(len(fit_Na_D1_B)):
        y += (fit_Na_D1_B[n] * (x)**n)
    # Save coordinates
    Na_D1_B.append(y) 
"""
#Na_D1
plt.subplots(figsize=(16.5, 11.7), dpi=300)
plt.plot(wavelength_object, flux_object_norm_A, linewidth=1, label="Dataset A")
plt.plot(wavelength_object, flux_object_norm_B, linewidth=1, label="Dataset B")
plt.plot(Na_D1_A_wavelength, (normal_distribution(Na_D1_A_wavelength, std_opt_A, avg_opt_A, c_opt_A)), label='Gaussische fitfunctie A')
plt.plot(Na_D1_B_wavelength, (normal_distribution(Na_D1_B_wavelength, std_opt_B, avg_opt_B, c_opt_B)), label='Gaussische fitfunctie B')

# plt.plot(Na_D1_A_wavelength, (second_orde_poly(Na_D1_A_wavelength, a_opt, b_opt, lambda_0_opt)), label='Tweede orde polynoom fitfunctie')
plt.errorbar(wavelength_object, flux_object_norm_A, yerr=flux_object_norm_A/SNR_A, markersize='1', fmt='.', ecolor='red', elinewidth=0.5)
plt.errorbar(wavelength_object, flux_object_norm_B, yerr=flux_object_norm_B/SNR_B, markersize='1', fmt='.', ecolor='red', elinewidth=0.5)
plt.ylim(0,)
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel("Normalized intensity")
plt.legend(loc=2, prop={'size': 6})
plt.show()

#####
"""
plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(wavelength_object,(flux_object_A-dark_A)/(tungstenflat_A-darkflat_A), label = 'absorption Mg-b1 A')
plt.plot(wavelength_object,(flux_object_B-dark_B)/(tungstenflat_B-darkflat_B), label = 'absorption Mg-b1 B')
plt.legend()
plt.show()

plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(wavelength_object, flux_object_norm_A, linewidth=1, label="Dataset A", color = 'green')
plt.plot(wavelength_object, flux_object_norm_B, linewidth=1, label="Dataset B", color = 'lime')
# plt.plot(Na_D1_A_wavelength, Na_D1_A, label='fitfunction A', linewidth=1, color = 'crimson')
# plt.plot(Na_D1_B_wavelength, Na_D1_B, label='fitfunction B', linewidth=1, color = 'fuchsia')
plt.plot(wavelength_object, flux_object_norm_B)
plt.ylim(0,)
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel("Normalized intensity")
plt.legend()
plt.show()
"""

R=696342000
error_R = 65000
c=299792458
error_c = 1

#gaussian results 

min_A_g = avg_opt_A 
error_A_g = pcov_n_A[2][2]**(1/2)

min_B_g = avg_opt_B
error_B_g = pcov_n_B[2][2]**(1/2)


t_formule = (2*np.pi*R/c)*(min_A_g +min_B_g)/(min_B_g - min_A_g)
def omlooptijd(min_A_g, error_A_g, min_B_g, error_B_g):
    lambda_gem = (min_A_g+min_B_g)/2
    delta_lambda = abs(lambda_gem - min_A_g)
    v = c * (delta_lambda/lambda_gem)
    T = ((2*np.pi*R)/v)
    print(f"{T} is de omlooptijd in seconden")
    print(f"{T/(60*60*24)} is de omlooptijd in dagen")

    error_T = (((2*np.pi*error_R/c)*((min_A_g+min_B_g)/(min_B_g-min_A_g)))**2 +
        ((2*np.pi*R/c)*((2*min_B_g*error_A_g)/((min_A_g-min_B_g)**2)))**2 +
        ((2*np.pi*R/c)*((2*min_A_g*error_B_g)/((min_B_g-min_A_g)**2)))**2)**(1/2)
    print(f"{error_T} is de error van de omlooptijd in seconden")
    print(f"{error_T/(60*60*24)} is de error van de omlooptijd in dagen")
    print(((2*np.pi*error_R/c)*((min_A_g+min_B_g)/(min_B_g-min_A_g)))**2)
    print(((2*np.pi*R/c)*((2*min_B_g*error_A_g)/((min_A_g-min_B_g)**2)))**2)
    print(((2*np.pi*R/c)*((2*min_A_g*error_B_g)/((min_B_g-min_A_g)**2)))**2)
omlooptijd(min_A_g, error_A_g, min_B_g, error_B_g)

#%%
Na_D2_A_wavelength = []
Na_D2_A_intensity = []
Na_D2_A_error = []

Na_D2_B_wavelength = []
Na_D2_B_intensity = []
Na_D2_B_error = []

# calculate rotation period with Mg-b1
for i in range(len(wavelength_object)):
    if 5895.70 < wavelength_object[i] <  5896.60:
        Na_D2_A_wavelength.append(wavelength_object[i])
        Na_D2_A_intensity.append(flux_object_norm_A[i])
        Na_D2_A_error.append(flux_object_norm_A[i]/SNR_A[i])

for i in range(len(wavelength_object)):
        if 5895.70 < wavelength_object[i] <  5896.50:
            Na_D2_B_wavelength.append(wavelength_object[i])
            Na_D2_B_intensity.append(flux_object_norm_B[i])
            Na_D2_B_error.append(flux_object_norm_B[i]/SNR_B[i])


def normal_distribution(x, std, avg, c):
    return -(np.e**(-(((x-avg)/std)**2)/2))/(std*np.sqrt(2*np.pi))+c



popt_n_A, pcov_n_A = curve_fit(normal_distribution, Na_D2_A_wavelength, Na_D2_A_intensity, p0=[1, 5896.10, 1], sigma=Na_D2_A_error)
std_opt_A , avg_opt_A, c_opt_A= popt_n_A
error_std_cov_A, error_avg_cov_A, error_c_cov_A = pcov_n_A
print(f'minimum gaussische functie {avg_opt_A}')
print(f'sqrt variantie en sigma std{(pcov_n_A[0][0]**(1/2)), std_opt_A}')

popt_n_B, pcov_n_B = curve_fit(normal_distribution, Na_D2_B_wavelength, Na_D2_B_intensity, p0=[1, 5896.20, 1], sigma=Na_D2_B_error)
std_opt_B , avg_opt_B, c_opt_B= popt_n_B
error_std_cov_B, error_avg_cov_B, error_c_cov_B = pcov_n_B
print(f'minimum gaussische functie {avg_opt_B}')
print(f'sqrt variantie en sigma std{(pcov_n_B[0][0]**(1/2)), std_opt_B}')

"""
fit_Na_D2_A= np.polynomial.polynomial.polyfit(Na_D2_A_wavelength,Na_D2_A_intensity, 5)
Na_D2_A = []
for x in Na_D2_A_wavelength:
    y = 0
    # Calculate y_coordinate
    for n in range(len(fit_Na_D2_A)):
        y += (fit_Na_D2_A[n] * (x)**n)
    # Save coordinates
    Na_D2_A.append(y) 

fit_Na_D2_B =  np.polynomial.polynomial.polyfit(Na_D2_B_wavelength,Na_D2_B_intensity, 5)
Na_D2_B = []
for x in Na_D2_B_wavelength:
    y = 0
    # Calculate y_coordinate
    for n in range(len(fit_Na_D2_B)):
        y += (fit_Na_D2_B[n] * (x)**n)
    # Save coordinates
    Na_D2_B.append(y) 
"""
#Na_D2
plt.subplots(figsize=(16.5, 11.7), dpi=300)
plt.plot(wavelength_object, flux_object_norm_A, linewidth=1, label="Dataset A")
plt.plot(wavelength_object, flux_object_norm_B, linewidth=1, label="Dataset B")
plt.plot(Na_D2_A_wavelength, (normal_distribution(Na_D2_A_wavelength, std_opt_A, avg_opt_A, c_opt_A)), label='Gaussische fitfunctie A')
plt.plot(Na_D2_B_wavelength, (normal_distribution(Na_D2_B_wavelength, std_opt_B, avg_opt_B, c_opt_B)), label='Gaussische fitfunctie B')

# plt.plot(Na_D2_A_wavelength, (second_orde_poly(Na_D2_A_wavelength, a_opt, b_opt, lambda_0_opt)), label='Tweede orde polynoom fitfunctie')
plt.errorbar(wavelength_object, flux_object_norm_A, yerr=flux_object_norm_A/SNR_A, markersize='1', fmt='.', ecolor='red', elinewidth=0.5)
plt.errorbar(wavelength_object, flux_object_norm_B, yerr=flux_object_norm_B/SNR_B, markersize='1', fmt='.', ecolor='red', elinewidth=0.5)
plt.ylim(0,)
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel("Normalized intensity")
plt.legend(loc=2, prop={'size': 6})
plt.show()

#####
"""
plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(wavelength_object,(flux_object_A-dark_A)/(tungstenflat_A-darkflat_A), label = 'absorption Mg-b1 A')
plt.plot(wavelength_object,(flux_object_B-dark_B)/(tungstenflat_B-darkflat_B), label = 'absorption Mg-b1 B')
plt.legend()
plt.show()

plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(wavelength_object, flux_object_norm_A, linewidth=1, label="Dataset A", color = 'green')
plt.plot(wavelength_object, flux_object_norm_B, linewidth=1, label="Dataset B", color = 'lime')
# plt.plot(Na_D2_A_wavelength, Na_D2_A, label='fitfunction A', linewidth=1, color = 'crimson')
# plt.plot(Na_D2_B_wavelength, Na_D2_B, label='fitfunction B', linewidth=1, color = 'fuchsia')
plt.plot(wavelength_object, flux_object_norm_B)
plt.ylim(0,)
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel("Normalized intensity")
plt.legend()
plt.show()
"""

R=696342000
error_R = 65000
c=299792458
error_c = 1

#gaussian results 

min_A_g = avg_opt_A 
error_A_g = pcov_n_A[2][2]**(1/2)

min_B_g = avg_opt_B
error_B_g = pcov_n_B[2][2]**(1/2)

#polynomial result

# min_A_p = lambda_0_opt
# error_A_p = pcov_p[2][2]**(1/2)

t_formule = (2*np.pi*R/c)*(min_A_g +min_B_g)/(min_B_g - min_A_g)
def omlooptijd(min_A_g, error_A_g, min_B_g, error_B_g):
    lambda_gem_2 = (min_A_g+min_B_g)/2
    delta_lambda_2 = abs(lambda_gem_2 - min_A_g)
    v_2= c * (delta_lambda_2/lambda_gem_2)
    T_2 = ((2*np.pi*R)/v_2)
    print(f"{T_2} is de omlooptijd in seconden")
    print(f"{T_2/(60*60*24)} is de omlooptijd in dagen")

    error_T = (((2*np.pi*error_R/c)*((min_A_g+min_B_g)/(min_B_g-min_A_g)))**2 +
        ((2*np.pi*R/c)*((2*min_B_g*error_A_g)/((min_A_g-min_B_g)**2)))**2 +
        ((2*np.pi*R/c)*((2*min_A_g*error_B_g)/((min_B_g-min_A_g)**2)))**2)**(1/2)
    print(f"{error_T} is de error van de omlooptijd in seconden")
    print(f"{error_T/(60*60*24)} is de error van de omlooptijd in dagen")
    print(((2*np.pi*error_R/c)*((min_A_g+min_B_g)/(min_B_g-min_A_g)))**2)
    print(((2*np.pi*R/c)*((2*min_B_g*error_A_g)/((min_A_g-min_B_g)**2)))**2)
    print(((2*np.pi*R/c)*((2*min_A_g*error_B_g)/((min_B_g-min_A_g)**2)))**2)
omlooptijd(min_A_g, error_A_g, min_B_g, error_B_g)
# Na_D1_A_wavelength = []
# Na_D1_A_intensity = []
# Na_D1_B_wavelength = []
# Na_D1_B_intensity = []

# # calculate rotztion period with Na-D1
# for i in range(len(wavelength_object)):
#     if 5889.9 < wavelength_object[i] < 5891.0:
#         Na_D1_A_wavelength.append(wavelength_object[i])
#         Na_D1_A_intensity.append(flux_object_norm_A[i])
#         Na_D1_B_wavelength.append(wavelength_object[i])
#         Na_D1_B_intensity.append(flux_object_norm_B[i])


# fit_Na_D1_A= np.polynomial.polynomial.polyfit(Na_D1_A_wavelength,Na_D1_A_intensity, 5)
# Na_D1_A = []
# for x in Na_D1_A_wavelength:
#     y = 0
#     # Calculate y_coordinate
#     for n in range(len(fit_Na_D1_A)):
#         y += (fit_Na_D1_A[n] * (x)**n)
#     # Save coordinates
#     Na_D1_A.append(y) 

# fit_Na_D1_B =  np.polynomial.polynomial.polyfit(Na_D1_B_wavelength,Na_D1_B_intensity, 5)
# Na_D1_B = []
# for x in Na_D1_B_wavelength:
#     y = 0
#     # Calculate y_coordinate
#     for n in range(len(fit_Na_D1_B)):
#         y += (fit_Na_D1_B[n] * (x)**n)
#     # Save coordinates
#     Na_D1_B.append(y) 

# #Na-D1
# plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
# plt.plot(wavelength_object,(flux_object_A-dark_A)/(tungstenflat_A-darkflat_A), label = 'absorption Na-D1 A')
# plt.plot(wavelength_object,(flux_object_B-dark_B)/(tungstenflat_B-darkflat_B), label = 'absorption Na-D1 B')
# plt.legend()
# plt.show()

# plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
# plt.plot(wavelength_object, flux_object_norm_A, linewidth=1, label="Dataset A", color = 'lightblue')
# plt.plot(wavelength_object, flux_object_norm_B, linewidth=1, label="Dataset B", color = 'blue')
# plt.plot(Na_D1_A_wavelength, Na_D1_A, label='fitfunctie A', linewidth=1, color = 'yellow')
# plt.plot(Na_D1_B_wavelength, Na_D1_B, label='fitfunctie B', linewidth=1, color = 'orange')
# plt.plot(wavelength_object, flux_object_norm_B)
# plt.ylim(0,)
# plt.xlabel('Wavelength (Angstrom)')
# plt.ylabel("Genormaliseerde Intensiteit")
# plt.legend()
# plt.show()


# min_Na_D1_A=Na_D1_A_wavelength[np.where(Na_D1_A == min(Na_D1_A))[0][0]]
# min_Na_D1_B=Na_D1_B_wavelength[np.where(Na_D1_B == min(Na_D1_B))[0][0]]
# print(np.where(Na_D1_A == min(Na_D1_A))[0][0], min(Na_D1_A))
# print(f"De golflengte van Na-D1 dataset A is {min_Na_D1_A}")
# print(np.where(Na_D1_B == min(Na_D1_B))[0][0], min(Na_D1_B))
# print(f"De golflengte van Na-D1 dataset B is {min_Na_D1_B}")

# R=696340000
# c=299792458

# lambda_gem_1 = (min_Na_D1_B + min_Na_D1_A)/2
# delta_lambda_1 = abs(min_Na_D1_B - lambda_gem_1)

# v_1 = c* (delta_lambda_1/lambda_gem_1)

# print(lambda_gem_1, delta_lambda_1, v_1)


# T_1 = ((2*np.pi*R)/v_1)
# print(f"{T_1} is the rotation time in seconds  calculated from Na-D1 line")
# print(f"{T_1/(60*60*24)} is the rotation time in days  calculated from Na-D1 line")

# %%
# Na_D2_A_wavelength = []
# Na_D2_A_intensity = []
# Na_D2_B_wavelength = []
# Na_D2_B_intensity = []

# # calculate rotztion period with Na-D2
# for i in range(len(wavelength_object)):
#     if 5895.9 < wavelength_object[i] < 5896.6:
#         Na_D2_A_wavelength.append(wavelength_object[i])
#         Na_D2_A_intensity.append(flux_object_norm_A[i])
#         Na_D2_B_wavelength.append(wavelength_object[i])
#         Na_D2_B_intensity.append(flux_object_norm_B[i])


# fit_Na_D2_A= np.polynomial.polynomial.polyfit(Na_D2_A_wavelength,Na_D2_A_intensity, 5)
# Na_D2_A = []
# for x in Na_D2_A_wavelength:
#     y = 0
#     # Calculate y_coordinate
#     for n in range(len(fit_Na_D2_A)):
#         y += (fit_Na_D2_A[n] * (x)**n)
#     # Save coordinates
#     Na_D2_A.append(y) 

# fit_Na_D2_B =  np.polynomial.polynomial.polyfit(Na_D2_B_wavelength,Na_D2_B_intensity, 5)
# Na_D2_B = []
# for x in Na_D2_B_wavelength:
#     y = 0
#     # Calculate y_coordinate
#     for n in range(len(fit_Na_D2_B)):
#         y += (fit_Na_D2_B[n] * (x)**n)
#     # Save coordinates
#     Na_D2_B.append(y) 

# #Na-D2
# plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
# plt.plot(wavelength_object,(flux_object_A-dark_A)/(tungstenflat_A-darkflat_A), label = 'absorption Na-D2 A')
# plt.plot(wavelength_object,(flux_object_B-dark_B)/(tungstenflat_B-darkflat_B), label = 'absorption Na-D2 B')
# plt.legend()
# plt.show()

# plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
# plt.plot(wavelength_object, flux_object_norm_A, linewidth=1, label="Dataset A", color = 'green')
# plt.plot(wavelength_object, flux_object_norm_B, linewidth=1, label="Dataset B", color = 'lime')
# plt.plot(Na_D2_A_wavelength, Na_D2_A, label='fitfunction A', linewidth=1, color = 'crimson')
# plt.plot(Na_D2_B_wavelength, Na_D2_B, label='fitfunction B', linewidth=1, color = 'fuchsia')
# plt.plot(wavelength_object, flux_object_norm_B)
# plt.ylim(0,)
# plt.xlabel('Wavelength (Angstrom)')
# plt.ylabel("Normalized intensity")
# plt.legend()
# plt.show()


# min_Na_D2_A=Na_D2_A_wavelength[np.where(Na_D2_A == min(Na_D2_A))[0][0]]
# min_Na_D2_B=Na_D2_B_wavelength[np.where(Na_D2_B == min(Na_D2_B))[0][0]]
# print(np.where(Na_D2_A == min(Na_D2_A))[0][0], min(Na_D2_A))
# print(f"The wavelegth of Na-D2  in dataset A is {min_Na_D2_A}")
# print(np.where(Na_D2_B == min(Na_D2_B))[0][0], min(Na_D2_B))
# print(f"The wavelength of Na-D2 in dataset B is {min_Na_D2_B}")

# R=696340000
# c=299792458

# lambda_gem_2 = (min_Na_D2_B + min_Na_D2_A)/2
# delta_lambda_2 = abs(min_Na_D2_B - lambda_gem_2)

# v_2 = c* (delta_lambda_2/lambda_gem_2)

# print(lambda_gem_2, delta_lambda_2, v_2)


# T_2 = ((2*np.pi*R)/v_2)
# print(f"{T_2} is the rotation time in seconds  calculated from Na-D2 line")
# print(f"{T_2/(60*60*24)}is the rotation time in days  calculated from Na-D2 line")









