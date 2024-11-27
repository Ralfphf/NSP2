# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:11:30 2023

@author: Rasjied
"""
import numpy as np
from scipy.stats import norm
import os
import glob
from astropy.io import fits
import numpy as np
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


main_folder_A = r'C:\Users\Ralfy\OneDrive - UvA\Natuur- & Sterrenkunde Bachelor\2e Jaar\NSP2 & ECPC\NSP2\Flux_raw_sunLimbA\Flux_raw_sunLimbA'
main_folder_B = r'C:\Users\Ralfy\OneDrive - UvA\Natuur- & Sterrenkunde Bachelor\2e Jaar\NSP2 & ECPC\NSP2\Flux_raw_sunLimbB\Flux_raw_sunLimbB'

N_order = 3
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



# %% Golflengte Kalibratie met polynoomfit

wavelength_list =   [6677.2817,
                     6538.1120,
                     6583.9059, 
                     6604.8534,
                     6591.4845,
                     6588.5396,
                     6554.1603,
                     6577.2145,
                     6684.2930,
                     6666.3589,
                     6664.0510,
                     6662.2686,
                     6660.6762,
                     6643.6976]

x_list =            [1752,
                     4656,
#hoogste punt        3747,
                     3748,                    
                     3319,
                     3594,
                     3654,
                     4343,
                     3883,
                     1592,
                     1997,
                     2048,
                     2088,
                     2124,
                     2496]

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
                     0.5,
                     0.5,
                     0.5]

'''
plt.plot(x_pixelvalues_A,thar_A)
plt.scatter(x_list,thar_A[x_list], c='red', label = 'calibration points' )
for index in range(len(x_list)):
    plt.text(x_list[index]+20, thar_A[x_list][index]+20, wavelength_list[index], size=8)
plt.legend()
plt.show()
'''
# %% Polynomial fit for wavelength calibration

fit_order = 4
#5 of hoger valt buiten 
fit_1 = np.polynomial.polynomial.polyfit(x_list,wavelength_list,fit_order,w=uncertainty_x)

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
'''
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [7, 2]})
fig.subplots_adjust(hspace=0)

ax1.set_title("Wavelength calibration fit (x-pixels vs wavelength)")
ax1.plot(x_pixelvalues_A, wavelength_object)
ax1.set_ylabel("Wavelength [Angstrom]")
ax1.errorbar(x_list, wavelength_list, yerr=np.abs(uncertainty_x*np.array(fit_1[1])), fmt='o', ecolor='red', capsize=3, label='Residuals with error bars')
ax1.scatter(x_list,wavelength_list, c='blue')



ax2.errorbar(x_list, residuals, yerr=np.abs(uncertainty_x*np.array(fit_1[1])), fmt='o', ecolor='red', capsize=3, label='Residuals with error bars')
ax2.scatter(x_list,residuals)
ax2.set_ylabel("Pixels")
ax2.set_ylabel("Residuals [Angstrom]")
ax2.axhline(0, color='black', linestyle='--', linewidth=1, label = 'model')
ax2.axhline(fit_1[1], color='gray', linestyle='--', linewidth=1, label = '1 pixel difference')
ax2.axhline(-1*fit_1[1], color='gray', linestyle='--', linewidth=1)
for index in range(len(x_list)):
    ax2.text(x_list[index], residuals[index], wavelength_list[index], size=8)
plt.legend()
plt.show()
'''



# %% first order flux correction:

plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.errorbar(wavelength_object,(flux_object_A-dark_A)/(tungstenflat_A-darkflat_A))
# yerr=(flux_object_A-dark_A)/((tungstenflat_A-darkflat_A)*SNR_A), markersize='1', fmt='.', ecolor='red', elinewidth=0.5)
plt.ylim(0,)
plt.show()

plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(wavelength_object,(flux_object_B-dark_B)/(tungstenflat_B-darkflat_B))
plt.ylim(0,)
plt.show()

# %% Nu aan jullie om lekker te normaliseren:

fit_order_norm = 10
fit_2_A = np.polynomial.polynomial.polyfit(wavelength_object,(flux_object_A-dark_A)/(tungstenflat_A-darkflat_A),fit_order_norm)

# x & y coordinaten van de fit
normalisation_fit_A= []
error_norm_fit_A=[]
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

H_alpha_A_wavelength = []
H_alpha_A_intensity = []
H_alpha_B_wavelength = []
H_alpha_B_intensity = []
H_alpha_A_error = []
H_alpha_B_error = []

for i in range(len(wavelength_object)):
    if 6562.1 < wavelength_object[i] < 6563.6:
        H_alpha_A_wavelength.append(wavelength_object[i])
        H_alpha_A_intensity.append(flux_object_norm_A[i])
        H_alpha_B_wavelength.append(wavelength_object[i])
        H_alpha_B_intensity.append(flux_object_norm_B[i])
        H_alpha_A_error.append(flux_object_norm_A[i]/SNR_A[i])
        H_alpha_B_error.append(flux_object_norm_B[i]/SNR_B[i])


def normal_distribution(x, std, avg):
    return y == (np.e**(-(((x-avg)/std)**2)/2))/(std*np.sqrt(2*np.pi))

curve_fit(normal_distribution, H_alpha_A_wavelength, H_alpha_A_intensity, sigma=H_alpha_A_error)
print(curve_fit)

fit_H_alpha_A = np.polynomial.polynomial.polyfit(H_alpha_A_wavelength,H_alpha_A_intensity, 2)
mu, std = norm.fit(H_alpha_A_intensity)
print(mu, std)

H_alpha_A = []
for x in H_alpha_A_wavelength:
    y = 0
    # Calculate y_coordinate
    for n in range(len(fit_H_alpha_A)):
        y += (fit_H_alpha_A[n] * (x)**n)
    # Save coordinates
    H_alpha_A.append(y) 

fit_H_alpha_B =  np.polynomial.polynomial.polyfit(H_alpha_B_wavelength,H_alpha_B_intensity, 2)


H_alpha_B = []
for x in H_alpha_B_wavelength:
    y = 0
    # Calculate y_coordinate
    for n in range(len(fit_H_alpha_B)):
        y += (fit_H_alpha_B[n] * (x)**n)
    # Save coordinates
    H_alpha_B.append(y) 

'''
print(fit_H_alpha_A)
print(np.polyder(fit_H_alpha_A, 1))


z = [-1.53200913e-09, 1.40776779e-05, 5.28039891e-02, -3.46544929e+02, -3.98134659e+06, 1.86652649e+10]
x1=np.linspace(6562.1, 6563.6, 1000)

for x in x1:
    if np.polyval(np.polyder(z, 1), x) < 0.0001:
        print(x)
plt.plot(x1, np.polyval(z, x1))
plt.show()
plt.plot(x1, np.polyval(np.polyder(z, 1), x1))
plt.show()


min_value = np.polyval(np.polyder(z, 1), 6562.1)
for x in x1:
    if np.polyval(np.polyder(z, 1), x) < min_value:
        min_value = np.polyval(np.polyder(z, 1), x)
        print(x)
'''

'''
plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
# plt.plot(wavelength_object,(flux_object_A-dark_A)/(tungstenflat_A-darkflat_A))
plt.plot(wavelength_object, flux_object_norm_A, linewidth=1, label="Dataset A")
# plt.plot(wavelength_object, flux_object_norm_B, linewidth=1, label="Dataset B")
plt.plot(H_alpha_A_wavelength, H_alpha_A, label='fitfunctie A', linewidth=1)
plt.plot(H_alpha_A_wavelength, norm.pdf(H_alpha_A_wavelength, mu, std))
plt.errorbar(wavelength_object, flux_object_norm_A, yerr=flux_object_norm_A/SNR_A, markersize='1', fmt='.', ecolor='red', elinewidth=0.5)
# plt.plot(H_alpha_B_wavelength, H_alpha_B, label='fitfunctie B', linewidth=1)
# plt.plot(wavelength_object, flux_object_norm_B)
plt.ylim(0,)
plt.xlabel('Wavelenght (Angstrom)')
plt.ylabel("Genormaliseerde Intensiteit")
plt.legend()
plt.show()
'''

'''
min_H_alpha_A=H_alpha_A_wavelength[np.where(H_alpha_A == min(H_alpha_A))[0][0]]
min_H_alpha_B=H_alpha_B_wavelength[np.where(H_alpha_B == min(H_alpha_B))[0][0]]
print(np.where(H_alpha_A == min(H_alpha_A))[0][0], min(H_alpha_A))
print(f"De golflengte van H-alpha dataset A is {min_H_alpha_A}")
print(np.where(H_alpha_B == min(H_alpha_B))[0][0], min(H_alpha_B))
print(f"De golflengte van H-alpha dataset B is {min_H_alpha_B}")

R=696340000
c=299792458

lambda0 = (min_H_alpha_B + min_H_alpha_A)/2
delta_lambda = abs(min_H_alpha_B - lambda0)

v = c* (delta_lambda/lambda0)

print(lambda0, delta_lambda, v)


T = ((2*np.pi*R)/v)
print(f"{T} is de omlooptijd in seconden")
print(f"{T/(60*60*24)} is de omlooptijd in dagen")
'''


# %%






