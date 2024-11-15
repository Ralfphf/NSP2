# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:11:30 2023

@author: Rasjied
"""
import numpy as np
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
# formatting of exported datasets: Thar,Tungsten,Bias,Dark,Object, SNR, darkflat
    
    # For example if you would like the flux of the object in order 3:

main_folder = r'C:\Users\Ralfy\OneDrive - UvA\Natuur- & Sterrenkunde Bachelor\2e Jaar\NSP2 & ECPC\NSP2\Flux_raw_sunLimbA\Flux_raw_sunLimbA'

N_order = 3
data_order_N = np.loadtxt(os.path.join(main_folder, "data_raw_order_{}.csv").format(N_order),  delimiter=',')

x_pixelvalues = np.arange(len(data_order_N[0]))
thar = data_order_N[0]
tungstenflat = data_order_N[1]
bias = data_order_N[2]
dark = data_order_N[3]
flux_object = data_order_N[4]
SNR = data_order_N[5]
darkflat = data_order_N[6]


plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(x_pixelvalues,thar, label = 'ThAr')
plt.plot(x_pixelvalues,tungstenflat, label = 'Tungsten')
plt.plot(x_pixelvalues,bias, label = 'Bias')
plt.plot(x_pixelvalues,dark, label = 'Dark')
plt.plot(x_pixelvalues,flux_object, label = 'Object')
plt.plot(x_pixelvalues,SNR, label = 'SNR')
plt.plot(x_pixelvalues,darkflat, label = 'darkflat')
plt.legend()
plt.show()

# %% Golflengte Kalibratie met polynoomfit

wavelength_list =   [6677.2817,
                     6538.1120,
                     6583.9059, 
                     6604.8534,
                     6591.4845,
                     6588.5396,
                     6554.1603]

x_list =            [1752,
                     4656,
                     3747,
                     3319,
                     3594,
                     3654,
                     4343]

uncertainty_x =     [0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5]


plt.plot(x_pixelvalues,thar)
plt.scatter(x_list,thar[x_list], c='red', label = 'calibration points' )
for index in range(len(x_list)):
    plt.text(x_list[index]+20, thar[x_list][index]+20, wavelength_list[index], size=8)
plt.legend()
plt.show()

# %% Polynomial fit for wavelength calibration

fit_order = 2
fit_1 = np.polynomial.polynomial.polyfit(x_list,wavelength_list,fit_order,w=uncertainty_x)

# x & y coordinaten van de fit
wavelength_object = []
for x in x_pixelvalues:
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

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [7, 2]})
fig.subplots_adjust(hspace=0)

ax1.set_title("Wavelength calibration fit (x-pixels vs wavelength)")
ax1.plot(x_pixelvalues, wavelength_object)
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




# %% first order flux correction:

plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(wavelength_object,(flux_object-dark)/(tungstenflat-darkflat))
plt.ylim(0,)
plt.show()

# %% Nu aan jullie om lekker te normaliseren:










# %%






