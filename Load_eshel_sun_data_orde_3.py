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
# formatting of exported datasets: Thar,Tungsten,Bias,Dark,Object, self.SNR_A, self.darkflat_A
    
    # For example if you would like the flux of the object in order 3:

class orde_3():
    def __init__(self):
        main_folder_A = r'C:\Users\Ralfy\OneDrive - UvA\Natuur- & Sterrenkunde Bachelor\2e Jaar\NSP2 & ECPC\NSP2\Flux_raw_sunLimbA\Flux_raw_sunLimbA'
        main_folder_B = r'C:\Users\Ralfy\OneDrive - UvA\Natuur- & Sterrenkunde Bachelor\2e Jaar\NSP2 & ECPC\NSP2\Flux_raw_sunLimbB\Flux_raw_sunLimbB'

        N_order = 3
        data_order_N_A = np.loadtxt(os.path.join(main_folder_A, "data_raw_order_{}.csv").format(N_order),  delimiter=',')
        data_order_N_B = np.loadtxt(os.path.join(main_folder_B, "data_raw_order_{}.csv").format(N_order),  delimiter=',')



        self.x_pixelvalues_A = np.arange(len(data_order_N_A[0]))
        self.thar_A = data_order_N_A[0]
        self.tungstenflat_A = data_order_N_A[1]
        self.bias_A = data_order_N_A[2]
        self.dark_A = data_order_N_A[3]
        self.flux_object_A = data_order_N_A[4]
        self.SNR_A = data_order_N_A[5]
        self.darkflat_A = data_order_N_A[6]


        self.x_pixelvalues_B = np.arange(len(data_order_N_B[0]))
        self.thar_B = data_order_N_B[0]
        self.tungstenflat_B = data_order_N_B[1]
        self.bias_B = data_order_N_B[2]
        self.dark_B = data_order_N_B[3]
        self.flux_object_B = data_order_N_B[4]
        self.SNR_B = data_order_N_B[5]
        self.darkflat_B = data_order_N_B[6]

        self.wavelength_list =   [6677.2817,
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

        self.x_list =            [1752,
                            4656,
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

        self.uncertainty_x =     [0.5,
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
        # %% Polynomial fit for wavelength calibration

        fit_order = 4
        #5 of hoger valt buiten 
        self.fit_1 = np.polynomial.polynomial.polyfit(self.x_list,self.wavelength_list,fit_order,w=self.uncertainty_x)

        # x & y coordinaten van de fit
        self.wavelength_object = []
        for x in self.x_pixelvalues_A:
            y = 0
            # Calculate y_coordinate
            for n in range(len(self.fit_1)):
                y += self.fit_1[n] * (x)**n       
            # Save coordinates
            self.wavelength_object.append(y)   

        fit_order_norm = 10
        fit_2_A = np.polynomial.polynomial.polyfit(self.wavelength_object,(self.flux_object_A-self.dark_A)/(self.tungstenflat_A-self.darkflat_A),fit_order_norm)

        # x & y coordinaten van de fit
        normalisation_fit_A= []
        for x in self.wavelength_object:
            y = 0
            # Calculate y_coordinate
            for n in range(len(fit_2_A)):
                y += (fit_2_A[n] * (x)**n) + 0.1
            # Save coordinates
            normalisation_fit_A.append(y)   

        fit_2_B = np.polynomial.polynomial.polyfit(self.wavelength_object,(self.flux_object_B-self.dark_B)/(self.tungstenflat_B-self.darkflat_B),fit_order_norm)

        # x & y coordinaten van de fit
        normalisation_fit_B= []
        for x in self.wavelength_object:
            y = 0
            # Calculate y_coordinate
            for n in range(len(fit_2_B)):
                y += (fit_2_B[n] * (x)**n) + 0.1
            # Save coordinates
            normalisation_fit_B.append(y)   

        self.flux_object_norm_A = (self.flux_object_A-self.dark_A)/((self.tungstenflat_A-self.darkflat_A)*normalisation_fit_A)
        self.flux_object_norm_B = (self.flux_object_B-self.dark_B)/((self.tungstenflat_B-self.darkflat_B)*normalisation_fit_B)
    

        self.H_alpha_A_wavelength = []
        H_alpha_A_intensity = []
        self.H_alpha_B_wavelength = []
        H_alpha_B_intensity = []
        H_alpha_A_error = []
        H_alpha_B_error = []

        for i in range(len(self.wavelength_object)):
            if 6561.83 < self.wavelength_object[i] < 6563.63:
                self.H_alpha_A_wavelength.append(self.wavelength_object[i])
                H_alpha_A_intensity.append(self.flux_object_norm_A[i])
                H_alpha_A_error.append(self.flux_object_norm_A[i]/self.SNR_A[i])

        for i in range(len(self.wavelength_object)):
            if 6561.89 < self.wavelength_object[i] < 6563.68:
                self.H_alpha_B_wavelength.append(self.wavelength_object[i])
                H_alpha_B_intensity.append(self.flux_object_norm_B[i])
                H_alpha_B_error.append(self.flux_object_norm_B[i]/self.SNR_B[i])

        def normal_distribution(x, std, avg, c):
            return -(np.e**(-(((x-avg)/std)**2)/2))/(std*np.sqrt(2*np.pi))+c

        popt_n_A, self.pcov_n_A = curve_fit(normal_distribution, self.H_alpha_A_wavelength, H_alpha_A_intensity, p0=[1, 6562.7, 1], sigma=H_alpha_A_error)
        self.std_opt_A , self.avg_opt_A, self.c_opt_A= popt_n_A

        popt_n_B, self.pcov_n_B = curve_fit(normal_distribution, self.H_alpha_B_wavelength, H_alpha_B_intensity, p0=[1, 6562.7, 1], sigma=H_alpha_B_error)
        self.std_opt_B , self.avg_opt_B, self.c_opt_B= popt_n_B

        self.normal_distribution = normal_distribution

    def all_datasets_graph(self):
        plt.plot(self.x_pixelvalues_A,self.thar_A, label = 'ThAr')
        plt.plot(self.x_pixelvalues_A,self.tungstenflat_A, label = 'Tungsten')
        plt.plot(self.x_pixelvalues_A,self.bias_A, label = 'Bias')
        plt.plot(self.x_pixelvalues_A,self.dark_A, label = 'Dark_A')
        plt.plot(self.x_pixelvalues_A,self.flux_object_A, label = 'Object')
        plt.plot(self.x_pixelvalues_A,self.SNR_A, label = 'SNR')
        plt.plot(self.x_pixelvalues_A,self.darkflat_A, label = 'darkflat')
        plt.legend()
        plt.show()

        plt.plot(self.x_pixelvalues_B,self.thar_B, label = 'ThAr')
        plt.plot(self.x_pixelvalues_B,self.tungstenflat_B, label = 'Tungsten')
        plt.plot(self.x_pixelvalues_B,self.bias_B, label = 'Bias')
        plt.plot(self.x_pixelvalues_B,self.dark_B, label = 'Dark')
        plt.plot(self.x_pixelvalues_B,self.flux_object_B, label = 'Object')
        plt.plot(self.x_pixelvalues_B,self.SNR_B, label = 'SNR')
        plt.plot(self.x_pixelvalues_B,self.darkflat_B, label = 'darkflat')
        plt.legend()
        plt.show()
        

    def wavelength_calibration(self):
        # Golflengte Kalibratie met polynoomfit
        
        plt.plot(self.x_pixelvalues_A,self.thar_A)
        plt.scatter(self.x_list,self.thar_A[self.x_list], c='red', label = 'calibration points' )
        for index in range(len(self.x_list)):
            plt.text(self.x_list[index]+20, self.thar_A[self.x_list][index]+20, self.wavelength_list[index], size=8)
        plt.legend()
        plt.show()
        


        #  Residuals berekenen
    def residuals_graph(self):
        residuals = []
        for i, x_value in enumerate(self.x_list):
            # Bereken de voorspelde waarde met de fit-coëfficiënten
            predicted_wavelength = sum(self.fit_1[n] * (x_value)**n for n in range(len(self.fit_1)))
            
            # Bereken het residual door het verschil te nemen tussen de werkelijke en voorspelde waarde
            residual = self.wavelength_list[i] - predicted_wavelength
            residuals.append(residual)
            
        # lekker plotten:

        fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [7, 2]})
        fig.subplots_adjust(hspace=0)

        ax1.set_title("Wavelength calibration fit (x-pixels vs wavelength)")
        ax1.plot(self.x_pixelvalues_A, self.wavelength_object)
        ax1.set_ylabel("Wavelength [Angstrom]")
        ax1.errorbar(self.x_list, self.wavelength_list, yerr=np.abs(self.uncertainty_x*np.array(self.fit_1[1])), fmt='o', ecolor='red', capsize=3, label='Residuals with error bars')
        ax1.scatter(self.x_list,self.wavelength_list, c='blue')



        ax2.errorbar(self.x_list, residuals, yerr=np.abs(self.uncertainty_x*np.array(self.fit_1[1])), fmt='o', ecolor='red', capsize=3, label='Residuals with error bars')
        ax2.scatter(self.x_list,residuals)
        ax2.set_ylabel("Pixels")
        ax2.set_ylabel("Residuals [Angstrom]")
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, label = 'model')
        ax2.axhline(self.fit_1[1], color='gray', linestyle='--', linewidth=1, label = '1 pixel difference')
        ax2.axhline(-1*self.fit_1[1], color='gray', linestyle='--', linewidth=1)
        for index in range(len(self.x_list)):
            ax2.text(self.x_list[index], residuals[index], self.wavelength_list[index], size=8)
        plt.legend()
        plt.show()
    

        # %% first order flux correction:
    def flux_graph(self):

        plt.errorbar(self.wavelength_object,(self.flux_object_A-self.dark_A)/(self.tungstenflat_A-self.darkflat_A))
        plt.ylim(0,)
        plt.show()

        plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
        plt.plot(self.wavelength_object,(self.flux_object_B-self.dark_B)/(self.tungstenflat_B-self.darkflat_B))
        plt.ylim(0,)
        plt.show()

        # %% Nu aan jullie om lekker te normaliseren:



    def fitted_spectrallines(self):
        plt.plot(self.wavelength_object, self.flux_object_norm_A, linewidth=1, label="Dataset A")
        plt.plot(self.wavelength_object, self.flux_object_norm_B, linewidth=1, label="Dataset B")
        plt.plot(self.H_alpha_A_wavelength, (self.normal_distribution(self.H_alpha_A_wavelength, self.std_opt_A, self.avg_opt_A, self.c_opt_A)), label='Gaussische fitfunctie A')
        plt.plot(self.H_alpha_B_wavelength, (self.normal_distribution(self.H_alpha_B_wavelength, self.std_opt_B, self.avg_opt_B, self.c_opt_B)), label='Gaussische fitfunctie B')

        plt.errorbar(self.wavelength_object, self.flux_object_norm_A, yerr=self.flux_object_norm_A/self.SNR_A, markersize='1', fmt='.', ecolor='red', elinewidth=0.5)
        plt.errorbar(self.wavelength_object, self.flux_object_norm_B, yerr=self.flux_object_norm_B/self.SNR_B, markersize='1', fmt='.', ecolor='red', elinewidth=0.5)
        plt.ylim(0,)
        plt.xlabel('Wavelenght (Angstrom)')
        plt.ylabel("Genormaliseerde Intensiteit")
        plt.legend()
        plt.show()

    def returns_H_alpha(self):
        return self.avg_opt_A, self.pcov_n_A[1][1]**(1/2), self.avg_opt_B, self.pcov_n_B[1][1]**(1/2)


        
if __name__ == '__main__':
    model = orde_3()
    model.fitted_spectrallines()