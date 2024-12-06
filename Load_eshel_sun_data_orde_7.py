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
# formatting of exported datasets: Thar,Tungsten,Bias,Dark,Object, self.SNR_A, self.darkflat_A
    
    # For example if you would like the flux of the object in order 3:

class orde_7():
    def __init__(self):

        main_folder_A = r'C:\Users\Ralfy\OneDrive - UvA\Natuur- & Sterrenkunde Bachelor\2e Jaar\NSP2 & ECPC\NSP2\Flux_raw_sunLimbA\Flux_raw_sunLimbA'
        main_folder_B = r'C:\Users\Ralfy\OneDrive - UvA\Natuur- & Sterrenkunde Bachelor\2e Jaar\NSP2 & ECPC\NSP2\Flux_raw_sunLimbB\Flux_raw_sunLimbB'

        N_order = 7
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



        # %% Golflengte Kalibratie met polynoomfit

        self.wavelength_list =   [5912.0853,
                            5914.7139,
                            5916.5992,
                            5928.8130  ,
                            5888.584,
                            5882.6242,
                            5885.7016,
                            5860.3102,
                            5834.2633,
                            5938.8252,
                            5908.9257,
                            5891.451,
                            ]

        self.x_list =            [3271,
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
                            ]

        # %% Polynomial fit for wavelength calibration

        fit_order = 3
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


        # %% Nu aan jullie om lekker te normaliseren:

        fit_order_norm = 50
        fit_2_A = np.polynomial.polynomial.polyfit(self.wavelength_object,(self.flux_object_A-self.dark_A)/(self.tungstenflat_A-self.darkflat_A),fit_order_norm)

        # x & y coordinaten van de fit
        self.normalisation_fit_A= []
        for x in self.wavelength_object:
            y = 0
            # Calculate y_coordinate
            for n in range(len(fit_2_A)):
                y += (fit_2_A[n] * (x)**n) + 0.1
            # Save coordinates
            self.normalisation_fit_A.append(y)   

        fit_2_B = np.polynomial.polynomial.polyfit(self.wavelength_object,(self.flux_object_B-self.dark_B)/(self.tungstenflat_B-self.darkflat_B),fit_order_norm)

        # x & y coordinaten van de fit
        self.normalisation_fit_B= []
        for x in self.wavelength_object:
            y = 0
            # Calculate y_coordinate
            for n in range(len(fit_2_B)):
                y += (fit_2_B[n] * (x)**n) + 0.1
            # Save coordinates
            self.normalisation_fit_B.append(y)   

        self.flux_object_norm_A = (self.flux_object_A-self.dark_A)/((self.tungstenflat_A-self.darkflat_A)*self.normalisation_fit_A)
        self.flux_object_norm_B = (self.flux_object_B-self.dark_B)/((self.tungstenflat_B-self.darkflat_B)*self.normalisation_fit_B)


        #Natrium lijn D1 fitfunctie
        self.Na_D1_A_wavelength = []
        Na_D1_A_intensity = []
        self.Na_D1_B_wavelength = []
        Na_D1_B_intensity = []
        Na_D1_A_error = []
        Na_D1_B_error = []
        Na_D2_A_error = []
        Na_D2_B_error = []

        for i in range(len(self.wavelength_object)):
            if 5895.8 < self.wavelength_object[i] < 5896.55:
                self.Na_D1_A_wavelength.append(self.wavelength_object[i])
                Na_D1_A_intensity.append(self.flux_object_norm_A[i])
                self.Na_D1_B_wavelength.append(self.wavelength_object[i])
                Na_D1_B_intensity.append(self.flux_object_norm_B[i])
                Na_D1_A_error.append(self.flux_object_norm_A[i]/self.SNR_A[i])
                Na_D1_B_error.append(self.flux_object_norm_B[i]/self.SNR_B[i])


        #Natrium lijn D2 fitfunctie
        self.Na_D2_A_wavelength = []
        Na_D2_A_intensity = []
        self.Na_D2_B_wavelength = []
        Na_D2_B_intensity = []

        for i in range(len(self.wavelength_object)):
            if 5889.6 < self.wavelength_object[i] < 5890.6:
                self.Na_D2_A_wavelength.append(self.wavelength_object[i])
                Na_D2_A_intensity.append(self.flux_object_norm_A[i])
                self.Na_D2_B_wavelength.append(self.wavelength_object[i])
                Na_D2_B_intensity.append(self.flux_object_norm_B[i])
                Na_D2_A_error.append(self.flux_object_norm_A[i]/self.SNR_A[i])
                Na_D2_B_error.append(self.flux_object_norm_B[i]/self.SNR_B[i])

        def normal_distribution(x, std, avg, c):
            return -(np.e**(-(((x-avg)/std)**2)/2))/(std*np.sqrt(2*np.pi))+c

        popt_D1_A, self.pcov_D1_A = curve_fit(normal_distribution, self.Na_D1_A_wavelength, Na_D1_A_intensity, p0=[1, 5896.3, 1], sigma=Na_D1_A_error)
        self.std_opt_D1_A , self.avg_opt_D1_A, self.c_opt_D1_A= popt_D1_A

        popt_D1_B, self.pcov_D1_B = curve_fit(normal_distribution, self.Na_D1_B_wavelength, Na_D1_B_intensity, p0=[1, 5896.3, 1], sigma=Na_D1_B_error)
        self.std_opt_D1_B , self.avg_opt_D1_B, self.c_opt_D1_B= popt_D1_B

        popt_D2_A, self.pcov_D2_A = curve_fit(normal_distribution, self.Na_D2_A_wavelength, Na_D2_A_intensity, p0=[1, 5889.9, 1], sigma=Na_D2_A_error)
        self.std_opt_D2_A , self.avg_opt_D2_A, self.c_opt_D2_A= popt_D2_A

        popt_D2_B, self.pcov_D2_B = curve_fit(normal_distribution, self.Na_D2_B_wavelength, Na_D2_B_intensity, p0=[1, 5889.9, 1], sigma=Na_D2_B_error)
        self.std_opt_D2_B , self.avg_opt_D2_B, self.c_opt_D2_B= popt_D2_B



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
        plt.plot(self.wavelength_object, (self.flux_object_A - self.dark_A)/(self.tungstenflat_A-self.darkflat_A))
        plt.plot(self.wavelength_object, self.normalisation_fit_A)
        plt.ylim(0,)
        plt.show()

        plt.plot(self.wavelength_object,(self.flux_object_B-self.dark_B)/(self.tungstenflat_B-self.darkflat_B))
        plt.plot(self.wavelength_object, self.normalisation_fit_B)
        plt.ylim(0,)
        plt.show()

    def fitted_spectrallines(self):

        plt.errorbar(self.wavelength_object, self.flux_object_norm_A, yerr=self.flux_object_norm_A/self.SNR_A, markersize='1', fmt='.', ecolor='red', elinewidth=0.5)
        plt.errorbar(self.wavelength_object, self.flux_object_norm_B, yerr=self.flux_object_norm_B/self.SNR_B, markersize='1', fmt='.', ecolor='red', elinewidth=0.5)

        plt.plot(self.Na_D1_A_wavelength, (self.normal_distribution(self.Na_D1_A_wavelength, self.std_opt_D1_A, self.avg_opt_D1_A, self.c_opt_D1_A)), label='Gaussische fitfunctie D1 A')
        plt.plot(self.Na_D1_B_wavelength, (self.normal_distribution(self.Na_D1_B_wavelength, self.std_opt_D1_B, self.avg_opt_D1_B, self.c_opt_D1_B)), label='Gaussische fitfunctie D1 B')

        plt.plot(self.Na_D2_A_wavelength, (self.normal_distribution(self.Na_D2_A_wavelength, self.std_opt_D2_A, self.avg_opt_D2_A, self.c_opt_D2_A)), label='Gaussische fitfunctie D2 A')
        plt.plot(self.Na_D2_B_wavelength, (self.normal_distribution(self.Na_D2_B_wavelength, self.std_opt_D2_B, self.avg_opt_D2_B, self.c_opt_D2_B)), label='Gaussische fitfunctie D2 B')


        plt.ylim(0,)
        plt.xlabel('Wavelenght (Angstrom)')
        plt.ylabel("Genormaliseerde Intensiteit")
        plt.legend()
        plt.show()

    def returns_D1(self): 
        return self.avg_opt_D1_A, self.pcov_D1_A[1][1]**(1/2), self.avg_opt_D1_B, self.pcov_D1_B[1][1]**(1/2)
    def returns_D2(self):
        return self.avg_opt_D2_A, self.pcov_D2_A[1][1]**(1/2), self.avg_opt_D2_B, self.pcov_D2_B[1][1]**(1/2)

if __name__ == '__main__':
    model = orde_7()
    model.residuals_graph()
    model.flux_graph()
    model.fitted_spectrallines()