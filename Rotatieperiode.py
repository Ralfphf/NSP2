from Load_eshel_sun_data_orde_3 import orde_3
from Load_eshel_sun_data_orde_7 import orde_7
from Load_eshel_sun_data_orde_13 import orde_13
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
model_H_alpha = orde_3()
model_Na = orde_7()
model_Mg = orde_13()

avg_opt_A, pcov_n_A, avg_opt_B, pcov_n_B = model_H_alpha.returns_H_alpha()
avg_opt_D1_A, pcov_D1_A, avg_opt_D1_B, pcov_D1_B = model_Na.returns_D1()
avg_opt_D2_A, pcov_D2_A, avg_opt_D2_B, pcov_D2_B = model_Na.returns_D2()
avg_opt_b1_A, pcov_b1_A, avg_opt_b1_B, pcov_b1_B = model_Mg.returns_b1()
avg_opt_b2_A, pcov_b2_A, avg_opt_b2_B, pcov_b2_B = model_Mg.returns_b2()
avg_opt_b3_A, pcov_b3_A, avg_opt_b3_B, pcov_b3_B = model_Mg.returns_b3()


R=696342000
error_R = 65000
c=299792458

def omlooptijd(min_A_g, error_A_g, min_B_g, error_B_g):
    lambda_gem = (min_A_g+min_B_g)/2
    delta_lambda = abs(lambda_gem - min_A_g)
    v = c * (delta_lambda/lambda_gem)
    T = ((2*np.pi*R)/v)/(60*60*24)
    print(f"{T} is de omlooptijd in dagen")

    error_T = ((((2*np.pi*error_R/c)*((min_A_g+min_B_g)/(min_B_g-min_A_g)))**2 +
        ((2*np.pi*R/c)*((2*min_B_g*error_A_g)/((min_A_g-min_B_g)**2)))**2 +
        ((2*np.pi*R/c)*((2*min_A_g*error_B_g)/((min_B_g-min_A_g)**2)))**2)**(1/2))/(60*60*24)
    print(f"{error_T} is de error van de omlooptijd in dagen")
    return T, error_T

T_H_alpha, T_error_H_alpha = omlooptijd(avg_opt_A, pcov_n_A, avg_opt_B, pcov_n_B)
T_Na_D1, T_error_Na_D1 = omlooptijd(avg_opt_D1_A, pcov_D1_A, avg_opt_D1_B, pcov_D1_B)
T_Na_D2, T_error_Na_D2 = omlooptijd(avg_opt_D2_A, pcov_D2_A, avg_opt_D2_B, pcov_D2_B)
T_Mg_b1, T_error_Mg_b1 = omlooptijd(avg_opt_b1_A, pcov_b1_A, avg_opt_b1_B, pcov_b1_B)
T_Mg_b2, T_error_Mg_b2 = omlooptijd(avg_opt_b2_A, pcov_b2_A, avg_opt_b2_B, pcov_b2_B)
T_Mg_b3, T_error_Mg_b3 = omlooptijd(avg_opt_b3_A, pcov_b3_A, avg_opt_b3_B, pcov_b3_B)

all_periods = [T_H_alpha, T_Na_D1, T_Na_D2, T_Mg_b1, T_Mg_b2, T_Mg_b3]
all_periods_name = ['H-alpha', 'Natrium D1', 'Natrium D2', 'Magnesium b1', 'Magnesium b2', 'Magnesium b3']
all_errors_periods = [T_error_H_alpha, T_error_Na_D1, T_error_Na_D2, T_error_Mg_b1, T_error_Mg_b2, T_error_Mg_b3]
number_of_lines = [1, 2, 3, 4, 5, 6]

    
line_list = []
error_line = []
def straight_line(x, p):
    return p 


popt_s, pcov_s = curve_fit(straight_line, number_of_lines, all_periods, p0=[25], sigma=all_errors_periods)
print(popt_s[0], pcov_s)
for i in range(len(number_of_lines)):
    line_list.append(popt_s[0])
    error_line.append(pcov_s[0][0])



fit_order_norm = 2
fit = np.polynomial.polynomial.polyfit(number_of_lines, all_periods, fit_order_norm, w=all_errors_periods)

# x & y coordinaten van de fit
omlooptijd_fit= []
for x in number_of_lines:
    y = 0
    # Calculate y_coordinate
    for n in range(len(fit)):
        y += (fit[n] * (x)**n) + 0.1
    # Save coordinates
    omlooptijd_fit.append(y)   
# lekker plotten:

plt.ylabel("Omlooptijd (dagen)")
plt.errorbar(all_periods_name, all_periods, yerr=all_errors_periods, fmt='o', ecolor='red', capsize=3, label='Residuals with error bars')
plt.scatter(all_periods_name, all_periods, c='blue')
plt.plot(all_periods_name, omlooptijd_fit)
plt.errorbar(all_periods_name, line_list, yerr=error_line, fmt='-')
plt.ylabel('Omlooptijd equator (dagen)')
plt.legend()
plt.show()

print(f"De gemiddelde omlooptijd is {popt_s[0]} dagen met een error van {pcov_s[0][0]}")