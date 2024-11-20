data_order_N_B = np.loadtxt(os.path.join(main_folder_B, "data_raw_order_{}.csv").format(N_order),  delimiter=',')

x_pixelvalues_B = np.arange(len(data_order_N_B[0]))
thar_B = data_order_N_B[0]
tungstenflat_B = data_order_N_B[1]
bias_B = data_order_N_B[2]
dark_B = data_order_N_B[3]
flux_object_B = data_order_N_B[4]
SNR_B = data_order_N_B[5]
darkflat_B = data_order_N_B[6]


plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(x_pixelvalues_B,thar_B, label = 'ThAr')
plt.plot(x_pixelvalues_B,tungstenflat_B, label = 'Tungsten')
plt.plot(x_pixelvalues_B,bias_B, label = 'Bias')
plt.plot(x_pixelvalues_B,dark_B, label = 'Dark_A')
plt.plot(x_pixelvalues_B,flux_object_B, label = 'Object')
plt.plot(x_pixelvalues_B,SNR_B, label = 'SNR')
plt.plot(x_pixelvalues_B,darkflat_B, label = 'darkflat')
plt.legend()
plt.show()

plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
plt.plot(wavelength_object,(flux_object_B-dark_B)/(tungstenflat_B-darkflat_B))
plt.ylim(0,)
plt.show()

# %% Nu aan jullie om lekker te normaliseren:

fit_order_norm = 10
fit_2 = np.polynomial.polynomial.polyfit(wavelength_object,(flux_object_B-dark_B)/(tungstenflat_B-darkflat_B),fit_order_norm)

# x & y coordinaten van de fit
normalisation_fit= []
for x in wavelength_object:
    y = 0
    # Calculate y_coordinate
    for n in range(len(fit_2)):
        y += (fit_2[n] * (x)**n) + 0.1
    # Save coordinates
    normalisation_fit.append(y)   

flux_object_norm = (flux_object_B-dark_B)/((tungstenflat_B-darkflat_B)*normalisation_fit)
plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
# plt.plot(wavelength_object,(flux_object_B-dark_B)/(tungstenflat_B-darkflat_B))
plt.plot(wavelength_object, flux_object_norm, linewidth=1)
# plt.plot(wavelength_object, flux_object_norm)
plt.ylim(0,)
plt.show()

print(np.where(flux_object_norm == min(flux_object_norm))[0][0], min(flux_object_norm))
print(f"De golflengte van H-alpha is {wavelength_object[np.where(flux_object_norm == min(flux_object_norm))[0][0]]}")

