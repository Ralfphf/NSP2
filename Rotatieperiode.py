from Load_eshel_sun_data_orde_3 import omlooptijd_3, parameters_3
from Load_eshel_sun_data_orde_7 import omlooptijd_7, parameters_7

min_A_g, error_A_g, min_B_g, error_B_g = parameters_3
T_3, error_T_3 = omlooptijd_3(min_A_g, error_A_g, min_B_g, error_B_g)
print(T_3, error_T_3)


