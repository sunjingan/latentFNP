import numpy as np
import pandas as pd

RMSE = [4.6716e-01, 4.2914e-01, 9.5297e-01, 3.4774e+01, 7.5965e+01, 5.5894e+01,
        4.2624e+01, 3.7964e+01, 3.5818e+01, 3.5080e+01, 3.1563e+01, 2.8136e+01,
        2.5520e+01, 2.4193e+01, 2.1998e+01, 2.3450e+01, 2.8007e+01, 6.6796e-09,
        4.7773e-08, 3.8743e-07, 2.6422e-06, 8.8568e-06, 1.9828e-05, 5.9750e-05,
        1.2555e-04, 2.0067e-04, 2.9275e-04, 4.5596e-04, 4.9556e-04, 4.3874e-04,
        6.0721e-01, 7.3854e-01, 9.6111e-01, 1.1055e+00, 1.1774e+00, 1.1365e+00,
        1.0036e+00, 9.0299e-01, 8.3325e-01, 7.8649e-01, 7.5484e-01, 6.9423e-01,
        5.1714e-01, 5.1639e-01, 5.7010e-01, 7.3007e-01, 8.5287e-01, 9.5192e-01,
        9.5369e-01, 8.4867e-01, 7.4229e-01, 6.7749e-01, 6.3604e-01, 6.3625e-01,
        6.1518e-01, 4.7171e-01, 4.3586e-01, 5.0815e-01, 3.8208e-01, 2.9663e-01,
        3.1992e-01, 3.6655e-01, 4.3531e-01, 4.5372e-01, 4.7696e-01, 5.1185e-01,
        6.1456e-01, 6.0399e-01, 6.8289e-01]

multi_level_vnames = [
    "z", "q","u","v","t"
]
single_level_vnames = ['u10', 'v10', 't2m', 'msl']

long2shortname_dict = {"geopotential": "z", "temperature": "t", "specific_humidity": "q", "relative_humidity": "r", "u_component_of_wind": "u", "v_component_of_wind": "v", "vorticity": "vo", "potential_vorticity": "pv", \
    "2m_temperature": "t2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10", "total_cloud_cover": "tcc", "total_precipitation": "tp", "toa_incident_solar_radiation": "tisr"}

height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

for i in range(len(single_level_vnames)):
    print(single_level_vnames[i],RMSE[i])

for i in range(len(multi_level_vnames)):
    for j in range(len(height_level)):
        idx_ = i*len(height_level) + j
        print(multi_level_vnames[i]+str(height_level[j]),RMSE[idx_+4])