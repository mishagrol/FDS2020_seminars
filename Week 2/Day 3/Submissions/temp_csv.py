import numpy as np
from pandas import DataFrame as pdf


my_array = np.array([[7.099999999999999645e-01,2.454099999999999966e+02,2.232999999999999829e+01],
[7.099999999999999645e-01,2.431800000000000068e+02,2.667999999999999972e+01],
[0.000000000000000000e+00,5.257500000000000000e+02,8.849999999999999645e+00],
[9.699999999999999734e-01,2.348000000000000043e+01,0.000000000000000000e+00]])

#rounded_array = np.round((np.reshape(array_of_results, (4,3))), 2)
#np.savetxt(result_csv, rounded_array, delimiter=",")
print(my_array)