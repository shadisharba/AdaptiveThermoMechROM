"""
Temperature dependent material parameters of Copper(Cu) and Fused Tungsten Carbide (FTC)
The units here differ from the paper cited in readme.md by converting meter to millimeter
"""

import scipy.integrate as integrate
import numpy as np

I2 = np.asarray([1., 1., 1., 0, 0, 0])
I4 = np.eye(6)
IxI = np.outer(I2, I2)
P1 = IxI / 3.0
P2 = I4 - P1

min_temperature = 293.00
max_temperature = 1300

poisson_ratio_cu = lambda x: 3.40000e-01 * x**0
conductivity_cu = lambda x: 4.20749e+05 * x**0 + -6.84915e+01 * x**1
heat_capacity_cu = lambda x: 2.94929e+03 * x**0 + 2.30217e+00 * x**1 + -2.95302e-03 * x**2 + 1.47057e-06 * x**3
cte_cu = lambda x: 1.28170e-05 * x**0 + 8.23091e-09 * x**1
elastic_modulus_cu = lambda x: 1.35742e+08 * x**0 + 5.85757e+03 * x**1 + -8.16134e+01 * x**2

thermal_strain_cu = lambda x: integrate.quad(cte_cu, min_temperature, x)[0] * I2
shear_modulus_cu = lambda x: elastic_modulus_cu(x) / (2. * (1. + poisson_ratio_cu(x)))
bulk_modulus_cu = lambda x: elastic_modulus_cu(x) / (3. * (1. - 2. * poisson_ratio_cu(x)))
stiffness_cu = lambda x: bulk_modulus_cu(x) * IxI + 2. * shear_modulus_cu(x) * P2

poisson_ratio_wsc = lambda x: 2.80000e-01 * x**0
conductivity_wsc = lambda x: 2.19308e+05 * x**0 + -1.87425e+02 * x**1 + 1.05157e-01 * x**2 + -2.01180e-05 * x**3
heat_capacity_wsc = lambda x: 2.39247e+03 * x**0 + 6.62775e-01 * x**1 + -2.80323e-04 * x**2 + 6.39511e-08 * x**3
cte_wsc = lambda x: 5.07893e-06 * x**0 + 5.67524e-10 * x**1
elastic_modulus_wsc = lambda x: 4.13295e+08 * x**0 + -7.83159e+03 * x**1 + -3.65909e+01 * x**2 + 5.48782e-03 * x**3

thermal_strain_wsc = lambda x: integrate.quad(cte_wsc, min_temperature, x)[0] * I2
shear_modulus_wsc = lambda x: elastic_modulus_wsc(x) / (2. * (1. + poisson_ratio_wsc(x)))
bulk_modulus_wsc = lambda x: elastic_modulus_wsc(x) / (3. * (1. - 2. * poisson_ratio_wsc(x)))
stiffness_wsc = lambda x: bulk_modulus_wsc(x) * IxI + 2. * shear_modulus_wsc(x) * P2
