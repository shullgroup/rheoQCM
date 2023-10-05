# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt

# add QCMFuncs
sys.path.append('../QCMFuncs')

import QCM_functions as qcm
plt.close('all')

# read the data
df = qcm.read_xlsx('../test_data/BCB_4.xlsx')

# pick a calculation
# we fit to the frequency shifts for the values before the colon
# we fit to the dissipation shifts for the values after the colon
calc = '3.5_5'

# solve for the properties
layers = {1:{'grho3':1e12, 'phi':1, 'drho':5e-3}}
soln = qcm.solve_for_props(df, calc, ['grho3', 'phi', 'drho'], layers)

# now make the property axes and plot the property values on it
props = qcm.make_prop_axes(['grho3.linear', 'phi', 'drho'], xunit = 'index')
qcm.plot_props(soln, props, fmt='+-', num = 'BCB properties')

