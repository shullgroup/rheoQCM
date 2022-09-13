# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt

# add QCMFuncs
sys.path.append('../QcmFuncs')

import QCM_functions as qcm
plt.close('all')

# read the data
df = qcm.read_xlsx('../test_data/BCB_4.xlsx')

# pick a calculation
# we fit to the frequency shifts for the values before the colon
# we fit to the dissipation shifts for the values after the colon
calc = '3.5:5'

# solve for the properties
soln = qcm.solve_for_props(df, calc)

# now make the property axes and plot the property values on it
props = qcm.make_prop_axes(xunit = 'index')
qcm.prop_plots(soln, props, fmt='+-', num = 'BCB properties')

# now generate the solution check
check = qcm.check_solution(soln, nplot = [1, 3, 5], 
                           plotsize = (5,3), df_lim = 'auto',
                           gammascale = 'log',
                           num = 'BCB solution check',
                           plot_df1 = True)
