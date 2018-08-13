#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 14:51:38 2018

@author: ken
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../QCMfuncs') 
from QCM_functions import make_prop_axes, make_vgp_axes
propfig = make_prop_axes('props', 't (min.)')
vgpfig = make_vgp_axes('vgp')

def prop_plot_from_csv(figure, csvfile, plotstr, legendtext):
    data = pd.read_csv(csvfile)    
    figure['drho_ax'].plot(data['xdata'].values, data['drho'].values, '+b', 
                label=legendtext)
   
    figure['grho_ax'].plot(data['xdata'].values, data['grho'].values, '+b', 
                label=legendtext)
    figure['phi_ax'].plot(data['xdata'].values, data['phi'].values, '+b', 
                label=legendtext)
   
data1file = '../figures/DGEBA-Jeffamine230_RT_3_355.txt'
legend1text = '1:1expoxy:amine'

data2file = '../figures/DGEBA-Jeffamine230_RT_5_355.txt'
legend2text = '2:1epoxy:amine'

prop_plot(data1file, 'b+', legend1text)
prop_plot(data2file, 'ro', legend2text)


vgpfig=plt.figure('vgp', figsize=(3,3))
vgp_ax = vgpfig.add_subplot(111)
vgp_ax.set_xlabel(r'$|G_3^*|\rho \: (Pa \cdot g/cm^3)$')
vgp_ax.set_ylabel(r'$\phi$ (deg.)')
                  
vgp_ax.semilogx(data1['grho'].values, data1['phi'].values, '+b', 
            label='1:1 exoxy:amine')
vgp_ax.semilogx(data2['grho'].values, data2['phi'].values, 'or', 
            label='2:1 exoxy:amine')

vgpfig.tight_layout()

