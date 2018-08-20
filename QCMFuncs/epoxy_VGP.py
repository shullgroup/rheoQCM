#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 14:51:38 2018

@author: ken
"""

import sys
sys.path.append('..') 
import QCM_functions as qcm

# make the property and van Gurp-Palmen axes
propfig = qcm.make_prop_axes('props', 't (min.)')
vgpfig = qcm.make_vgp_axes('vgp')

# define text files for plotting
data1file = 'figures/DGEBA-Jeffamine230_RT_3_355.txt'
legend1text = '1:1expoxy:amine'
data2file = 'figures/DGEBA-Jeffamine230_RT_5_355.txt'
legend2text = '2:1epoxy:amine'

# add the data to the texts
qcm.prop_plot_from_csv(propfig, data1file, 'b+', legend1text)
qcm.prop_plot_from_csv(propfig, data2file, 'ro', legend2text)
qcm.vgp_plot_from_csv(vgpfig, data1file, 'b+', legend1text)
qcm.vgp_plot_from_csv(vgpfig, data2file, 'ro', legend2text)



