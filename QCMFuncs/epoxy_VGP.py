#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 14:51:38 2018

@author: ken
"""
#%%
# import sys
# sys.path.append('..') 
import matplotlib.pyplot as plt

try:
    from Qifeng_sampledefs import sample_dict
except ImportError: # in case running Jupyter in the QCM_py folder
    from QCMFuncs.Qifeng_sampledefs import sample_dict
try:
    import QCM_functions as qcm
except ImportError: # in case running Jupyter in the QCM_py folder
    import QCMFuncs.QCM_functions as qcm

parms = {}  # parameters to pass to qcm.analyze
parms['dataroot'] = qcm.find_dataroot('qifeng')
parms['figlocation'] = 'datadir' # save data in 
sample = sample_dict()  # read sample dictionary

######################################################################


#%% DGEBA-Jeffamine230
# make the property and van Gurp-Palmen axes
propfig = qcm.make_prop_axes('props', 't (min.)')
vgpfig = qcm.make_vgp_axes('vgp')

# define text files for plotting
base_fig_name1 = qcm.find_base_fig_name(sample['DGEBA-Jeffamine230_RT_3'], parms)
legend1text = '1:1expoxy:amine'
base_fig_name2 = qcm.find_base_fig_name(sample['DGEBA-Jeffamine230_RT_5'], parms)
legend2text = '2:1epoxy:amine'

# add the data to the texts
qcm.prop_plot_from_csv(propfig, base_fig_name1 + '_355.txt', 'b+', legend1text)
qcm.prop_plot_from_csv(propfig, base_fig_name1 + '_133.txt', 'bx', legend1text)
qcm.prop_plot_from_csv(propfig, base_fig_name2 + '_355.txt', 'ro', legend2text)
qcm.prop_plot_from_csv(propfig, base_fig_name2 + '_133.txt', 'rv', legend2text)
propfig['drho_ax'].legend()
qcm.vgp_plot_from_csv(vgpfig, base_fig_name1 + '_355.txt', 'b+', legend1text)
qcm.vgp_plot_from_csv(vgpfig, base_fig_name1 + '_133.txt', 'bx', legend1text)
qcm.vgp_plot_from_csv(vgpfig, base_fig_name2 + '_355.txt', 'ro', legend2text)
qcm.vgp_plot_from_csv(vgpfig, base_fig_name2 + '_133.txt', 'rv', legend2text)

plt.legend()
plt.show()



#%% DGEBA-Jeffamine400
# make the property and van Gurp-Palmen axes
propfig = qcm.make_prop_axes('props', 't (min.)')
vgpfig = qcm.make_vgp_axes('vgp')
# define text files for plotting
base_fig_name3 = qcm.find_base_fig_name(sample['DGEBA-Jeffamine400_RT_3'], parms)


# add the data to the texts
qcm.prop_plot_from_csv(propfig, base_fig_name3 + '_355.txt', 'b+', '355')
qcm.prop_plot_from_csv(propfig, base_fig_name3 + '_353.txt', 'bx', '353')
propfig['drho_ax'].legend()
qcm.vgp_plot_from_csv(vgpfig, base_fig_name3  + '_355.txt', 'b+', '355')
qcm.vgp_plot_from_csv(vgpfig, base_fig_name3  + '_353.txt', 'bx', '353')

plt.legend()
plt.show()



#%% DGEBA-PACM
# make the property and van Gurp-Palmen axes
propfig = qcm.make_prop_axes('props', 't (min.)')
vgpfig = qcm.make_vgp_axes('vgp')

# define text files for plotting
base_fig_name4 = qcm.find_base_fig_name(sample['DGEBA-PACM_RT'], parms)
legend4text = 'DGEBA-PACM'
base_fig_name5 = qcm.find_base_fig_name(sample['DGEBA-PACM_RT_2'], parms)
legend5text = 'DGEBA-PACM_2'

# add the data to the texts
qcm.prop_plot_from_csv(propfig, base_fig_name4 + '_355.txt', 'b+', legend4text +  ' 355')
qcm.prop_plot_from_csv(propfig, base_fig_name4 + '_353.txt', 'bx', legend4text +  ' 133')
qcm.prop_plot_from_csv(propfig, base_fig_name5 + '_355.txt', 'ro', legend5text +  ' 355')
qcm.prop_plot_from_csv(propfig, base_fig_name5 + '_133.txt', 'rv', legend5text +  ' 133')
propfig['drho_ax'].legend()
qcm.vgp_plot_from_csv(vgpfig, base_fig_name4 + '_355.txt', 'b+', legend4text +  ' 355')
qcm.vgp_plot_from_csv(vgpfig, base_fig_name4 + '_353.txt', 'bx', legend4text +  ' 133')
qcm.vgp_plot_from_csv(vgpfig, base_fig_name5 + '_355.txt', 'ro', legend5text +  ' 355')
qcm.vgp_plot_from_csv(vgpfig, base_fig_name5 + '_133.txt', 'rv', legend5text +  ' 133')

plt.legend()
plt.show()


#%% DGEBA-Jeffamine2000
# make the property and van Gurp-Palmen axes
propfig = qcm.make_prop_axes('props', 't (min.)')
vgpfig = qcm.make_vgp_axes('vgp')

# add the data to the texts
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT'], parms) + '_133.txt', 'bo',
    'DGEBA-Jeffamine2000 133 (1:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT'], parms) + '_355.txt', 'bx',
    'DGEBA-Jeffamine2000 355 (1:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_2'], parms) + '_133.txt', 'go',
    'DGEBA-Jeffamine2000_2 133 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_2'], parms) + '_355.txt', 'gx',
    'DGEBA-Jeffamine2000_2 355 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_3'], parms) + '_133.txt', 'ro',
    'DGEBA-Jeffamine2000_3 133 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_3'], parms) + '_355.txt', 'rx',
    'DGEBA-Jeffamine2000_3 355 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_3_2'], parms) + '_133.txt', 'co',
    'DGEBA-Jeffamine2000_3_2 133 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_3_2'], parms) + '_355.txt', 'cx',
    'DGEBA-Jeffamine2000_3_2 355 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_4'], parms) + '_133.txt', 'mo',
    'DGEBA-Jeffamine2000_4 133 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_4'], parms) + '_355.txt', 'mx',
    'DGEBA-Jeffamine2000_4 355 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_4_2'], parms) + '_133.txt', 'yo',
    'DGEBA-Jeffamine2000_4_2 133 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_4_2'], parms) + '_355.txt', 'yx',
    'DGEBA-Jeffamine2000_4_2 355 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_5'], parms) + '_133.txt', 'ko',
    'DGEBA-Jeffamine2000_5 133 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_5'], parms) + '_355.txt', 'kx',
    'DGEBA-Jeffamine2000_5 355 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_6'], parms) + '_133.txt', 'ro',
    'DGEBA-Jeffamine2000_6 133 (2:1)')
qcm.prop_plot_from_csv(propfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_6'], parms) + '_355.txt', 'rx',
    'DGEBA-Jeffamine2000_6 355 (2:1)')

propfig['drho_ax'].legend()
propfig['drho_ax'].set_xscale('log')
propfig['grho_ax'].set_xscale('log')
propfig['phi_ax'].set_xscale('log')

qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT'], parms) + '_133.txt', 'bo',
    'DGEBA-Jeffamine2000 133 (1:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT'], parms) + '_355.txt', 'bx',
    'DGEBA-Jeffamine2000 355 (1:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_2'], parms) + '_133.txt', 'go',
    'DGEBA-Jeffamine2000_2 133 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_2'], parms) + '_355.txt', 'gx',
    'DGEBA-Jeffamine2000_2 355 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_3'], parms) + '_133.txt', 'ro',
    'DGEBA-Jeffamine2000_3 133 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_3'], parms) + '_355.txt', 'rx',
    'DGEBA-Jeffamine2000_3 355 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_3_2'], parms) + '_133.txt', 'co',
    'DGEBA-Jeffamine2000_3_2 133 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_3_2'], parms) + '_355.txt', 'cx',
    'DGEBA-Jeffamine2000_3_2 355 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_4'], parms) + '_133.txt', 'mo',
    'DGEBA-Jeffamine2000_4 133 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_4'], parms) + '_355.txt', 'mx',
    'DGEBA-Jeffamine2000_4 355 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_4_2'], parms) + '_133.txt', 'yo',
    'DGEBA-Jeffamine2000_4_2 133 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_4_2'], parms) + '_355.txt', 'yx',
    'DGEBA-Jeffamine2000_4_2 355 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_5'], parms) + '_133.txt', 'ko',
    'DGEBA-Jeffamine2000_5 133 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_5'], parms) + '_355.txt', 'kx',
    'DGEBA-Jeffamine2000_5 355 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_6'], parms) + '_133.txt', 'ro',
    'DGEBA-Jeffamine2000_6 133 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, 
    qcm.find_base_fig_name(sample['DGEBA-Jeffamine2000_RT_6'], parms) + '_355.txt', 'rx',
    'DGEBA-Jeffamine2000_6 355 (2:1)')

plt.legend()
plt.show()


#%% ALL
# make the property and van Gurp-Palmen axes
propfig = qcm.make_prop_axes('props', 't (min.)')
vgpfig = qcm.make_vgp_axes('vgp')

# add the data to the texts
qcm.prop_plot_from_csv(propfig, base_fig_name1 + '_355.txt', 'co', 'DGEBA-Jeffamine230 355 (1:1)')
qcm.prop_plot_from_csv(propfig, base_fig_name1 + '_133.txt', 'cv', 'DGEBA-Jeffamine230 133 (1:1)')
qcm.prop_plot_from_csv(propfig, base_fig_name2 + '_355.txt', 'bo', 'DGEBA-Jeffamine230 355 (2:1)')
# qcm.prop_plot_from_csv(propfig, base_fig_name2 + '_133.txt', 'bv', 'DGEBA-Jeffamine230 133 (2:1)')
qcm.prop_plot_from_csv(propfig, base_fig_name3 + '_355.txt', 'go', 'DGEBA-Jeffamine400 355 (2:1)')
# qcm.prop_plot_from_csv(propfig, base_fig_name3 + '_353.txt', 'gv', 'DGEBA-Jeffamine400 353 (2:1)')
qcm.prop_plot_from_csv(propfig, base_fig_name4 + '_355.txt', 'ro', 'DGEBA-PACM 355 (2:1)')
# qcm.prop_plot_from_csv(propfig, base_fig_name4 + '_353.txt', 'rv', 'DGEBA-PACM 353 (2:1)')
# qcm.prop_plot_from_csv(propfig, base_fig_name5 + '_355.txt', 'rs', 'DGEBA-PACM_2 355 (2:1)')
# qcm.prop_plot_from_csv(propfig, base_fig_name5 + '_133.txt', 'rd', 'DGEBA-PACM_2 353 (2:1)')
propfig['drho_ax'].legend()
propfig['drho_ax'].set_xscale('log')
propfig['grho_ax'].set_xscale('log')
propfig['phi_ax'].set_xscale('log')
qcm.vgp_plot_from_csv(vgpfig, base_fig_name1 + '_355.txt', 'co', 'DGEBA-Jeffamine230 355 (1:1)')
qcm.vgp_plot_from_csv(vgpfig, base_fig_name1 + '_133.txt', 'cv', 'DGEBA-Jeffamine230 133 (1:1)')
qcm.vgp_plot_from_csv(vgpfig, base_fig_name2 + '_355.txt', 'bo', 'DGEBA-Jeffamine230 355 (2:1)')
# qcm.vgp_plot_from_csv(vgpfig, base_fig_name2 + '_133.txt', 'bv', 'DGEBA-Jeffamine230 133 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, base_fig_name3 + '_355.txt', 'go', 'DGEBA-Jeffamine400 355 (2:1)')
# qcm.vgp_plot_from_csv(vgpfig, base_fig_name3 + '_353.txt', 'gv', 'DGEBA-Jeffamine400 353 (2:1)')
qcm.vgp_plot_from_csv(vgpfig, base_fig_name4 + '_355.txt', 'ro', 'DGEBA-PACM 355 (2:1)')
# qcm.vgp_plot_from_csv(vgpfig, base_fig_name4 + '_353.txt', 'rv', 'DGEBA-PACM 353 (2:1)')
# qcm.vgp_plot_from_csv(vgpfig, base_fig_name5 + '_355.txt', 'rs', 'DGEBA-PACM_2 355 (2:1)')
# qcm.vgp_plot_from_csv(vgpfig, base_fig_name5 + '_133.txt', 'rd', 'DGEBA-PACM_2 353 (2:1)')

plt.legend()
plt.show()

