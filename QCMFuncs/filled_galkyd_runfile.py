#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import QCM_functions as qcm
import os
import hdf5storage
import legacy_functions as legacy

w00_1 = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/LFS_GZO0_01_data'
w00_ref = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/LFS_G_08_data'
w05_1 = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/LFS_GZO5_01_data'
w10_1 = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/LFS_GZO10_01_data'
w10_2 = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/LFS_GZO10_02_data'
w20_1 = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/LFS_GZO20_01_data'
w20_2 = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/LFS_GZO20_02_data'
w20_3 = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/LFS_GZO20_03_data'
w40_1 = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/LFS_GZO40_01_data'
w50_1 = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/LFS_GZO50_01_data'  # no calc file for this one - no actual solutions
w50_2 = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/LFS_GZO50_02_data'
w50_3 = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/LFS_GZO50_03_data'

wwet = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.5])
wdry = wwet/(0.43*wwet+0.57)
phi = (wdry/5.6)/((wdry/5.6)+(1-wdry)/1.2)
norm = 1+10*phi


#%% now compare results for increasing volume fractions, all using 353
fig1 = qcm.make_prop_axes('fig1', 'time')
legacy.plot_qcmprops(fig1,w00_1,'353','0% ZnO', '+-')
legacy.plot_qcmprops(fig1,w05_1,'353','5% ZnO', 'o-')
legacy.plot_qcmprops(fig1,w10_2,'353','10% ZnO', 'v-')
legacy.plot_qcmprops(fig1,w20_3,'353','20% ZnO', 's-')
legacy.plot_qcmprops(fig1,w40_1,'353','40% ZnO', '^-')
#legacy.plot_qcmprops(fig1,w50_2,'353','50% ZnO', 'D-')

fig1['phi_ax'].legend()
fig1['figure'].savefig('/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/filledqcm.svg')

#%% make Van Gurp-Palmen plots
fig2 = qcm.make_vgp_axes('fig2')

legacy.plot_vgp(fig2,w00_1,'353','0% ZnO', '+-')
legacy.plot_vgp(fig2,w05_1,'353','5% ZnO', 'o-')
legacy.plot_vgp(fig2,w10_2,'353','10% ZnO', 'v-')
legacy.plot_vgp(fig2,w20_3,'353','20% ZnO', 's-')
legacy.plot_vgp(fig2,w40_1,'353','40% ZnO', '^-')
#legacy.plot_vgp(fig2,w50_2,'353','50% ZnO', 'D-')

fig2['vgp_ax'].legend()
fig2['figure'].savefig('/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/vgp.pdf')