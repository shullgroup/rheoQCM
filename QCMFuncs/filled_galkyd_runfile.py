#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import QCM_functions as qcm
import DMA_functions as dma
import hdf5storage
import legacy_functions as legacy

# define locations of the QCM data files
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

# now define locations of the DMA files
dmadir = '/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/DMA/'
ZnO_00_2 = 'Alkyd_sample_2_150423_shifted'
ZnO_00_4a = 'Alkyd_sample_4_150602_shifted'
ZnO_00_4b = 'Alkyd_sample_4_150602_shiftedv2'
ZnO_00_6 = 'Alkyd_sample_6_150728_shifted'
ZnO_05 = 'ZnO_5_percent_sample_3_150702_shifted'
ZnO_10 = 'ZnO_10_percent_sample_2_150709_shifted'
ZnO_20a = 'ZnO_20_percent_sample_2_150709_shifted'
ZnO_20b = 'ZnO_20_percent_sample_3_150709_shifted'
ZnO_20c = 'ZnO_20_percent_sample_4_5_Combined_150716_shifted'
ZnO_20d = 'ZnO_20_percent_sample_4_150716_shifted'

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

#%%  generate the dictionaries used by the plot function
parms = {}
Tref = 20
sample_ZnO_00_2 = dma.DMAread_sturdy(dmadir, ZnO_00_2, parms)
sample_ZnO_00_4b = dma.DMAread_sturdy(dmadir, ZnO_00_4b, parms)
sample_ZnO_00_6 = dma.DMAread_sturdy(dmadir, ZnO_00_6, parms)
sample_ZnO_05 = dma.DMAread_sturdy(dmadir, ZnO_05, parms)
sample_ZnO_10 = dma.DMAread_sturdy(dmadir, ZnO_10, parms)
sample_ZnO_20a = dma.DMAread_sturdy(dmadir, ZnO_20a, parms)
sample_ZnO_20b = dma.DMAread_sturdy(dmadir, ZnO_20b, parms)
sample_ZnO_20c = dma.DMAread_sturdy(dmadir, ZnO_20c, parms)
sample_ZnO_20d = dma.DMAread_sturdy(dmadir, ZnO_20d, parms)

#%%  comparision of the different unfilled samples
parms = {'Trange':[-20, 50]}
dma.DMAplot(sample_ZnO_00_2, parms, Tref)
dma.DMAplot(sample_ZnO_00_4b, parms, Tref)
dma.DMAplot(sample_ZnO_00_6, parms, Tref)

#%%  comparison of the 20% filled samples 
parms = {'Trange':[-20, 80], 'make_spline':'yes'}
plt.close('all')
dma.DMAplot(sample_ZnO_20a, parms, Tref)
dma.DMAplot(sample_ZnO_20b, parms, Tref)
dma.DMAplot(sample_ZnO_20c, parms, Tref)
dma.DMAplot(sample_ZnO_20d, parms, Tref)

#%%
parm = {}
Tref = 20
plt.close('all')
parms = {'Trange':[-20, 50]}
parms['etype']= 'estor'
sample_ZnO_00_2 = dma.DMAread_sturdy(dmadir, ZnO_00_2, parms)
parms['figinfo'] = dma.DMAplot(sample_ZnO_00_2, parms, Tref)
parms['Trange'] = [-20, 60]

sample_ZnO_05 = dma.DMAread_sturdy(dmadir, ZnO_05, parms)
parms['figinfo'] = dma.DMAplot(sample_ZnO_05, parms, Tref)

sample_ZnO_10 = dma.DMAread_sturdy(dmadir, ZnO_10, parms)
parms['figinfo'] = dma.DMAplot(sample_ZnO_10, parms, Tref)

sample_ZnO_20d = dma.DMAread_sturdy(dmadir, ZnO_20d, parms)
parms['figinfo'] = dma.DMAplot(sample_ZnO_20d, parms, Tref)

plt.figure('dma').savefig('dmasummary.svg')
plt.figure('vgp').savefig('vgpsummary.svg')
