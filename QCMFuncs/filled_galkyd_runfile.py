#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import QCM_functions as qcm
import DMA_functions as dma
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

w_wet=np.array([0, 0.05, 0.1, 0.2, 0.4])
w_dry=w_wet/(0.43*w_wet+0.57)
phi_ZnO = (w_dry/5.6)/(w_dry/5.6+(1-w_dry)/1.2)
rho=5.6*phi_ZnO+(1-phi_ZnO)*1.2

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
phi000props = legacy.plot_qcmprops(fig1,w00_1,'353',r'$\phi_{\mathrm{ZnO}}=0$', '+-')
phi019props = legacy.plot_qcmprops(fig1,w05_1,'353',r'$\phi_{\mathrm{ZnO}}=0.02$', 'o-')
phi040props = legacy.plot_qcmprops(fig1,w10_2,'353',r'$\phi_{\mathrm{ZnO}}=0.04$', 'v-')
phi086props = legacy.plot_qcmprops(fig1,w20_3,'353',r'$\phi_{\mathrm{ZnO}}=0.09$', 's-')
phi200props = legacy.plot_qcmprops(fig1,w40_1,'353',r'$\phi_{\mathrm{ZnO}}=0.20$', '^-')

#legacy.plot_qcmprops(fig1,w50_2,'353','50% ZnO', 'D-')
fig1['phi_ax'].legend(labelspacing=0)
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
Tref = 20
plt.close('all')
parms = {'Trange_plot':[-10, 50], 'Trange_spline':[-30, 70]}
parms['colortype']='g'
parms['filltype']='full'
parms['markertype']='o'
parms['markersize']=2.5
parms['labeltext']=r'$\phi_{\mathrm{ZnO}}=0$'
sample_ZnO_00_2 = dma.DMAread_sturdy(dmadir, ZnO_00_2, parms)
parms['figinfo'] = dma.DMAplot(sample_ZnO_00_2, parms, Tref)

parms['colortype']='k'
parms['labeltext']=r'$\phi_{\mathrm{ZnO}}=0.02$'
sample_ZnO_05 = dma.DMAread_sturdy(dmadir, ZnO_05, parms)
parms['figinfo'] = dma.DMAplot(sample_ZnO_05, parms, Tref)

parms['colortype']='r'
parms['labeltext']=r'$\phi_{\mathrm{ZnO}}=0.04$'
sample_ZnO_10 = dma.DMAread_sturdy(dmadir, ZnO_10, parms)
parms['figinfo'] = dma.DMAplot(sample_ZnO_10, parms, Tref)

parms['colortype']='b'
parms['labeltext']=r'$\phi_{\mathrm{ZnO}}=0.09$'
sample_ZnO_20d = dma.DMAread_sturdy(dmadir, ZnO_20d, parms)
parms['figinfo'] = dma.DMAplot(sample_ZnO_20d, parms, Tref)

# adjust y limit for phase angle plots
parms['figinfo']['dma_ax2'].set_ylim(top=60)
parms['figinfo']['vgp_ax'].set_ylim(top=60)

# now add QCM points to the vgp plot
grho3_qcm = np.array([])
phi_qcm = np.array([])
faT_qcm = 1.5e7*np.ones(5)

for datadict in [phi000props, phi019props, phi040props, phi086props, phi200props]:
    grho3_qcm = np.append(grho3_qcm, datadict['grho3'][-1])
    phi_qcm = np.append(phi_qcm, datadict['phi'][-1])

g_qcm = grho3_qcm/(rho[0:5]*1000)
    
parms['figinfo']['vgp_ax'].plot(3*g_qcm, phi_qcm, 'rx', markersize=10, label='QCM')
parms['figinfo']['estar_ax'].plot(faT_qcm, 3*g_qcm, 'rx', markersize=10, label='QCM')

# now add the nanoindentation points to the estar plot
nu = np.array([0.5, 0.5])
Er = np.array([2.3e8, 4.11e9])
Estar_indent = Er*(1-nu **2 )

parms['figinfo']['estar_ax'].plot(1, Estar_indent[0], 'go', markersize=8,
     label='indent')
parms['figinfo']['estar_ax'].plot(1, Estar_indent[1], 'bs', markersize=8,
     label='indent')

# add legend info
parms['figinfo']['estar_ax'].legend(loc='center', bbox_to_anchor=(-0.9, 0.5))
parms['figinfo']['dma_ax1'].legend(loc='best', labelspacing=0, handlelength=1,
     borderaxespad=0.2)
parms['figinfo']['dma_ax2'].legend(loc='best', labelspacing=0, borderpad=0.2,
     borderaxespad=-0.2, handlelength=1, framealpha=1)

# add titles for subplots
parms['figinfo']['dma_ax1'].set_title('(a)')
parms['figinfo']['dma_ax2'].set_title('(b)')
parms['figinfo']['dma_ax3'].set_title('(c)')

plt.figure('dma').tight_layout()
plt.figure('vgp').tight_layout()

plt.figure('dma').savefig('/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/dmasummary.eps')
plt.figure('vgp').savefig('/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/vgpsummary.eps')

#%%  now plot the conc. dependence of G for the qcm data

qcm_gfig = plt.figure('qcm_g', figsize=(3,3))
qcm_gax = qcm_gfig.add_subplot(111)
qcm_gax.plot(phi_ZnO, g_qcm, 'b+', markersize=14, label='QCM')
qcm_gax.set_xlabel(r'$\phi_{\mathrm{ZnO}}$')
qcm_gax.set_ylabel(r'$|G^*|$ (Pa)')

def fill_fit(phi):
    return 1+2.5*phi+14.1*phi ** 2

fill_fit = np.vectorize(fill_fit)

phi_for_plot = np.linspace(0, phi_ZnO.max(), 100)

qcm_gax.plot(phi_for_plot, 9.7e8*fill_fit(phi_for_plot), '-b', label='Guth-Gold')

qcm_gax.legend()
qcm_gfig.tight_layout()

plt.figure('qcm_g').savefig('/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/qcm_gvals.eps')
 



