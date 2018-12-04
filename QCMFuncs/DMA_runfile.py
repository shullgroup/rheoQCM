#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ken
"""

# %% import the routines we need and set up the combined Van Gurp-Palmen fig.
import DMA_sampledefs
import DMA_functions as dma
import matplotlib.pyplot as plt
import QCM_functions as qcm
parms = {}

#%% make the dmaplots from Lauren's data
# start with pure galkyd sample
sampletemp = dma.DMAread_strudy(DMA_sampledefs.PBD_14())

# %% make plots each sample
plt.close('all')
Tref = 20
parms['show_springpot_fit'] = 'yes'
# 1-4 PBD sample
sampletemp = dma.DMAread(DMA_sampledefs.PBD_14())
samplePBD = qcm.DMAplot(sampletemp, parms, Tref)

# 1-4 PI sample
sampletemp = qcm.DMAread(DMA_sampledefs.PI_14())
samplePI_14 = qcm.DMAcalc(sampletemp, parms, Tref)

# sbr sample
sampletemp = qcm.DMAread(DMA_sampledefs.sbr())
samplesbr = qcm.DMAcalc(sampletemp, parms, Tref)

# 3-4 PI sample
sampletemp = qcm.DMAread(DMA_sampledefs.PI_34())
samplePI_34 = qcm.DMAcalc(sampletemp, parms, Tref)

# %% make the VGP plots
plt.close('all')
vgpfig = plt.figure('VGP', figsize=(6, 6))

addVGPplot(samplePBD, samplePBD, vgpfig, 1)
addVGPplot(samplePI_14, samplePI_14, vgpfig, 2)
addVGPplot(samplesbr, samplesbr, vgpfig, 3)
addVGPplot(samplePI_34, samplePI_34, vgpfig, 4)

vgpfig.tight_layout()
vgpfig.savefig('figures/vgpfig.svg')
plt.show()
# %% make the fig that shows the dma data with the springpot fits
Rheodata(samplesbr, Tref)

# now make the fig that illustrates the narrowing of the possible values of B
Bcompare(samplesbr, Tref)

# %% compare the old and new SBRsamples
plt.close('all')

sampletemp = qcm.DMAread(DMA_sampledefs.sbr35())
DMAplot(sampletemp, samplesbr, Tref)

sampletemp = qcm.DMAread(DMA_sampledefs.sbr36())
DMAplot(sampletemp, samplesbr, Tref)

sampletemp = qcm.DMAread(DMA_sampledefs.sbr37())
DMAplot(sampletemp, samplesbr, Tref)

sampletemp = qcm.DMAread(DMA_sampledefs.sbr38())
DMAplot(sampletemp, samplesbr, Tref)

sampletemp = qcm.DMAread(DMA_sampledefs.sbr39())
DMAplot(sampletemp, samplesbr, Tref)

# %% unfilled SBR
plt.close('all')
Tref = -20
sampletemp = qcm.DMAread(DMA_sampledefs.sbr39())
sample39sbr = qcm.DMAcalc(sampletemp, parms, Tref)

# %% SBR 50 series
plt.close('all')
parms = {'sp_parms' : sample39sbr['sp_parms']}
parms['aTfit'] = 'ref'
parms['show_springpot_fit'] = 'no'
parms['mark'] = '+'
parms['add_titles'] = 'no'
parms['new_axes'] = 'yes'
parms['tandelta'] = 'yes'
parms['calc_plot'] = 'no'

sampletemp = qcm.DMAread(DMA_sampledefs.sbr50())
if parms['aTfit'] == 'self':
    sampletemp = qcm.DMAcalc(sampletemp, parms, Tref)
parms['figinfo'] = DMAplot(sampletemp, parms, Tref)
parms['new_axes'] = 'no'
dmadata1 = parms['figinfo']['dmadata1']
data_idx = list(dmadata1.keys())[0]
legend_plots=[dmadata1[data_idx]]
legend_text = ['NU50']

parms['mark'] = 's'
sampletemp = qcm.DMAread(DMA_sampledefs.sbr51())
if parms['aTfit'] == 'self':
    sampletemp = qcm.DMAcalc(sampletemp, parms, Tref)
figinfo = DMAplot(sampletemp, parms, Tref)
dmadata1 = figinfo['dmadata1']
data_idx = list(dmadata1.keys())[0]
legend_plots.append(dmadata1[data_idx])
legend_text.append('NU51')

parms['mark'] = 'o'
sampletemp = qcm.DMAread(DMA_sampledefs.sbr52())
if parms['aTfit'] == 'self':
    sampletemp = qcm.DMAcalc(sampletemp, parms, Tref)
figinfo = DMAplot(sampletemp, parms, Tref)
dmadata1 = figinfo['dmadata1']
data_idx = list(dmadata1.keys())[0]
legend_plots.append(dmadata1[data_idx])
legend_text.append('NU52')

parms['mark'] = '^'
sampletemp = qcm.DMAread(DMA_sampledefs.sbr53())
if parms['aTfit'] == 'self':
    sampletemp = qcm.DMAcalc(sampletemp, parms, Tref)
figinfo = DMAplot(sampletemp, parms, Tref)
dmadata1 = figinfo['dmadata1']
data_idx = list(dmadata1.keys())[0]
legend_plots.append(dmadata1[data_idx])
legend_text.append('NU53')

parms['mark'] = 'x'
sampletemp = qcm.DMAread(DMA_sampledefs.sbr39())
if parms['aTfit'] == 'self':
    sampletemp = qcm.DMAcalc(sampletemp, parms, Tref)
figinfo = DMAplot(sampletemp, parms, Tref)
dmadata1 = figinfo['dmadata1']
data_idx = list(dmadata1.keys())[0]
legend_plots.append(dmadata1[data_idx])
legend_text.append('NU39')

figinfo['dma_ax1'].legend(legend_plots, legend_text)
figinfo['dma_ax3'].set_xlim(left=-45, right=-10)
figinfo['dma_ax3'].set_ylim(bottom=1e-2, top=1e6)

figinfo['dmafig'].set_size_inches(8, 5)
figinfo['dmafig'].savefig('dmafig_fix_aT.pdf')

# %%
plt.close('all')

sampletemp = qcm.DMAread(DMA_sampledefs.sbr34())
DMAplot(sampletemp, samplesbr, Tref)

sampletemp = qcm.DMAread(DMA_sampledefs.sbr39())
DMAplot(sampletemp, samplesbr, Tref)

sampletemp = qcm.DMAread(DMA_sampledefs.sbr44())
DMAplot(sampletemp, samplesbr, Tref)

plt.show()