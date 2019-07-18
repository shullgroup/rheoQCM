#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 00:29:00 2018

@author: ken
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/home/ken/Mydocs/Python/QCM_py/QCMFuncs')
import QCM_functions as qcm


thick_glass = {'drho':5e-3, 'grho3':1e12, 'phi':1}
weakglass = {'drho':1e-5, 'grho3':1e10, 'phi':30}
water = {'drho':np.inf, 'grho3':1e8, 'phi':90}
thin_water = {'drho':1e-5, 'grho3':1e8, 'phi':90}
rubber = {'drho':5e-5, 'grho3':1e10, 'phi':45}
johannsmann = {'drho':4e-4, 'grho3':5e11, 'phi':11.3}


fig = {}; ax = {}
parms = {'squaren':True}
parms['nvals'] = np.arange(1, 23, 2)
def makefig(num, parms):
    squaren = parms.get('squaren', True)
    fig[num], ax[num] = plt.subplots(1,2, figsize=(6,3))
    ax[num][0].set_ylabel(r'$\Delta f/n$ (Hz)')
    if squaren:
        ax[num][0].set_xlabel(r'$n^2$')
        ax[num][1].set_xlabel(r'$n^2$')
    else:
        ax[num][0].set_xlabel(r'$n$')
        ax[num][1].set_xlabel(r'$n$')
    ax[num][1].set_ylabel(r'$\Delta \Gamma/n$ (Hz)')
    fig[num].tight_layout()


def addplot_both(fignum, film, parms):
    squaren = parms.get('squaren', True)
    nvals = parms['nvals']
    if squaren:
        xdata = nvals ** 2
    else:
        xdata = nvals
    delfstar_sla = np.zeros(len(nvals),dtype=complex)
    delfstar_LL = np.zeros(len(nvals),dtype=complex)
    layers = {'film':film}

    for i,n in enumerate(nvals):
        delfstar_sla[i] = qcm.calc_delfstar(n, layers, 'sla')/n
        delfstar_LL[i] = qcm.calc_delfstar(n, layers, 'LL')/n

    ax[fignum][0].plot(xdata, delfstar_sla.real, '+-', label='SLA')
    ax[fignum][0].plot(xdata, delfstar_LL.real, 'x-',  label='LL')
    ax[fignum][0].legend()
    
    ax[fignum][1].plot(xdata, delfstar_sla.imag, '+-', label='SLA')
    ax[fignum][1].plot(xdata, delfstar_LL.imag, 'x-', label='LL')
    ax[fignum][1].legend()


def addplot_LL(fignum, film, legend, parms):  
    squaren = parms.get('squaren', True)
    nvals = parms['nvals']
    if squaren:
        xdata = nvals ** 2
    else:
        xdata = nvals

    delfstar_LL = np.zeros(len(nvals),dtype=complex)
    layers = {'film':film}

    for i,n in enumerate(nvals):
        delfstar_LL[i] = qcm.calc_delfstar(n, layers, 'LL')/n
        
    ax[fignum][0].plot(xdata, delfstar_LL.real, 'x-', label=legend)
    ax[fignum][1].plot(xdata, delfstar_LL.imag, 'x-', label=legend)

plt.close('all')
makefig(1, parms)
film = {'drho':4e-4, 'grho3':5e11, 'phi':11.3}
addplot_both(1, film, parms)

makefig(2, parms)
films = {1:{'drho':4e-4, 'grho3':2e11, 'phi':11.3},
         2:{'drho':4e-4, 'grho3':5e11, 'phi':11.3},
         3:{'drho':4e-4, 'grho3':1e12, 'phi':11.3}}

legends = {1:'$|G^*|=2x10^8$ Pa',
           2:'$|G^*|=5x10^8$ Pa',
           3:'$|G^*|=10^9$ Pa'}

for n in [1, 2, 3]:
    addplot_LL(2, films[n], legends[n], parms)

ax[2][1].legend()

makefig(3, parms)
films = {1:{'drho':4e-4, 'grho3':5e11, 'phi':5},
         2:{'drho':4e-4, 'grho3':5e11, 'phi':10},
         3:{'drho':4e-4, 'grho3':5e11, 'phi':20}}
legends = {1:'$\phi=5^\circ$',
           2:'$\phi=10^\circ$',
           3:'$\phi=20^\circ$'}

for n in [1, 2, 3]:
    addplot_LL(3, films[n], legends[n], parms)
ax[3][0].legend()
ax[3][1].legend()

# now save the figures
fig[1].savefig('fig1.svg')
fig[2].savefig('fig2.svg')
fig[3].savefig('fig3.svg')

#%%
thick_glass = {'drho':5e-3, 'grho3':1e12, 'phi':1}
makefig(4, parms)
addplot_both(4, thick_glass, parms)
fig[4].savefig('fig4.svg')

parms['nvals'] = [1, 3, 5]
parms['squaren'] = False

makefig(5, parms)
addplot_both(5, thick_glass, parms)
fig[5].savefig('fig5.svg')