#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 09:17:40 2018

@author: ken
"""

import matplotlib.pyplot as plt
import QCM_functions as qcm
import numpy as np

overlayer = {'grho3':1e12, 'phi':1}
n = 3
grho3 = 1e12
phi = 5

drho=np.linspace(0, 1e-4, 10)

fig = plt.figure('confinement', figsize=(6,3))
ax_f = fig.add_subplot(121)
ax_g = fig.add_subplot(122)
ax_f.set_xlabel('$d_f$ (nm)')
ax_g.set_xlabel('$d_f$ (nm)')
ax_f.set_ylabel(r'$\Delta f$'+ ' (nm)')
ax_g.set_ylabel(r'$\Delta \Gamma$'+ ' (nm)')

dm = [0, 1, 2, 3]  #membrane thickness in microns
for d_mem in dm:
    overlayer['drho'] = d_mem*1e-3
    delfstar = qcm.delfstarcalc(n, drho, grho3, phi, overlayer)
    ax_f.plot(drho*1e6, delfstar.real, '-+', label=str(d_mem)+' $\mu$m')
    ax_g.plot(drho*1e6, delfstar.imag, '-+', label=str(d_mem)+' $\mu$m')
    
ax_f.legend()
ax_g.legend()
fig.tight_layout()
fig.savefig('confinement.pdf')

