#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# these are a bunch of functions to utlize old data files - generally
# not useful for new data
import numpy as np
import hdf5storage
import pdb

def plot_qcmprops(propfig,inputfile,calctype,labeltext, sym):
    props={}
    cstr = 'c' + calctype
    data = hdf5storage.loadmat(inputfile+'_calc.mat')
    dlam3 = data['d3sol'][cstr][0]
    drho = data['drhosol'][cstr][0,0][0]
    grho3 = data['grho3sol'][cstr][0,0][0]
    phi = data['phisol'][cstr][0,0][0]
    idx = data['idxsol'][cstr][0,0][0]-1
    idx = np.array(idx)
    time = data['time'][0]/(60*24)
    idx = idx[np.where(time[idx]>1)]
    
    propfig['drho_ax'].plot(time[idx], 1000*drho[idx], sym, label=labeltext)
    propfig['grho_ax'].plot(time[idx], grho3[idx]/1000, sym, label=labeltext)
    propfig['phi_ax'].plot(time[idx], phi[idx], sym, label=labeltext)
    
    timescale(propfig,'linear')
    maxtime(propfig, 350)

    props['drho'] = drho[~np.isnan(drho)]
    props['phi'] = phi[~np.isnan(phi)]
    props['grho3'] = grho3[~np.isnan(grho3)]
    return props

    
def timescale(propfig, scale):
    # makes the time axis either logarithmic or linear
    propfig['drho_ax'].set_xscale(scale)
    propfig['grho_ax'].set_xscale(scale)
    propfig['phi_ax'].set_xscale(scale)
    propfig['drho_ax'].set_xlabel('time (days)')
    propfig['grho_ax'].set_xlabel('time (days)')
    propfig['phi_ax'].set_xlabel('time (days)')
   
    
def maxtime(propfig, rightlimit):
    # makes the time axis either logarithmic or linear
    propfig['drho_ax'].set_xlim(right=rightlimit)
    propfig['grho_ax'].set_xlim(right=rightlimit)
    propfig['phi_ax'].set_xlim(right=rightlimit)


def plot_vgp(vgpfig,inputfile,calctype,labeltext, sym):
    cstr = 'c' + calctype
    data = hdf5storage.loadmat(inputfile+'_calc.mat')
    grho3 = data['grho3sol'][cstr][0,0][0]
    phi = data['phisol'][cstr][0,0][0]
    idx = data['idxsol'][cstr][0,0][0]-1
    idx = np.array(idx)
    time = data['time'][0]/(60*24)
    idx = idx[np.where(time[idx]>1)]

    vgpfig['vgp_ax'].plot(grho3[idx]/1000, phi[idx], sym, label=labeltext)
    