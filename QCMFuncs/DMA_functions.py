#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ken
"""

import numpy as np
import matplotlib.pyplot as plt
import QCM_functions as qcm
from scipy.optimize import fsolve, curve_fit
import hdf5storage
from scipy.interpolate import LSQUnivariateSpline
import pandas as pd


def DMAread_delgado(sample, parms):
    # this reads the dMA data and puts it into the dmadata dictionary so we
    # can more easilty plot it later
    dmadir = parms['dmadir']
    file = sample['dmadata']['filename']

    # read in the DMA data from the orignal data file
    dma__df = pd.read_csv(dmadir + file +
                          '/mech_data.txt',  header=[0, 1], sep='\t',
                          encoding="ISO-8859-1")
    
    freq = np.array(dma__df)[:, 0].astype(float)
    estor = np.array(dma__df)[:, 1].astype(float)
    eloss = np.array(dma__df)[:, 2].astype(float)
    tanphi = np.array(dma__df)[:, 3].astype(float)
    phi = np.rad2deg(np.arctan(tanphi))
    estar = (estor ** 2 + eloss ** 2) ** 0.5

    input_Tfile = (dmadir + file + '/temp_data.txt')

    # now read in the file that contains temperatures and shift factors
    # column 0 is T (C)
    # column 1 is aT (we don't care much about the other columns
    T_df = pd.read_csv(input_Tfile, header=[0, 1],
                       sep='\t', encoding="ISO-8859-1")
    
    T = np.array(T_df)[:, 0].astype(float)
    aT_raw = np.array(T_df)[:, 1].astype(float)

    Tnum = T.shape[0]  # the number of temperatures
    ftotal = freq.shape[0]
    fnum = ftotal/Tnum  # number of frequencies at each temp
    
    if fnum != np.fix(fnum):
        print('non-integral fnum')
    else:
        fnum = int(fnum)
        
    dma_np = np.zeros([ftotal, 5])
    
    dma_np[:, 0] = freq
    dma_np[:, 1] = estar
    dma_np[:, 2] = phi
    dma_np[:, 3] = estor
    dma_np[:, 4] = eloss
    dma_np = np.reshape(dma_np, (Tnum, fnum, 5))
    
    dmadata = {}
    dmadata['f'] = dma_np[:, :, 0]
    dmadata['estar'] = dma_np[:, :, 1]
    dmadata['phi'] = dma_np[:, :, 2]    
    dmadata['estor'] = dma_np[:, :, 3]
    dmadata['eloss'] = dma_np[:, :, 4]
    dmadata['dma_np'] = dma_np
    dmadata['aT_raw'] = aT_raw
    dmadata['T'] = T

    sample['dmadata'] = dmadata
    return sample


def DMAread_sturdy(file, parms):
    dmadir = parms['dmadir']
    # a read file for Laruen's MAT files
    data_input = hdf5storage.loadmat(dmadir + file)
    # not surewhy dtype needs to be specified here, but it seems to be 
    # necessary at least for estor
    freq = np.array(data_input['rawdata'][2,0][:,0], dtype=np.float64)
    estor = np.array(data_input['rawdata'][2,1][:,0], dtype=np.float64)
    eloss = np.array(data_input['rawdata'][2,2][:,0], dtype=np.float64)
    T = np.array(data_input['T'][0,:]) # these are Temps 
    aT_raw = np.array(data_input['shift_factors'][:,0])
    estar =(estor**2+eloss**2)**0.5
    phi = np.degrees(np.arctan(eloss/estor))
    ftotal = freq.shape[0]
    Tnum = data_input['T'].shape[1]  # the number of temperatures
    fnum = ftotal/Tnum  # number of frequencies at each temp
    
    if fnum != np.fix(fnum):
        print('non-integral fnum')
    else:
        fnum = int(fnum)
    dma_np = np.zeros([ftotal, 5])
    
    dma_np[:, 0] = freq
    dma_np[:, 1] = estar
    dma_np[:, 2] = phi
    dma_np[:, 3] = estor
    dma_np[:, 4] = eloss
    dma_np = np.reshape(dma_np, (Tnum, fnum, 5))
    
    dmadata = {}
    dmadata['f'] = dma_np[:, :, 0]
    dmadata['estar'] = dma_np[:, :, 1]
    dmadata['phi'] = dma_np[:, :, 2]    
    dmadata['estor'] = dma_np[:, :, 3]
    dmadata['eloss'] = dma_np[:, :, 4]
    dmadata['dma_np'] = dma_np
    dmadata['aT_raw'] = aT_raw
    dmadata['T'] = T
    
    sample={'dmadata': dmadata}
    return sample


def DMAplot(sample, parms, Tref):
    dmadata = sample['dmadata']
    figinfo = parms.get('figinfo',{})
    Trange = parms.get('Trange', {})
    markertype = parms.get('markertype', '+')
    markersize = parms.get('markersize', 8)
    colortype = parms.get('colortype', 'r')
    filltype = parms.get('filltype', 'none')
    sp_parms = parms.get('sp_parms', {})
    # older versions had sp_parms in sample instead of parms
    labeltext = parms.get('labeltext','nolab')
    
    Trange['spline']=Trange.get('spline', [Trange['plot'][0]-10,
                                Trange['plot'][1]+10])

    # read in the temperatures and shift factors
    T = dmadata['T']
    aT = dmadata['aT_raw']

    # deterimine Temp. indices to use for plotting and spline fits
    Tind = {}
    Tind['plot'] = np.where((T >= Trange['plot'][0]) & (T <= Trange['plot'][1]))[0]
    Tind['spline'] = np.where((T >= Trange['spline'][0]) & (T <= Trange['spline'][1]))[0]
    
    qcmsize = 8  # marker size for qcm points
    
    # generate spline fits to the DMA data
    faT_plot, estar_spline_raw, phi_spline_raw = make_property_splines(dmadata, aT, Tind)
    
    
    # now we add the QCM points and do the fits
    K = 2.4e9  # bulk modulus of rubber from Polymer 35, 2759–2763 (1994)
    
    # now we read in the QCM data
    qcmdata = sample.get('qcmdata',{})
    qcm_G = np.array(qcmdata.get('qcm_rhog',[]))/np.array(qcmdata.get('rho', []))
    qcm_E = 9*K * qcm_G/(3*K + qcm_G)
    qcm_phi = np.array(qcmdata.get('qcm_phi', []))
    qcm_T = np.array(qcmdata.get('qcm_T', []))
    nqcm_aT = qcm_E.shape[0]
    qcm_aT = np.zeros(nqcm_aT)
    
    # determine the shift factors by getting Estar values to match DMA data
    for i in np.arange(nqcm_aT):
        def ftosolve(logf):
            return abs(estar_spline_raw(logf) - np.log(qcm_E[i]))
        
        logf_guess = np.log(sp_parms['fref_raw']) - 5  # usually works as a guess
        qcm_aT[i] = np.exp(fsolve(ftosolve, logf_guess))/1.5e7
 
    # develop spline fits 
    T_total = np.append(T[Tind['spline']], qcm_T)
    aT_total = np.append(aT[Tind['spline']], qcm_aT)

    # generate spline fit for raw shift factors 
    aT_spline_raw = LSQUnivariateSpline(T_total, np.log(aT_total), [0])
    
    # determine raw reference temperature
    Tref_raw = fsolve(aT_spline_raw, 25)
    aTfix = np.exp(aT_spline_raw(Tref)-aT_spline_raw(Tref_raw))
    
    def aT_spline(T):
        return np.exp(aT_spline_raw(T))/aTfix

    # adjust shift factors based on ref. temp.
    aT = aT/aTfix
    qcm_aT = qcm_aT/aTfix
    faT_plot = faT_plot/aTfix
    
    # redefine the spline fits to account for the correct ref. temp
    def estar_spline(faT):
        return np.exp(estar_spline_raw(np.log(aTfix*faT)))

    def phi_spline(faT):
        return phi_spline_raw(np.log(aTfix*faT))

    # make new dma plot if it doesn't exist
    if not plt.fignum_exists('dma'):
        dmafig = plt.figure('dma', figsize=(9, 3))
        dma_ax1 = dmafig.add_subplot(1, 3, 1)
        dma_ax1.loglog(True)
        dma_ax1.set_xlabel('$fa_T$ (s$^{-1}$)')
        dma_ax1.set_ylabel('$|E^*|$ (Pa)')
        dma_ax2 = dmafig.add_subplot(1, 3, 2)
        dma_ax2.semilogx(True)

        if parms.get('tandelta', 'no') == 'yes':
            dma_ax2.set_ylabel(r'$\tan(\delta)$')
        else:
            dma_ax2.set_ylabel('$\\phi$ (deg.)')

        dma_ax2.set_xlabel('$fa_T$ (s$^{-1}$)')

        dma_ax3 = dmafig.add_subplot(1, 3, 3)
        dma_ax3.semilogy(True)
        dma_ax3.set_xlabel(r'$T\:(^{\circ}$C)')
        dma_ax3.set_ylabel('$a_T$')
    else:
        dmafig = figinfo['dmafig']
        dma_ax1 = figinfo['dma_ax1']
        dma_ax2 = figinfo['dma_ax2']
        dma_ax3 = figinfo['dma_ax3']

    # set up the van Gurp-Palmen plot
    if not plt.fignum_exists('vgp'):
        vgpfig = plt.figure('vgp', figsize=(8,3))
        vgp_ax = vgpfig.add_subplot(122)
        vgp_ax.semilogx(True)
        vgp_ax.set_ylabel(r'$\phi$ (deg.)')
        vgp_ax.set_xlabel('$|E^*|$ (Pa)')
        vgp_ax.set_title('(b)')
        estar_ax = vgpfig.add_subplot(121)
        estar_ax.loglog(True)
        estar_ax.set_xlabel(r'$fa_T$ s$^{-1}$')
        estar_ax.set_ylabel('$|E^*|$ (Pa)')
        estar_ax.set_title('(a)')
    else:
        vgpfig = figinfo['vgpfig']
        vgp_ax = figinfo['vgp_ax']
        estar_ax = figinfo['estar_ax']
        
    # reset the color cycling
    dma_ax1.set_prop_cycle(None)
    dma_ax2.set_prop_cycle(None)
    dma_ax3.set_prop_cycle(None)
    vgp_ax.set_prop_cycle(None)
    estar_ax.set_prop_cycle(None)

    # take the tangent of phi if that is desired
    if parms.get('tandelta', 'no') == 'yes':
        phidata_for_plot = np.tan(np.radians(dmadata['phi']))
    else:
        phidata_for_plot = dmadata['phi']

    # plot the DMA data        
    faT = np.empty_like(dmadata['f'])
    dmadata1 = {}
    dmadata2 = {}
    dmadata3 = {}

    for Tindex in Tind['plot']:
        faT[Tindex, :] = dmadata['f'][Tindex, :] * aT[Tindex]
        dmadata1[Tindex], = (dma_ax1.plot(faT[Tindex, :],
                             dmadata['estar'][Tindex, :], marker=markertype,
                             linestyle='none', fillstyle=filltype, markersize=markersize))
        dmadata2[Tindex], = (dma_ax2.plot(faT[Tindex, :],
                             phidata_for_plot[Tindex, :], marker=markertype,
                             linestyle='none', fillstyle=filltype, markersize=markersize))
        dmadata3[Tindex], = dma_ax3.plot(T[Tindex], aT[Tindex],
                marker=markertype, fillstyle='none', markersize = 12)
        
    # add the QCM data to the plots
    qcmdata1, = dma_ax1.plot(1.5e7*qcm_aT, qcm_E, 'ro', markersize=qcmsize)
    qcmdata2, = dma_ax2.plot(1.5e7*qcm_aT, qcm_phi, 'ro',
                             markersize=qcmsize)
    qcmdata3, = dma_ax3.plot(qcm_T, qcm_aT, 'ro', markersize=qcmsize)
    qcmdata4, = estar_ax.plot(1.5e7*qcm_aT, qcm_E, 'ro', markersize=qcmsize)
    qcmdata5, = vgp_ax.plot(qcm_E, qcm_phi, 'ro', markersize=qcmsize)
    

    # add the VFT fit for the shift factors
    T_for_fit = np.linspace(T[Tind['plot']].min(), T[Tind['plot']].max(), 100)
            
    if parms.get('add_vft_fit', False):
        VFT_fit = np.exp(qcm.vogel(T_for_fit, Tref, B, Tinf))
        aTfit, = dma_ax3.plot(T_for_fit, VFT_fit, 'b-', label='fit')
        
    # add simple spline fits to the shift factor plot
    if parms.get('add_aT_spline', True):
        aTspline_plt, = dma_ax3.plot(T_for_fit, aT_spline(T_for_fit), 'b-')
    
    # add legend to the shift factor plot
    if parms.get('aTlegend', False):
        dma_ax3.legend([dmadata1[Tind['plot'].min()], qcmdata3, aTfit],
                   ['DMA', 'QCM', 'fit'])                     
        
    # now we create the springpot fits and add them to the plot
    if parms.get('show_springpot_fit', False):
        faTfit = qcm.springpot_f(sp_parms)
        Estar = qcm.springpot(faTfit, sp_parms)
        fit1, = dma_ax1.loglog(faTfit, abs(Estar), 'b-')
        fit2, = dma_ax2.semilogx(faTfit, np.angle(Estar, deg=True), 'b-')
        

    # now add titles to plots
    if parms.get('add_titles', True):
        dma_ax1.set_title('(a)')
        dma_ax2.set_title('(b)')
        dma_ax3.set_title('(c)')                        
        
    # add splines to different plots
    vgp_ax.semilogx(estar_spline(faT_plot), phi_spline(faT_plot),
                    color=colortype, linewidth=1)
    estar_ax.loglog(faT_plot, estar_spline(faT_plot),
                   color=colortype, linewidth=1, label=labeltext,
                   markevery=1000, marker=markertype, 
                   markerfacecolor = 'None')
    
        

    # only show on the primary property axes if desired
    if parms.get('show_splines', False):
        dma_ax1.loglog(faT_plot, estar_spline(faT_plot),
                       color=colortype, linewidth=1, label=labeltext)
        dma_ax2.semilogx(faT_plot, phi_spline(faT_plot),
                         color=colortype, linewidth=1, label=labeltext)
        
    # add symbols to vgp plot
    if parms.get('add_vgp_symbols', False):
        Estarmin = 3e8  #  minimum estar for symbols to be added to vgp plot
        vgp_plot_sym = np.logspace(np.log10(faT_plot[0]),
                                   np.log10(faT_plot[-1]), 10)
        
        
        estar_ax.loglog(vgp_plot_sym, estar_spline(vgp_plot_sym),
                        color=colortype,
                        marker=markertype, linestyle='None', fillstyle='none',
                        linewidth=1)
        vgp_plot_sym = vgp_plot_sym[np.where(estar_spline(vgp_plot_sym)
                                            <Estarmin)]
        
        vgp_ax.semilogx(estar_spline(vgp_plot_sym), 
                        phi_spline(vgp_plot_sym), color=colortype,
                        marker=markertype, linestyle='None', fillstyle='none',
                        label=labeltext, linewidth=1)
        
    # reset limits as necessary
    dma_ax1.set_ylim(bottom=5e5, top=3e9)
    estar_ax.set_ylim(bottom=5e5, top=3e9)
    dmafig.tight_layout()
    vgpfig.tight_layout()
    
    # 
    figinfo['dmafig'] = dmafig
    figinfo['dma_ax1'] = dma_ax1
    figinfo['dma_ax2'] = dma_ax2
    figinfo['dma_ax3'] = dma_ax3
    figinfo['vgpfig'] = vgpfig
    figinfo['vgp_ax'] = vgp_ax
    figinfo['estar_ax'] = estar_ax
    
    dmadata['aT'] = aT
    dmadata['qcm_aT'] = qcm_aT
    dmadata['Tind'] = Tind
    dmadata['sp_parms'] = sp_parms
    dmadata['faT_plot'] = faT_plot
    dmadata['estar_spline'] = estar_spline
    dmadata['phi_spline'] = phi_spline
    dmadata['qcm_E'] = qcm_E
    dmadata['qcm_phi'] = qcm_phi
    
    parms['figinfo'] = figinfo

    return parms


def make_knots(numpy_array, num_knots, parms):           
    knot_interval = (np.max(numpy_array)-np.min(numpy_array))/(num_knots+1)
    minval = np.min(numpy_array)+knot_interval
    maxval = np.max(numpy_array)-knot_interval
    knots = np.linspace(minval, maxval, num_knots)
    return knots

def make_property_splines(dmadata, aT, Tind):
    # returns spline fits of the master curves for Estar, phi and aT
    dma_np = dmadata['dma_np']

    # restrict data to the desired temperatures for the spline fit and plots
    dma_master = {}
    dma_master['spline'] = dma_np[Tind['spline'],:,:]
    dma_master['spline'][:,:,0] = dma_master['spline'][:,:,0] * aT[Tind['spline'], None]
    dma_master['plot'] = dma_np[Tind['plot'],:,:]
    dma_master['plot'][:,:,0] = dma_master['plot'][:,:,0] * aT[Tind['plot'], None]
  
    # now reshape to put all the temperature data together
    dma_size = {'spline':dma_master['spline'].shape,
                'plot':dma_master['plot'].shape}

    dma_master['spline'] = np.reshape(dma_master['spline'],
              (dma_size['spline'][0]*dma_size['spline'][1], 5))
    dma_master['plot'] = np.reshape(dma_master['plot'],
              (dma_size['plot'][0]*dma_size['plot'][1], 5))
    
    # now sort according to faT, eliminating any values of the 
    # log(faT) that are equal, as required by LSQUnivariateSpline 
    faT_spline, indices_spline = np.unique(np.log(dma_master['spline'][:,0]),
                                    return_index=True)
    faT_plot = np.unique(dma_master['plot'][:,0])
    dma_master['spline'] = dma_master['spline'][indices_spline]
    
    # extract quantities to determine spline fits
    freq_master_spline = dma_master['spline'][:, 0]
    estar_master_spline = dma_master['spline'][:,1]
    phi_master_spline = dma_master['spline'][:,2]
    
    # make spline fits to property data
    knots = make_knots(np.log(freq_master_spline), 5, {})
    estar_spline = LSQUnivariateSpline(np.log(freq_master_spline), np.log(estar_master_spline), knots)   
    phi_spline = LSQUnivariateSpline(np.log(freq_master_spline), phi_master_spline, knots)
    
    return faT_plot, estar_spline, phi_spline


def Bcompare(sample, Tref):
    qcmsize = 8  # marker size for qcm points
    # read in the parameters from the sample file

    qcmdata = sample['qcmdata']
    dmadata = sample['dmadata']
    sp_parms = sample['sp_parms']
    aTfix = sample['aTfix']
    T_total = sample['T_total']
    Tind = dmadata['Tind']

    # set up the plots
    Bcompfig = plt.figure(sample['title']+'comp', figsize=(8, 3))
    Bcomp_ax1 = Bcompfig.add_subplot(1, 2, 1)
    Bcomp_ax1.loglog(True)
    Bcomp_ax1.set_xlabel('$fa_T$ (s$^{-1}$)')
    Bcomp_ax1.set_ylabel('$|E^*|$ (Pa)')
    Bcomp_ax1.set_ylim(bottom=3e5, top=3e9)

    Bcomp_ax2 = Bcompfig.add_subplot(1, 2, 2)
    Bcomp_ax2.semilogy(True)
    Bcomp_ax2.set_xlabel(r'$T\:(^ \circ$C)')
    Bcomp_ax2.set_ylabel('$a_T$')

    # put the DMA data on the plots
    dmadata1 = {}
    dmadata2 = {}

    faT = np.empty_like(dmadata['f'])
    for Tindex in Tind:
        faT[Tindex, :] = (dmadata['f'][Tindex, :] * dmadata['aT_raw'][Tindex]
                          / aTfix)
        dmadata1[Tindex], = Bcomp_ax1.plot(faT[Tindex, :],
                                           dmadata['estar'][Tindex, :], '+')
        dmadata2[Tindex], = Bcomp_ax2.plot(dmadata['T'][Tindex],
                                           dmadata['aT_raw'][Tindex]/aTfix,
                                           '+', markersize=16)

    # add QCM points to the plots
    qcmdata1, = Bcomp_ax1.plot(1.5e7*qcmdata['qcm_aT']/aTfix, qcmdata['qcm_E'], 'ro',
                               markersize=qcmsize)
    qcmdata2, = Bcomp_ax2.plot(qcmdata['qcm_T'], qcmdata['qcm_aT']/aTfix, 'ro',
                               markersize=qcmsize)

    # add the fits for the orginal values B, Tinf
    faTfit = qcm.springpot_f(sp_parms)
    Estarfit = qcm.springpot(faTfit, sp_parms)
    fitE, = Bcomp_ax1.plot(faTfit, abs(Estarfit), 'b-')
    B1 = sp_parms['B']  # this is the standard, best fit value
    Tinf1 = sp_parms['Tinf']  # standard, best fit value
    Tforplot = np.linspace(min(T_total), max(T_total), 100)
    fitaT1, = Bcomp_ax2.plot(Tforplot, np.exp(qcm.vogel(Tforplot, Tref, B1,
                                                    Tinf1)), '-b')

    # now plot the curves with other values of B
    # first we specify the alternative values we want to use
    B2 = 0.5*B1
    B3 = 2*B1

    # now redefine the vogel functions with fixed values of B
    def vogel2(T, Tref, Tinf):
        return -B2/(Tref-Tinf) + B2/(T-Tinf)

    def vogel3(T, Tref, Tinf):
        return -B3/(Tref-Tinf) + B3/(T-Tinf)

    # use orignal values of Tref and Tinf for initial guess
    vft_guess = np.array([Tref, Tinf1])

    # now we do the curve fits to get the best fit values of Tref and Tinf
    # that are consistent with the DMA data
    [Tref2, Tinf2], pcov = curve_fit(vogel2, dmadata['T'][Tind],
                                     np.log(dmadata['aT'][Tind]), vft_guess)
    [Tref3, Tinf3], pcov = curve_fit(vogel3, dmadata['T'][Tind],
                                     np.log(dmadata['aT'][Tind]), vft_guess)

    # now we add these alternate curves to the aT plot
    fitaT2, = Bcomp_ax2.plot(Tforplot, np.exp(qcm.vogel(Tforplot, Tref2, B2,
                                                    Tinf2)), '--g')
    fitaT3, = Bcomp_ax2.plot(Tforplot, np.exp(qcm.vogel(Tforplot, Tref3, B3,
                                                    Tinf3)), '-.r')

    # now add titles to plots
    Bcomp_ax1.set_title('(a)')
    Bcomp_ax2.set_title('(b)')

    # add the legend to the Estar plot
    Bcomp_ax1.legend([dmadata1[Tind[0]], qcmdata1, fitE],
                     ['DMA', 'QCM', 'fit'])

    # build the legend text for the aT plot
    B1text = ('$B=$' + '{:.0f}'.format(B1) + ', ' + r'$T_{\infty}=$' +
              '{:.0f}'.format(Tinf1)+r'$^{\circ}$C')
    B2text = ('$B=$' + '{:.0f}'.format(B2) + ', ' + r'$T_{\infty}=$' +
              '{:.0f}'.format(Tinf2)+r'$^{\circ}$C')
    B3text = ('$B=$' + '{:.0f}'.format(B3) + ', ' + r'$T_{\infty}=$' +
              '{:.0f}'.format(Tinf3)+r'$^{\circ}$C')

    # now add the legend to the aT plot
    Bcomp_ax2.legend([fitaT2, fitaT1, fitaT3], [B2text, B1text, B3text])

    # clean things up, refresh the plot and write the figure file
    Bcompfig.tight_layout()

    Bcompfig.savefig('../figures/'+sample['title']+'_bcomp.svg')


def Rheodata(sample, Tref):
    # Read in the things we need from the sample dictionary

    dmadata = sample['dmadata']
    sp_parms = sample['sp_parms']
    aTfix = sample['aTfix']
    Tind = dmadata['Tind']

    # set up the plots
    Rheodatafig = plt.figure(sample['title']+'rheo', figsize=(8, 3))
    Rheodata_ax1 = Rheodatafig.add_subplot(1, 2, 1)
    Rheodata_ax1.loglog(True)
    Rheodata_ax1.set_xlabel('$fa_T$ (s$^{-1}$)')
    Rheodata_ax1.set_ylabel('$|E^*|$ (Pa)')
    Rheodata_ax1.set_ylim(bottom=3e5, top=3e9)

    Rheodata_ax2 = Rheodatafig.add_subplot(1, 2, 2)
    Rheodata_ax2.semilogx(True)
    Rheodata_ax2.set_xlabel('$fa_T$ (s$^{-1}$)')
    Rheodata_ax2.set_ylabel(r'$\phi$ (deg.)')

    # put the DMA data on the plots
    dmadata1 = {}
    dmadata2 = {}

    faT = np.empty_like(dmadata['f'])
    for Tindex in Tind:
        faT[Tindex, :] = (dmadata['f'][Tindex, :] * dmadata['aT_raw'][Tindex] /
                          aTfix)
        dmadata1[Tindex], = Rheodata_ax1.plot(faT[Tindex, :],
                                              dmadata['estar'][Tindex, :], '+')
        dmadata2[Tindex], = Rheodata_ax2.plot(faT[Tindex, :],
                                              dmadata['phi'][Tindex, :], '+')

    # add the fits for the orginal values B, Tinf
    faTfit = qcm.springpot_f(sp_parms)
    Estarfit = qcm.springpot(faTfit, sp_parms)
    fitE, = Rheodata_ax1.plot(faTfit, abs(Estarfit), 'b-')
    fitphi, = Rheodata_ax2.plot(faTfit, np.angle(Estarfit, deg=True), 'b-')

    # now add titles to plots
    Rheodata_ax1.set_title('(a)')
    Rheodata_ax2.set_title('(b)')

    # add the legend to the Estar plot
    Rheodata_ax1.legend([dmadata1[Tind[0]], fitE],
                        ['DMA', 'fit'], loc='lower right')

    # add the lines that illustrate the location of fref
    fref = sample['sp_parms']['fref_raw']/aTfix
    slope_low = sample['sp_parms']['phi'][1]/90
    slope_high = sample['sp_parms']['phi'][2]/90
    Eg = sample['sp_parms']['E'][2]
    xdata_high = np.logspace(np.log10(fref)-1, np.log10(fref)+3, 10)
    xdata_low = np.logspace(np.log10(fref)-3, np.log10(fref)+1, 10)
    ydata_low = Eg*(xdata_low/fref)**slope_low
    ydata_high = Eg*(xdata_high/fref)**slope_high

    Rheodata_ax1.loglog(xdata_low, ydata_low, '--b')
    Rheodata_ax1.loglog(xdata_high, ydata_high, '--b')

    # add the annotation
    Rheodata_ax1.text(1e4, 3e8, '($f_{ref}a_T$, $E_g$)')

    Rheodata_ax1.annotate('($f_{ref}a_T$, $E_g$)', xy=(fref, Eg),
                          xytext=(1e4, 3e8), arrowprops=dict(facecolor='black',
                          connectionstyle='arc3', arrowstyle='-|>'),)

    # clean things up, refresh the plot and write the figure file
    Rheodatafig.tight_layout()
    Rheodatafig.savefig('../figures/'+sample['title']+'_Rheodata.svg')


def addVGPplot(sample, samplefits, vgpfig, fignum):
    qcmsize = 8  # marker size for qcm points
    # Read the fits from the appropriate sample file

    dmadata = sample['dmadata']

    # set up the van Gurp-Palmen plot
    # this assumes that DMAplot has been run to update the sample ditionary
    vgpfig = plt.figure('VGP')
    vgp_ax = vgpfig.add_subplot(2, 2, fignum)
    vgp_ax.semilogx(True)
    vgp_ax.set_ylabel(r'$\phi$ (deg.)')
    vgp_ax.set_xlabel('$|E^*|$ (Pa)')
    vgp_ax.set_xlim(left=3e5, right=3e9)

    Tind = dmadata['Tind']

    # put the DMA data on the Van Gurp-Palmen Plot
    dmadata1 = {}

    for Tindex in Tind:
        dmadata1[Tindex], = vgp_ax.plot(dmadata['estar'][Tindex, :],
                                        dmadata['phi'][Tindex, :], '+')

    # read in factors needed for the fit

    # faTfit = qcm.springpot_f(sp_parms)

    # # calc. and plot the magnitude and phase angle of E from the fit function
    # fit_E = np.absolute(qcm.springpot(faTfit, sp_parms))
    # fit_phi = np.angle(qcm.springpot(faTfit, sp_parms), deg=True)
    # fit, = vgp_ax.plot(fit_E, fit_phi, 'b-')

    # read in the QCM data and add them to the plot
    K = 2.4e9  # bulk modulus of rubber from Polymer 35, 2759–2763 (1994).
    qcm_G = sample['qcmdata']['qcm_rhog']/sample['qcmdata']['rho']
    qcm_E = 9*K * qcm_G/(3*K + qcm_G)
    qcm_phi = sample['qcmdata']['qcm_phi']
    qcmdata, = vgp_ax.plot(qcm_E, qcm_phi, 'ro', markersize=qcmsize)
    vgp_ax.set_title(sample['title'])
    vgp_ax.legend([dmadata1[Tind[0]], qcmdata, fit], ['DMA', 'QCM', 'fit'])

    vgpfig.savefig('../figures/vgpfig.svg')


def afmplot(sample, Tref):
    Trange = sample['Trange']  # can reset this if needed
    Tfit = np.linspace(Trange[0], max(sample['qcm_T']), 100)
    f_afm = 2000
    df_afm = pd.read_excel(sample['datapath'] + sample['afmfile'],
                           header=[0, 1])
    afmdata = df_afm.values

    T_afm = afmdata[:, 1].astype(float)
    phi_afm = np.degrees(np.arctan(afmdata[:, 2].astype(float)))
    d0_afm = afmdata[:, 3].astype(float)
    p0_afm = afmdata[:, 4].astype(float)
    dmax_afm = afmdata[:, 5].astype(float)
    R_afm = afmdata[:, 6].astype(float)

    # These are the continuous fits over the full range
    sp_parms = sample['sp_parms']

    logaTfit = qcm.vogel(Tfit, sample['Tref_raw'], sample['B'],
                     sample['Tinf'])
    aTfit = np.exp(logaTfit)
    Estarfit = qcm.springpot(f_afm*aTfit, sp_parms)
    phifit = np.degrees(np.angle(Estarfit))

    # These are just evaluated at the AFM points
    logaT_afm = qcm.vogel(T_afm, sample['Tref_raw'], sample['B'],
                      sample['Tinf'])
    aT_afm = 10**logaT_afm
    Estar_afm = qcm.springpot(f_afm*aT_afm, sp_parms)
    abs_Sstar_afm = p0_afm/d0_afm
    a_afm = 1e9*(3/8)*abs_Sstar_afm/abs(Estar_afm)
    ah_afm = (dmax_afm*R_afm)**0.5  # Hertzian contact radius from max disp.
    absEstar_afm = 1e9*(3/8)*abs_Sstar_afm/ah_afm
    G_off_afm = abs_Sstar_afm*d0_afm**2/(4*np.pi*a_afm**2)

    # Now set up the plots with axes, labels, etc.
    afmfig = plt.figure('AFM1')
    afm_ax1 = afmfig.add_subplot(1, 3, 2)
    afm_ax1.semilogy(True)
    afm_ax1.set_xlabel(r'$T\:^\circ$C')
    afm_ax1.set_ylabel('$|E^*|$ (Pa)')
    afm_ax1.set_autoscale_on(True)

    afm_ax2 = afmfig.add_subplot(1, 3, 1)
    afm_ax2.set_xlabel(r'$T\:^\circ$C')
    afm_ax2.set_ylabel(r'$\phi$ (deg.)')
    afm_ax2.set_autoscale_on(True)

    afm_ax3 = afmfig.add_subplot(1, 3, 3)
    afm_ax3.set_xlabel(r'$T\:^\circ$C')
    afm_ax3.set_ylabel(r'$a$ (nm)')
    afm_ax3.set_autoscaley_on(True)

    # Plot the stuff we need to plot
    afm_ax1.plot(Tfit, abs(Estarfit), 'b-')
    afm_ax1.plot(T_afm, abs(Estar_afm), 'r+')
    afm_ax2.plot(Tfit, phifit, 'b-')
    afm_ax2.plot(T_afm, phi_afm, 'r+')
    afm_ax3.plot(T_afm, a_afm, 'r+')

    # autoscale on y axis doesn't seem to work, so we deal with it here
    afm_ax1.set_ylim(bottom=3e5, top=3e9)
    afm_ax3.set_ylim(bottom=getlimits(a_afm)[0], top=getlimits(a_afm)[1])

    afmfig.tight_layout()
    afmfig.canvas.draw()

    afmfig2 = plt.figure('AFM2')
    # delta/a vs. T
    afm_ax4 = afmfig2.add_subplot(2, 3, 1)
    afm_ax4.set_xlabel(r'$T\:^\circ$C')
    afm_ax4.set_ylabel(r'$\delta/a$')
    afm_ax4.plot(T_afm, d0_afm/a_afm, '+r')
    afm_ax4.set_autoscale_on(True)

    # a/R vs. T
    afm_ax5 = afmfig2.add_subplot(2, 3, 2)
    afm_ax5.set_xlabel(r'$T\:^\circ$C')
    afm_ax5.set_ylabel('$a/Rlegend$')
    afm_ax5.plot(T_afm, a_afm/R_afm, '+r')
    afm_ax5.set_autoscale_on(True)

    # a/ah vs. T
    afm_ax6 = afmfig2.add_subplot(2, 3, 3)
    afm_ax6.set_xlabel(r'$T\:^\circ$C')
    afm_ax6.set_ylabel('$a/a_h$')
    afm_ax6.plot(T_afm, a_afm/ah_afm, '+r')
    afm_ax6.set_autoscale_on(True)

    # calc E (with a = ah) vs. T
    afm_ax7 = afmfig2.add_subplot(2, 3, 4)
    afm_ax7.semilogy(True)
    afm_ax7.set_xlabel(r'$T\:^\circ$C')
    afm_ax7.set_ylabel('$E$ (Pa)')
    afm_ax7.plot(Tfit, abs(Estarfit), 'b-')
    afm_ax7.plot(T_afm, absEstar_afm, '+r')
    afm_ax7.set_ylim(bottom=1e6, top=1e8)
    afm_ax7.set_xlim(left=-15, right=55)
    afm_ax7.set_autoscale_on(True)

    # delta_max/R vs. T
    afm_ax8 = afmfig2.add_subplot(2, 3, 5)
    afm_ax8.set_xlabel(r'$T\:^\circ$C')
    afm_ax8.set_ylabel(r'$\delta_{mafm_ax}/R$')
    afm_ax8.plot(T_afm, dmax_afm/R_afm, '+r')
    afm_ax8.set_autoscale_on(True)

    # debonding G vs. T
    afm_ax9 = afmfig2.add_subplot(2, 3, 6)
    afm_ax9.set_xlabel(r'$T\:^\circ$C')
    afm_ax9.set_ylabel(r'$G_{off}$')
    afm_ax9.plot(T_afm, G_off_afm, '+r')
    afm_ax9.set_autoscale_on(True)

    afmfig2.tight_layout()


def springpot_plot(sample):
    # plots the frequency response of different individual springpots,
    # with each branch in a different color.
    fref = sample['fref_raw']
    phi = sample['phi']
    E = sample['E']
    typ = sample['typ']
    n_br = typ.shape[0]  # number of series branches
    n_sp = typ.sum()  # number of springpot elements
    nf = 5  # number of points at each frequency

    sp = np.empty((nf, n_sp), dtype=np.complex)  # element compliance
    plots = {}
    legendtext = {}
    colors = ['k', 'r', 'g']
    markers = ['s', 'o', '^']

    fmax = 10*fref
    fmin = 0.1*fref*(min(E)/max(E))**(1/(max(phi)/90))
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    for i in np.arange(n_sp):
        sp[:, i] = E[i]*(1j*(f/fref)) ** (phi[i]/90)

    fig = plt.figure('SP')
    ax_sp = fig.add_subplot(1, 1, 1)
    ax_sp.loglog(True)
    ax_sp.set_xlabel('$fa_T$ (s$^{-1}$)')
    ax_sp.set_ylabel('$|E^*|$ (Pa)')
    ax_sp.set_autoscale_on(True)
    ax_sp.set_ylim(bottom=0.3*np.amin(abs(sp)), top=3*np.amax(abs(sp)))
    sp_vec = np.append(0, typ.cumsum())
    legendtext = {}

    for j in np.arange(n_sp):
        legendtext[j] = '_nolegend_'  # turn off legends for now

    for i in np.arange(n_br):
        legendtext[sp_vec[i]] = 'Branch '+str(i+1)
        sp_i = np.arange(sp_vec[i], sp_vec[i+1])
        for j in sp_i:
            plots, = ax_sp.plot(f, abs(sp[:, j]), color=colors[i],
                                linewidth=0.5, linestyle='--',
                                marker=markers[i], label=legendtext[j])

    # Now plot the full springpot model fit
    f = np.logspace(np.log10(fmin), np.log10(fmax), 100)
    sp_parms = {}
    sp_parms['fref'] = sample['fref_raw']
    sp_parms['phi'] = sample['phi']
    sp_parms['E'] = sample['E']
    sp_parms['typ'] = sample['typ']
    ax_sp.plot(f, abs(qcm.springpot(f, sp_parms)), '-b', linewidth=2,
               label='total')

    plt.legend(ncol=4)

    return fig


def getlimits(f):
    # used to set plot limits for a numpy array
    flim = [min(f)-0.1*(max(f)-min(f)), max(f)+0.1*(max(f)-min(f))]
    return flim
