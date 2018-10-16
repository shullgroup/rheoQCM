#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:19:59 2018

@author: ken
"""
import numpy as np
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:01:39 2018

@author: ken
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit
import hdf5storage
from scipy.interpolate import LSQUnivariateSpline
import pandas as pd
import pdb


def DMAread(sample):
    # this reads the dMA data and puts it into the dmadata dictionary so we
    # can more easilty plot it later
    dmadata = sample['dmadata']

    # read in the DMA data from the orignal data file
    dma__df = pd.read_csv(dmadata['datapath'] + dmadata['filename'] +
                          '/mech_data.txt',  header=[0, 1], sep='\t',
                          encoding="ISO-8859-1")
    dma_np = np.array(dma__df)[:, 0:4].astype(float)

    # column 0:  frequency
    # column 1  E prime in Pa
    # column 2  E double prime in Pa
    # column 3  tangent of phi

    # Add calculated values
    dma_np = np.append(dma_np, np.zeros((dma_np.shape[0], 3)), axis=1)

    dma_np[:, 4] = np.rad2deg(np.arctan(dma_np[:, 3]))
    dma_np[:, 5] = (dma_np[:, 1] ** 2 + dma_np[:, 2] ** 2) ** 0.5

    # column 4: phi in degrees
    # column 5 is magnitude of E
    # column 6 is going to be faT
    input_Tfile = (dmadata['datapath'] + dmadata['filename'] +
                   '/temp_data.txt')

    # now read in the file that contains temperatures and shift factors
    # column 0 is T (C)
    # column 1 is aT (we don't care much about the other columns
    T_df = pd.read_csv(input_Tfile, header=[0, 1],
                       sep='\t', encoding="ISO-8859-1")
    T_np = np.array(T_df)
    T_np = T_np[:, 0:3].astype(float)
    Tnum = T_np.shape[0]  # the number of temperatures
    fnum = int(dma_np.shape[0]/Tnum)  # number of frequencies at each temp

    dma_np = np.reshape(dma_np, (Tnum, fnum, 7))

    # put the data into a dictionary that we can more easily deal with later
    dmadata['T'] = T_np[:, 0]
    dmadata['aT_raw'] = T_np[:, 1]
    dmadata['f'] = dma_np[:, :, 0]
    dmadata['phi'] = dma_np[:, :, 4]
    dmadata['estar'] = dma_np[:, :, 5]

    # determine temperature indices where temp. is in the desired range
    Trange = dmadata['Trange']
    dmadata['Tind'] = np.where((dmadata['T'] >= Trange[0]) & (dmadata['T'] <=
                               Trange[1]))[0]

    sample['dmadata'] = dmadata

    return sample


def DMAread_sturdy(datadir, file, parms):
    # a read file for Laruen's MAT files
    sample = {}
    data_input = hdf5storage.loadmat(datadir + file)
    # not surewhy dtype needs to be specified here, but it seems to be 
    # necessary at least for estor
    freq = np.array(data_input['rawdata'][2,0][:,0], dtype=np.float64)
    estor = np.array(data_input['rawdata'][2,1][:,0], dtype=np.float64)
    eloss = np.array(data_input['rawdata'][2,2][:,0], dtype=np.float64)
    T = np.array(data_input['rawdata'][2,4][:,0]) # these are Temps for each freq. point
    estar =(estor**2+eloss**2)**0.5
    phi = np.degrees(np.arctan(eloss/estor))
    ftotal = freq.shape[0]
    Tnum = data_input['T'].shape[1]  # the number of temperatures
    fnum = ftotal/Tnum  # number of frequencies at each temp
    if fnum != np.fix(fnum):
        print('non-integral fnum')
    else:
        fnum = int(fnum)
    dma_np = np.zeros([ftotal, 6])
    dma_np[:, 0] = freq
    dma_np[:, 1] = estar
    dma_np[:, 2] = phi
    dma_np[:, 3] = T
    dma_np[:, 4] = estor
    dma_np[:, 5] = eloss
    
    dma_np = np.reshape(dma_np, (Tnum, fnum, 6))
    dmadata = {}
    dmadata['f'] = dma_np[:, :, 0]
    dmadata['estar'] = dma_np[:, :, 1]
    dmadata['phi'] = dma_np[:, :, 2]    
    dmadata['aT_raw'] = np.squeeze(data_input['shift_factors'])
    dmadata['T'] = np.squeeze(data_input['T'])
    dmadata['estor'] = dma_np[:, :, 4]
    dmadata['eloss'] = dma_np[:, :, 5]
    dmadata['dma_np'] = dma_np
    
    # determine temperature indices where temp. is in the desired range
    sample['dmadata'] = dmadata
    return sample



def DMAcalc(sampledata, parms, Tref):
    # we stick with the standard
    qcmsize = 8  # marker size for qcm points
    # calculates shift factors - generally only need to run this once

    # Read the fits from the appropriate sample file
    dmadata = sampledata['dmadata']
    qcmdata = sampledata.get('qcmdata',{})
    sp_parms = sampledata.get('sp_parms',{})

    # now we add the QCM points and do the fits
    K = 2.4e9  # bulk modulus of rubber from Polymer 35, 2759–2763 (1994).
    # now we read in the data

    qcm_G = np.array(qcmdata.get('qcm_rhog',[]))/np.array(qcmdata.get('rho', []))
    qcm_E = 9*K * qcm_G/(3*K + qcm_G)
    qcm_phi = np.array(qcmdata.get('qcm_phi', []))
    qcm_T = np.array(qcmdata.get('qcm_T', []))
    nqcm_aT = qcm_E.shape[0]
    qcm_aT = np.zeros(nqcm_aT)

#    # %%  this is what we need to do to get the spline fit to work: taken
#    # from https://scicomp.stackexchange.com/questions/19693/is-my-restricted-
#    # natural-cubic-spline-equation-wrong
#
#    n = 100
#    x = linspace(0, 1, n)
#    y = (((exp(1.2*x) + 1.5*sin(7*x))-1)/3) + normal(0, 0.15, size=n)
#    t = [0.2, 0.4, 0.6, 0.8]
#    spl = LSQUnivariateSpline(x, y, t)
#    plt.plot(x, spl(x), '-', x, y, 'o')

    # temporarily change fref to fref_raw, since we are fitting to the original
    # raw data
    sp_parms['fref'] = sp_parms['fref_raw']
    for i in np.arange(nqcm_aT):
        def ftosolve(f):
            return (abs(springpot(np.array([f]), sampledata['sp_parms']))
                    - qcm_E[i])
        f_guess = 0.01*sp_parms['fref_raw']  # usually works as a guess
        # take abs value in case fsolve returns the negative of the freq.
        qcm_aT[i] = np.abs(fsolve(ftosolve, f_guess)/1.5e7)

    vft_guess = np.array([sp_parms['Tref_raw'], sp_parms['B'],
                          sp_parms['Tinf']])
    Tind = dmadata['Tind']
    T_total = np.append(dmadata['T'][Tind], qcmdata['qcm_T'])
    aT_total = np.append(dmadata['aT_raw'][Tind], qcm_aT)

    [Tref_raw, B, Tinf], pcov = curve_fit(vogel, T_total,
                                          np.log(aT_total), vft_guess)

    # now we adjust the shift factors based on ref. temp
    aTfix = np.exp(vogel(Tref, Tref_raw, B, Tinf))
    sampledata['aTfix'] = aTfix
    dmadata['aT'] = dmadata['aT_raw']/aTfix
    sp_parms['fref'] = sp_parms['fref_raw']/aTfix

    # Calculate Tg as Temp where fref = 1 Hz.
    def fvogel(T):
        return vogel(T, Tref_raw, B, Tinf)-np.log(sp_parms['fref_raw'])
    Tg_guess = Tinf+50
    sp_parms['Tg'] = fsolve(fvogel, Tg_guess)

    # update the sample dictionary
    dmadata['Tref_raw'] = Tref_raw
    dmadata['Tind'] = Tind

    sp_parms['B'] = B
    sp_parms['Tinf'] = Tinf

    qcmdata['qcm_aT'] = qcm_aT
    qcmdata['qcm_E'] = qcm_E

    sampledata['T_total'] = T_total
    sampledata['aT_total'] = aT_total
    sampledata['qcmdata'] = qcmdata
    sampledata['dmadata'] = dmadata
    sampledata['sp_parms'] = sp_parms
    parms['sp_parms'] = sp_parms

    # make the plot
    if parms.get('calc_plot', 'yes') == 'yes':
        figinfo = DMAplot(sampledata, parms, Tref)

        dma_ax1 = figinfo['dma_ax1']
        dma_ax2 = figinfo['dma_ax2']
        dma_ax3 = figinfo['dma_ax3']
        dmafig = figinfo['dmafig']
        dmadata1 = figinfo['dmadata1']
        fit3 = figinfo['fit3']

        # add QCM points to the plots
        qcm_aT = qcm_aT/aTfix
        qcmdata1, = dma_ax1.plot(1.5e7*qcm_aT, qcm_E, 'ro', markersize=qcmsize)
        qcmdata2, = dma_ax2.plot(1.5e7*qcm_aT, qcm_phi, 'ro',
                                 markersize=qcmsize)
        qcmdata3, = dma_ax3.plot(qcm_T, qcm_aT, 'ro', markersize=qcmsize)

        dma_ax3.legend([dmadata1[Tind[0]], qcmdata3, fit3],
                       ['DMA', 'QCM', 'fit'])

        dmafig.tight_layout() 
        dmafig.savefig('../figures/'+sampledata['title']+'.svg')

    # print the parameters we care about to the screen
    print('')
    print('sample=', sampledata['title'])
    print('Tg=', sp_parms['Tg'])
    print('B=', B)
    print('Tinf=', Tinf)
    print('Tref=', Tref)
    print('fref=', sp_parms['fref'])
    print('E=', sp_parms['E'])
    print('phi=', sp_parms['phi'])

    return sampledata


def DMAplot(sampledata, parms, Tref):
    make_spline = parms.get('make_spline', 'no')
    figinfo = parms.get('figinfo',{})
    Trange = parms.get('Trange', [-200, 300])  # default range designed to include everything
    sp_parms = parms.get('sp_parms', {})
    dmadata = sampledata['dmadata']
    if 'mark' not in parms:
        parms['mark'] = '+'
    
    # read in the shift factors and the corresponding temperatures
    Tind =     dmadata['Tind'] = np.where((dmadata['T'] >= Trange[0]) & (dmadata['T'] <= Trange[1]))[0]
    T = dmadata['T']
    aT = dmadata['aT_raw']
    
    # now adjust the shift factors to the reference temperature
    knots = make_knots(T[Tind], 3, {})
    aT_spline = LSQUnivariateSpline(T[Tind], np.log(aT[Tind]), knots)   
    aTfix = np.exp(aT_spline(Tref))
    aT = aT/aTfix
    
    # determine which shift factors to use
    if parms.get('aTfit', 'self') != 'self':
        aT = np.exp(vogel(dmadata['T'], Tref, sp_parms['B'],
                         sp_parms['Tinf']))

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
        dma_ax3.set_xlabel(r'$T\:(^ \circ$C)')
        dma_ax3.set_ylabel('$a_T$')
    else:
        dmafig = figinfo['dmafig']
        dma_ax1 = figinfo['dma_ax1']
        dma_ax2 = figinfo['dma_ax2']
        dma_ax3 = figinfo['dma_ax3']

    # set up the van Gurp-Palmen plot
    if not plt.fignum_exists('vgp'):
        vgpfig = plt.figure('vgp', figsize=(3,3))
        vgp_ax = vgpfig.add_subplot(111)
        vgp_ax.semilogx(True)
        vgp_ax.set_ylabel(r'$\phi$ (deg.)')
        vgp_ax.set_xlabel('$|E^*|$ (Pa)')
    else:
        dmafig = parms['figinfo']['dmafig']
        vgpfig = parms['figinfo']['vgpfig']
        vgp_ax = parms['figinfo']['vgp_ax']

    # put the DMA data on the Van Gurp-Palmen Plot
    dmadata1 = {}


    # reset the color cycling
    dma_ax1.set_prop_cycle(None)
    dma_ax2.set_prop_cycle(None)
    dma_ax3.set_prop_cycle(None)
    vgp_ax.set_prop_cycle(None)

    # put the DMA data on the plots
    dmadata1 = {}
    dmadata2 = {}
    dmadata3 = {}
    dmadata4 = {}

    # take the tangent of phi if that is desired
    if parms.get('tandelta', 'no') == 'yes':
        phidata_for_plot = np.tan(np.radians(dmadata['phi']))
    else:
        phidata_for_plot = dmadata['phi']
        
    faT = np.empty_like(dmadata['f'])
    for Tindex in Tind:
        faT[Tindex, :] = dmadata['f'][Tindex, :] * aT[Tindex]
        dmadata1[Tindex], = (dma_ax1.plot(faT[Tindex, :],
                             dmadata['estar'][Tindex, :], marker=parms['mark'],
                             linestyle='none'))
        dmadata2[Tindex], = (dma_ax2.plot(faT[Tindex, :],
                             phidata_for_plot[Tindex, :], marker=parms['mark'],
                             linestyle='none'))
        dmadata3[Tindex], = dma_ax3.plot(T[Tindex],
                                         aT[Tindex], '+', markersize=12)
        
#        dmadata4[Tindex], = vgp_ax.plot(dmadata['estar'][Tindex, :],
#                                        dmadata['phi'][Tindex, :], '+')
               
    # make necessary adjustments to axis limits
    vgp_ax.set_ylim(bottom=0)
    dma_ax2.set_ylim(bottom=0)
    dma_ax1.set_ylim(bottom=1e6, top=5e9)
    
    # now we create the springpot fits and add them to the plot
    if parms.get('show_springpot_fit', 'no') == 'yes':
        faTfit = springpot_f(sp_parms)
        Estar = springpot(faTfit, sp_parms)
        fit1, = dma_ax1.loglog(faTfit, abs(Estar), 'b-')
        fit2, = dma_ax2.semilogx(faTfit, np.angle(Estar, deg=True), 'b-')


    # generate VFT fit if the parameters exist for it
    if 'B' in parms: 
        fit3, = dma_ax3.plot(T[Tind], np.exp(vogel(T[Tind], Tref, sp_parms.get('B', 1000),
                                      sp_parms.get('Tinf', -50), '-b')))

    # now add titles to plots
    if parms.get('add_titles', 'no') == 'yes':
        dma_ax1.set_title(sampledata['title'])
        dma_ax2.set_title('$T_{ref}=$'+str(Tref) + r'$^{\circ}$' +
                          'C \n $f_{ref}=$' +
                          '{:.1e}'.format(sp_parms['fref']) + ' s$^{-1}$')
        dma_ax3.set_title('$B=${:.0f}'.format(sp_parms['B']) + '\n' +
                          r'$T_{\infty}=$' +
                          '{:.0f}'.format(sp_parms['Tinf']) + r'$^{\circ}$C')
        
    # organize the data so we can make a master plot
                  
    dma_master = dmadata['dma_np']
    # adjust frequency by the shift factors
    dma_master[:,:,0] = dma_master[:,:,0] * aT[:, None]
    # restrict data to the temperaures in Tind
    dma_master = dma_master[Tind,:,:]
    # now reshape to put all the temperature data together
    dma_size = dma_master.shape
    dma_master = np.reshape(dma_master, (dma_size[0]*dma_size[1], 6))
    # now sort according to faT
    dma_master = dma_master[dma_master[:, 0].argsort()]
        
    freq_master = dma_master[:, 0]
    estar_master = dma_master[:,1]
    phi_master = dma_master[:,2]
    T_master = dma_master[:,3]
    estor_master = dma_master[:,4]
    eloss_master = dma_master[:,5]
        
    # add spline fit of shift factors
    dma_ax3.plot(T[Tind], np.exp(aT_spline(T[Tind]))/aTfix, 'b-')    
    dmafig.tight_layout()
    vgpfig.tight_layout()
    
    # make spline fits to property data
    knots = make_knots(np.log(freq_master), 3, {})
    estar_spline = LSQUnivariateSpline(np.log(freq_master), np.log(estar_master), knots)   
    phi_spline = LSQUnivariateSpline(np.log(freq_master), phi_master, knots)   
    
    # add master splines
    dma_ax1.loglog(freq_master, np.exp(estar_spline(np.log(freq_master))), 'g-')
    dma_ax2.semilogx(freq_master, phi_spline(np.log(freq_master)), 'g-')
    vgp_ax.semilogx(np.exp(estar_spline(np.log(freq_master))), phi_spline(np.log(freq_master)), 'g-')
    figinfo = {}

    figinfo['dmafig'] = dmafig
    figinfo['dma_ax1'] = dma_ax1
    figinfo['dma_ax2'] = dma_ax2
    figinfo['dma_ax3'] = dma_ax3
    figinfo['vgpfig'] = vgpfig
    figinfo['vgp_ax'] = vgp_ax

    return figinfo

phi_spline(np.log(freq_master))
def make_knots(numpy_array, num_knots, parms):                  
    knot_interval = (np.max(numpy_array)-np.min(numpy_array))/(num_knots+1)
    minval = np.min(numpy_array)+knot_interval
    maxval = np.max(numpy_array)-knot_interval
    knots = np.linspace(minval, maxval, num_knots)
    return knots

phi_spline(np.log(freq_master))

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
    faTfit = springpot_f(sp_parms)
    Estarfit = springpot(faTfit, sp_parms)
    fitE, = Bcomp_ax1.plot(faTfit, abs(Estarfit), 'b-')
    B1 = sp_parms['B']  # this is the standard, best fit value
    Tinf1 = sp_parms['Tinf']  # standard, best fit value
    Tforplot = np.linspace(min(T_total), max(T_total), 100)
    fitaT1, = Bcomp_ax2.plot(Tforplot, np.exp(vogel(Tforplot, Tref, B1,
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
    fitaT2, = Bcomp_ax2.plot(Tforplot, np.exp(vogel(Tforplot, Tref2, B2,
                                                    Tinf2)), '--g')
    fitaT3, = Bcomp_ax2.plot(Tforplot, np.exp(vogel(Tforplot, Tref3, B3,
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
    faTfit = springpot_f(sp_parms)
    Estarfit = springpot(faTfit, sp_parms)
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


def addVGPplot(sampledata, samplefits, vgpfig, fignum):
    qcmsize = 8  # marker size for qcm points
    # Read the fits from the appropriate sample file

    sp_parms = samplefits['sp_parms']
    dmadata = sampledata['dmadata']

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

    faTfit = springpot_f(sp_parms)

    # calc. and plot the magnitude and phase angle of E from the fit function
    fit_E = np.absolute(springpot(faTfit, sp_parms))
    fit_phi = np.angle(springpot(faTfit, sp_parms), deg=True)
    fit, = vgp_ax.plot(fit_E, fit_phi, 'b-')

    # read in the QCM data and add them to the plot
    K = 2.4e9  # bulk modulus of rubber from Polymer 35, 2759–2763 (1994).
    qcm_G = sampledata['qcmdata']['qcm_rhog']/sampledata['qcmdata']['rho']
    qcm_E = 9*K * qcm_G/(3*K + qcm_G)
    qcm_phi = sampledata['qcmdata']['qcm_phi']
    qcmdata, = vgp_ax.plot(qcm_E, qcm_phi, 'ro', markersize=qcmsize)
    vgp_ax.set_title(sampledata['title'])
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

    logaTfit = vogel(Tfit, sample['Tref_raw'], sample['B'],
                     sample['Tinf'])
    aTfit = np.exp(logaTfit)
    Estarfit = springpot(f_afm*aTfit, sp_parms)
    phifit = np.degrees(np.angle(Estarfit))

    # These are just evaluated at the AFM points
    logaT_afm = vogel(T_afm, sample['Tref_raw'], sample['B'],
                      sample['Tinf'])
    aT_afm = 10**logaT_afm
    Estar_afm = springpot(f_afm*aT_afm, sp_parms)
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
    ax_sp.plot(f, abs(springpot(f, sp_parms)), '-b', linewidth=2,
               label='total')

    plt.legend(ncol=4)

    return fig


def springpot(f, sp_parms):
    # this function supports a combination of different springpot elments
    # combined in series, and then in parallel.  For example, if typ is
    # [1,2,3],  there are three branches
    # in parallel with one another:  the first one is element 1, the
    # second one is a series comination of eleents 2 and 3, and the third
    # one is a series combination of 4, 5 and 6.
    fref = sp_parms['fref']
    phi = sp_parms['phi']
    E = sp_parms['E']
    typ = sp_parms['typ']
    nf = f.shape[0]  # number of frequencies
    n_br = typ.shape[0]  # number of series branches
    n_sp = typ.sum()  # number of springpot elements
    sp_comp = np.empty((nf, n_sp), dtype=np.complex)  # element compliance
    br_E = np.empty((nf, n_br), dtype=np.complex)  # branch stiffness

    for i in np.arange(n_sp):
        sp_comp[:, i] = 1/(E[i]*(1j*(f/fref)) ** (phi[i]/90))

    sp_vec = np.append(0, typ.cumsum())
    for i in np.arange(n_br):
        sp_i = np.arange(sp_vec[i], sp_vec[i+1])
        br_E[:, i] = 1/sp_comp[:, sp_i].sum(1)

    return br_E.sum(1)


def springpot_f(sp_parms):
    # this function returns a sensible array of frequencies to use to plot
    # the springpot fit
    fref = sp_parms['fref']
    E = sp_parms['E']
    phi = sp_parms['phi']
    fmax = 1e3*fref
    fmin = 1e-3*fref*(min(E)/max(E))**(1/(max(phi)/90))
    faTfit = np.logspace(np.log10(fmin), np.log10(fmax), 100)
    return faTfit


def vogel(T, Tref, B, Tinf):
    logaT = -B/(Tref-Tinf) + B/(T-Tinf)
    return logaT


def getlimits(f):
    # used to set plot limits for a numpy array
    flim = [min(f)-0.1*(max(f)-min(f)), max(f)+0.1*(max(f)-min(f))]
    return flim
