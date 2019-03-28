#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:19:59 2018

@author: ken
"""

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import os
import hdf5storage
from pathlib import Path
import pandas as pd
import pdb

# for reading h5 data file
try:
    from DataSaver_copy import DataSaver
except:
    from QCMFuncs.DataSaver_copy import DataSaver

data_saver = DataSaver()


Zq = 8.84e6  # shear acoustic impedance of at cut quartz
f1 = 5e6  # fundamental resonant frequency
openplots = 4
drho_q = Zq/(2*f1)
e26 = 9.65e-2
electrode_default = {'drho':2.8e-3, 'grho3':3.0e14, 'phi':0}


def close_on_click(event):
    # used so plots close in response to a some event
    global openplots
    plt.close()
    openplots = openplots - 1
    return


def find_dataroot(owner):
    # returns root data directory as first potential option from an input list
    # if none of the possibilities exist, we return 'none'
    if owner == 'schmitt':
        dataroots = ['/home/ken/k-shull@u.northwestern.edu/'+
                     'Group_Members/Research-Schmitt/data/Schmitt/',
                     r'/Volumes/GoogleDrive/My Drive/Research-Schmitt/'+
                     r'data/Schmitt']
    elif owner == 'qifeng':
        dataroots = ['/home/ken/k-shull@u.northwestern.edu/Group_Members/'+
                     'Research-Wang/CHiMaD/QCM_sample/data/',
                     r'C:\Users\ShullGroup\Documents\User Data\WQF\GoogleDriveSync'+
                     r'\Research-Wang\CHiMaD\QCM_sample\data']
    elif owner == 'taghon':
        dataroots = ['/home/ken/k-shull@u.northwestern.edu/Group_Members/'+
                     'Research-Taghon/QCM/merefiles/data/']
    elif owner == 'depolo':
        dataroots = ['/home/ken/k-shull@u.northwestern.edu/'+
                     r'Group_Members/Research-Depolo/data/',
                     r'C:\Users\Gwen dePolo\gwendepolo2023@u.northwestern.edu\Research-Depolo\data']
    elif owner == 'sturdy':
        dataroots = ['/home/ken/Mydocs/People/Sturdy/Filled_Galkyd_Paper/data/QCM/']
    else:
        dataroots = [os.getcwd()] # current folder

    for directory in dataroots:
        if os.path.exists(directory):
            return directory
    print('cannot find root data directory')
    return 'none'


def fstar_err_calc(fstar):
    # calculate the error in delfstar
    g_err_min = 10 # error floor for gamma
    f_err_min = 50 # error floor for fd
    err_frac = 3e-2 # error in f or gamma as a fraction of gamma
    # start by specifying the error input parameters
    fstar_err = np. zeros(1, dtype=np.complex128)
    fstar_err = (f_err_min + err_frac*fstar.imag + 1j*
                 (g_err_min + err_frac*fstar.imag))
    return fstar_err


def sauerbreyf(n, drho):
    return 2*n*f1 ** 2*drho/Zq


def sauerbreym(n, delf):
    return delf*Zq/(2*n*f1 ** 2)


def grho(n, material):
    grho3 = material['grho3']
    phi = material['phi']
    return grho3*(n/3) ** (phi/90)


def grho_from_dlam(n, drho, dlam, phi):
    return (drho*n*f1*np.cos(np.deg2rad(phi/2))/dlam) ** 2


def grho3_bulk(delfstar):
    return (np.pi*Zq*abs(delfstar[3])/f1) ** 2


def phi_bulk(n, delfstar):
    return -np.degrees(2*np.arctan(np.real(delfstar[n]) /
                       np.imag(delfstar[n])))


def calc_D(n, material, delfstar):
    drho = material['drho']
    # set switch to handle ase where drho = 0
    if drho == 0:
        return 0
    else:
        return 2*np.pi*(n*f1+delfstar)*drho/zstar_bulk(n, material)

def zstar_bulk(n, material):
    grho3 = material['grho3']
    grho = grho3*(n/3)**(material['phi']/90)  #check for error here
    grhostar = grho*np.exp(1j*np.pi*material['phi']/180)
    return grhostar ** 0.5

def calc_delfstar_sla(ZL):
    return f1*1j/(np.pi*Zq)*ZL

def calc_ZL(n, layers, delfstar):
    # layers is a dictionary of dictionaries
    # each dictionary is named according to the layer number
    # layer 1 is closest to the quartz

    N = len(layers)
    Z = {}; D = {}; L = {}; S = {}

    # we use the matrix formalism to avoid typos.
    for i in np.arange(1, N):
        Z[i] = zstar_bulk(n, layers[i])
        D[i] = calc_D(n, layers[i], delfstar)
        L[i] = np.array([[np.cos(D[i])+1j*np.sin(D[i]), 0],
                 [0, np.cos(D[i])-1j*np.sin(D[i])]])

    # get the terminal matrix from the properties of the last layer
    D[N] = calc_D(n, layers[N], delfstar)
    Zf_N = 1j*zstar_bulk(n, layers[N])*np.tan(D[N])

    # if there is only one layer, we're already done
    if N == 1:
        return Zf_N

    Tn = np.array([[1+Zf_N/Z[N-1], 0],
          [0, 1-Zf_N/Z[N-1]]])

    uvec = L[N-1]@Tn@np.array([[1.], [1.]])

    for i in np.arange(N-2, 0, -1):
        S[i] = np.array([[1+Z[i+1]/Z[i], 1-Z[i+1]/Z[i]],
          [1-Z[i+1]/Z[i], 1+Z[i+1]/Z[i]]])
        uvec = L[i]@S[i]@uvec

    rstar = uvec[1,0]/uvec[0,0]
    return Z[1]*(1-rstar)/(1+rstar)



def calc_delfstar(n, layers, calctype):
    if not layers: # layers is empty {}
        return np.nan

    # there is data
    if 'overlayer' in layers:
        ZL = calc_ZL(n, {1:layers['film'], 2:layers['overlayer']}, 0)
        ZL_ref = calc_ZL(n, {1:layers['overlayer']}, 0)
        del_ZL = ZL-ZL_ref
    else:
        del_ZL = calc_ZL(n, {1:layers['film']}, 0)

    if calctype != 'LL':
        # use the small load approximation in all cases where calctype
        # is not explicitly set to 'LL'
        return calc_delfstar_sla(del_ZL)

    else:
        # this is the most general calculation
        # use defaut electrode if it's not specified
        if 'electrode' not in layers:
            layers['electrode'] = electrode_default

        layers_all = {1:layers['electrode'], 2:layers['film']}
        layers_ref = {1:layers['electrode']}
        if 'overlayer' in layers:
            layers_all[3]=layers['overlayer']
            layers_ref[2] = layers['overlayer']

        ZL_all = calc_ZL(n, layers_all, 0)
        delfstar_sla_all = calc_delfstar_sla(ZL_all)
        ZL_ref = calc_ZL(n, layers_ref, 0)
        delfstar_sla_ref = calc_delfstar_sla(ZL_ref)


        def solve_Zmot(x):
            delfstar = x[0] + 1j*x[1]
            Zmot = calc_Zmot(n,  layers_all, delfstar)
            return [Zmot.real, Zmot.imag]

        sol = optimize.root(solve_Zmot, [delfstar_sla_all.real,
                                         delfstar_sla_all.imag])
        dfc = sol.x[0] + 1j* sol.x[1]

        def solve_Zmot_ref(x):
            delfstar = x[0] + 1j*x[1]
            Zmot = calc_Zmot(n,  layers_ref, delfstar)
            return [Zmot.real, Zmot.imag]

        sol = optimize.root(solve_Zmot_ref, [delfstar_sla_ref.real,
                                             delfstar_sla_ref.imag])
        dfc_ref = sol.x[0] + 1j* sol.x[1]

        return dfc-dfc_ref



def calc_Zmot(n, layers, delfstar):
    om = 2 * np.pi *(n*f1 + delfstar)
    g0 = 10 # Half bandwidth of unloaed resonator (intrinsic dissipation on crystalline quartz)
    Zqc = Zq * (1 + 1j*2*g0/(n*f1))
    dq = 330e-6  # only needed for piezoelectric stiffening calc.
    epsq = 4.54; eps0 = 8.8e-12; C0byA = epsq * eps0 / dq; ZC0byA = C0byA / (1j*om)
    ZPE = -(e26/dq)**2*ZC0byA  # ZPE accounts for oiezoelectric stiffening anc
    # can always be neglected as far as I can tell

    Dq = om*drho_q/Zq
    secterm = -1j*Zqc/np.sin(Dq)
    ZL = calc_ZL(n, layers, delfstar)
    # eq. 4.5.9 in book
    thirdterm = ((1j*Zqc*np.tan(Dq/2))**-1 + (1j*Zqc*np.tan(Dq/2) + ZL)**-1)**-1
    Zmot = secterm + thirdterm  +ZPE

    return Zmot



def calc_dlam(n, film):
    return calc_D(n, film, 0).real/(2*np.pi)


def dlam(n, dlam3, phi):
    return dlam3*(int(n)/3) ** (1-phi/180)


def normdelfstar(n, dlam3, phi):
    return -np.tan(2*np.pi*dlam(n, dlam3, phi)*(1-1j*np.tan(np.deg2rad(phi/2)))) / \
        (2*np.pi*dlam(n, dlam3, phi)*(1-1j*np.tan(np.deg2rad(phi/2))))


def rhcalc(nh, dlam3, phi):
    return normdelfstar(nh[0], dlam3, phi).real / \
        normdelfstar(nh[1], dlam3, phi).real


def rh_from_delfstar(nh, delfstar):
    # nh here is the calc string (i.e., '353')
    n1 = int(nh[0])
    n2 = int(nh[1])
    return (n2/n1)*delfstar[n1].real/delfstar[n2].real


def rdcalc(nh, dlam3, phi):
    return -(normdelfstar(nh[2], dlam3, phi).imag / \
        normdelfstar(nh[2], dlam3, phi).real)


def rd_from_delfstar(n, delfstar):
    # dissipation ratio calculated for the relevant harmonic
    return -delfstar[n].imag/delfstar[n].real


def solve_for_props(soln_input):
    # solve the QCM equations to determine the properties
    layers = soln_input['layers']
    nhplot = soln_input.get('nhplot', [1, 3, 5])
    calctype = soln_input.get('calctype', 'SLA')
    nh = soln_input['nh']
    n1 = int(nh[0])
    n2 = int(nh[1])
    n3 = int(nh[2])
    delfstar = soln_input['delfstar']

    # first pass at solution comes from rh and rd
    rd_exp = -delfstar[n3].imag/delfstar[n3].real
    rh_exp = (n2/n1)*delfstar[n1].real/delfstar[n2].real

    if 'prop_guess' in soln_input:
        soln1_guess = guess_from_props(soln_input['propguess'])
    elif rd_exp > 0.5:
        soln1_guess = bulk_guess(delfstar)
    else:
        soln1_guess = thinfilm_guess(delfstar)

    lb = np.array([0, 0])  # lower bounds on dlam3 and phi
    ub = np.array([5, 90])  # upper bonds on dlam3 and phi

    # we solve the problem initially using the harmonic and dissipation
    # ratios, using the small load approximation
    def ftosolve(x):
        return [rhcalc(nh, x[0], x[1])-rh_exp, rdcalc(nh, x[0], x[1])-rd_exp]

    soln1 = optimize.least_squares(ftosolve, soln1_guess, bounds=(lb, ub))

    dlam3 = soln1['x'][0]
    phi = soln1['x'][1]
    drho = (sauerbreym(n1, delfstar[n1].real) /
            normdelfstar(n1, dlam3, phi).real)
    grho3 = grho_from_dlam(3, drho, dlam3, phi)

    # we solve it again to get the Jacobian with respect to our actual
    # input variables - this is helpfulf for the error analysis
    x0 = np.array([drho, grho3, phi])

    lb = np.array([0, 1e7, 0])  # lower bounds drho, grho3, phi
    ub = np.array([1e-2, 1e13, 90])  # upper bounds drho, grho3, phi

    # now solve a second time in order to get the proper jacobian for the
    # error calculation, using either the SLA or LL methods

    def ftosolve2(x):
        layers['film'] = {'drho':x[0], 'grho3':x[1], 'phi':x[2]}
        return ([delfstar[n1].real-calc_delfstar(n1, layers, calctype).real,
                 delfstar[n2].real-calc_delfstar(n2, layers, calctype).real,
                 delfstar[n3].imag-calc_delfstar(n3, layers, calctype).imag])

    # put the input uncertainties into a 3 element vector
    delfstar_err = np.zeros(3)

    delfstar_err[0] = fstar_err_calc(delfstar[n1]).real
    delfstar_err[1] = fstar_err_calc(delfstar[n2]).real
    delfstar_err[2] = fstar_err_calc(delfstar[n3]).imag

    # initialize the output uncertainties
    err = {}
    err_names = ['drho', 'grho3', 'phi']
    # recalculate solution to give the uncertainty, if solution is viable
    if np.all(lb < x0) and np.all(x0 < ub):
        soln2 = optimize.least_squares(ftosolve2, x0, bounds=(lb, ub))
        drho = soln2['x'][0]
        grho3 = soln2['x'][1]
        phi = soln2['x'][2]
        film = {'drho':drho, 'grho3':grho3, 'phi':phi}
        dlam3 = calc_dlam(3, film)
        jac = soln2['jac']
        jac_inv = np.linalg.inv(jac)

        # define sensibly names partial derivatives for further use
        deriv = {}
        for k in [0, 1, 2]:
            deriv[err_names[k]]={0:jac_inv[k, 0], 1:jac_inv[k, 1], 2:jac_inv[k, 2]}
            err[err_names[k]] = ((jac_inv[k, 0]*delfstar_err[0])**2 +
                                (jac_inv[k, 1]*delfstar_err[1])**2 +
                                (jac_inv[k, 2]*delfstar_err[2])**2)**0.5
    else:
        film = {'drho':np.nan, 'grho3':np.nan, 'phi':np.nan, 'dlam3':np.nan}
        deriv = {}
        for k in [0, 1, 2]:
            err[err_names[k]] = np.nan

    # now back calculate delfstar, rh and rd from the solution
    delfstar_calc = {}
    rh = {}
    rd = {}
    for n in nhplot:
        print('layers', layers)
        delfstar_calc[n] = calc_delfstar(n, layers, calctype)
        rd[n] = rd_from_delfstar(n, delfstar_calc)
    rh = rh_from_delfstar(nh, delfstar_calc)

    soln_output = {'film': film, 'dlam3': dlam3,
                   'delfstar_calc': delfstar_calc, 'rh': rh, 'rd': rd}

    soln_output['err'] = err
    soln_output['delfstar_err'] = delfstar_err
#    soln_output['deriv'] = deriv
    return soln_output



def null_solution(nhplot):
    film = {'drho':np.nan, 'grho3':np.nan, 'phi':np.nan, 'dlam3':np.nan}

    soln_output = {'film':film, 'dlam3':np.nan,
                   'err':{'drho':np.nan, 'grho3':np.nan, 'phi': np.nan}}

    delfstar_calc = {}
    rh = {}
    rd = {}
    for n in nhplot:
        delfstar_calc[n] = np.nan
        rd[n] = np.nan
    rh = np.nan
    soln_output['rd'] = rd
    soln_output['rh'] = rh
    soln_output['delfstar_calc'] = delfstar_calc

    return soln_output

def find_base_fig_name(sample, parms):
    # specify the location for the output figure files
    figlocation = parms.get('figlocation', 'figures')
    datadir = sample.get('datadir', '')
    filmfile = sample.get('filmfile', '')
    samplename = sample.get('samplename', 'null_name')
    if figlocation == 'datadir':
        base_fig_name = os.path.join(parms['dataroot'], datadir, filmfile)
    else:
        # check in which folder we are running
        cwd = os.getcwd()
        basename = os.path.basename(cwd) # name of current folder
        if basename == 'QCMFuncs': # running in QCMFuncs
            base_fig_path = os.path.join(cwd, 'figures')
        else: # running out of QCMFuns. Probably in QCM_py
            if 'QCMFuncs' in os.listdir(): # QCMFuncs is a subfolder
                base_fig_path = os.path.join(cwd, 'QCMFuncs', 'figures')
            else: # QCMFuncs is not a subfolder
                base_fig_path = os.path.join(cwd, 'figures') # save in figures/ directly

        if not os.path.exists(base_fig_path):
            os.mkdir(base_fig_path)

        base_fig_name = os.path.join(base_fig_path, samplename)

    print('path', base_fig_name)

    return base_fig_name

def analyze(sample, parms):
    global openplots
    sample['layers'] = {}
    if 'overlayer' in sample:
        sample['layers']['overlayer']=sample['overlayer']
    # add the appropriate file root to the data path
    sample['dataroot'] = parms['dataroot']
    # read in the optional inputs, assigning default values if not assigned
    nhplot = sample.get('nhplot', [1, 3, 5])
    # firstline = sample.get('firstline', 0)
    sample['xlabel'] = sample.get('xlabel',  't (min.)')
    Temp = np.array(sample.get('Temp', [22]))

    # set the appropriate value for xdata
    if Temp.shape[0] != 1:
        sample['xlabel'] = r'$T \: (^\circ$C)'

    sample['nhcalc'] = sample.get('nhcalc', ['355'])

    # set the color dictionary for the different harmonics
    colors = {1: [1, 0, 0], 3: [0, 0.5, 0], 5: [0, 0, 1]}
    parms['colors'] = colors

    # initialize the dictionary we'll use to keep track of the points to plot
    idx = {}

    # plot and process the film data
    film = process_raw(sample, 'film')
    # plot and process bare crystal data
    bare = process_raw(sample, 'bare')

    # if there is only one temperature, than we use time as the x axis, using
    # up to ten user-selected points
    if Temp.shape[0] == 1:
        if film['filmindex'] is not None:
            nx = len(film['filmindex'])
        else:
            nx = min(parms.get('nx',np.inf), film['n_in_range'])
    else:
        nx = Temp.shape[0]

    # move getting index out of for loop for getting index from dict
    if film['filmindex'] is not None:
        film['idx'] = film['filmindex']
    else:
        film['idx'] = pickpoints(Temp, nx, film)

    bare['idx'] = pickpoints(Temp, nx, bare)

    # pick the points that we want to analyze and add them to the plots
    for data_dict in [bare, film]:
        data_dict['fstar_err'] = {}
        idx = data_dict['idx']
        for n in nhplot:
            data_dict['fstar_err'][n] = np.zeros(data_dict['n_all'], dtype=np.complex128)
            for i in idx:
                data_dict['fstar_err'][n][i] = fstar_err_calc(data_dict['fstar'][n][i])
                t = data_dict['t'][i]
                f = data_dict['fstar'][n][i].real/n
                g = data_dict['fstar'][n][i].imag
                f_err = data_dict['fstar_err'][n][i].real/n
                g_err = data_dict['fstar_err'][n][i].imag
                data_dict['f_ax'].errorbar(t, f, yerr=f_err, color=colors[n],
                                      label='n='+str(n), marker='x')
                data_dict['g_ax'].errorbar(t, g, yerr=g_err, color=colors[n],
                                       label='n='+str(n), marker='x')

    # adjust nhcalc to account to only include calculations for for which
    # the data exist
    sample['nhcalc'] = nhcalc_in_nhplot(sample['nhcalc'], nhplot)

    # there is nothing left to do if nhcalc has no values
    if not(sample['nhcalc']):
        return

    # now calculate the frequency and dissipation shifts
    delfstar = {}
    delfstar_err = {}
    film['fstar_ref']={}

    # if the number of temperatures is 1, we use the average of the
    # bare temperature readings
    for n in nhplot:
        film['fstar_ref'][n] = np.zeros(film['n_all'], dtype=np.complex128)
        if Temp.shape[0] == 1:
            bare['fstar'][n] = bare['fstar'][n][~np.isnan(bare['fstar'][n])]
            film['fstar_ref'][n][film['idx']] = (np.average(bare['fstar'][n]) *
                                                 np.ones(nx))
        else:
            film['fstar_ref'][n][film['idx']] = bare['fstar'][n][bare['idx']]

    for i in np.arange(nx):
        idxf = film['idx'][i]
        delfstar[i] = {}
        delfstar_err[i] ={}
        for n in nhplot:
            delfstar[i][n] = (film['fstar'][n][idxf] - film['fstar_ref'][n][idxf])
            delfstar_err[i][n] = fstar_err_calc(film['fstar'][n][idxf])

    sample['delfstar'] = delfstar
    sample['delfstar_err'] = delfstar_err
    sample['film'] = film
    sample['bare'] = bare

    # set the appropriate value for xdata
    if Temp.shape[0] == 1:
        sample['xdata'] = film['t'][film['idx']]
    else:
        sample['xdata'] = Temp

    # set up the property axes
    sample['propfig'] = make_prop_axes('prop_'+sample['samplename'],
          sample['xlabel'])

    solve_from_delfstar(sample, parms)

    # tidy up the property figure
    cleanup_propfig(sample, parms)


def find_idx_in_range(t, t_range):
    if t_range[0] == t_range[1]:
        idx = np.arange(t.shape[0]).astype(int)
    else:
        idx = np.where((t >= t_range[0]) &
                       (t <= t_range[1]))[0]
    return idx


def nhcalc_in_nhplot(nhcalc_in, nhplot):
    # there is probably a more elegant way to do this
    # only consider harmonics in nhcalc that exist in nhplot
    nhcalc_out = []
    nhplot = list(set(nhplot))
    for nh in nhcalc_in:
        nhlist = list(set(nh))
        nhlist = [int(i) for i in nhlist]
        if all(elem in nhplot for elem in nhlist):
            nhcalc_out.append(nh)
    return nhcalc_out


def solve_from_delfstar(sample, parms):
    # this function is used if we already have a bunch of delfstar values
    # and want to obtain the solutions from there
    # now set the markers used for the different calculation types
    markers = {'131': '>', '133': '^', '353': '+', '355': 'x', '3': 'x'}
    colors = parms.get('colors',{1: [1, 0, 0], 3: [0, 0.5, 0], 5: [0, 0, 1]})
    imagetype = parms.get('imagetype', 'svg')
    # get film info (containing raw data plot, etc. if it exists)
    sample['film']=sample.get('film',{})
    close_on_click_switch = parms.get('close_on_click_switch', True)

    base_fig_name = find_base_fig_name(sample, parms)
    imagetype = parms.get('imagetype', 'svg')
    nhplot = sample.get('nhplot', [1, 3, 5])
    delfstar = sample['delfstar']
    nx = len(delfstar)  # this is the number of data points

    if 'xdata' in sample:
        xdata = sample['xdata']
    else:
        xdata=np.arange(nx)
        sample['xlabel'] = 'index'
    if 'propfig' in sample:
        propfig = sample['propfig']
    else:
        propfig = make_prop_axes('props', sample['xlabel'])

    # set up the consistency check axes
    checkfig = {}
    for nh in sample['nhcalc']:
        checkfig[nh] = make_check_axes(sample, nh)
        if close_on_click_switch and not run_from_ipython():
            # when code is run with IPython don't use the event
            checkfig[nh]['figure'].canvas.mpl_connect('key_press_event',
                                                    close_on_click)

    # now do all of the calculations and plot the data
    soln_input = {'nhplot': nhplot}
    soln_input['layers']=sample.get('layers',{})
    results = {}
    for nh in sample['nhcalc']:
        # initialize all the dictionaries
        results[nh] = {'film':{'drho':np.zeros(nx), 'drho_err':np.zeros(nx),
                              'grho3':np.zeros(nx), 'grho3_err':np.zeros(nx),
                              'phi':np.zeros(nx), 'phi_err':np.zeros(nx)},
                       'dlam3':np.zeros(nx),
                       'rd': {}, 'rh': {}, 'delfstar_calc': {}}
        for n in nhplot:
            results[nh]['delfstar_calc'][n] = (np.zeros(nx,
                                               dtype=np.complex128))
            results[nh]['rd'][n] = np.zeros(nx)

        results[nh]['rh'] = np.zeros(nx)

        for i in np.arange(nx):
            # obtain the solution for the properties
            soln_input['nh'] = nh
            soln_input['delfstar'] = delfstar[i]
            if (np.isnan(delfstar[i][int(nh[0])].real) or
                np.isnan(delfstar[i][int(nh[1])].real) or
                np.isnan(delfstar[i][int(nh[2])].imag)):
                soln = null_solution(nhplot)
            else:
                soln = solve_for_props(soln_input)

            results[nh]['film']['drho'][i] = soln['film']['drho']
            results[nh]['film']['grho3'][i] = soln['film']['grho3']
            results[nh]['film']['phi'][i] = soln['film']['phi']
            results[nh]['film']['drho_err'][i] = soln['err']['drho']
            results[nh]['film']['grho3_err'][i] = soln['err']['grho3']
            results[nh]['film']['phi_err'][i] = soln['err']['phi']
            results[nh]['dlam3'][i] = soln['dlam3']

            for n in nhplot:
                results[nh]['delfstar_calc'][n][i] = (
                 soln['delfstar_calc'][n])
                results[nh]['rd'][n][i] = soln['rd'][n]
            results[nh]['rh'][i] = soln['rh']

            # add actual values of delf, delg for each harmonic to the
            # solution check figure
            for n in nhplot:
                checkfig[nh]['delf_ax'].plot(xdata[i], delfstar[i][n].real
                                             / n, '+', color=colors[n])
                checkfig[nh]['delg_ax'].plot(xdata[i], delfstar[i][n].imag,
                                             '+', color=colors[n])
            # add experimental rh, rd to solution check figure
            checkfig[nh]['rh_ax'].plot(xdata[i], rh_from_delfstar(nh,
                                       delfstar[i]), '+', color=colors[n])

            for n in nhplot:
                checkfig[nh]['rd_ax'].plot(xdata[i], rd_from_delfstar(n,
                                           delfstar[i]), '+', color=colors[n])

        # add the calculated values of rh, rd to the solution check figures
        checkfig[nh]['rh_ax'].plot(xdata, results[nh]['rh'], '-')
        for n in nhplot:
            checkfig[nh]['rd_ax'].plot(xdata, results[nh]['rd'][n], '-',
                                       color=colors[n])

        # add calculated delf and delg to solution check figures
        for n in nhplot:
            (checkfig[nh]['delf_ax'].plot(xdata,
             results[nh]['delfstar_calc'][n].real / n, '-',
             color=colors[n], label='n='+str(n)))
            (checkfig[nh]['delg_ax'].plot(xdata,
             results[nh]['delfstar_calc'][n].imag, '-', color=colors[n],
             label='n='+str(n)))

        # add legend to the solution check figures
        checkfig[nh]['delf_ax'].legend()
        checkfig[nh]['delg_ax'].legend()

        if 'xscale' in sample:
            checkfig[nh]['delf_ax'].set_xscale(sample['xscale'])
            checkfig[nh]['delg_ax'].set_xscale(sample['xscale'])
            checkfig[nh]['rh_ax'].set_xscale(sample['xscale'])
            checkfig[nh]['rd_ax'].set_xscale(sample['xscale'])


        # tidy up the solution check figure
        checkfig[nh]['figure'].tight_layout()
        checkfig[nh]['figure'].savefig(base_fig_name + '_'+nh +
                                       '.' + imagetype)

        # get the property data to add to the property figure
        drho = 1000*results[nh]['film']['drho']
        grho3 = results[nh]['film']['grho3']/1000
        phi = results[nh]['film']['phi']
        drho_err = 1000*results[nh]['film']['drho_err']
        grho3_err = results[nh]['film']['grho3_err']/1000
        phi_err = results[nh]['film']['phi_err']

        # this is where we determine what marker to use.  We change it if
        # we have specified a different marker in the sample dictionary
        if 'forcemarker' in sample:
            markers[nh] = sample['forcemarker']

        # add property data with error bars to the figure
        propfig['drho_ax'].errorbar(xdata, drho, yerr=drho_err,
                                    marker=markers[nh], label=nh)
        propfig['grho3_ax'].errorbar(xdata, grho3, yerr=grho3_err,
                                    marker=markers[nh], label=nh)
        propfig['phi_ax'].errorbar(xdata, phi, yerr=phi_err,
                                   marker=markers[nh], label=nh)
        output_data = np.stack((xdata, drho, grho3, phi), axis=-1)
        np.savetxt(base_fig_name+'_'+nh+'.txt', output_data,
                   delimiter=',', header='xdata,drho,grho,phi', comments='')

        # add values of d/lam3 to the film raw data figure
        if 'rawfig' in sample['film']:
            sample['film']['dlam3_ax'].plot(xdata, results[nh]['dlam3'], '+', label=nh)

    # add legend to the the dlam3 figure and set the x axis label
    if 'rawfig' in sample['film']:
        sample['film']['dlam3_ax'].legend()
        sample['film']['dlam3_ax'].set_xlabel(sample['xlabel'])

    print('done with ', base_fig_name, 'press any key to close plots and continue')

    if close_on_click_switch and not run_from_ipython():
        # when code is run with IPython, don't use the event
        propfig['figure'].canvas.mpl_connect('key_press_event', close_on_click)
        if 'rawfig' in sample['film']:
            sample['film']['rawfig'].canvas.mpl_connect('key_press_event', close_on_click)
            sample['bare']['rawfig'].canvas.mpl_connect('key_press_event', close_on_click)

    openplots = 3 + len(checkfig)
    if not run_from_ipython():
        # when code is run with IPython, don't use key_press_event
        while openplots>0:
            plt.pause(1)

    sample['results'] = results
    return sample

def cleanup_propfig(sample, parms):
    make_legend = parms.get('make_legend', True)
    make_titles = parms.get('make_titles', True)
    propfig = sample['propfig']
    # add legends to the property figure
    if make_legend:
        propfig['drho_ax'].legend()
        propfig['grho3_ax'].legend()
        propfig['phi_ax'].legend()
    # add axes titles to the property figure
    if make_titles:
        propfig['drho_ax'].set_title('(a)')
        propfig['grho3_ax'].set_title('(b)')
        propfig['phi_ax'].set_title('(c)')
    # adjust linear and log axes as desired
    if 'xscale' in sample:
        propfig['drho_ax'].set_xscale(sample['xscale'])
        propfig['grho3_ax'].set_xscale(sample['xscale'])
        propfig['phi_ax'].set_xscale(sample['xscale'])
    if 'grho3scale' in sample:
        propfig['grho3_ax'].set_yscale(sample['grho3scale'])
    propfig['figure'].tight_layout()
    # write to standard location specified in sample definition
    propfig['figure'].savefig(find_base_fig_name(sample, parms)+
          '_prop.'+parms.get('imagetype', 'svg'))



def pickpoints(Temp, nx, data_dict):
    t_in = data_dict['t']
    idx_in = data_dict['idx_in_range']
    idx_file = data_dict['idx_file']
    idx_out = np.array([], dtype=int)
    if Temp.shape[0] == 1:
        try:
            idx_out = np.loadtxt(idx_file, dtype=int)
            if len(idx_out) > nx:
                idx_out = idx_out[:nx]
            elif len(idx_out) < nx:
                add_idx = np.ones(nx-len(idx_out), dtype=int) * idx_out[-1]
                idx_out = np.concatenate((idx_out, add_idx))
        except:
            t = np.linspace(min(t_in[idx_in]), max(t_in[idx_in]), nx)
            for n in np.arange(nx):
                idx_out = np.append(idx_out, (np.abs(t[n] - t_in)).argmin())
            idx_out = np.asarray(idx_out)

    elif Path(idx_file).is_file():
        idx_out = np.loadtxt(idx_file, dtype=int)
    else:
        # make the correct figure active
        plt.figure(data_dict['rawfigname'])
        print('click on plot to pick ', str(nx), 'points')
        pts = plt.ginput(nx, timeout=0)
        pts = np.array(pts)[:, 0]
        for n in np.arange(nx):
            idx_out = np.append(idx_out, (np.abs(pts[n] - t_in)).argmin())
        idx_out = np.asarray(idx_out)
        np.savetxt(idx_file, idx_out, fmt='%4i')

    return idx_out


def close_existing_fig(figname):
    if plt.fignum_exists(figname):
        plt.close(figname)
    return


def make_prop_axes(propfigname, xlabel):
    # set up the property plot
    close_existing_fig(propfigname)
    fig = plt.figure(propfigname, figsize=(9, 3))
    drho_ax = fig.add_subplot(131)
    drho_ax.set_xlabel(xlabel)
    drho_ax.set_ylabel(r'$d\rho$ (g/m$^2$)')

    grho3_ax = fig.add_subplot(132)
    grho3_ax.set_xlabel(xlabel)
    grho3_ax.set_ylabel(r'$|G_3^*|\rho$ (Pa $\cdot$ g/cm$^3$)')

    phi_ax = fig.add_subplot(133)
    phi_ax.set_xlabel(xlabel)
    phi_ax.set_ylabel(r'$\phi$ (deg.)')

    fig.tight_layout()

    return {'figure': fig, 'drho_ax': drho_ax, 'grho3_ax': grho3_ax,
            'phi_ax': phi_ax}


def make_vgp_axes(vgpfigname):
    close_existing_fig(vgpfigname)
    fig = plt.figure(vgpfigname, figsize=(3, 3))
    vgp_ax = fig.add_subplot(111)
    vgp_ax.set_xlabel((r'$|G_3^*|\rho$ (Pa $\cdot$ g/cm$^3$)'))
    vgp_ax.set_ylabel(r'$\phi$ (deg.)')
    fig.tight_layout()
    return {'figure': fig, 'vgp_ax':vgp_ax}


def prop_plot_from_csv(figure, csvfile, plotstr, legendtext):
    data = pd.read_csv(csvfile)
    figure['drho_ax'].plot(data['xdata'].values, data['drho'].values, plotstr,
                label=legendtext)
    figure['grho3_ax'].plot(data['xdata'].values, data['grho'].values, plotstr,
                label=legendtext)
    figure['phi_ax'].plot(data['xdata'].values, data['phi'].values, plotstr,
                label=legendtext)


def vgp_plot_from_csv(figure, csvfile, plotstr, legendtext):
    data = pd.read_csv(csvfile)
    figure['vgp_ax'].semilogx(data['grho'].values, data['phi'].values, plotstr,
                label=legendtext)


def process_raw(sample, data_type):
    colors = {1: [1, 0, 0], 3: [0, 0.5, 0], 5: [0, 0, 1]}
    # specify the film filenames
    firstline = sample.get('firstline', 0)
    nhplot = sample.get('nhplot', [1, 3, 5])
    trange = sample.get(data_type+'trange', [0, 0])
    datadir = sample.get('datadir', '')

    data_dict = {}

    filetype = sample.get('filetype', 'mat') # mat or h5
    if filetype == 'mat': # mat file to read
        data_dict['file'] = os.path.join(sample['dataroot'], datadir, sample[data_type+'file'] + '.' + filetype)
        data_dict['data'] = hdf5storage.loadmat(data_dict['file'])

        # extract the frequency data from the appropriate file
        freq = data_dict['data']['abs_freq'][firstline:, 0:7]
        # get rid of all the rows that don't have any data
        freq = freq[~np.isnan(freq[:, 1:]).all(axis=1)]
        # reference frequencies are the first data points for the bare crystal data
        sample['freqref'] = sample.get('freqref', freq[0, :])

        # extract frequency information
        data_dict['t'] = freq[:, 0]
        data_dict['fstar'] = {}

        for n in nhplot:
            data_dict['fstar'][n] = freq[:, n] +1j*freq[:, n+1] - sample['freqref'][n]

        # set key for getting index to plot from txt file
        data_dict['idx_file'] = os.path.join(sample['dataroot'], datadir, sample[data_type+'file']+'_film_idx.txt')

    elif filetype == 'h5': # h5 file to read
        data_dict['file'] = os.path.join(sample['dataroot'], datadir, sample['filmfile'] + '.' + filetype)
        filmchn = sample.get('filmchn', 'samp') # default is samp
        # load file
        data_saver.load_file(data_dict['file'])

        # read data from file
        if data_type == 'film': # get film data
            df = data_saver.reshape_data_df(filmchn, mark=True, dropnanrow=False, dropnancolumn=False, deltaval=False, norm=False, unit_t='m', unit_temp='C')
        elif data_type == 'bare': # get bare data
            df = data_saver.reshape_data_df(filmchn+'_ref', mark=True, dropnanrow=False, dropnancolumn=False, deltaval=False, norm=False, unit_t='m', unit_temp='C')
            pass

        # reference frequencies are the first data points for the bare crystal data
        # get t
        data_dict['t'] = df.t.values # make it as ndarray, unit in min
        data_dict['fstar'] = {}
        for n in nhplot:
            data_dict['fstar'][n] = df['f'+str(n)].values +1j*df['g'+str(n)].values

        # set key for getting index to plot from txt file
        data_dict['idx_file'] = os.path.join(sample['dataroot'], datadir, sample['filmfile']+'_film_idx.txt')

    # get index to plot from *_sampledefs.py
    if 'filmindex' in sample:
        data_dict['filmindex'] = np.array(sample['filmindex'], dtype=int)
    else:
        data_dict['filmindex'] = None



    # figure out how man total points we have
    data_dict['n_all'] = data_dict['t'].shape[0]

    #  find all the time points between specified by timerange
    #  if the min and max values for the time range are equal, we use
    #    all the points
    data_dict['idx_in_range'] = find_idx_in_range(data_dict['t'], trange)
    data_dict['n_in_range'] = data_dict['idx_in_range'].shape[0]

    # rewrite nhplot to account for the fact that data may not exist for all
    # of the harmonics
    data_dict['n_exist'] = np.array([]).astype(int)

    for n in nhplot:
        if not all(np.isnan(data_dict['fstar'][n])):
            data_dict['n_exist'] = np.append(data_dict['n_exist'], n)

    # make the figure with its axis
    rawfigname = 'raw_'+data_type+'_'+sample['samplename']
    close_existing_fig(rawfigname)
    if data_type == 'bare':
        numplots=2
    else:
        numplots=3

    data_dict['rawfig'] = plt.figure(rawfigname, figsize=(numplots*3,3))

    data_dict['f_ax'] = data_dict['rawfig'].add_subplot(1,numplots,1)
    data_dict['f_ax'].set_xlabel('t (min.)')
    data_dict['f_ax'].set_ylabel(r'$\Delta f_n/n$ (Hz)')
    data_dict['f_ax'].set_title(data_type)

    data_dict['g_ax'] = data_dict['rawfig'].add_subplot(1,numplots,2)
    data_dict['g_ax'].set_xlabel('t (min.)')
    data_dict['g_ax'].set_ylabel(r'$\Gamma$ (Hz)')
    data_dict['g_ax'].set_title(data_type)

    if numplots == 3:
        data_dict['dlam3_ax'] = data_dict['rawfig'].add_subplot(1,numplots,3)
        data_dict['dlam3_ax'].set_xlabel('t (min.)')
        data_dict['dlam3_ax'].set_ylabel(r'$d/\lambda_3$')
        data_dict['dlam3_ax'].set_title(data_type)

    if 'xscale' in sample:
        data_dict['f_ax'].set_xscale(sample['xscale'])
        data_dict['g_ax'].set_xscale(sample['xscale'])
        if numplots == 3:
            data_dict['dlam3_ax'].set_xscale(sample['xscale'])


    # plot the raw data
    for n in nhplot:
        t = data_dict['t'][data_dict['idx_in_range']]
        f = data_dict['fstar'][n][data_dict['idx_in_range']].real/n
        g = data_dict['fstar'][n][data_dict['idx_in_range']].imag
        (data_dict['f_ax'].plot(t, f, color=colors[n], label='n='+str(n)))
        (data_dict['g_ax'].plot(t, g, color=colors[n], label='n='+str(n)))

    # add the legends
    data_dict['f_ax'].legend()
    data_dict['g_ax'].legend()

    data_dict['rawfig'].tight_layout()
    data_dict['rawfigname'] = rawfigname

    return data_dict


def make_check_axes(sample, nh):
    close_existing_fig(nh + 'solution check')
    #  compare actual annd recaulated frequency and dissipation shifts.
    fig = plt.figure(nh + '_solution check_'+sample['samplename'])
    delf_ax = fig.add_subplot(221)
    delf_ax.set_xlabel(sample['xlabel'])
    delf_ax.set_ylabel(r'$\Delta f/n$ (Hz)')

    delg_ax = fig.add_subplot(222)
    delg_ax.set_xlabel(sample['xlabel'])
    delg_ax.set_ylabel(r'$\Delta \Gamma$ (Hz)')

    rh_ax = fig.add_subplot(223)
    rh_ax.set_xlabel(sample['xlabel'])
    rh_ax.set_ylabel(r'$r_h$')

    rd_ax = fig.add_subplot(224)
    rd_ax.set_xlabel(sample['xlabel'])
    rd_ax.set_ylabel(r'$r_d$')

    fig.tight_layout()

    return {'figure': fig, 'delf_ax': delf_ax, 'delg_ax': delg_ax,
            'rh_ax': rh_ax, 'rd_ax': rd_ax}


def plot_spectra(fig_dict, sample, idx_vals):
    datadir = sample.get('datadir','')
    if not 'fig' in fig_dict:
        print('making new figure')
        fig = plt.figure('spectra', figsize=(9, 9))
        G_ax = {}  # conductance plots
        B_ax = {}  # susceptance plots
        Nyquist_ax = {}  # Nyquist plot
        plot_num = 1

        for n in [1, 3, 5]:
            # set up the different axis
            G_ax[n] = fig.add_subplot(3,3, (n+1)/2)
            G_ax[n].set_xlabel('$f$ (Hz)')
            G_ax[n].set_ylabel('$G$ (S)')
            B_ax[n] = fig.add_subplot(3,3, 3+(n+1)/2)
            B_ax[n].set_xlabel('$f$ (Hz)')
            B_ax[n].set_ylabel('$B$ (S)')
            Nyquist_ax[n] = fig.add_subplot(3,3, 6+(n+1)/2)
            Nyquist_ax[n].set_xlabel('$B$ (S)')
            Nyquist_ax[n].set_ylabel('$G$ (S)')
        fig.tight_layout()

    else:
        fig = fig_dict['fig']
        G_ax = fig_dict['G_ax']
        B_ax = fig_dict['B_ax']
        Nyquist_ax = fig_dict['Nyquist_ax']
        plot_num = fig_dict['plot_num']+1

    # define dictionaries for the relevant axes
    # the following command controls the way the offsets are handled
    # 4 is the default, but 2 seems to work better here
    plt.rcParams['axes.formatter.offset_threshold'] = 2

       # read the data
    spectra_file = os.path.join(datadir, sample['filmfile'] + '_raw_spectras.mat')
    spectra = hdf5storage.loadmat(spectra_file)

    for n in [1, 3, 5]:
        for idx in idx_vals:
            # read the data
            # this is for version 'e' of Josh's program, where the columns are
            # frequency, exp. G, B, fit G,B and residuals for G, B
            f = spectra['raw_spectra_'+str(n)][idx][3][:, 0]
            G_exp = spectra['raw_spectra_'+str(n)][idx][3][:, 1]
            B_exp = spectra['raw_spectra_'+str(n)][idx][3][:, 2]
            G_fit= spectra['raw_spectra_'+str(n)][idx][3][:, 3]
            B_fit = spectra['raw_spectra_'+str(n)][idx][3][:, 4]
            G_res = spectra['raw_spectra_'+str(n)][idx][3][:, 5]
            B_res = spectra['raw_spectra_'+str(n)][idx][3][:, 6]

            # now make the plots
            G_ax[n].plot(f, G_exp, 'b-')
            B_ax[n].plot(f, B_exp, 'b-')
            Nyquist_ax[n].plot(G_exp, B_exp, 'b-')
            G_ax[n].plot(f, G_fit, 'r-')
            B_ax[n].plot(f, B_fit, 'r-')
            G_ax[n].plot(f, G_res, 'g-')
            B_ax[n].plot(f, B_res, 'g-')

        # add label to plots
        text_y = spectra['raw_spectra_'+str(n)][idx_vals[0]][3][:, 1].max()
        max_idx = spectra['raw_spectra_'+str(n)][idx_vals[0]][3][:, 1].argmax()
        text_x = spectra['raw_spectra_'+str(n)][idx_vals[0]][3][max_idx, 0]
        G_ax[n].text(text_x, text_y, str(plot_num), horizontalalignment='center')


    return  {'fig': fig, 'G_ax': G_ax, 'B_ax': B_ax,
            'Nyquist_ax': Nyquist_ax, 'plot_num': plot_num}


def contour(function, parms):
    # set up the number of points and establisth the great
    n = parms.get('n', 100)
    phi = parms.get('phi', np.linspace(0, 90, n))
    dlam = parms.get('dlam', np.linspace(0, 0.5, n))
    dlam_i, phi_i = np.meshgrid(dlam, phi)

    # calculate the z values and reset things to -1 at dlam=0
    z = normdelfstar(3, dlam_i, phi_i)
    z[:,0] = -1

    # now make the contour plots

    fig = plt.figure('contour', figsize=(6, 3))

    flevels = np.linspace(-2, 0, 1000)
    glevels = np.linspace(0, 2, 1000)

    # start with frequency shift
    f_ax = fig.add_subplot(121)
    f_ax.set_xlabel(r'$d/\lambda_n$')
    f_ax.set_ylabel(r'$\phi$ (deg.)')
    f_ax.set_title(r'$\Delta f_n/f_{sn}$')
    f_map = f_ax.contourf(dlam_i, phi_i, z.real, flevels, cmap="gnuplot2",
                          extend='both')
    fig.colorbar(f_map, ax=f_ax, ticks=np.linspace(min(flevels),
                 max(flevels), 6))

    # we need this to get rid of the lines in the contour plot
    for c in f_map.collections:
        c.set_edgecolor("face")

    # now plot the dissipation
    g_ax = fig.add_subplot(122)
    g_ax.set_xlabel(r'$d/\lambda_n$')
    g_ax.set_ylabel(r'$\phi$ (deg.)')
    g_ax.set_title(r'$\Delta \Gamma_n/f_{sn}$')
    g_map = g_ax.contourf(dlam_i, phi_i, z.imag, glevels, cmap="gnuplot2_r",
                          extend='both')
    fig.colorbar(g_map, ax=g_ax, ticks=np.linspace(min(glevels),
                 max(glevels), 6))
    for c in g_map.collections:
        c.set_edgecolor("face")

    fig.tight_layout()

    return

def bulk_props(delfstar):
    # get the bulk solution for grho and phi
    grho3 = (np.pi*Zq*abs(delfstar[3])/f1) ** 2
    phi = -np.degrees(2*np.arctan(delfstar[3].real /
                      delfstar[3].imag))

    return [grho3, phi]


def bulk_guess(delfstar):
    # get the bulk solution for grho and phi
    grho3 = (np.pi*Zq*abs(delfstar[3])/f1) ** 2
    phi = -np.degrees(2*np.arctan(delfstar[3].real /
                      delfstar[3].imag))

    # calculate rho*lambda
    lamrho3 = np.sqrt(grho3)/(3*f1*np.cos(np.deg2rad(phi/2)))

    # we need an estimate for drho.  We oNy use thi approach if it is
    # reasonably large.  We'll put it at the quarter wavelength condition
    # for now
    drho = lamrho3/4
    film = {'drho':drho, 'grho3':grho3, 'phi':phi}
    dlam3 = calc_dlam(3, film)

    return [dlam3, min(phi, 90)]


def guess_from_props(film):
    dlam3 = calc_dlam(3, film)
    return [dlam3, film['phi']]


def thinfilm_guess(delfstar):
    # really a placeholder function until we develop a more creative strategy
    # for estimating the starting point
    return [0.05, 5]

def make_knots(numpy_array, num_knots):
    # makes num_knots eveNy spaced knots along array
    knot_interval = (np.max(numpy_array)-np.min(numpy_array))/(num_knots+1)
    minval = np.min(numpy_array)+knot_interval
    maxval = np.max(numpy_array)-knot_interval
    knots = np.linspace(minval, maxval, num_knots)
    return knots

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


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def print_test(variable):
    print(f1)