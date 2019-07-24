#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:19:59 2018

@author: ken
"""

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import pandas as pd

Zq = 8.84e6  # shear acoustic impedance of at cut quartz
f1 = 5e6  # fundamental resonant frequency
openplots = 4
drho_q = Zq/(2*f1)
e26 = 9.65e-2
electrode_default = {'drho':2.8e-3, 'grho3':3.0e14, 'phi':0}

# find indices of an array where the values are closet to the ones specified
def find_nearest_idx(values, array):
    # find index of a point with value closest to the one specified
    # make values a numpy array if it isn't already
    values = np.asarray(values).reshape(1, -1)[0,:]
    idx = np.zeros(values.size, dtype=int)
    for i in np.arange(values.size):
        idxval = np.searchsorted(array, values[i], side="left")
        if idxval > 0 and (idxval == len(array) or np.abs(values[i] - array[idxval-1]) < 
                        np.abs(values[i] - array[idxval])):
            idx[i] = idxval-1
        else:
            idx[i] = idxval
    return idx


def find_idx_in_range(t, t_range):
    if t_range[0] == t_range[1]:
        idx = np.arange(t.shape[0]).astype(int)
    else:
        idx = np.where((t >= t_range[0]) &
                       (t <= t_range[1]))[0]
    return idx


def close_on_click(event):
    # used so plots close in response to a some event
    global openplots
    plt.close()
    openplots = openplots - 1
    return


def add_D_axis(ax):
    # add right hand axis with dissipation
    axD = ax.twinx()
    axD.set_ylabel(r'$\Delta D_n$ (ppm)')
    axlim = ax.get_ylim()
    ylim = tuple(2*lim/5 for lim in axlim)
    axD.set_ylim(ylim)
    return axD


def fstar_err_calc(n, delfstar, layers):
    # calculate the error in delfstar
    g_err_min = 1 # error floor for gamma
    f_err_min = 1 # error floor for fd
    err_frac = 1e-2 # error in f or gamma as a fraction of gamma
    if 'overlayer' in layers:
        fstar = delfstar + calc_delfstar(n, {'film':layers['overlayer']}, 
                                             'SLA')
    else:
        fstar = delfstar
            
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


def grho_bulk(n, delfstar):
    return (np.pi*Zq*abs(delfstar[n])/f1) ** 2


def phi_bulk(n, delfstar):
    return -np.degrees(2*np.arctan(np.real(delfstar[n]) /
                       np.imag(delfstar[n])))

    
def deltarho_bulk(n, delfstar):
    #decay length multiplied by density
    return -Zq*abs(delfstar[n])**2/(2*n*f1**2*delfstar[n].real)


def calc_D(n, material, delfstar,calctype):
    drho = material['drho']
    # set switch to handle ase where drho = 0
    if drho == 0:
        return 0
    else:
        return 2*np.pi*(n*f1+delfstar)*drho/zstar_bulk(n, 
                       material,calctype)


def zstar_bulk(n, material, calctype):
    grho3 = material['grho3']
    if calctype != 'QCMD':
        grho = grho3*(n/3)**(material['phi']/90) 
        grhostar = grho*np.exp(1j*np.pi*material['phi']/180)
    else:
        # Qsense version: constant G', G" linear in omega
        greal=grho3*np.cos(np.radians(material['phi']))
        gimag=(n/3)*np.sin(np.radians(material['phi']))
        grhostar=(gimag**2+greal**2)**(0.5)*(np.exp(1j*
                 np.radians(material['phi'])))        
    return grhostar ** 0.5


def calc_delfstar_sla(ZL):
    return f1*1j/(np.pi*Zq)*ZL


def calc_ZL(n, layers, delfstar, calctype):
    # layers is a dictionary of dictionaries
    # each dictionary is named according to the layer number
    # layer 1 is closest to the quartz

    N = len(layers)
    Z = {}; D = {}; L = {}; S = {}

    # we use the matrix formalism to avoid typos.
    for i in np.arange(1, N):
        Z[i] = zstar_bulk(n, layers[i],calctype)
        D[i] = calc_D(n, layers[i], delfstar, calctype)
        L[i] = np.array([[np.cos(D[i])+1j*np.sin(D[i]), 0],
                 [0, np.cos(D[i])-1j*np.sin(D[i])]])

    # get the terminal matrix from the properties of the last layer
    D[N] = calc_D(n, layers[N], delfstar, calctype)
    Zf_N = 1j*zstar_bulk(n, layers[N], calctype)*np.tan(D[N])

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
        ZL = calc_ZL(n, {1:layers['film'], 2:layers['overlayer']},
                         0,calctype)
        ZL_ref = calc_ZL(n, {1:layers['overlayer']}, 0, calctype)
        del_ZL = ZL-ZL_ref
    else:
        del_ZL = calc_ZL(n, {1:layers['film']}, 0, calctype)

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

        ZL_all = calc_ZL(n, layers_all, 0, calctype)
        delfstar_sla_all = calc_delfstar_sla(ZL_all)
        ZL_ref = calc_ZL(n, layers_ref, 0,calctype)
        delfstar_sla_ref = calc_delfstar_sla(ZL_ref)


        def solve_Zmot(x):
            delfstar = x[0] + 1j*x[1]
            Zmot = calc_Zmot(n,  layers_all, delfstar, calctype)
            return [Zmot.real, Zmot.imag]

        sol = optimize.root(solve_Zmot, [delfstar_sla_all.real,
                                         delfstar_sla_all.imag])
        dfc = sol.x[0] + 1j* sol.x[1]

        def solve_Zmot_ref(x):
            delfstar = x[0] + 1j*x[1]
            Zmot = calc_Zmot(n,  layers_ref, delfstar, calctype)
            return [Zmot.real, Zmot.imag]

        sol = optimize.root(solve_Zmot_ref, [delfstar_sla_ref.real,
                                             delfstar_sla_ref.imag])
        dfc_ref = sol.x[0] + 1j* sol.x[1]

        return dfc-dfc_ref


def calc_Zmot(n, layers, delfstar, calctype):
    om = 2 * np.pi *(n*f1 + delfstar)
    g0 = 10 # Half bandwidth of unloaed resonator (intrinsic dissipation on crystalline quartz)
    Zqc = Zq * (1 + 1j*2*g0/(n*f1))
    dq = 330e-6  # only needed for piezoelectric stiffening calc.
    epsq = 4.54; eps0 = 8.8e-12; C0byA = epsq * eps0 / dq; ZC0byA = C0byA / (1j*om)
    ZPE = -(e26/dq)**2*ZC0byA  # ZPE accounts for oiezoelectric stiffening anc
    # can always be neglected as far as I can tell

    Dq = om*drho_q/Zq
    secterm = -1j*Zqc/np.sin(Dq)
    ZL = calc_ZL(n, layers, delfstar,calctype)
    # eq. 4.5.9 in book
    thirdterm = ((1j*Zqc*np.tan(Dq/2))**-1 + (1j*Zqc*np.tan(Dq/2) + ZL)**-1)**-1
    Zmot = secterm + thirdterm  +ZPE

    return Zmot


def calc_dlam(n, film, calctype):
    return calc_D(n, film, 0, calctype).real/(2*np.pi)


def calc_lamrho(n, grho3, phi):
    # calculate lambda*rho
    grho=grho3*(n/3) ** (phi/90)
    return np.sqrt(grho)/(n*f1*np.cos(np.deg2rad(phi/2)))


def calc_deltarho(n, grho3, phi):
    # calculate delta*rho (decay length times density)
    return calc_lamrho(n, grho3, phi)/(2*np.pi*np.tan(np.radians(phi/2)))


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

    return [drho, grho3, min(phi, 90)]


def thinfilm_guess(delfstar, nh):
    # really a placeholder function until we develop a more creative strategy
    # for estimating the starting point
    n1 = int(nh[0])
    n2 = int(nh[1])
    n3 = int(nh[2])


    rd_exp = -delfstar[n3].imag/delfstar[n3].real
    rh_exp = (n2/n1)*delfstar[n1].real/delfstar[n2].real
    lb = np.array([0, 0])  # lower bounds on dlam3 and phi
    ub = np.array([5, 90])  # upper bonds on dlam3 and phi

    # we solve the problem initially using the harmonic and dissipation
    # ratios, using the small load approximation
    # we also neglect the overlayer in this first calculation
    def ftosolve(x):
        return [rhcalc(nh, x[0], x[1])-rh_exp, rdcalc(nh, x[0], x[1])-rd_exp]
    guess = [0.05, 5]
    soln = optimize.least_squares(ftosolve, guess, bounds=(lb, ub))

    dlam3 = soln['x'][0]
    phi = soln['x'][1]
    drho = (sauerbreym(n1, delfstar[n1].real) /
            normdelfstar(n1, dlam3, phi).real)
    grho3 = grho_from_dlam(3, drho, dlam3, phi)
    return drho, grho3, phi


def solve_for_props(soln_input):
    # solve the QCM equations to determine the properties
    if 'overlayer' in soln_input:
        layers={'overlayer':soln_input['overlayer']}
    else:
        layers={}
    nhplot = soln_input.get('nhplot', [1, 3, 5])
    calctype = soln_input.get('calctype', 'SLA')
    nh = soln_input['nh']
    n1 = int(nh[0])
    n2 = int(nh[1])
    n3 = int(nh[2])
    delfstar = soln_input['delfstar']
    rd_exp = -delfstar[n3].imag/delfstar[n3].real

    if 'prop_guess' in soln_input:
        guess=soln_input['prop_guess']
        drho, grho3, phi = guess['drho'], guess['grho3'], guess['phi']
    elif rd_exp > 0.5:
        drho, grho3, phi = bulk_guess(delfstar)
    else:
        drho, grho3, phi = thinfilm_guess(delfstar, nh)

    # set initial guess and upper and lower bounds
    x0 = np.array([drho, grho3, phi])
    lb = np.array([0, 1e5, 0])  # lower bounds drho, grho3, phi
    ub = np.array([1e-2, 1e13, 90])  # upper bounds drho, grho3, phi

    # otain the solutoin, using either the SLA or LL methods

    def ftosolve(x):
        layers['film'] = {'drho':x[0], 'grho3':x[1], 'phi':x[2]}
        return ([delfstar[n1].real-calc_delfstar(n1, layers, calctype).real,
                 delfstar[n2].real-calc_delfstar(n2, layers, calctype).real,
                 delfstar[n3].imag-calc_delfstar(n3, layers, calctype).imag])

    # put the input uncertainties into a 3 element vector
    delfstar_err = np.zeros(3)

    delfstar_err[0] = fstar_err_calc(n1, delfstar[n1], layers).real
    delfstar_err[1] = fstar_err_calc(n2, delfstar[n2], layers).real
    delfstar_err[2] = fstar_err_calc(n3, delfstar[n3], layers).imag

    # initialize the output uncertainties
    err = {}
    err_names = ['drho', 'grho3', 'phi']
    # recalculate solution to give the uncertainty, if solution is viable

    if np.all(lb < x0) and np.all(x0 < 1.1*ub):
        soln2 = optimize.least_squares(ftosolve, x0, bounds=(lb, ub))
        drho = soln2['x'][0]
        grho3 = soln2['x'][1]
        phi = soln2['x'][2]
        film = {'drho':drho, 'grho3':grho3, 'phi':phi}
        dlam3 = calc_dlam(3, film, calctype)
        jac = soln2['jac']
        try:
            jac_inv = np.linalg.inv(jac)
        except:
            jac_inv = np.zeros([3,3])

        # define sensibly named partial derivatives for further use
        deriv = {}
        for k in [0, 1, 2]:
            deriv[err_names[k]]={0:jac_inv[k, 0], 1:jac_inv[k, 1], 2:jac_inv[k, 2]}
            err[err_names[k]] = ((jac_inv[k, 0]*delfstar_err[0])**2 +
                                (jac_inv[k, 1]*delfstar_err[1])**2 +
                                (jac_inv[k, 2]*delfstar_err[2])**2)**0.5
        # reset erros to zero if they are bigger than the actual values
        if err['drho']>drho:
            err['drho']=0
        if err['grho3']>grho3:
            err['grho3']=0
        if err['phi']>phi:
            err['phi']=0
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
    soln_output['deriv'] = deriv
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


def solve_from_delfstar_bulk(delfstar, ncalc):
    # this function is used if we already have a bunch of delfstar values
    # and want to obtain the solution to bulk layers with a semiinfinite thickness
    # get film info (containing raw data plot, etc. if it exists)
    deltarho={}
    grho={}
    phi={}
    for n in ncalc:
        deltarho[n]=np.zeros(len(delfstar))
        grho[n]=np.zeros(len(delfstar))
        phi[n]=np.zeros(len(delfstar))
        for i in np.arange(len(delfstar)):
            deltarho[n][i]=deltarho_bulk(n, delfstar[i])
            grho[n][i]=grho_bulk(n, delfstar[i])
            phi[n][i]=phi_bulk(n, delfstar[i]) 
    return deltarho, grho, phi


def solve_from_delfstar(sample, parms):
    # this function is used if we already have a bunch of delfstar values
    # and want to obtain the solutions from there
    # now set the markers used for the different calculation types
    markers = {'131': '>', '133': '^', '353': '+', '355': 'x', '3': 'x'}
    colors = parms.get('colors',{1: [1, 0, 0], 3: [0, 0.5, 0], 5: [0, 0, 1]})
    calctype = parms.get('calctype', 'SLA')
    # get film info (containing raw data plot, etc. if it exists)
    sample['film']=sample.get('film',{})
    sample['samplename']= sample.get('samplename','noname')
    close_on_click_switch = parms.get('close_on_click_switch', True)
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
        sample['propfig']=propfig

    # set up the consistency check axes
    checkfig = {}
    for nh in sample['nhcalc']:
        checkfig[nh] = make_check_axes(sample, nh)
        if close_on_click_switch and not run_from_ipython():
            # when code is run with IPython don't use the event
            checkfig[nh]['figure'].canvas.mpl_connect('key_press_event',
                                                    close_on_click)
    # now do all of the calculations and plot the data
    soln_input = {'nhplot': nhplot, 'calctype':calctype}
    if 'overlayer' in sample:
        soln_input['overlayer']=sample['overlayer']
    if 'prop_guess' in sample:
        soln_input['prop_guess']=sample['prop_guess']
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
            soln_input['prop_guess']=soln['film']
            for n in nhplot:
                results[nh]['delfstar_calc'][n][i] = (
                 soln['delfstar_calc'][n])
                results[nh]['rd'][n][i] = soln['rd'][n]
            results[nh]['rh'][i] = soln['rh']
            # add actual values of delf, delg for each harmonic to the
            # solution check figure
            for n in nhplot:
                checkfig[nh]['delf_ax'].plot(xdata[i], delfstar[i][n].real/n,
                                             '+', color=colors[n])
                checkfig[nh]['delg_ax'].plot(xdata[i], delfstar[i][n].imag/n,
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
             results[nh]['delfstar_calc'][n].real/n, '-',
             color=colors[n], label='n='+str(n)))
            (checkfig[nh]['delg_ax'].plot(xdata,
             results[nh]['delfstar_calc'][n].imag/n, '-', color=colors[n],
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
        checkfig[nh]['D_ax'] = add_D_axis(checkfig[nh]['delg_ax'])
        checkfig[nh]['figure'].tight_layout()
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
        # add values of d/lam3 to the film raw data figure
        if 'rawfig' in sample['film']:
            sample['film']['dlam3_ax'].plot(xdata, results[nh]['dlam3'], '+', label=nh)
    # add legend to the the dlam3 figure and set the x axis label
    if 'rawfig' in sample['film']:
        sample['film']['dlam3_ax'].legend()
        sample['film']['dlam3_ax'].set_xlabel(sample['xlabel'])
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
    # tidy up the property figure
    cleanup_propfig(sample, parms)
    sample['checkfig'] = checkfig
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
    return propfig


def close_existing_fig(figname):
    if plt.fignum_exists(figname):
        plt.close(figname)
    return


def make_prop_axes(propfigname, xlabel):
    # set up the standard property plot
    close_existing_fig(propfigname)
    fig = plt.figure(propfigname, figsize=(9, 3))
    drho_ax = fig.add_subplot(131)
    drho_ax.set_xlabel(xlabel)
    drho_ax.set_ylabel(r'$d\rho$ ($\mu$m$\cdot$g/cm$^3$)')

    grho3_ax = fig.add_subplot(132)
    grho3_ax.set_xlabel(xlabel)
    grho3_ax.set_ylabel(r'$|G_3^*|\rho$ (Pa $\cdot$ g/cm$^3$)')

    phi_ax = fig.add_subplot(133)
    phi_ax.set_xlabel(xlabel)
    phi_ax.set_ylabel(r'$\phi$ (deg.)')

    fig.tight_layout()

    return {'figure': fig, 'drho_ax': drho_ax, 'grho3_ax': grho3_ax,
            'phi_ax': phi_ax}

    
def make_prop_axes_bulk(propfigname, xlabel):
    # set up the standard property plot
    close_existing_fig(propfigname)
    fig = plt.figure(propfigname, figsize=(9, 3))
    deltarho_ax = fig.add_subplot(131)
    deltarho_ax.set_xlabel(xlabel)
    deltarho_ax.set_ylabel(r'$\delta\rho$ ($\mu m\cdot$g/cm$^3$)')

    grho_ax = fig.add_subplot(132)
    grho_ax.set_xlabel(xlabel)
    grho_ax.set_ylabel(r'$|G_n^*|\rho$ (Pa $\cdot$ g/cm$^3$)')
    # adjust tick label format for grho_ax    
    grho_ax.ticklabel_format(axis='y', style='sci', 
       scilimits=(0,0), useMathText=True)

    phi_ax = fig.add_subplot(133)
    phi_ax.set_xlabel(xlabel)
    phi_ax.set_ylabel(r'$\phi_n$ (deg.)')

    fig.tight_layout()

    return {'figure': fig, 'deltarho_ax':deltarho_ax, 'grho_ax':grho_ax,
            'phi_ax':phi_ax}


def make_vgp_axes(vgpfigname):
    close_existing_fig(vgpfigname)
    fig = plt.figure(vgpfigname, figsize=(3, 3))
    vgp_ax = fig.add_subplot(111)
    vgp_ax.set_xlabel((r'$|G_3^*|\rho$ (Pa $\cdot$ g/cm$^3$)'))
    vgp_ax.set_ylabel(r'$\phi$ (deg.)')
    fig.tight_layout()
    return {'figure': fig, 'vgp_ax':vgp_ax}


def make_check_axes(sample, nh):
    close_existing_fig(nh + 'solution check')
    #  compare actual annd recaulated frequency and dissipation shifts.
    fig = plt.figure(nh + '_solution check_'+sample['samplename'])
    delf_ax = fig.add_subplot(221)
    delf_ax.set_xlabel(sample['xlabel'])
    delf_ax.set_ylabel(r'$\Delta f/n$ (Hz)')

    delg_ax = fig.add_subplot(222)
    delg_ax.set_xlabel(sample['xlabel'])
    delg_ax.set_ylabel(r'$\Delta \Gamma/n$ (Hz)')

    rh_ax = fig.add_subplot(223)
    rh_ax.set_xlabel(sample['xlabel'])
    rh_ax.set_ylabel(r'$r_h$')

    rd_ax = fig.add_subplot(224)
    rd_ax.set_xlabel(sample['xlabel'])
    rd_ax.set_ylabel(r'$r_d$')

    fig.tight_layout()

    return {'figure': fig, 'delf_ax': delf_ax, 'delg_ax': delg_ax,
            'rh_ax': rh_ax, 'rd_ax': rd_ax}


def delfstar_from_xlsx(directory, file):  # build delfstar dictionary from excel file
    df = pd.read_excel(directory+file, sheet_name=None, header=0)['S_channel']
    delfstar={}
    for i in np.arange(len(df)):
        delfstar[i]={}
        for n in [1, 3, 5, 7, 9]:
            if 'delf'+str(n) in df.keys():
                delfstar[i][n] = df['delf'+str(n)][i] + 1j*df['delg'+str(n)][i]
    time = df['t']  # time in seconds
    return time, delfstar


def bulk_props(delfstar):
    # get the bulk solution for grho and phi
    grho3 = (np.pi*Zq*abs(delfstar[3])/f1) ** 2
    phi = -np.degrees(2*np.arctan(delfstar[3].real /
                      delfstar[3].imag))

    return [grho3, phi]


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
