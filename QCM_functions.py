#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:19:59 2018

@author: ken
"""
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import hdf5storage
from pathlib import Path
from functools import reduce

zq = 8.84e6  # shear acoustic impedance of quartz
f1 = 5e6  # fundamental resonant frequency


def cosd(phi):  # need to define matlab-like functions that accept degrees
    return np.cos(np.deg2rad(phi))


def sind(phi):  # need to define matlab-like functions that accept degrees
    return np.sin(np.deg2rad(phi))


def tand(phi):  # need to define matlab-like functions that accept degrees
    return np.tan(np.deg2rad(phi))


def sauerbreyf(n, drho):
    return 2*n*f1 ** 2*drho/zq


def sauerbreym(n, delf):
    return delf*zq/(2*n*f1 ** 2)


def grho(n, grho3, phi):
    return grho3*(n/3) ** (phi/90)


def grho_from_dlam(n, drho, dlam, phi):
    return (drho*n*f1*cosd(phi/2)/dlam) ** 2


def D(n, drho, grho3, phi):
    return 2*np.pi*drho*n*f1*(cosd(phi/2) - 1j*sind(phi/2)) / \
        (grho(n, grho3, phi)) ** 0.5


def delfstarcalc(n, drho, grho3, phi):
    return -sauerbreyf(n, drho)*np.tan(D(n, drho, grho3, phi)) / \
        D(n, drho, grho3, phi)


def d_lamcalc(n, drho, grho3, phi):
    return drho*n*f1*cosd(phi/2)/np.sqrt(grho(n, grho3, phi))


def thin_film_gamma(n, drho, jdprime_rho):
    return 8*np.pi ** 2*n ** 3*f1 ** 4*drho ** 3*jdprime_rho / (3*zq)
    # same master expression, replacing grho3 with jdprime_rho3


def grho3(jdprime_rho3, phi):
    return sind(phi)/jdprime_rho3


def dlam(n, dlam3, phi):
    return dlam3*(int(n)/3) ** (1-phi/180)


def normdelfstar(n, dlam3, phi):
    return -np.tan(2*np.pi*dlam(n, dlam3, phi)*(1-1j*tand(phi/2))) / \
        (2*np.pi*dlam(n, dlam3, phi)*(1-1j*tand(phi/2)))


def rhcalc(nh, dlam3, phi):
    return np.real(normdelfstar(nh[0], dlam3, phi)) / \
        np.real(normdelfstar(nh[1], dlam3, phi))


def rhexp(nh, delfstar):
    n1 = int(nh[0])
    n2 = int(nh[1])
    return (n2/n1)*np.real(delfstar[n1])/np.real(delfstar[n2])


def rdcalc(nh, dlam3, phi):
    return -np.imag(normdelfstar(nh[2], dlam3, phi)) / \
        np.real(normdelfstar(nh[2], dlam3, phi))


def rdexp(nh, delfstar):
    n3 = int(nh[2])
    return -np.imag(delfstar[n3])/np.real(delfstar[n3])


def solvefunction(soln_input):
    nh = soln_input['nh']
    nhplot = soln_input['nhplot']
    delfstar = soln_input['delfstar']
    err = soln_input['err']
    soln1_guess = soln_input['soln1_guess']
    n1 = int(nh[0])
    n2 = int(nh[1])
    n3 = int(nh[2])

    # first pass at solution comes from rh and rd
    rd_exp = -np.imag(delfstar[n3])/np.real(delfstar[n3])
    rh_exp = (n2/n1)*np.real(delfstar[n1])/np.real(delfstar[n2])
    lb = [0, 0]  # lower bounds on dlam3 and phi
    ub = [1, 90]  # upper bonds on dlam3 and phi

    def ftosolve(x):
        return [rhcalc(nh, x[0], x[1])-rh_exp, rdcalc(nh, x[0], x[1])-rd_exp]

    soln1 = least_squares(ftosolve, soln1_guess, bounds=(lb, ub))

    # put the properties into dictionaries that will include the first
    # and second solutions
    dlam3 = {'soln1': soln1['x'][0]}
    phi = {'soln1': soln1['x'][1]}
    drho = {'soln1': (sauerbreym(n1, np.real(delfstar[n1])) /
            np.real(normdelfstar(n1, dlam3['soln1'], phi['soln1'])))}
    grho3 = {'soln1': grho_from_dlam(3, drho['soln1'],
                                     dlam3['soln1'], phi['soln1'])}

    # now define guesses for drho, grho3, phi
    x0 = np.array([drho['soln1'], grho3['soln1'], phi['soln1']])

    # we solve it again to get the Jacobian with respect to our actual
    # input variables - this is helpfulf for the error analysis
    def ftosolve2(x):
        return ([np.real(delfstar[n1]) -
                np.real(delfstarcalc(n1, x[0], x[1], x[2])),
                np.real(delfstar[n2]) -
                np.real(delfstarcalc(n2, x[0], x[1], x[2])),
                np.imag(delfstar[n3]) -
                np.imag(delfstarcalc(n3, x[0], x[1], x[2]))])

    soln2 = least_squares(ftosolve2, x0)
    drho['soln2'] = soln2['x'][0]
    grho3['soln2'] = soln2['x'][1]
    phi['soln2'] = soln2['x'][2]
    dlam3['soln2'] = d_lamcalc(3, drho['soln2'], grho3['soln2'], phi['soln2'])

    delfstar_calc = {}
    for n in nhplot:
        delfstar_calc[n] = delfstarcalc(n, drho['soln2'], grho3['soln2'],
                                        phi['soln2'])

#    errin = [delfn1err, delfn2err, delgn3err]
#    Jinv = inv(J)
#    errout = (Jinv ** 2*errin ** 2) ** 0.5
#    err['drho'] = errout[1]
#    err['grho3'] = errout[2]
#    err['phi'] = errout[3]
#    err['dlam3'] = NaN
#    err['jdprime_rho'] = NaN

    soln_output = {'drho': drho, 'grho3': grho3, 'phi': phi, 'dlam3': dlam3,
                   'delfstar_calc': delfstarcalc}

    return soln_output


def QCManalyze(sample):
    # read in the optional inputs, assigning default values if not assigned
    nhplot = sample.get('nhplot', [1, 3, 5])
    firstline = sample.get('firstline', 0)
    baretrange = sample.get('baretrange', [0, 0])
    filmtrange = sample.get('filmtrange', [0, 0])
    sample['xlabel'] = sample.get('xlabel',  't (min.)')
    Temp = np.array(sample.get('Temp', [22]))
    sample['nhcalc'] = sample.get('nhcalc', [])

    # now read in the variables that must exist in the sample dictionary
    filmfile = sample['datadir'] + sample['filmfile'] + '.mat'
    filmdata = hdf5storage.loadmat(filmfile)

    # build the relevant filenames
    barefile = sample['datadir'] + sample['barefile'] + '.mat'
    film_idx_file = Path(sample['datadir']+sample['filmfile']+'_film_idx.txt')
    baredata = hdf5storage.loadmat(barefile)
    bare_idx_file = Path(sample['datadir']+sample['filmfile']+'_bare_idx.txt')

    # define uncertainties in freqency, dissipation
    err = {}
    err['f1'] = [50, 3e-3]
    err['f3'] = [50, 3e-3]
    err['f5'] = [50, 3e-3]

    err['g1'] = [10, 3e-3]
    err['g3'] = [10, 3e-3]
    err['g5'] = [10, 3e-3]

    # set the color dictionary for the different harmonics
    colors = {1: [1, 0, 0], 3: [0, 0.5, 0], 5: [0, 0, 1]}

    # now set the markers used for the different calculation types
    markers = {'131': '>', '133': '^', '353': '+', '355': 'x'}

    # initialize the dictionary we'll use to keep track of the points to plot
    idx = {}

    # read in the frequency data
    barefreq = baredata['abs_freq'][firstline:, 0:7]
    filmfreq = filmdata['abs_freq'][firstline:, 0:7]

    # get rid of all the rows that don't have any data
    barefreq = barefreq[~np.isnan(barefreq).all(axis=1)]
    filmfreq = filmfreq[~np.isnan(filmfreq).all(axis=1)]

    # reference frequencies are the first data points for the bare crystal data
    freqref = barefreq[0, :]

    # extract frequency information
    bare_t = barefreq[:, 0]
    film_t = filmfreq[:, 0]
    baref = {}
    filmf = {}
    bareg = {}
    filmg = {}

    for n in nhplot:
        baref[n] = barefreq[:, n] - freqref[n]
        filmf[n] = filmfreq[:, n] - freqref[n]
        bareg[n] = barefreq[:, n+1]
        filmg[n] = filmfreq[:, n+1]

    #  find all the time points between specified by timerange
    #  if the min and max values for the time range are equal, we use
    #    all the points
    idx['b_all'] = idx_in_range(bare_t, baretrange)
    idx['f_all'] = idx_in_range(film_t, filmtrange)

    # figure out how man total points we have for bare and film data
    n_all = {'b': idx['b_all'].shape[0], 'f': idx['f_all'].shape[0]}

    # if there is only one temperature, than we use time as the x axis, using
    # up to ten user-selected points
    if Temp.shape[0] == 1:
        nx = min(10, n_all['b'], n_all['f'])
    else:
        nx = Temp.shape[0]

    # rewrite nhplot to account for the fact that data may not exist for all
    # of the harmonics
    n_exist_bare = np.array([]).astype(int)
    n_exist_film = np.array([]).astype(int)
    for n in nhplot:
        if not all(np.isnan(baref[n])):
            n_exist_bare = np.append(n_exist_bare, n)
        if not all(np.isnan(filmf[n])):
            n_exist_film = np.append(n_exist_film, n)

    # only include harmonics were both film and bare data exist
    nhplot = reduce(np.intersect1d, (nhplot, n_exist_bare, n_exist_film))

    # make the axes for the raw data
    rawfig = make_raw_axes(sample)

    # plot the raw data
    for n in nhplot:
        (rawfig['baref_ax'].plot(bare_t[idx['b_all']], baref[n][idx['b_all']]
         / n, color=colors[n], label='n='+str(n)))
        (rawfig['bareg_ax'].plot(bare_t[idx['b_all']], bareg[n][idx['b_all']],
         color=colors[n], label='n='+str(n)))
        (rawfig['filmf_ax'].plot(film_t[idx['f_all']], filmf[n][idx['f_all']]
         / n, color=colors[n], label='n='+str(n)))
        (rawfig['filmg_ax'].plot(film_t[idx['f_all']], filmg[n][idx['f_all']],
         color=colors[n], label='n='+str(n)))

    # add the legends
    rawfig['baref_ax'].legend()
    rawfig['bareg_ax'].legend()
    rawfig['filmf_ax'].legend()
    rawfig['filmg_ax'].legend()

    # pick the points that we want to analyze
    idx['b'] = pickpoints(Temp, nx, idx['b_all'],
                          bare_t, bare_idx_file)
    idx['f'] = pickpoints(Temp, nx, idx['f_all'],
                          film_t, film_idx_file)

    # now add these points to the plots
    rawplots = {}
    for n in nhplot:
        rawplots[str(n)+'1'] = (rawfig['baref_ax'].plot(bare_t[idx['b']],
                                baref[n][idx['b']]/n, 'x', color=colors[n]))
        rawplots[str(n)+'1'] = (rawfig['bareg_ax'].plot(bare_t[idx['b']],
                                bareg[n][idx['b']], 'x', color=colors[n]))
        rawplots[str(n)+'1'] = (rawfig['filmf_ax'].plot(film_t[idx['f']],
                                filmf[n][idx['f']]/n, 'x', color=colors[n]))
        rawplots[str(n)+'1'] = (rawfig['filmg_ax'].plot(film_t[idx['f']],
                                filmg[n][idx['f']], 'x', color=colors[n]))

    # adjust nhcalc to account to only include calculations for for which
    # the data exist
    sample['nhcalc'] = nhcalc_in_nhplot(sample['nhcalc'], nhplot)

    # there is nothing left to do if nhcalc has no values
    if not(sample['nhcalc']):
        return

    # now calculate the frequency and dissipation shifts
    delfstar = {}
    for i in np.arange(nx):
        delfstar[i] = {}
        for n in nhplot:
            delfstar[i][n] = (filmf[n][idx['f'][i]] -
                              baref[n][idx['b'][i]] + 1j *
                              (filmg[n][idx['f'][i]] - bareg[n][idx['b'][i]]))

    # set up the property axes
    propfig = make_prop_axes(sample)
    checkfig = {}
    for nh in sample['nhcalc']:
        checkfig[nh] = make_check_axes(sample, nh)

    # set the appropriate value for xdata
    if Temp.shape[0] == 1:
        xdata = film_t[idx['f']]
    else:
        sample['xlabel'] = r'$T \: ^\circ$C'
        xdata = Temp

    # now do all of the calculations and plot the data
    soln_input = {'err': err, 'nhplot': nhplot}
    results = {}


    for nh in sample['nhcalc']:
        soln_input['soln1_guess'] = [0.05, 1]
        results[nh] = {'drho': np.zeros(nx), 'grho3': np.zeros(nx),
                       'phi': np.zeros(nx), 'dlam3': np.zeros(nx)}
        for i in np.arange(nx):
            # obtain the solution for the properties
            soln_input['nh'] = nh
            soln_input['delfstar'] = delfstar[i]
            soln = solvefunction(soln_input)
            results[nh]['drho'][i] = soln['drho']['soln2']
            results[nh]['grho3'][i] = soln['grho3']['soln2']
            results[nh]['phi'][i] = soln['phi']['soln2']
            results[nh]['dlam3'][i] = soln['dlam3']['soln2']

            # add actual values of delf, delg for each harmonic to the
            # solution check figure
            for n in nhplot:
                checkfig[nh]['delf_ax'].plot(xdata[i], np.real(delfstar[i][n])
                                             / n, '+', color=colors[n])
                checkfig[nh]['delg_ax'].plot(xdata[i], np.imag(delfstar[i][n]),
                                             '+', color=colors[n])
            # add rh, rd to solution check figure
            checkfig[nh]['rh_ax'].plot(xdata[i], rhexp(nh, delfstar[i]),
                                       '+', color=colors[n])
            checkfig[nh]['rd_ax'].plot(xdata[i], rdexp(nh, delfstar[i]),
                                       '+', color=colors[n])

        # add the calculated values of rh, rd to the solution check figures
        checkfig[nh]['rh_ax'].plot(xdata, rhcalc(nh, results[nh]['dlam3'],
                                   results[nh]['phi']), '-')
        checkfig[nh]['rd_ax'].plot(xdata, rdcalc(nh, results[nh]['dlam3'],
                                   results[nh]['phi']), '-')

        # add calculated delf and delg to solution check figures
        for n in nhplot:
            (checkfig[nh]['delf_ax'].plot(xdata,  np.real(delfstarcalc(n,
             results[nh]['drho'], results[nh]['grho3'],
             results[nh]['phi'])) / n, '-', color=colors[n],
             label='n='+str(n)))
            (checkfig[nh]['delg_ax'].plot(xdata,  np.imag(delfstarcalc(n,
             results[nh]['drho'], results[nh]['grho3'],
             results[nh]['phi'])), '-', color=colors[n],
             label='n='+str(n)))

        # add legen to the solution check figures
        checkfig[nh]['delf_ax'].legend()
        checkfig[nh]['delg_ax'].legend()

        # tidy up the solution check figure
        checkfig[nh]['figure'].tight_layout()
        checkfig[nh]['figure'].savefig(sample['datadir'] + sample['filmfile'] +
                                       '_'+nh+'.png')

        # add property data to the property figure
        propfig['drho_ax'].plot(xdata, 1000*results[nh]['drho'],
                                marker=markers[nh], label=nh)
        propfig['grho_ax'].plot(xdata, results[nh]['grho3']/1000,
                                marker=markers[nh], label=nh)
        propfig['phi_ax'].plot(xdata, results[nh]['phi'],
                               marker=markers[nh], label=nh)

    # add legends to the property figure
    propfig['drho_ax'].legend()
    propfig['grho_ax'].legend()
    propfig['phi_ax'].legend()

    # tidy up the raw data and property figures
    rawfig['figure'].tight_layout()
    rawfig['figure'].savefig(sample['datadir']+sample['filmfile']+'_raw.png')
    propfig['figure'].tight_layout()
    propfig['figure'].savefig(sample['datadir']+sample['filmfile']+'_prop.png')


def idx_in_range(t, t_range):
    if t_range[0] == t_range[1]:
        idx = np.arange(t.shape[0]).astype(int)
    else:
        idx = np.where((t > t_range[0]) &
                       (t < t_range[1]))[0]
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


def pickpoints(Temp, nx, idx_in, t_in, idx_file):
    idx_out = np.array([], dtype=int)
    if Temp.shape[0] == 1:
        t = np.linspace(min(t_in), max(t_in), nx)
        for n in np.arange(nx):
            idx_out = np.append(idx_out, (np.abs(t[n] - t_in)).argmin())
        idx_out = np.asarray(idx_out)

    elif idx_file.is_file():
        idx_out = np.loadtxt(idx_file, dtype=int)
    else:
        pts = plt.ginput(nx)
        pts = np.array(pts)[:, 0]
        for n in np.arange(nx):
            idx_out = np.append(idx_out, (np.abs(pts[n] - t_in)).argmin())
        idx_out = np.asarray(idx_out)
        np.savetxt(idx_file, idx_out, fmt='%4i')

    return idx_out


def make_prop_axes(sample):
    # set up the property plot
    fig = plt.figure('prop', figsize=(9, 3))
    drho_ax = fig.add_subplot(131)
    drho_ax.set_xlabel(sample['xlabel'])
    drho_ax.set_ylabel(r'$d\rho\: (g/m^2)$')

    grho_ax = fig.add_subplot(132)
    grho_ax.set_xlabel(sample['xlabel'])
    grho_ax.set_ylabel(r'$|G_3^*|\rho \: (Pa \cdot g/cm^3)$')

    phi_ax = fig.add_subplot(133)
    phi_ax.set_xlabel(sample['xlabel'])
    phi_ax.set_ylabel(r'$\phi$ (deg.)')

    fig.tight_layout()

    return {'figure': fig, 'drho_ax': drho_ax, 'grho_ax': grho_ax,
            'phi_ax': phi_ax}


def make_raw_axes(sample):
    # We make a figure that includes the bare crystal data and the film data
    fig = plt.figure('Raw')

    baref_ax = fig.add_subplot(221)
    baref_ax.set_xlabel(sample['xlabel'])
    baref_ax.set_ylabel(r'$\Delta f_n/n$ (Hz)')
    baref_ax.set_title('bare crystal')

    bareg_ax = fig.add_subplot(222)
    bareg_ax.set_xlabel(sample['xlabel'])
    bareg_ax.set_ylabel(r'$\Gamma$ (Hz)')
    bareg_ax.set_title('bare crystal')

    filmf_ax = fig.add_subplot(223)
    filmf_ax.set_xlabel(sample['xlabel'])
    filmf_ax.set_ylabel(r'$\Delta f_n/n$ (Hz)')
    filmf_ax.set_title('film')

    filmg_ax = fig.add_subplot(224)
    filmg_ax.set_xlabel(sample['xlabel'])
    filmg_ax.set_ylabel(r'$\Gamma$ (Hz)')
    filmg_ax.set_title('film')

    fig.tight_layout()

    return {'figure': fig, 'baref_ax': baref_ax, 'bareg_ax': bareg_ax,
            'filmf_ax': filmf_ax, 'filmg_ax': filmg_ax}


def make_check_axes(sample, nh):
    #  compare actual annd recaulated frequency and dissipation shifts.
    fig = plt.figure(nh + 'solution check')
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
