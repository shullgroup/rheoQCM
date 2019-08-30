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

try:
  from kww import kwwc, kwws
  # kwwc returns: integral from 0 to +infinity dt cos(omega*t) exp(-t^beta)
  # kwws returns: integral from 0 to +infinity dt sin(omega*t) exp(-t^beta)
except ImportError:
  pass

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


def rhcalc(calc, dlam3, phi):
    return normdelfstar(calc[0], dlam3, phi).real / \
        normdelfstar(calc[1], dlam3, phi).real


def rh_from_delfstar(calc, delfstar):
    # calc here is the calc string (i.e., '353')
    n1 = int(calc[0])
    n2 = int(calc[1])
    return (n2/n1)*delfstar[n1].real/delfstar[n2].real


def rdcalc(calc, dlam3, phi):
    return -(normdelfstar(calc[2], dlam3, phi).imag / \
        normdelfstar(calc[2], dlam3, phi).real)


def rd_from_delfstar(n, delfstar):
    # dissipation ratio calculated for the relevant harmonic
    return -delfstar[n].imag/delfstar[n].real

    
def bulk_props(delfstar):
    # get the bulk solution for grho and phi
    grho3 = (np.pi*Zq*abs(delfstar[3])/f1) ** 2
    phi = -np.degrees(2*np.arctan(delfstar[3].real /
                      delfstar[3].imag))
    return [grho3, min(phi, 90)]


def thinfilm_guess(delfstar, calc):
    # really a placeholder function until we develop a more creative strategy
    # for estimating the starting point
    n1 = int(calc[0])
    n2 = int(calc[1])
    n3 = int(calc[2])

    rd_exp = -delfstar[n3].imag/delfstar[n3].real
    rh_exp = (n2/n1)*delfstar[n1].real/delfstar[n2].real
    lb = np.array([0, 0])  # lower bounds on dlam3 and phi
    ub = np.array([5, 90])  # upper bonds on dlam3 and phi

    # we solve the problem initially using the harmonic and dissipation
    # ratios, using the small load approximation
    # we also neglect the overlayer in this first calculation
    def ftosolve(x):
        return [rhcalc(calc, x[0], x[1])-rh_exp, rdcalc(calc, x[0], x[1])-rd_exp]
    guess = [0.05, 5]
    soln = optimize.least_squares(ftosolve, guess, bounds=(lb, ub))

    dlam3 = soln['x'][0]
    phi = soln['x'][1]
    drho = (sauerbreym(n1, delfstar[n1].real) /
            normdelfstar(n1, dlam3, phi).real)
    grho3 = grho_from_dlam(3, drho, dlam3, phi)
    return drho, grho3, phi


def solve_for_props(delfstar, **kwargs):
    # solve the QCM equations to determine the properties

    df = delfstar.T  # transpose the input dataframe
    if 'overlayer' in kwargs.keys():
        layers={'overlayer':kwargs['overlayer']}
    else:
        layers={}

    nplot = kwargs.get('nplot', [1, 3, 5])
    calctype = kwargs.get('calctype', 'SLA')
    filmtype = kwargs.get('filmtype', 'thin')

    calc = kwargs.get('calc') # specify the values used in the n1:n2,n3 calculation
    n1 = int(calc[0]); n2 = int(calc[1]); n3 = int(calc[2])

    # set upper and lower bounds
    lb = kwargs.get('lb', [1e5, 0, 0])
    ub = kwargs.get('ub', [1e13, 90, 3e-2])
    lb = np.array(lb)  # lower bounds drho, grho3, phi
    ub = np.array(ub)  # upper bounds drho, grho3, phi


    # we use the first data point to get an initial guess
    if 'guess' in kwargs.keys():
        guess=kwargs['guess']
        drho, grho3, phi = guess['drho'], guess['grho3'], guess['phi']
    elif filmtype=='bulk':
        n3=n1 # bulk solution uses delta f and delta gamma from same harmonic
        grho3, phi = bulk_props(df[0])
        drho = np.inf
        lb = lb[0:2]
        ub = ub[0:2]
    else:
        drho, grho3, phi = thinfilm_guess(df[0], calc)

    # set up the initial guess
    if filmtype == 'bulk':
        x0 = np.array([grho3, phi])
    else:
        x0 = np.array([grho3, phi, drho])

    # add extra rows to the dataframe for all the calculated values
    newrows = []
    for n in nplot:
        newrows = newrows + ['df_calc'+str(n)]
        
    newrows = newrows+['drho', 'drho_err', 'grho3', 'grho3_err', 'phi', 'phi_err',
                       'dlam3',  'delfstar_err', 'deriv']

    data = {}
    for element in newrows:
        data[element] = {}
        for i in np.arange(df.shape[1]):
            data[element][i] = {}

    newrows = pd.DataFrame(columns=df.columns, index=newrows)
    df=df.append(newrows)

    # obtain the solution, using either the SLA or LL methods
    for i in df.columns:
        if filmtype != 'bulk':
            def ftosolve(x):
                layers['film'] = {'grho3':x[0], 'phi':x[1], 'drho':x[2]}
                return ([df[i][n1].real-calc_delfstar(n1, layers, calctype).real,
                         df[i][n2].real-calc_delfstar(n2, layers, calctype).real,
                         df[i][n3].imag-calc_delfstar(n3, layers, calctype).imag])
        else:
            def ftosolve(x):
                layers['film'] = {'drho':np.inf, 'grho3':x[0], 'phi':x[1]}
                return ([df[i][n1].real-calc_delfstar(n1, layers, calctype).real,
                         df[i][n1].imag-calc_delfstar(n1, layers, calctype).imag])

        # initialize the output uncertainties
        err = {}
        soln = optimize.least_squares(ftosolve, x0, bounds=(lb, ub))
        grho3 = soln['x'][0]
        phi = soln['x'][1]

        if filmtype != 'bulk':
            drho = soln['x'][2]

        layers['film'] = {'drho':drho, 'grho3':grho3, 'phi':phi}
        dlam3 = calc_dlam(3, layers['film'], calctype)
        jac = soln['jac']

        try:
            deriv = np.linalg.inv(jac)
        except:
            deriv = np.zeros([len(x0),len(x0)])

        # put the input uncertainties into a 3 element vector
        delfstar_err = np.zeros(3)
        delfstar_err[0] = fstar_err_calc(n1, df[i][n1], layers).real
        delfstar_err[1] = fstar_err_calc(n2, df[i][n2], layers).real
        delfstar_err[2] = fstar_err_calc(n3, df[i][n3], layers).imag

        # determine error from Jacobian
        err = {}
        for k in np.arange(len(x0)):
            for m in np.arange(len(x0)):
                err[k]=0
                err[k] = err[k]+(deriv[k, m]*delfstar_err[m])**2
            err[k] = np.sqrt(err[k])

        if filmtype == 'bulk':
            err[2]=np.nan

        # now back calculate delfstar from the solution
        delfstar_calc = {}

        for n in nplot:
            delfstar_calc[n] = calc_delfstar(n, layers, calctype)

        for n in nplot:
            data['df_calc'+str(n)][i]=round(delfstar_calc[n],1)

        data['grho3'][i] = grho3
        data['phi'][i] = phi
        data['drho'][i] = drho

        data['grho3_err'][i] = err[0]
        data['phi_err'][i] = err[1]
        data['drho_err'][i] = err[2]

        data['dlam3'][i] = dlam3
        data['delfstar_err'][i] = delfstar_err
        data['deriv'][i] = deriv

        # set up the initial guess
        if filmtype == 'bulk':
            x0 = np.array([grho3, phi])
        else:
            x0 = np.array([grho3, phi, drho])

    # add these calculated values to existing dataframe
    new_info = pd.DataFrame(data)
    new_df = pd.concat([delfstar, new_info], axis=1)
    return new_df


def make_prop_axes(**kwargs):
    # set up the standard property plot
    filmtype = kwargs.get('filmtype','thin')
    num = kwargs.get('num','property fig')

    if filmtype != 'bulk':
        fig, ax = plt.subplots(1,3, figsize=(9, 3), num=num)
        ax[2].set_ylabel(r'$d\rho$ ($\mu$m$\cdot$g/cm$^3$)')
    else:
        fig, ax = plt.subplots(1,2, figsize=(6, 3), num=num)

    ax[0].set_ylabel(r'$|G_3^*|\rho$ (Pa $\cdot$ g/cm$^3$)')
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    ax[1].set_ylabel(r'$\phi$ (deg.)')

    # set xlabel to 'index' by default
    for i in np.arange(len(ax)):
        ax[i].set_xlabel('index')
    fig.tight_layout()
    return fig, ax


def make_delf_axes(**kwargs):
    num = kwargs.get('num','delf fig')
    fig, ax = plt.subplots(1,2, figsize=(6,3), num=num)
    for i in [0, 1]:
        ax[i].set_xlabel(r'index')
    ax[0].set_ylabel(r'$\Delta f/n$ (kHz)')
    ax[1].set_ylabel(r'$\Delta \Gamma /n$ (kHz)')
    fig.tight_layout()
    return fig, ax


def make_vgp_axes():
    fig, ax = plt.subplots(1,1, figsize=(3, 3))
    ax.set_xlabel((r'$|G_3^*|\rho$ (Pa $\cdot$ g/cm$^3$)'))
    ax.set_ylabel(r'$\phi$ (deg.)')
    fig.tight_layout()
    return fig, ax


def delfstar_from_xlsx(directory, file, **kwargs):  # delfstar datafrome from excel file
    # exclude all rows where the sindicated harmonics are not marked
    restrict_to_marked = kwargs.get('restrict_to_marked',[3])
    nvals = kwargs.get('nvals',[1,3,5])

    df = pd.read_excel(directory+file, sheet_name=None, header=0)['S_channel']

    df['keep_row']=1  # keep all rows we are told to check for specific marks
    for n in restrict_to_marked:
        df['keep_row'] = df['keep_row']*df['mark'+str(n)]

    # delete rows we don't want to keep
    df = df[df.keep_row==1] # Delete all rows that are not appropriately marked

    # add each of the values of delfstar
    for n in nvals:
        df[n] = df['delf'+str(n)] + 1j*df['delg'+str(n)].round(1)

    df = df.filter(items=['t']+['temp']+nvals)
    df = df.reset_index()
    return df

def gstar_maxwell(wtau):  # Maxwell element
    return 1j*wtau/(1+1j*wtau)


def gstar_kww_single(wtau, beta):  # Transform of the KWW function
    return wtau*(kwws(wtau, beta)+1j*kwwc(wtau, beta)) 
gstar_kww = np.vectorize(gstar_kww_single)


def springpot(w, sp_parms):
    # this function supports a combination of different springpot elments
    # combined in series, and then in parallel.  For example, if type is
    # [1,2,3],  there are three branches
    # in parallel with one another:  the first one is element 1, the
    # second one is a series comination of eleents 2 and 3, and the third
    # one is a series combination of 4, 5 and 6.
    
    # specify which elements are kww or Maxwell elements
    kww = sp_parms.get('kww',[])
    maxwell = sp_parms.get('maxwell',[])
    
    # beta doesn't exist for maxwell element
    beta = sp_parms.get('beta',[0]) # 1 be default indicates mawell elment
    # type is a build in function name. we can try to use other name
    sp_type = sp_parms.get('type',[1]) # single element by default
    
    # make values numpy arrays if they aren't already
    tau = np.asarray(sp_parms['tau']).reshape(1, -1)[0,:]
    beta = np.asarray(beta).reshape(1, -1)[0,:]
    g0 = np.asarray(sp_parms['g0']).reshape(1, -1)[0,:]
    sp_type = np.asarray(sp_type).reshape(1, -1)[0,:]
    
    nw = len(w)  # number of frequencies
    n_br = len(sp_type)  # number of series branches
    n_sp = sp_type.sum()  # number of springpot elements
    sp_comp = np.empty((nw, n_sp), dtype=np.complex)  # element compliance
    br_g = np.empty((nw, n_br), dtype=np.complex)  # branch stiffness

    for i in np.arange(n_sp):
        if i in maxwell:  # Maxwell element
            sp_comp[:, i] = 1/(g0[i]*gstar_maxwell(w*tau[i]))            
        elif i in kww:  #  kww (stretched exponential) elment
            sp_comp[:, i] = 1/(g0[i]*gstar_kww(w*tau[i], beta[i]))
        else:  # power law springpot element
            sp_comp[:, i] = 1/(g0[i]*(1j*w*tau[i]) ** beta[i])

    sp_vec = np.append(0, type.cumsum())
    for i in np.arange(n_br):
        sp_i = np.arange(sp_vec[i], sp_vec[i+1])
        br_g[:, i] = 1/sp_comp[:, sp_i].sum(1)

    return br_g.sum(1)


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