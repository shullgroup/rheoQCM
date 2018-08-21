'''
class for fitting of G and B
'''
import numpy as np
from scipy.optimize import leastsq
from lmfit import Model, Minimizer, minimize, Parameters, fit_report, printfuncs
from lmfit.models import ConstantModel

class GBFitting:
    def __init__(self):
        # self.gmodel = Model(self.fun_G)
        # self.bmodel = Model(self.fun_B)
        self.gmodel, self.bmodel, self.gpars, self.bpars = make_model(n=1)



def fun_G(x, amp, cen, wid, phi):
    ''' 
    function of relation between frequency (f) and conductance (G) 
    '''
    return amp * (4 * wid**2 * x**2 * np.cos(phi) - 2 * wid * x * np.sin(phi) * (cen**2 - x**2)) / (4 * wid**2 * x**2 + (cen**2 -x**2)**2)

def fun_B(x, amp, cen, wid, phi):
    ''' 
    function of relation between frequency (f) and susceptance (B) 
    '''
    return amp * (4 * wid**2 * x**2 * np.sin(phi) + 2 * wid * x * np.cos(phi) * (cen**2 - x**2)) / (4 * wid**2 * x**2 + (cen**2 -x**2)**2)

def make_model(n=1):
    '''
    make complex model for multiple peaks
    input:
    n:    number of peaks
    '''
    gmod = ConstantModel(prefix='g_')
    gpars = gmod.make_params(c=0)
    bmod = ConstantModel(prefix='b_')
    bpars = bmod.make_params(c=0)

    for i in np.arange(1, n+1):
        # gmod and bmod sharing the same varible so use the same prefix
        gmod_i = Model(fun_G, prefix='p'+str(i)+'_')
        gpars.update(gmod_i.make_params())
        gpars['p'+str(i)+'_amp'].set(0, min=0)
        gpars['p'+str(i)+'_cen'].set()
        gpars['p'+str(i)+'_wid'].set(1, min=1)
        gpars['p'+str(i)+'_phi'].set(0, min=-np.pi/2, max=np.pi/2)
        bmod_i = Model(fun_B, prefix='p'+str(i)+'_')
        bpars.update(bmod_i.make_params())
        bpars['p'+str(i)+'_amp'].set(0, min=0)
        bpars['p'+str(i)+'_cen'].set()
        bpars['p'+str(i)+'_wid'].set(1, min=1)
        bpars['p'+str(i)+'_phi'].set(0, min=-np.pi/2, max=np.pi/2)
        gmod += gmod_i
        bmod += bmod_i
    
    return gmod, bmod, gpars, bpars

def res_GB(params, f=None, G=None, B=None):
    '''
    function of residual of both G and B
    '''
    residual_G = G - gmod.eval(params, x=f)
    residual_B = B - bmod.eval(params, x=f)
    return np.concatenate((residual_G, residual_B))

def set_params(f, G, B, n):
    ''' set the parameters for fitting '''

    params = Parameters()
    for i in np.arange(1, n+1):
        params.add(
            'p'+str(i)+'_amp',              # amplitude (G)
            value=np.max(G) - np.min(G),    # init: peak height
            min=0,                          # lb
            max=np.inf,                     # ub
        )
        params.add(
            'p'+str(i)+'_cen',              # center 
            value=np.mean(f),               # init: average f
            min=np.min(f),                  # lb: assume peak is in the range of f
            max=np.max(f),                  # ub: assume peak is in the range of f
        )
        params.add(
            'p'+str(i)+'_wid',                  # width (fwhm)
            value=(np.max(f) - np.min(f)) / 2,  # init: half range
            min=1,                              # lb
            max=np.max(f) - np.min(f),          # ub: assume peak is in the range of f
        )
        params.add(
            'p'+str(i)+'_phi',              # phase shift
            value=0,                        # init value: peak height
            min=-np.pi / 2,                 # lb
            max=np.pi / 2,                  # ub
        )
      
    params.add(
        'g_c',              # initialize G_offset
        value=np.min(G),    # init G_offset = mean(G)
        # min=-np.inf,        # lb
        # max=np.max(G)/2,    # ub
    )        
    params.add(
        'b_c',              # initialize B_offset
        value=np.mean(B),   # init B_offset = mean(B)
        # min=np.min(B)/2,    # lb
        # max=np.min(B)/2,    # ub
    )
    return params

def minimize_GB(residual, f, G, B, n):
    '''
    use leasesq to fit
    '''
    # we creat a new set of params (gpars and dpars set in make_model() are for the case if we want to use them seperately)
    # refine params with data
    params = set_params(f, G, B, n)
    # minimize with leastsq
    # mini = Minimizer(residual, params, fcn_args=(f, G, B))
    # result = mini.leastsq(xtol=1.e-10, ftol=1.e-10)
    result = minimize(residual, params, method='leastsq', args=(f, G, B), xtol=1.e-10, ftol=1.e-10)

    print(fit_report(result)) 
    printfuncs.report_fit(result.params)
    result.params.pretty_print()
    print('chisqr', result.chisqr)
    print('redchi', result.redchi)
    print('nfev', result.nfev)
    print('success', result.success)
    print('message', result.message)
    return result

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    try:
        from AccessMyVNA_dummy import AccessMyVNA
    except:
        from Modules.AccessMyVNA_dummy import AccessMyVNA 

    gbfitting = GBFitting()
    # print(gbfitting.gmodel.param_names)
    # print(gbfitting.bmodel.param_names)
    # print(gbfitting.gpars)
    # print(gbfitting.bpars)

    gmod = gbfitting.gmodel
    bmod = gbfitting.bmodel
    accvna = AccessMyVNA()
    _, f, G = accvna.GetScanData(nWhata=-1, nWhatb=15)
    _, _, B = accvna.GetScanData(nWhata=-1, nWhatb=16)
    # G = G * 1e9
    # B = B * 1e9
    params = set_params(f, G, B, 1)
    gmod.eval(params, x=f)

    result = minimize_GB(res_GB, f, G, B, 1)

    fig = plt.figure()
    plt.plot(f, G, 'bo')
    plt.plot(f, gmod.eval(result.params, x=f), 'k--')
    plt.twinx()
    plt.plot(f, B, 'go')
    plt.plot(f, bmod.eval(result.params, x=f), 'k--')
    plt.show()
    exit(0)


    result = gmod.fit(G, params=params, x=f)
    print(result.fit_report())

    plot_components = False
    # plot results
    fig = plt.figure()
    plt.plot(f, G, 'bo')
    if plot_components:
        # generate components
        comps = result.eval_components(x=f)
        plt.plot(f, 10*comps['jump'], 'k--')
        plt.plot(f, 10*comps['gaussian'], 'r-')
    else:
        plt.plot(f, result.init_fit, 'k--')
        plt.plot(f, result.best_fit, 'r-')

    result = bmod.fit(B, params=params, x=f)
    print(result.fit_report())

    plot_components = False
    # plot results
    fig = plt.figure()
    plt.plot(f, B, 'bo')
    if plot_components:
        # generate components
        comps = result.eval_components(x=f)
        plt.plot(f, 10*comps['jump'], 'k--')
        plt.plot(f, 10*comps['gaussian'], 'r-')
    else:
        plt.plot(f, result.init_fit, 'k--')
        plt.plot(f, result.best_fit, 'r-')
    plt.show()