'''
class for fitting of G and B
'''
import numpy as np
from lmfit import Model, Minimizer, minimize, Parameters, fit_report, printfuncs
from lmfit.models import ConstantModel

class GBFitting:
    def __init__(self):
        # self.gmodel = Model(self.fun_G)
        # self.bmodel = Model(self.fun_B)
        self.gmodel, self.bmodel, self.gpars, self.bpars = make_models_pars(n=1)



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

def make_gmod(n):
    '''
    make complex model of G (w/o params) for multiple (n) peaks
    input:
    n:    number of peaks
    '''
    gmod = ConstantModel(prefix='g_')
    for i in np.arange(1, n+1):
        gmod_i = Model(fun_G, prefix='p'+str(i)+'_')
        gmod += gmod_i
    return gmod

def make_bmod(n):
    '''
    make complex model of B (w/o params) for multiple (n) peaks
    input:
    n:    number of peaks
    '''
    bmod = ConstantModel(prefix='g_')
    for i in np.arange(1, n+1):
        bmod_i = Model(fun_B, prefix='p'+str(i)+'_')
        bmod += bmod_i
    return bmod

def make_models(n=1):
    '''
    make complex model for multiple peaks
    input:
    n:    number of peaks
    '''
    gmod = ConstantModel(prefix='g_')
    bmod = ConstantModel(prefix='b_')

    for i in np.arange(1, n+1):
        # gmod and bmod sharing the same varible so use the same prefix
        gmod_i = Model(fun_G, prefix='p'+str(i)+'_')
        bmod_i = Model(fun_B, prefix='p'+str(i)+'_')
        gmod += gmod_i
        bmod += bmod_i
    
    return gmod, bmod

def make_models_pars(n=1):
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

def res_GB(params, f, G, B, **kwargs):
    '''
    residual of both G and B
    '''
    eps = kwargs.get('eps', None)
    n = kwargs.get('n', 1)
    # eps = 100
    # eps = (G - np.amin(G))
    # eps = pow((G - np.amin(G)*1.001), 1/2)
    gmod, bmod = make_models(n)
    residual_G = G - gmod.eval(params, x=f)
    residual_B = B - bmod.eval(params, x=f)

    if eps is None:
        return np.concatenate((residual_G, residual_B))
    else:
        return np.concatenate((residual_G * eps, residual_B * eps))

def set_params(f, G, B, n=1):
    ''' set the parameters for fitting '''

    params = Parameters()
    for i in np.arange(1, n+1):
        params.add(
            'p'+str(i)+'_amp',              # amplitude (G)
            value=np.amax(G) - np.amin(G),    # init: peak height
            min=0,                          # lb
            max=np.inf,                     # ub
        )
        params.add(
            'p'+str(i)+'_cen',              # center 
            value=np.mean(f),               # init: average f
            min=np.amin(f),                  # lb: assume peak is in the range of f
            max=np.amax(f),                  # ub: assume peak is in the range of f
        )
        params.add(
            'p'+str(i)+'_wid',                  # width (fwhm)
            value=(np.amax(f) - np.amin(f)) / 2,  # init: half range
            min=1,                              # lb
            max=np.amax(f) - np.amin(f),          # ub: assume peak is in the range of f
        )
        params.add(
            'p'+str(i)+'_phi',              # phase shift
            value=0,                        # init value: peak height
            min=-np.pi / 2,                 # lb
            max=np.pi / 2,                  # ub
        )
      
    params.add(
        'g_c',              # initialize G_offset
        value=np.amin(G),    # init G_offset = mean(G)
        # min=-np.inf,        # lb
        # max=np.amax(G)/2,    # ub
    )        
    params.add(
        'b_c',              # initialize B_offset
        value=np.mean(B),   # init B_offset = mean(B)
        # min=np.amin(B)/2,    # lb
        # max=np.amin(B)/2,    # ub
    )
    return params

def minimize_GB(f, G, B, n=1, cen_guess=None, wid_guess=None, factor=None):
    '''
    use leasesq to fit
    '''
   
    # set data for fitting
    if not None in [cen_guess, wid_guess, factor]:
        condition = np.where((f >= cen_guess - wid_guess * factor) & (f <= cen_guess + wid_guess * factor))
        f, G, B = f[condition], G[condition], B[condition]

    # eps = None
    eps = pow((G - np.amin(G)*1.001), 1/2) # residual weight
     
    # set params with data
    params = set_params(f, G, B, n)
    
    # minimize with leastsq
    # mini = Minimizer(residual, params, fcn_args=(f, G, B))
    # result = mini.leastsq(xtol=1.e-10, ftol=1.e-10)
    result = minimize(res_GB, params, method='leastsq', args=(f, G, B), kws={'eps': eps, 'n': n}, xtol=1.e-18, ftol=1.e-18)

    print(fit_report(result)) 
    print('success', result.success)
    print('message', result.message)
    print('lmdif_message', result.lmdif_message)
    return result

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    try:
        from AccessMyVNA_dummy import AccessMyVNA
    except:
        from Modules.AccessMyVNA_dummy import AccessMyVNA 

    # gbfitting = GBFitting()
    # # print(gbfitting.gmodel.param_names)
    # # print(gbfitting.bmodel.param_names)
    # # print(gbfitting.gpars)
    # # print(gbfitting.bpars)

    # gmod = gbfitting.gmodel
    # bmod = gbfitting.bmodel
    accvna = AccessMyVNA()
    _, f, G = accvna.GetScanData(nWhata=-1, nWhatb=15)
    _, _, B = accvna.GetScanData(nWhata=-1, nWhatb=16)
    # G = G * 1e3
    # B = B * 1e3

    n = 1

    result = minimize_GB(f, G, B, n, )
    params = set_params(f, G, B, n)
    result = minimize(res_GB, params, method='leastsq', args=(f, G, B), kws={'eps': pow((G - np.amin(G)*1.001), 1/2), 'n': n}, xtol=1.e-10, ftol=1.e-10)

    print(fit_report(result)) 
    print('success', result.success)
    print('message', result.message)
    print('lmdif_message', result.lmdif_message)
    print('params', result.params.get('p1_cen').value)
    print('params', result.params.get('p1_cen').stderr)
    print('params', result.params.valuesdict())
    # exit(0)
    gmod, bmod = make_models(n)

    plt.figure()
    plt.plot(f, G, 'bo')
    plt.plot(f, gmod.eval(result.params, x=f), 'k--')
    plt.twinx()
    plt.plot(f, B, 'go')
    plt.plot(f, bmod.eval(result.params, x=f), 'k--')

    plt.figure()
    plt.plot(G, B, 'bo')
    plt.plot(gmod.eval(result.params, x=f), bmod.eval(result.params, x=f), 'k--')

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