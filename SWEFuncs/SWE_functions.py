# swe.py
import sympy as sp
import numpy as np
from sympy.physics.quantum import TensorProduct
import scipy.optimize as optimize
from scipy.interpolate import LinearNDInterpolator, interp1d
import matplotlib.pyplot as plt

# display and preview not used here, but are handy to make available when 
# calling the swe module

# material parameters
# keep evertying in SI units to avoid confusion
parms = {
         r'mu_L': 183e3,   # 'paralllel' shear modulus
         r'mu_T': 49.7e3, # 'perpendicular' shear modulus
         r'E_L': 0.942e6, # 1e6 # 'parallel' extensional modulus
         'c_2': -0.43,  # isotropic strain hardening parameter
         'c_4': 18.6, # strain hardening parameter for fiber extension
         'a': 7e-5, # indenter half-width
         'h': 1.1e-3, # material thickness
         'l': 0.02, # indenter length
         'fr_s':0.1, # fractional error in contact stiffness ratio
         'fr_E':0.1} # fractional error in E_L/mu_T


# invariants
i={}
for n in [1,4,5,6,7,8,9]:
    i[n]=sp.symbols(f'I_{n}') 

# strains
lambda_L, delta = sp.symbols(['lambda_L', 'delta'],  positive = True)

# symbolic forms for angle used to define group velocity
theta_g = sp.symbols('theta_g', real=True)

# parameters used in HGY formulation
c_2, c_4, mu_T, mu_L, E_L, beta = sp.symbols(['c_2', 'c_4', 'mu_T', 
            'mu_L', 'E_L',  'beta'], positive = True)

# rotation angle arond 1 axis for incremental strain
theta = sp.symbols('theta', real=True)

# define various ratios used in contact stiffness equations
a, h = sp.symbols(['a', 'h'])

# hydrostatic pressure term
p = sp.symbols('p')

# vector pointing along fiber axis
m = sp.Matrix([[0],[0],[1]])

# strain energy funtion from  Hegde et al. Int. J. of Non-Linear Mech. 
# 160, 104663 (2024) (http://dx.doi.org/10.1016/j.ijnonlinmec.2024.104663)
W =((mu_T/(2*c_2))*(sp.exp(c_2*(i[1]-3))-1)+((E_L+mu_T-4*mu_L)/(2*c_4))*
                (sp.exp(c_4*(sp.sqrt(i[4])-1)**2)-1) +
                ((mu_T-mu_L)/2)*(2*i[4]-i[5]-1))

# define the incremental strain energy function and incrementa moduli
mu_lt, mu_tl, mu_tt, E_ll, phi_i, zeta_i, sigma_l = sp.symbols(['mu_lt', 'mu_tl',
        'mu_tt', 'E_ell_ell', 'phi_i', 'zeta_i', 'sigma_l'])

Wi = sp.Rational(1,2)*(mu_tt*(i[1]-3)+
    (E_ll+mu_tt-4*mu_lt)*(sp.sqrt(i[4])-1)**2
    +(mu_tt-mu_lt)*(2*i[4]-i[5]-1))
    # +(i[4]-1)*(i[6]-i[7]))

# F for extensional prestrain
F0 = (sp.Matrix([[1/sp.sqrt(lambda_L), 0, 0],
              [0, 1/sp.sqrt(lambda_L), 0],
              [0, 0, lambda_L]]))

# rotation matrix for rotation of theta degrees around axis 0
R = sp.Matrix([[1, 0, 0],
           [0, sp.cos(theta), -sp.sin(theta)],
           [0, sp.sin(theta), sp.cos(theta)]])

def F(idx, incremental = False):
    '''
    F is the deformation gradient tensor.  idx is a tuple describing the 
    deformation: 1st number is displacement direction, 2nd is gradient direction.
    example values of idx are: (descriptions are with theta = 0)
    (1,1) or (1,1) transverse extension (lambda_L = 1 only)
    (2,2) longitudinal extension 
    (0,1) transverse polarization, transverse propagation (tt)
    (0,2) or (1,2) transverse polarization, longitudinal propagation (tl)
    (2,1) or (2,0) longitudinal polarization, transverse propagation (lt)
    
    use of a third number in the tuple means we are computing the ful
    angular dependence of the SH or SV modes as follows
    (0,2,1) horizontal shear mode as a function of angle 
    (1,2,1) vertical shear mode as a functiono of angle 
    mu = {'SH': swe.modulus((0,2,1)),
          'SV': swe.modulus((1,2,1))}
    
    Set incremental to true if F only describes the incremental deformatoin
    '''
           
    F_inc = (sp.Matrix([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]))
    
    F_inc[idx[0],idx[1]] = delta # this is the increment to F
    
    if idx==(0,0):  # transverse extensiont, lambda_L = 1 by definition
        F_inc[1,1] = sp.sqrt(1/(beta*delta))
        F_inc[2,2] = sp.sqrt(beta/delta)
    
    if idx==(1,1):  # transverse extensiont, lambda_L = 1 by definition
        F_inc[0,0] = sp.sqrt(1/(beta*delta))
        F_inc[2,2] = sp.sqrt(beta/delta)
    
    if idx==(2,2): # extension along fiber axes
        F_inc[0,0] = sp.sqrt(1/delta)
        F_inc[1,1] = sp.sqrt(1/delta)
    
    # we put a third element in the idx tuple if we want to use the angle
    if len(idx)>2:
        F_inc = R*F_inc*R.transpose()
    
    if incremental:
        return F_inc
    else:
        return F_inc*F0
                
def C(F):
    # right Cauchy-Green Tensor
    return F.T*F

def B(F):
    # left Cauch-Green Tensor
    return F*F.T


def I1(F):
    # first strain invariant
    return B(F).trace()

def I4(F):
    # fourth strain invariant for fiber directed along x3
    return m.dot(C(F) * m)

def I5(F):
    # fifth strain invariant for fiber directed along x3
    return m.dot((C(F) * C(F)) * m)

def I6(F, prestress):
    # sixth strain invariant for prestressed material
    return (prestress * C(F)).trace()

def I7(F, prestress):
    # seventh strain invariant for prestressed material
    return (prestress * C(F) * C(F)).trace()

def I8(F, prestress):
    # eighth strain invariant for prestressed fiber directed along x3
    return m.dot(prestress * C(F) * m)

def I9(F, prestress):
    # eighth strain invariant for prestressed fiber directed along x3
    return m.dot(prestress * C(F) * C(F) * m)

# register functions
I = {}
I[1] = I1
I[4] = I4
I[5] = I5
I[6] = I6
I[7] = I7
I[8] = I8
I[9] = I9

# values of I using prestrained reference state
i0 = {}
i0[1] = 3
i0[4] = 1
i0[5] = 1
i0[6] = sigma_l
i0[7] = sigma_l
i0[8] = sigma_l
i0[9] = sigma_l

idx = np.arange(3) # used for nested loops
     
def stress(idx):
    # Ogden Eq. 2.47
    f = F(idx)
    tp4 = TensorProduct(f*m, f*m).reshape(3,3)
    tp5 = (TensorProduct(f*m, B(f)*f*m).reshape(3,3)+
           TensorProduct(B(f)*f*m, f*m).reshape(3,3))
    
    # Wd[n] is dW/di[n]
    Wd={}
    for n in [1,4,5]:
        Wd[n]=sp.diff(W, i[n])

    sigma = -p*sp.eye(3) + 2*Wd[1]*B(f)+2*Wd[4]*tp4+2*Wd[5]*tp5
    
    pval = sp.solve(sigma[0,0], p)[0]
    sigma = sigma.subs(p, pval)
    for n in [1,4,5]:
        sigma = sigma.subs(i[n], I[n](f))
    return sp.simplify(sigma)

# define the prestress, which is always uniaxial along the symmetry
# axis for now
prestress = (sp.Matrix([[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, sigma_l]]))

def stressi(idx):  # stress from incremental strain energy funcion
    # Ogden and Singh Eq. 34
    f = F(idx, incremental=True)
    tp4 = TensorProduct(f*m, f*m).reshape(3,3)
    tp5 = (TensorProduct(f*m, B(f)*f*m).reshape(3,3)+
           TensorProduct(B(f)*f*m, f*m).reshape(3,3))
   # Sig = f*prestress*f.T
    
    
    # we don't use I8 and I9 currently, but maybe we will someday
    Wd={}
    for n in [1,4,5,6,7]:
        Wd[n]=sp.diff(Wi, i[n])
        
    sigma = (-p*sp.eye(3) + 2*Wd[1]*B(f)+2*Wd[4]*tp4+2*Wd[5]*tp5)
 #       +     2*Wd[6]*Sig + 2*Wd[7]*(B(f)*Sig+Sig*B(f)))
    
    pval = sp.solve(sigma[0,0], p)[0]
    sigma = sigma.subs(p, pval)
    
    for n in [1,4,5]:
        sigma = sigma.subs(i[n], I[n](f))
        
    for n in [6,7]:
        sigma = sigma.subs(i[n], I[n](f, prestress))

    return sp.simplify(sigma)


def modulus(idx):
    # calculate differential modulus from stress function
    if len(idx)>2:  # for theta dependence of SH and SV modes
        stress_tensor= R.transpose()*stress(idx)*R
    else:
        stress_tensor = stress(idx)
    stress_component = stress_tensor[idx[0], idx[1]]
    mod = sp.diff(stress_component, delta)
    if idx[0]!=idx[1] : # don't need to do this for the extension cases
        mod = mod.subs(delta, 0)
    else:
        mod = mod.subs(delta, 1)
    mod = sp.simplify(mod)
    return mod

def modulusi(idx):
    # calculate differential modulus from stress function
    if len(idx)>2:  # for theta dependence of SH and SV modes
        stress_tensor= R.transpose()*stressi(idx)*R
    else:
        stress_tensor = stressi(idx)
    stress_component = stress_tensor[idx[0], idx[1]]
    mod = sp.diff(stress_component, delta)
    if idx[0]!=idx[1] : # don't need to do this for the extension cases
        mod = mod.subs(delta, 0)
    else:
        mod = mod.subs(delta, 1)
    mod = sp.simplify(mod)
    return mod

# it's useful to have expressions for some specific incremental moduli
# for the linearized mode we need the stress incrementa modluli
mu_lt_sym = modulus((2, 1))
mu_tl_sym = modulus((1, 2))
mu_tt_sym = modulus((0, 1))
E_ll_sym = modulus((2, 2))
sigma_l_sym = stress((2,2))[2,2].subs(delta, 1)
#sigma_l_sym = mu_tl_sym - mu_lt_sym - gives same result as above
phi_i_sym = mu_lt_sym/mu_tt_sym-1
zeta_i_sym = sp.Rational(1,4)*(E_ll_sym/mu_tt_sym-3)


# these functions for mu_SH and mu_SV are the  new simplified versions, but
# are equivalent to what is in the paper

mu_SH = (mu_tt*(1+phi_i)+sigma_l)*(sp.cos(theta))**2+mu_tt*(sp.sin(theta))**2

mu_SV = (mu_tt*(1+phi_i+(zeta_i-phi_i)*(sp.sin(2*theta))**2)+
                  sigma_l*((sp.cos(theta))**2-sp.Rational(1,2)*(sp.sin(2*theta))**2))


def lambdify2(function, xvars, parms):
    '''
    function is a symbolic function of many variables
    this returns a lambdified version of the function as a function of 
    the symbol variables in the  list xvars, substituting the other ones 
    for the values in the parms dictionary
    '''
    variables = list(function.free_symbols)
    for xvar in xvars:
        try:
            variables.remove(xvar)
        except:
            print(f'problem removing {xvar}')
    for var in variables:
        try:
            function = function.subs(var, parms[var.name])
        except:
            print(f'problem substituting for {var}')
            
    # check to see if the function is a constant
    if function.free_symbols == set():
        const = float(function)
        
        def plot_function(var):
            var_array = np.asarray(var)
            # preserve shape/dtype; if scalar input, return scalar
            if var.shape == ():
                return const  
            else :
                return np.full_like(var_array, const, dtype=float)

    else:
        plot_function = sp.lambdify(xvars, function, modules=["numpy"])
        
    return plot_function


def guess_beta(lamtval, parms):
    # low strain value for beta, used as initial guess calc_beta
    return 1+(lamtval-1)*(1-4*parms[r'mu_T']/(parms[r'E_L']+
                                                 parms[r'mu_T']))

def calc_beta(lamtval, parms):
    '''
    determine the value of beta that ensures all lateral normal stresses are
    zero.  The stress function has the other transverse stress being zero,
    so this function insures that the longitudinal stress is zero
    '''
    sigma = stress((1,1))[2,2]
    sigma = sigma.subs(delta, lamtval)
    ftosolve = lambdify2(sigma, [beta], parms)    
    guess = guess_beta(lamtval, parms)
    soln = optimize.least_squares(ftosolve, guess, bounds=(0.5*guess, 2*guess))    
    return soln['x']

def transverse_extension_single(lamtval, parms, **kwargs):
    # returns true stress for extension in transverse direction
    # change beta_calc to True in call if you want to 
    # rigorously calculate beta for transverse extension
    beta_calc = kwargs.get('beta_calc', False)
    if beta_calc:
        beta_val = calc_beta(lamtval, parms)[0]
    else:
        beta_val = guess_beta(lamtval, parms)
    sigma = stress((1,1))[1,1]
    sigma = sigma.subs(beta, beta_val).subs(lambda_L, 1)
    stress_func = lambdify2(sigma, [delta], parms)
    return stress_func(lamtval)

def group_velocity(mu):
    # input here is mu (either SH or SV)
    v = sp.sqrt(mu)
    v2 = v*sp.sin(theta)+sp.diff(v, theta)*sp.cos(theta)
    v3 = v*sp.cos(theta)-sp.diff(v, theta)*sp.sin(theta)
    vg = sp.sqrt(v2**2+v3**2) # group velocity
    theta_g = theta + sp.atan(sp.diff(v,theta)/v) # angle of propagation
    return vg, theta_g

transverse_extension = np.vectorize(transverse_extension_single)
calc_beta_vec = np.vectorize(calc_beta)
guess_beta_vec = np.vectorize(guess_beta)

# Now we have some functions related to the indentation experiments
# specify the varables
rs = sp.symbols (r'r_s', real=True, positive=True) #s_perp/s_parallel
alpha, beta, A = sp.symbols(['alpha', 'beta', 'A'])
rmu = sp.symbols('r_mu', real=True, positive=True) #mu_L/mu_T
rE = sp.symbols('r_E', real=True, positive = True)
frs, frE = sp.symbols(['fr_s', 'fr_E'], real = True, positive =True)

def mu_L_error(df, x_val, x_err, s_ratio, s_ratio_err):
    # create a dictionary for the relvant values of the stiffness ratio
    s = {0:s_ratio, # this is the actual value of r
        -1:(1-s_ratio_err)*s_ratio, # this is the lower limit of r, given the error
         1:(1+s_ratio_err)*s_ratio}  # this is the upper limit of r, given the error
    
    # now we define a similar dictionary for the relevant values of the 
    # normalized value of E_L
    x = {0:x_val, # this is the actual value of r
        -1:(1-x_err)*x_val, # this is the lower limit of r, given the error
         1:(1+x_err)*x_val}  # this is the upper limit of r, given the error 

    yval = {}
    for s_idx in [-1, 0, 1]:
        yval[s_idx]={}
        for x_idx in [-1, 0, 1]:
            yval[s_idx][x_idx]=fixed_s_ratio_func(df,s[s_idx])(x[x_idx])
    
    # calculate error from uncertinty in s values
    yerr_s=[[yval[0][0]-yval[-1][0]], [yval[1][0]-yval[0][0]]]
    yerr_s = np.array(yerr_s)
    
    # now do the same thing for error from uncertainty in x (x is normallized E_L)
    yerr_x=[ [yval[0][0]-yval[0][-1]], [yval[0][1]-yval[0][0]]]
    yerr_x = np.array(yerr_x)
    
    return (yerr_x**2+yerr_s**2)**(0.5)
    
def ah_corr(ah):
    # ah here is the a/h ratio
    return -0.27 *np.log(ah/2)

def s_parallel(lambda_L_vals, parms):
    # returns parallel contact modulus for different pre-strains
    ah = parms['a']/parms['h']
    mu_Tt = lambdify2(modulus((0,1)), [lambda_L], parms)(lambda_L_vals)
    s_parallel = mu_Tt/ah_corr(ah)
    return s_parallel

def s_perp(lambda_L_vals, df, parms, shear_type):
    '''
    Calculate contact modulus for indenter aligned perpendicular to symmetry axis.
    
    Parameters
    ----------
    lambda_L_vals : array of floats
        extension ratio in fiber direction.
        
    df : dataframe
        dataframe with FEA data used to calculate the contact stiffnesses.

    parms : dictionary
        parameter dictionary containingthe elastic constants.
    shear_type : string
        'tl' or 'lt' - generally assumed to be tl in the paper

    Returns
    -------
    s_perp : array of floats
        low-strain contact stiffness useing mu_tl or mu_lt as the shear modulus.
    '''
    
    Ell = lambdify2(modulus((2,2)), [lambda_L], parms)(lambda_L_vals)
    mu_T = lambdify2(modulus((0,1)), [lambda_L], parms)(lambda_L_vals)
    if shear_type=='tl':
        mu_L = lambdify2(modulus((0,2)), [lambda_L], parms)(lambda_L_vals)
    elif shear_type=='lt':
        mu_L = lambdify2(modulus((2,0)), [lambda_L], parms)(lambda_L_vals)
    else:
        print(f'shear_type of {shear_type} is not valid')
        return
    x = Ell/(3*mu_T)
    y = mu_L/mu_T
    
    rs=s_ratio_func(df)(x,y)

    s_perp = s_parallel(lambda_L_vals, parms)*rs
    return s_perp

def print_latex(expression):
    # print the latex code for the specified expression
    latex_expression = sp.latex(expression)
    print(latex_expression)

def save_image(expression, filename):
    with open(filename, 'wb') as outputfile:
        preamble = "\\documentclass[10pt]{standalone}\n" \
                    "\\usepackage{amsmath,amsfonts}\\begin{document}"
        sp.preview(expression, viewer='BytesIO', outputbuffer=outputfile,
                output = 'pdf', preamble=preamble)
           
def get_ah_values(df):
    '''
    Extract all a/h values from the FEA data into a list

    Parameters
    ----------
    df : dataframe
        Input dataframe
    

    Returns
    -------
    List of all unique a/h values

    '''
    

    return   df['ah'].unique().tolist()

def choose_ah(df, ah):
    '''
    Refine fea dataframe to only include specified value within 10% of specified 
    value.

    Parameters
    ----------
    df : dataframe 
        Iinput dataframe that may contain lots of a/h values
    
    Returns
    -------
    Dataframe limited to specified a/h value.
    '''
    df['ah'] = df['a']/df['h']
    idx = df.index[abs(df.ah-ah)/ah<0.1].tolist()
    df_prop = df.loc[idx]
    return df_prop

    
def s_ratio_func(df):
    '''
    Fitting function for contact stiffness ratio, obtained from FEA data.

    Parameters
      ----------
      df : dataframe
          input dataframe with FEA data - should have single value of a/h
          
      Returns
      -------
      Function
          function of x, y, where x is E_L/3mu_T and y
          is mu_L/mu_T.
    '''
    
    def loginter2d(x, y, z):
        # function definition to create 2d interpolation in log domain for x,y
        logx = np.log10(x)
        logy = np.log10(y)
        linzfunc = LinearNDInterpolator(list(zip(logx, logy)), z)
        logzfunc = lambda a, b : linzfunc(np.log10(a), np.log10(b))
        return logzfunc
    
    # each value of a/h has a different function, which we put in a zfunc dict.
    
    x = df['Ef3Gt']  # this is our x axis variable
    y = df['GfGt']  # we cycle over all values of ths property
    z = df['rs']
    
    # Create an interpolating function using LinearNDzfunc
    return loginter2d(x, y, z)


def fixed_s_ratio_func(df, s_ratio):
    '''
    Fitting function for mu_L/mu_T vs E_L/3mu_T at fixed
    value of the contact stiffness ratio, r_perp/r_parallel.

    Parameters
    ----------
    df : dataframe
        Input dataframe with FEA data, should have single value for both a/h, 
        and l/a. 
    
    rs : Contact stiffness ratio (s_perp/s_parallel).
        
    Returns
    -------
    Function
        function of x where x is E_L/3mu_T and y
        is .
    '''

    xmin = df.Ef3Gt.min()
    xmax = df.Ef3Gt.max()
    ymin = df.GfGt.min()
    ymax = df.GfGt.max()
    x = np.geomspace(xmin, xmax, 100)
    y = np.array([], dtype=float)
    sfunc = s_ratio_func(df)
    for xval in x:
        if (np.isnan(sfunc(xval, ymax)) or 
            s_ratio>sfunc(xval, ymax) or
            s_ratio<sfunc(xval, ymin)):
            y = np.append(y, np.nan)
        else:
            def ftosolve(yval):
                return sfunc(xval, yval)-s_ratio
            soln = optimize.least_squares(ftosolve, ymax, 
                                 bounds=(ymin, ymax))
            y = np.append(y, soln['x'])
    log_interp = interp1d(np.log10(x), y)   
    return lambda x : log_interp(np.log10(x)) 

    
def make_s_ratio_plot(df, xrange, yrange):
    '''
    Make a plot of the contact modulus ratio (s_perp/s_parallel)

    Parameters
    ----------
    df : datframe
        input dataframe with FEA dat (single a/h value for all data)
    xrange : list of two floats
        range of x data to plot.
    yrange : TYPE
        range of y data to plot.
        
    Returns
    -------
    Fig and ax handles for contour plot of stiffness ratio as a function of
    E_L/3mu_T and mu_L/mu_T.
    '''
    
    zfunc = s_ratio_func(df)

    fig, ax = plt.subplots(1,1, figsize = (4, 3), constrained_layout = True)
    
    x_new = np.linspace(xrange[0], xrange[1], 300)
    y_new = np.linspace(yrange[0], yrange[1], 300)

    X_new, Y_new = np.meshgrid(x_new, y_new)
    Z_new = s_ratio_func(df)(X_new,Y_new)
    
    # Plot the interpolated function
    image = ax.pcolormesh(X_new, Y_new, Z_new, cmap='viridis',
                     vmin=zfunc(xrange[0], yrange[0]),
                     vmax=zfunc(xrange[1], yrange[1]))
    fig.colorbar(image, label = r'$s_\perp/s_\parallel$')
    
    
    # Redefine format_coord
    def format_coord(x_val, y_val):
        z_val = zfunc(x_val, y_val)
        return f'x={x_val:.2f}, y={y_val:.2f}, z={z_val:.2f}'    
    
    ax.format_coord = format_coord
       
    ax.set_xlabel(r'$E_L/(3\mu_T)$')
    ax.set_ylabel('$\mu_L/\mu_T$')
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
   
    fig.show()

    
    return fig, ax

def add_s_ratio_line(df, ax, s_ratio, fmt, **kwargs):
    '''
    Add line to the plot at specified value of s_ratio

    Parameters
    ----------
    df : dataframe
        input dataframe with FEA data
    ax : axis handle
        axis to plot on.
    s_ratio : float
        falue of contact stiffness ratio corresponding to the line
    fmt : format strong for plotting the line

    Returns
    -------
    No return - just updates the axis by adding the line

    '''
    label = kwargs.get('label', fr'$s_\perp/s_\parallel$={s_ratio:.2f}')
    [xmin, xmax] = ax.get_xlim()
    [ymin, ymax] = ax.get_ylim()
    xr = np.linspace(xmin, xmax, 100)
    yr = fixed_s_ratio_func(df, s_ratio)(xr)
    ax.plot(xr, yr, fmt, label = label)
    ax.set_ylim([ymin, ymax])
    return 

def wrap_for_python_code(s: str, width=80, indent=4):
    """
    Return Python source code that reconstructs string `s`, formatted so that
    each source line is ≤ width characters. Uses adjacent string literals.
    """
    max_chunk = width - indent - 2  # subtract 2 for the quotes
    chunks = [s[i:i+max_chunk] for i in range(0, len(s), max_chunk)]

    indent_spaces = " " * indent
    lines = ["(\n"]
    lines += [f'{indent_spaces}"{c}"\n' for c in chunks]
    lines.append(")\n")

    return "".join(lines)
    