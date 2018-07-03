
# data file 
samp = {
    'index',
    'time',
    'f1',
    'g1',
    'f3',
    'g3',
    'temp', # if temp module established
    'marked',
}

# test_para
samp_para = {
    'f0',
    't0',
    'tshift',
    'ref', # file name of ref type
}
# ref may have defferent rows with samp
# if the ref is from external reference
# if the ref is from the same test with samp, they should have the same idxes and time pattern
ref = {
    'index',
    'time',
    'f1',
    'g1',
    'f3',
    'g3',
    'temp', # if temp module established
    'marked'
}

# has the same rows with samp
# more columns may add 
calc = {
    'index',
    'delf1',
    'delfcal1',
    'delg1',
    'delgcal1',
    'delf3',
    'delfcal3',
    'delg3',
    'delgcal3',
    'drho',
    'Grho',
    'phi',
    'dlam',
    'lamrho',
    'delrho',
    'delfdelfsn',
    'delgdelfsn',
    'rh',
    'rlam',
}

# save as dict
calc_para = {
    'nhcal',
    'nhplot'
}

# raw spectra in a separate file
# n is the index in the data file
raw = {
    index: {
        't': {},
        'samp': {
            1: {
                'f': {},
                'G': {},
                'B': {},  
            },
            3: {
                'f': {},
                'G': {},
                'B': {},
            },
        },
        'ref': { # if reference measured together
            '''
            ndarray or DataFrame of data
            f, G1, B1, G3, B3,...
            '''
        },
        'temp': [] # if temp module is established
    }
}

settings = {

}

# program settings
settings = {
    'ver': {}, # version
    # ...
}


