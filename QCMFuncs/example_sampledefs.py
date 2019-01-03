'''
This file define two files tested from the same sample, tested by Matlab program and Python program, respectively.
'''

def sample_dict():
    sample = {}  # individual sample dictionaries get added to this

    samplename = 'polymer_matlab'
    sample[samplename] = {
        'samplename': samplename,
        'filetype': 'mat',                            # file type of data file(s) 'mat' or 'h5'. default is 'mat'.
        # 'filmchn': 'samp',                            # For 'h5' only. define the channel where film data is stored. defaule is 'samp'
        'datadir': 'polymer_matlab',                                # the relative path of file in 'dataroot'. Note: don't include '.' at the beginning of the relative path
        'barefile': 'polymer_matlab_bare',                 # For 'mat' only. bare file name. 'h5' file will use the bare defined in its file
        'filmfile': 'polymer_matlab',                # file name where the film data is stored
        # 'baretrange': [0, 900],                       # index range to be used for bare. default is [0, 0]
        # 'filmtrange': [0, 900],                       # index range to be used for film. default is [0, 0]
        # 'filmindex': range(0, 156, 10),               # array form (range, list, tuple) of index of points to be calculated and plotted. 
        # 'xscale': 'linear',                           # x-axis (time or temperature) scale. 'linear' or 'log'. default is 'linear'
        # 'freqref':                                  # leave this commented
        # 'Temp': [120, 110, 100, 90, 80, 70, 60, 40],  # For 'mat' only. list of temperature steps. default is [22]
        # 'xtype': 't',                                 # For 'h5' only. data used for x-axis. 't' or 'temp'. default is 't'
        'nhplot': [3, 5],                             # list of harmonics to be plotted. default is [1, 3, 5]
        'nhcalc': ['355'],                            # list of harmonic combinations to be calculated. default is ['353']
        }

    samplename = 'polymer_h5'
    sample[samplename] = {
        'samplename': samplename,
        'filetype': 'h5',                            # file type of data file(s) 'mat' or 'h5'. default is 'mat'.
        'filmchn': 'samp',                            # For 'h5' only. define the channel where film data is stored. defaule is 'samp'
        'datadir': 'polymer_h5',                                # the relative path of file in 'dataroot'. Note: don't include '.' at the beginning of the relative path
        'filmfile': 'polymer',                # file name where the film data is stored
        # 'baretrange': [0, 900],                       # index range to be used for bare. default is [0, 0]
        # 'filmtrange': [0, 900],                       # index range to be used for film. default is [0, 0]
        # 'filmindex': range(0, 156, 10),               # array form (range, list, tuple) of index of points to be calculated and plotted. 
        # 'xscale': 'linear',                           # x-axis (time or temperature) scale. 'linear' or 'log'. default is 'linear'
        # 'freqref':                                  # leave this commented
        # 'Temp': [120, 110, 100, 90, 80, 70, 60, 40],  # For 'mat' only. list of temperature steps. default is [22]
        # 'xtype': 't',                                 # For 'h5' only. data used for x-axis. 't' or 'temp'. default is 't'
        'nhplot': [3, 5],                             # list of harmonics to be plotted. default is [1, 3, 5]
        'nhcalc': ['355'],                            # list of harmonic combinations to be calculated. default is ['353']
        }

    return sample
    
