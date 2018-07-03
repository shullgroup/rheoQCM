# note that the values of fref used for the springpot models are for the
# reference temperature from the original DMA data, which is a bit
# arbitrary


def sample_dict():
    sample = {}  # individual sample dictionaries get added to this
    tempdic = {}  # need to start with empty dictionary for each sample
    tempdic['samplename'] = 'PS_3k_cool'
    tempdic['datadir'] = 'example_data/PS_3k/'
    tempdic['barefile'] = 'Urma_bare_cool'
    tempdic['filmfile'] = 'Wilfred_3k_cool'
    tempdic['baretrange'] = [0, 900]  # default is [0, 0]
    tempdic['filmtrange'] = [0, 900]  # default is [0, 0]
    tempdic['Temp'] = [120, 110, 100, 90, 80, 70, 60, 40]  # default is [22]
    tempdic['nhplot'] = [3, 5]  # default is [1, 3, 5]
    tempdic['nhcalc'] = ['355']  # default is ['353']
    sample[tempdic['samplename']] = tempdic

    #  PMMA sample - Meredith
    tempdic = {}
    tempdic['samplename'] = 'PMMA_75k'
    tempdic['datadir'] = 'example_data/PMMA/'
    tempdic['barefile'] = 'Angela_bare'
    tempdic['filmfile'] = 'Angela_PMMA_75k_film2'
    tempdic['Temp'] = [120, 110, 100, 90, 80, 70, 60, 50, 40]
    tempdic['nhcalc'] = ['355']
    tempdic['nhplot'] = [3, 5]
    sample[tempdic['samplename']] = tempdic

    #  PMMA sample - Tom sample 4
    samplename = 'PMMA_75k_S04'
    sample[samplename] = {
            'samplename': samplename,
            'datadir': 'example_data/PMMA/',
            'barefile': 'QCMS04_bare',
            'filmfile': 'QCMS04_75kPMMA',
            'nhplot': [3, 5]
            }

    #  PMMA sample - Tom sample 5
    samplename = 'PMMA_75k_S05'
    sample[samplename] = {
            'samplename': samplename,
            'datadir': 'example_data/PMMA/',
            'barefile': 'QCMS05_bare',
            'filmfile': 'QCMS05_75kPMMA',
            'nhplot': [3, 5]
            }

    return sample
