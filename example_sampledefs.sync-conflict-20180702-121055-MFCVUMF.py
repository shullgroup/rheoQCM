# note that the values of fref used for the springpot models are for the
# reference temperature from the original DMA data, which is a bit
# arbitrary


def PS_3k_cool():
    sample = {}
    sample['datadir'] = 'example_data/PS_3k/'
    sample['barefile'] = 'Urma_bare_cool'
    sample['filmfile'] = 'Wilfred_3k_cool'
    sample['baretrange'] = [0, 900]  # default is [0, 0]
    sample['filmtrange'] = [0, 900]  # default is [0, 0]
    sample['Temp'] = [120, 110, 100, 90, 80, 70, 60, 40] # default is [22]
    sample['nhplot'] = [3, 5]  # default is [1, 3, 5]
    sample['nhcalc'] = ['355'] # default is ['353']
    return sample


def PMMA_75k():
    sample = {}
    sample['datadir'] = 'example_data/PMMA/'
    sample['barefile'] = 'Angela_bare'
    sample['filmfile'] = 'Angela_PMMA_75k_film2'
    sample['Temp'] = [120, 110, 100, 90, 80, 70, 60, 50, 40]
    sample['nhcalc'] = ['355']
    sample['nhplot'] = [3, 5]
    return sample



