# note that the values of fref used for the springpot models are for the
# reference temperature from the original DMA data, which is a bit
# arbitrary

def sample_dict():
    sample = {}  # individual sample dictionaries get added to this

    samplename = 'PS_3k_cool'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'data/Taghon/PS_3k/',
        'barefile': 'Urma_bare_cool',
        'filmfile': 'Wilfred_3k_cool',
        'baretrange': [0, 900],  # default is [0, 0]
        'filmtrange': [0, 900],  # default is [0, 0]
        'Temp': [120, 110, 100, 90, 80, 70, 60, 40],  # default is [22]
        'nhplot': [3, 5],  # default is [1, 3, 5]
        'nhcalc': ['355'],  # default is ['353']
        }

    #  PMMA sample - Meredith
    samplename = 'PMMA_75k_T01'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'data/Taghon/PMMA/',
        'barefile': 'Angela_bare',
        'filmfile': 'Angela_PMMA_75k_film2',
        'Temp': [120, 110, 100, 90, 80, 70, 60, 50, 40],
        'nhcalc': ['355'],
        'nhplot': [3, 5]
        }

    #  PMMA sample - Tom sample 4
    samplename = 'PMMA_75k_S04'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'data/Schmitt/PMMA/',
        'barefile': 'QCMS04_bare',
        'filmfile': 'QCMS04_75kPMMA',
        'nhplot': [3, 5]
        }

    #  PMMA sample - Tom sample 5
    samplename = 'PMMA_75k_S05'
    sample[samplename] = {
        'samplename': samplename,
        'datadir': 'data/Schmitt/PMMA/',
        'barefile': 'QCMS05_bare',
        'filmfile': 'QCMS05_75kPMMA',
        'nhplot': [3, 5]
        }

    #  20180629
    samplename = 'DGEBA-Jeffamine2000_RT'
    sample[samplename] = {
        'samplename': 'DGEBA-Jeffamine2000_RT',
        'datadir': 'data/Qifeng/20180629/',
        'barefile': 'bare_air',
        'filmfile': 'DGEBA-Jeffamine2000_RT',
        'firstline': 50,
        'filmtrange': [1000, 6000],
        'soln_type': 'bulk',
        'nhplot': [1, 3, 5],
        'nhcalc': ['355', '353']
        }

    #  One of David's bare crystal datasets
    samplename = 'Bare_xtal'
    sample[samplename] = {
        'samplename': 'bare_room_temp',
        'datadir': 'data/Delgado_QCM/bare_xtal/',
        'filmfile': 'bare_room_temp',
        'nhplot': [3, 5]
         }

    return sample
