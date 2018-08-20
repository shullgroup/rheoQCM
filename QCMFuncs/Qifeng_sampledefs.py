# note that the values of fref used for the springpot models are for the
# reference temperature from the original DMA data, which is a bit
# arbitrary


def sample_dict():
    sample = {}  # individual sample dictionaries get added to sample

    #  20180502
    samplename = 'cryt_2_BCB_air'
    sample[samplename] = {
    'samplename': 'cryt_2_BCB_air',
    'datadir': '20180502/',
    'barefile': 'cryt_2_bare_air',
    'filmfile': 'cryt_2_BCB_air',
    'filmtrange': [1000, 10000],
    'nhcalc': ['355', '353'],
    'nhplot': [1, 3, 5]
    }

    samplename = 'cryt_2_BCB_LN2'
    sample[samplename] = {
    'samplename': 'cryt_2_BCB_LN2',
    'datadir': '20180502/',
    'barefile': 'cryt_2_bare_LN2',
    'filmfile': 'cryt_2_BCB_LN2',
    # 'filmtrange': [4000, 5000],
    'nhcalc': ['355', '353'],
    'nhplot': [1, 3, 5]
    }

    samplename = 'cryt_2_BCB_air_after_LN2'
    sample[samplename] = {
    'samplename': 'cryt_2_BCB_air_after_LN2',
    'datadir': '20180502/',
    'barefile': 'cryt_2_bare_air',
    'filmfile': 'DGEBA-cryt_2_BCB_air_after_LN2',
    # 'filmtrange': [4000, 5000],
    'nhcalc': ['355', '353'],
    'nhplot': [1, 3, 5]
    }

    #  20180629
    samplename = 'DGEBA-Jeffamine2000_RT'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT',
    'datadir': '20180629/',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT',
    'firstline': 50,
    # 'filmtrange': [4000, 5000],
    'nhcalc': ['355', '353'],
    'nhplot': [1, 3, 5]
    }

    #  20180711
    samplename = 'DGEBA-Jeffamine230_RT'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine230_RT',
    'datadir': '20180711/',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine230_RT',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'nhcalc': ['355'],
    'nhplot': [3, 5]
    }

    #  20180713
    samplename = 'DGEBA-Jeffamine230_RT_2'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine230_RT_2',
    'datadir': '20180713/',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine230_RT_2',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'nhcalc': ['355'],
    'nhplot': [3, 5]
    }

    #  20180724 1:1 Good
    samplename = 'DGEBA-Jeffamine230_RT_3'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine230_RT_3',
    'datadir': '20180724/',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine230_RT_3',
    'firstline': 1,
    'filmtrange': [500, 10000],
    'nhcalc': ['355'],
    'nhplot': [3, 5]
    }

    #  20180726 2:1 too thick
    samplename = 'DGEBA-Jeffamine230_RT_4'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine230_RT_4',
    'datadir': '20180726/',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine230_RT_4',
    'firstline': 2,
    # 'filmtrange': [1, 10],
    'nhcalc': ['355'],
    'nhplot': [3, 5]
    }

    #  20180727 2:1 Good
    samplename = 'DGEBA-Jeffamine230_RT_5'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine230_RT_5',
    'datadir': '20180727/',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine230_RT_5',
    'firstline': 2,
    'filmtrange': [500, 50000],
    'nhcalc': ['355'],
    'nhplot': [3, 5]
    }

    #  20180803 2:1 
    samplename = 'DGEBA-Jeffamine2000_RT_2'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT_2',
    'datadir': '20180803/',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT_2',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'nhcalc': ['355'],
    'nhplot': [1, 3, 5]
    }

    #  20180806 2:1 
    samplename = 'DGEBA-Jeffamine2000_RT_3'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT_3',
    'datadir': '20180806/',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT_3',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'nhcalc': ['355', '353'],
    'nhplot': [1, 3, 5]
    }

    #  20180807 2:1 
    samplename = 'DGEBA-Jeffamine2000_RT_3_2'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT_3_2',
    'datadir': '20180807/',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT_3_2',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'nhcalc': ['355', '353'],
    'nhplot': [1, 3, 5]
    }

    #  20180808 2:1 
    samplename = 'DGEBA-Jeffamine2000_RT_4'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT_4',
    'datadir': '20180808/',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT_4',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'nhcalc': ['355', '353'],
    'nhplot': [1, 3, 5]
    }

    return sample
