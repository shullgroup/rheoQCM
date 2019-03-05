# note that the values of fref used for the springpot models are for the
# reference temperature from the original DMA data, which is a bit
# arbitrary


def sample_dict():
    sample = {}  # individual sample dictionaries get added to sample

    # 20190201 D400_5_N2_CO2 samp (CO2)
    samplename = 'D400_5_N2_CO2__samp'
    sample[samplename] = {
    'samplename': 'D400_5_N2_CO2__samp',
    'datadir': '20190201',
    'filetype': 'h5',
    'filmchn': 'samp',
    'filmfile': 'D400_5_N2_CO2',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['133', '353', '355'],
    'nhplot': [1, 3, 5]
    }
    # 20190201 D400_5_N2_CO2 ref (N2)
    samplename = 'D400_5_N2_CO2__ref'
    sample[samplename] = {
    'samplename': 'D400_5_N2_CO2__ref',
    'datadir': '20190201',
    'filetype': 'h5',
    'filmchn': 'ref',
    'filmfile': 'D400_5_N2_CO2',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['353', '355'],
    'nhplot': [1, 3, 5]
    }


    # 20190129 D400_4_N2_RH samp
    samplename = 'D400_4_N2_RH__samp'
    sample[samplename] = {
    'samplename': 'D400_4_N2_RH__samp',
    'datadir': '20190129',
    'filetype': 'h5',
    'filmchn': 'samp',
    'filmfile': 'D400_4_N2_RH',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['353', '355'],
    'nhplot': [1, 3, 5]
    }
    # 20190129 D400_4_N2_RH ref
    samplename = 'D400_4_N2_RH__ref'
    sample[samplename] = {
    'samplename': 'D400_4_N2_RH__ref',
    'datadir': '20190129',
    'filetype': 'h5',
    'filmchn': 'ref',
    'filmfile': 'D400_4_N2_RH',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['353', '355'],
    'nhplot': [1, 3, 5]
    }




    # 20190118 temp_ramp_23_143_23 samp
    samplename = 'temp_ramp_23_143_23__samp'
    sample[samplename] = {
    'samplename': 'temp_ramp_23_143_23__samp',
    'datadir': '20190118',
    'filetype': 'h5',
    'filmchn': 'samp',
    'filmfile': 'temp_ramp_23_143_23',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['353', '355'],
    'nhplot': [1, 3, 5]
    }

    # 20190118 temp_ramp_23_143_23.h5
    samplename = 'temp_ramp_23_143_23__ref'
    sample[samplename] = {
    'samplename': 'temp_ramp_23_143_23__ref',
    'datadir': '20190118',
    'filetype': 'h5',
    'filmchn': 'ref',
    'filmfile': 'temp_ramp_23_143_23',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['353', '355'],
    'nhplot': [1, 3, 5]
    }




    # 20181029 DGEBA-PACM_RT_2_cured 
    samplename = 'DGEBA-PACM_RT_2_cured'
    sample[samplename] = {
    'samplename': 'DGEBA-PACM_RT_2_cured',
    'datadir': '20181029',
    'barefile': 'DGEBA-PACM_RT_2_bare_air',
    'filmfile': 'DGEBA-PACM_RT_2_cured',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['353', '355'],
    'nhplot': [1, 3, 5]
    }

    # 20181029 DGEBA-PACM_RT_cured 
    samplename = 'DGEBA-PACM_RT_cured'
    sample[samplename] = {
    'samplename': 'DGEBA-PACM_RT_cured',
    'datadir': '20181029',
    'barefile': 'DGEBA-PACM_RT_bare_air',
    'filmfile': 'DGEBA-PACM_RT_cured',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['353', '355'],
    'nhplot': [1, 3, 5]
    }

    # 20181029 DGEBA-Jeffamine400_RT_3_cured 
    samplename = 'DGEBA-Jeffamine400_RT_3_cured'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine400_RT_3_cured',
    'datadir': '20181029',
    'barefile': 'DGEBA-Jeffamine400_RT_3_bare_air',
    'filmfile': 'DGEBA-Jeffamine400_RT_3_cured',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['353', '355'],
    'nhplot': [1, 3, 5]
    }
    
    # 20181029 DGEBA-Jeffamine400_RT_2_cured 
    samplename = 'DGEBA-Jeffamine400_RT_2_cured'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine400_RT_2_cured',
    'datadir': '20181029',
    'barefile': 'DGEBA-Jeffamine400_RT_2_bare_air',
    'filmfile': 'DGEBA-Jeffamine400_RT_2_cured',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['353', '355'],
    'nhplot': [1, 3, 5]
    }

    
    # 20181029 DGEBA-Jeffamine230_RT_5_cured 
    samplename = 'DGEBA-Jeffamine230_RT_5_cured'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine230_RT_5_cured',
    'datadir': '20181029',
    'barefile': 'DGEBA-Jeffamine230_RT_5_bare_air',
    'filmfile': 'DGEBA-Jeffamine230_RT_5_cured',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['353', '355'],
    'nhplot': [1, 3, 5]
    }
    
    # 20181029 DGEBA-Jeffamine230_RT_3_cured 
    samplename = 'DGEBA-Jeffamine230_RT_3_cured'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine230_RT_3_cured',
    'datadir': '20181029',
    'barefile': 'DGEBA-Jeffamine230_RT_3_bare_air',
    'filmfile': 'DGEBA-Jeffamine230_RT_3_cured',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    # 'filmindex': range(0, 156, 10),
    # 'xscale': 'log',
    'nhcalc': ['353', '355'],
    'nhplot': [1, 3, 5]
    }

    # 20180917 2:1 
    samplename = 'DGEBA-Jeffamine2000_RT_7'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT_7',
    'datadir': '20180917',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT_7',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    'filmindex': list(range(0, 156, 10)) + [156, 158, 160, 162, 163, 164],
    # 'xscale': 'log',
    'nhcalc': ['133'],
    'nhplot': [1, 3,]
    }

    # 20180914 2:1 
    samplename = 'DGEBA-Jeffamine2000_RT_6'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT_6',
    'datadir': '20180914',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT_6',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    'filmindex': range(3, 50, 5),
    'xscale': 'log',
    'nhcalc': ['133', '355', '353'],
    'nhplot': [1, 3, 5]
    }

    # 20180828 2:1 
    samplename = 'DGEBA-Jeffamine2000_RT_5'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT_5',
    'datadir': '20180828',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT_5',
    'firstline': 0,
    # 'filmtrange': [1, 10],
    'xscale': 'log',
    'nhcalc': ['133', '355', '353'],
    'nhplot': [1, 3, 5]
    }

    #  20180824 2:1 
    samplename = 'DGEBA-PACM_RT_2'
    sample[samplename] = {
    'samplename': 'DGEBA-PACM_RT_2',
    'datadir':    '20180824',
    'barefile':   'bare_air',
    'filmfile':   'DGEBA-PACM_RT_2',
    'firstline':  0,
    # 'filmtrange': [1, 10],
    'filmindex':  list(range(0,51, 5)) + list(range(60, 121, 10)) + [128],
    'xscale': 'log',
    'nhcalc':     ['133', '355'],
    'nhplot':     [1, 3, 5]
    }

    #  20180823 2:1 
    samplename = 'DGEBA-PACM_RT'
    sample[samplename] = {
    'samplename': 'DGEBA-PACM_RT',
    'datadir':    '20180823',
    'barefile':   'bare_air',
    'filmfile':   'DGEBA-PACM_RT',
    'firstline':  0,
    # 'filmtrange': [1, 10],
    'filmindex':  list(range(0, 79, 5)) + [78],
    'xscale': 'log',
    'nhcalc':     ['355'],
    'nhplot':     [1, 3, 5]
    }

    #  20180817 2:1 
    samplename = 'DGEBA-Jeffamine400_RT_3'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine400_RT_3',
    'datadir':    '20180817',
    'barefile':   'bare_air',
    'filmfile':   'DGEBA-Jeffamine400_RT_3',
    'firstline':  0,
    # 'filmtrange': [1, 10],
    'filmindex':  list(range(0,31, 5)) + [32, 33, 34] + list(range(35, 100, 5)) + [102, 103],
    'xscale': 'log',
    'nhcalc':     ['133', '355', ],
    'nhplot':     [1, 3, 5]
    }

    #  20180814 2:1 
    samplename = 'DGEBA-Jeffamine400_RT_2'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine400_RT_2',
    'datadir': '20180814',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine400_RT_2',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'filmindex':  list(range(0, 90, 5)) + [91, 92],
    'xscale': 'log',
    'nhcalc': ['355', '353'],
    'nhplot': [1, 3, 5]
    }

    #  20180813 2:1 
    samplename = 'DGEBA-Jeffamine400_RT'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine400_RT',
    'datadir': '20180813',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine400_RT',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'xscale': 'log',
    'nhcalc': ['355', '353'],
    'nhplot': [1, 3, 5]
    }

#  20180810 2:1 
    samplename = 'DGEBA-Jeffamine2000_RT_4_2'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT_4_2',
    'datadir': '20180810',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT_4_2',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'xscale': 'log',
    'nhcalc': ['133', '355', '353'],
    'nhplot': [1, 3, 5]
    }

    #  20180808 2:1 
    samplename = 'DGEBA-Jeffamine2000_RT_4'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT_4',
    'datadir': '20180808',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT_4',
    'firstline': 1,
    'filmtrange': [0, 1000],
    'xscale': 'log',
    'nhcalc': ['133', '355'],
    'nhplot': [1, 3, 5]
    }

    #  20180807 2:1 
    samplename = 'DGEBA-Jeffamine2000_RT_3_2'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT_3_2',
    'datadir': '20180807',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT_3_2',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'xscale': 'log',
    'nhcalc': ['133', '355', '353'],
    'nhplot': [1, 3, 5]
    }

    #  20180806 2:1 
    samplename = 'DGEBA-Jeffamine2000_RT_3'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT_3',
    'datadir': '20180806',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT_3',
    'firstline': 1,
    'filmtrange': [100, 1000],
    'xscale': 'log',
    'nhcalc': ['133', '355', '353'],
    'nhplot': [1, 3, 5]
    }

    #  20180803 2:1 
    samplename = 'DGEBA-Jeffamine2000_RT_2'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT_2',
    'datadir': '20180803',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT_2',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'xscale': 'log',
    'nhcalc': ['133', '355'],
    'nhplot': [1, 3, 5]
    }

    #  20180727 2:1 Good
    samplename = 'DGEBA-Jeffamine230_RT_5'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine230_RT_5',
    'datadir': '20180727',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine230_RT_5',
    'firstline': 2,
    # 'filmtrange': [500, 50000],
    'filmindex':  list(range(0, 96, 5)) + [96, 97],
    'xscale': 'log',
    'nhcalc': ['133', '355'],
    'nhplot': [1, 3, 5]
    }

    #  20180726 2:1 too thick
    samplename = 'DGEBA-Jeffamine230_RT_4'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine230_RT_4',
    'datadir': '20180726',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine230_RT_4',
    'firstline': 2,
    # 'filmtrange': [1, 10],
    'xscale': 'log',
    'nhcalc': ['355'],
    'nhplot': [3, 5]
    }

    #  20180724 1:1 Good
    samplename = 'DGEBA-Jeffamine230_RT_3'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine230_RT_3',
    'datadir': '20180724',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine230_RT_3',
    'firstline': 0,
    # 'filmtrange': [500, 10000],
    'filmindex':  list(range(0, 130, 5)) +  [140, 145, 149, 150],
    'xscale': 'log',
    'nhcalc': ['133', '355'],
    'nhplot': [1, 3, 5]
    }

    #  20180713
    samplename = 'DGEBA-Jeffamine230_RT_2'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine230_RT_2',
    'datadir': '20180713',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine230_RT_2',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'xscale': 'log',
    'nhcalc': ['355'],
    'nhplot': [3, 5]
    }

    # %%  20180711
    samplename = 'DGEBA-Jeffamine230_RT'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine230_RT',
    'datadir': '20180711',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine230_RT',
    'firstline': 1,
    # 'filmtrange': [1, 10],
    'xscale': 'log',
    'nhcalc': ['355'],
    'nhplot': [3, 5]
    }

    # %%  20180629
    samplename = 'DGEBA-Jeffamine2000_RT'
    sample[samplename] = {
    'samplename': 'DGEBA-Jeffamine2000_RT',
    'datadir': '20180629',
    'barefile': 'bare_air',
    'filmfile': 'DGEBA-Jeffamine2000_RT',
    'firstline': 50,
    'filmtrange': [0, 1000],
    'xscale': 'log',
    'nhcalc': ['133', '355', '353'],
    'nhplot': [1, 3, 5]
    }

    samplename = 'cryt_2_BCB_air_after_LN2'
    sample[samplename] = {
    'samplename': 'cryt_2_BCB_air_after_LN2',
    'datadir': '20180502',
    'barefile': 'cryt_2_bare_air',
    'filmfile': 'DGEBA-cryt_2_BCB_air_after_LN2',
    # 'filmtrange': [4000, 5000],
    'nhcalc': ['355', '353'],
    'nhplot': [1, 3, 5]
    }

    samplename = 'cryt_2_BCB_LN2'
    sample[samplename] = {
    'samplename': 'cryt_2_BCB_LN2',
    'datadir': '20180502',
    'barefile': 'cryt_2_bare_LN2',
    'filmfile': 'cryt_2_BCB_LN2',
    # 'filmtrange': [4000, 5000],
    'nhcalc': ['355', '353'],
    'nhplot': [1, 3, 5]
    }

    #  20180502
    samplename = 'cryt_2_BCB_air'
    sample[samplename] = {
    'samplename': 'cryt_2_BCB_air',
    'datadir': '20180502',
    'barefile': 'cryt_2_bare_air',
    'filmfile': 'cryt_2_BCB_air',
    'filmtrange': [1000, 10000],
    'nhcalc': ['355', '353'],
    'nhplot': [1, 3, 5]
    }

    return sample