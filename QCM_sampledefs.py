# note that the values of fref used for the springpot models are for the
# reference temperature from the original DMA data, which is a bit
# arbitrary


def PMMA_75k_Tom():
    sample = {}
    sample['datadir'] = '../data/Schmitt/'
    sample['barefile'] = 'QCMS01_bare_practice'
    sample['filmfile'] = 'QCMS01_75kPMMA_practice'
    sample['nhcalc'] = ['355']
    sample['nhplot'] = [3, 5]
    return sample


def PMMA_75k():
    sample = {}
    sample['datadir'] = '../data/Taghon/PMMA/'
    sample['barefile'] = 'Angela_bare'
    sample['filmfile'] = 'Angela_PMMA_75k_film2'
    sample['Temp'] = [120, 110, 100, 90, 80, 70, 60, 50, 40]
    sample['nhcalc'] = ['355']
    sample['nhplot'] = [3, 5]
    return sample


def PS_3k_1():
    sample = {}
    sample['datadir'] = '../data/Taghon/PS_3k/'
    sample['barefile'] = 'Bare_3k_1'
    sample['filmfile'] = 'PS_3k_1'
    sample['baretrange'] = [0, 660]
    sample['filmtrange'] = [0, 660]
    sample['Temp'] = [30, 60, 80, 90, 100, 110]
    sample['nhplot'] = [3, 5]
    sample['nhcalc'] = ['355']
    sample['firstline'] = 0
    sample['xlabel'] = 't (min.)'
    return sample


def PS_3k_cool():
    sample = {}
    sample['datadir'] = '../data/Taghon/PS_3k/'
    sample['barefile'] = 'Urma_bare_cool'
    sample['filmfile'] = 'Wilfred_3k_cool'
    sample['baretrange'] = [0, 900]
    sample['filmtrange'] = [0, 900]
    sample['Temp'] = [120, 110, 100, 90, 80, 70, 60, 40]
    sample['nhplot'] = [3, 5]
    sample['nhcalc'] = ['355']
    sample['firstline'] = 0
    sample['xlabel'] = 't (min.)'
    return sample


def PSF_spun():
    sample = {}
    sample['datadir'] = ('../data/Delgado_QCM/test_bare_psf_04122018_' +
                         'spin_coat_film/')
    sample['barefile'] = 'bare_room_temp'
    sample['filmfile'] = 'psf_room_temp'
    sample['nhcalc'] = ['133', '131']
    return sample


def PSF_spun_130():
    sample = {}
    sample['datadir'] = ('../data/Delgado_QCM/test_bare_psf_04122018_' +
                         'spin_coat_film/')
    sample['barefile'] = 'bare_at_130'
    sample['filmfile'] = 'psf_130'
    sample['nhcalc'] = ['133', '131']
    return sample


def PSF_spun_heated_cooled():
    sample = {}
    sample['datadir'] = ('../data/Delgado_QCM/test_bare_psf_04122018_' +
                         'spin_coat_film/')
    sample['barefile'] = 'bare_room_temp'
    sample['filmfile'] = 'psf_room_temp_next_day_041418'

    return sample


def PSF_float():
    sample = {}
    sample['datadir'] = ('../data/Delgado_QCM/test_bare_psf_042618_float_film/')
    sample['barefile'] = 'bare_room_temp'
    sample['filmfile'] = 'psf_room_temp_after_run'
    sample['nhcalc'] = ['133']
    return sample


def PSF_float_130():
    sample = {}
    sample['datadir'] = ('../data/Delgado_QCM/test_bare_psf_042618_float_film/')
    sample['barefile'] = 'bare_130'
    sample['filmfile'] = 'psf_130'
    sample['nhcalc'] = ['133']
    return sample


def PSF_float_heated_cooled():
    sample = {}
    sample['datadir'] = ('../data/Delgado_QCM/test_bare_psf_042618_float_film/')
    sample['barefile'] = 'bare_room_temp'
    sample['filmfile'] = 'psf_room_temp_after_run'
    sample['nhcalc'] = ['133']
    return sample
