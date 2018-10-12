import numpy as np
# note that the values of fref used for the springpot models are for the
# reference temperature from the original DMA data, which is a bit
# arbitrary


def sbr():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'SBR'

    # specify the DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = '0425161'
    dmadata['Trange'] = [-50, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    # specify the QCM data
    qcmdata = {}
    qcmdata['qcm_T'] = np.array([20, 30, 40])
    qcmdata['qcm_rhog'] = np.array([8.89e7, 7.40e7, 3.90e7])
    qcmdata['rho'] = np.array([0.9])  # sample density in g/cm^3
    qcmdata['qcm_phi'] = np.array([44.7, 46.1, 46.2])
    sample['qcmdata'] = qcmdata

    # now include the AFM data
    sample['afmf'] = 2000
    sample['afmfile'] = '072716_SBR_AFM.xlsx'

    # now add the springpot models
    sp_parms = {}
    sp_parms['typ'] = np.array([1, 2])
    sp_parms['fref_raw'] = 1e7
    sp_parms['fref'] = 6.94e7
    sp_parms['phi'] = np.array([2.3, 55, 2])
    sp_parms['E'] = np.array([1.76e6, 1.12e9, 1.12e9])
    sp_parms['Tref_raw'] = -10
    sp_parms['Tref'] = 20
    sp_parms['Tinf'] = -78.4
    sp_parms['B'] = 481
    sample['sp_parms'] = sp_parms

    return sample


def sbr50():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'NU50'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_50_Duke'
    dmadata['Trange'] = [-40, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    # now add the springpot models
    sp_parms = {}
    sp_parms['typ'] = np.array([1, 2])
    sp_parms['fref_raw'] = 1e7
    sp_parms['fref'] = 6.94e7
    sp_parms['phi'] = np.array([2.3, 55, 2])
    sp_parms['E'] = np.array([1.76e6, 1.12e9, 1.12e9])
    sp_parms['Tref_raw'] = -10
    sp_parms['Tref'] = 20
    sp_parms['Tinf'] = -78.4
    sp_parms['B'] = 481
    sample['sp_parms'] = sp_parms    # specify the QCM data
    qcmdata = {}
    qcmdata['qcm_T'] = np.array([20, 30, 40])
    qcmdata['qcm_rhog'] = np.array([8.89e7, 7.40e7, 3.90e7])
    qcmdata['rho'] = np.array([0.9])  # sample density in g/cm^3
    qcmdata['qcm_phi'] = np.array([44.7, 46.1, 46.2])
    sample['qcmdata'] = qcmdata

    return sample


def sbr51():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'NU51'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_51_Duke'
    dmadata['Trange'] = [-40, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    # now add the springpot models
    sp_parms = {}
    sp_parms['typ'] = np.array([1, 2])
    sp_parms['fref_raw'] = 1e7
    sp_parms['fref'] = 6.94e7
    sp_parms['phi'] = np.array([2.3, 55, 2])
    sp_parms['E'] = np.array([1.76e6, 1.12e9, 1.12e9])
    sp_parms['Tref_raw'] = -10
    sp_parms['Tref'] = 20
    sp_parms['Tinf'] = -78.4
    sp_parms['B'] = 481
    sample['sp_parms'] = sp_parms

    # specify the QCM data
    qcmdata = {}
    qcmdata['qcm_T'] = np.array([20, 30, 40])
    qcmdata['qcm_rhog'] = np.array([8.89e7, 7.40e7, 3.90e7])
    qcmdata['rho'] = np.array([0.9])  # sample density in g/cm^3
    qcmdata['qcm_phi'] = np.array([44.7, 46.1, 46.2])
    sample['qcmdata'] = qcmdata

    return sample


def sbr52():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'NU52'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_52_Duke'
    dmadata['Trange'] = [-40, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    # now add the springpot models
    sp_parms = {}
    sp_parms['typ'] = np.array([1, 2])
    sp_parms['fref_raw'] = 1e7
    sp_parms['fref'] = 6.94e7
    sp_parms['phi'] = np.array([2.3, 55, 2])
    sp_parms['E'] = np.array([1.76e6, 1.12e9, 1.12e9])
    sp_parms['Tref_raw'] = -10
    sp_parms['Tref'] = 20
    sp_parms['Tinf'] = -78.4
    sp_parms['B'] = 481
    sample['sp_parms'] = sp_parms

    # specify the QCM data
    qcmdata = {}
    qcmdata['qcm_T'] = np.array([20, 30, 40])
    qcmdata['qcm_rhog'] = np.array([8.89e7, 7.40e7, 3.90e7])
    qcmdata['rho'] = np.array([0.9])  # sample density in g/cm^3
    qcmdata['qcm_phi'] = np.array([44.7, 46.1, 46.2])
    sample['qcmdata'] = qcmdata

    return sample


def sbr53():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'NU53'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_53_Duke'
    dmadata['Trange'] = [-40, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    # now add the springpot models
    sp_parms = {}
    sp_parms['typ'] = np.array([1, 2])
    sp_parms['fref_raw'] = 1e7
    sp_parms['fref'] = 6.94e7
    sp_parms['phi'] = np.array([2.3, 55, 2])
    sp_parms['E'] = np.array([1.76e6, 1.12e9, 1.12e9])
    sp_parms['Tref_raw'] = -10
    sp_parms['Tref'] = 20
    sp_parms['Tinf'] = -78.4
    sp_parms['B'] = 481
    sample['sp_parms'] = sp_parms

    # specify the QCM data
    qcmdata = {}
    qcmdata['qcm_T'] = np.array([20, 30, 40])
    qcmdata['qcm_rhog'] = np.array([8.89e7, 7.40e7, 3.90e7])
    qcmdata['rho'] = np.array([0.9])  # sample density in g/cm^3
    qcmdata['qcm_phi'] = np.array([44.7, 46.1, 46.2])
    sample['qcmdata'] = qcmdata

    return sample


def sbr54():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'SBR54'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_54_N121_120517'
    dmadata['Trange'] = [-40, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    return sample


def sbr55():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'SBR55'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_52_N121_121817'
    dmadata['Trange'] = [-40, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    return sample


def sbr34():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'SBR35'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_34_Mix_2_min_0_N660_092917'
    dmadata['Trange'] = [-40, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    return sample


def sbr35():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'SBR35'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_35_Mix_4_min_25_N660_092517'
    dmadata['Trange'] = [-40, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    return sample


def sbr36():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'SBR36'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_36_Mix_4_min_15_N660_092517_good'
    dmadata['Trange'] = [-40, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    return sample


def sbr37():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'SBR37'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_37_Mix_4_min_10_N660_092217'
    dmadata['Trange'] = [-40, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    return sample


def sbr38():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'SBR38'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_38_Mix_4_min_1_N660_092217'
    dmadata['Trange'] = [-40, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    return sample


def sbr39():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'SBR39'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_39_Mix_4_min_0_N660_092217'
    dmadata['Trange'] = [-40, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    # specify the QCM data
    qcmdata = {}
    qcmdata['qcm_T'] = np.array([20, 30, 40])
    qcmdata['qcm_rhog'] = np.array([8.89e7, 7.40e7, 3.90e7])
    qcmdata['rho'] = np.array([0.9])  # sample density in g/cm^3
    qcmdata['qcm_phi'] = np.array([44.7, 46.1, 46.2])
    sample['qcmdata'] = qcmdata

    # now add the springpot models
    sp_parms = {}
    sp_parms['typ'] = np.array([1, 2])
    sp_parms['fref_raw'] = 1e7
    sp_parms['fref'] = 6.94e7
    sp_parms['phi'] = np.array([2.3, 65, 2])
    sp_parms['E'] = np.array([1.76e6, 1.12e9, 1.12e9])
    sp_parms['Tref_raw'] = -10
    sp_parms['Tref'] = 20
    sp_parms['Tinf'] = -78.4
    sp_parms['B'] = 481
    sample['sp_parms'] = sp_parms

    return sample


def sbr44():
    # sample crosslinked at NU by David
    sample = {}
    sample['title'] = 'SBR44'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'NU_44_Mix_6_min_0_N660_092917'
    dmadata['Trange'] = [-50, -10]  # Temp. range to include
    sample['dmadata'] = dmadata

    return sample


def PI_34():
    sample = {}
    sample['title'] = '3-4 PI'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'crosslinked0_5_3_4042217_no_transition'
    dmadata['Trange'] = [-20, 30]  # Temp. range to include
    sample['dmadata'] = dmadata

    # specify the QCM data
    qcmdata = {}
    qcmdata['qcm_T'] = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130])
    qcmdata['qcm_rhog'] = np.array([2.3e8, 1.3e8, 7.1e7, 4.0e7, 2.1e7,
                                   1.5e7, 1.1e7, 7.8e6, 4.8e6])
    qcmdata['rho'] = 0.9  # sample density in g/cm^3
    qcmdata['qcm_phi'] = np.array([38.9, 47.0, 53.4, 56.0, 67, 71, 64, 63, 61])
    sample['qcmdata'] = qcmdata

    # now add the springpot models
    sp_parms = {}
    sp_parms['typ'] = np.array([1, 2])
    sp_parms['fref_raw'] = 1.5
    sp_parms['fref'] = 1.42e5
    sp_parms['phi'] = np.array([2, 69, 1.8])
    sp_parms['E'] = np.array([1.2e6, 1.4e9, 1.4e9])
    sp_parms['Tref_raw'] = -10
    sp_parms['Tref'] = 20
    sp_parms['Tinf'] = -57.3
    sp_parms['B'] = 595
    sample['sp_parms'] = sp_parms

    return sample


def sbr2():
    # sample crosslinked by David, used by Matt for AFM
    sample = {}
    sample['title'] = 'SBR-AFM'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = '0630161_sbr'
    dmadata['Trange'] = [-60, 0]  # Temp. range to include
    sample['dmadata'] = dmadata

    # specify starting values for Vogel fits
    sample['Tref_raw'] = -10
    sample['Tinf'] = -100
    sample['B'] = 700

    # now add the springpot models
    sp_parms = {}
    E_rub = 8.5e5
    E_rub_ratio = 10
    phi_rub = 15
    fref = 6e6
    phi_trans = 66
    E_glass = 1.8e9
    phi_glass = 12

    E_rub_shift = E_rub*(E_glass/E_rub)**(phi_rub/phi_trans)

    sp_parms['typ'] = np.array([1, 2, 3])
    sp_parms['fref_raw'] = fref
    sp_parms['phi'] = np.array([0, 0, phi_rub, phi_trans, phi_glass, 0])
    sp_parms['E'] = np.array([E_rub, E_rub*E_rub_ratio, E_rub_shift,
                             E_glass, E_glass, E_glass])
    sample['sp_parms'] = sp_parms

    return sample


def PI_14():
    sample = {}
    sample['title'] = '1-4 PI'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = 'GY_3_03_natsyn_12_21'
    dmadata['Trange'] = [-75, -30]  # Temp. range to include
    sample['dmadata'] = dmadata

    # specify the QCM data
    qcmdata = {}
    qcmdata['qcm_T'] = np.array([20])
    qcmdata['qcm_rhog'] = np.array([3.2e7])
    qcmdata['rho'] = 0.9  # sample density in g/cm^3
    qcmdata['qcm_phi'] = np.array([60])
    sample['qcmdata'] = qcmdata

    # now add the springpot models
    sp_parms = {}
    sp_parms['typ'] = np.array([1, 2])
    sp_parms['fref_raw'] = 9e5
    sp_parms['fref'] = 4.16e8
    sp_parms['phi'] = np.array([3, 76, 2])
    sp_parms['E'] = np.array([1.7e6, 2e9, 2e9])
    sp_parms['Tref_raw'] = -10
    sp_parms['Tref'] = 20
    sp_parms['Tinf'] = -116
    sp_parms['B'] = 762
    sample['sp_parms'] = sp_parms

    return sample


def PBD_14():
    sample = {}
    sample['title'] = 'PBD'

    # specify DMA data
    dmadata = {}
    dmadata['datapath'] = '../data/DMA/'
    dmadata['filename'] = '02_1_pbd'
    dmadata['Trange'] = [-115, -75]  # Temp. range to include
    sample['dmadata'] = dmadata

    # specify the QCM data
    qcmdata = {}
    qcmdata['qcm_T'] = np.array([20])
    qcmdata['qcm_rhog'] = np.array([3e6])
    qcmdata['rho'] = 0.9  # sample density in g/cm^3
    qcmdata['qcm_phi'] = np.array([50])
    sample['qcmdata'] = qcmdata

    # now add the springpot models
    sp_parms = {}
    sp_parms['typ'] = np.array([1, 2])
    sp_parms['fref_raw'] = 100
    sp_parms['fref'] = 6.80e10
    sp_parms['phi'] = np.array([0, 59, 2])
    sp_parms['E'] = np.array([2.8e6, 2e9, 2e9])
    sp_parms['Tref_raw'] = -90
    sp_parms['Tref'] = 20
    sp_parms['Tinf'] = -155
    sp_parms['B'] = 845

    sample['sp_parms'] = sp_parms

    return sample
