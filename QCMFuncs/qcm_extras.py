#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:04:00 2018

@author: ken
"""
import numpy as np
Zq = 8.84e6  # shear acoustic impedance of at cut quartz
f1 = 5e6  # fundamental resonant frequency


def thin_film_gamma(n, drho, jdprime_rho):
    return 8*np.pi ** 2*n ** 3*f1 ** 4*drho ** 3*jdprime_rho / (3*Zq)
    # same master expression, replacing grho3 with jdprime_rho3


def grho3(jdprime_rho3, phi):
    return np.sin(np.deg2rad(phi))/jdprime_rho3

