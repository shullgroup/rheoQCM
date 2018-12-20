#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:45:35 2018

@author: ken
"""

import sympy as sym
import numpy as np

def get_ZL(uvec):
    rstar = uvec[1]/uvec[0]
    ZL = Z[0]*(1-rstar)/(1+rstar)
    return sym.simplify(ZL)


numlayers = 3  # the number of layers to consider
D = {}
Z = {}
L = {}
S = {}

for i in np.arange(numlayers):
    D[i] = sym.Symbol('D'+str(i))
    Z[i] = sym.Symbol('Z'+str(i))
    L[i] = sym.Matrix([[sym.cos(D[i])+sym.I*sym.sin(D[i]),0],
                     [0, sym.cos(D[i])-sym.I*sym.sin(D[i])]])
    
for i in np.arange(numlayers-1):
    S[i] = sym.Matrix([[1+Z[i+1]/Z[i], 1-Z[i+1]/Z[i]],
                       [1-Z[i+1]/Z[i], 1+Z[i+1]/Z[i]]])
    
Top = sym.Matrix([[1], [1]])

# single layer
uvec = L[0]*Top
print('one layer: ', get_ZL(uvec))

# two layer
uvec = L[0]*S[0]*L[1]*Top
print('two layers: ', get_ZL(uvec))

# three layers
uvec = L[0]*S[0]*L[1]*S[1]*L[2]*Top
print('three layers: ', get_ZL(uvec))



