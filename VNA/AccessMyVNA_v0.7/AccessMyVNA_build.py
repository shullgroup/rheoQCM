import os

from cffi import FFI

ffi = FFI()

# ffi.set_source("_vna",
#     None
#     # r'''
#     # #include "AccessMyVNAdll.h"
#     # ''', # header
#     # # The important thing is to inclue libc in the list of libraries we're linking against:
#     # libraries=["./AccessMyVNA_v0.7/AccessMyVNAdll.cpp"],
# )

# with open(r'./VNA/AccessMyVNA_v0.7/AccessMyVNAdll.h') as f:
#     ffi.cdef(f.read())
# #     # print(f.read())

ffi.cdef('int MyVNAInit(void);')
C = ffi.dlopen(r'./VNA/AccessMyVNA_v0.7/AccessMyVNAdll.dll')

def initvna():
    C.MyVNAInit()

if __name__ == "__main__":
    initvna()
    # ffi.compile()