import os

from cffi import FFI

ffi = FFI()

ffi.set_source("AccesMyVNA_ffi",
    "#include <./VNA/AccessMyVNA_v0.7/AccessMyVNAdll.h>", # header
    # The important thing is to inclue libc in the list of libraries we're linking against:
    libraries=["c"],
)

with open(r'./VNA/AccessMyVNA_v0.7/AccessMyVNAdll.h') as f:
    ffi.cdef(f.read())
    # print(f.read())

if __name__ == "__main__":
    ffi.compile()