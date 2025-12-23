import os
import sys

if sys.platform == "win32":
    # MKL pip package on Windows puts DLLs in <sys.prefix>/Library/bin
    mkl_path = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(mkl_path): os.add_dll_directory(mkl_path)
elif sys.platform == "linux":
    # On Linux, MKL libs are typically in <sys.prefix>/lib
    mkl_path = os.path.join(sys.prefix, "lib")
    if os.path.isdir(mkl_path): os.add_dll_directory(mkl_path)
    # Note: LD_LIBRARY_PATH or RPATH usually handles this.
    # If not, one might need ctypes.CDLL(os.path.join(mkl_path, "libmkl_rt.so"))
    print("Tried to add mkl path: ", mkl_path)


from .spsolve import spsolve
from .spsolve_triangular import spsolve_triangular
from .spsolve_pardiso import solve as spsolve_pardiso