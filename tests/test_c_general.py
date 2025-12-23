import numpy as np
from scipy.sparse import csr_array, spdiags, random
from spsolve import spsolve
from scipy.sparse.linalg import spsolve as scipy_spsolve

from .spsolve_test_base import random_b, reset_rng, __REPEAT__

def test_c_general_solve():
    # Create a random sparse matrix
    np.random.seed(42)
    n = 1000
    # Create a density 0.1 matrix
    A = csr_array(spdiags(5+np.random.rand(n), 0, n, n) + random(n, n, density=0.1))
    
    b = random_b(n, 1000)
    
    print("Solving with custom spsolve...")
    # Solve using my spsolve
    x = spsolve(A, b.copy())
    
    # Check residual
    res = A @ x - b
    norm = np.linalg.norm(res)
    print(f"Residual norm: {norm}")
    assert norm < 1e-8, f"Residual too high: {norm}"
    
    # Compare with reference
    print("Solving with scipy...")
    x_ref = scipy_spsolve(A, b)
    
    diff = np.linalg.norm(x - x_ref)
    print(f"Difference from Scipy: {diff}")
    assert np.allclose(x, x_ref, atol=1e-6)

if __name__ == "__main__":
    test_full_solve()
