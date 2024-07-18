import numpy as np
from ..gmps import gmps
def add(psi1, psi2, chi_max, eps, factor=1., compress=True):
    assert(psi1.L == psi2.L)
    Bs = []
    B1 = psi1.Bs[0]
    B2 = psi2.Bs[0]
    Bs.append(np.zeros((1, B1.shape[1]+B2.shape[1], B1.shape[2]), dtype=complex))
    Bs[-1][0, :B1.shape[1], :] = psi1.norm * B1[0, :, :].copy()
    Bs[-1][0, B1.shape[1]:, :] = psi2.norm * factor * B2[0, :, :].copy()
    for i in range(1, psi1.L-1):
        B1 = psi1.Bs[i]
        B2 = psi2.Bs[i]
        Bs.append(np.zeros((B1.shape[0]+B2.shape[0], B1.shape[1]+B2.shape[1], B1.shape[2]), dtype=complex))
        Bs[-1][:B1.shape[0], :B1.shape[1], :] = B1.copy()
        Bs[-1][B1.shape[0]:, B1.shape[1]:, :] = B2.copy()
    B1 = psi1.Bs[psi1.L-1]
    B2 = psi2.Bs[psi1.L-1]
    Bs.append(np.zeros((B1.shape[0]+B2.shape[0], 1, B1.shape[2]), dtype=complex))
    Bs[-1][:B1.shape[0], 0, :] = B1[:, 0, :].copy()
    Bs[-1][B1.shape[0]:, 0, :] = B2[:, 0, :].copy()
    
    # compress the result
    result = gmps.GMPS(Bs, [None]*psi1.L)
    result.canonical = 'none'
    error = 0
    if compress:
        error = result.canonicalize(chi_max, eps)
    return result, error
