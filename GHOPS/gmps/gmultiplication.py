import grassmanntn as gtn
from ..gmps import gmps
import numpy as np

def GNmultiply(psi,Ws,chi_max,eps,inplace=False,compress= True):
    assert(len(Ws)==psi.L)
    Ws= gtn.dense(data=Ws,statistics=(-1,-1,-1,-1))
    Bs = [None]*psi.L
    Bs[0] = gtn.einsum('ijk,mkn->ijmn',psi.Bs[0],Ws[0,:,:,:])
    Bs[0] = Bs[0].join_legs('i(jk)m')
    for i in range(1, psi.L-1):
        Bs[i] = gtn.einsum('ijk,mnkl->ijmnl',psi.Bs[i], Ws[i])
        Bs[i] = gtn.einsum('ijklm->ikjlm',Bs[i])
        Bs[i] = Bs[i].join_legs('(ij)(kl)m')
    Bs[-1] = gtn.einsum('ijk,mkn->ijmn')
    Bs[-1] = gtn.einsum('ijkl->ikjl',Bs[-1])
    Bs[-1]= Bs[-1].join_legs('(ij)km')
    if inplace == True:
        psi.Bs = Bs
        psi.canonical = False
        if compress:
            return psi.canonicalize(chi_max, eps)
        return 0
    else:
        result = gmps.GMPS(Bs, [None]*psi.L)
        result.norm = psi.norm
        result.canonical = False
        error = 0
        if compress:
            error = result.canonicalize(chi_max, eps)
        return result, error