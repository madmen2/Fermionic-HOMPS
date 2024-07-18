import numpy as np
import grassmanntn as gtn
from scipy.linalg import svd
import sparse as sp

class GMPS:
    """ 
    Class representing a Grassmannian Matrix Product State (GMPS)
    
    Attributes
    ----------
    Bs : List of np.darray
        List of the grassmann tensors. One tensor for each physical site. Each site has three indices. One
        physical index and two Grassmann indices. 
    L: int
    number of sites

    canonical : str
        one of {'none', 'left', 'right'}. Describes if the MPS is either not in
        canonical form, in left-canonical form, or in right-canonical form.
    norm : complex
        additional overall scalar factor of the MPS
    
    statistics: list
        List of the statistics of the sites. Can be either 'bosonic' or 'fermionic'. +1 is a non-conjugate index
        -1 is a conjugate index and 0 is a bosonic index. 
    encoder : str
        The encoding of the Grassmannian tensors. Can be either 'Canonical' or 'Parity-Preserving'
    
    """
    
    def __init__(self, Bs, canonical ='None',encoder='Parity-Preserving',norm=1.0):
        self.Bs = Bs
        self.L = len(Bs)
        self.canonical = canonical
        self.norm = norm
        self.encoder = encoder
        #generate statistics 
        self.statistics = [0]*self.L
        grassman_list=[]
        for i in range(self.L):
            grassman_list.append(gtn.dense(data=self.Bs[i],statistics=(-1,1,1)))
        self.Bs =grassman_list
        
    def get_theta_2(self, i):
        contract =  gtn.einsum('ijk,jmn->imnk',self.Bs[i],self.Bs[i+1])
        return gtn.einsum('imnk->imkn',contract)
    
    def copy(self):
        return GMPS([np.copy(B) for B in self.Bs], self.canonical, self.encoder, self.norm)
    
    def get_bond_dims(self):
        return [self.Bs[i].shape[1] for i in range(self.L-1)]
    
    def get_average_bond_dim(self):
        return np.mean(self.get_bond_dims())
    
    def get_max_bond_dim(self):
        return np.max(self.get_bond_dims())
    
    def compute_left_environment(self):
        
        if self.canonical == 'left':
            return [np.eye(self.Bs[i].shape[0], self.Bs[i].shape[0]) for i in range(self.L)]
        contr = np.pones((1,1))
        contr= gtn.dense(data=contr, statistics=(1,1))
        envs = [contr.copy()]
        for i in range(self.L-1):
            contr = gtn.einsum('ij,ikl->jkl',contr,self.Bs[i])
            contr=gtn.einsum('ijk,imk->jm',contr,contr.hconjugate('|ijk'))
            envs.append(contr)
        return envs
    
    
    def compute_right_environment(self):
        
        if self.canonical == 'right':
            return [np.eye(self.Bs[i].shape[0],self.Bs[i].shape[0]) for i in range(self.L)]
        contr = np.ones((1,1))
        contr= gtn.dense(data=contr, statistics=(1,1))
        envs = [None]*self.L
        envs[self.L-1] = contr.copy()
        for i in range(self.L-1,0,-1):
            contr = gtn.einsum('ij,kim->kjm',contr,self.Bs[i])
            contr = gtn.einsum('ijk,mij->km',contr,contr.hconjugate('|ijk'))
            envs[i-1]=contr.copy()
        return envs
    
    def conver_grassmann_varriables(op):
        basis = ["phi_0","conj0","phi_1","conj1","phi_2","conj2","phi_3","conj3"]
        coeff,coords = op[0][0].get_coeff(basis=["phi_0","conj0","phi_1","conj1","phi_2","conj2","phi_3","conj3"])
        
    
    def site_expectation_value(self,op):
        
        left_env = self.compute_left_environment()
        right_env = self.compute_right_environment()
        
        norm = gtn.einsum('ij,kim->jmn',right_env[0],self.Bs[0])
        norm=gtn.einsum('ijk,mik->jm', norm,self.Bs[0].hconjugate('|ijk'))
        norm= norm.item()
        
        results=[]
        for i in range(self.L):
            contr = gtn.einsum('ij,kin->jkn',left_env[i],self.Bs[i])
            contr = gtn.einsum('ijk,km ->ijm',contr,op)
            contr=gtn.einsum('ijk,imk->jm',contr,self.Bs[i].hconjugate('|ijk'))
            contr=gtn.einsum('ij,ij',contr,right_env[i])
            results.append(contr.item()/norm)
        return results
    

    
    @staticmethod
    def init_GHOMPS_GMPS(psi0,N_bath,N_trunc,chi_max=1):
        Bs = [None]*(N_bath +1)
        chi = min(psi0.size,chi_max)
        B_physical = np.zeros([1,chi,psi0.size],dtype= complex)
        
        B_physical[0,0,:] = psi0
        B_physical = gtn.dense(data=B_physical,statistics=(-1,1,1),encoder='parity-preserving',format='standard')
        Bs[0] = B_physical
        for i in range(int(N_bath)):
            chi_prime = min(chi_max, min(chi*N_trunc, N_trunc**(N_bath-i-1)))
            B_bath = np.zeros([chi, chi_prime, N_trunc], dtype=complex)
            B_bath[0, 0, 0] = 1.
            B_bath= gtn.dense(data=B_bath,statistics=(-1,1,1),encoder='parity-preserving',format='standard')
            Bs[i+1] = B_bath
        statistics = (-1,1,1)
        gmps=[]
        for i in range(len(Bs)):
            gmps.append(gtn.dense(data=Bs[i],statistics=statistics,encoder='parity-preserving',format='standard'))
        return GMPS(gmps)
   
    def sanity_check(self):
        """
        Checks if all bond dimensions match up and if self.canonical is set correctly
        """
        # Check if the bond dimensions match up
        assert(self.Bs[0].shape[0] == 1)
        assert(self.Bs[-1].shape[1] == 1)
        for i in range(self.L-1):
            assert(self.Bs[i].shape[1] == self.Bs[i+1].shape[0])
        # Check for canonicalization
        if self.canonical == 'left':
            assert(self.is_left_canonical())
        elif self.canonical == 'right':
            assert(self.is_right_canonical())
    
    def is_right_canonical(self):
        contr= np.ones((1,1))
        contr = gtn.dense(data=contr, statistics=(1,1))
        for i in range(self.L-1,0,-1):
            contr = gtn.einsum('ij,kim->jkm',contr,self.Bs[i])
            contr= gtn.einsum('ijk,mik->jm',contr,contr.hconjugate('|ijk'))
            if not np.all(np.isclose(contr, np.eye(contr.shape[0]))):
                return False
        return True        
        
    def to_state_vector(self):
        contr = gtn.einsum('ij->ji', self.Bs[0][0])
        for i in range(1, self.L):
            contr = gtn.einsum('ij,jkl->ikl', contr, self.Bs[i])
            contr = gtn.einsum('ijk->ikj', contr) 
            contr= contr.join_legs('(ij)k')    
        
        return self.norm * contr[:, 0]

        
    def is_left_canonical(self):
        contr = np.ones((1,1))
        contr = gtn.dense(data=contr, statistics=(1,1))
        for i in range(self.L-1):
            contr= gtn.einsum('ij,imn>jmn',contr,self.Bs[i])
            contr = gtn.einsum('ijk,imk->jm', contr, self.Bs[i].hconjugate('|ijk'))
            if not np.all(np.isclose(contr, np.eye(contr.shape[0]))):
                return False
        return True
    
    def canonicalize(self,chi_max=0,eps=0):
        #sweep left ro right using SVDs to compute singular values
        for i in range(self.L-1):
            B=  gtn.einsum('ijk->ikj',B)
            B = B.join_legs('(ij)k')
            U,S,V,norm_factor,error_temp = split_and_truncate(B,chi_max,eps)
            chi_new= S.size
            V = np.array(V).reshape(chi_new,chi_vr,chi_i)
            V = np.transpose(V, (0, 2, 1))
            V = gtn.dense(data=V,statistics=(-1,1,1))
            self.Bs[i] = V
            B = self.Bs[i+1]
            B= gtn.einsum('ijk,jm->ikm', B,U)
            B = gtn.einsum('ijk,km->ijm', B, np.diag(S))
            B = gtn.einsum('ijk->ikj', B)
            # Sweep right to left using SVDs to compute singular values
        error =0.0
        for i in range(self.L-1,0,-1):
            chi_vl,chi_vr,chi_i=B.shape
            B=  gtn.einsum('ijk->ikj',B)
            B= gtn.join_legs('i(jk)')
            U,S,V, norm_factor,error_temp = split_and_truncate(B,chi_max,eps)
            chi_new = S.size
            self.norm *= norm_factor
            error += error_temp
            V = np.array(V).reshape(chi_new,chi_vr,chi_i)
            V = np.transpose(V, (0, 2, 1))
            V = gtn.dense(data=V,statistics=(-1,1,1))
            self.Bs[i] = V 
            B = self.Bs[i-1]
            B= gtn.einsum('ijk,jm->ikm', B,U)
            B = gtn.einsum('ijk,km->ijm', B, np.diag(S))
            B = gtn.einsum('ijk->ikj', B)
        self.Bs[0] = B 
        self.canonical = 'right'
        if error > 1.e-3:
            print("[WARNING]: Large error",error, " > 1.e-3 detected")

def split_and_truncate(A,chi_max=0,eps=0):

    U,S,V = A.svd('i|j')   
    if chi_max>0:
        chi_new = min(chi_max,np.sum(np.array(S)>eps))   
    else:
        chi_new= np.sum(np.array(S)>=eps)
    assert chi_new>=1
    piv = np.argsort(S)[::-1][:chi_new]
    error = np.sum(S[chi_new:]**2)
    U,S,V = U[:,piv], S[piv], V[piv,:] 
    #renormalize
    norm = np.linalg.norm(S)    
    if norm < 1.e-7:
        print("[WARNING]: Small singular Values, norm(S) < 1e-7")
    S = S/norm
    
    return U, S, V, norm, error



