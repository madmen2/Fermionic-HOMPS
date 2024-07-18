import numpy as np
import grassmanntn as gtn
from ..gmps import gmps
from scipy.sparse.linalg import expm_multiply

class GTDVP_Engine: 
    
    def __init__(self,psi,model,dt,chi_max,eps,N_trunc):
        self.psi = psi
        self.model = model
        self.dt=dt
        self.chi_max = chi_max  
        self.eps=eps
        self.N_trunc = N_trunc
        self.LPs= [None]*psi.L
        self.RPs= [None]*psi.L
        D = self.model.H_mpo[0].shape[0]
        chi =self.psi.Bs[0].shape[0]
        LP = np.zeros([chi, D, chi], dtype="float")  # vL wL* vL*
        RP = np.zeros([chi, D, chi], dtype="float")  # vR* wR* vR
        LP[:, 0, :] = np.eye(chi)
        RP[:, D - 1, :] = np.eye(chi)
        LP =gtn.dense(data=LP, statistics=(-1,1,1))
        RP =gtn.dense(data=RP, statistics=(-1,1,1))
        self.LPs[0] = LP
        self.RPs[-1] = RP
        # initialize necessary RPs
        for i in range(psi.L - 1, 0, -1):
            self.update_RP(i) 
    
    def convert_grassmann(self,op):
        #4x4 matrix
        first=op[0,3,:,:]
        
        
            
    def update_RP(self,i):
        j =i-1
        B = self.psi.Bs[i]
        RP = self.RPs[i]
        Bc = B.hconjugate('i|jk')
        W=self.model.H_mpo[i]

        """       #print(W[0,3,:,:][0])
        first = W[0,3,:,:]
        print()
        coords=[]
        #create 4x4 matrix
        first_matrix = np.eye(self.N_trunc,self.N_trunc,dtype=object)
        for i in range(self.N_trunc):
            for j in range(self.N_trunc):
                coeffs,coord=first[i][j].get_coeff(basis=[f"phi_{j}"])
                first_matrix[i][j]=coeffs[0]
        W[0,3,:,:]=first_matrix
        print(type(W[0,3,:,:][0][0]))  """
        W = gtn.dense(data=W, statistics=(-1,-1,-1,-1))
        RP = gtn.einsum('ijk,kmn->ijmn',B,RP)
        RP = gtn.einsum('ijkm,njfk->imnf', RP,W)
        RP = gtn.einsum('ijkm,bjm->ikb',RP,Bc)
        self.RPs[j] = RP
        
    def update_LP(self,i):
        j = 1+1
        LP= self.LPs[i]
        B = self.psi.Bs[i]
        Bc = B.hconjugate('i|jk')
        print(Bc)
        W=self.model.H_mpo[i]
        W = gtn.dense(data=W, statistics=(-1,-1,-1,-1))   
        LP = gtn.einsum('ijk,kmn->ijmn',LP,B)
        LP=gtn.einsum('ijkm,bigm->jkbg',W,LP)
        LP = gtn.einsum('ijk,mikn->jmn',Bc,LP)
        self.Lps[j] = LP
        

        

class GTVP1_Engine(GTDVP_Engine):
    def __init__(self,psi,model,dt,N_trunc,chi_max=0,eps=0,mode='svd'):
        super().__init__(psi,model,dt,chi_max,eps,N_trunc)
        self.mode=mode
        
        
    def compute_Heff_Twosite(LP,RP,W1,W2):
        W1= gtn.dense(data=W1, statistics=(-1,-1,-1,-1))
        W2= gtn.dense(data=W2, statistics=(-1,-1,-1,-1))
        result = gtn.einsum('ijk,jlmn->iklmn',LP,W1)
        result = gtn.einsum('ijklm,kpqr - >ijlmpqr',result,W2)
        result = gtn.einsum('ijklmno,pmq->ijklopq',result,RP)
        result = gtn.einsum('ijklmnop->ikmpjlno',result)
        result = result.join_legs('(ijkl)(mnop)')
    
        return result
    
    
    def compute_Heff_Onesite(LP,RP,W):
        W = gtn.dense(data=W, statistics=(-1,1,1,-1),format='matrix')
        result = gtn.einsum('ijk,jmno->ikmno',LP,W)  
        result = gtn.einsum('jklmn,plq->jkmnpq',result,RP)
        result = gtn.einsum('ijklmn->inkjml',result)
        result = result.join_legs('(ijk)(lmn)',make_format = 'matrix', intermediate_stat=(-1,-1))
        return result
    
    def compute_Heff_zero_site(LP,RP):
        result = gtn.einsum('ijk,mjn->ikmn',LP,RP)
        result = gtn.einsum('ijkm->imjk',result)
        result.info()
        result = result.join_legs('(im)(jk)') 
        return result
    
    def sweep(self):
        for i in range(self.psi.L-1):
            self.update_site(i)
            self.update_bond(i,sweep_right=True)
        self.update_site(self.psi.L - 1)
        # sweep from right to left
        for i in range(self.psi.L - 1, 0, -1):
            self.update_site(i)
            self.update_bond(i, sweep_right=False)
        # update first site
        self.update_site(0)
        

    def update_site(self,i):
        psi = self.psi.Bs[i]
        psi_shape = psi.shape
        # compute effective one-site Hamiltonian
        Heff = GTVP1_Engine.compute_Heff_Onesite(self.LPs[i], self.RPs[i], self.model.H_mpo[i])
        # evolve 1-site wave function forwards in time
        psi = GTVP1_Engine.evolve(psi, Heff, self.dt/2)
        psi = np.reshape(psi, psi_shape)
        # put back into MPS
        self.psi.Bs[i] = psi
            
        
        
    def update_bond(self,i,sweep_right):
        C = None
        self.psi.Bs[i] = gtn.dense(data=self.psi.Bs[i], statistics=(-1,1,1))
        B= gtn.einsum('ijk->ikj',self.psi.Bs[i]) 
        if sweep_right:
            B = B.join_legs('(ij)k',intermediate_stat=(1,1))
            U,S,V,norm,_ = gmps.split_and_truncate(B,self.chi_max,self.eps)
            self.psi.norm += norm
            B= U.split_legs('ijk')    
            self.psi.Bs[i] = gtn.einsum('ijk->ikj',B)
            C = gtn.einsum('ij,jk->ik',np.diag(S),V)
            self.update_LP(i)
            Heff = self.compute_Heff_zero_site(self.LPs[i+1], self.RPs[i])
        
        else:
            B = B.join_legs('i(jk)')
            U,S,V,norm,_ = gmps.split_and_truncate(B,self.chi_max,self.eps)
            self.psi.norm += norm
            B = U.split_legs('ijk')
            self.psi.Bs[i] = gtn.einsum('ijk->ikj',B)
            C = gtn.einsum('ij,jk->ik',V,np.diag(S))
            self.update_RP(i)
            Heff = self.compute_Heff_zero_site(self.LPs[i], self.RPs[i]) 
            
        C = C.join_legs('(ij)')
        C = self.evolve(C,Heff,-self.dt/2)
        C = C.split_legs('(i)(j)')
        
        if sweep_right: 
            self.psi.Bs[i] = gtn.einsum('ij,jkm->ikm',C,self.psi.Bs[i+1])
        else: 
            self.psi.Bs[i-1] = gtn.einsum('ijk,jm->ikm')
            self.psi.Bs[i-1]= gtn.einsum('ijk->ikj',self.psi.Bs[i-1])
                    
    def evolve(psi,H,dt):
        exp = []
        for i in range(H.shape[0]):
            exp.append(gtn.arith.exp(-1.j * H[0][i]*dt)*psi[0][0][0])
        print(exp)
        return expm_multiply(-1.j * H * dt, psi[0][0])
    