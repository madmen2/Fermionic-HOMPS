from ..gutil import operators
import grassmanntn as gtn
import numpy as np 

class GHOMPSModel:
    def __init__(self, g, w, h, L, N_trunc,mode, rescale_aux=True, 
                 alternative_realization=False, gamma_terminator=0):

        self.N_bath = g.shape[0]
        assert(self.N_bath == w.shape[0])
        self.L = L
        self.mode = mode
        self.N_trunc = N_trunc
        self.N = self.N_bath + 1
        self.alternative_realization = alternative_realization
        # get some useful operators
        sigma_x, sigma_z, eye, sigma_y = operators.generate_physical_operators()
        self.eye = eye
        
        epsilon, episilon_conj, N,eye_aux ,berezin= operators.generate_grassmann_indices(self.N_trunc)
        self._H_mpo_template = np.array([None]*(self.N_bath + 1))
        self.H_mpo = np.array([None]*(self.N_bath + 1))
        self._H_mpo_template[0] = np.zeros((4, 4, 2, 2), dtype=object)
        self._H_mpo_template[0][0, 0, :, :] = -1.j * eye
        if self.alternative_realization:
            self._H_mpo_template[0][0, 1, :, :] = -1.j * L
            self._H_mpo_template[0][0, 2, :, :] = 1.j * np.conj(L).T
            self._H_mpo_template[0][0, 3, :, :] = h - 1.j * gamma_terminator * np.conj(L).T@L
        else:
            self._H_mpo_template[0][0, 1, :, :] = 1.j * L
            self._H_mpo_template[0][0, 2, :, :] = -1.j * np.conj(L).T
            self._H_mpo_template[0][0, 3, :, :] = h
        self.H_mpo[0] = self._H_mpo_template[0].copy()
        for i in range(self.N_bath):
            self._H_mpo_template[i+1] = np.zeros((4, 4, N_trunc, N_trunc), dtype=object)

            self._H_mpo_template[i+1][0, 0, :, :] = eye_aux
            self._H_mpo_template[i+1][1, 1, :, :] = eye_aux
            self._H_mpo_template[i+1][2, 2, :, :] = eye_aux
            self._H_mpo_template[i+1][3, 3, :, :] = eye_aux
            if rescale_aux:
                self._H_mpo_template[i+1][0, 3, :, :] = w[i]*episilon_conj@epsilon
                self._H_mpo_template[i+1][1, 3, :, :] = g[i]/np.sqrt(np.abs(g[i]))*episilon_conj
                self._H_mpo_template[i+1][2, 3, :, :] = np.sqrt(np.abs(g[i]))*epsilon
            else:
        
                self._H_mpo_template[i+1][0, 3, :, :] = w[i]*epsilon
                self._H_mpo_template[i+1][1, 3, :, :] = g[i]*episilon_conj@N
                self._H_mpo_template[i+1][2, 3, :, :] = N
            self.H_mpo[i+1] = self._H_mpo_template[i+1].copy()
            
    def update_mpo_linear(self,zt):
        self.H_mpo = [W.copy() for W in self._H_mpo_template]
        if self.alternative_realization:
            self.H_mpo[0][0, 3, :, :] += zt * self.L
        else:
            self.H_mpo[0][0, 3, :, :] += 1.j * np.conj(zt) * self.L
                        
    def update_mpo_nonlinear(self, zt, memory, expL):
        self.H_mpo = [W.copy() for W in self._H_mpo_template]
        if self.alternative_realization:
            self.H_mpo[0][0, 3, :, :] += (zt + memory) * self.L
            self.H_mpo[0][0, 2, :, :] -= 1.j * expL * self.eye


    def compute_update_mpo(self):
            N = len(self.H_mpo)
            self.update_mpo = [None] * N
            factor = np.power(-1.j, 1/N)
            for i in range(N):
                self.update_mpo[i] = factor * gtn.einsum('ijkl->ijlk',gtn.dense(data=self.H_mpo[i].copy(),
                                                                                statistics=(-1,-1,-1,-1))) 
