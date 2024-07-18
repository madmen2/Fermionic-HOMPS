import grassmanntn as gtn
import numpy as np

class GrassmanGaussianNoise:
    

    def __init__(self,g,w,duration): 
        
        self.g=g
        self.w=w
        self.duration=duration
        
    def grassman_number_generator(self,N_steps):
        self.N_steps=N_steps
        random_int = np.random.random(self.N_steps)
        string=[]
        for i in random_int:
            string.append(str(i))
        gn = gtn.set_ac(string)
        strings=[]
        for i in range(self.N_steps):
            input_str = str(gn[i])   
            asterisk_position = input_str.find('*')
            result = input_str[asterisk_position + 1:]
            strings.append(result)
                    
        final=[]
        for i in range(self.N_steps):
            final.append(float(strings[i]))
        return final
    
    """    def alpha(self,tau):
            self.tau=tau
            arg1 = np.multiply.outer(np.real(self.w), np.abs(self.tau))
            arg2 = np.multiply.outer(np.imag(self.w), self.tau)
            return np.sum(self.g[:, np.newaxis]*np.exp(-arg1 - 1j*arg2), axis=0)"""
    
    def initialize(self, N_steps):
        self.N_steps = N_steps
        self.ts = np.linspace(0, self.duration, N_steps)
        self.z =self.grassman_number_generator(N_steps)
        
    def sample_process(self):
        self.tau = np.random.random(self.N_steps)
        arg1 = np.multiply.outer(np.real(self.w), np.abs(self.tau))
        arg2 = np.multiply.outer(np.imag(self.w), self.tau)
        Z=np.array(1j*np.sum(self.g[:, np.newaxis]*np.exp( -arg1-1j*arg2), axis=0))
        
        generator= 'phi_'
        generators=[]
        for i in range(Z.shape[0]):
            generators.append(generator+str(i))
        
        gen=gtn.set_ac(generators)
        prod=gen*Z
        return prod
    
    def sample_process2(self): 
        list = []
        for i in np.random.random(len(self.z)):
            for j in np.random.random(len(self.z)):
                if i != j:
                    if np.mean(self.sample_process2(i)*self.sample_process2(j),axis=0)== 0 and np.mean(self.sample_process2(i)*np.conj(self.sample_process2(j)),axis=0)==self.alpha(i-j):
                        list.append(self.sample_process2(i)),list.append(self.sample_process2(j))
                        
        grassman_noise = np.array(list)
        return grassman_noise
    
  