import numpy as np
import scipy.sparse as sparse
import grassmanntn as gtn

def generate_physical_operators():
    """
    Computes some basic operators in the physical Hilbert space as np.ndarray
        
    Returns
    -------
    np.ndarray
        sigma_x pauli matrix, shape (2, 2): 
    np.ndarray
        sigma_z pauli matrix, shape (2, 2)
    np.ndarray
        identity matrix, shape (2, 2)
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_y = np.array([[0, -1.j], [1.j, 0]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    return sigma_x, sigma_z, eye, sigma_y



def generate_dissapative_qubit(omega):

    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = 0.5*omega*sigma_z
    return H

def generate_spin_boson_hamiltonian(delta=1, epsilon=0):
    """
    Computes the system Hamiltonian of the spin-boson model as np.ndarray
    
    Parameters
    ----------
    delta : float 
        parameter for the spin-boson model
    epsilon : float
        parameter for the spin-boson model
        
    Returns
    -------
    np.ndarray
        Hamiltonian of the spin-boson model, shape (2, 2) 
        $H = -1/2\Delta\sigma_x+1/2\epsilon\sigma_z$
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = -0.5*delta*sigma_x + 0.5*epsilon*sigma_z
    return H

def generate_grassmann_indices(N_trunc):
    generators_epsilon = []
    for i in range(N_trunc):
        generators_epsilon.append('phi_'+str(i))
    c1 = gtn.set_ac(generators_epsilon) 
    generators_epsilon_conj = []
    for i in range(N_trunc):
        generators_epsilon_conj.append('phi_'+str(i)+'_conj')       
    c2 = gtn.set_ac(generators_epsilon_conj)
    eye_aux=np.eye(N_trunc,dtype=complex )
    return c1, c2, c1*c2,eye_aux

def generate_physical_operators_sparse():
    """
    Computes some basic operators in the physical Hilbert space as scipy.sparse matrices
        
    Returns
    -------
    scipy.sparse.csr_matrix
        sigma_x pauli matrix, shape (2, 2): 
    scipy.sparse.csr_matrix
        sigma_z pauli matrix, shape (2, 2)
    scipy.sparse.csr_matrix
        identity matrix, shape (2, 2)
    """
    sigma_x = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    sigma_z = sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
    eye = sparse.csr_matrix(np.eye(2, dtype=complex))
    return sigma_x, sigma_z, eye

def generate_spin_boson_hamiltonian_sparse(delta=1, epsilon=0):
    """
    Computes the system Hamiltonian of the spin-boson model as scipy.sparse matrices
    
    Parameters
    ----------
    delta : float 
        parameter for the spin-boson model
    epsilon : float
        parameter for the spin-boson model
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Hamiltonian of the spin-boson model, shape (2, 2) 
        $H = -1/2\Delta\sigma_x+1/2\epsilon\sigma_z$
    """
    sigma_x = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    sigma_z = sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
    H = -0.5*delta*sigma_x + 0.5*epsilon*sigma_z
    return H

def generate_auxiallary_operators_sparse(N_trunc, rescale_aux=True):
    """
    Computes the auxillary operators that are used in some of the
    HOPS/HOMPS implementations as scipy sparse matrices
    
    Parameters
    ----------
    N_trunc : int
        dimension of the auxillary Hilbert space
    rescale_aux : int
        If this is set to true (default), the raising and lowering
        operators will include the factors sqrt(n + 1) and sqrt(n)
        respectively (this is necessary for HOMPS with rescaled 
        auxillary vectors)

    Returns
    -------
    scipy.sparse.csr_matrix
        number operator N. Is of shape (N_trunc, N_trunc)
    scipy.sparse.csr_matrix
        raising operator b_dagger. is of shape (N_trunc, N_trunc)
    scipy.sparse.csr_matrix
        lowering operator b. is shape (N_trunc, N_trunc)
    scipy.sparse.csr_matrix
        identity operator. is of shape (N_trunc, N_trunc)
    """
    # number operator
    N = sparse.csr_matrix(np.diag(np.arange(0, N_trunc, dtype=complex)))
    # lowering operator
    b = sparse.csr_matrix(np.diag(np.sqrt(np.arange(1, N_trunc, dtype=complex)), 1))
    # raising operator
    b_dagger = sparse.csr_matrix(np.diag(np.sqrt(np.arange(1, N_trunc, dtype=complex)), -1))
    
    # lowering operator
    if rescale_aux:
        b = sparse.csr_matrix(np.diag(np.sqrt(np.arange(1, N_trunc, dtype=complex)), 1))
    else:
        b = sparse.csr_matrix(np.diag(np.sqrt(np.ones(N_trunc-1, dtype=complex)), 1))
    # raising operator
    if rescale_aux:
        b_dagger = sparse.csr_matrix(np.diag(np.sqrt(np.arange(1, N_trunc, dtype=complex)), -1))
    else:
        b_dagger = sparse.csr_matrix(np.diag(np.sqrt(np.ones(N_trunc-1, dtype=complex)), -1))
    # identity operator
    eye = sparse.csr_matrix(np.eye(N_trunc, dtype=complex))
    # return the three operators
    return N, b_dagger, b, eye
