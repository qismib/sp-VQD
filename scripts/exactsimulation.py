#Exact simulation
import numpy as np
import scipy.linalg as la

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


#definining Ising hamiltonian
def H_Ising (n_spins, J, h):
    ZZ = SparsePauliOp.from_sparse_list([["ZZ", [i,i+1], J] for i in range(n_spins-1)], n_spins)
    X = SparsePauliOp.from_sparse_list([["X", [i], h] for i in range(n_spins)], n_spins)
    H = ZZ + X
    return H

#Creating exact evolution operator
def U_Ising(n_spins, J, h, t):
    H = H_Ising(n_spins, J, h).to_matrix()
    return la.expm(-1j*H*t)

#measuring observables
def exact_meas_of_X (n_spins, j, h, inter):
    exp_value =  []

    #Initial state array
    qc = QuantumCircuit(n_spins)
    init_state =np.array(Statevector(qc))

    #Observale definition as matrix
    obs = SparsePauliOp.from_sparse_list([["X", [i], 1] for i in range(n_spins)], n_spins)
    obs_mat = obs.to_matrix()

    for t in inter:
        evolved_state =  U_Ising(n_spins, j, h, t) @ init_state
        measured_state = obs_mat @ evolved_state
        exp_value.append((evolved_state.transpose().conjugate() @ measured_state).real/n_spins)
    return exp_value

def exact_meas_of_Z (n_spins, j, h, inter):
    exp_value =  []


    qc = QuantumCircuit(n_spins)
    init_state =np.array(Statevector(qc))


    obs = SparsePauliOp.from_sparse_list([["Z", [i], 1] for i in range(n_spins)], n_spins)
    obs_mat = obs.to_matrix()

    for t in inter:
        evolved_state =  U_Ising(n_spins, j, h, t) @ init_state
        measured_state = obs_mat @ evolved_state
        exp_value.append((evolved_state.transpose().conjugate() @ measured_state).real/n_spins)
    return exp_value
