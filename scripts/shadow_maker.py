import numpy as np
import matplotlib.pyplot as plt
import json

from exactsimulation import *
from noise_model import *

from qiskit import QuantumCircuit, execute, Aer, ClassicalRegister, assemble
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector, Operator, DensityMatrix
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.providers.aer.noise import NoiseModel
from qiskit_ibm_provider import IBMProvider
from qiskit.providers.fake_provider import FakeManilaV2, FakeVigo


def make_snapshots(circuit, parameters, n_spins, shadow_size, backend):
    X = Pauli("X")
    Y = Pauli("Y")
    Z = Pauli("Z")
    H = Operator(1/np.sqrt(2)*np.array([[1,1],[1,-1]]))
    S = Operator(np.array([[1,0],[0,-1j]], dtype=complex))
    I = Operator(Pauli("I"))
    #S daga
    pauli_ens = [H, H @ S, I]
    #pauli_ens = [X, Y, Z]
    rand_rotations = np.random.randint(3, size=(shadow_size,n_spins))
    meas_result = np.zeros((shadow_size, n_spins), dtype=int)
    circuit = circuit.bind_parameters(parameters)
    for k in range(shadow_size):
        temp_circuit = circuit.copy()
        for i in range(n_spins):
            temp_circuit.append(pauli_ens[rand_rotations[k, i]], [i])
        temp_circuit.measure_all()
        job = backend.run(temp_circuit, shots = 1).result().get_counts()
        meas_result[k, :] = [int(x) for x in [*list(job.keys())[0]]]

    return (meas_result, rand_rotations)

def snap_to_shadow(base, rotation):
    n_spins = len(base)

    state_0 = np.array([[1, 0],[0,0]])
    state_1 = np.array([[0, 0],[0,1]])
    states = [state_0, state_1]
    hadamard = 1/np.sqrt(2)*np.array([[1,1],[1,-1]])
    s_gate = np.array([[1,0],[0,-1j]], dtype=complex)
    id = np.identity(2)
    gate_ens = [hadamard, hadamard @ s_gate, id]
    shadow = [1]
    for i in range(n_spins):
        temp_shad = 3*gate_ens[rotation[i]].conj().T @ states[base[i]] @ gate_ens[rotation[i]] - id
        shadow = np.kron(shadow, temp_shad)
    return shadow

def make_shadow(circuit, parameters, n_spins, shadow_size, backend):
    base, rotations = make_snapshots(circuit, parameters, n_spins, shadow_size, backend)
    shadow_size, n_spins = base.shape
    rho = np.zeros((2**n_spins, 2**n_spins))
    for i in range(shadow_size):
        rho = rho + snap_to_shadow(base[i], rotations[i])
    return rho/shadow_size

def shadow_expectation(circuit, observable, parameters, n_spins, number_of_experiments, shadow_size, backend):
    mean = []
    for k in range(number_of_experiments):
        base, rotations = make_snapshots(circuit, parameters, n_spins, shadow_size, backend)

        shadow_size, n_spins = base.shape

        #Decompose observable in Pauli rotations: 0->X, 1->Y, 2->Z
        obs =  observable.to_list()
        obs_to_indices = np.zeros(n_spins)
        for i in obs:
            i = [*i[0]]
            for j in range(n_spins):
                if i[j] == "X":
                    obs_to_indices[j] = 0
                if i[j] == "Y":
                    obs_to_indices[j] = 1
                if i[j] == "Z":
                    obs_to_indices[j] = 2
        count = 0
        exp = 0
        for k in range(shadow_size):
            eigenvalue = 0
            if np.array_equal(rotations[k], obs_to_indices):
                for i in base[k]:
                    if i == 0:
                        eigenvalue += (1)
                    if i == 1:
                        eigenvalue += (-1)

                exp += eigenvalue
                count+=1
                mean.append(exp/count)

    return np.mean(mean)


def find_shadow_size(observable, error, sampling_error):
    obs = observable.to_list()
    pauli_obs = [Pauli(o[0]) for o in obs]

    n_spins = int(np.log2((observable.to_matrix()).shape[0]))
    M = len(obs)

    number_of_experiments = np.ceil(2*np.log(2*M/sampling_error))

    norm = lambda op: np.linalg.norm(op - np.trace(op) / 2 ** n_spins, ord=np.inf)** 2
    shadow_size = np.ceil(34/error**2*max([norm(o.to_matrix()) for o in pauli_obs]))

    return int(shadow_size), int(number_of_experiments)
