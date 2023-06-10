import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector, Pauli
from qiskit.primitives import Estimator
from qiskit.visualization import circuit_drawer

#-------------------------------------
#Variational ansatz costruction
def ansatz(depth, n_spins,par):
    qc_ans = QuantumCircuit(n_spins)
    for l in range(depth):
        for i in range(n_spins):
            if l%2 == 0:
                qc_ans.rx(par[i+l*n_spins],i)
            if l%2 == 1:
                qc_ans.ry(par[i+l*n_spins],i)
        for j in range(n_spins-1):
            qc_ans.rzz(par[depth*n_spins+j+l*(n_spins-1)],j,j+1)
    return qc_ans

#------------------------------------
#Trotter step
def trotter_evol(n_spins, dt, J, h):

    qc_trotter = QuantumCircuit(n_spins)

    for i in range(n_spins):
        qc_trotter.rx(2*J*dt, i)
    for  i in range(n_spins-1):
        qc_trotter.rzz(2*dt*h, i,i+1)

    return qc_trotter

#--------------------------------------------------
#|0...0><0....0| projection
def projector (n_spins):
    proj = SparsePauliOp(["I", "Z"], coeffs=[1/2,1/2])
    proj0 = proj
    for i in range(n_spins-1):
        proj0 = proj0 ^ proj
    return proj0
#---------------------------------------------------
#Overlap costruction
def GlobalCostFunction(n_spins, depth, dparev, par, J, h, dt):
    proj = projector(n_spins)

    parev = [0]*len(par)
    for k in range(len(par)):
        parev[k] = par[k] + dparev[k]

    estimator = Estimator()

    qc_over = QuantumCircuit(n_spins)
    qc_over = qc_over.compose(ansatz(depth, n_spins, parev))
    qc_over.barrier()
    qc_over = qc_over.compose(trotter_evol(n_spins, dt, J, h).inverse())
    qc_over.barrier()
    qc_over = qc_over.compose(ansatz(depth, n_spins, par).inverse())
    qc_over.barrier()
    overlap = estimator.run(qc_over, proj).result().values

    L = (1 - overlap)
    return L

#-----------------------------------------------------
#Gradient evaluetion and parameter evolution
def Global_L_gradient(s, n_spins, depth, dparev, par, J, h, i, dt):
    dpar_sup = [0]*len(dparev)
    dpar_inf = [0]*len(dparev)
    for k in range(len(dparev)):
        dpar_inf[k] = dparev[k]
        dpar_sup[k] = dparev[k]
    dpar_inf[i] = dpar_inf[i] - s
    dpar_sup[i] = dpar_sup[i] + s
    cost_plus = GlobalCostFunction(n_spins, depth, dpar_sup, par, J, h, dt)
    cost_minus = GlobalCostFunction(n_spins, depth, dpar_inf, par, J, h, dt)
    variation = (cost_plus-cost_minus)/(2*np.sin(s))

    return variation[0]


def Global_parameter_evolution (s, eta, n_spins, depth, dparev, par, J, h, dt):
    for i in range (len(par)):
        dparev[i] = dparev[i] - eta * Global_L_gradient(s, n_spins, depth, dparev, par, J, h, i, dt)

    return dparev

#-----------------------------------------------
#Total magnetizzation expectation value i.e. <sigma>/n_spins
def expectation_X(depth , n_spins, par, shots):
    qc_x = QuantumCircuit(n_spins)
    qc_x = qc_x.compose(ansatz(depth, n_spins, par))
    #X = SparsePauliOp.from_sparse_list(["X", [0], 1] , n_spins)
    X = SparsePauliOp.from_sparse_list([["X", [i], 1] for i in range(n_spins)], n_spins)
    estimator = Estimator()
    exp_value  = estimator.run(qc_x, X).result().values
    return exp_value/n_spins

def expectation_Z(depth , n_spins, par, shots):
    qc_z = QuantumCircuit(n_spins)
    qc_z = qc_z.compose(ansatz(depth, n_spins, par))
    #Z = SparsePauliOp.from_sparse_list("Z", [0], 1 , n_spins)
    Z = SparsePauliOp.from_sparse_list([["Z", [i], 1] for i in range(n_spins)], n_spins)
    estimator = Estimator()
    exp_value  = estimator.run(qc_z, Z).result().values

    return exp_value/n_spins
