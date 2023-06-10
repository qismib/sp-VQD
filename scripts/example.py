import numpy as np
import matplotlib.pyplot as plt

from exactsimulation import *
from pVQD import *

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector, Pauli
from qiskit.primitives import Estimator
from qiskit.visualization import circuit_drawer


N = 3 #Spins
h = 1
J = 1/4
depth = 3 #ansatz depth
l_rate = 0.05
s = np.pi / 2
dt = 0.05 #Time step
shots = [800, 8000]

#Setting initial parameters to '0'
par  = [0]*(depth*N+depth*(N-1))
dparev  = [0.01]*len(par)
inter = np.linspace(0, 0.5, 10)

expectation_values = {'X 800': [],'X 8000': [],'Z 800': [],'Z 8000': []}

expectation_values['X 800'].append(expectation_X(depth , N, par, shots[0]))
expectation_values['X 8000'].append(expectation_X(depth , N, par, shots[1]))
expectation_values['Z 800'].append(expectation_Z(depth , N, par, shots[0]))
expectation_values['Z 8000'].append(expectation_Z(depth , N, par, shots[1]))

for t in inter:
    L = 1
    for i in range(100):
        L = GlobalCostFunction(N, depth, dparev, par, J, h, dt)
        dparev = Global_parameter_evolution(s, l_rate, N, depth, dparev, par, J, h, dt)

        print("L: ",L)

    for k in range(len(par)):
        par[k] = par[k] + dparev[k]
    print("Parameters: ",par)
    print("------------------------")
    expectation_values['X 800'].append(expectation_X(depth , N, par, shots[0]))
    expectation_values['X 8000'].append(expectation_X(depth , N, par, shots[1]))
    expectation_values['Z 800'].append(expectation_Z(depth , N, par, shots[0]))
    expectation_values['Z 8000'].append(expectation_Z(depth , N, par, shots[1]))


exact_x = exact_meas_of_X (N, J, h, inter)
exact_z = exact_meas_of_Z (N, J, h, inter)
















#---------------------------------------------------
#plotting result
plt.figure(1)
plt.subplot(211)
plt.plot(inter, exact_x,linestyle="dashed",color="black",label="Exact")
plt.plot(inter, expectation_values['X 8000'],color="C1",label="pVQD 8000 steps",linestyle="",marker=".")
plt.plot(inter, expectation_values['X 800'],color="C0",label="pVQD 800 steps",linestyle="",marker=".")

plt.ylabel(r'$\langle\sigma_x\rangle$')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(inter, exact_z,linestyle="dashed",color="black",label="Exact")
plt.plot(inter, expectation_values['Z 800'],color="C0",label="pVQD 800 steps",linestyle="",marker=".")
plt.plot(inter, expectation_values['Z 8000'],color="C1",label="pVQD 8000 steps",linestyle="",marker=".")

plt.xlabel('time')
plt.ylabel(r'$\langle\sigma_z\rangle$')

plt.legend()
plt.grid()
plt.show()
