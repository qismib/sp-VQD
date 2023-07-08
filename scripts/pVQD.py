import numpy as np
import matplotlib.pyplot as plot
import json

from exactsimulation import *
from noise_model import *

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.providers.aer.noise import NoiseModel
from qiskit_ibm_provider import IBMProvider
from qiskit.providers.fake_provider import FakeManilaV2, FakeVigo


def ei(i, n):
    v = np.zeros(n)
    v[i] = 1.0
    return v[:]

#ansatz definition
def ansatz(n_spins, par, depth):
    count = 0
    qc_ans = QuantumCircuit(n_spins)
    for l in range(depth):
        for i in range(n_spins):
            if l%2 == 0:
                qc_ans.rx(par[count],i)
                count += 1
            if l%2 == 1:
                qc_ans.ry(par[count],i)
                count += 1
        for j in range(n_spins-1):
            qc_ans.rzz(par[count],j,j+1)
            count += 1

    if (depth%2 == 1):
        for i in range(n_spins):
            qc_ans.ry(par[count],i)
            count = count +1
    if (depth%2 == 0):
        for i in range(n_spins):
            qc_ans.rx(par[count],i)
            count = count +1

    return qc_ans

def trotter_evolution(n_spins, dt, magnetic_fields):

    qc_trotter = QuantumCircuit(n_spins)

    for i in range(n_spins):
        qc_trotter.rx(2*magnetic_fields[1]*dt, i)
    for  i in range(n_spins-1):
        qc_trotter.rzz(2*dt*magnetic_fields[0], i,i+1)

    return qc_trotter



def global_projector (n_spins):
    proj = SparsePauliOp(["I", "Z"], coeffs=[1/2,1/2])
    proj0 = proj
    for i in range(n_spins-1):
        proj0 = proj0 ^ proj
    return proj0

def local_projector(n_spins):

    proj0 = SparsePauliOp(["I", "Z"], coeffs=[1/2,1/2])
    I = Pauli("I")

    localproj = proj0
    for i in range(1,n_spins):
        localproj = localproj ^ I
    for j in range(1,n_spins):
        partialproj = proj0
        for i in range(1, n_spins):
            if j == i:
                partialproj = partialproj ^ I
            else:
                partialproj = partialproj ^ proj0
        localproj = localproj + partialproj
    return localproj/n_spins



class pVQD:

    def __init__(self,trotter, n_spins, parameters, shift, depth, magnetic_fields):

        self.n_spins = n_spins
        self.magnetic_fields = magnetic_fields
        self.parameters = parameters
        self.shift = shift
        self.num_parameters = len(parameters)
        self.depth = depth
        self.trotter_evolution = trotter

        self.parameters_vec = ParameterVector("pars", self.num_parameters)
        self.ansatz = ansatz(self.n_spins, self.parameters_vec, self.depth)

        self.start = ParameterVector("r", self.num_parameters)
        self.evoluted = ParameterVector("l", self.num_parameters)





    def circuit_construction(self):

        qc = QuantumCircuit(self.n_spins)

        qc = qc.compose((self.ansatz.assign_parameters({self.parameters_vec: self.evoluted})))
        qc = qc.compose((self.trotter_evolution).inverse())
        qc = qc.compose((self.ansatz.assign_parameters({self.parameters_vec: self.start})).inverse())

        return qc


    def overlap_measurement(self, circuit, shifted_par,estimator, projector):
        par = shifted_par+(self.parameters).tolist()
        results = estimator.run(circuit, projector, par).result()
        return results

    def measurements(self, estimator, sim_config):
        result = []
        error_vec = []
        qc_z = QuantumCircuit(self.n_spins)
        qc_z = qc_z.compose(self.ansatz)

        Z = SparsePauliOp.from_sparse_list([["Z", [i], 1] for i in range(self.n_spins)], self.n_spins)
        job_meas = estimator.run(qc_z, Z, self.parameters).result()
        result.append(job_meas.values[0]/self.n_spins)

        if sim_config[1] != 'noshots':
            meta = job_meas.metadata[0]
            err = np.sqrt(meta['variance'].real/meta['shots'])
            error_vec.append(err)
        else:
            error_vec.append(0)

        qc_x = QuantumCircuit(self.n_spins)
        qc_x = qc_x.compose(self.ansatz)
        X = SparsePauliOp.from_sparse_list([["X", [i], 1] for i in range(self.n_spins)], self.n_spins)
        job_meas = estimator.run(qc_x, X, self.parameters).result()
        result.append(job_meas.values[0]/self.n_spins)
        if sim_config[1] != 'noshots':
            meta = job_meas.metadata[0]
            err = np.sqrt(meta['variance'].real/meta['shots'])
            error_vec.append(err)
        else:
            error_vec.append(0)

        return result, error_vec

    def cost_and_gradient(self, circuit, estimator, projector,s, sim_config):

        shifted_par = [(self.parameters+self.shift).tolist()]
        for i in range(self.num_parameters):
            shifted_par.append((self.parameters + self.shift + ei(i, self.num_parameters)*s).tolist())
            shifted_par.append((self.parameters + self.shift - ei(i, self.num_parameters)*s).tolist())

        result = []
        error_vec = []
        for val in shifted_par:
            overlap = self.overlap_measurement(circuit, val, estimator, projector)
            result.append(1 - overlap.values[0].real)
            if sim_config[1] != 'noshots':
                meta = overlap.metadata[0]
                err = np.sqrt(meta['variance'].real/meta['shots'])
                error_vec.append(err)
            else:
                error_vec.append(0)

        g = np.zeros(self.num_parameters)
        E = [result[0], error_vec[0]]
        for i in range(self.num_parameters):
            g[i] = (result[1+2*i]-result[2+2*i])/(2*np.sin(s))

        return E,g

    def cost_and_gradient_spsa(self, circuit, estimator, projector, k, sim_config):
        a = 0.16
        c = 0.1
        A = 1
        alpha = 0.602
        gamma = 0.101

        a_k = a/np.power(A + k, alpha)
        c_k = c/np.power(k, gamma)

        Delta = np.random.binomial(1, 1/2, self.num_parameters)
        Delta = np.where(Delta == 0, -1, Delta)

        shifted_par = [(self.parameters+self.shift).tolist()]

        shifted_par.append((self.parameters + self.shift + c_k*Delta).tolist())
        shifted_par.append((self.parameters + self.shift - c_k*Delta).tolist())

        result = []
        error_vec = []
        for val in shifted_par:
            overlap = self.overlap_measurement(circuit, val, estimator, projector)
            result.append(1 - overlap.values[0].real)
            if sim_config[1] != 'noshots':
                meta = overlap.metadata[0]
                err = np.sqrt(meta['variance']/meta['shots'])
                error_vec.append(err)
            else:
                error_vec.append(0)

        g = np.zeros(self.num_parameters)
        E = [result[0], error_vec[0]]
        for i in range(self.num_parameters):
            g[i] = (result[1]-result[2])/(2*c_k* Delta[i])

        return E,g, a_k




    def adam_gradient(self, m, v, t, g):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        alpha = np.ones(self.num_parameters)*0.001

        if t == 0:
            t = 1

        m = beta1*m + (1 - beta1)*g
        v = beta2*v + (1 - beta2)*np.power(g, 2)
        alpha = alpha * m/(1 - np.power(beta1,t))/(np.sqrt(v/(1 - np.power(beta2,t))) + epsilon)


        return self.shift - alpha, m, v


    def run(self, time_step, n_time_steps, sim_config = ['global', 'noshots'], optimizer = 'gradient_descent', shots = 8000, ln=0.01, s = np.pi/2, max_iter = 100):

        #global or local cost function
        if sim_config[0] == 'global':
            print("Using global cost function.")
            proj = global_projector(self.n_spins)
        if sim_config[0] == 'local':
            print("Using local cost function.")
            proj = local_projector(self.n_spins)

        qc = self.circuit_construction()

        #simulation configuration
        if sim_config[1] == 'noshots':
            print("Simulating without shots.")
            estimator = Estimator()
        if sim_config[1] == 'shots':
            print("Simulating with "+ str(shots) +" shots.")
            estimator = AerEstimator(run_options={"shots": shots}, approximation = True)
        if sim_config[1] == 'noise':
            provider = IBMProvider()
            backend = provider.get_backend('ibm_lagos')
            #backend = FakeVigo()
            noise_model = NoiseModel.from_backend(backend)
            coup = backend.configuration().coupling_map
            #noise_model = noiser()
            print("Simulating with noise and "+ str(shots) + " shots.")
            estimator = AerEstimator(backend_options={"method": "density_matrix","coupling_map": coup,"noise_model": noise_model}, run_options={"shots": shots})
            #estimator = AerEstimator(backend_options={"method": "density_matrix","noise_model": noise_model}, run_options={"shots": shots})




        log = {'S_z': [], 'S_z_err': [], 'S_x': [], 'S_x_err': [], 'Cost_Function': [], 'Cost_Function_err': [], 'Optimization_Steps': [], 'initial_Cost_Functuion': [], 'Parameters': [], 'n_Time_Steps': n_time_steps, 'time_step': time_step}

        opt_curve = {'Cost' : [], 'variance' : []}


        if optimizer == 'gradient_descent':
            print("Using gradient descent optimizer.")
        if optimizer == 'adam':
            print("Using adam optimizer.")
        if optimizer == 'spsa':
            print("Using spsa gradient evaluation.")

        #measure at time 0
        init_shift = self.shift

        meas, err = self.measurements(estimator, sim_config)

        log['S_z'].append(meas[0])
        log['S_z_err'].append(err[0])
        log['S_x'].append(meas[1])
        log['S_x_err'].append(err[1])


        #system evolution with pVQD

        for t in range(n_time_steps):
            curr_L = 1
            #self.shift = init_shift
            print("Time step: ", t)
            print("=========================================")
            count = 0
            first_shifted_par = (self.parameters + self.shift).tolist()
            log['initial_Cost_Functuion'].append(1-self.overlap_measurement(qc, first_shifted_par,estimator, proj).values[0].real)


            if optimizer == 'adam':
                m = np.zeros(self.num_parameters)
                v = np.zeros(self.num_parameters)



            #Optimization for a single time step
            while curr_L > 0.00001 and count<max_iter:

                if optimizer != 'spsa':
                    L, grad = self.cost_and_gradient(qc, estimator, proj, s, sim_config)
                else:
                    L, grad, a_k = self.cost_and_gradient_spsa(qc, estimator, proj, count + 1, sim_config)
                    self.shift = self.shift - a_k * grad

                if optimizer ==  'gradient_descent':
                    self.shift = self.shift - ln*grad

                if optimizer == 'adam':
                    self.shift, m, v = self.adam_gradient(m, v, count, grad)

                print("Gradient: ", grad)
                print("Shifts: ", self.shift)
                curr_L = L[0]
                print("Cost function: ", curr_L)
                opt_curve['Cost'].append(L[0])
                opt_curve['variance'].append(L[1])
                count+=1

            print("Optimization steps required: ", count)
            self.parameters = self.parameters + self.shift
            meas, errors = self.measurements(estimator, sim_config)
            log['S_z'].append(meas[0])
            log['S_z_err'].append(err[0])
            log['S_x'].append(meas[1])
            log['S_x_err'].append(err[1])
            log['Cost_Function'].append(L[0])
            log['Cost_Function_err'].append(L[1])
            log['Optimization_Steps'].append(count)
            log['Parameters'].append(list(self.parameters))

        file_name = "data/"+sim_config[0] + "_" + sim_config[1] + "_" + str(n_time_steps) +"_steps_"+optimizer+".dat"
        with open(file_name, "w") as f:
            json.dump(log, f)

        file_name = "data/optcurve.dat"
        with open(file_name, "w") as f:
            json.dump(opt_curve, f)

        return log
