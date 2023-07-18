from mitiq import zne
from qiskit import execute
from qiskit.providers.aer       import QasmSimulator
from noise_model import *
import numpy as np
from qiskit_aer.primitives import Estimator as AerEstimator


def zne_find_probabilities(circuit, parameters, backend, shots=8000, scale_factors = [1,2,3,4,5], zne_order=3):

    circuit = circuit.bind_parameters(parameters)
    circuit.measure_all()
    circuit = circuit.decompose(reps = 2)

    scaled_circuits = [zne.scaling.fold_global(circuit, factor) for factor in scale_factors]
    measurement_counts = execute(scaled_circuits, backend=backend, optimization_level=0, shots=shots).result().get_counts()
    ordered_bitstrings_keys = dict(sorted(measurement_counts[0].items())).keys()
    probs = {key : 0 for key in ordered_bitstrings_keys}

    expectation = 0
    for key in ordered_bitstrings_keys:
        probability = []
        for i in range(len(scale_factors)):
            check1=np.array([key for i in range(len(measurement_counts[i].keys()))])
            check2=np.array([l for l in measurement_counts[i].keys()])
            if np.any(check1 == check2):
                probability.append(measurement_counts[i][key]/shots)
            else:
                probability.append(0)

        zne_probability = zne.PolyFactory.extrapolate(scale_factors, probability, order=zne_order)
        if zne_probability < 0:
            zne_probability = 0
        if zne_probability > 1:
            zne_probability = 1
        probs[key] = zne_probability
    return probs


def zne_expectation(circuit, parameters, backend, shots=8000, scale_factors = [1,2,3,4,5], zne_order=3):
    probabilities = zne_find_probabilities(circuit, parameters, backend, shots, scale_factors, zne_order)
    expectation = 0
    for key in probabilities.keys():
        eigenvalue = 0
        for partial_eigenvalue in [*key]:
            if partial_eigenvalue == '1':
                eigenvalue = eigenvalue - 1
            else:
                eigenvalue = eigenvalue + 1
        expectation = expectation + (probabilities[key] * eigenvalue)
    return expectation


def zne_overlap(circuit, parameters, backend, shots=8000, scale_factors = [1,2,3,4,5], zne_order=3):
    probabilities = zne_find_probabilities(circuit, parameters, backend, shots, scale_factors, zne_order)
    overlap = probabilities['000']
    return overlap
