import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import random
import json
from IPython.display import clear_output
import matplotlib.pyplot as plt
from copy import deepcopy

# Data processing, modeling and model evaluation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.metrics import plot_confusion_matrix

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap,ZFeatureMap,RealAmplitudes
from qiskit.quantum_info import SparsePauliOp, Kraus, SuperOp
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, ADAM
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import NeuralNetwork, EstimatorQNN

# from qiskit.tools.visualization import plot_histogram
# from qiskit_aer import AerSimulator
# from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError, pauli_error

algorithm_globals.random_seed = 12345


# We now define an pool curcuit
def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

# We now define an Pooling Layer
def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def predict(model, x, y):
    y_pre = model.predict(x)
    mae = mean_absolute_error(y, y_pre)
    rmse = np.sqrt(mean_squared_error(y, y_pre))
    r2 = r2_score(y, y_pre)
    return mae, rmse, r2, y_pre
