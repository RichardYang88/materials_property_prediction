#!/usr/bin/env python
# coding: utf-8

from common import *
# load dataset
Elem_train = np.load("./data/pca/Elem_train.npy")
Elem_test = np.load("./data/pca/Elem_test.npy")
Prop_train = np.load("./data/pca/Prop_train.npy")
Prop_test = np.load("./data/pca/Prop_test.npy")
mod_nm = "QRANN"

def attention_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="attention Layer")
    params = ParameterVector(param_prefix, length=(num_qubits-1)*2+1)
    param_idx=0
    for i in range(num_qubits-1):
        qc.rx(params[param_idx], i)
        param_idx+=1
    qc.barrier()

    qc.h(num_qubits-1)
    
    for i in range(num_qubits-2):
        qc.cx(i, i+1)
    qc.cry(params[param_idx], num_qubits-2, num_qubits-1)
    param_idx+=1
    qc.h(num_qubits-1)
    qc.barrier()

    for i in range(num_qubits-1):
        qc.ry(params[param_idx], i)
        param_idx+=1
    qc.barrier()

    for i in range(num_qubits-1):
        qc.cx(num_qubits-1, i)
    qc.barrier()

    qc.h(num_qubits-3)
    qc.h(num_qubits-2)
    qc.barrier()

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

# print(attention_layer(5, "a1").decompose())

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    print(f"iter: {len(objective_func_vals)}   obj_func_eval: {obj_func_eval}")

feature_map = ZFeatureMap(4)
ansatz = QuantumCircuit(5, name="Ansatz")
ansatz.compose(attention_layer(5, "a1"), list(range(5)), inplace=True)
ansatz.compose(pool_layer([0, 1], [2, 3], "p1"), list(range(4)), inplace=True)
ansatz.compose(attention_layer(3, "a2"), list(range(2, 5)), inplace=True)

# Combining the feature map and ansatz
circuit = QuantumCircuit(5)
circuit.compose(feature_map, range(4), inplace=True)
circuit.compose(ansatz, range(5), inplace=True)

observable = SparsePauliOp.from_list([("IZZII", 2)])

qann = EstimatorQNN(
    circuit=circuit.decompose(),
    observables=observable,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
)
print("weight_params len:",len(qann.weight_params))

print("\n\n============= small sample training ===============")
part = 20
epoch = Elem_train.shape[0]//part
perf = []
perf_t = []
iter_list = []
Prop_list = []
loss = []
for i in range(epoch):
    init_pt = np.random.rand(len(qann.weight_params))
    Elem_train_part = Elem_train[i*part:(i+1)*part]
    Prop_train_part = Prop_train[i*part:(i+1)*part]

    objective_func_vals = []
    model = NeuralNetworkRegressor(
        neural_network=qann,
        optimizer=L_BFGS_B(maxiter=200, ftol=1e-7),
        callback=callback_graph,
        initial_point=init_pt,
    )

    model.fit(Elem_train_part, Prop_train_part)
    mae, rmse, r2, Prop_pred = predict(model, Elem_test, Prop_test)
    mae_t, rmse_t, r2_t, _ = predict(model, Elem_train_part, Prop_train_part)
    print(f'testset perform: R^2={r2:.4f}, rmse={rmse:.4f}, mae={mae:.4f}')
    perf.append([mae, rmse, r2])
    perf_t.append([mae_t, rmse_t, r2_t])
    iter_list.append(len(objective_func_vals))
    Prop_list.append(Prop_pred.squeeze())
    loss.append(objective_func_vals)

perf = np.array(perf)
perf_t = np.array(perf_t)
iter_list = np.array(iter_list)
Prop_list = np.array(Prop_list)

iter_min = iter_list.min()
for i in range(epoch):
    loss[i] = loss[i][:iter_min]
loss = np.array(loss)

np.save("./dump_small_sample/perf/{}.npy".format(mod_nm), perf)
np.save("./dump_small_sample/perf_t/{}.npy".format(mod_nm), perf_t)
np.save("./dump_small_sample/iter/{}.npy".format(mod_nm), iter_list)
np.save("./dump_small_sample/Prop/{}.npy".format(mod_nm), Prop_list)
np.save("./dump_small_sample/loss/{}.npy".format(mod_nm), loss)

print("predict trainset")
print(perf_t.shape)
print(perf_t)
print(perf_t.mean(axis=0))
print(perf_t.std(axis=0))
print(perf_t.max(axis=0))
print(perf_t.min(axis=0))

print("predict testset")
print(perf.shape)
print(perf)
print(perf.mean(axis=0))
print(perf.std(axis=0))
print(perf.max(axis=0))
print(perf.min(axis=0))

print("iteration list")
print(iter_list.shape)
print(iter_list)
print(iter_list.mean(axis=0))
print(iter_list.std(axis=0))
print(iter_list.max(axis=0))
print(iter_list.min(axis=0))


print("\n\n============= full sample training ===============")
part = Elem_train.shape[0]
epoch = 5
perf = []
perf_t = []
iter_list = []
Prop_list = []
loss = []
for i in range(epoch):
    init_pt = np.random.rand(len(qann.weight_params))
    Elem_train_part = deepcopy(Elem_train)
    Prop_train_part = deepcopy(Prop_train)

    objective_func_vals = []
    model = NeuralNetworkRegressor(
        neural_network=qann,
        optimizer=L_BFGS_B(maxiter=200, ftol=1e-7),
        callback=callback_graph,
        initial_point=init_pt,
    )

    model.fit(Elem_train_part, Prop_train_part)
    mae, rmse, r2, Prop_pred = predict(model, Elem_test, Prop_test)
    mae_t, rmse_t, r2_t, _ = predict(model, Elem_train_part, Prop_train_part)
    print(f'testset perform: R^2={r2:.4f}, rmse={rmse:.4f}, mae={mae:.4f}')
    perf.append([mae, rmse, r2])
    perf_t.append([mae_t, rmse_t, r2_t])
    iter_list.append(len(objective_func_vals))
    Prop_list.append(Prop_pred.squeeze())
    loss.append(objective_func_vals)

perf = np.array(perf)
perf_t = np.array(perf_t)
iter_list = np.array(iter_list)
Prop_list = np.array(Prop_list)

iter_min = iter_list.min()
for i in range(epoch):
    loss[i] = loss[i][:iter_min]
loss = np.array(loss)

np.save("./dump_full_sample/perf/{}.npy".format(mod_nm), perf)
np.save("./dump_full_sample/perf_t/{}.npy".format(mod_nm), perf_t)
np.save("./dump_full_sample/iter/{}.npy".format(mod_nm), iter_list)
np.save("./dump_full_sample/Prop/{}.npy".format(mod_nm), Prop_list)
np.save("./dump_full_sample/loss/{}.npy".format(mod_nm), loss)

print("predict trainset")
print(perf_t.shape)
print(perf_t)
print(perf_t.mean(axis=0))
print(perf_t.std(axis=0))
print(perf_t.max(axis=0))
print(perf_t.min(axis=0))

print("predict testset")
print(perf.shape)
print(perf)
print(perf.mean(axis=0))
print(perf.std(axis=0))
print(perf.max(axis=0))
print(perf.min(axis=0))

print("iteration list")
print(iter_list.shape)
print(iter_list)
print(iter_list.mean(axis=0))
print(iter_list.std(axis=0))
print(iter_list.max(axis=0))
print(iter_list.min(axis=0))
