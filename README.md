# Materials property prediction

The quantum residual attention neural network (QRANN),run by executing the qrann.py file

# Load the training and testing datasets
LOAD Elem_train, Elem_test, Prop_train, Prop_test

# Set the random seed for reproducibility
SET random_seed = 12345

# Define the Pool Circuit for pooling layer
FUNCTION pool_circuit(params):
CREATE quantum circuit with 2 qubits
APPLY Rz, Rx, Ry gates as per parameters
RETURN quantum circuit

# Define the Pooling Layer for quantum circuit
FUNCTION pool_layer(sources, sinks, param_prefix):
CREATE quantum circuit with num_qubits
PARAMS = generate parameters based on sources and sinks
FOR each pair of source, sink:
APPLY pool_circuit to each pair with relevant parameters
RETURN quantum circuit

# Define the Attention Layer for quantum circuit
FUNCTION attention_layer(num_qubits, param_prefix):
CREATE quantum circuit with num_qubits
APPLY parameterized gates (RX, CRY, RY, CNOT, and Hadamard)
RETURN quantum circuit

# Define the prediction function (evaluate model performance)
FUNCTION predict(model, inputs, true_values):
PREDICT the output from the model using inputs
CALCULATE MAE, RMSE, and R^2 as performance metrics
RETURN performance metrics and predictions

# Define the callback function to track objective function during optimization
FUNCTION callback_graph(weights, obj_func_eval):
CLEAR the output screen
STORE objective function values
PRINT current iteration and objective function evaluation

# Create the feature map (quantum circuit for encoding data)
CREATE feature_map (e.g., ZFeatureMap for 4 qubits)

# Create the quantum neural network (QNN) ansatz (ansatz circuit)
CREATE ansatz circuit with multiple layers (attention + pooling layers)
COMPOSE attention_layer(5, "a1") for first part
COMPOSE pool_layer for pooling part
COMPOSE attention_layer(3, "a2") for second part

# Combine the feature map and ansatz into a full quantum circuit
COMPOSE full quantum circuit from feature_map and ansatz

# Define the observable for QNN
CREATE observable (e.g., SparsePauliOp)

# Instantiate the Estimator QNN
CREATE EstimatorQNN using full quantum circuit and observables

# Start training on small data samples (mini-batch training)
SET batch_size = 20
SET epochs = number of training batches
FOR each epoch:
# Initialize parameters and select a mini-batch of data
INIT random parameters for the model
SELECT training data batch (Elem_train_part, Prop_train_part)

# Train the model using L-BFGS-B optimizer
FIT the model on current batch data
EVALUATE the model on both training and test data
PRINT performance metrics (MAE, RMSE, R^2)

# Store performance metrics for each iteration
RECORD performance metrics and predictions

# Save the results of mini-batch training
SAVE performance metrics, predictions, losses, and iteration counts to disk

# Print final performance statistics for training and testing datasets
PRINT statistics (mean, std, max, min) for training and testing results

# Start full training on the entire dataset
SET batch_size = total training size
SET epochs = 5
FOR each epoch:
# Initialize parameters and use full dataset
INIT random parameters for the model
SELECT full training dataset (Elem_train, Prop_train)

# Train the model on the full dataset
FIT the model on the entire dataset
EVALUATE the model on both training and test data
PRINT performance metrics (MAE, RMSE, R^2)

# Store performance metrics for each epoch
RECORD performance metrics and predictions

# Save the results of full sample training
SAVE performance metrics, predictions, losses, and iteration counts to disk

# Print final performance statistics for training and testing datasets
PRINT statistics (mean, std, max, min) for training and testing results