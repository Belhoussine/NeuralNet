# Machine Learning Library
## Usage Notes:
Python3 is needed to use this library.
```
git clone https://github.com/Belhoussine/NeuralNet
cd NeuralNet
pip3 install requirements.txt
```

## Neural Network Specifications

### 1. Artificial Neural Network:
- [x] Supports multiple layers
- [x] Supports multiple neurons per layer
- [ ] Train:
    - [x] Forward Propagation
    - [ ] Back Propagation  
    - [x] Run in Epochs
    - [x] Supports mini batches
- [x] Predict
- [x] Verbose training phase 

### 2. Activation Functions:

- [x] Sigmoid (Non linear mapping between 0 and 1)
- [x] Softmax (Non Linear Probability Distribution)
- [x] ReLU (Rectified Linear Unit)
- [x] Leaky ReLU (Leaking ReLU on negative values)
- [x] TanH (Hyperbolic Tangent)
- [x] ELU (Exponential Linear Unit)

### 3. Loss Functions:

- [x] RMSE (Root Mean Squared Error)
- [x] MSE (Mean Squared Error)
- [x] SSE (Sum Squared Error)
- [x] MAE (Mean Absolute Error)
- [x] LogCosH (Log of Hyperbolic cosine)
- [x] Huber (Hyperbolic Tangent)
- [ ] Cross Entropy (Logistic Loss)
- [ ] Least Squares 

### 3. Optimization Algorithms:

- [ ] Batch Gradient Descent
- [ ] SGD (Stochastic Gradient Descent)
- [ ] Mini-Batch Gradient Descent
- [ ] General Purpose Gradient Descent
- [ ] ADAM (Adaptive Moment Estimation)
- [ ] RMSProp

### 4. Utility Functions:

- [x] Download MNIST dataset from remote server
- [x] Flatten (Convert 2D Matrix to vector)
- [x] One Hot Encoding (Convert numerical to categorical)
- [x] One Hot Decoding (Convert categorical to numerical)
- [x] Normalization Function (Linear Mapping between 0 and 1)
- [x] Accurary function (Compute Model Accuracy)
- [x] Activate (Applies given activation function)
- [x] Compute Loss (with chosen loss function)
- [ ] Optimize (Applies given optimizer on model)
- [x] Shuffle (Shuffles training data)
