# Car Price Prediction Model

## 1. Data Preparation

The dataset is loaded from a CSV file.
Some columns (like running) are cleaned and normalized.
Categorical features (e.g., color, type, status, model, motor_type) are one-hot encoded.
The wheel column is set to 1 for all rows.
Independent (X) and dependent (y) variables are extracted for training.

## 2. Model Evolution

### Basic Linear Model

The first approach was a simple linear regression model using torch.nn.Linear.
This model had limited capacity and struggled with complex relationships in the data.

### Adding Adam Optimizer

The optimization strategy was improved by replacing the standard gradient descent with the Adam optimizer.
Adam provides adaptive learning rates, which help the model converge faster.

### Increasing Model Complexity (Linear Layers 32 & 64)

The model was expanded to include multiple fully connected (Linear) layers:
First layer: 64 neurons
Second layer: 32 neurons
Third layer: 16 neurons
Output layer: 1 neuron (predicting price)
ReLU activation was used to introduce non-linearity.

### Adding Dropout and Batch Normalization

Batch Normalization (BatchNorm1d) was introduced after each linear layer to stabilize training and accelerate convergence.

Dropout layers (Dropout(0.3)) were added to prevent overfitting by randomly disabling neurons during training.

The final architecture is as follows:

```python
class CustomNN(nn.Module):
    def __init__(self, n_coeff):
        super(CustomNN, self).__init__()
        
        self.fc1 = nn.Linear(n_coeff, 64)
        self.bn1 = nn.BatchNorm1d(64)  
        self.dropout1 = nn.Dropout(0.3) 
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(16, 1) 
        
        self.activation = nn.ReLU()  

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.activation(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        return self.fc4(x)
```
3. Training Process
Loss function: Mean Squared Error (MSE) was used to measure prediction accuracy.

Optimizer: Adam optimizer with a learning rate of 0.001.

Epochs: The model was trained for 800 epochs.

Training loop:

```
def train_model(epochs=800, lr=0.001):
    torch.manual_seed(442)
    coeffs = init_coeffs()
    optimizer = torch.optim.Adam(coeffs.parameters(), lr=lr)
    for i in range(epochs): 
        loss = one_epoch(coeffs, optimizer, lr=lr)
        losses.append(loss)
    return coeffs
```
4. Prediction & Loss Calculation

Predictions are generated using:

```
def calc_preds(coeffs, indeps, denormalize=False):
    preds = coeffs(indeps)
    if denormalize:
        preds = preds * trn_dep_std + trn_dep_mean
    return preds
```

Loss is computed as:

```
def calc_loss(coeffs, indeps, deps): 
    preds = calc_preds(coeffs, indeps, denormalize=False)
    return torch.nn.functional.mse_loss(preds, deps.view(-1, 1))
```

5. Conclusion
This model was improved step-by-step from a basic linear model to a more complex deep learning architecture with batch normalization and dropout, allowing for better generalization and improved performance.

