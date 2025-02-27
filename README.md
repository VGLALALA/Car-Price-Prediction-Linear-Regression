# Car Price Prediction Model

This repository contains a deep learning model for car price prediction, progressing through multiple improvements from a basic linear model to an optimized architecture with dropout and batch normalization.

## 1. Data Preparation

- The dataset is loaded from a CSV file.
- Some columns (like `running`) are cleaned and normalized.
- Categorical features (e.g., `color`, `type`, `status`, `model`, `motor_type`) are one-hot encoded.
- The `wheel` column is set to `1` for all rows.
- Independent (`X`) and dependent (`y`) variables are extracted for training.

## 2. Model Evolution

### **Step 1: Basic Linear Model**
- The first approach was a simple linear regression model using `torch.nn.Linear`.
- This model had limited capacity and struggled with complex relationships in the data.

### **Step 2: Adding Adam Optimizer**
- The optimization strategy was improved by replacing the standard gradient descent with the **Adam optimizer**.
- Adam provides adaptive learning rates, which help the model converge faster.

### **Step 3: Increasing Model Complexity (Linear Layers 32 & 64)**
- The model was expanded to include multiple fully connected (`Linear`) layers:
  - **First layer:** 64 neurons
  - **Second layer:** 32 neurons
  - **Third layer:** 16 neurons
  - **Output layer:** 1 neuron (predicting price)
- **ReLU activation** was used to introduce non-linearity.

### **Step 4: Adding Dropout and Batch Normalization**
- **Batch Normalization (`BatchNorm1d`)** was introduced after each linear layer to stabilize training and accelerate convergence.
- **Dropout layers (`Dropout(0.3)`)** were added to prevent overfitting by randomly disabling neurons during training.
- The final architecture is as follows:

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

## 3. Mean Squared Error (MSE) Loss

Loss is computed as:
The **Mean Squared Error (MSE) Loss** function is used as the loss metric for this model. It measures the squared difference between predictions (`x`) and true values (`y`). The unreduced (i.e., with `reduction='none'`) loss is computed as:

\[\ell(x, y) = L = \{ l_1, l_2, ..., l_N \}^T, \quad l_n = (x_n - y_n)^2\]

where \( N \) is the batch size.

If `reduction` is set to a value other than `'none'` (default is `'mean'`), then:

\[\ell(x, y) =
egin{cases} 
	ext{mean}(L), & 	ext{if reduction} = 'mean' \ 
	ext{sum}(L), & 	ext{if reduction} = 'sum' 
\end{cases}\]

- **Mean (`'mean'`)**: The operation divides the total loss by \( N \), ensuring loss values remain normalized.
- **Sum (`'sum'`)**: The loss is computed as the sum of all squared differences without division by \( N \).

To avoid division by \( N \), one can set `reduction='sum'`.

The loss is computed in the training loop as follows:

```python
def calc_loss(coeffs, indeps, deps): 
    preds = calc_preds(coeffs, indeps, denormalize=False)
    return torch.nn.functional.mse_loss(preds, deps.view(-1, 1))
```

## 4. Training Process
- **Loss function**: Mean Squared Error (MSE) was used to measure prediction accuracy.
- **Optimizer**: Adam optimizer with a learning rate of `0.001`.
- **Epochs**: The model was trained for 800 epochs.
- Training loop:

  ```python
  def train_model(epochs=800, lr=0.001):
      torch.manual_seed(442)
      coeffs = init_coeffs()
      optimizer = torch.optim.Adam(coeffs.parameters(), lr=lr)
      for i in range(epochs): 
          loss = one_epoch(coeffs, optimizer, lr=lr)
          losses.append(loss)
      return coeffs
  ```

## 5. Prediction & Loss Calculation
- Predictions are generated using:

  ```python
  def calc_preds(coeffs, indeps, denormalize=False):
      preds = coeffs(indeps)
      if denormalize:
          preds = preds * trn_dep_std + trn_dep_mean
      return preds
  ```

## 6. Conclusion
This model was improved step-by-step from a basic linear model to a more complex deep learning architecture with batch normalization and dropout, allowing for better generalization and improved performance.
