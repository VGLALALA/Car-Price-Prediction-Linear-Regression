# 🚗 Car Price Prediction Model

This repository contains a deep learning model for car price prediction, progressing through multiple improvements from a basic linear model to an optimized architecture with dropout and batch normalization.

---

## 📊 Data Preparation

### **Loading the Dataset**

The dataset is loaded from a CSV file. If running in a Kaggle environment, the dataset is accessed from the competition directory. Otherwise, it is downloaded and extracted.

```python
import os
from pathlib import Path
import pandas as pd

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle:
    path = Path('../input/cartrain')
else:
    path = Path('')
    if not path.exists():
        import zipfile, kaggle
        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f'{path}.zip').extractall(path)

df = pd.read_csv(path/'train.csv')
```

### **Feature Selection**

A subset of relevant features is selected for training. The `indep_cols` list contains independent variables (`X`), including categorical and numerical features.

```python
indep_cols = [
    "year",
    "running",
    "motor_volume",
    "color_beige", "color_black", "color_blue", "color_brown", "color_cherry",
    "color_clove", "color_golden", "color_gray", "color_green", "color_orange",
    "color_other", "color_pink", "color_purple", "color_red", "color_silver",
    "color_skyblue", "color_white",
    "type_Coupe", "type_Universal", "type_hatchback", "type_minivan / minibus",
    "type_pickup", "type_sedan", "type_suv",
    "status_crashed", "status_excellent", "status_good", "status_new", "status_normal",
    "model_hyundai", "model_kia", "model_mercedes-benz", "model_nissan", "model_toyota",
    "motor_type_diesel", "motor_type_gas", "motor_type_hybrid",
    "motor_type_petrol", "motor_type_petrol and gas"
]
```

### **Data Normalization and Preprocessing**

A function is defined to clean and normalize the dataset:

```python
def dataNormalize(df, mode="train"):
    # Convert 'running' column from string to numerical values (handling miles to km conversion)
    df[['number', 'unit']] = df['running'].str.split(expand=True)
    df['number'] = df['number'].astype(float)
    df.loc[df['unit'] == 'miles', 'number'] *= 1.60934  # Convert miles to kilometers
    df['running'] = df['number']
    df.drop(columns=['number', 'unit'], inplace=True)
    df["running"] = pd.to_numeric(df["running"], errors="coerce")
    
    # Convert 'price' column to float if in training mode
    if mode == "train":
        df['price'] = df['price'].astype(float)
    
    # One-hot encode categorical features
    categorical_columns = ["color", "type", "status", "model", "motor_type"]
    df = pd.get_dummies(df, columns=categorical_columns, dtype=int, prefix=categorical_columns)
    
    # Set 'wheel' column to 1 for all rows
    df["wheel"] = 1  
    
    return df

df = dataNormalize(df)
```

### **Summary of Data Preparation**

- **Loading the dataset**: Reads data from a CSV file, automatically downloading it if necessary.
- **Feature selection**: Defines independent variables for training.
- **Preprocessing**:
  - Splits and converts `running` column into numerical values (converts miles to km if needed).
  - Converts `price` to float type in training mode.
  - One-hot encodes categorical columns.
  - Sets the `wheel` column to `1` for all rows.

The dataset now contains **42 processed columns** after normalization and feature engineering.

---

## 🏗️ Model Evolution

### **Basic Linear Model**

The first approach was a simple linear regression model using `torch.nn.Linear`, which had limited capacity and struggled with complex relationships in the data.

#### **Original Model Implementation**

```python
import torch

torch.manual_seed(442)

n_coeff = t_indep.shape[1]
coeffs = torch.rand(n_coeff) - 0.5
preds = (t_indep * coeffs).sum(axis=1)
loss = torch.abs(preds - t_dep).mean()
```
#### **Increasing Model Complexity**

```python
  class CustomNN(nn.Module):
      def __init__(self, n_coeff):
          super(CustomNN, self).__init__()
          
          self.fc1 = nn.Linear(n_coeff, 64)
          
          self.fc2 = nn.Linear(64, 32)

          self.fc3 = nn.Linear(32, 16)

          self.fc4 = nn.Linear(16, 1) 
          
          self.activation = nn.ReLU()  

      def forward(self, x):
          x = self.activation(self.fc1(x))
          
          x = self.activation(self.fc2(x))

          x = self.activation(self.fc3(x))

          return self.fc4(x)  
  ```

#### **Adding Dropout and Batch Normalization**
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
---

### **Adding Adam Optimizer**

The optimization strategy was improved by replacing standard gradient descent with the **Adam optimizer**, which provides adaptive learning rates to speed up convergence.

#### **Adam Optimizer Implementation**

```python
optimizer = torch.optim.Adam(coeffs.parameters(), lr=lr)

def one_epoch(coeffs, optimizer, lr):
    optimizer.zero_grad()
    loss = calc_loss(coeffs, trn_indep, trn_dep)
    loss.backward()
    optimizer.step()
    return round(loss.item(), 3)
```

---

## 📉 Mean Squared Error (MSE) Loss

Loss is computed using **Mean Squared Error (MSE) Loss**:

[MSELoss Documentation](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)

```python
def calc_loss(coeffs, indeps, deps):
    preds = calc_preds(coeffs, indeps, denormalize=False)
    return torch.nn.functional.mse_loss(preds, deps.view(-1, 1))
```

---

## 🎯 Training Process

- **Loss function:** Mean Squared Error (MSE) measures prediction accuracy.
- **Optimizer:** Adam optimizer with a learning rate of `0.001`.
- **Epochs:** The model was trained for **800 epochs**.

#### **Training Loop**

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

---

## Final Result
![Final Result](4.png)
