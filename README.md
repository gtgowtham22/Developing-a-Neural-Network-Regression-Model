# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1082" height="546" alt="546060726-cc3d99c0-5c33-4092-a242-00398dcb3334" src="https://github.com/user-attachments/assets/837e3489-3c2b-49bd-975a-95997e01cfde" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM:


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
     

dataset1 = pd.read_csv('DL-Exp1 - Sheet1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values
     

print(dataset1.head(5))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
     

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
     

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
     

# Name:SAJEN MURALI
# Register Number:212223220089
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x
     

# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)
     

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss=criterion(ai_brain(X_train),y_train)
    loss.backward()
    optimizer.step()


    ai_brain.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

     

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)


with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

     
Test Loss: 324.846680

loss_df = pd.DataFrame(ai_brain.history)
     

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
     


X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')



### Name:GOWTHAM G T

### Register Number:212224110017



### Dataset Information
<img width="207" height="173" alt="Screenshot 2026-04-27 083046" src="https://github.com/user-attachments/assets/b2ec682a-7c24-43c2-a2e7-e7c0a3e9da8a" />



### OUTPUT

<img width="586" height="180" alt="Screenshot 2026-04-27 083121" src="https://github.com/user-attachments/assets/2677cf50-c060-40d1-8ea0-ef4223ad3adb" />

### Training Loss Vs Iteration Plot
<img width="658" height="386" alt="Screenshot 2026-04-27 083027" src="https://github.com/user-attachments/assets/251c3dbe-6d63-4a4a-851f-e6a16c2e9e27" />


### New Sample Data Prediction
<img width="739" height="86" alt="Screenshot 2026-04-27 083141" src="https://github.com/user-attachments/assets/3f6170f9-1b23-4f78-b78b-53051fb775c0" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
