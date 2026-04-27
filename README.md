<img width="181" height="117" alt="Screenshot 2026-03-22 102133" src="https://github.com/user-attachments/assets/1738cedd-e76e-4c61-a549-2d8209645e22" />
# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

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

## PROGRAM

### Name:GOWTHAM G T

### Register Number:212224110017

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here

```

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
