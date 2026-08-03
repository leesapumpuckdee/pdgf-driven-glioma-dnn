import numpy as np #for arrays, normalization, splitting data
import pandas as pd #mainly for reading csv files
import torch #used for building the nn
import torch.nn as nn #used for building the nn layers like linear, conv, etc
import torch.nn.functional as F #used for activation functions like reLu, sigmoid, etc,
import torch.optim as optim #gives optimization algorithms such as SGD.

#dataset: 60 simulation data points generated via 5x5x5 Cartesian grid
#parameter sweeps using TumorDyn finite-volume PDE solver
#limited dataset size reflects computational cost of numerical simulations
#model performance validated on held-out test set (15% split)

#loading dataset
#the header uses commas, but the data rows use tabs
#so we skip the original header and provide the correct column names ourselves
data = pd.read_csv(
    "60point_125tumorbatch.csv",
    sep="\t",
    skiprows=1,
    names=["alpha_a", "gamma_a", "mu", "Gperc", "Nperc", "Tterm"]
)

#making sure every column contains numerical values
data = data.apply(pd.to_numeric, errors="raise")

print("Columns:", data.columns.tolist())
print(data.head())

#first 3 columns are input parameters: alpha_a, gamma_a, and mu
x = data.iloc[:, 0:3].to_numpy(dtype=np.float32)

#last 3 columns are outputs: Gperc, Nperc, and Tterm
y = data.iloc[:, 3:6].to_numpy(dtype=np.float32)

print("Input shape:", x.shape)
print("Output shape:", y.shape)

#splitting data into training, validation, + testing sets 70/15/15
#training set is used to update the weights of the model
#validation set is used to check the model and help choose hyperparameters
#testing set is used to evaluate the model after training is complete
num_data = x.shape[0] #total num of datapoints which is 125 for right now
indices = np.arange(num_data) #creates row indices for data

np.random.seed(1) #makes the random split reproducible so we can get same split every time we run the code
torch.manual_seed(1) #makes the model's random starting weights reproducible
np.random.shuffle(indices) #shuffles the indices so we can randomly split the data

train_end = int(0.7 * num_data)
val_end = int(0.85 * num_data)

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

#splitting the original data before normalization
x_train_raw = x[train_indices]
y_train_raw = y[train_indices]

x_val_raw = x[val_indices]
y_val_raw = y[val_indices]

x_test_raw = x[test_indices]
y_test_raw = y[test_indices]


#normalizing data because columns have diff ranges
#normalization means to subtract the mean & divide by the STD
#only the training data is used to find the mean and STD
#this keeps the validation and testing data unseen during training
x_mean = np.mean(x_train_raw, axis=0)
x_std = np.std(x_train_raw, axis=0)

#only normalizing Tterm
#using mean=0 and std=1 leaves Gperc and Nperc unchanged
#i am only normalizing Tterm here because gperc and nperc are already in the range of 0-1 because they are proportions!
y_mean = np.array([
    0.0,
    0.0,
    np.mean(y_train_raw[:, 2])
], dtype=np.float32)

y_std = np.array([
    1.0,
    1.0,
    np.std(y_train_raw[:, 2])
], dtype=np.float32)

#prevents division by zero if a column has no variation
x_std[x_std == 0] = 1.0

if y_std[2] == 0:
    y_std[2] = 1.0

#normalizing the training, validation, and testing data
#using the mean and STD from the training data
x_train = (x_train_raw - x_mean) / x_std
x_val = (x_val_raw - x_mean) / x_std
x_test = (x_test_raw - x_mean) / x_std

y_train = (y_train_raw - y_mean) / y_std
y_val = (y_val_raw - y_mean) / y_std
y_test = (y_test_raw - y_mean) / y_std

#converting numpy arrays into pytorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("Number of training data:", x_train.shape[0])
print("Number of validation data:", x_val.shape[0])
print("Number of test data:", x_test.shape[0])

#shallow nn has 1 hidden layer, a dnn has multiple,
#to make it deep we add 1 more hidden layer between them!!!
#similar to shallow MNIST nn but changed for regression instead of classification
class Net(nn.Module): #deep nn for predicting pdgf-driven glioma tumor behavior.
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 32) #input layer to first hidden layer
        self.fc2 = nn.Linear(32, 32) #first hidden layer to second hidden layer makes it deep!
        self.fc3 = nn.Linear(32, 3) #second hidden layer to output layer
        #the shallow MNIST nn had 1 hidden layer, this has 2 so its a small dnn
        #3 inputs - 32 neurons - 32 neurons - 3 outputs

    def forward(self, x): #forward function defines how data moves through the nn
     #ReLU keeps positive values and turns neg values into 0
        x = F.relu(self.fc1(x)) #activation function for first hidden layer ReLU instead of tanh
        x = F.relu(self.fc2(x)) #activation function for second hidden layer 
        x = self.fc3(x) #output layer (no activation function for regression) no softmax b/c regression
        return x
       
model = Net()
print(model)

#loss function + optimizer, because outputs are real-valued we used MSE (aka mean square error)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005) 
#the SGD optimizer updates the model parameters in the direction that lowers the loss (ch. 7)
#the entire training set is currently used for each update because the dataset is small
#model.parameters gives SGD the weights and biases
#lr is the learning rate controlling the step size of each update

#lists for saving loss history
epoch_history = []
training_loss_history = []
validation_loss_history = []

#training loop
num_epochs = 20000 #number of times we go through the entire training dataset
for epoch in range(num_epochs):
    model.train() 
    train_outputs = model(x_train) #forward pass; inputs go through nn
    loss = criterion(train_outputs, y_train) #compute loss; how far prediction is from true training outputs
    optimizer.zero_grad() #zero the gradients
    loss.backward() #backpropagation ch. 6!! computes gradients of the loss w.r.t model weights
    optimizer.step() #update weights mentioned in ch. 7 because it performs SGD update step; reduces loss

    if epoch % 100 == 0:
        model.eval()

        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)

            epoch_history.append(epoch)
            training_loss_history.append(loss.item())
            validation_loss_history.append(val_loss.item())

            print(f"Epoch [{epoch}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

#save loss history to a CSV file
loss_history = pd.DataFrame({
    "Epoch": epoch_history,
    "Training_Loss": training_loss_history,
    "Validation_Loss": validation_loss_history
})

loss_history.to_csv("loss_history.csv", index=False)

print("Loss history saved to loss_history.csv")

#model evaluation on test data
model.eval()
with torch.no_grad():
    test_output = model(x_test)
    test_loss = criterion(test_output, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

#converting predictions back to original scale for interpretation of predictions
test_predictions = test_output.numpy() * y_std + y_mean
test_actuals = y_test.numpy() * y_std + y_mean

print("Test Predictions (Gperc, Nperc, Tterm):")
for i in range(len(test_predictions)):
    print(f"Predicted: {test_predictions[i]}, Actual: {test_actuals[i]}")

results = pd.DataFrame({
    "Actual_Gperc": test_actuals[:, 0],
    "Predicted_Gperc": test_predictions[:, 0],

    "Actual_Nperc": test_actuals[:, 1],
    "Predicted_Nperc": test_predictions[:, 1],

    "Actual_Tterm": test_actuals[:, 2],
    "Predicted_Tterm": test_predictions[:, 2]
})

results.to_csv("test_predictions.csv", index=False)

print("Test predictions saved to test_predictions.csv")

#save model
torch.save(model.state_dict(), "tumor_dnn_model.pth")
np.save("x_mean.npy", x_mean)
np.save("x_std.npy", x_std)
np.save("y_mean.npy", y_mean)
np.save("y_std.npy", y_std)
#important because any new input MUST be normalized using same mean + STD
print("Model and normalization parameters saved.")

#saving the data split so the plotting file uses the exact same test data
np.save("train_indices.npy", train_indices)
np.save("val_indices.npy", val_indices)
np.save("test_indices.npy", test_indices)

print("Data split indices saved.")

#future notes from zeng! (ch. 9 related)
#if model overfit, we can try adding droppout layers, regularization, or early stopping but for now this model will remain a simple dnn