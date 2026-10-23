import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import KFold
from pathlib import Path

#output folder
output_dir = Path("09_04_2026")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Results will be saved in: {output_dir.resolve()}")


#loading the datasettt
data = pd.read_csv(
    "regression-09-04-2026.csv",
    sep="\t",
    skiprows=1,
    names=[
        "alpha_a",
        "gamma_a",
        "mu",
        "Gperc",
        "Nperc",
        "Tterm"
    ]
)

#make sure all values are numerical
data = data.apply(pd.to_numeric, errors="raise")

#inputs
x = data.iloc[:, 0:3].to_numpy(dtype=np.float32)

#outputs
y = data.iloc[:, 3:6].to_numpy(dtype=np.float32)

print("Total number of simulations:", len(x))


#create development and final test sets
#the final test set is NOT used during cross-validation

num_data = x.shape[0]

indices = np.arange(num_data)

#fixed seed so the split is reproducible
np.random.seed(1)
np.random.shuffle(indices)

#85% development data
#15% final test data

dev_end = int(0.85 * num_data)

development_indices = indices[:dev_end]
test_indices = indices[dev_end:]

x_dev = x[development_indices]
y_dev = y[development_indices]

x_test_raw = x[test_indices]
y_test_raw = y[test_indices]

print("Development samples:", len(x_dev))
print("Final test samples:", len(x_test_raw))

#save indices so we can use the exact same test set later
np.save(
    output_dir / "cv_development_indices.npy",
    development_indices
)

np.save(
    output_dir / "cv_test_indices.npy",
    test_indices
)


#defining our dnn

#hidden_size is now a hyperparameter
#so we can compare 16, 32, and 64 neurons

class Net(nn.Module):

    def __init__(self, hidden_size):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(
            3,
            hidden_size
        )

        self.fc2 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.fc3 = nn.Linear(
            hidden_size,
            3
        )

    def forward(self, x):

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        #linear output because this is regression
        x = self.fc3(x)

        return x


#create 5-folds

#the development data will be divided into 5 folds
#each fold becomes the validation set exactly once

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)


#cross-validation function

def cross_validate(
    learning_rate,
    hidden_size,
    num_epochs
):

    #store validation loss from each fold
    fold_losses = []

    #repeat training 5 times
    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(x_dev)
    ):

        print(f"Fold {fold + 1}/5")
        print(f"Learning rate = {learning_rate}")
        print(f"Hidden size = {hidden_size}")

        #split

        x_train_raw = x_dev[train_idx]
        y_train_raw = y_dev[train_idx]

        x_val_raw = x_dev[val_idx]
        y_val_raw = y_dev[val_idx]

        print(
            "Training samples:",
            len(x_train_raw)
        )

        print(
            "Validation samples:",
            len(x_val_raw)
        )


        #normalization

        #IMPORTANT:
        #calculate normalization values ONLY
        #using the training portion of this fold

        x_mean = np.mean(
            x_train_raw,
            axis=0
        )

        x_std = np.std(
            x_train_raw,
            axis=0
        )

        #avoid division by zero
        x_std[x_std == 0] = 1.0


        #only normalize Tterm
        #Gperc and Nperc remain unchanged

        y_mean = np.array([
            0.0,
            0.0,
            np.mean(
                y_train_raw[:, 2]
            )
        ], dtype=np.float32)

        y_std = np.array([
            1.0,
            1.0,
            np.std(
                y_train_raw[:, 2]
            )
        ], dtype=np.float32)

        if y_std[2] == 0:
            y_std[2] = 1.0


        #normalize training data

        x_train = (
            x_train_raw - x_mean
        ) / x_std

        y_train = (
            y_train_raw - y_mean
        ) / y_std


        #normalize validation data
        #using TRAINING statistics

        x_val = (
            x_val_raw - x_mean
        ) / x_std

        y_val = (
            y_val_raw - y_mean
        ) / y_std

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import KFold
from pathlib import Path


#output folder

output_dir = Path("09_04_2026")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Results will be saved in: {output_dir.resolve()}")


#loading the dataset

data = pd.read_csv(
    "regression-09-04-2026.csv",
    sep="\t",
    skiprows=1,
    names=[
        "alpha_a",
        "gamma_a",
        "mu",
        "Gperc",
        "Nperc",
        "Tterm"
    ]
)

#make sure all values are numerical
data = data.apply(pd.to_numeric, errors="raise")

#inputs
x = data.iloc[:, 0:3].to_numpy(dtype=np.float32)

#outputs
y = data.iloc[:, 3:6].to_numpy(dtype=np.float32)

print("Total number of simulations:", len(x))


#creating development and final test sets

#the final test set is NOT used during cross-validation

num_data = x.shape[0]

indices = np.arange(num_data)

#fixed seed so the split is reproducible
np.random.seed(1)
np.random.shuffle(indices)

#85% development data
#15% final test data

dev_end = int(0.85 * num_data)

development_indices = indices[:dev_end]
test_indices = indices[dev_end:]

x_dev = x[development_indices]
y_dev = y[development_indices]

x_test_raw = x[test_indices]
y_test_raw = y[test_indices]

print("Development samples:", len(x_dev))
print("Final test samples:", len(x_test_raw))

#save indices so we can use the exact same test set later

np.save(
    output_dir / "cv_development_indices.npy",
    development_indices
)

np.save(
    output_dir / "cv_test_indices.npy",
    test_indices
)


#defining our dnn

#hidden_size is passed into the network
#for now we keep hidden_size fixed at 32

class Net(nn.Module):

    def __init__(self, hidden_size):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(
            3,
            hidden_size
        )

        self.fc2 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.fc3 = nn.Linear(
            hidden_size,
            3
        )

    def forward(self, x):

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        #linear output because this is regression
        x = self.fc3(x)

        return x


#=========================================================
# CREATE 5 FOLDS
#=========================================================

#the development data will be divided into 5 folds
#each fold becomes the validation set exactly once

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)


#cross-validation function

def cross_validate(
    learning_rate,
    hidden_size,
    num_epochs
):

    #store validation loss from each fold
    fold_losses = []

    #repeat training 5 times
    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(x_dev)
    ):

        print("\n================================")
        print(f"Fold {fold + 1}/5")
        print(f"Learning rate = {learning_rate}")
        print(f"Hidden size = {hidden_size}")
        print("================================")


        #splitting the development data into training and validation sets for this fold

        x_train_raw = x_dev[train_idx]
        y_train_raw = y_dev[train_idx]

        x_val_raw = x_dev[val_idx]
        y_val_raw = y_dev[val_idx]

        print(
            "Training samples:",
            len(x_train_raw)
        )

        print(
            "Validation samples:",
            len(x_val_raw)
        )


        #noramlization

        #calculate normalization values ONLY
        #using the training portion of this fold

        x_mean = np.mean(
            x_train_raw,
            axis=0
        )

        x_std = np.std(
            x_train_raw,
            axis=0
        )

        #avoid division by zero
        x_std[x_std == 0] = 1.0


        #only normalize Tterm
        #Gperc and Nperc remain unchanged

        y_mean = np.array([
            0.0,
            0.0,
            np.mean(
                y_train_raw[:, 2]
            )
        ], dtype=np.float32)

        y_std = np.array([
            1.0,
            1.0,
            np.std(
                y_train_raw[:, 2]
            )
        ], dtype=np.float32)

        if y_std[2] == 0:
            y_std[2] = 1.0


        #normalize training data

        x_train = (
            x_train_raw - x_mean
        ) / x_std

        y_train = (
            y_train_raw - y_mean
        ) / y_std


        #normalize validation data
        #using TRAINING statistics

        x_val = (
            x_val_raw - x_mean
        ) / x_std

        y_val = (
            y_val_raw - y_mean
        ) / y_std


        #convert to PyTorch tensors

        x_train = torch.tensor(
            x_train,
            dtype=torch.float32
        )

        y_train = torch.tensor(
            y_train,
            dtype=torch.float32
        )

        x_val = torch.tensor(
            x_val,
            dtype=torch.float32
        )

        y_val = torch.tensor(
            y_val,
            dtype=torch.float32
        )


        #create a new neural network for this fold

        #important:
        #every fold starts with a fresh neural network

        torch.manual_seed(1)

        model = Net(
            hidden_size=hidden_size
        )

        #same loss function as the original DNN
        criterion = nn.MSELoss()

        #learning rate changes depending on
        #which learning rate we are testing
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate
        )


        #training loop for this fold

        for epoch in range(num_epochs):

            model.train()

            #forward pass
            train_outputs = model(
                x_train
            )

            #calculate training loss
            loss = criterion(
                train_outputs,
                y_train
            )

            #clear gradients from previous step
            optimizer.zero_grad()

            #backpropagation
            loss.backward()

            #update weights and biases
            optimizer.step()


        #validation loss for this fold

        model.eval()

        with torch.no_grad():

            val_outputs = model(
                x_val
            )

            val_loss = criterion(
                val_outputs,
                y_val
            )


        #save validation loss for this fold

        fold_losses.append(
            val_loss.item()
        )

        print(
            f"Fold {fold + 1} "
            f"Validation Loss: "
            f"{val_loss.item():.8f}"
        )


   #after all folds are complete, calculate mean and std of validation loss

    mean_validation_loss = np.mean(
        fold_losses
    )

    std_validation_loss = np.std(
        fold_losses
    )

    print("5-FOLD CROSS-VALIDATION SUMMARY")

    print(
        "Validation losses:",
        fold_losses
    )

    print(
        f"Mean validation loss: "
        f"{mean_validation_loss:.8f}"
    )

    print(
        f"STD validation loss: "
        f"{std_validation_loss:.8f}"
    )


    return (
        mean_validation_loss,
        std_validation_loss,
        fold_losses
    )


#these are the hyperparameters we want to test during cross-validation

#for now, only change the learning rate
#keep the hidden-layer size fixed at 32

learning_rates = [
    0.01,
    0.005,
    0.001
]

hidden_size = 32

#keep the same number of epochs for every learning rate
#so the comparison is fair

num_epochs = 20000


#store all cross-validation results

cv_results = []


#actually run cross-validation for each learning rate

for learning_rate in learning_rates:

    print("TESTING LEARNING RATE")
    print(f"Learning rate = {learning_rate}")


    mean_loss, std_loss, fold_losses = (
        cross_validate(
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            num_epochs=num_epochs
        )
    )


    #save results for this learning rate

    cv_results.append({

        "Learning_Rate":
            learning_rate,

        "Mean_Validation_Loss":
            mean_loss,

        "STD_Validation_Loss":
            std_loss,

        "Fold_1":
            fold_losses[0],

        "Fold_2":
            fold_losses[1],

        "Fold_3":
            fold_losses[2],

        "Fold_4":
            fold_losses[3],

        "Fold_5":
            fold_losses[4]
    })


#creating results DataFrame for easier analysis

results_df = pd.DataFrame(
    cv_results
)

#sort so the learning rate with the
#LOWEST mean validation loss appears first

results_df = results_df.sort_values(
    "Mean_Validation_Loss"
)

print("\n\n")
print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")


print(results_df)


#save results to CSV so we can analyze them later

results_df.to_csv(
    output_dir / "learning_rate_5fold_cv.csv",
    index=False
)

print(
    "\nCross-validation results saved to:",
    output_dir / "learning_rate_5fold_cv.csv"
)


#this will print the best learning rate!
#the first row is the best because
#the table was sorted from lowest to highest validation loss

best_result = results_df.iloc[0]

best_learning_rate = (
    best_result["Learning_Rate"]
)

print("BEST LEARNING RATE")

print(
    "Best learning rate:",
    best_learning_rate
)

print(
    "Mean validation loss:",
    best_result[
        "Mean_Validation_Loss"
    ]
)

print(
    "Standard deviation:",
    best_result[
        "STD_Validation_Loss"
    ]
)

#my next steps for hyperparameter tuning:

#1. compare the mean 5-fold validation loss for
#   learning rates 0.01, 0.005, and 0.001
#
#2. select the learning rate with the LOWEST
#   mean validation loss
#
#3. once the best learning rate is selected,
#   repeat 5-fold cross-validation to test other
#   hyperparameters if needed, such as:
#
#       hidden size = 16, 32, 64
#
#4. after all hyperparameters are selected,
#   train ONE final neural network using ALL
#   development data
#
#5. evaluate that final model ONCE on the
#   untouched 15% test set
#
#6. calculate final MAE, RMSE, and R^2
#
#7. after validating the final surrogate,
#   move on to the deterministic inverse
#   parameter-estimation problem
#
#IMPORTANT:
#the final test set should NOT be used to choose
#the learning rate or any other hyperparameters
        