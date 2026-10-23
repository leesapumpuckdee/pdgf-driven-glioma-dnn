import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path


#project folders
project_dir = Path(__file__).resolve().parent.parent

data_dir = project_dir / "data"
results_dir = project_dir / "results"


#same network structure as trained sigmoid model
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        raw_output = self.fc3(x)

        Gperc = torch.sigmoid(
            raw_output[:, 0:1]
        )

        Nperc = torch.sigmoid(
            raw_output[:, 1:2]
        )

        Tterm = raw_output[:, 2:3]

        return torch.cat(
            (Gperc, Nperc, Tterm),
            dim=1
        )


#load trained surrogate
model = Net()

model.load_state_dict(
    torch.load(
        results_dir / "tumor_dnn_model_sigmoid.pth",
        weights_only=True
    )
)

model.eval()


#freeze neural network weights
for parameter in model.parameters():
    parameter.requires_grad = False


print("Trained surrogate loaded and frozen.")


#load normalization values
x_mean = np.load(
    results_dir / "x_mean.npy"
)

x_std = np.load(
    results_dir / "x_std.npy"
)

y_mean = np.load(
    results_dir / "y_mean.npy"
)

y_std = np.load(
    results_dir / "y_std.npy"
)


#example experimental/target output
#replace these later with actual tumor data
Y0 = np.array([
    0.25,
    0.60,
    55.0
], dtype=np.float32)


#initial parameter guess X0
X0 = np.array([
    88500.0,
    218.5,
    0.355
], dtype=np.float32)


#normalize X0
X0_normalized = (
    X0 - x_mean
) / x_std


#convert to tensor
X0_tensor = torch.tensor(
    X0_normalized,
    dtype=torch.float32
).unsqueeze(0)


#normalize target
Y0_normalized = (
    Y0 - y_mean
) / y_std

Y0_tensor = torch.tensor(
    Y0_normalized,
    dtype=torch.float32
).unsqueeze(0)


#surrogate prediction at X0
prediction = model(
    X0_tensor
)


#objective function
#C(X) = ||Y0 - m(X)||^2
cost = torch.sum(
    (
        Y0_tensor
        - prediction
    ) ** 2
)


print(
    "Initial parameter guess:",
    X0
)

print(
    "Initial cost C(X0):",
    cost.item()
)

#next steps:
#1. compute the gradient of the cost function C(X)
#   with respect to alpha_a, gamma_a, and mu
#
#2. use the gradient to update the parameter guess
#   using standard gradient descent
#
#3. try a few different inverse learning rates eta 
#
#4. save C(X) at each iteration
#
#5. plot C(X) versus iteration to see how quickly
#   the optimization converges and whether eta is
#   too small, reasonable, or too large, be experimental!!!!
#
#6. once the gradient-descent version is working,
#   compare the estimated parameters to the known
#   parameter values and later move to real
#   experimental tumor targets