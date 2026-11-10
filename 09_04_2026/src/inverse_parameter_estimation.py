import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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


#new parameter combination that was NOT one of the
#original PDE simulation points

X_true = np.array([
    90000.0,   #alpha_a
    150.0,     #gamma_a
    0.350      #mu
], dtype=np.float32)


#normalize X_true

X_true_normalized = (
    X_true - x_mean
) / x_std


X_true_tensor = torch.tensor(
    X_true_normalized,
    dtype=torch.float32
).unsqueeze(0)


#evaluate trained surrogate:
#Y0 = m_theta*(X_true)

with torch.no_grad():

    Y0_normalized_tensor = model(
        X_true_tensor
    )


#convert prediction back to original scale

Y0 = (
    Y0_normalized_tensor.numpy()[0]
    * y_std
    + y_mean
)


print("\nKnown off-grid input X_true:")
print(X_true)

print("\nm_theta*(X_true) = Y0:")
print(Y0)

print("Gperc =", Y0[0])
print("Nperc =", Y0[1])
print("Tterm =", Y0[2])

#initial guess for gradient descent
X0 = np.array([
    88500.0,
    218.5,
    0.355
], dtype=np.float32)


#normalize X0
X0_normalized = (
    X0 - x_mean
) / x_std


Y0_normalized = (
    Y0 - y_mean
) / y_std


Y0_tensor = torch.tensor(
    Y0_normalized,
    dtype=torch.float32
).unsqueeze(0)

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


#trying diff learning rates

#these learning rates are for changing alpha_a, gamma_a,
#and mu -- NOT for training the neural network

#chosen from previous eta testing
eta = 0.5

num_iterations = 200


def run_gradient_descent(eta):

    #start from the SAME initial guess for every eta
    x_guess = torch.tensor(
        X0_normalized,
        dtype=torch.float32,
        requires_grad=True
    )

    cost_history = []

    for iteration in range(num_iterations):

        #surrogate prediction
        prediction = model(
            x_guess.unsqueeze(0)
        )

        #objective function
        #C(X) = ||Y0 - m(X)||^2
        cost = torch.sum(
            (
                Y0_tensor
                - prediction
            ) ** 2
        )

        #save C(X) for this iteration
        cost_history.append(
            cost.item()
        )

        #clear gradient from previous iteration
        if x_guess.grad is not None:
            x_guess.grad.zero_()

        #compute gradient of C with respect to X
        cost.backward()

        #standard gradient descent:
        #X_new = X_old - eta * gradient
        with torch.no_grad():

            x_guess -= (
                eta
                * x_guess.grad
            )

    #convert final normalized parameters
    #back to original parameter scale

    estimated_X = (
        x_guess.detach().numpy()
        * x_std
        + x_mean
    )

    return cost_history, estimated_X

all_cost_histories = {}

#run experiment for each eta

all_cost_histories = {}

for eta in [eta]:

    cost_history, estimated_X = run_gradient_descent(eta)

    #save this eta's cost history
    all_cost_histories[eta] = cost_history

    print("\n-------------------------")
    print("eta =", eta)

    print(
        "Initial cost:",
        cost_history[0]
    )

    print(
        "Final cost:",
        cost_history[-1]
    )

    print(
        "Estimated parameters:",
        estimated_X
    )
#plotting 

plt.figure(
    figsize=(8, 6)
)

for eta in [eta]:

    plt.plot(
        range(num_iterations),
        all_cost_histories[eta],
        label=f"eta = {eta}"
    )

plt.xlabel("Iteration")
plt.ylabel("Cost C(X)")

plt.title(
    "Cost Function vs Iteration "
    "for Different Learning Rates"
)

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    results_dir /
    "inverse_cost_vs_iteration.png",
    dpi=300
)

plt.show()


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
#   too small, reasonable, or too large, be experimental!!!! COMPLETEDEDEDEDED
#
#6. once the gradient-descent version is working,
#   compare the estimated parameters to the known
#   parameter values and later move to real
#   experimental tumor targets 
# test 1
