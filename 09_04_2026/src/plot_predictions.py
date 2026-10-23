import numpy as np #for numerical operations and array manipulations
import pandas as pd #for loading the dataset and doing some basic data manipulation
import torch #this is for loading the trained model and making predictions
import torch.nn as nn #this is for creating the neural network architecture
import torch.nn.functional as F #this is for activation functions like relu, sigmoid, etc.
import matplotlib #plotting library!
matplotlib.use("Agg")  #this is needed to make matplotlib work on the cluster without a display
import matplotlib.pyplot as plt #this import must come after matplotlib.use("Agg") to avoid errors on the cluster!!!!! IMPORTANT
from pathlib import Path

#folder containing the trained model files + where plots will be saved
output_dir = Path("09_04_2026")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Reading/saving model results in: {output_dir.resolve()}")

#loading dataset! 
data = pd.read_csv(
    "regression-09-04-2026.csv", 
    sep="\t", 
    skiprows=1, #this skips the first row of the file, containing the file names
    names=["alpha_a", "gamma_a", "mu", "Gperc", "Nperc", "Tterm"]
)

data = data.apply( 
    pd.to_numeric,
    errors="raise"
)  #making sure every column contains numerical values

#first 3 columns are input parameters: alpha_a, gamma_a, and mu
x = data.iloc[:, 0:3].to_numpy(dtype=np.float32)

#last 3 columns are outputs: Gperc, Nperc, and Tterm
y = data.iloc[:, 3:6].to_numpy(dtype=np.float32)

#our normalization parameters created during training!!
#it is important to reuse the TRAINING means and standard deviations here
#recalculating them from the test set would leak test information and give the model data on a different scale
x_mean = np.load(output_dir / "x_mean.npy")
x_std = np.load(output_dir / "x_std.npy")
y_mean = np.load(output_dir / "y_mean.npy")
y_std = np.load(output_dir / "y_std.npy")

test_indices = np.load(output_dir / "test_indices.npy")

#selecting testing data
x_test_raw = x[test_indices]
y_test_raw = y[test_indices]

#converting testing data into pytorch tensors
#normalizing inside torch.tensor() is a compact choice; some scripts do these as two separate steps
#the result is the same as long as the saved means and standard deviations match the trained model
x_test = torch.tensor((x_test_raw - x_mean) / x_std, dtype=torch.float32)
y_test = torch.tensor((y_test_raw - y_mean) / y_std, dtype=torch.float32)

#this is from the DNN  
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
#map_location="cpu" makes loading work even if the model was originally saved on a GPU
model.load_state_dict(
    torch.load(
        output_dir / "tumor_dnn_model.pth",
        map_location="cpu"
    )
) #loading the trained model
model.eval()  #putting the model in evaluation mode

#model used to predict testing outputs
with torch.no_grad():  #disabling gradient calculation to save memory and computations
    test_predictions_scaled = model(x_test)  #predicting outputs for testing data

#converting tensors to numpy arrays
test_predictions_scaled = test_predictions_scaled.numpy()
y_test_scaled = y_test.numpy()

#converting back to og scale!! das why saving values r important
#this manual multiplication and addition is the inverse of z-score normalization: original = scaled * std + mean
#using saved NumPy arrays instead of a scaler object is simple and keeps this consistent with the training script
test_predictions = test_predictions_scaled * y_std + y_mean
test_actuals = y_test_scaled * y_std + y_mean

output_names = ["Gperc", "Nperc", "Tterm"]

#titles for the plots
display_names = [
    "Glioma Cell Proportion",
    "Immune Cell Proportion",
    "Terminal Time"
]

#units for the plots
units = [
    "Proportion",
    "Proportion",
    "Time"
]

#calculate metrics for each output
errors = test_predictions - test_actuals
absolute_errors = np.abs(errors)

#axis=0 is important because it calculates one metric for each output column instead of one number for the whole model
mae = np.mean(absolute_errors, axis=0)  #mean absolute error for each output
rmse = np.sqrt(np.mean(errors**2, axis=0))  #root mean squared error for each output

#correct r2 calculation for multiple outputs
#ss_res = the variation of predicted values around the actual values
#ss_tot = the variation of actual values around their mean
ss_res = np.sum((test_actuals - test_predictions)**2, axis=0)
ss_tot = np.sum(
    (test_actuals - np.mean(test_actuals, axis=0))**2,
    axis=0
)

#prevent division by 0 in case ss_tot is 0 for any output
#this is a manual multi-output R2 calculation instead of using sklearn.metrics.r2_score
#returning nan is intentional because R2 is undefined if every actual value for an output is identical
r2 = np.full(ss_res.shape, np.nan, dtype=np.float64)
np.divide(ss_res, ss_tot, out=r2, where=ss_tot != 0)
r2 = 1 - r2

#printing metrics for each output
print("\nMean Absolute Error (MAE) for each output:")
for i, name in enumerate(output_names):
    print(f"{name}: {mae[i]:.8f}")

print("\nRoot Mean Squared Error (RMSE) for each output:")
for i, name in enumerate(output_names):
    print(f"{name}: {rmse[i]:.8f}")

print("\nR-squared (R2) for each output:")
for i, name in enumerate(output_names):
    print(f"{name}: {r2[i]:.4f}")

#the fun part! styling w/ matplotlib

#rcParams changes the defaults for every plot created after this line, not just the next figure
#this avoids repeating the same poster styling for all four saved figures
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#F8FAFC"
})

plot_colors = [
    "#C05AD9",  #g% - purple
    "#7A7A7A",  #n% - gray
    "#16A34A"   #ttime - green
]


def get_plot_limits(actual, predicted):
    #using the same limits on both axes makes the y=x line a fair visual comparison
    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())

    data_range = maximum - minimum
    padding = data_range * 0.12 if data_range != 0 else 0.1

    return minimum - padding, maximum + padding


def add_perfect_prediction_line(ax, lower_limit, upper_limit, show_label=False):
    #the y=x reference line shows where predicted values exactly equal actual values
    label = "Perfect prediction" if show_label else None

    ax.plot(
        [lower_limit, upper_limit],
        [lower_limit, upper_limit],
        linestyle="--",
        linewidth=2,
        color="#DC2626",
        label=label,
        zorder=2
    )


def label_test_points(
    ax,
    actual,
    predicted,
    label_prefix,
    offset,
    color=None
):
    #these labels show test-set order, not original row numbers from the full dataset
    for point_number, (x_value, y_value) in enumerate(
        zip(actual, predicted),
        start=1
    ):
        annotation_options = {
            "xytext": offset,
            "textcoords": "offset points",
            "fontsize": 13,
            "fontweight": "bold"
        }

        if color is not None:
            annotation_options["color"] = color

        ax.annotate(
            f"{label_prefix}{point_number}",
            (x_value, y_value),
            **annotation_options
        )


def add_metrics_box(ax, column, fontsize, box_padding):
    metrics_text = (
        f"MAE: {mae[column]:.4f}\n"
        f"RMSE: {rmse[column]:.4f}\n"
        f"$R^2$: {r2[column]:.4f}"
    )

    #transAxes uses relative coordinates, so the box stays in the upper-left corner
    #even when the numerical ranges are very different between outputs
    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=fontsize,
        bbox={
            "boxstyle": f"round,pad={box_padding}",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": 0.95
        }
    )


#3-panel figure for all outputs
fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5.2),
    constrained_layout=False
)

for column, ax in enumerate(axes):
    actual = test_actuals[:, column]
    predicted = test_predictions[:, column]
    lower_limit, upper_limit = get_plot_limits(actual, predicted)

    ax.scatter(
        actual,
        predicted,
        s=95,
        color=plot_colors[column],
        edgecolor="white",
        linewidth=1.6,
        alpha=0.9,
        zorder=3
    )

    add_perfect_prediction_line(ax, lower_limit, upper_limit)
    label_test_points(
        ax,
        actual,
        predicted,
        label_prefix="",
        offset=(5, 5),
        color="#334155"
    )

    ax.set_xlim(lower_limit, upper_limit)
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_xlabel(f"Actual {units[column]}", fontsize=15)
    ax.set_ylabel(f"Predicted {units[column]}", fontsize=15)
    ax.set_title(display_names[column], fontsize=18, pad=6)

    add_metrics_box(
        ax,
        column,
        fontsize=11,
        box_padding=0.45
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.7,
        alpha=0.4
    )

plt.tight_layout(
    rect=[0.02, 0.08, 0.98, 0.92],
    w_pad=1.5
)

plt.savefig(
    output_dir / "actual_vs_predicted_all_outputs.svg",
    format="svg",
    bbox_inches="tight",
    pad_inches=0.03,
    facecolor="white"
)

plt.close(fig)


#individual figure for each output
for column in range(3):
    actual = test_actuals[:, column]
    predicted = test_predictions[:, column]
    lower_limit, upper_limit = get_plot_limits(actual, predicted)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    ax.scatter(
        actual,
        predicted,
        s=150,
        color=plot_colors[column],
        edgecolor="white",
        linewidth=2,
        alpha=0.9,
        zorder=3
    )

    add_perfect_prediction_line(
        ax,
        lower_limit,
        upper_limit,
        show_label=True
    )

    label_test_points(
        ax,
        actual,
        predicted,
        label_prefix="Test ",
        offset=(7, 7)
    )

    ax.set_xlim(lower_limit, upper_limit)
    ax.set_ylim(lower_limit, upper_limit)

    #equal aspect keeps the perfect-prediction line from looking distorted
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"Actual {display_names[column]}")
    ax.set_ylabel(f"Predicted {display_names[column]}")
    ax.set_title(
        f"Actual vs. Predicted\n{display_names[column]}",
        pad=14
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.8,
        alpha=0.45
    )

    add_metrics_box(
        ax,
        column,
        fontsize=13,
        box_padding=0.55
    )

    ax.legend(loc="lower right")
    plt.tight_layout()

    #SVG is a vector format, im using svg so when i resize it on my poster it doesn't get pixelated like a raster image would
    plt.savefig(
        output_dir / f"actual_vs_predicted_{output_names[column]}.svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)


print("\nPoster-quality figures saved successfully:")
print(f"  {output_dir / 'actual_vs_predicted_all_outputs.svg'}")
print(f"  {output_dir / 'actual_vs_predicted_Gperc.svg'}")
print(f"  {output_dir / 'actual_vs_predicted_Nperc.svg'}")
print(f"  {output_dir / 'actual_vs_predicted_Tterm.svg'}")