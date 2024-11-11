import json
import matplotlib.pyplot as plt

# Load data from JSON file
with open("report/training.json", "r") as file:
    data = json.load(file)

# Extract epochs and metrics
epochs = list(map(int, data.keys()))
train_rmse = [data[str(epoch)]["train_rmse"] for epoch in epochs]
train_mse = [data[str(epoch)]["train_mse"] for epoch in epochs]
val_rmse = [data[str(epoch)]["val_rmse"] for epoch in epochs]
val_mse = [data[str(epoch)]["val_mse"] for epoch in epochs]

# Plotting
plt.figure(figsize=(10, 5))

# RMSE Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_rmse, marker='s', color='b', label="Train RMSE", linewidth=1.2)
plt.plot(epochs, val_rmse, marker='s', color='r', label="Validation RMSE", linewidth=1.2)
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("RMSE", fontsize=13)
plt.legend(fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=12)

# MSE Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_mse, marker='s', color='b', label="Train MSE", linewidth=1.2)
plt.plot(epochs, val_mse, marker='s', color='r', label="Validation MSE", linewidth=1.2)
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("MSE", fontsize=13)
plt.legend(fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig('training.png')

