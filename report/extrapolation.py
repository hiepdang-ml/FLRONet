import json
import matplotlib.pyplot as plt

# Load data from JSON file
with open('report/extrapolate.json', 'r') as f:
    data = json.load(f)

# Extract extrapolation data
extrapolation_indices = list(data["extrapolation"].keys())
extrapolation_mae = [data["extrapolation"][key]["MAE"] for key in extrapolation_indices]
extrapolation_rmse = [data["extrapolation"][key]["RMSE"] for key in extrapolation_indices]

# Interpolation data
interpolation_mae = data["interpolation"]["MAE"]
interpolation_rmse = data["interpolation"]["RMSE"]

# Plotting MAE and RMSE in horizontal subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# RMSE subplot
ax1.plot(extrapolation_indices, extrapolation_rmse, color='b', marker='s', linestyle='--', label='Extrapolation RMSE')
ax1.axhline(y=interpolation_rmse, color='r', linestyle='--')
ax1.text(2, interpolation_rmse, f'Avg. Interpolation RMSE = {interpolation_rmse}', color='r', ha='center', va='bottom')
ax1.set_xlabel('Prediction Step', fontsize=13)
ax1.set_ylabel('RMSE', fontsize=13)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(fontsize=12)

# MAE subplot
ax2.plot(extrapolation_indices, extrapolation_mae, color='b', marker='s', linestyle='--', label='Extrapolation MAE')
ax2.axhline(y=interpolation_mae, color='r', linestyle='--')
ax2.text(2, interpolation_mae, f'Avg. Interpolation MAE = {interpolation_mae}', color='r', ha='center', va='bottom')
ax2.set_xlabel('Prediction Step', fontsize=13)
ax2.set_ylabel('MAE', fontsize=13)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig('time_extrapolation.png')