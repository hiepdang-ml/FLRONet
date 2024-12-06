import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load the data from the JSON file
with open('report/dropout.json', 'r') as f:
    data = json.load(f)

# Extract values for plotting
steps = list(range(len(data['flronet-fno'])))
rmse_flronetfno = [data['flronet-fno'][str(i)]['RMSE'] for i in steps]
mae_flronetfno = [data['flronet-fno'][str(i)]['MAE'] for i in steps]
rmse_flronetunet = [data['flronet-unet'][str(i)]['RMSE'] for i in steps]
mae_flronetunet = [data['flronet-unet'][str(i)]['MAE'] for i in steps]
rmse_flronetmlp = [data['flronet-mlp'][str(i)]['RMSE'] for i in steps]
mae_flronetmlp = [data['flronet-mlp'][str(i)]['MAE'] for i in steps]
rmse_fno3d = [data['fno3d'][str(i)]['RMSE'] for i in steps]
mae_fno3d = [data['fno3d'][str(i)]['MAE'] for i in steps]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

# Custom legends
legend_elements = [
    Line2D([0], [0], color='tab:blue', marker='s', linestyle='-', linewidth=3, markersize=6, label='flronet-fno'),
    Line2D([0], [0], color='tab:green', marker='s', linestyle='-', linewidth=3, markersize=6, label='flronet-mlp'),
    Line2D([0], [0], color='tab:red', marker='s', linestyle='-', linewidth=3, markersize=6, label='flronet-unet'),
    Line2D([0], [0], color='tab:brown', marker='s', linestyle='-', linewidth=3, markersize=6, label='fno3d')
]

# MAE subplot
marker_positions = [0, 5, 10, 15, 20]
axs[0].plot(steps, mae_flronetfno, label='flronet-fno', linestyle='-', color='tab:blue', linewidth=3)
axs[0].plot(marker_positions, [mae_flronetfno[i] for i in marker_positions], marker='s', color='tab:blue', linestyle='None', markersize=6)
axs[0].plot(steps, mae_flronetmlp, label='flronet-mlp', linestyle='-', color='tab:green', linewidth=3)
axs[0].plot(marker_positions, [mae_flronetmlp[i] for i in marker_positions], marker='s', color='tab:green', linestyle='None', markersize=6)
axs[0].plot(steps, mae_flronetunet, label='flronet-unet', linestyle='-', color='tab:red', linewidth=3)
axs[0].plot(marker_positions, [mae_flronetunet[i] for i in marker_positions], marker='s', color='tab:red', linestyle='None', markersize=6)
axs[0].plot(steps, mae_fno3d, label='fno3d', linestyle='-', color='tab:brown', linewidth=3)
axs[0].plot(marker_positions, [mae_fno3d[i] for i in marker_positions], marker='s', color='tab:brown', linestyle='None', markersize=6)
axs[0].set_xlabel(r'$d$', fontsize=18)
axs[0].set_ylabel('MAE [m/s]', fontsize=18)
axs[0].tick_params(axis='both', labelsize=16)
axs[0].set_ylim(0, None)
axs[0].legend(handles=legend_elements, fontsize=16, frameon=False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# RMSE subplot
axs[1].plot(steps, rmse_flronetfno, label='flronet-fno', linestyle='-', color='tab:blue', linewidth=3)
axs[1].plot(marker_positions, [rmse_flronetfno[i] for i in marker_positions], marker='s', color='tab:blue', linestyle='None', markersize=6)
axs[1].plot(steps, rmse_flronetmlp, label='flronet-mlp', linestyle='-', color='tab:green', linewidth=3)
axs[1].plot(marker_positions, [rmse_flronetmlp[i] for i in marker_positions], marker='s', color='tab:green', linestyle='None', markersize=6)
axs[1].plot(steps, rmse_flronetunet, label='flronet-unet', linestyle='-', color='tab:red', linewidth=3)
axs[1].plot(marker_positions, [rmse_flronetunet[i] for i in marker_positions], marker='s', color='tab:red', linestyle='None', markersize=6)
axs[1].plot(steps, rmse_fno3d, label='fno3d', linestyle='-', color='tab:brown', linewidth=3)
axs[1].plot(marker_positions, [rmse_fno3d[i] for i in marker_positions], marker='s', color='tab:brown', linestyle='None', markersize=6)
axs[1].set_xlabel(r'$d$', fontsize=18)
axs[1].set_ylabel('RMSE [m/s]', fontsize=18)
axs[1].tick_params(axis='both', labelsize=16)
axs[1].set_ylim(0, None)
axs[1].legend(handles=legend_elements, fontsize=16, frameon=False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Adjust layout and save
plt.tight_layout(pad=1)
plt.savefig('report/dropout.png', dpi=300)