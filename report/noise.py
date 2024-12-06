import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load the data from the JSON file
with open('report/noise.json', 'r') as f:
    data = json.load(f)

# Extract noise levels and metrics for flronet-fno and flronet-unet
noise_levels = [float(k) for k in data['flronet-fno'].keys()]
rmse_flronetfno = [v['RMSE'] for v in data['flronet-fno'].values()]
mae_flronetfno = [v['MAE'] for v in data['flronet-fno'].values()]
rmse_flronetmlp = [v['RMSE'] for v in data['flronet-mlp'].values()]
mae_flronetmlp = [v['MAE'] for v in data['flronet-mlp'].values()]
rmse_flronetunet = [v['RMSE'] for v in data['flronet-unet'].values()]
mae_flronetunet = [v['MAE'] for v in data['flronet-unet'].values()]
rmse_fno3d = [v['RMSE'] for v in data['fno3d'].values()]
mae_fno3d = [v['MAE'] for v in data['fno3d'].values()]

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
marker_positions = [0., 0.05, 0.1, 0.15, 0.2]
marker_indices = [list(noise_levels).index(pos) for pos in marker_positions]  # Find indices

axs[0].plot(noise_levels, mae_flronetfno, color='tab:blue', label='flronet-fno', linewidth=3)
axs[0].plot(marker_positions, [mae_flronetfno[i] for i in marker_indices], marker='s', color='tab:blue', linestyle='None', markersize=6)
axs[0].plot(noise_levels, mae_flronetmlp, color='tab:green', label='flronet-mlp', linewidth=3)
axs[0].plot(marker_positions, [mae_flronetmlp[i] for i in marker_indices], marker='s', color='tab:green', linestyle='None', markersize=6)
axs[0].plot(noise_levels, mae_flronetunet, color='tab:red', label='flronet-unet', linewidth=3)
axs[0].plot(marker_positions, [mae_flronetunet[i] for i in marker_indices], marker='s', color='tab:red', linestyle='None', markersize=6)
axs[0].plot(noise_levels, mae_fno3d, color='tab:brown', label='fno3d', linewidth=3)
axs[0].plot(marker_positions, [mae_fno3d[i] for i in marker_indices], marker='s', color='tab:brown', linestyle='None', markersize=6)
axs[0].set_xlabel(r'$\epsilon$', fontsize=20)
axs[0].set_ylabel('MAE [m/s]', fontsize=18)
axs[0].tick_params(axis='both', labelsize=16)
filtered_levels = [level for level in noise_levels if (level * 100) % 5 == 0]
axs[0].set_xticks(filtered_levels)
axs[0].set_xticklabels([f"{int(level * 100)}%" for level in filtered_levels])
axs[0].legend(handles=legend_elements, fontsize=16, frameon=False)
axs[0].set_ylim(0, 0.5)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# RMSE subplot
axs[1].plot(noise_levels, rmse_flronetfno, color='tab:blue', label='flronet-fno', linewidth=3)
axs[1].plot(marker_positions, [rmse_flronetfno[i] for i in marker_indices], marker='s', color='tab:blue', linestyle='None', markersize=6)
axs[1].plot(noise_levels, rmse_flronetmlp, color='tab:green', label='flronet-mlp', linewidth=3)
axs[1].plot(marker_positions, [rmse_flronetmlp[i] for i in marker_indices], marker='s', color='tab:green', linestyle='None', markersize=6)
axs[1].plot(noise_levels, rmse_flronetunet, color='tab:red', label='flronet-unet', linewidth=3)
axs[1].plot(marker_positions, [rmse_flronetunet[i] for i in marker_indices], marker='s', color='tab:red', linestyle='None', markersize=6)
axs[1].plot(noise_levels, rmse_fno3d, color='tab:brown', label='fno3d', linewidth=3)
axs[1].plot(marker_positions, [rmse_fno3d[i] for i in marker_indices], marker='s', color='tab:brown', linestyle='None', markersize=6)
axs[1].set_xlabel(r'$\epsilon$', fontsize=20)
axs[1].set_ylabel('RMSE [m/s]', fontsize=18)
axs[1].tick_params(axis='both', labelsize=16)
filtered_levels = [level for level in noise_levels if (level * 100) % 5 == 0]
axs[1].set_xticks(filtered_levels)
axs[1].set_xticklabels([f"{int(level * 100)}%" for level in filtered_levels])
axs[1].legend(handles=legend_elements, fontsize=16, frameon=False)
axs[1].set_ylim(0, 0.7)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Adjust layout and show
plt.tight_layout()
plt.savefig('report/noise.png', dpi=300)
plt.show()