import json
import matplotlib.pyplot as plt

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

# MAE subplot
axs[0].plot(noise_levels, mae_flronetfno, color='tab:blue', label='flronet-fno', linewidth=4)
axs[0].plot(noise_levels, mae_flronetmlp, color='tab:green', label='flronet-mlp', linewidth=4)
axs[0].plot(noise_levels, mae_flronetunet, color='tab:red', label='flronet-unet', linewidth=4)
axs[0].plot(noise_levels, mae_fno3d, color='tab:brown', label='fno3d', linewidth=4)
axs[0].set_xlabel(r'$\epsilon$', fontsize=18)
axs[0].set_ylabel('MAE [m/s]', fontsize=18)
axs[0].tick_params(axis='both', labelsize=16)
axs[0].set_xticks(noise_levels)
axs[0].set_xticklabels([f"{int(level * 100)}%" for level in noise_levels])
axs[0].legend(fontsize=16, frameon=False)
axs[0].set_ylim(0, 0.3)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# RMSE subplot
axs[1].plot(noise_levels, rmse_flronetfno, color='tab:blue', label='flronet-fno', linewidth=4)
axs[1].plot(noise_levels, rmse_flronetmlp, color='tab:green', label='flronet-mlp', linewidth=4)
axs[1].plot(noise_levels, rmse_flronetunet, color='tab:red', label='flronet-unet', linewidth=4)
axs[1].plot(noise_levels, rmse_fno3d, color='tab:brown', label='fno3d', linewidth=4)
axs[1].set_xlabel(r'$\epsilon$', fontsize=18)
axs[1].set_ylabel('RMSE [m/s]', fontsize=18)
axs[1].tick_params(axis='both', labelsize=16)
axs[1].set_xticks(noise_levels)
axs[1].set_xticklabels([f"{int(level * 100)}%" for level in noise_levels])
axs[1].legend(fontsize=16, frameon=False)
axs[1].set_ylim(0, 0.4)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Adjust layout and show
plt.tight_layout()
plt.savefig('report/noise.png', dpi=300)
plt.show()