import json
import matplotlib.pyplot as plt

# Load the data from the JSON file
with open('report/noise.json', 'r') as f:
    data = json.load(f)

# Extract noise levels and metrics for FLRONet and UNet
noise_levels = [float(k) for k in data['FLRONet'].keys()]
rmse_flronet = [v['RMSE'] for v in data['FLRONet'].values()]
mae_flronet = [v['MAE'] for v in data['FLRONet'].values()]
rmse_unet = [v['RMSE'] for v in data['UNet'].values()]
mae_unet = [v['MAE'] for v in data['UNet'].values()]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)

# MAE subplot
axs[0].plot(noise_levels, mae_flronet, marker='o', color='royalblue', label='FLRONet')
axs[0].plot(noise_levels, mae_unet, marker='o', color='firebrick', linestyle='--', label='UNet')
axs[0].set_xlabel(r'$\epsilon$', fontsize=14)
axs[0].set_ylabel('MAE [m/s]', fontsize=14)
axs[0].tick_params(axis='both', labelsize=12)
axs[0].set_xticks(noise_levels)
axs[0].set_xticklabels([f"{int(level * 100)}%" for level in noise_levels])
axs[0].legend(fontsize=12)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# RMSE subplot
axs[1].plot(noise_levels, rmse_flronet, marker='o', color='royalblue', label='FLRONet')
axs[1].plot(noise_levels, rmse_unet, marker='o', color='firebrick', linestyle='--', label='UNet')
axs[1].set_xlabel(r'$\epsilon$', fontsize=14)
axs[1].set_ylabel('RMSE [m/s]', fontsize=14)
axs[1].tick_params(axis='both', labelsize=12)
axs[1].set_xticks(noise_levels)
axs[1].set_xticklabels([f"{int(level * 100)}%" for level in noise_levels])
axs[1].legend(fontsize=12)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Adjust layout and show
plt.tight_layout()
plt.savefig('report/noise.png', dpi=300)
plt.show()