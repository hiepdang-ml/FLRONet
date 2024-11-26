import json
import matplotlib.pyplot as plt

# Load the data from the JSON file
with open('report/dropout.json', 'r') as f:
    data = json.load(f)

# Extract values for plotting
indices = list(range(len(data['FLRONet'])))
rmse_flronet = [data['FLRONet'][str(i)]['RMSE'] for i in indices]
mae_flronet = [data['FLRONet'][str(i)]['MAE'] for i in indices]
rmse_unet = [data['UNet'][str(i)]['RMSE'] for i in indices]
mae_unet = [data['UNet'][str(i)]['MAE'] for i in indices]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)

# MAE subplot
axs[0].plot(indices, mae_flronet, label='FLRONet', marker='o', linestyle='-', color='royalblue', linewidth=1.7, markersize=4.5)
axs[0].plot(indices, mae_unet, label='UNet', marker='o', linestyle='--', color='firebrick', linewidth=1.7, markersize=4.5)
axs[0].set_xlabel('#Dropped Sensors', fontsize=16)
axs[0].set_ylabel('MAE [m/s]', fontsize=16)
axs[0].tick_params(axis='both', labelsize=14)
axs[0].legend(fontsize=14, frameon=False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# RMSE subplot
axs[1].plot(indices, rmse_flronet, label='FLRONet', marker='o', linestyle='-', color='royalblue', linewidth=1.7, markersize=4.5)
axs[1].plot(indices, rmse_unet, label='UNet', marker='o', linestyle='--', color='firebrick', linewidth=1.7, markersize=4.5)
axs[1].set_xlabel('#Dropped Sensors', fontsize=16)
axs[1].set_ylabel('RMSE [m/s]', fontsize=16)
axs[1].tick_params(axis='both', labelsize=14)
axs[1].legend(fontsize=14, frameon=False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Adjust layout and save
plt.tight_layout(pad=1)
plt.savefig('report/dropout.png', dpi=300)