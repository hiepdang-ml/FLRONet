import json
import matplotlib.pyplot as plt

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

# MAE subplot
axs[0].plot(steps, mae_flronetfno, label='flronet-fno', linestyle='-', color='tab:blue', linewidth=4)
axs[0].plot(steps, mae_flronetmlp, label='flronet-mlp', linestyle='-', color='tab:green', linewidth=4)
axs[0].plot(steps, mae_flronetunet, label='flronet-unet', linestyle='-', color='tab:red', linewidth=4)
axs[0].plot(steps, mae_fno3d, label='fno3d', linestyle='-', color='tab:brown', linewidth=4)
axs[0].set_xlabel(r'$d$', fontsize=18)
axs[0].set_ylabel('MAE [m/s]', fontsize=18)
axs[0].tick_params(axis='both', labelsize=16)
axs[0].set_ylim(0, None)
axs[0].legend(fontsize=16, frameon=False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# RMSE subplot
axs[1].plot(steps, rmse_flronetfno, label='flronet-fno', linestyle='-', color='tab:blue', linewidth=4)
axs[1].plot(steps, rmse_flronetmlp, label='flronet-mlp', linestyle='-', color='tab:green', linewidth=4)
axs[1].plot(steps, rmse_flronetunet, label='flronet-unet', linestyle='-', color='tab:red', linewidth=4)
axs[1].plot(steps, rmse_fno3d, label='fno3d', linestyle='-', color='tab:brown', linewidth=4)
axs[1].set_xlabel(r'$d$', fontsize=18)
axs[1].set_ylabel('RMSE [m/s]', fontsize=18)
axs[1].tick_params(axis='both', labelsize=16)
axs[1].set_ylim(0, None)
axs[1].legend(fontsize=16, frameon=False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Adjust layout and save
plt.tight_layout(pad=1)
plt.savefig('report/dropout.png', dpi=300)