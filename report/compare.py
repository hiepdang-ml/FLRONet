import json
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'

# Load the data from the JSON file
with open('report/compare.json', 'r') as f:
    data = json.load(f)

# Extract values for plotting
v_flronetfno = [case['v'] for case in data['flronet-fno'].values()]
mae_flronetfno = [case['MAE'] for case in data['flronet-fno'].values()]
rmse_flronetfno = [case['RMSE'] for case in data['flronet-fno'].values()]

v_flronetmlp = [case['v'] for case in data['flronet-mlp'].values()]
mae_flronetmlp = [case['MAE'] for case in data['flronet-mlp'].values()]
rmse_flronetmlp = [case['RMSE'] for case in data['flronet-mlp'].values()]

v_flronetunet = [case['v'] for case in data['flronet-unet'].values()]
mae_flronetunet = [case['MAE'] for case in data['flronet-unet'].values()]
rmse_flronetunet = [case['RMSE'] for case in data['flronet-unet'].values()]

v_fno3d = [case['v'] for case in data['fno3d'].values()]
mae_fno3d = [case['MAE'] for case in data['fno3d'].values()]
rmse_fno3d = [case['RMSE'] for case in data['fno3d'].values()]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# MAE subplot
axs[0].plot(v_flronetfno, mae_flronetfno, label='flronet-fno', marker='s', color='tab:blue', linestyle='-', linewidth=2)
axs[0].plot(v_flronetmlp, mae_flronetmlp, label='flronet-mlp', marker='s', color='tab:green', linestyle='-', linewidth=2)
axs[0].plot(v_flronetunet, mae_flronetunet, label='flronet-unet', marker='s', color='tab:red', linestyle='-', linewidth=2)
axs[0].plot(v_fno3d, mae_fno3d, label='fno3d', marker='s', color='tab:brown', linestyle='-', linewidth=2)
axs[0].set_xticks(v_flronetfno)
axs[0].set_xticklabels([f"{v:.1f}" for v in v_flronetfno], fontsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[0].set_xlabel(r'$v_0$ [m/s]', fontsize=16)
axs[0].set_ylabel('MAE [m/s]', fontsize=18)
axs[0].set_ylim(0, 0.15)
axs[0].legend(fontsize=14, frameon=False, loc='upper left')
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_linewidth(1.3)
axs[0].spines['bottom'].set_linewidth(1.3)
axs[0].tick_params(axis='both', which='major', length=6, width=1.3)

# RMSE subplot
axs[1].plot(v_flronetfno, rmse_flronetfno, label='flronet-fno', marker='s', color='tab:blue', linestyle='-', linewidth=2)
axs[1].plot(v_flronetmlp, rmse_flronetmlp, label='flronet-mlp', marker='s', color='tab:green', linestyle='-', linewidth=2)
axs[1].plot(v_flronetunet, rmse_flronetunet, label='flronet-unet', marker='s', color='tab:red', linestyle='-', linewidth=2)
axs[1].plot(v_fno3d, rmse_fno3d, label='fno3d', marker='s', color='tab:brown', linestyle='-', linewidth=2)
axs[1].set_xticks(v_flronetfno)
axs[1].set_xticklabels([f"{v:.1f}" for v in v_flronetfno], fontsize=14)
axs[1].tick_params(axis='y', labelsize=14)
axs[1].set_xlabel(r'$v_0$ [m/s]', fontsize=16)
axs[1].set_ylabel('RMSE [m/s]', fontsize=18)
axs[1].set_ylim(0, 0.25)
axs[1].legend(fontsize=14, frameon=False, loc='upper left')
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_linewidth(1.3)
axs[1].spines['bottom'].set_linewidth(1.3)
axs[1].tick_params(axis='both', which='major', length=6, width=1.3)

# Adjust layout and show
plt.tight_layout(pad=0.5)
plt.subplots_adjust(wspace=0.25)  # Increase space between subplots
plt.savefig('report/compare.png', dpi=300)


