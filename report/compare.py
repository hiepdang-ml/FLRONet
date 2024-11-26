import json
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'

# Load the data from the JSON file
with open('report/compare.json', 'r') as f:
    data = json.load(f)

# Extract values for plotting
v_flronet = [case['v'] for case in data['flronet'].values()]
mae_flronet = [case['MAE'] for case in data['flronet'].values()]
rmse_flronet = [case['RMSE'] for case in data['flronet'].values()]

v_unet = [case['v'] for case in data['unet'].values()]
mae_unet = [case['MAE'] for case in data['unet'].values()]
rmse_unet = [case['RMSE'] for case in data['unet'].values()]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))

# MAE subplot
axs[0].plot(v_flronet, mae_flronet, label='FLRONet', marker='s', color='royalblue', linestyle='-')
axs[0].plot(v_unet, mae_unet, label='UNet', marker='s', color='firebrick', linestyle='--')
axs[0].set_xticks(v_flronet)
axs[0].set_xticklabels([f"{v:.1f}" for v in v_flronet], fontsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[0].set_xlabel(r'$v_0$ [m/s]', fontsize=16)
axs[0].set_ylabel('MAE [m/s]', fontsize=18)
axs[0].legend(fontsize=14, frameon=False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_linewidth(1.3)
axs[0].spines['bottom'].set_linewidth(1.3)
axs[0].tick_params(axis='both', which='major', length=6, width=1.3)

# RMSE subplot
axs[1].plot(v_flronet, rmse_flronet, label='FLRONet', marker='s', color='royalblue', linestyle='-')
axs[1].plot(v_unet, rmse_unet, label='UNet', marker='s', color='firebrick', linestyle='--')
axs[1].set_xticks(v_flronet)
axs[1].set_xticklabels([f"{v:.1f}" for v in v_flronet], fontsize=14)
axs[1].tick_params(axis='y', labelsize=14)
axs[1].set_xlabel(r'$v_0$ [m/s]', fontsize=16)
axs[1].set_ylabel('RMSE [m/s]', fontsize=18)
axs[1].legend(fontsize=14, frameon=False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_linewidth(1.3)
axs[1].spines['bottom'].set_linewidth(1.3)
axs[1].tick_params(axis='both', which='major', length=6, width=1.3)

# Adjust layout and show
plt.tight_layout(pad=0.5)
plt.subplots_adjust(wspace=0.25)  # Increase space between subplots
plt.savefig('report/compare.png', dpi=300)


