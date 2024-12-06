import matplotlib.pyplot as plt

scales = ['1x', '2x', '4x', '8x']
flronet_v100 = [0.14, 0.17, 0.25, 0.47]
flronet_a100 = [0.09, 0.11, 0.16, 0.3]

fno3d_v100 = [0.17, 0.2, 0.28, 0.51]
fno3d_a100 = [0.11, 0.13, 0.17, 0.32]

positions = [float(x.replace('x', '')) for x in scales]

fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))

# V100 subplot
axs[0].plot(positions, flronet_v100, marker='s', linestyle='-', color='tab:blue', label="flronet-fno", linewidth=2, markersize=6)
axs[0].plot(positions, fno3d_v100, marker='s', linestyle='-', color='tab:red', label="fno3d", linewidth=2, markersize=6)
axs[0].set_title("Tesla V100", fontsize=16)
axs[0].set_xlabel("Scale", fontsize=14, labelpad=10)
axs[0].set_ylabel("Inference Time [s]", fontsize=16, labelpad=10)
axs[0].set_xticks(positions)
axs[0].set_xticklabels(scales, fontsize=14)
axs[0].tick_params(axis='both', which='major', length=6, width=1.3, labelsize=14)
axs[0].legend(fontsize=14, frameon=False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_linewidth(1.3)
axs[0].spines['bottom'].set_linewidth(1.3)

# A100 subplot
axs[1].plot(positions, flronet_a100, marker='s', linestyle='-', color='tab:blue', label="flronet-fno", linewidth=2, markersize=6)
axs[1].plot(positions, fno3d_a100, marker='s', linestyle='-', color='tab:red', label="fno3d", linewidth=2, markersize=6)
axs[1].set_title("A100 SXM4", fontsize=16)
axs[1].set_xlabel("Scale", fontsize=14, labelpad=10)
axs[1].set_xticks(positions)
axs[1].set_xticklabels(scales, fontsize=14)
axs[1].tick_params(axis='both', which='major', length=6, width=1.3, labelsize=14)
axs[1].legend(fontsize=14, frameon=False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_linewidth(1.3)
axs[1].spines['bottom'].set_linewidth(1.3)

plt.tight_layout()
plt.savefig('report/super_res_space.png', dpi=300)