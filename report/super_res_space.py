import matplotlib.pyplot as plt

scales = ['1x', '2x', '4x', '8x']
tesla_v100 = [0.29, 0.35, 0.5, 0.95]
a100 = [0.17, 0.22, 0.33, 0.6]

# Create the figure
plt.figure(figsize=(7, 3.5))

# Convert scales to positions
positions = [float(x.replace('x','')) for x in scales]

# Plotting the data
plt.plot(positions, tesla_v100, marker='s', linestyle='--', color='firebrick', label="Tesla V100", linewidth=2.5, markersize=8)
plt.plot(positions, a100, marker='s', linestyle='--', color='royalblue', label="A100 SXM4", linewidth=2.5, markersize=8)

# Add labels and ticks
plt.xlabel("Scale", fontsize=18, labelpad=10)
plt.ylabel("Inference Time [s]", fontsize=18, labelpad=10)
plt.xticks(positions, scales, fontsize=16)  # Map positions to scales
plt.yticks(fontsize=16)

# Add legend
plt.legend(fontsize=16, frameon=False)

# Customize spines
ax = plt.gca()  # Get the current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.3)
ax.spines['bottom'].set_linewidth(1.3)
ax.tick_params(axis='both', which='major', length=6, width=1.3)

# Tight layout and save
plt.tight_layout(pad=0.5)
plt.savefig('report/super_res_space.png', dpi=300)
