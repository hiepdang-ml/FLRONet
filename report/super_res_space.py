import matplotlib.pyplot as plt

# Data for plotting
scales = ['1x', '2x', '4x']
tesla_v100 = [0.57, 0.65, 0.84]
a100 = [0.33, 0.42, 0.55]

plt.figure(figsize=(10, 6))
plt.plot(scales, tesla_v100, marker='x', linestyle='--', color='green', label="Tesla V100", linewidth=2, markersize=8)
plt.plot(scales, a100, marker='x', linestyle='--', color='blue', label="A100 SMX4", linewidth=2, markersize=8)

plt.xlabel("Scale", fontsize=18)
plt.ylabel("Inference Time (s)", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)

plt.tight_layout()
plt.savefig('super_res_time.png')