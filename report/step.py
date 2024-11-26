import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the JSON data
with open('report/step.json', 'r') as f:
    data = json.load(f)

# Prepare the data for plotting
index_list = []
metric_list = []
value_list = []

for key, values in data.items():
    for mae_value in values['mae']:
        index_list.append(key)
        metric_list.append('MAE')
        value_list.append(mae_value)
    for rmse_value in values['rmse']:
        index_list.append(key)
        metric_list.append('RMSE')
        value_list.append(rmse_value)

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    "Step": index_list,
    "Metric": metric_list,
    "Value": value_list
})

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Initialize the figure
fig, axs = plt.subplots(1, 2, figsize=(12, 3))

# MAE Catplot
sns.boxplot(
    data=plot_data[plot_data["Metric"]=="MAE"],
    x="Step", y="Value", ax=axs[0],
    showfliers=False,
)
axs[0].set_xlabel("Step", fontsize=18)
axs[0].set_ylabel("MAE [m/s]", fontsize=18)
axs[0].tick_params(axis='x', labelsize=12)
axs[0].tick_params(axis='y', labelsize=12)
axs[0].grid(False)

for i, patch in enumerate(axs[0].patches):
    if i in {0, 5, 10, 15, 20}:
        patch.set_facecolor('red')
    else:
        patch.set_facecolor('dodgerblue')
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

medians = axs[0].lines[4::5]
for median in medians:
    median.set_color('black')
    median.set_linewidth(2)

# RMSE Catplot
sns.boxplot(
    data=plot_data[plot_data["Metric"]=="RMSE"],
    x="Step", y="Value", ax=axs[1],
    showfliers=False,
)
axs[1].set_xlabel("Step", fontsize=18)
axs[1].set_ylabel("RMSE [m/s]", fontsize=18)
axs[1].tick_params(axis='x', labelsize=12)
axs[1].tick_params(axis='y', labelsize=12)
axs[1].grid(False)

for i, patch in enumerate(axs[1].patches):
    if i in {0, 5, 10, 15, 20}:
        patch.set_facecolor('red')
    else:
        patch.set_facecolor('dodgerblue')
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

medians = axs[1].lines[4::5]
for median in medians:
    median.set_color('black')
    median.set_linewidth(2)

# Adjust layout and show
plt.tight_layout()
plt.savefig('report/step.png', dpi=300)

