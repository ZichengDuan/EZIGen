import matplotlib.pyplot as plt
import numpy as np

# Define methods and scores
methods = ["DreamBooth", "Elite", "BLIP-Diff.", "FastComposer",
           "SSR-Encoder", "Subject Diffusion", "Ours", "PhotoMaker"]
prompt_consistency = [0.239, 0.217, 0.237, 0.243, 0.235, 0.228, 0.237, 0.279]
id_preservation = [0.273, 0.231, 0.227, 0.514, 0.391, 0.605, 0.598, 0.495]

# Define unique colors for each method
colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

# Create a more aesthetically pleasing plot with better aspect ratio and fewer ticks
plt.figure(figsize=(3, 2))  # Make it slightly wider
plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
# Plot each method with a different color
for i, method in enumerate(methods):
    # if "Ours" in method:
    #     plt.scatter(id_preservation[i], prompt_consistency[i], zorder=2, s=75, c='red', marker="*")
    if method in ["FastComposer", "Subject Diffusion", "PhotoMaker"]:
        plt.scatter(id_preservation[i], prompt_consistency[i], color=colors[2], s=10, zorder=2)
    else:
        plt.scatter(id_preservation[i], prompt_consistency[i], color=colors[3], s=10, zorder=2)

# Annotate each point with method names in black
for i, method in enumerate(methods):
    if "Ours" in method:
        plt.text(id_preservation[i] - 0.001, prompt_consistency[i],  # Move text to the left
                 method, fontsize=5, ha='right', weight='bold', color='black')
    else:
        plt.text(id_preservation[i] + 0.001, prompt_consistency[i] - 0.002, 
                 method, fontsize=6, ha='left', color='black')

# Set labels and title
plt.ylabel("Prompt Consistency", fontsize=5, labelpad=-5)
plt.xlabel("ID Preservation", fontsize=5, labelpad=-5)

# Adjust tick marks for clarity
import matplotlib.pyplot as plt
import numpy as np

# 让 X 轴和 Y 轴刻度减少一个
y_ticks = np.linspace(0.21, 0.28, num=2)  # 生成 3 个刻度
x_ticks = np.linspace(0.22, 0.61, num=2)  # 生成 4 个刻度

plt.yticks(x_ticks, labels=[f"{tick:.3f}" for tick in x_ticks], fontsize=4)  # 保持小数位对齐
plt.xticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks], fontsize=4)  # 统一格式

# Set axis limits for better spacing
plt.ylim(0.21, 0.28)
plt.xlim(0.22, 0.61)
plt.grid(True, linestyle="--", alpha=0.6)

# Create legend inside the top-left corner
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], markersize=4, label="W/ in-domain data"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[3], markersize=4, label="W/o in-domain data"),
]

plt.legend(handles=legend_elements, loc='upper left', fontsize=5, frameon=True)

# Reduce white margins
plt.tight_layout(pad=0.05)

# Show the plot
plt.savefig("human_res.png", bbox_inches='tight', dpi=300)