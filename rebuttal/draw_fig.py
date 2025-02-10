import matplotlib.pyplot as plt
import numpy as np

# Define methods and scores
methods = ["Textual Inversion", "DreamBooth", "Elite", "BLIP-Diff.", "IP-Adapter",
           "SSR-Encoder", "BootPIG", "Subject Diffusion", "Ours", "Ours-SDXL"]
clip_t_scores = [0.261, 0.306, 0.296, 0.298, 0.274, 0.308, 0.311, 0.293, 0.316, 0.323]
dino_scores = [0.561, 0.672, 0.647, 0.589, 0.608, 0.612, 0.674, 0.711, 0.718, 0.722]

# Define unique colors for each method
colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

# Create a more aesthetically pleasing plot with better aspect ratio and fewer ticks
plt.figure(figsize=(3, 2))  # Make it slightly wider
plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
# Plot each method with a different color
for i, method in enumerate(methods):
    if "Ours" in method:
        plt.scatter(clip_t_scores[i], dino_scores[i], zorder=2, s=75, c='red', marker="*")
    elif method in ["Textual Inversion", "DreamBooth"]:
        plt.scatter(clip_t_scores[i], dino_scores[i], color=colors[0], s=10, zorder=2)
    else:
        plt.scatter(clip_t_scores[i], dino_scores[i], color=colors[1], s=10, zorder=2)

# Annotate each point with method names in black
for i, method in enumerate(methods):
    if "Ours" in method:
        plt.text(clip_t_scores[i] - 0.001, dino_scores[i],  # Move text to the left
                 method, fontsize=5, ha='right', weight='bold', color='black')
    elif method == "DreamBooth":
        plt.text(clip_t_scores[i] - 0.001, dino_scores[i],  # Move text to the left
                 method, fontsize=6, ha='right', color='black')
    else:
        plt.text(clip_t_scores[i] + 0.001, dino_scores[i] - 0.002, 
                 method, fontsize=6, ha='left', color='black')

# Set labels and title
plt.xlabel("CLIP-T Score", fontsize=5, labelpad=-5)
plt.ylabel("DINO Score", fontsize=5, labelpad=-5)

# Adjust tick marks for clarity
import matplotlib.pyplot as plt
import numpy as np

# 让 X 轴和 Y 轴刻度减少一个
x_ticks = np.linspace(0.26, 0.33, num=2)  # 生成 3 个刻度
y_ticks = np.linspace(0.55, 0.73, num=2)  # 生成 4 个刻度

plt.xticks(x_ticks, labels=[f"{tick:.3f}" for tick in x_ticks], fontsize=4)  # 保持小数位对齐
plt.yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks], fontsize=4)  # 统一格式

# Set axis limits for better spacing
plt.xlim(0.26, 0.33)
plt.ylim(0.55, 0.73)
plt.grid(True, linestyle="--", alpha=0.6)

# Create legend inside the top-left corner
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=4, label="Tuning based"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=4, label="Tuning free"),
]

plt.legend(handles=legend_elements, loc='upper left', fontsize=5, frameon=True)

# Reduce white margins
plt.tight_layout(pad=0.05)

# Show the plot
plt.savefig("compare_dino_clipt.png", bbox_inches='tight', dpi=300)