import numpy as np
import matplotlib.pyplot as plt


# Function to create grouped bar charts for multiple comparisons
def plot_grouped_bar_comparisons(data, labels, title, filename):
    x = np.arange(len(labels))  # Number of groups (Min, Avg, Max)
    bar_width = 0.35

    optimized_values = [item[0] for item in data]
    baseline_values = [item[1] for item in data]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars_optimized = ax.bar(
        x - bar_width / 2,
        optimized_values,
        bar_width,
        label="Optimized",
        color="skyblue",
    )
    bars_baseline = ax.bar(
        x + bar_width / 2,
        baseline_values,
        bar_width,
        label="Baseline",
        color="orange",
    )

    # Add numbers on top of each bar
    for bar in bars_optimized:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01 * yval,
            f"{int(yval)}",
            ha="center",
            fontsize=14,
        )

    for bar in bars_baseline:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01 * yval,
            f"{int(yval)}",
            ha="center",
            fontsize=14,
        )

    ax.set_xlabel("Metrics", fontsize=16)
    ax.set_ylabel("Cycles", fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    ax.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# Data for the first plot
data_1 = [(104, 111648), (15066, 428117), (31960, 747462)]
labels_1 = ["Min", "Avg", "Max"]
title_1 = "Real - 1433, 7"

# Data for the second plot
data_2 = [(172, 79776), (146211, 306956), (293632, 536208)]
labels_2 = ["Min", "Avg", "Max"]
title_2 = "Synthetic - 1024, 32"

# Generate the plots
plot_grouped_bar_comparisons(data_1, labels_1, title_1, "figs/real_1433_7_grouped.pdf")
plot_grouped_bar_comparisons(
    data_2, labels_2, title_2, "figs/synthetic_1024_32_grouped.pdf"
)
