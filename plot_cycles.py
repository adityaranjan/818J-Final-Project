import matplotlib.pyplot as plt
import numpy as np


# Data for the first plot
optimized_phases_1 = [32, 74224, 31924]
baseline_phases_1 = [111378, 0, 750330]
labels_1 = ["Agg Only", "Overlapped", "Combination Only"]
title_1 = "Real - 1433, 7"

# Data for the second plot
optimized_phases_2 = [28, 53040, 293604]
baseline_phases_2 = [79584, 0, 617952]
labels_2 = ["Agg Only", "Overlapped", "Combination Only"]
title_2 = "Synthetic - 1024, 32"


# Adjusted function for proper stacked bar comparison
def plot_stacked_bar_vertical(optimized, baseline, labels, title, filename):
    bar_width = 0.35
    x = np.array([0, 1])  # Two bars for Optimized and Baseline

    # Add zero for Overlapped in Baseline if needed
    if len(baseline) < len(labels):
        baseline = baseline[:1] + [0] + baseline[1:]

    # Split the data for stacking
    optimized_bars = np.array(optimized)
    baseline_bars = np.array(baseline)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Stacked bar for Optimized
    bar_optimized_1 = ax.bar(
        x[0], optimized_bars[0], bar_width, label=labels[0], color="mediumseagreen"
    )
    bar_optimized_2 = ax.bar(
        x[0],
        optimized_bars[1],
        bar_width,
        bottom=optimized_bars[0],
        label=labels[1],
        color="skyblue",
    )
    bar_optimized_3 = ax.bar(
        x[0],
        optimized_bars[2],
        bar_width,
        bottom=optimized_bars[0] + optimized_bars[1],
        label=labels[2],
        color="orange",
    )

    # Stacked bar for Baseline
    bar_baseline_1 = ax.bar(x[1], baseline_bars[0], bar_width, color="mediumseagreen")
    bar_baseline_2 = ax.bar(
        x[1], baseline_bars[1], bar_width, bottom=baseline_bars[0], color="skyblue"
    )
    bar_baseline_3 = ax.bar(
        x[1],
        baseline_bars[2],
        bar_width,
        bottom=baseline_bars[0] + baseline_bars[1],
        color="orange",
    )

    # Add the sum of each stacked bar on top
    sum_optimized = np.sum(optimized_bars)
    sum_baseline = np.sum(baseline_bars)

    ax.text(
        x[0],
        sum_optimized + 0.02 * sum_optimized,
        f"{int(sum_optimized)}",
        ha="center",
        fontsize=14,
    )
    ax.text(
        x[1],
        sum_baseline + 0.005 * sum_baseline,
        f"{int(sum_baseline)}",
        ha="center",
        fontsize=14,
    )

    ax.set_ylabel("Cycles", fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(["Optimized", "Baseline"], fontsize=16)
    ax.legend(labels, fontsize=16)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# Generate updated plots
plot_stacked_bar_vertical(
    optimized_phases_1,
    baseline_phases_1,
    labels_1,
    title_1,
    "figs/real_1433_7_stacked.pdf",
)

plot_stacked_bar_vertical(
    optimized_phases_2,
    baseline_phases_2,
    labels_2,
    title_2,
    "figs/synthetic_10024_32_stacked.pdf",
)
