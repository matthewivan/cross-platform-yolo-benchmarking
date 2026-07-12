#!/usr/bin/env python3
"""
create_plot.py — all plots, driven entirely by the CONFIG dict below.

Replaces create_plot.py, create_plot_khadas.py and create_plot_radxa_vs_khadas.py.
The two old scatter scripts baked data inline; this reads it from the CSVs
benchmark.py writes, so adding a board = adding one dict entry (no editing arrays).

Adds Jetson AGX Orin. Produces:
  * per-resolution latency boxplots across ALL boards
  * latency-vs-start-temperature scatter (the old overlay), any board subset
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# CONFIG — add a board by adding one entry here. That's the whole change.
# ---------------------------------------------------------------------------
BASE_PATH = "./"
RESOLUTIONS = [256, 320, 640]

BOARDS = {
    "Pi Zero 2W": {
        "color": "green",
        "files": {256: "rpi_benchmark_results_256model_correctms.csv",
                  320: "rpi_benchmark_results_320model.csv",
                  640: "rpi_benchmark_results_640model.csv"},
    },
    "Radxa Zero 3W": {
        "color": "orange",
        "files": {256: "radxa_10_benchmark_results_256model.csv",
                  320: "radxa_10_benchmark_results_320model.csv",
                  640: "radxa_10_benchmark_results_640model.csv"},
    },
    "Khadas Edge 2": {
        "color": "blue",
        "files": {256: "khadas_10_benchmark_256model.csv",
                  320: "khadas_10_benchmark_320model.csv",
                  640: "khadas_10_benchmark_640model.csv"},
    },
    "Jetson AGX Orin": {
        "color": "red",
        "files": {256: "jetson_10_benchmark_256model.csv",
                  320: "jetson_10_benchmark_320model.csv",
                  640: "jetson_10_benchmark_640model.csv"},
    },
}


# ---------------------------------------------------------------------------
def load_csv(board, res):
    """Return the DataFrame for a board+res, or None if the CSV is missing."""
    fname = BOARDS[board]["files"].get(res)
    if not fname:
        return None
    path = os.path.join(BASE_PATH, fname)
    if not os.path.exists(path):
        print(f"[skip] missing {path}")
        return None
    return pd.read_csv(path)


def boxplot_all_boards(res):
    """One boxplot per resolution comparing every board that has data."""
    series, labels, colors = [], [], []
    for board, cfg in BOARDS.items():
        df = load_csv(board, res)
        if df is None or "mean_latency_ms" not in df:
            continue
        series.append(df["mean_latency_ms"])
        labels.append(board)
        colors.append(cfg["color"])
    if not series:
        return

    plt.figure(figsize=(2 + 1.6 * len(series), 6))
    bp = plt.boxplot(series, labels=labels, patch_artist=True,
                     medianprops=dict(color="black"))
    for box, c in zip(bp["boxes"], colors):
        box.set(facecolor=c, alpha=0.7)
    plt.title(f"Inference Latency by Board ({res}×{res})")
    plt.ylabel("Mean Latency (ms)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15)
    out = f"latency_boxplot_{res}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[wrote] {out}")


def scatter_latency_vs_temp(res, boards=None):
    """Latency vs start temperature, colored by CPU load (the old overlay)."""
    boards = boards or list(BOARDS)
    load_colors = {0: "blue", 25: "green", 50: "orange", 75: "red", 100: "purple"}
    markers = ["o", "^", "s", "D", "v", "P"]
    board_marker = {b: markers[i % len(markers)] for i, b in enumerate(boards)}

    plt.figure(figsize=(9, 6))
    plotted = False
    for board in boards:
        df = load_csv(board, res)
        if df is None or "temp_start_C" not in df:
            continue
        for load, c in load_colors.items():
            sub = df[df["cpu_load_percent"] == load]
            if sub.empty:
                continue
            plt.scatter(sub["temp_start_C"], sub["mean_latency_ms"],
                        color=c, marker=board_marker[board], alpha=0.8)
            plotted = True
    if not plotted:
        plt.close()
        return

    plt.xlabel("Start Temperature (°C)")
    plt.ylabel("Mean Latency (ms)")
    plt.title(f"Latency vs Temperature by CPU Load ({res}×{res})")
    load_handles = [Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=c, markersize=8)
                    for c in load_colors.values()]
    load_labels = [f"{l}% CPU" for l in load_colors]
    dev_handles = [Line2D([0], [0], marker=board_marker[b], color="k",
                          linestyle="None", markersize=8) for b in boards]
    plt.legend(handles=load_handles + dev_handles,
               labels=load_labels + boards,
               loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    out = f"latency_vs_temp_{res}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[wrote] {out}")


def main():
    for res in RESOLUTIONS:
        boxplot_all_boards(res)
        scatter_latency_vs_temp(res)


if __name__ == "__main__":
    main()
