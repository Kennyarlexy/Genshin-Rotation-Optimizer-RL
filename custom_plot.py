import os

import matplotlib.pyplot as plt

def plot_time_series(data, filename="time_series_plot.png"):
    """
    Plots a time series from a list of numbers and saves it to a file.

    Parameters:
        data (list): A list of numerical values representing the time series.
        filename (str): The name of the file to save the plot as.
    """
    if not data:
        raise ValueError("The data list is empty.")

    # Ensure the "plots" directory exists
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)

    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='o', linestyle='-', color='teal', linewidth=2, markersize=6)
    plt.title("Time Series Plot", fontsize=16, fontweight='bold')
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()