import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_time_series(data, filename="time_series_plot.png", window=30):
    """
    Plots a time series and moving average, then saves it to a file.

    Parameters:
        data (list): A list of numerical values representing the time series.
        filename (str): The name of the file to save the plot as.
    """
    if not data:
        raise ValueError("The data list is empty.")

    moving_avg = pd.Series(data).rolling(window=window).mean()

    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)

    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='o', linestyle='-', color='teal', linewidth=2, markersize=6, label='Actual Reward')    
    plt.plot(moving_avg, color='orange', linestyle='-', linewidth=2, label=f'{window}-Episode Moving Average')

    plt.title("Reward Per Episode", fontsize=16, fontweight='bold')
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# Example of how to run it
if __name__ == '__main__':
    sample_rewards = [10, 15, 12, 20, 25, 22, 30, 28, 35, 40, 38, 33, 45, 50, 48]
    plot_time_series(sample_rewards, "reward_plot_with_ma.png")
    print("Plot saved to plots/reward_plot_with_ma.png")