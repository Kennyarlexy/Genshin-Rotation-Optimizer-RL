import os
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for moving average calculation

def plot_time_series(data, filename="time_series_plot.png"):
    """
    Plots a time series and its 5-point moving average, then saves it to a file.

    Parameters:
        data (list): A list of numerical values representing the time series.
        filename (str): The name of the file to save the plot as.
    """
    if not data:
        raise ValueError("The data list is empty.")

    # --- Start of new code ---
    # Calculate the 5-point moving average using pandas
    # This creates NaN for the first 4 values, which matplotlib ignores when plotting.
    moving_avg = pd.Series(data).rolling(window=10).mean()
    # --- End of new code ---

    # Ensure the "plots" directory exists
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)

    plt.figure(figsize=(10, 6))

    # Modified: Added a label for the legend
    plt.plot(data, marker='o', linestyle='-', color='teal', linewidth=2, markersize=6, label='Actual Reward')
    
    # --- Start of new code ---
    # Plot the moving average in orange
    plt.plot(moving_avg, color='orange', linestyle='-', linewidth=2, label='10-Episode Moving Average')
    # --- End of new code ---

    plt.title("Reward Per Episode", fontsize=16, fontweight='bold')
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- Start of new code ---
    # Add a legend to identify the lines
    plt.legend()
    # --- End of new code ---
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# Example of how to run it
if __name__ == '__main__':
    sample_rewards = [10, 15, 12, 20, 25, 22, 30, 28, 35, 40, 38, 33, 45, 50, 48]
    plot_time_series(sample_rewards, "reward_plot_with_ma.png")
    print("Plot saved to plots/reward_plot_with_ma.png")