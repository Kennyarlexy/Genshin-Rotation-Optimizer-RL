import numpy as np

class WelfordNormalizer:
    """
    An online normalizer that calculates the running mean and standard deviation
    of a data stream using Welford's algorithm.

    This allows for a separation between updating the statistics ('fit') and
    applying the normalization ('transform').
    """
    def __init__(self):
        self.count = 1e-4  # Initialize with a small value to avoid division by zero
        self.mean = 0.0
        self.M2 = 0.0  # The sum of squared differences from the mean

    @property
    def variance(self):
        """Calculates the current variance."""
        return self.M2 / self.count

    @property
    def std(self):
        """Calculates the current standard deviation."""
        # Use max to prevent sqrt of a negative number due to floating point errors
        return np.sqrt(max(self.variance, 1e-8))

    def update(self, new_data):
        """
        Updates the running statistics with a new data point.
        This is the 'fit' part of the process.
        """
        self.count += 1
        delta = new_data - self.mean
        self.mean += delta / self.count
        
        # This is the core of Welford's algorithm for updating M2
        delta2 = new_data - self.mean
        self.M2 += delta * delta2

    def transform(self, data):
        """
        Normalizes a data point using the current running mean and std.
        This does NOT update the statistics.
        """
        # Add a small epsilon for numerical stability
        return (data - self.mean) / (self.std + 1e-8)

    def fit_transform(self, new_data):
        """
        A convenience method that first updates the stats and then
        returns the normalized version of the new data point.
        """
        self.update(new_data)
        return self.transform(new_data)
    