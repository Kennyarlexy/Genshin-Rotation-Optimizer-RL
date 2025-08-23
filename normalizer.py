class WelfordNormalizer:
    def __init__(self):
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.data_count = 1e-4  # Prevent division by zero

    def transform(self, new_data):
        self.data_count += 1
        delta = new_data - self.reward_mean
        self.reward_mean += delta / self.data_count
        delta2 = new_data - self.reward_mean
        self.reward_var += delta * delta2  # Welford update

        reward_std = max((self.reward_var / self.data_count) ** 0.5, 1e-6)
        normalized_reward = (new_data - self.reward_mean) / reward_std

        return normalized_reward