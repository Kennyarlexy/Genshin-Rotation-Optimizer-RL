import keras
from keras import layers
import numpy as np


class ActorCritic(keras.Model):
    def __init__(self, state_size, n_actions):
        super().__init__()
        self.state_size  = state_size
        self.action_size = n_actions

        # Shared layers
        self.embedding = layers.Embedding(input_dim=n_actions+1, output_dim=8, input_length=state_size, name='action_embedding', mask_zero=True)
        self.lstm_1    = layers.LSTM(48, name='lstm_feature_extractor_1')

        self.concat = layers.Concatenate()

        # Actor layers
        self.actor_hidden_1 = layers.Dense(128, activation='relu')
        self.actor_hidden_2 = layers.Dense(48, activation='relu')
        self.actor_output = layers.Dense(n_actions, activation='softmax')

        # Critic layers
        self.critic_hidden_1 = layers.Dense(64, activation='relu')
        self.critic_hidden_2 = layers.Dense(16, activation='relu')
        self.critic_output = layers.Dense(1, activation='linear')

    def call(self, inputs):
        action_seq = inputs

        action_features = self.embedding(action_seq)
        action_features = self.lstm_1(action_features)
        # features = self.concat([action_features, cd_features])
        features = action_features

        actor = self.actor_hidden_1(features)
        actor = self.actor_hidden_2(actor)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(features)
        critic = self.critic_hidden_2(critic)
        critic = self.critic_output(critic)

        return actor, critic
    
    @property
    def actor_vars(self):
        return self.actor_hidden_1.trainable_variables + self.actor_output.trainable_variables

    @property
    def critic_vars(self):
        return self.critic_hidden_1.trainable_variables + self.critic_output.trainable_variables

    @property
    def shared_vars(self):
        return self.embedding.trainable_variables + self.lstm_1.trainable_variables

# Example usage
if __name__ == "__main__":
    state_size = 30
    action_size = 4

    model = ActorCritic(state_size, action_size)
    action_seq = [1, 2, 3] + [0]*27
    print(action_seq)

    input_tensor = np.array(action_seq, dtype=np.int32).reshape((1, -1))
    print(input_tensor)
    policy, value = model.predict(input_tensor)        

    print("Policy:", policy)
    print("Value:", value)

    print(type(policy), type(value))
    print(model.summary())