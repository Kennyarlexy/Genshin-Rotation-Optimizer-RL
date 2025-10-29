import tensorflow as tf
import keras
from keras import layers


class ActorCritic(keras.Model):
    def __init__(self, state_size, n_actions):
        super().__init__()
        self.state_size  = state_size
        self.action_size = n_actions

        # Shared layers
        self.one_hot_layer = keras.layers.CategoryEncoding(
            num_tokens=n_actions+1,
            output_mode='one_hot'
        )
        self.flatten_layer = keras.layers.Flatten()
        self.embedding = layers.Embedding(input_dim=n_actions+1, output_dim=3, name='action_embedding', mask_zero=True)
        self.lstm_1    = layers.LSTM(16, return_sequences=True)
        self.lstm_2    = layers.LSTM(64)
        self.conv_1d   = layers.Conv1D(filters=4, kernel_size=3, activation='relu')
        self.max_pooling = layers.MaxPooling1D(pool_size=2)

        # Actor layers
        self.actor_hidden_1 = layers.Dense(32, activation='relu')
        self.actor_hidden_2 = layers.Dense(32, activation='relu')
        self.actor_output = layers.Dense(n_actions, activation='softmax')

        # Critic layers
        self.critic_hidden_1 = layers.Dense(32, activation='relu')
        self.critic_hidden_2 = layers.Dense(16, activation='relu')
        self.critic_output = layers.Dense(1, activation='linear')

    def call(self, inputs):
        action_seq = inputs

        # features = self.embedding(action_seq)
        features = self.one_hot_layer(action_seq)
        lstm_features = self.lstm_1(features)
        lstm_features = self.lstm_2(lstm_features)
        # features = self.conv_1d(features)
        # features = self.max_pooling(features)
        # features = self.lstm_1(features)

        # features = self.one_hot_layer(action_seq)
        features = self.flatten_layer(features)


        actor = self.actor_hidden_1(features)
        actor = self.actor_hidden_2(actor)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(lstm_features)
        # critic = self.critic_hidden_2(critic)
        critic = self.critic_output(critic)

        return actor, critic

# Example usage
if __name__ == "__main__":
    state_size = 20
    action_size = 2

    sample_input = tf.zeros((1, 20))
    model = ActorCritic(state_size, action_size)
    model(sample_input)
    print(model.summary())