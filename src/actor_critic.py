import tensorflow as tf
import keras
from custom_layers import OneHotWithMasking
from keras import layers
from env import GcsimState


class ActorCritic(keras.Model):
    def __init__(self, seq_len, n_actions, n_special_actions):
        super().__init__()
        self.seq_len = seq_len
        self.action_size = n_actions

        # Shared layers
        self.one_hot_layer_1 = layers.CategoryEncoding(
            num_tokens=n_actions+n_special_actions,
            output_mode='one_hot'
        )
        self.one_hot_layer_2 = OneHotWithMasking(depth=n_actions+n_special_actions)
        
        self.concat        = layers.Concatenate()
        self.flatten_layer = layers.Flatten()
        self.embedding     = layers.Embedding(input_dim=n_actions+n_special_actions, output_dim=3, name='action_embedding', mask_zero=True)
        self.lstm_1        = layers.LSTM(16, return_sequences=True)
        self.lstm_2        = layers.LSTM(64)
        self.dense_1       = layers.Dense(32, activation='relu')
        self.conv_1d       = layers.Conv1D(filters=4, kernel_size=3, activation='relu')
        self.max_pooling   = layers.MaxPooling1D(pool_size=2)

        # Actor layers
        self.actor_hidden_1 = layers.Dense(64, activation='relu')
        self.actor_hidden_2 = layers.Dense(32, activation='relu')
        self.actor_output   = layers.Dense(n_actions, activation='softmax')

        # Critic layers
        self.critic_hidden_1 = layers.Dense(64, activation='relu')
        self.critic_hidden_2 = layers.Dense(32, activation='relu')
        self.critic_output   = layers.Dense(1, activation='linear')

        self.build(input_shape=(None, self.seq_len))

    @tf.function
    def call(self, inputs):
        action_seq, duration_left = inputs
        
        # return self._call_ver_1(action_seq)
        # return self._call_ver_2(action_seq)
        # return self._call_ver_3(action_seq)
        # return self._call_ver_4(action_seq)
        # return self._call_ver_5(action_seq)
        return self._call_ver_6(action_seq, duration_left)
    
    def _call_ver_1(self, inputs):
        one_hot_features = self.one_hot_layer_1(inputs)
        lstm_features    = self.lstm_2(one_hot_features)

        actor = self.flatten_layer(one_hot_features)
        actor = self.actor_hidden_1(actor)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(lstm_features)
        critic = self.critic_output(critic)

        return actor, critic
    
    def _call_ver_2(self, inputs):
        one_hot_features = self.one_hot_layer_1(inputs)
        one_hot_features = one_hot_features[:, :, 1:]

        lstm_inputs = self.one_hot_layer_2(inputs)
        lstm_features = self.lstm_2(lstm_inputs)

        actor = self.flatten_layer(one_hot_features)
        actor = self.actor_hidden_1(actor)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(lstm_features)
        critic = self.critic_output(critic)

        return actor, critic
    
    def _call_ver_3(self, inputs):
        inputs = self.one_hot_layer_2(inputs)
        features = self.lstm_2(inputs)

        actor = self.actor_hidden_1(tf.stop_gradient(features))
        actor = self.actor_hidden_2(actor)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(features)
        critic = self.critic_output(critic)

        return actor, critic
    
    def _call_ver_4(self, inputs):
        one_hot_features = self.one_hot_layer_2(inputs)

        zeros_cnt = tf.reduce_sum(tf.cast(inputs == 0, tf.int32))
        is_start_neuron = tf.constant([[int(zeros_cnt == self.seq_len - 1)]], dtype=tf.float32)

        actor_input = one_hot_features[:, 1:, :-1]
        actor = self.flatten_layer(actor_input)
        actor = self.actor_hidden_1(tf.concat([is_start_neuron, actor], axis=-1))
        actor = self.actor_hidden_2(actor)
        actor = self.actor_output(actor)

        critic = self.lstm_2(one_hot_features)
        critic = self.critic_hidden_1(critic)
        critic = self.critic_output(critic)

        return actor, critic
    
    def _call_ver_5(self, inputs):
        one_hot_features = self.one_hot_layer_2(inputs)
        lstm_features    = self.lstm_2(one_hot_features)

        actor = self.actor_hidden_1(lstm_features)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(lstm_features)
        critic = self.critic_output(critic)

        return actor, critic

    def _call_ver_6(self, action_seq, duration_left):
        one_hot_features = self.one_hot_layer_2(action_seq)
        lstm_features    = self.lstm_2(one_hot_features)

        concat_features = self.concat([lstm_features, duration_left])

        actor = self.actor_hidden_1(concat_features)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(concat_features)
        critic = self.critic_output(critic)

        return actor, critic

    def _call_ver_7(self, action_seq, duration_left):
        one_hot_features = self.one_hot_layer_2(action_seq)
        action_frames = tf.expand_dims(action_frames, axis=-1)
        seq_features = self.concat([one_hot_features, action_frames])
        
        lstm_features    = self.lstm_2(seq_features)

        concat_features = self.concat([lstm_features, duration_left])

        actor = self.actor_hidden_1(concat_features)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(concat_features)
        critic = self.critic_output(critic)

        return actor, critic
    

# Example usage
if __name__ == "__main__":
    seq_len = 20
    n_actions = 2

    model = ActorCritic(seq_len, n_actions, n_special_actions=2)
    print(model.summary())