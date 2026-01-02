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
        self.n_special_action = n_special_actions

        # Shared layers
        self.one_hot_layer_1 = layers.CategoryEncoding(
            num_tokens=n_actions+n_special_actions,
            output_mode='one_hot'
        )
        self.one_hot_layer_2 = OneHotWithMasking(depth=n_actions+n_special_actions)
        self.masking = layers.Masking(mask_value=-1)
        
        self.concat        = layers.Concatenate()
        self.flatten_layer = layers.Flatten()
        self.embedding     = layers.Embedding(input_dim=n_actions+n_special_actions, output_dim=3, name='action_embedding', mask_zero=True)
        self.lstm_1        = layers.LSTM(16, return_sequences=True)
        self.lstm_2        = layers.LSTM(64)
        self.lstm_3        = layers.LSTM(16)
        self.lstm_4        = layers.Bidirectional(self.lstm_1)
        self.lstm_5        = layers.Bidirectional(self.lstm_2)
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
        action_seq = inputs["action_seq"]
        action_frames = inputs["action_frames"]
        relative_action_frames = inputs["relative_action_frames"]
        duration_left = inputs["duration_left"]
        
        # return self._call_ver_6(action_seq, duration_left)
        # return self._call_ver_7(action_seq, action_frames, duration_left)
        # return self._call_ver_8(action_seq, action_frames, duration_left)
        # return self._call_ver_9(action_seq, action_frames, duration_left)
        # return self._call_ver_10(action_seq, action_frames, duration_left)
        # return self._call_ver_11(action_seq, relative_action_frames, duration_left)
        return self._call_ver_12(action_seq, relative_action_frames, duration_left)
    
    def _call_ver_6(self, action_seq, duration_left):
        one_hot_features = self.one_hot_layer_2(action_seq)
        lstm_features    = self.lstm_2(one_hot_features)

        concat_features = self.concat([lstm_features, duration_left])

        actor = self.actor_hidden_1(concat_features)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(concat_features)
        critic = self.critic_output(critic)

        return actor, critic

    def _call_ver_7(self, action_seq, action_frames, duration_left):
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
    
    def _call_ver_8(self, action_seq, action_frames, duration_left):
        one_hot_features = self.one_hot_layer_2(action_seq)        
        lstm_features    = self.lstm_2(one_hot_features)

        paddings = [[0, 0], [0, self.seq_len - tf.shape(action_frames)[1]]]
        action_frames = tf.pad(action_frames, paddings, "CONSTANT")

        concat_features = self.concat([lstm_features, duration_left, action_frames[:, 1:]])

        actor = self.actor_hidden_1(concat_features)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(concat_features)
        critic = self.critic_output(critic)

        return actor, critic
    
    def _call_ver_9(self, action_seq, action_frames, duration_left):
        action_seq = self.one_hot_layer_2(action_seq)        
        lstm_features_1 = self.lstm_2(action_seq)

        action_frames = self.masking(tf.expand_dims(action_frames, axis=-1))
        lstm_features_2 = self.lstm_3(action_frames)

        concat_features = self.concat([lstm_features_1, lstm_features_2, duration_left])

        actor = self.actor_hidden_1(concat_features)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(concat_features)
        critic = self.critic_output(critic)

        return actor, critic
    
    def _call_ver_10(self, action_seq, action_frames, duration_left):
        is_padding = (action_seq == 0)
        is_padding_expanded = tf.expand_dims(is_padding, axis=-1)
        one_hot_features = self.one_hot_layer_1(action_seq)
        one_hot_features = tf.where(is_padding_expanded, -1.0, one_hot_features)
    
        one_hot_features = self.masking(one_hot_features)
        action_frames = self.masking(tf.expand_dims(action_frames, axis=-1))

        seq_features = self.concat([one_hot_features, action_frames])
        lstm_features = self.lstm_2(seq_features)

        concat_features = self.concat([lstm_features, duration_left])

        actor = self.actor_hidden_1(concat_features)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(concat_features)
        critic = self.critic_output(critic)

        return actor, critic
    
    def _call_ver_11(self, action_seq, relative_action_frames, duration_left):
        is_padding = (action_seq == 0)
        is_padding_expanded = tf.expand_dims(is_padding, axis=-1)
        one_hot_features = self.one_hot_layer_1(action_seq)
        one_hot_features = tf.where(is_padding_expanded, -1.0, one_hot_features)

        one_hot_features = self.masking(one_hot_features)
        relative_action_frames = self.masking(tf.expand_dims(relative_action_frames, axis=-1))

        seq_features = self.concat([one_hot_features, relative_action_frames])
        lstm_features = self.lstm_2(seq_features)

        concat_features = self.concat([lstm_features, duration_left])

        actor = self.actor_hidden_1(concat_features)
        actor = self.actor_hidden_2(actor)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(concat_features)
        critic = self.critic_hidden_2(critic)
        critic = self.critic_output(critic)

        return actor, critic
    
    def _call_ver_12(self, action_seq, relative_action_frames, duration_left):
        batch_size, _ = action_seq.shape
        
        is_padding = tf.expand_dims((action_seq == 0), axis=-1)

        action_embedding = self.one_hot_layer_1(action_seq)
        action_embedding = tf.where(is_padding, -1.0, action_embedding)
        action_embedding = self.masking(action_embedding)

        # Shape: (B, seq) --> [[0, 0, 0, ...], [1, 1, 1, ...], ...]
        batch_idx = tf.broadcast_to(tf.expand_dims(tf.range(batch_size), axis=1), (batch_size, self.seq_len))

        # Shape: (B, seq) --> [[0, 1, 2, ...], [0, 1, 2, ...], ...]
        seq_idx = tf.broadcast_to(tf.expand_dims(tf.range(self.seq_len), axis=0), (batch_size, self.seq_len))
        
        # Shape: (B, seq); the <None> action will be masked later
        action_idx = tf.where((action_seq != 0), action_seq - 1, 0)
        
        # Coordinate tensor with shape: (B, seq, 3)
        full_indices = tf.stack([batch_idx, seq_idx, action_idx], axis=-1)
        
        frame_embedding = tf.scatter_nd(full_indices, relative_action_frames, (batch_size, self.seq_len, self.action_size + 1))
        frame_embedding = tf.where(is_padding, -1.0, frame_embedding)
        frame_embedding = self.masking(frame_embedding)

        seq_embedding = self.concat([action_embedding, frame_embedding])
        lstm_features = self.lstm_4(seq_embedding)
        lstm_features = self.lstm_5(lstm_features)

        concat_features = self.concat([lstm_features, duration_left])

        actor = self.actor_hidden_1(concat_features)
        actor = self.actor_hidden_2(actor)
        actor = self.actor_output(actor)

        critic = self.critic_hidden_1(concat_features)
        critic = self.critic_hidden_2(critic)
        critic = self.critic_output(critic)

        return actor, critic

# Example usage
if __name__ == "__main__":

    seq_len = 20
    n_actions = 2

    model = ActorCritic(seq_len, n_actions, n_special_actions=2)
    print(model.summary())