"""Custom Keras objects used by the CNN-LSTM-Attention model.

These classes are required to (de)serialize ``best_pso_ransomware_model.h5``;
pass :data:`CUSTOM_OBJECTS` to ``tf.keras.models.load_model``.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall


class F2Score(tf.keras.metrics.Metric):
    """Custom F2 Score metric for TensorFlow 2.10"""
    def __init__(self, name='f2_score', **kwargs):
        super(F2Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()

        beta = 2.0
        numerator = (1 + beta**2) * (precision * recall)
        denominator = (beta**2 * precision) + recall + tf.keras.backend.epsilon()
        f2 = numerator / denominator
        return f2

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


class AttentionLayer(layers.Layer):
    """Enhanced attention mechanism optimized for TensorFlow 2.10"""
    def __init__(self, attention_dim=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.supports_masking = True

    def build(self, input_shape):
        self.W_a = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_a = self.add_weight(
            name='attention_bias',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u_a = self.add_weight(
            name='attention_context',
            shape=(self.attention_dim,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        uit = tf.nn.tanh(tf.tensordot(inputs, self.W_a, axes=1) + self.b_a)
        ait = tf.tensordot(uit, self.u_a, axes=1)

        if mask is not None:
            ait = tf.where(mask, ait, -1e9)

        ait = tf.nn.softmax(ait, axis=1)
        ait_expanded = tf.expand_dims(ait, axis=-1)
        weighted_input = inputs * ait_expanded
        output = tf.reduce_sum(weighted_input, axis=1)

        return output, ait

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'attention_dim': self.attention_dim})
        return config


CUSTOM_OBJECTS = {
    'F2Score': F2Score,
    'AttentionLayer': AttentionLayer,
}
