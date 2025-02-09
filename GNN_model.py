'''
@Author: Hongliang Zhou
'''
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LeakyReLU


#Data the context as input
logging.getLogger().setLevel(logging.INFO)

'''
Define sparse GAT model
'''
@keras.saving.register_keras_serializable(package="MyLayers")
class SparseDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=LeakyReLU(alpha=0.1)):
        super(SparseDenseLayer, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # Initialize weights
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'activation': self.activation})
        return config

    def call(self, inputs):
        # Perform sparse-dense matrix multiplication
        # output = tf.sparse.sparse_dense_matmul(inputs, self.W)
        output = tf.linalg.matmul(inputs, self.W, a_is_sparse = True)
        # Apply activation function
        if self.activation is not None:
            output = self.activation(output)
        return output

@tf.keras.saving.register_keras_serializable(package="MyLayers")
class GATConv(layers.Layer):
    """
    A simplified multi-head Graph Attention (GAT) convolution layer.
    Processes node features and logs attention scores each call.

    Args:
      input_dim: Dimensionality of input node features.
      output_dim: Dimensionality of output node features per head.
      num_heads: Number of attention heads.
      dropout_rate: Dropout rate on attention coefficients.
    """
    def __init__(self, input_dim, output_dim, num_heads=1, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.att_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self._logged_attention_scores = []

    def build(self, input_shape):
        # input_shape: [node_features shape, edges shape]
        if input_shape[0][-1] != self.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.input_dim}, but got {input_shape[0][-1]}"
            )
        initializer = tf.keras.initializers.GlorotUniform()
        self.W = self.add_weight(
            shape=(self.input_dim, self.num_heads * self.output_dim),
            initializer=initializer,
            name='W'
        )
        self.a = self.add_weight(
            shape=(self.num_heads, 2 * self.output_dim),
            initializer=initializer,
            name='a'
        )
        self.bias = self.add_weight(
            shape=(self.num_heads * self.output_dim,),
            initializer='zeros',
            name='bias'
        )
        super().build(input_shape)

    def attention_scores(self, h, edges):
        # h: [num_nodes, num_heads, output_dim], edges: [E, 2]
        src = edges[:, 0]
        dst = edges[:, 1]
        h_src = tf.gather(h, src)  # [E, num_heads, output_dim]
        h_dst = tf.gather(h, dst)  # [E, num_heads, output_dim]
        h_cat = tf.concat([h_src, h_dst], axis=-1)  # [E, num_heads, 2 * output_dim]
        e = tf.reduce_sum(self.a * h_cat, axis=-1)   # [E, num_heads]
        return e

    def call(self, inputs, training=False):
        # inputs is a tuple: (node_features, edges)
        x, edges = inputs  # x: [num_nodes, input_dim], edges: [E, 2]
        num_nodes = tf.shape(x)[0]
        xW = tf.matmul(x, self.W)
        h = tf.reshape(xW, [num_nodes, self.num_heads, self.output_dim])
        e = self.attention_scores(h, edges)
        e = tf.nn.leaky_relu(e, alpha=0.2)
        src = edges[:, 0]
        max_e = tf.math.unsorted_segment_max(e, src, num_segments=num_nodes)
        max_e_edge = tf.gather(max_e, src)
        exp_e = tf.exp(e - max_e_edge)
        sum_exp_e = tf.math.unsorted_segment_sum(exp_e, src, num_segments=num_nodes)
        sum_exp_e_edge = tf.gather(sum_exp_e, src)
        alpha = exp_e / (sum_exp_e_edge + 1e-10)
        if training:
            alpha = self.att_dropout(alpha, training=training)
        self._logged_attention_scores.append(alpha)
        dst = edges[:, 1]
        alpha_exp = tf.expand_dims(alpha, axis=-1)  # [E, num_heads, 1]
        messages = alpha_exp * tf.gather(h, src)      # [E, num_heads, output_dim]
        out = tf.math.unsorted_segment_sum(messages, dst, num_segments=num_nodes)
        out = tf.reshape(out, [num_nodes, self.num_heads * self.output_dim])
        out = out + self.bias
        return out

    def get_attention_scores(self):
        return self._logged_attention_scores

    def reset_attention_scores(self):
        self._logged_attention_scores = []
