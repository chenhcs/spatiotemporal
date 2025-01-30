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

@keras.saving.register_keras_serializable(package="MyLayers")
class GATConv(layers.Layer):
    """
    A simplified GAT (Graph Attention) convolution layer (multi-head)
    that logs its attention scores every time call() is invoked.

    Arguments:
      input_dim: The dimensionality of input node features.
      output_dim: The dimensionality of output node features per head.
      num_heads: Number of attention heads.
      dropout_rate: Float, dropout on attention coefficients if desired.
      ...
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_heads=1,
                 dropout_rate=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        initializer = tf.keras.initializers.GlorotUniform()

        # Weight matrix for node features -> transformed features
        self.W = self.add_weight(
            shape=(self.input_dim, self.num_heads * self.output_dim),
            initializer=initializer,
            name='W'
        )

        # Attention parameter 'a' for each head, shape (num_heads, 2 * output_dim)
        self.a = self.add_weight(
            shape=(self.num_heads, 2 * self.output_dim),
            initializer=initializer,
            name='a'
        )

        # Optionally, a bias term (for each head, or shared)
        self.bias = self.add_weight(
            shape=(self.num_heads * self.output_dim,),
            initializer='zeros',
            name='bias'
        )

        # Dropout
        self.att_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

        # A place to store per-call attention (list of Tensors)
        self._logged_attention_scores = []

    def attention_scores(self, h, edges):
        """
        Given node features h [num_nodes, num_heads, output_dim]
        and edges [E, 2], compute the *raw* attention coefficient for each edge,
        per head. Return shape [E, num_heads].

        This is the heart of GAT where you use 'a' to compute e_{ij}.
        """
        # 1) Gather features for source and target
        src = edges[:, 0]  # shape [E]
        dst = edges[:, 1]  # shape [E]

        # Gather node features
        h_src = tf.gather(h, src)  # shape [E, num_heads, output_dim]
        h_dst = tf.gather(h, dst)  # shape [E, num_heads, output_dim]

        # 2) Concatenate source & target for each head
        # shape [E, num_heads, 2*output_dim]
        h_cat = tf.concat([h_src, h_dst], axis=-1)

        # 3) Multiply by 'a' (batch dot). We have a of shape [num_heads, 2*output_dim].
        # shape: e => [E, num_heads]
        e = tf.reduce_sum(self.a * h_cat, axis=-1)

        return e  # shape [E, num_heads]

    def call(self, inputs, training=False):
        """
        Forward pass of GATConv.
        inputs = [node_features, edges]
          node_features: shape [num_nodes, input_dim]
          edges: shape [E, 2], each row [source, target]
        """
        x, edges = inputs

        num_nodes = tf.shape(x)[0]
        xW = tf.matmul(x, self.W)
        h = tf.reshape(xW, [num_nodes, self.num_heads, self.output_dim])
        e = self.attention_scores(h, edges)  # shape [E, num_heads]
        e = tf.nn.leaky_relu(e, alpha=0.2)  # shape [E, num_heads]

        src = edges[:, 0]  # shape [E]
        max_e = tf.math.unsorted_segment_max(e, src, num_segments=num_nodes)
        max_e_edge = tf.gather(max_e, src)  # shape [E, num_heads]
        exp_e = tf.exp(e - max_e_edge)

        sum_exp_e = tf.math.unsorted_segment_sum(exp_e, src, num_segments=num_nodes)
        # shape [num_nodes, num_heads]

        sum_exp_e_edge = tf.gather(sum_exp_e, src)  # shape [E, num_heads]

        alpha = exp_e / (sum_exp_e_edge + 1e-10)  # shape [E, num_heads]

        # 5) Optionally dropout on alpha
        if training:
            alpha = self.att_dropout(alpha, training=training)  # shape [E, num_heads]

        # 6) Log these attention scores
        self._logged_attention_scores.append(alpha)

        # 7) Message passing: sum of (alpha * neighbor_features)
        dst = edges[:, 1]  # shape [E]

        # Expand alpha for broadcast: shape [E, num_heads, 1]
        alpha_exp = tf.expand_dims(alpha, axis=-1)
        # shape => [E, num_heads, out_dim]
        messages = alpha_exp * tf.gather(h, src)

        out = tf.math.unsorted_segment_sum(messages, dst, num_segments=num_nodes)
        out = tf.reshape(out, [num_nodes, self.num_heads * self.output_dim])
        out = out + self.bias

        return out  # shape [N, num_heads * out_dim]

    def get_attention_scores(self):
        """
        Returns the list of attention Tensors logged across calls,
        in the order they occurred.

        Each element is shape [E, num_heads] if you only store alpha in call().
        """
        return self._logged_attention_scores

    def reset_attention_scores(self):
        """
        Clears the stored attention scores.
        Useful if you want to run multiple inferences
        and keep logs separate.
        """
        self._logged_attention_scores = []