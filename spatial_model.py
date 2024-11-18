from PIL.features import features

import tensorflow as tf

import model
import numpy as np

import logging
from scipy.spatial import Delaunay

from tensorflow import keras
from tensorflow.keras import layers



from model import FeedForward

logging.getLogger().setLevel(logging.INFO)

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix


def closeness_graph(context, threshold=10):
    """
    Computes the adjacency graph of a set of points where edges connect nodes that are within a specified distance.

    Parameters:
    context (numpy array): 2D numpy array where each row represents a point in n-dimensional space.
    threshold (float): The maximum distance between nodes to consider an edge.

    Returns:
    adj_matrix (tf.sparse.SparseTensor): Adjacency matrix representing the graph.
    """
    # Build a KDTree for efficient neighbor lookup
    tree = cKDTree(context)

    # Query the tree for all pairs within the threshold distance
    pairs = tree.query_pairs(r=threshold)

    # Initialize lists to construct the sparse adjacency matrix
    row = []
    col = []
    data = []

    for i, j in pairs:
        # Since the graph is undirected, add both (i, j) and (j, i)
        row.extend([i, j])
        col.extend([j, i])
        data.extend([1, 1])

    n_points = context.shape[0]

    # Create a sparse matrix in COOrdinate format
    adj_sparse = coo_matrix((data, (row, col)), shape=(n_points, n_points))

    # Convert the sparse matrix to a TensorFlow sparse tensor
    adj_tf_sparse = tf.sparse.SparseTensor(
        indices=np.vstack((adj_sparse.row, adj_sparse.col)).T,
        values=adj_sparse.data.astype(np.float32),
        dense_shape=adj_sparse.shape
    )

    # Ensure the indices are sorted in row-major order
    adj_tf_sparse = tf.sparse.reorder(adj_tf_sparse)

    return adj_tf_sparse


#Data the context as input
def delaunay_to_graph(context):
    """
    Computes the Delaunay triangulation of a set of points and returns an adjacency matrix as a TensorFlow sparse tensor.

    Parameters:
    context (numpy array): 2D numpy array where each row represents a point in 2D space.

    Returns:
    adj_matrix (tf.sparse.SparseTensor): Adjacency matrix representing the graph.
    """

    delaunay = Delaunay(context)
    simplices = delaunay.simplices

    # Create adjacency list
    edges = set()
    for simplex in simplices:
        # Create edges between all pairs of vertices in each simplex
        for i in range(3):
            for j in range(i + 1, 3):
                # Sort the tuple so that (i, j) is the same as (j, i)
                edge = tuple(sorted((simplex[i], simplex[j])))
                edges.add(edge)

    # Convert edges to numpy arrays (adjacency matrix format)
    edges = np.array(list(edges))
    row, col = edges[:, 0], edges[:, 1]
    data = np.ones(len(edges), dtype=np.float32)

    #Create a sparse adjacency matrix in TensorFlow
    adj_matrix = tf.sparse.SparseTensor(indices=np.stack([row, col], axis=1),
                                        values=data,
                                        dense_shape=[len(context), len(context)])

    # Since the adjacency matrix is symmetric, we also need to include the reverse edges
    adj_matrix_symmetric = tf.sparse.add(adj_matrix, tf.sparse.transpose(adj_matrix))

    return adj_matrix_symmetric



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        # Store the configuration strings for serialization
        self.kernel_initializer_config = keras.initializers.serialize(self.kernel_initializer)
        self.kernel_regularizer_config = keras.regularizers.serialize(self.kernel_regularizer)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out, attention_scores_norm

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'kernel_initializer': self.kernel_initializer_config,
            'kernel_regularizer': self.kernel_regularizer_config,
        })
        return config


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])[0]
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)

    def attention_scores(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain attention scores from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])[1]
            for attention_layer in self.attention_layers
        ]
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads,
            'merge_type': self.merge_type,
        })
        return config


class GraphAttentionNetwork(keras.Model):
    def __init__(
        self,
        node_states,
        edges,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads)
            for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_states, edges = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs

    # Compute the attention scores for each GAT layer
    def attention_scores(self, node_states, edges):
        x = self.preprocess(node_states)
        scores = []
        for attention_layer in self.attention_layers:
            scores.append(attention_layer.attention_scores([x, edges]))
            x = attention_layer([x, edges]) + x
        # scores[i][j] is the i'th layer's j'th head.
        return scores

    def train_step(self, data):
        indices, labels = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self([self.node_states, self.edges])
            # Compute loss
            loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        # Compute gradients
        grads = tape.gradient(loss, self.trainable_weights)
        # Apply gradients (update weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update metric(s)
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        indices = data
        # Forward pass
        outputs = self([self.node_states, self.edges])
        # Compute probabilities
        return tf.nn.softmax(tf.gather(outputs, indices))

    def test_step(self, data):
        indices, labels = data
        # Forward pass
        outputs = self([self.node_states, self.edges])
        # Compute loss
        loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        # Update metric(s)
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update({
            'node_states': self.node_states.numpy(),
            'edges': self.edges.numpy(),
            'hidden_units': self.hidden_units,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        node_states = tf.constant(config.pop('node_states'))
        edges = tf.constant(config.pop('edges'))
        return cls(node_states=node_states, edges=edges, **config)


#GCN
class GraphConvolution(layers.Layer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="kernel",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                regularizer=self.kernel_regularizer,
                trainable=True,
                name="bias",
            )
        self.built = True


    def call(self, inputs):
        node_states, adjacency = inputs
        # Normalize the adjacency matrix
        degrees = tf.sparse.reduce_sum(adjacency, output_is_sparse=False, axis=1)
        degree_inv_sqrt = 1/tf.math.sqrt(degrees)
        degree_inv_sqrt = tf.where(
            tf.math.is_inf(degree_inv_sqrt), tf.zeros_like(degree_inv_sqrt), degree_inv_sqrt
        )
        degree_mat_inv_sqrt = tf.linalg.diag(degree_inv_sqrt)
        adjacency_normalized = tf.matmul(
            tf.sparse.sparse_dense_matmul(degree_mat_inv_sqrt, adjacency), degree_mat_inv_sqrt
        )

        # Perform the graph convolution
        node_states_transformed = tf.matmul(node_states, self.kernel)
        output = tf.matmul(adjacency_normalized, node_states_transformed)

        if self.use_bias:
            output += self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class GraphConvolutionalNetwork(keras.Model):
    def __init__(
        self,
        node_states,
        adjacency,
        hidden_units,
        num_layers,
        output_dim,
        self_loops = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.adjacency = tf.cast(adjacency, tf.float32)
        self.self_loops = self_loops
        self.preprocess = layers.Dense(hidden_units, activation="relu")
        if self.self_loops:
            self.adjacency = tf.sparse.add(adjacency, tf.sparse.eye(adjacency.shape[0]))
        self.conv_layers = [
            GraphConvolution(hidden_units, activation="relu") for _ in range(num_layers)
        ]

        self.output_layer = GraphConvolution(output_dim)

    def call(self, inputs):
        node_states, adjacency = inputs
        x = self.preprocess(node_states)
        for conv_layer in self.conv_layers:
            x = conv_layer([x, adjacency])# Residual connection
        outputs = self.output_layer([x, adjacency])
        return outputs


    def train_step(self, data):
        indices, labels = data

        with tf.GradientTape() as tape:
            outputs = self([self.node_states, self.adjacency])
            loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        indices = data
        outputs = self([self.node_states, self.adjacency])
        return tf.nn.softmax(tf.gather(outputs, indices))

    def test_step(self, data):
        indices, labels = data
        outputs = self([self.node_states, self.adjacency])
        loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}





'''GraphSAGE'''
# GraphSAGE Layer
class GraphSAGELayer(layers.Layer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        # Weight matrix for concatenated self and neighbor features
        self.kernel = self.add_weight(
            shape=(2 * feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="kernel",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                regularizer=self.kernel_regularizer,
                trainable=True,
                name="bias",
            )
        super().build(input_shape)

    def call(self, inputs):
        node_states, adjacency = inputs
        # Compute degree inverse
        degrees = tf.sparse.reduce_sum(adjacency, axis=1)
        degree_inv = tf.math.reciprocal_no_nan(degrees)
        degree_inv_mat = tf.linalg.diag(degree_inv)
        # Normalize adjacency matrix
        adjacency_normalized = tf.sparse.sparse_dense_matmul(degree_inv_mat, adjacency)
        # Aggregate neighbor states
        neighbor_states = tf.matmul(adjacency_normalized, node_states)
        # Concatenate self and neighbor states
        concat_states = tf.concat([node_states, neighbor_states], axis=1)
        # Apply linear transformation
        output = tf.matmul(concat_states, self.kernel)
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

# GraphSAGE Model
class GraphSAGENetwork(keras.Model):
    def __init__(
        self,
        node_states,
        adjacency,
        hidden_units,
        num_layers,
        output_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.adjacency = tf.cast(adjacency, tf.float32)
        x = self.preprocess(node_states)
        self.sage_layers = [
            GraphSAGELayer(hidden_units, activation="relu") for _ in range(num_layers)
        ]
        self.output_layer = GraphSAGELayer(output_dim)

    def call(self, inputs):
        node_states, adjacency = inputs
        x = node_states
        for sage_layer in self.sage_layers:
            x = sage_layer([x, adjacency])
        outputs = self.output_layer([x, adjacency])
        return outputs

    def train_step(self, data):
        indices, labels = data
        with tf.GradientTape() as tape:
            outputs = self([self.node_states, self.adjacency])
            logits = tf.gather(outputs, indices)
            loss = self.compiled_loss(labels, logits)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        indices, labels = data
        outputs = self([self.node_states, self.adjacency])
        logits = tf.gather(outputs, indices)
        loss = self.compiled_loss(labels, logits)
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        indices = data
        outputs = self([self.node_states, self.adjacency])
        logits = tf.gather(outputs, indices)
        return tf.nn.softmax(logits)
