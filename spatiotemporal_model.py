import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import RNN
from GNN_model import GATConv


class GATRNNCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=1, dropout_rate=0.0, **kwargs):
        """
        Custom RNN cell using multi-layer GATs.

        Args:
          input_dim: Dimensionality of external input features at each time step.
          hidden_dim: Dimensionality of the hidden state (must be divisible by num_heads).
          num_layers: Number of stacked GATConv layers.
          num_heads: Number of attention heads.
          dropout_rate: Dropout rate on attention coefficients.
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.gat_per_head_dim = hidden_dim // num_heads
        self.gat_layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.gat_layers.append(
                GATConv(
                    input_dim=in_dim,
                    output_dim=self.gat_per_head_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate
                )
            )
        self.linear = layers.Dense(hidden_dim)

    @property
    def state_size(self):
        # The state is per-node: shape [num_nodes, hidden_dim].
        return tf.TensorShape([None, self.hidden_dim])

    @property
    def output_size(self):
        return self.hidden_dim

    def call(self, inputs, states, training=None):
        """
        Args:
          inputs: A tuple (X_t, A_t) where:
                   X_t: [batch, num_nodes, input_dim]
                   A_t: [batch, E, 2]
          states: List containing the previous hidden state.
                  It may be shape [batch, hidden_dim] or [batch, num_nodes, hidden_dim].
          training: Boolean indicating training mode.
        Returns:
          A tuple (output, new_states) with shape [batch, num_nodes, hidden_dim].
        """
        # Unpack the tuple input.
        x_t, A_t = inputs  # x_t: [batch, num_nodes, input_dim], A_t: [batch, E, 2]
        batch_size = tf.shape(x_t)[0]
        num_nodes = tf.shape(x_t)[1]

        prev_state = states[0]
        if prev_state.shape.rank == 2:
            prev_state = tf.expand_dims(prev_state, axis=1)  # [batch, 1, hidden_dim]
            prev_state = tf.tile(prev_state, [1, num_nodes, 1])  # [batch, num_nodes, hidden_dim]

        def apply_gat(x_A):
            x_sample, A_sample = x_A  # x_sample: [num_nodes, input_dim], A_sample: [E, 2]
            h = x_sample
            for gat in self.gat_layers:
                h = gat([h, A_sample], training=training)
            return h

        gat_output = tf.map_fn(apply_gat, (x_t, A_t), dtype=tf.float32)
        combined = tf.concat([gat_output, prev_state], axis=-1)  # [batch, num_nodes, 2 * hidden_dim]
        next_state = self.linear(combined)  # [batch, num_nodes, hidden_dim]
        return next_state, [next_state]


# ------------------- Test Case -------------------

if __name__ == '__main__':
    # Toy example
    input_dim = 16      # Dimensionality of external input node features.
    hidden_dim = 32     # Hidden state dimensionality (must be divisible by num_heads).
    num_nodes = 10      # Number of nodes per graph.
    num_layers = 2      # Number of stacked GATConv layers.
    num_heads = 4       # Number of attention heads.
    dropout_rate = 0.1  # Dropout rate for attention coefficients.
    E = 9               # Number of edges per graph.

    # Create an instance of the custom GAT RNN cell.
    gat_rnn_cell = GATRNNCell(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate
    )

    # Wrap the custom cell with TensorFlow's built-in RNN layer.
    rnn_layer = tf.keras.layers.RNN(gat_rnn_cell, return_sequences=True)

    # Example inputs.
    B = 2  # Batch size
    T = 5  # Sequence length

    # External node features: shape (B, T, num_nodes, input_dim)
    example_X = tf.random.uniform(shape=(B, T, num_nodes, input_dim))

    # Graph edge data: shape (B, T, E, 2). These can change over time.
    example_A = tf.random.uniform(
        shape=(B, T, E, 2),
        minval=0, maxval=num_nodes, dtype=tf.int32
    )

    # Randomly generate an initial state
    initial_state = tf.random.uniform(shape=(B, num_nodes, hidden_dim))

    # Call the RNN with the tuple (X, A)
    output_sequence = rnn_layer((example_X, example_A), initial_state=[initial_state])
    print("Output shape:", output_sequence.shape)
