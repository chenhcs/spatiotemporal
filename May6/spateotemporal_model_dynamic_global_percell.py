import logging
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

#import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.ops import math_ops as ops
from tensorflow.python.ops import nn
#from GNN_model import GATConv
#import tensorflow_addons as tfa

#import tensorflow_text

import collections
import os
import pathlib
import re
import string
import sys
import tempfile

class CellEmbedding(tf.keras.layers.Layer):
  def __init__(self, d_model, gene_size):
    super().__init__()
    self.d_model = d_model
    self.proj_kernel = self.add_weight(shape=(gene_size, d_model), initializer='random_normal', trainable=True)

  def call(self, x):
    x = tf.einsum('blm,mn->bln', x, self.proj_kernel)
    return x

class GeneEmbedding(tf.keras.layers.Layer):
  def __init__(self, d_model, gene_size):
    super().__init__()
    self.d_model = d_model
    self.gene_size = gene_size
    #self.embedding = [tf.keras.layers.Conv1D(d_model if i < 4 else gene_size * d_model, 1, activation='relu') for i in range(5)]
    self.proj_kernel_token = self.add_weight(shape=(gene_size, d_model), initializer='random_normal', trainable=True)
    self.proj_kernel_disc_value = self.add_weight(shape=(d_model, 1), initializer='random_normal', trainable=True)
    self.proj_kernel_value = self.add_weight(shape=(d_model, 1), initializer='random_normal', trainable=True)
    self.proj_kernel2_value = self.add_weight(shape=(d_model, d_model), initializer='random_normal', trainable=True)
    self.scale_factor = self.add_weight(shape=(), initializer="ones", trainable=True)
    self.proj_kernel_dict_value = self.add_weight(shape=(42, d_model), initializer='random_normal', trainable=True)
    self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
    self.self_attention = BaseAttentionGene(
        num_heads=1,
        key_dim=d_model,
        dropout=0.1)
    #self.ffn = FeedForward(d_model, 128)

  def call(self, x, gene_size):
    '''
    def frange(start, stop, step):
        while start <= stop:
            yield start
            start += step
    '''
    x_token = tf.einsum('blm,mn->blmn', tf.ones_like(x, dtype=tf.float32), self.proj_kernel_token)
    x_value = tf.einsum('blm,nk->blmn', x, self.proj_kernel_value)
    x_value = self.leaky_relu(x_value)
    x_value = tf.einsum('blmn,nk->blmk', x_value, self.proj_kernel2_value) + self.scale_factor * x_value
    x_value = tf.nn.softmax(x_value, axis=-1)
    x_value = tf.einsum('blmn,nk->blm', x_value, self.proj_kernel_disc_value)

    '''
    bin_edges = lst = [round(x, 1) for x in frange(-2, 2.01, 0.1)]
    # Assign each value to a bin
    x_value = tf.raw_ops.Bucketize(input=x, boundaries=list(bin_edges[1:-1]))  # boundaries must exclude -inf/+inf
    x_value = tf.nn.embedding_lookup(self.proj_kernel_dict_value, x_value)
    '''

    #x = x_token + x_value
    #x = tf.math.multiply(x_token, x_value)
    x = tf.einsum('blmn,blm->blmn', x_token, x_value)
    bs = tf.shape(x)[0]
    ls = tf.shape(x)[1]
    x = tf.reshape(x, tf.concat([[bs * ls], tf.shape(x)[2:]], axis=0))
    x = self.self_attention(x)
    #x = self.ffn(x)
    x = self.self_attention(x)
    #x = self.ffn(x)
    #x = self.self_attention(x)
    #x = self.ffn(x)
    x = tf.reshape(x, tf.concat([[bs, ls], tf.shape(x)[1:]], axis=0))
    return x_token[:, :, :gene_size, :], x[:, :, :gene_size, :]

class CustomAttentionCell(tf.keras.layers.Attention):
    def __init__(self, **kwargs):
        super().__init__()

    def _calculate_values(self, vq, vk, vexp):
        scores1 = tf.einsum('blmn,bkpn->blkmp', vq, vk)
        v = tf.einsum('blkmp,bkp->blkm', scores1, vexp)
        return v

    def _apply_scores(self, scores, value, scores_mask=None, training=False):
        """Applies attention scores to the given value tensor.

        To use this method in your attention layer, follow the steps:

        * Use `query` tensor of shape `(batch_size, Tq)` and `key` tensor of
            shape `(batch_size, Tv)` to calculate the attention `scores`.
        * Pass `scores` and `value` tensors to this method. The method applies
            `scores_mask`, calculates
            `attention_distribution = softmax(scores)`, then returns
            `matmul(attention_distribution, value).
        * Apply `query_mask` and return the result.

        Args:
            scores: Scores float tensor of shape `(batch_size, Tq, Tv)`.
            value: Value tensor of shape `(batch_size, Tv, dim)`.
            scores_mask: A boolean mask tensor of shape `(batch_size, 1, Tv)`
                or `(batch_size, Tq, Tv)`. If given, scores at positions where
                `scores_mask==False` do not contribute to the result. It must
                contain at least one `True` value in each line along the last
                dimension.
            training: Python boolean indicating whether the layer should behave
                in training mode (adding dropout) or in inference mode
                (no dropout).

        Returns:
            Tensor of shape `(batch_size, Tq, dim)`.
            Attention scores after masking and softmax with shape
                `(batch_size, Tq, Tv)`.
        """
        if scores_mask is not None:
            padding_mask = ops.logical_not(scores_mask)
            # Bias so padding positions do not contribute to attention
            # distribution.  Note 65504. is the max float16 value.
            max_value = 65504.0 if scores.dtype == "float16" else 1.0e9
            scores -= max_value * ops.cast(padding_mask, dtype=scores.dtype)

        weights = nn.softmax(scores)
        if training and self.dropout > 0:
            weights = backend.random.dropout(
                weights,
                self.dropout,
                seed=self.seed_generator,
            )
        return tf.einsum('blk,blkm->blm', weights, value), weights

    def call(
        self,
        inputs,
        mask=None,
        training=None,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        #self._validate_call_args(inputs=inputs, mask=mask)
        q = inputs[0]
        vexp = inputs[1][0]
        vq = inputs[1][1] #blmn
        vk = inputs[1][2] #bkpn  blkm
        k = inputs[2]
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(query=q, key=k)
        v = self._calculate_values(vq=vq, vk=vk, vexp=vexp)
        if v_mask is not None:
            # Mask of shape [batch_size, 1, Tv].
            v_mask = tf.expand_dims(v_mask, axis=-2)
        if self.causal or use_causal_mask:
            # Creates a lower triangular mask, so position i cannot attend to
            # positions j>i. This prevents the flow of information from the
            # future into the past.
            scores_shape = tf.shape(scores)
            # causal_mask_shape = [1, Tq, Tv].
            causal_mask_shape = tf.concat(
                [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0
            )
            causal_mask = _lower_triangular_mask(causal_mask_shape)
        else:
            causal_mask = None
        scores_mask = _merge_masks(v_mask, causal_mask)
        result, attention_scores = self._apply_scores(
            scores=scores, value=v, scores_mask=scores_mask, training=training
        )
        if q_mask is not None:
            # Mask of shape [batch_size, Tq, 1].
            q_mask = tf.expand_dims(q_mask, axis=-1)
            result *= tf.cast(q_mask, dtype=result.dtype)
        if return_attention_scores:
            return result, attention_scores
        return result

def _lower_triangular_mask(shape):
    """Creates a lower-triangular boolean mask over the last 2 dimensions."""
    row_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-2)
    col_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-1)
    return tf.greater_equal(row_index, col_index)

def _merge_masks(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return tf.logical_and(x, y)

class BaseAttentionCell(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = CustomAttentionCell(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class BaseAttentionGene(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttentionCell):
  def call(self, x_query, context_key, context_vexp, x_vq, context_vk):
    attn_output, attn_scores = self.mha(
        inputs=[x_query, (context_vexp, x_vq, context_vk), context_key],
        use_causal_mask = True,
        return_attention_scores=True)
    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    #x = self.add([x, attn_output])
    x = attn_output
    #x = self.layernorm(x)

    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x

class DecoderCrossLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderCrossLayer, self).__init__()

    self.cross_attention = CrossAttention(
        use_scale=True,
        dropout=dropout_rate)

    #self.ffn = FeedForward(d_model, dff)

  def call(self, x_query, context_key, context_vexp, x_vq, context_vk):
    x = self.cross_attention(x_query, context_key, context_vexp, x_vq, context_vk)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    #x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x

'''
class SpatialCellEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=1, dropout_rate=0.0, **kwargs):
        """
        Customized RNN cell using multi-layer GATs, can be intialized and used as input to the RNN in keras.

        Args:
          input_dim: Dimensionality of external input features at each time step
          hidden_dim: Dimensionality of the hidden state (must be divisible by num_heads)
          num_layers: Number of stacked GATConv layers
          num_heads: Number of attention heads for each GATConv layer
          dropout_rate: Dropout rate on attention coefficients
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
            in_dim = input_dim #if i == 0 else hidden_dim
            self.gat_layers.append(
                GATConv(
                    input_dim=in_dim,
                    output_dim=self.gat_per_head_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate
                )
            )
        # Linear layer has the same hidden dimension as the GATConv
        self.linear = tf.keras.layers.Dense(hidden_dim)

    @property
    def state_size(self):
        # The state is per-node: shape [num_nodes, hidden_dim].
        return tf.TensorShape([None, self.hidden_dim])

    @property
    def output_size(self):
        return self.hidden_dim

    def call(self, inputs, training=None):
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
        # Unpack the tuple input
        x_t, A_t = inputs  # x_t: [batch, num_nodes, input_dim], A_t: [batch, E, 2]
        batch_size = tf.shape(x_t)[0]
        time_length = tf.shape(x_t)[1]
        num_nodes = tf.shape(x_t)[2]
        num_edges = tf.shape(A_t)[2]
        x_t = tf.reshape(x_t, [-1, num_nodes, self.input_dim])
        A_t = tf.reshape(A_t, [-1, num_edges, 2])

        def apply_gat(x_A):
            x_sample, A_sample = x_A  # x_sample: [num_nodes, input_dim], A_sample: [E, 2]
            x_t_ = x_sample
            i = 0
            for gat in self.gat_layers:
                i += 1
                h, x_t_ = gat([x_t_, A_sample], training=training)
            return h, x_t_

        # Batch training
        gat_output, x_t_ = tf.map_fn(apply_gat, (x_t, A_t), dtype=(tf.float32, tf.float32))

        #Add any post-processing here; in practice, we should apply a readout function or extract the node feature for the target node.

        #combined = tf.concat([gat_output, prev_state], axis=-1)  # Combine the hidden dimension and the previous_hidden [batch, num_nodes, 2 * hidden_dim]
        next_state = self.linear(gat_output)  # [batch, num_nodes, hidden_dim]

        next_state = tf.squeeze(next_state)
        x_t_ = tf.squeeze(x_t_)

        next_state = tf.reshape(next_state, [batch_size, time_length, self.hidden_dim])
        x_t_ = tf.reshape(x_t_, [batch_size, time_length, self.input_dim])

        return next_state, x_t_
'''

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, target_gene_size, ligand_gene_size, receptor_gene_size, tf_gene_size, num_heads, dff,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.target_gene_size = target_gene_size
    self.ligand_gene_size = ligand_gene_size
    self.receptor_gene_size = receptor_gene_size
    self.tf_gene_size = tf_gene_size

    #three cell emebddings for each cell from (1) target genes, (2) TFs, (3) signaling genes (ligands + receptors) respectively
    self.embed_cell_target = CellEmbedding(d_model=d_model, gene_size=target_gene_size)
    self.embed_cell_tf = CellEmbedding(d_model=d_model, gene_size=tf_gene_size)
    self.embed_cell_ligrecp = CellEmbedding(d_model=d_model, gene_size=ligand_gene_size + receptor_gene_size)

    #embed genes in each cell for (1) target genes, (2) TFs, (3) signaling gene (ligand + receptor) respectively
    self.embed_gene_target = GeneEmbedding(d_model=d_model, gene_size=ligand_gene_size + receptor_gene_size + target_gene_size + tf_gene_size)
    self.embed_gene_tf = GeneEmbedding(d_model=d_model, gene_size=ligand_gene_size + receptor_gene_size + tf_gene_size + target_gene_size)
    self.embed_gene_ligrecp = GeneEmbedding(d_model=d_model, gene_size=ligand_gene_size + receptor_gene_size + tf_gene_size + target_gene_size)

    self.dec_cross_layers_tf = DecoderCrossLayer(d_model=target_gene_size, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)

    self.dec_cross_layers_ligand = DecoderCrossLayer(d_model=target_gene_size, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)

    self.last_attn_scores = None

  def call(self, target_exp, tf_exp, ligrecp_exp):
    cell_embedding_ligrecp = self.embed_cell_ligrecp(ligrecp_exp)
    ligrecp_gene_embedding_global, ligrecp_gene_embedding_percell = self.embed_gene_ligrecp(tf.concat([ligrecp_exp, tf_exp, target_exp], axis=-1), self.ligand_gene_size + self.receptor_gene_size)
    ligand_gene_embedding_global = ligrecp_gene_embedding_global[:, :, :self.ligand_gene_size, :]
    receptor_gene_embedding_global = ligrecp_gene_embedding_global[:, :, self.ligand_gene_size:, :]
    ligand_gene_embedding_percell = ligrecp_gene_embedding_percell[:, :, :self.ligand_gene_size, :]
    receptor_gene_embedding_percell = ligrecp_gene_embedding_percell[:, :, self.ligand_gene_size:, :]

    cell_embedding_tf = self.embed_cell_tf(tf_exp)
    tf_gene_embedding_global, tf_gene_embedding_percell = self.embed_gene_tf(tf.concat([tf_exp, ligrecp_exp, target_exp], axis=-1), self.tf_gene_size)

    cell_embedding_target = self.embed_cell_target(target_exp)  # Shape `(batch_size, seq_len, d_model)`.
    target_gene_embedding_global, target_gene_embedding_percell = self.embed_gene_target(tf.concat([target_exp, ligrecp_exp, tf_exp], axis=-1), self.target_gene_size)

    receptor_exp = ligrecp_exp[:, :, self.ligand_gene_size:]
    target_exp_y1_global = self.dec_cross_layers_tf(cell_embedding_target, cell_embedding_tf, tf_exp, (target_gene_embedding_global), (tf_gene_embedding_global))
    target_exp_y2_global = self.dec_cross_layers_ligand(cell_embedding_target, cell_embedding_ligrecp, receptor_exp, (target_gene_embedding_global), (receptor_gene_embedding_global))
    target_exp_y1_percell = self.dec_cross_layers_tf(cell_embedding_target, cell_embedding_tf, tf_exp, (target_gene_embedding_percell), (tf_gene_embedding_percell))
    target_exp_y2_percell = self.dec_cross_layers_ligand(cell_embedding_target, cell_embedding_ligrecp, receptor_exp, (target_gene_embedding_percell), (receptor_gene_embedding_percell))

    self.last_attn_scores = self.dec_cross_layers_tf.last_attn_scores
    self.x_vq1_global = target_gene_embedding_global #target gene embedding (batch_size, seq_len, genes, d_model)
    self.tf_vk1_global = tf_gene_embedding_global #tf gene embedding (batch_size, seq_len, genes, d_model)
    self.recp_vk_global = receptor_gene_embedding_global #signaling gene embedding (batch_size, seq_len, genes, d_model)
    self.x_vq1_percell = target_gene_embedding_percell #target gene embedding (batch_size, seq_len, genes, d_model)
    self.tf_vk1_percell = tf_gene_embedding_percell #tf gene embedding (batch_size, seq_len, genes, d_model)
    self.recp_vk_percell = receptor_gene_embedding_percell #signaling gene embedding (batch_size, seq_len, genes, d_model)

    network_tf_global = tf.einsum('blmn,bkpn->blkmp', target_gene_embedding_global, tf_gene_embedding_global)
    network_recp_global = tf.einsum('blmn,bkpn->blkmp', target_gene_embedding_global, receptor_gene_embedding_global)
    network_tf_percell = tf.einsum('blmn,bkpn->blkmp', target_gene_embedding_percell, tf_gene_embedding_percell)
    network_recp_percell = tf.einsum('blmn,bkpn->blkmp', target_gene_embedding_percell, receptor_gene_embedding_percell)

    ##tf_exp_y3 = lr_model(ligrecp_exp, ligand_gene_embedding, receptor_gene_embedding)
    ##We can build a neural network to predict tf/target gene expression from the product of ligand and receptor genes
    ##We can use gene embeddings to reconstruct the weights of the first layer of the neural network
    ##The first layer of the neural network has the weights W of the size (n_ligand*n_receptor, h_dim)
    ##On the other hand, for each cell, we have the ligand gene embeddings "stored in ligand_gene_embedding" of size (n_ligand, h_dim), and receptor gene embeddings "stored in receptor_gene_embedding" of size (n_receptor, h_dim)
    ##we can be reconstructed the weight matrix W by the pairwise elementwise product of every row in "ligand_gene_embedding" with every row in "receptor_gene_embedding", which has the shape of (n_ligand, n_receptor, h_dim)
    return target_exp_y1_global, target_exp_y2_global, target_exp_y1_percell, target_exp_y2_percell, network_tf_global, network_recp_global, network_tf_percell, network_recp_percell

class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff, ligand_gene_size, receptor_gene_size, tf_gene_size, target_gene_size, dropout_rate=0.1):
    super().__init__()

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model, target_gene_size=target_gene_size,
                           ligand_gene_size=ligand_gene_size,
                           receptor_gene_size=receptor_gene_size,
                           tf_gene_size=tf_gene_size,
                           num_heads=num_heads, dff=dff,
                           dropout_rate=dropout_rate)

  def call(self, inputs):
    ligrecp_exp, tf_exp, target_exp = inputs

    target_exp_y1_global, target_exp_y2_global, target_exp_y1_percell, target_exp_y2_percell, network_tf_global, network_recp_global, network_tf_percell, network_recp_percell = self.decoder(target_exp, tf_exp, ligrecp_exp)  # (batch_size, target_len, d_model)

    logits1_global = target_exp_y1_global
    logits2_global = target_exp_y2_global
    logits1_percell = target_exp_y1_percell
    logits2_percell = target_exp_y2_percell

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits1_global._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return target_exp_y1_global, target_exp_y2_global, target_exp_y1_percell, target_exp_y2_percell, network_tf_global, network_recp_global, network_tf_percell, network_recp_percell

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def masked_loss(label, pred):
  mask = label != -99999
  loss_object = tf.keras.losses.MeanSquaredError(
    name='mean_squared_error')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss

def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred))

tlength = 3 #length of time points in each sample

#training data
tf_exp = np.load('data_triple/tf_array_train.npy') #shape: [sample, time, genes]
tf_exp = tf_exp[:, :tlength, :]

'''
ligand_exp = np.load('data_triple/ligand_array_train.npy')
ligand_exp = ligand_exp[:, :tlength, :]

recp_exp = np.load('data_triple/recep_array_train.npy')
recp_exp = recp_exp[:, :tlength, :]

ligrecp_exp = np.concatenate((ligand_exp, recp_exp), axis=-1)
'''

ligrecp_exp = np.load('data_triple/lr_pair_array_train.npy')
ligrecp_exp = ligrecp_exp[:, :tlength, :]

target_exp = np.load('data_triple/target_array_train.npy') #input
target_exp_y = np.load('data_triple/label_array_train.npy') #label
#target_exp = target_exp[:, :, np.r_[30:46]]
#target_exp_y = target_exp_y[:, :, np.r_[30:46]]

#validation data
tf_exp_val = np.load('data_triple/tf_array_test.npy')
tf_exp_val = tf_exp_val[:, :tlength, :]

'''
ligand_exp_val = np.load('data_triple/ligand_array_test.npy')
ligand_exp_val = ligand_exp_val[:, :tlength, :]

recp_exp_val = np.load('data_triple/recep_array_test.npy')
recp_exp_val = recp_exp_val[:, :tlength, :]

ligrecp_exp_val = np.concatenate((ligand_exp_val, recp_exp_val), axis=-1)
'''

ligrecp_exp_val = np.load('data_triple/lr_pair_array_test.npy')
ligrecp_exp_val = ligrecp_exp_val[:, :tlength, :]

target_exp_val = np.load('data_triple/target_array_test.npy') #input
target_exp_y_val = np.load('data_triple/label_array_test.npy') #ground truth label
#target_exp_val = target_exp_val[:, :, np.r_[30:46]]
#target_exp_y_val = target_exp_y_val[:, :, np.r_[30:46]]

print(tf_exp.shape, ligrecp_exp.shape, target_exp.shape, target_exp_y.shape)

num_layers = 1
d_model = 32 #embedding size
dff = 32 #dense layer neuron number
num_heads = 1
dropout_rate = 0.0
ligand_gene_size = 0 #ligand_exp.shape[-1]
receptor_gene_size = ligrecp_exp.shape[-1] #recp_exp.shape[-1] 
tf_gene_size = tf_exp.shape[-1]
target_gene_size = target_exp.shape[-1]

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    ligand_gene_size=ligand_gene_size,
    receptor_gene_size=receptor_gene_size,
    tf_gene_size=tf_gene_size,
    target_gene_size=target_gene_size,
    dropout_rate=dropout_rate)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

transformer.compile(
    loss={"output_1": masked_loss, "output_2": masked_loss, "output_3": masked_loss, "output_4": masked_loss, "output_5": l1_loss, "output_6": l1_loss, "output_7": l1_loss, "output_8": l1_loss},
    loss_weights={'output_1': 1.0, 'output_2': 1.0, 'output_3': 1.0, 'output_4': 1.0, 'output_5': 0.5, 'output_6': 0.5, 'output_7': 0.5, 'output_8': 0.5},
    optimizer=optimizer,
    metrics={"output_1":tf.keras.metrics.MeanSquaredError()}
    )


eps = 50

history = transformer.fit((ligrecp_exp, tf_exp, target_exp), {"output_1": target_exp_y, "output_2": target_exp_y, "output_3": target_exp_y, "output_4": target_exp_y, "output_5": target_exp_y, "output_6": target_exp_y, "output_7": target_exp_y, "output_8": target_exp_y},
                batch_size=3,
                epochs=eps,
                validation_data=([ligrecp_exp_val, tf_exp_val, target_exp_val], target_exp_y_val))

predictions, predictions2, _, _, _, _, _, _ = transformer([ligrecp_exp[:1000], tf_exp[:1000], target_exp[:1000]]) #ligand, context, input, ligand_neighbor, graph
print(predictions.shape)

attn_scores = transformer.decoder.dec_cross_layers_tf.last_attn_scores

plt.clf()
plt.matshow(predictions[:101, 2, :])
plt.savefig('results_test/pred_train.png')

plt.clf()
plt.matshow(target_exp_y[:101, 2, :])
plt.savefig('results_test/label_train.png')

predictions, predictions2, _, _, _, _, _, _ = transformer([ligrecp_exp_val[:1000], tf_exp_val[:1000], target_exp_val[:1000]])
print(predictions.shape)
#np.save('predictions.npy', predictions)

plt.clf()
plt.matshow(predictions[:101, 2, :])
plt.savefig('results_test/pred.png')

plt.clf()
plt.matshow(target_exp_y_val[:101, 2, :])
plt.savefig('results_test/label.png')

#construct and visualize regulatory networks
for b in range(1): #int(len(context)/1000) + 1
    predictions1, predictions2, _, _, _, _, _, _ = transformer([ligrecp_exp_val[b * 1000: (b+1) * 1000], tf_exp_val[b * 1000: (b+1) * 1000], target_exp_val[b * 1000: (b+1) * 1000]])
    x_vq1 = transformer.decoder.x_vq1_global
    tf_vk1 = transformer.decoder.tf_vk1_global

    recp_vk1 = transformer.decoder.recp_vk_global
    #x_vq2 = transformer.decoder.x_vq2
    #tf_vk2 = transformer.decoder.tf_vk2
    #x_vq3 = transformer.decoder.x_vq3
    #tf_vk3 = transformer.decoder.tf_vk3
    print(x_vq1.shape)
    print(tf_vk1.shape)
    #print(recp_vk1.shape)

    vq = [x_vq1]
    vk = [tf_vk1]
    lvk = [recp_vk1]
    weight_nt = np.zeros((tf_vk1.shape[-2], x_vq1.shape[-2]))
    weight_nt_lr = np.zeros((recp_vk1.shape[-2], x_vq1.shape[-2]))
    for i in range(1):
        x_vq = vq[i]
        tf_vk = vk[i]
        recp_vk = lvk[i]
        #for i in range(attn_scores.shape[0]):
        #x_vq_ = x_vq[i]
        #tf_vk_ = tf_vk[i]
        x_vq_ = tf.reduce_mean(x_vq, axis=0)
        tf_vk_ = tf.reduce_mean(tf_vk, axis=0)
        recp_vk_ = tf.reduce_mean(recp_vk, axis=0)
        x_vq_ = tf.reduce_mean(x_vq_, axis=0)
        tf_vk_ = tf.reduce_mean(tf_vk_, axis=0)
        recp_vk_ = tf.reduce_mean(recp_vk_, axis=0)
        weight = np.matmul(tf_vk_, np.transpose(x_vq_))
        weight_nt += np.abs(weight)
        weight_lr = np.matmul(recp_vk_, np.transpose(x_vq_))
        weight_nt_lr += np.abs(weight_lr)
        print(tf_vk_.shape, x_vq_.shape, weight.shape, weight_nt.shape)
    plt.clf()
    ax = plt.gca()
    ax.matshow(np.abs(weight_nt))
    plt.savefig('weight_test_triple/attention_weights_sample_' + str(b * 1000) + '.png')
    np.save('weight_test_triple/attention_weights_sample_' + str(b * 1000) + '.npy', weight_nt)
    plt.clf()
    ax = plt.gca()
    ax.matshow(np.abs(weight_nt_lr))
    plt.savefig('weight_test_triple/attention_weights_lr_sample_' + str(b * 1000) + '.png')
    np.save('weight_test_triple/attention_weights_lr_sample_' + str(b * 1000) + '.npy', weight_nt_lr)
    weight_all = np.vstack((weight_nt_lr, weight_nt))
    plt.clf()
    ax = plt.gca()
    ax.matshow(np.abs(weight_all))
    plt.savefig('weight_test_triple/attention_weights_all_sample_' + str(b * 1000) + '.png')
    np.save('weight_test_triple/attention_weights_all_sample_' + str(b * 1000) + '.npy', weight_all)


for b in range(int(len(ligrecp_exp_val)/1000) + 1):
    predictions1, predictions2, _, _, _, _, _, _ = transformer([ligrecp_exp_val[b * 1000: (b+1) * 1000], tf_exp_val[b * 1000: (b+1) * 1000], target_exp_val[b * 1000: (b+1) * 1000]])
    x_vq = transformer.decoder.x_vq1_percell
    tf_vk = transformer.decoder.tf_vk1_percell
    recp_vk = transformer.decoder.recp_vk_percell
    print(x_vq.shape)
    print(tf_vk.shape)
    print(recp_vk.shape)
    #if True:
    for i in range(predictions1.shape[0]):
        x_vq_ = x_vq[i]
        tf_vk_ = tf_vk[i]
        recp_vk_ = recp_vk[i]
        #x_vq_ = tf.reduce_mean(x_vq, axis=0)
        #tf_vk_ = tf.reduce_mean(tf_vk, axis=0)
        np.save('weight_test_triple/weights/weight_target_sample_' + str(b * 1000 + i) + '.npy', x_vq_)
        np.save('weight_test_triple/weights/weight_context_sample_' + str(b * 1000 + i) + '.npy', tf_vk_)
        np.save('weight_test_triple/weights/weight_lignad_sample_' + str(b * 1000 + i) + '.npy', recp_vk_)

        x_vq_ = tf.reduce_mean(x_vq_, axis=0)
        tf_vk_ = tf.reduce_mean(tf_vk_, axis=0)
        recp_vk_ = tf.reduce_mean(recp_vk_, axis=0)
        weight = np.matmul(tf_vk_, np.transpose(x_vq_))
        print(tf_vk_.shape, x_vq_.shape, weight.shape)
        plt.clf()
        ax = plt.gca()
        ax.matshow(np.abs(weight))
        plt.savefig('weight_test_triple/sample/attention_weights_sample_' + str(b * 1000 + i) + '.png')
        #np.save('weight_test_triple/sample/attention_weights_sample_' + str(b * 1000 + i) + '.npy', weight)

        weight = np.matmul(recp_vk_, np.transpose(x_vq_))
        print(recp_vk_.shape, x_vq_.shape, weight.shape)
        print(str(b * 1000 + i), tf.reduce_mean(tf_exp_val[b * 1000 + i], axis=0))
        plt.clf()
        ax = plt.gca()
        ax.matshow(np.abs(weight))
        plt.savefig('weight_test_triple/sample/attention_weights_lr_sample_' + str(b * 1000 + i) + '.png')
        #np.save('weight_test_triple/sample/attention_weights_lr_sample_' + str(b * 1000 + i) + '.npy', weight)

################################################################

head = 0
for b in range(int(len(ligrecp_exp_val)/1000) + 1):
    predictions1, predictions2, _, _, _, _, _, _ = transformer([ligrecp_exp[b * 1000: (b+1) * 1000], tf_exp[b * 1000: (b+1) * 1000], target_exp[b * 1000: (b+1) * 1000]])
    #print(b, predictions.shape)
    attn_scores = transformer.decoder.dec_cross_layers_tf.last_attn_scores
    attn_scores_lg = transformer.decoder.dec_cross_layers_ligand.last_attn_scores
    for i in range(attn_scores.shape[0]):
        attention = attn_scores[i]
        print(attention.shape)
        plt.clf()
        ax = plt.gca()
        ax.matshow(attention)
        plt.savefig('fig_test/attention_weights_sample_' + str(b * 1000 + i) + '.png')
        #np.save('data_test/attention_weights_sample_' + str(b * 1000 + i) + '.npy', attention)
        attention = attn_scores_lg[i]
        print(attention.shape)
        plt.clf()
        ax = plt.gca()
        ax.matshow(attention)
        plt.savefig('fig_test/attention_weights_sample_' + str(b * 1000 + i) + '_ligand.png')
