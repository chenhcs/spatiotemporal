import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_addons as tfa

import tensorflow_text

import collections
import os
import pathlib
import re
import string
import sys
import tempfile

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)

class QueryPositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, d_model, gene_size):
    super().__init__()
    self.d_model = d_model
    #self.embedding = tf.keras.layers.Conv1D(d_model, 1, activation='relu', input_shape=(None, gene_size))
    self.proj_kernel = self.add_weight(shape=(gene_size, d_model), initializer='random_normal', trainable=True)

  def call(self, x):
    x = tf.einsum('blm,mn->blmn', x, self.proj_kernel)
    return x

class ValuePositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, d_model, gene_size):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Dense(d_model, activation='relu')
    #self.embedding = tf.keras.layers.Conv1D(d_model, 1, activation='relu', input_shape=(None, gene_size))

  def call(self, x):
    #length = tf.shape(x)[1]
    x = self.embedding(x)
    return x

class CustomAttention(tf.keras.layers.Attention):
    def _calculate_scores(self, query, key):
        scores = tf.einsum('blmn,bkpn->blk', query, key)
        return scores

    def call(
        self,
        inputs,
        mask=None,
        training=None,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        self._validate_call_args(inputs=inputs, mask=mask)
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(query=q, key=k)
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

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = CustomAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
  def call(self, x, x_query, context_key, context_value):
    attn_output, attn_scores = self.mha(
        inputs=[x_query, context_value, context_key],
        use_causal_mask = True,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

class GlobalSelfAttention(BaseAttention):
  def call(self, x, x_query, x_key, x_value):
    attn_output = self.mha(
        inputs=[x_query, x_value, x_key],
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class CausalSelfAttention(BaseAttention):
  def call(self, x, x_query, x_key, x_value):
    attn_output = self.mha(
        inputs=[x_query, x_value, x_key],
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
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

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        use_scale=True,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, x_query, x_key, x_value):
    x = self.self_attention(x, x_query, x_key, x_value)
    x = self.ffn(x)
    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, gene_size, num_heads,
               dff, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.exp_bias = self.add_weight(shape=(gene_size, ), initializer='random_normal', trainable=True)
    self.query_pos_embedding = QueryPositionalEmbedding(d_model=d_model, gene_size=gene_size)
    self.key_pos_embedding = QueryPositionalEmbedding(d_model=d_model, gene_size=gene_size)
    self.value_pos_embedding = ValuePositionalEmbedding(d_model=gene_size, gene_size=gene_size)

    self.enc_layers = [
        EncoderLayer(d_model=gene_size,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    #self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)

    for i in range(self.num_layers):
        x_query = self.query_pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        x_key = self.key_pos_embedding(x)
        x_value = self.value_pos_embedding(x)
        x = self.enc_layers[i](x, x_query, x_key, x_value)

    return x  # Shape `(batch_size, seq_len, d_model)`.

class DecoderCausalLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderCausalLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        use_scale=True,
        dropout=dropout_rate)

  def call(self, x, x_query, x_key, x_value):
    x = self.causal_self_attention(x, x_query, x_key, x_value)

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

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, x_query, context_key, context_value):
    x = self.cross_attention(x, x_query, context_key, context_value)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, gene_size, context_gene_size, num_heads, dff,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    #self.pos_embedding = PositionalEmbedding(d_model=d_model, gene_size=gene_size)
    self.exp_bias = self.add_weight(shape=(gene_size, ), initializer='random_normal', trainable=True)
    self.query_pos_embedding = QueryPositionalEmbedding(d_model=d_model, gene_size=gene_size)
    self.key_pos_embedding = QueryPositionalEmbedding(d_model=d_model, gene_size=gene_size)
    self.value_pos_embedding = ValuePositionalEmbedding(d_model=gene_size, gene_size=gene_size)
    self.context_key_pos_embedding = QueryPositionalEmbedding(d_model=d_model, gene_size=context_gene_size)
    self.context_value_pos_embedding = ValuePositionalEmbedding(d_model=gene_size, gene_size=context_gene_size)
    #self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_causal_layers = [
        DecoderCausalLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.dec_cross_layers = [
        DecoderCrossLayer(d_model=gene_size, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    #x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    #x = self.dropout(x)
    #x = x + self.exp_bias
    context_key = self.context_key_pos_embedding(context)
    context_value = self.context_value_pos_embedding(context)

    for i in range(self.num_layers):
        x_query = self.query_pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        x_key = self.key_pos_embedding(x)
        x_value = self.value_pos_embedding(x)
        x = self.dec_causal_layers[i](x, x_query, x_key, x_value)
        x_query = self.query_pos_embedding(x)
        x = self.dec_cross_layers[i](x, x_query, context_key, context_value)


    self.last_attn_scores = self.dec_cross_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x

class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff, context_gene_size, target_gene_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model, gene_size=context_gene_size,
                           num_heads=num_heads, dff=dff,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model, gene_size=target_gene_size,
                           context_gene_size=context_gene_size,
                           num_heads=num_heads, dff=dff,
                           dropout_rate=dropout_rate)

    #self.final_layer = tf.keras.layers.Dense(target_gene_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = x #self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits

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


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)   #change
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')

def plot_attention_head(in_tokens, translated_tokens, attention):
  # The model didn't generate `<START>` in the output. Skip it.
  translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention)
  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(translated_tokens)))

  labels = [label.decode('utf-8') for label in in_tokens.numpy()]
  ax.set_xticklabels(
      labels, rotation=90)

  labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
  ax.set_yticklabels(labels)
  plt.savefig('fig/attention_weights.png')


'''
#test example

context = np.array([
[[1., 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
[[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
])

input = np.array([
[[0., 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]], [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
[[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
[[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]
])

label = input[:, 1:, :].copy()
input = input[:, :5, :].copy()

context_val = np.array([
[[0., 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], [[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
[[0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
[[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
])

input_val = np.array([
[[0., 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]], [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
[[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
[[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
])

label_val = input_val[:, 1:, :].copy()
input_val = input_val[:, :5, :].copy()

print(context.shape, input.shape, label.shape, context_val.shape, input_val.shape, label_val.shape)

'''

context = np.load('data/context_array_train_cont_.npy') #np.random.rand(1000, 100, 200) * 10
context = context[:, :30, :]
for i in range(context.shape[-1]):
    context[:, :, i] -= np.min(context[:, :, i])
print(np.min(context))
input = np.load('data/target_array_train_cont_.npy') #np.random.rand(1000, 100, 300) * 10
for i in range(input.shape[-1]):
    input[:, :, i] -= np.min(input[:, :, i])
label = np.load('data/label_array_train_cont_.npy') #np.random.rand(1000, 100, 300) * 10
for i in range(label.shape[-1]):
    label[:, :, i] -= np.min(label[:, :, i])
#label = label - input
context_val = np.load('data/context_array_test_cont_.npy') #np.random.rand(1000, 100, 200) * 10
context_val = context_val[:, :30, :]
for i in range(context_val.shape[-1]):
    context_val[:, :, i] -= np.min(context_val[:, :, i])
input_val = np.load('data/target_array_test_cont_.npy') #np.random.rand(1000, 100, 300) * 10
for i in range(input_val.shape[-1]):
    input_val[:, :, i] -= np.min(input_val[:, :, i])
label_val = np.load('data/label_array_test_cont_.npy') #np.random.rand(1000, 100, 300) * 10
for i in range(label_val.shape[-1]):
    label_val[:, :, i] -= np.min(label_val[:, :, i])
index_all = np.arange(len(input_val))
#label_val = label_val - input_val
#np.random.shuffle(index_all)
#input = input_[index_all[:80000]] #input_val[index_all[:20000]]
#context = context_[index_all[:80000]]
#label = label_[index_all[:80000]]
input_val = input_val[index_all[:20000]] #input_[index_all[80000:]]
context_val = context_val[index_all[:20000]]
label_val = label_val[index_all[:20000]]

print(context.shape)

num_layers = 1
d_model = 32
dff = 512
num_heads = 1
dropout_rate = 0.0
context_gene_size = context.shape[-1]
target_gene_size = label.shape[-1]

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    context_gene_size=context_gene_size,
    target_gene_size=target_gene_size,
    dropout_rate=dropout_rate)

output = transformer([context[:101], input[:101]])

transformer.summary()

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[tfa.metrics.SpearmansRank()])

output = transformer([context[:101], input[:101]])

print(context.shape)
print(input.shape)
print(output.shape)

attn_scores = transformer.decoder.dec_cross_layers[-1].last_attn_scores
print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)
print(attn_scores[0])

weight = transformer.decoder.context_value_pos_embedding.get_weights()
print(weight[0].shape)

history = transformer.fit((context, input), label,
                batch_size=64,
                epochs=20)
                #validation_data=([context_val, input_val], label_val))

weight = transformer.decoder.context_value_pos_embedding.get_weights()
weight = np.squeeze(weight[0])
print(weight)
print(np.max(weight))
plt.clf()
ax = plt.gca()
ax.matshow(weight)
plt.savefig('results/recp2tar_weights.png')
np.save('results/recp2tar_weights.npy', weight)
