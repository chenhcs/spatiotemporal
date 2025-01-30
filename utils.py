import os
import numpy as np
import scipy
import tensorflow as tf
import networkx as nx

from tensorflow import keras
from keras import Input, Model
from keras.layers import Dropout, BatchNormalization, Dense, Softmax
from scipy.spatial import cKDTree, Delaunay
from scipy.sparse import csr_matrix, coo_matrix

from simulation import h5ad
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
)

from GNN_model import SparseDenseLayer, GATConv

# Attempt to import scanpy; install if missing
try:
    import scanpy as sc
except:
    os.system('python -m pip install scanpy')
    import scanpy as sc

# # Attempt to import scvi; install if missing
# try:
#     import scvi
# except:
#     os.system('python -m pip install -U scvi-tools')
#     import scvi

def create_model(input_dim, hidden_units, num_heads, num_classes):
    """Build a GAT-based model with sparse input."""
    x_in = Input(shape=(input_dim,))
    a_in = Input(shape=(2,), dtype=tf.int64)

    # Pre-processing Layer
    x_0 = SparseDenseLayer(hidden_units * num_heads)(x_in)
    x_1 = Dropout(0.1)(x_0)
    x_1 = BatchNormalization()(x_1)
    x_1 += x_0

    x_2 = GATConv(hidden_units * num_heads, hidden_units, num_heads=num_heads)([x_1, a_in])
    x_2 = Dropout(0.1)(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 += x_1

    x_3 = GATConv(hidden_units * num_heads, hidden_units, num_heads=num_heads)([x_2, a_in])
    x_3 = Dropout(0.1)(x_3)
    x_3 = BatchNormalization()(x_3)
    x_3 += x_2

    # Post-processing Layer
    out = Dense(num_classes)(x_3)
    out = Softmax()(out)

    return Model(inputs=[x_in, a_in], outputs=out)

def preprocess(adata):
    """
    Basic preprocessing:
    - Filter cells and genes,
    - Normalize and apply log1p,
    - Retain highly variable genes.
    """
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.normalize_total(adata, target_sum=1e3)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata

def evaluate_metrics(model,
                     node_states_full,
                     edges_full,
                     labels_full,
                     indices,
                     num_classes,
                     batch_size=1024):
    """
    Evaluate on a set of indices (test/val) to compute
    accuracy, F1, precision, recall, AUC, top-2 accuracy.
    """
    dataset = tf.data.Dataset.from_tensor_slices(indices).batch(batch_size)
    all_preds, all_true = [], []

    for batch_indices in dataset:
        preds = model([node_states_full, edges_full], training=False)
        batch_preds = tf.gather(preds, batch_indices)
        batch_labels = tf.gather(labels_full, batch_indices)
        all_preds.append(batch_preds.numpy())
        all_true.append(batch_labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    pred_labels = np.argmax(all_preds, axis=1)

    acc = accuracy_score(all_true, pred_labels)
    f1 = f1_score(all_true, pred_labels, average='macro')
    precision = precision_score(all_true, pred_labels, average='macro', zero_division=0)
    recall = recall_score(all_true, pred_labels, average='macro', zero_division=0)

    # Multi-class AUC
    try:
        all_true_onehot = tf.one_hot(all_true, depth=num_classes).numpy()
        auc_val = roc_auc_score(all_true_onehot, all_preds, average='macro', multi_class='ovr')
    except ValueError:
        auc_val = float('nan')

    top2_acc = top_k_accuracy_score(all_true, all_preds, k=2)

    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc_val,
        "top2_acc": top2_acc,
    }

def load_mouse_midbrain_data(path, batchs=None):
    """
    Load and preprocess real data, then optionally subset by batch.
    Retains the full AnnData object with scvi embeddings (if available).
    """
    # Load the entire dataset
    if batchs is None:
        adata = h5ad.open_h5ad(path)
        if 'Cell Types' not in adata.obs.keys():
            adata.obs['Cell Types'] = adata.obs['annotation']
        adata = adata[adata.obs['Cell Types'].notna()]
        coordinates = adata.obsm['spatial']
        return adata, coordinates

    # Otherwise, subset the loaded data by batch
    adata = h5ad.open_h5ad(path)
    adata = preprocess(adata)
    adata_time_point = adata[adata.obs.Batch.isin(batchs)].copy()

    # If 'Cell Types' is missing, create it from 'annotation'
    if "Cell Types" not in adata_time_point.obs.keys():
        adata_time_point.obs["Cell Types"] = adata_time_point.obs["annotation"]

    # Filter out cells with no cell types
    adata_time_point = adata_time_point[adata_time_point.obs["Cell Types"].notna()].copy()
    coordinates = adata_time_point.obsm["spatial"]
    return adata_time_point, coordinates

def encode_cell_types(adata, encode=True):
    """
    Maps each unique cell type to an integer index (or vice versa).
    """
    idx_cell_types = {}
    cell_idx_types = {}
    for idx, cell_type in enumerate(np.unique(adata.obs['Cell Types'])):
        cell_idx_types[cell_type] = idx
        idx_cell_types[idx] = cell_type

    if encode:
        cell_types = adata.obs['Cell Types'].map(cell_idx_types)
    else:
        cell_types = adata.obs['Cell Types']
    return cell_types, idx_cell_types.values()

def sparse_tensor_to_networkx(adj_sparse, cell_type_map, directed=False):
    """
    Converts a TF SparseTensor adjacency matrix to a NetworkX graph.
    """
    if not tf.executing_eagerly():
        tf.compat.v1.enable_eager_execution()

    indices = adj_sparse.indices.numpy()
    values = adj_sparse.values.numpy()
    edge_list = [(int(i), int(j), float(w)) for (i, j), w in zip(indices, values)]

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_weighted_edges_from(edge_list)
    nx.set_node_attributes(G, cell_type_map, 'cell_type')
    return G

def acquire_sparse_data(X):
    """
    Converts a (possibly) CSR/CSC matrix to a TF SparseTensor.
    Ensures indices are sorted.
    """
    if not isinstance(X, scipy.sparse.coo_matrix):
        X_coo = X.tocoo()
    else:
        X_coo = X

    row_indices = X_coo.row
    col_indices = X_coo.col
    values = X_coo.data
    indices = np.vstack((row_indices, col_indices)).T.astype(np.int64)
    dense_shape = X_coo.shape

    sparse_tensor = tf.sparse.SparseTensor(
        indices=indices,
        values=values.astype(np.float32),
        dense_shape=dense_shape
    )
    return tf.sparse.reorder(sparse_tensor)

def sparse_tensor_to_csr(sparse_tensor):
    """
    Converts a TF SparseTensor to a scipy csr_matrix.
    """
    indices = sparse_tensor.indices.numpy()
    values = sparse_tensor.values.numpy()
    dense_shape = sparse_tensor.dense_shape.numpy()
    return csr_matrix((values, (indices[:, 0], indices[:, 1])), shape=dense_shape)

def closeness_graph(context, threshold=10):
    """
    Builds an adjacency graph for points within 'threshold' distance.
    Returns a TF SparseTensor adjacency matrix.
    """
    tree = cKDTree(context)
    pairs = tree.query_pairs(r=threshold)

    row, col, data = [], [], []
    for i, j in pairs:
        row.extend([i, j])
        col.extend([j, i])
        data.extend([1, 1])

    n_points = context.shape[0]
    adj_sparse = coo_matrix((data, (row, col)), shape=(n_points, n_points))

    adj_tf_sparse = tf.sparse.SparseTensor(
        indices=np.vstack((adj_sparse.row, adj_sparse.col)).T,
        values=adj_sparse.data.astype(np.float32),
        dense_shape=adj_sparse.shape
    )
    return tf.sparse.reorder(adj_tf_sparse)

def delaunay_to_graph(context):
    """
    Builds an adjacency matrix (as TF SparseTensor) via Delaunay triangulation in 2D.
    """
    delaunay = Delaunay(context)
    simplices = delaunay.simplices

    edges = set()
    for simplex in simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = tuple(sorted((simplex[i], simplex[j])))
                edges.add(edge)

    edges = np.array(list(edges))
    row, col = edges[:, 0], edges[:, 1]
    data = np.ones(len(edges), dtype=np.float32)

    adj_matrix = tf.sparse.SparseTensor(
        indices=np.stack([row, col], axis=1),
        values=data,
        dense_shape=[len(context), len(context)]
    )
    # Symmetrize
    adj_matrix_symmetric = tf.sparse.add(adj_matrix, tf.sparse.transpose(adj_matrix))
    return adj_matrix_symmetric

def top_k_accuracy_score(y_true, y_prob, k=5):
    """
    Fraction of samples whose true label is in the top-k predicted classes.
    """
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k_preds[i] for i in range(len(y_true)))
    return correct / len(y_true)

def create_subgraph(all_edges, train_indices):
    """
    Filter the global edges to keep only edges among the specified 'train_indices'.
    Then reindex those edges from 0..(train_size - 1).
    Returns (sub_edges, old_to_new_map).
    """
    train_indices = np.array(train_indices)
    old_to_new = {}
    for new_id, old_id in enumerate(train_indices):
        old_to_new[old_id] = new_id

    edges_np = all_edges.numpy()
    mask = np.isin(edges_np[:, 0], train_indices) & np.isin(edges_np[:, 1], train_indices)
    filtered_edges = edges_np[mask]

    remapped = np.array([[old_to_new[u], old_to_new[v]] for (u, v) in filtered_edges], dtype=np.int32)
    return tf.constant(remapped, dtype=tf.int32), old_to_new
