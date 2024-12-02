from simulation import h5ad
from sklearn.preprocessing import LabelEncoder
import numpy as np
from spatial_model import GraphAttentionNetwork
import scipy

def load_data(path):
    """Load and preprocess the real data."""
    adata = h5ad.open_h5ad(path)
    if 'Cell Types' not in adata.obs.keys():
        adata.obs['Cell Types'] = adata.obs['annotation']
    # Filter out cells without cell types
    adata = adata[adata.obs['Cell Types'].notna()]
    coordinates = adata.obsm['spatial']
    return adata, coordinates

def encode_cell_types(adata, encode=True):
    """Encode cell types into numerical labels."""
    # le = LabelEncoder()
    # cell_types = le.fit_transform(adata.obs['Cell Types'])
    idx_cell_types = {}
    cell_idx_types = {}

    for idx, cell_type in enumerate(np.unique(adata.obs['Cell Types'])):
        cell_idx_types[cell_type] = idx
        idx_cell_types[idx] = cell_type

    if encode ==True:
        cell_types = adata.obs['Cell Types'].map(cell_idx_types)
        # print(idx_cell_types.values())
    else:
        cell_types = adata.obs['Cell Types']
    return cell_types, idx_cell_types.values()

import tensorflow as tf
import networkx as nx

def sparse_tensor_to_networkx(adj_sparse, cell_type_map, directed=False):
    """
    Converts a TensorFlow SparseTensor adjacency matrix to a NetworkX graph.

    Parameters:
    - adj_sparse: TensorFlow SparseTensor representing the adjacency matrix.
    - directed: Boolean indicating whether the graph is directed.

    Returns:
    - G: A NetworkX graph.
    """
    # Ensure TensorFlow is executing eagerly
    if not tf.executing_eagerly():
        tf.compat.v1.enable_eager_execution()

    # print(cell_type_map.keys())
    # print(cell_type_map.values())
    # Extract indices and values from the SparseTensor
    indices = adj_sparse.indices.numpy()
    values = adj_sparse.values.numpy()

    # Create an edge list with weights
    edge_list = [(int(i), int(j), float(weight)) for (i, j), weight in zip(indices, values)]

    # Create the NetworkX graph
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_weighted_edges_from(edge_list)

    nx.set_node_attributes(G, cell_type_map, 'cell_type')

    return G

def acquire_sparse_data(X):
    # Check if adata.X is in COO format
    if not isinstance(X, scipy.sparse.coo_matrix):
        adata_X_coo = X.tocoo()
    else:
        adata_X_coo = X

    # Extract row indices, column indices, and data values
    row_indices = adata_X_coo.row
    col_indices = adata_X_coo.col
    values = adata_X_coo.data

    # Stack row and column indices
    indices = np.vstack((row_indices, col_indices)).T.astype(np.int64)

    # Get the shape of the dense matrix
    dense_shape = adata_X_coo.shape

    # Create a TensorFlow SparseTensor
    sparse_tensor = tf.sparse.SparseTensor(
        indices=indices,
        values=values.astype(np.float32),
        dense_shape=dense_shape
    )

    # Ensure the indices are sorted (required by TensorFlow)
    sparse_tensor = tf.sparse.reorder(sparse_tensor)
    return sparse_tensor
    # print(sparse_tensor)
# # Example usage
# if __name__ == "__main__":
#     # Sample SparseTensor adjacency matrix
#     adj_sparse = tf.SparseTensor(
#         indices=[[0, 1], [1, 2], [2, 0]],
#         values=[1.0, 2.0, 3.0],
#         dense_shape=[3, 3]
#     )
#
#     # Convert to NetworkX graph
#     G = sparse_tensor_to_networkx(adj_sparse, directed=False)
#
#     # Display the edges with weights
#     print("Edges with weights:")
#     for u, v, data in G.edges(data=True):
#         print(f"({u}, {v}, weight={data['weight']})")
#
