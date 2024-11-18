from simulation import h5ad
from sklearn.preprocessing import LabelEncoder
import numpy as np
from spatial_model import GraphAttentionNetwork


def load_data(path):
    """Load and preprocess the data."""
    adata = h5ad.open_h5ad(path)
    # Filter out cells without cell types
    adata = adata[adata.obs['Cell Types'].notna()]
    coordinates = adata.obsm['spatial']
    return adata, coordinates

def encode_cell_types(adata):
    """Encode cell types into numerical labels."""
    # le = LabelEncoder()
    # cell_types = le.fit_transform(adata.obs['Cell Types'])
    idx_cell_types = {}
    cell_idx_types = {}
    for idx, cell_type in enumerate(np.unique(adata.obs['Cell Types'])):
        cell_idx_types[cell_type] = idx
        idx_cell_types[idx] = cell_type
    cell_types = adata.obs['Cell Types'].map(cell_idx_types)
    print(idx_cell_types.values())
    return cell_types, idx_cell_types.values()


