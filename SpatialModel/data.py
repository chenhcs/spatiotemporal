


import scanpy as sc
import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm


# Distance threshold (e.g., 100 units)
threshold_distance = 2

# Load data
adata = sc.read_h5ad('data/simulation/dynamics_10_DS5_lr_small_direct_spatial-4.h5ad')

# Normalize the data
expmat = adata.X.copy()
# If expmat is sparse, convert it to a dense array.
if hasattr(expmat, "toarray"):
    expmat = expmat.toarray()

for i in range(expmat.shape[1]):
    exparray = expmat[:, i]
    max_val = np.max(exparray)
    min_val = np.min(exparray)
    # mean = exparray.mean()
    # std = exparray.std()
    # Normalize each column with min-max scaling.
    expmat[:, i] = (expmat[:, i] - min_val) / (max_val - min_val + 1e-8)
    # expmat[:, i] = (exparray - mean)/std

adata.X = expmat
print(expmat)

# Load the ligand genes, receptor genes, and transcription factors (tf) genes.
ligands = adata.var['Gene'][-4:].to_numpy()
receptors = adata.var['Gene'][1:7].to_numpy()
tfs = adata.var['Gene'][45:51].to_numpy()
# Create a mapping from gene names to their column indices.
gene_to_idx = {gene: idx for idx, gene in enumerate(adata.var['Gene'])}
ligand_indices = [gene_to_idx[gene] for gene in ligands]
receptor_indices = [gene_to_idx[gene] for gene in receptors]
tf_indices = [gene_to_idx[gene] for gene in tfs]

# Load the cell types and spatial coordinates.
cell_types = np.sort(adata.obs['Cell Types'].unique())
spatial_coords = adata.obs['Cell To XY']
spatial_coords = spatial_coords.apply(lambda x: [int(i) for i in x.split('_')])
spatial_coords = np.vstack(spatial_coords)

# Precompute a KDTree for each sender cell type (used for k-NN neighbor selection).
sender_indices_by_type = {}
sender_trees = {}
for ct in cell_types:
    inds = np.where(adata.obs['Cell Types'] == ct)[0]
    sender_indices_by_type[ct] = inds
    if inds.size > 0:
        sender_trees[ct] = KDTree(spatial_coords[inds])
    else:
        sender_trees[ct] = None

# Now, extract features for each sender–receiver cell-type pair.
max_neighbors = 5
for b_idx, b in enumerate(tqdm(cell_types, desc="Receiver cell types")):
    # Get indices for cells of the receiver cell type b.
    receiver_indices = np.where(adata.obs['Cell Types'] == b)[0]

    for a_idx, a in enumerate(tqdm(cell_types, desc=f"  Sender cell types for {b}", leave=False)):
        X_s = []
        y_s = []
        count = 0
        # For each receiver cell, extract features based on neighboring sender cells of type a.
        for idx_c in receiver_indices:
            center_coord = spatial_coords[idx_c].reshape(1, -1)
            # Query k-nearest neighbors from the KDTree for sender cell type 'a'.
            if sender_trees[a] is not None:
                tree_sender = sender_trees[a]
                dist, knn_indices = tree_sender.query(center_coord, k=max_neighbors)
                knn_indices = knn_indices.flatten()
                dist = dist.flatten()
                # Filter neighbors by distance threshold.
                valid_mask = dist <= threshold_distance
                neighbor_inds = sender_indices_by_type[a][knn_indices[valid_mask]]
                if np.sum(valid_mask)>3:
                    count+=1
            else:
                pass

            # Compute the local ligand context from the k-NN (if any valid neighbors exist).
            if len(neighbor_inds) > 0:
                # Extract the submatrix for rows=neighbors and columns=ligand_indices.
                local_ligand_matrix = expmat[np.ix_(neighbor_inds, ligand_indices)]
                local_lig_vec = np.mean(local_ligand_matrix, axis=0)
                # Get the local receptor expressions and TF expressions for the receiver cell.
                local_recpt_vec = expmat[idx_c, receptor_indices]
                local_tfs = expmat[idx_c, tf_indices]
                # Compute the outer product (ligand * receptor) and flatten it.
                x = np.outer(local_lig_vec, local_recpt_vec).flatten()
                X_s.append(x)
                y_s.append(local_tfs)

            else:
                pass
                # local_lig_vec = np.zeros(len(ligands))
                # print("No neighbors found between {} and {}".format(b_idx, a_idx))

        print(count)
        if count < 50:
            print('No possible interactions between {} and {}'.format(a_idx, b_idx))
        X_s = np.array(X_s)
        y_s = np.array(y_s)
        # Save the features for this sender–receiver combination.
        np.save(f'data/LR_interaction_data/X_{a_idx}_{b_idx}.npy', X_s)
        np.save(f'data/LR_interaction_data/y_{a_idx}_{b_idx}.npy', y_s)
