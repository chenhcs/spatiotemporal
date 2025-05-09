# Updates
### May 8
- In spateotemporal_model_dynamic_global_percell_average_neighbor.py, predict receptor expression from averaged neighboring ligand expression.
- In spateotemporal_model_dynamic_global_percell_neighbor_attention.py, predict receptor expression from neighboring ligand expression, while adding attention weights to each neighboring cell.
- Updated infer_pseudotime_metacell_01norm_seperate_neighbor.py. For ligand expression array, instead of averaging all the neighbors, ligand expression of each neighboring cell is saved.

### May 6
- In infer_pseudotime_metacell_01norm.py, added the function to merge cells into meta cells. Change z score normalization to min max normalization.
- In spateotemporal_model_dynamic_global_percell.py, updated the way to embed expression values, added l1 normalization of regulatory networks.
- In spateotemporal_model_dynamic_global_percell.py, the current model learns a global regulatory netowork for the whole dataset from gene token embeddings only, while learns a cell specific regulatory network from gene token + expression embeddings.

# spatiotemporal

reconstruct pseudotime - learn_pseudotime.py

sample cells from pseudotime trajectory - prepare_data.py

model - model.py

GAT model - GNN_model.py

spatial model (GAT) pretraining - spatial_model_pretrain.py

oversampling method - GraphSMOTE.py

functional tools - utils.py

data analysis and compute attention flow - attention_flow.py

## Simulation data
The simulated spatial data is save as `simulation/st_simu.h5ad`. The cell types are saved in adata.obs['Cell Types'], while the spatial coordinates of the cells are saved in adata.obsm['spatial']. If `NaN` in adata.obs['Cell Tags'], then the spot is empty, otherwise the spot contains a cell.
