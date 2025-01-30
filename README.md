# spatiotemporal

reconstruct pseudotime - learn_pseudotime.py

sample cells from pseudotime trajectory - prepare_data.py

model - model.py

GAT model - GNN_model.py

spatial model (GAT) pretraining - spatial_model_pretrain.py

oversampling method - GraphSMOTE.py

Functional tools - utils.py

Tools for data analysis and compute attention flow - attention_flow.py

## Simulation data
The simulated spatial data is save as `simulation/st_simu.h5ad`. The cell types are saved in adata.obs['Cell Types'], while the spatial coordinates of the cells are saved in adata.obsm['spatial']. If `NaN` in adata.obs['Cell Tags'], then the spot is empty, otherwise the spot contains a cell.
