# spatiotemporal

reconstruct pseudotime - learn_pseudotime.py

sample cells from pseudotime trajectory - prepare_data.py

model - model.py

## Simulation data
The simulated spatial data is save as `simulation/st_simu.h5ad`. The cell types are saved in adata.obs['Cell Types'], while the spatial coordinates of the cells are saved in adata.obsm['spatial']. If `NaN` in adata.obs['Cell Tags'], then the spot is empty, otherwise the spot contains a cell.
