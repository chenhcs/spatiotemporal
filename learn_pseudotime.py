import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15,5)

adata = ad.read_h5ad('Dorsal_midbrain_cell_bin.h5ad')
adata1 = a[a.obs['annotation'] == 'GlioB']
adata2 = a[a.obs['annotation'] == 'RGC']
adata3 = a[a.obs['annotation'] == 'NeuB']
adata = ad.concat([adata1, adata2, adata3])
sc.pp.neighbors(adata, n_neighbors=45, use_rep='X_umap')
sc.tl.draw_graph(adata)
adata.uns['iroot'] = np.flatnonzero(adata.obs['annotation'] == 'RGC')[0]
sc.tl.dpt(adata)
sc.pl.umap(adata, color='dpt_pseudotime')
plt.savefig('umap.png')

#adata = ad.read_h5ad('data/midbrain_pseudotime.h5ad')
#sc.pl.umap(adata, color='Hes1')
x, y, c = [], [], []
a1 = adata[adata.obs['Time point'] == 'E16.5']
for idx in range(len(a1)):
    x.append(a1.obsm['spatial'][idx][0])
    y.append(a1.obsm['spatial'][idx][1])
    c.append(a1.obs['dpt_pseudotime'][idx])
plt.scatter(x, y, c=c, s=1)
plt.savefig('fig/midbrain_e16.png')
