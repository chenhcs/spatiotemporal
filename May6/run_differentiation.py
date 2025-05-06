import numpy as np
import pandas as pd
from sergio import sergio
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt

df = pd.read_csv('../data_sets/De-noised_100G_4T_300cPerT_dynamics_10_DS5_small_direct-2_positive/bMat_cID10.tab', sep='\t', header=None, index_col=None)
bMat = df.values

n_sc = 300
sim = sergio(number_genes=58, number_bins = 4, number_sc = n_sc, noise_params = 0.3, decays=0.8, sampling_state = 1, noise_params_splice = 0.1, noise_type='dpd', dynamics=True, bifurcation_matrix= bMat, splice_ratio=1.5)
sim.build_graph(input_file_taregts ='../data_sets/De-noised_100G_4T_300cPerT_dynamics_10_DS5_small_direct-2_positive/Interaction_cID_10_lr.txt', input_file_regs='../data_sets/De-noised_100G_4T_300cPerT_dynamics_10_DS5_small_direct-2_positive/Regs_cID_10_lr.txt', shared_coop_state=2)
sim.simulate_dynamics()
exprU, exprS = sim.getExpressions_dynamics()
exprU_clean = np.concatenate(exprU, axis = 1)
exprS_clean = np.concatenate(exprS, axis = 1)

expr_clean_dif = exprU_clean + exprS_clean

expr_clean_dif = np.transpose(expr_clean_dif)
print(expr_clean_dif.shape)

cell_ids = [j for j in range(expr_clean_dif.shape[0])]
cell_types = [(cell_id // n_sc) for cell_id in cell_ids]
gene_ids = [j for j in range(len(expr_clean_dif[1]))]
adata = ad.AnnData(X=expr_clean_dif,
                    var={'Gene': gene_ids},
                    obs={'Cell Types': cell_types})

adata.write('dynamics_10_DS5_lr_small_direct-2_positive.h5ad')
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_neighbors=100)
sc.tl.umap(adata)
color = ["Cell Types"]
color.extend([str(i) for i in range(10)])
print(color)
sc.pl.umap(adata, color=color)
print(adata)
plt.savefig('umap_ds5_lr_small_direct-2_positive.png')
