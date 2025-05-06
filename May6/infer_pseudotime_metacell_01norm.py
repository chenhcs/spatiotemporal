import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sys import argv
import pandas as pd

def draw_path(adata, path, cnt, t):
    plt.clf()
    spatial = adata.obsm['X_umap']
    plt.scatter(spatial[:, 0], spatial[:, 1], color='grey', s=1)
    plt.scatter(spatial[path[0], 0], spatial[path[0], 1], color='red', s=10)
    for i in range(len(path) - 1):
        plt.plot([spatial[path[i], 0], spatial[path[i + 1], 0]], [spatial[path[i], 1], spatial[path[i + 1], 1]], color='red')
        plt.scatter(spatial[path[i + 1], 0], spatial[path[i + 1], 1], color='red', s=1)
    plt.savefig('fig/path_' + str(t) + '_' + str(cnt) + '_temporal.png')


def run_umap(adata, n_neighbors=100, min_dist=0.01):
    print("--------------------------------------")
    print("         Creating UMAP graph          ")
    print("--------------------------------------")

    #adata = ad.AnnData(X=adata.X.T, obs=adata.var, var=adata.obs)
    print(f"Gene Expression Matrix: {adata.X.shape[0]} Single Cells, {adata.X.shape[1]} Genes")

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    color = ["Cell Types"]
    color.extend([str(i) for i in range(len(adata.var.index))])
    print(color)
    sc.pl.umap(adata, color=color)
    return adata

def merge_metacell(adata):
    cluster_labels = adata.obs["clusters"]

    X = adata.X
    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    expr_df["cluster"] = cluster_labels.values
    cluster_sum = expr_df.groupby("cluster").mean()

    umap = adata.obsm["X_umap"]
    umap_df = pd.DataFrame(umap, index=adata.obs_names, columns=["UMAP1", "UMAP2"])
    umap_df["cluster"] = cluster_labels.values
    mean_umap = umap_df.groupby("cluster")[["UMAP1", "UMAP2"]].mean()

    adata_meta = sc.AnnData(
        X=cluster_sum.values,
        obs=pd.DataFrame(index=cluster_sum.index),
        var=adata.var.copy()
    )

    adata_meta.obsm["X_umap"] = mean_umap.values

    sc.pp.neighbors(adata_meta, n_neighbors=10, use_rep='X_umap')

    return adata_meta

#################################Run umap for adata
#adata = ad.read_h5ad('dynamics_10_DS5_lr_01.h5ad')
adata = ad.read_h5ad('../sc_simu-2_positive.h5ad')
adata_ori = adata.copy()
adata = adata[:, :-6] #ignore ligand gene expression when infering pseudotime
print(adata)
print(adata.X)
print(adata.obs)
print(set(adata.obs['Cell Types']))

adata = run_umap(adata) #run umap on genes excluding ligands

sc.tl.leiden(adata, key_added="clusters", resolution=30)

###################################################
adata_ori.obs = adata.obs
adata_ori.uns = adata.uns
adata_ori.obsm = adata.obsm
adata_ori.obsp = adata.obsp

###################################################
#find the nearest 5 neighbors of each cell
spatial_neighbors = {}
d = pdist(adata_ori.obsm['spatial'])
d2 = squareform(d)
print(d2.shape)
for i in range(len(d2)):
    nnidx = np.argsort(d2[i])
    spatial_neighbors[i] = nnidx[:6]

#average expression of neighbors
adata_neighbor = adata_ori.copy()
for i in spatial_neighbors:
    for j in spatial_neighbors[i][1:]:
        adata_neighbor.X[i, :] += adata_ori.X[j, :]

adata_neighbor.obs = adata.obs
adata_neighbor.uns = adata.uns
adata_neighbor.obsm = adata.obsm
adata_neighbor.obsp = adata.obsp
###################################################

X1 = adata_ori[:, ['45', '46', '47', '48', '49', '50']].X
X2 = adata_neighbor[:, ['58', '59', '60', '61', '62', '63']].X

lr_products = X1[:, :, None] * X2[:, None, :]

X_new = lr_products.reshape(adata.n_obs, -1)

# Optionally generate variable names
var_names = [str(i) for i in range(len(['45', '46', '47', '48', '49', '50']) * len(['58', '59', '60', '61', '62', '63']))]

# Create a new AnnData object
adata_lr = ad.AnnData(X=X_new)
adata_lr.var_names = var_names
adata_lr.obs = adata.obs
adata_lr.uns = adata.uns
adata_lr.obsm = adata.obsm
adata_lr.obsp = adata.obsp
print(adata_lr.var)
###################################################

adata = merge_metacell(adata_ori)
adata_neighbor = merge_metacell(adata_neighbor)
adata_lr = merge_metacell(adata_lr)

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.pp.normalize_total(adata_neighbor)
sc.pp.log1p(adata_neighbor)

sc.pp.normalize_total(adata_lr)
sc.pp.log1p(adata_lr)

print(adata_lr)

print(adata)
idx = np.argsort(adata.obsm['X_umap'][:, 0])[0]
print(idx)
adata.uns['iroot'] = idx
sc.tl.dpt(adata)
color = ["dpt_pseudotime"]
color.extend(['45', '1', '7', '8', '11', '12', '46', '2', '15', '16', '47', '3', '20', '21', '48', '4', '28', '29', '49', '5', '36', '50', '6', '40', '58', '59', '60', '61', '62', '63'])
#print(color)
#sc.pl.umap(adata, color=color)
#print(adata)
#plt.savefig('umap.png')

#z normalization
for g in range(len(adata.var)):
    exparray = adata.X[:, g]
    std = np.std(exparray)
    if std > 0:
        adata.X[:, g] = (adata.X[:, g] - np.min(exparray)) / (np.max(exparray) - np.min(exparray)) #std
print(np.mean(adata.X, axis=0))
print(np.min(adata.X, axis=0))

for g in range(len(adata_neighbor.var)):
    exparray = adata_neighbor.X[:, g]
    std = np.std(exparray)
    if std > 0:
        adata_neighbor.X[:, g] = (adata_neighbor.X[:, g] - np.min(exparray)) / (np.max(exparray) - np.min(exparray)) #std
print(np.mean(adata_neighbor.X, axis=0))
print(np.min(adata_neighbor.X, axis=0))

expmat = adata_lr.X.copy()
for g in range(len(adata_lr.var)):
    exparray = expmat[:, g]
    std = np.std(exparray)
    if std > 0:
        expmat[:, g] = (exparray - np.min(exparray)) / (np.max(exparray) - np.min(exparray)) #np.max(exparray) #std
adata_lr.X = expmat
print(np.mean(adata_lr.X, axis=0))
print(np.min(adata_lr.X, axis=0))

plt.clf()
sc.pl.umap(adata, color=color)
print(adata)
plt.savefig('umap_nz.png')

plt.clf()
sc.pl.umap(adata_neighbor, color=['58', '59', '60', '61', '62', '63'])
print(adata_neighbor)
plt.savefig('umap_nz_neighbor.png')

plt.clf()
sc.pl.umap(adata_lr, color=[str(i) for i in range(len(adata_lr.var))])
print(adata_lr)
plt.savefig('umap_lr_pair.png')

from scipy import stats

all_ligand = ['58', '59', '60', '61', '62', '63']
all_recep = ['45', '46', '47', '48', '49', '50']
all_tf = ['1', '2', '3', '4', '5', '6']
#all_tf = ['62', '77', '82', '97']
all_targets = list(str(i) for i in np.arange(64) if str(i) not in (all_ligand + all_recep + all_tf)) #
all_genes = list(set(all_targets).union(all_ligand).union(all_recep).union(all_tf))

#corfall = []
#corball = []
#exp_array = adata.X
#for i in temporal_neighbors_l:
#    j = temporal_neighbors_l[i][1]
#    k = np.random.randint(len(temporal_neighbors_l))
#    cor1 = stats.spearmanr(exp_array[i], exp_array[j])
#    cor2 = stats.spearmanr(exp_array[i], exp_array[k])
#    corfall.append(cor1.statistic)
#    corball.append(cor2.statistic)
#    print(cor1.statistic, cor2.statistic, np.mean(corfall), np.mean(corball))

#plt.clf()
#plt.figure(figsize=(7,6))
idx_all = np.arange(len(adata))
np.random.shuffle(idx_all)
#sample cell path from embeddings, two iterations for training and testing
for t in range(2):
    #identify neighbors from embedding
    temporal_neighbors_l = {}
    temporal_decendent = {}
    d = pdist(adata.obsm['X_umap'])
    d2 = squareform(d)
    print(d2.shape)
    for i in range(len(d2)):
        nnidx = np.argsort(d2[i])
        temporal_neighbors_l[i] = nnidx[:5]
    print("done")

    for i in range(len(d2)):
        nnidx = np.argsort(d2[i])
        temporal_decendent[i] = []
        for j in range(1, 11):
            if adata.obs['dpt_pseudotime'][nnidx[j]] > adata.obs['dpt_pseudotime'][i]:
                temporal_decendent[i].append(nnidx[j])
                if len(temporal_decendent[i]) == 2:
                    break

    #adata_neighbor = adata.copy()
    #cnt_i = 0
    #for i in spatial_neighbors:
    #    cnt_i += 1
    #    print(cnt_i, len(spatial_neighbors))
    #    for j in spatial_neighbors[i][1:]:
    #        adata_neighbor.X[i, :] += adata.X[j, :]

    #plt.clf()
    #sc.pl.umap(adata_neighbor, color=['58', '59', '60', '61', '62', '63'])
    #plt.savefig('umap_ds5_neighbor.png')

    temporal_neighbors = {}
    cnt_i = 0
    for i in temporal_neighbors_l:
        cnt_i += 1
        print(cnt_i, len(temporal_neighbors_l))
        nnidx = []
        for j in temporal_neighbors_l[i]:
            if adata.obs['dpt_pseudotime'][j] > adata.obs['dpt_pseudotime'][i]: # and j in temporal_neighbors_l[i]:
                nnidx.append((j, adata.obs['dpt_pseudotime'][j]))
        nnidx = sorted(nnidx, key=lambda x: -x[1])
        if len(nnidx) > 1:
            temporal_neighbors[i] = [nnidx[0][0], nnidx[1][0]]
        elif len(nnidx) > 0:
            temporal_neighbors[i] = [nnidx[0][0]]
        else:
            temporal_neighbors[i] = temporal_decendent[i]
    print("done")

    all_paths = []
    cnt = 0
    len_path = 3
    for repeat in range(10):
        for i in temporal_neighbors:
            path = [i]
            cur = i
            for k in range(len_path):
                nnidx = temporal_neighbors[cur]
                if len(nnidx) == 0:
                    break
                next = np.random.choice(nnidx, size=1)[0]
                path.append(next)
                cur = next
            #print(len(path), path)
            if len(path) == len_path + 1:
                all_paths.append(path)
                print(i)
                print(path)
                rnd = np.random.randint(500)
                if cnt < 10:
                    draw_path(adata, path, cnt, t)
                    cnt += 1
        print(len(all_paths))

    ligand_array = []
    recep_array = []
    tf_array = []
    target_array = []
    label_array = []
    lr_pair_array = []
    for p in all_paths:
        if len(tf_array) % 100 == 0:
            print(len(tf_array))
        ligand = np.squeeze(adata_neighbor[p, all_ligand].X.toarray())
        receptor = np.squeeze(adata[p, all_recep].X.toarray())
        #ligand = np.hstack((ligand, context))
        tf = np.squeeze(adata[p, all_tf].X.toarray())
        target = np.squeeze(adata[p, all_targets].X.toarray())
        lr_pair = np.squeeze(adata_lr[p, :].X.toarray())
        ligand_array.append(ligand)
        recep_array.append(receptor)
        tf_array.append(tf)
        target_array.append(target)
        lr_pair_array.append(lr_pair)

    ligand_array = np.array(ligand_array)
    recep_array = np.array(recep_array)
    tf_array = np.array(tf_array)
    target_array = np.array(target_array)
    lr_pair_array = np.array(lr_pair_array)

    print(ligand_array.shape)
    print(tf_array.shape)
    print(target_array.shape)
    print(recep_array.shape)
    print(lr_pair_array.shape)

    #lable is input target gene shifted by 1 time point
    label_array = target_array[:, 1:, :]
    target_array = target_array[:, :len_path, :]
    print(ligand_array[:1,0,:], 'ligand')
    print(tf_array[:1,0,:], 'tf')
    print(target_array[0])
    plt.clf()
    index_all = np.arange(len(ligand_array))
    np.random.shuffle(index_all)
    plt.imshow(np.hstack((ligand_array[index_all[:200],0,:], tf_array[index_all[:200],0,:])))
    plt.savefig('fig/tf_array' + str(t) + '.png')
    if t == 0:
        np.save('data_triple/recep_array_train.npy', recep_array)
        np.save('data_triple/ligand_array_train.npy', ligand_array)
        np.save('data_triple/tf_array_train.npy', tf_array)
        np.save('data_triple/target_array_train.npy', target_array)
        np.save('data_triple/label_array_train.npy', label_array)
        np.save('data_triple/lr_pair_array_train.npy', lr_pair_array)
        np.save('data_triple/all_paths_train.npy', np.array(all_paths))
        adata.write('data_triple/adata_norm_train.h5ad')
        #np.save('data/all_lig_recp.npy', np.array(all_lig_recp))
        #np.save('data/all_targets.npy', np.array(all_targets))
    else:
        np.save('data_triple/recep_array_test.npy', recep_array)
        np.save('data_triple/ligand_array_test.npy', ligand_array)
        np.save('data_triple/tf_array_test.npy', tf_array)
        np.save('data_triple/target_array_test.npy', target_array)
        np.save('data_triple/label_array_test.npy', label_array)
        np.save('data_triple/lr_pair_array_test.npy', lr_pair_array)
        np.save('data_triple/all_paths_test.npy', np.array(all_paths))
        adata.write('data_triple/adata_norm_test.h5ad')
