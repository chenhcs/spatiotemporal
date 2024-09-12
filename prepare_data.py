import pyreadr
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.setrecursionlimit(10000)

def draw_path(adata, path, cnt, t):
    plt.clf()
    spatial = adata.obsm['X_umap']
    plt.scatter(spatial[:, 0], spatial[:, 1], color='grey', s=1)
    plt.scatter(spatial[path[0], 0], spatial[path[0], 1], color='red', s=3)
    for i in range(len(path) - 1):
        plt.plot([spatial[path[i], 0], spatial[path[i + 1], 0]], [spatial[path[i], 1], spatial[path[i + 1], 1]], color='red')
        plt.scatter(spatial[path[i + 1], 0], spatial[path[i + 1], 1], color='red', s=1)
    plt.savefig('fig/path_' + str(t) + '_' + str(cnt) + '_temporal.png')


all_lig_recp = set()
result = pyreadr.read_r('../data/LR_pairs_Jin_2020_mouse.rda')
lig2recp = {}
for i in range(len(result['LR_pairs_Jin_2020_mouse'])):
    for l in result['LR_pairs_Jin_2020_mouse']['ligand'][i].split(','):
        for r in result['LR_pairs_Jin_2020_mouse']['receptor'][i].split(','):
            if l in lig2recp:
                lig2recp[l].append(r)
                all_lig_recp.add(l)
                all_lig_recp.add(r)
                lig2recp[l] = list(set(lig2recp[l]))
            else:
                lig2recp[l] = [r]
                all_lig_recp.add(l)
                all_lig_recp.add(r)
print(len(all_lig_recp))
print(list(all_lig_recp)[:10])

result = pyreadr.read_r('../data/TF_PPR_KEGG_mouse.rda')
all_tfs = set([])
recp2TF = {}
all_targets = set()
for i in range(len(result['TF_PPR_KEGG_mouse'])):
    for r in result['TF_PPR_KEGG_mouse']['receptor'][i].split(','):
        for t in result['TF_PPR_KEGG_mouse']['tf'][i].split(','):
            if r in recp2TF:
                recp2TF[r].append(t)
                all_tfs.add(t)
                all_targets.add(t)
                recp2TF[r] = list(set(recp2TF[r]))
            else:
                recp2TF[r] = [t]
                all_tfs.add(t)
                all_targets.add(t)

TF2tar = {}
with open('../data/TF_TG_TRRUSTv2_RegNetwork_High_mouse.txt') as fr:
    tf = ''
    tar = []
    for line in fr:
        if '$' in line:
            if tf != '':
                TF2tar[tf] = tar
                tar = []
            tf = line.split('$')[1].split('\n')[0]
        else:
            for t in line.split('"'):
                if t[0].isalpha():
                    tar.append(t)
                    all_targets.add(t)
print(len(all_targets))
print(list(all_targets)[:10])

adata = ad.read_h5ad('../data/midbrain_pseudotime.h5ad')
sc.pp.filter_genes(adata, min_cells=100)
print(adata.obs['annotation'])
expmat = adata.X.toarray()
adata.X = expmat

print(adata.obs)

all_lig_recp = all_lig_recp.intersection(adata.var.index)
all_targets = all_targets.intersection(adata.var.index)
print(len(all_lig_recp))
print(len(all_targets))
all_lig_recp = list(all_lig_recp)
all_targets = list(all_targets)
all_genes = list(set(all_targets).union(all_lig_recp))

temporal_neighbors = {}
temporal_neighbors_l = {}
d = pdist(adata.obsm['X_umap'])
d2 = squareform(d)
print(d2.shape)
for i in range(len(d2)):
    nnidx = np.argsort(d2[i])
    temporal_neighbors[i] = nnidx[:6]
    temporal_neighbors_l[i] = nnidx[:11]
print("done")

expmat_neighbor = expmat.copy()
cnt_i = 0
for i in temporal_neighbors_l:
    cnt_i += 1
    #print(cnt_i, len(temporal_neighbors_l))
    for j in temporal_neighbors_l[i][1:]:
        expmat_neighbor[i, :] += expmat[j, :]

adata.X = expmat_neighbor
expmat = adata.X.copy()

'''
plt.clf()
plt.matshow(np.clip(expmat, 0, 50))
plt.savefig('fig/data.png')

a = a
'''
#exp_mean = np.mean(adata.X, axis=0)
#print(exp_mean.shape)
for i in range(expmat.shape[1]):
    exparray = expmat[:, i]
    std = np.std(exparray)
    #print(i, len(exparray[np.nonzero(exparray)]))
    if std > 0:
        expmat[:, i] = (expmat[:, i] - np.mean(exparray)) / std
#expmat = np.argsort(np.argsort(expmat, axis=1), axis=1)
#expmat_binary = np.zeros(expmat.shape)
#expmat_binary[np.where(expmat > 4200)] = 1

expmat_binary = expmat
adata.X = expmat_binary
print(adata.X)
print(np.mean(adata.X, axis=0))
adata.write('data/norm_data.h5ad')

for gene in ['Hmgb2', 'Dbi', 'Fabp7', 'Glis3', 'H2afv', 'Hes5', 'Ptn', 'Rspo1', 'Tnc', 'Tuba1b', 'Vim']:
    plt.clf()
    plt.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], c=np.squeeze(adata[:, [gene]].X.toarray()), s=1, alpha=0.5, cmap='Reds')
    plt.savefig('fig/' + gene + '.pdf')

plt.clf()
sort_idx = np.argsort(adata.obs['dpt_pseudotime'])
expmat_binary = expmat_binary[sort_idx]
sum = [np.mean(np.multiply(np.arange(expmat_binary.shape[0]), expmat_binary[:, i])) for i in range(expmat_binary.shape[1])]
sort_idx = np.argsort(sum)
expmat_binary = expmat_binary[:, sort_idx]
#plt.matshow(expmat_binary)
#plt.savefig('fig/data.png')

plt.clf()
sns.clustermap(expmat_binary, row_cluster=False)
plt.savefig('fig/data.png')
#a = a
#adata.uns['iroot'] = np.flatnonzero(adata.obs['leiden']  == '0')[2]
#sc.tl.dpt(adata)
#print(adata)
#adata = adata[adata.obs['timepoint']=='E13.5']
#print(adata)

from scipy import stats

temporal_neighbors = {}
temporal_neighbors_l = {}
d = pdist(adata.obsm['X_umap'])
d2 = squareform(d)
print(d2.shape)
for i in range(len(d2)):
    nnidx = np.argsort(d2[i])
    temporal_neighbors[i] = nnidx[:6]
    temporal_neighbors_l[i] = nnidx[:11]
print("done")

corfall = []
corball = []
exp_array = adata.X
for i in temporal_neighbors_l:
    j = temporal_neighbors_l[i][1]
    k = np.random.randint(len(temporal_neighbors_l))
    cor1 = stats.spearmanr(exp_array[i], exp_array[j])
    cor2 = stats.spearmanr(exp_array[i], exp_array[k])
    corfall.append(cor1.statistic)
    corball.append(cor2.statistic)
    print(cor1.statistic, cor2.statistic, np.mean(corfall), np.mean(corball))

idx_all = np.arange(len(adata))
np.random.shuffle(idx_all)
adata_ = adata.copy()
for t in range(2):
    if t == 0:
        adata = adata_[idx_all[:int(len(idx_all) * 0.5)]]
    else:
        adata = adata_[idx_all[int(len(idx_all) * 0.5):]]

    spatial_neighbors = {}
    spatial_neighbors_l = {}
    d = pdist(adata.obsm['spatial'])
    d2 = squareform(d)
    print(d2.shape)
    for i in range(len(d2)):
        nnidx = np.argsort(d2[i])
        spatial_neighbors[i] = nnidx[:6]
        spatial_neighbors_l[i] = nnidx[:51]
    print("done")

    temporal_neighbors = {}
    temporal_neighbors_l = {}
    d = pdist(adata.obsm['X_umap'])
    d2 = squareform(d)
    print(d2.shape)
    for i in range(len(d2)):
        nnidx = np.argsort(d2[i])
        temporal_neighbors[i] = nnidx[:6]
        temporal_neighbors_l[i] = nnidx[:11]
    print("done")

    adata_neighbor = adata.copy()
    cnt_i = 0
    for i in spatial_neighbors:
        cnt_i += 1
        print(cnt_i, len(spatial_neighbors))
        for j in spatial_neighbors[i][1:]:
            adata_neighbor.X[i, :] += adata.X[j, :]

    spatialtemporal_neighbors = {}
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
            spatialtemporal_neighbors[i] = [nnidx[0][0], nnidx[1][0]]
        elif len(nnidx) > 0:
            spatialtemporal_neighbors[i] = [nnidx[0][0]]
        else:
            spatialtemporal_neighbors[i] = []
    print("done")

    all_paths = []
    cnt = 0
    len_path = 30
    for repeat in range(10):
        for i in spatialtemporal_neighbors:
            path = [i]
            cur = i
            for k in range(len_path):
                nnidx = spatialtemporal_neighbors[cur]
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
                if cnt < 20:
                    draw_path(adata, path, cnt, t)
                    cnt += 1
        print(len(all_paths))

    context_array = []
    target_array = []
    label_array = []
    for p in all_paths:
        if len(context_array) % 100 == 0:
            print(len(context_array))
        #context = []
        #target = []
        #for c in p:
        context = np.squeeze(adata_neighbor[p, all_lig_recp].X.toarray())
        target = np.squeeze(adata[p, all_targets].X.toarray())
        context_array.append(context)
        target_array.append(target[:len_path])
        label_array.append(target[1:])
    context_array = np.array(context_array)
    target_array = np.array(target_array)
    label_array = np.array(label_array)
    print(context_array.shape)
    print(target_array.shape)
    print(label_array.shape)
    print(context_array[0])
    print(target_array[0])
    print(label_array[0])
    if t == 0:
        np.save('data/context_array_train_cont_.npy', context_array)
        np.save('data/target_array_train_cont_.npy', target_array)
        np.save('data/label_array_train_cont_.npy', label_array)
        np.save('data/all_paths_train.npy', np.array(all_paths))
        adata.write('data/adata_norm_train.h5ad')
        np.save('data/all_lig_recp.npy', np.array(all_lig_recp))
        np.save('data/all_targets.npy', np.array(all_targets))
    else:
        np.save('data/context_array_test_cont_.npy', context_array)
        np.save('data/target_array_test_cont_.npy', target_array)
        np.save('data/label_array_test_cont_.npy', label_array)
        np.save('data/all_paths_test.npy', np.array(all_paths))
        adata.write('data/adata_norm_test.h5ad')
