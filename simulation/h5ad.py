
import anndata as ad

##############################################
# h5ad Read and Write
##############################################

def open_h5ad(path):
    adata = ad.read_h5ad(path)
    return adata

def gene_expr_to_h5ad(gene_expr, ad_path, n_sc):
    gene_ids = [i for i in range(len(gene_expr))]
    cell_ids = [j for j in range(len(gene_expr[0]))]
    cell_types = [(cell_id // n_sc) for cell_id in cell_ids]
    adata = ad.AnnData(X=gene_expr.T,
                        var={'Gene': gene_ids},
                        obs={'Cell Type': cell_types})
    adata.write_h5ad(ad_path)
    return adata
