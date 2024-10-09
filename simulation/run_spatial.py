import scanpy as sc
import anndata as ad

from spatial import *
import matplotlib.pyplot as plt

def run_spatial(path, n_genes, n_bins, n_sc, T=0.5, lr=3):
    # T is the temperature to use
    # lr is the ratio of cells with receptors to ligands

    print("--------------------------------------")
    print("   Determining spatial coordinates    ")
    print("--------------------------------------")

    simulate_spatial_expression_data(path, n_genes, n_bins, n_sc, T, lr)

n_genes = 100
n_bins = 4
n_sc = 300
run_spatial("dynamics_10_DS5.h5ad", n_genes, n_bins, n_sc)
