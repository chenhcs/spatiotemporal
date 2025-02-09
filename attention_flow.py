"""
@Author: Hongliang Zhou
@Description: Providing tools for the visualization of the attention weights and delauny graph.
"""

import pip
import math
import os
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import entropy
from sklearn.preprocessing import normalize
from tqdm import tqdm


# --------------------------------------------------------------------
# If you have utility install checks:
def install_if_not_exist(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])

install_if_not_exist('seaborn')
install_if_not_exist('tqdm')
# --------------------------------------------------------------------

from GNN_model import GATConv

def compute_and_visualize_entropy(adj_mat: tf.sparse.SparseTensor,
                                  edge_weights,
                                  output_dir,
                                  num_bins=50):
    """
    Computes and visualizes the entropy of each node's attention weights
    over its neighbors for each layer and head.

    Parameters:
    adj_mat: tf.sparse.SparseTensor representing the adjacency matrix.
    edge_weights: list of lists of tensors, where edge_weights[layer][head]
                  is a tensor representing the attention weights of that head in that layer.
    num_bins: int, number of bins for the histogram.

    Returns:
    None (plots histograms)
    """
    edge_index = adj_mat.indices.numpy().T  # shape: [2, num_edges]
    num_nodes = adj_mat.shape[0]
    num_layers = len(edge_weights)
    num_heads = len(edge_weights[0])

    # Build neighbor list
    neighbor_dict = {i: [] for i in range(num_nodes)}
    for idx in range(edge_index.shape[1]):
        src = int(edge_index[0, idx])
        dst = int(edge_index[1, idx])
        neighbor_dict[src].append((dst, idx))

    entropies = {}

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            head_weights = edge_weights[layer_idx][head_idx].numpy()  # shape [E]
            node_entropies = []
            max_entropies = []
            normalized_entropies = []

            for node in range(num_nodes):
                neighbors = neighbor_dict[node]
                degrees = len(neighbors)
                if degrees == 0:
                    continue

                # Extract attention for edges that come from 'node'
                attn_weights = np.array([head_weights[idx] for (_, idx) in neighbors])
                attn_sum = attn_weights.sum()
                if attn_sum == 0:
                    # If sum is zero (extremely rare), skip or fix
                    continue

                attn_weights /= attn_sum  # ensure sum=1
                node_entropy = entropy(attn_weights, base=np.e)
                node_entropies.append(node_entropy)

                max_entropy = np.log(degrees)
                max_entropies.append(max_entropy)
                normalized_entropy = node_entropy / max_entropy if max_entropy > 0 else 0
                normalized_entropies.append(normalized_entropy)

            key = f"Layer {layer_idx + 1}, Head {head_idx + 1}"
            entropies[key] = {
                'entropy': node_entropies,
                'max_entropy': max_entropies,
                'normalized_entropy': normalized_entropies
            }

    sns.set(style="whitegrid")
    num_plots = num_layers * num_heads
    cols = min(num_heads, 4)
    rows = (num_plots + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 4 * rows))
    plot_idx = 1

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            key = f"Layer {layer_idx + 1}, Head {head_idx + 1}"
            data = entropies[key]['entropy']
            max_data = entropies[key]['max_entropy']

            plt.subplot(rows, cols, plot_idx)
            sns.histplot(data, bins=num_bins, kde=True, color='blue', edgecolor='black', label='Entropy Dist')
            sns.histplot(max_data, bins=num_bins, kde=True, color='red', edgecolor='yellow', label='Uniform Dist Entropy')
            plt.title(key)
            plt.xlabel('Entropy')
            plt.ylabel('Frequency')
            plt.legend()
            plot_idx += 1

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'attention_graph_entropy_distribution.pdf'), format='pdf')
    plt.show()


def plot_pairwise_attention_weights(adj_mat, edge_weights, node_types, class_names, threshold=None):
    """
    Computes and visualizes the average attentional weights between each pair of classes
    for each layer and head.

    Parameters:
    adj_mat: tf.sparse.SparseTensor (adjacency)
    edge_weights: list of lists of tensors, shape [num_layers][num_heads], each [E]
    node_types: tf.Tensor of int
    class_names: list of str
    threshold: float, optional
    """
    adj_indices = adj_mat.indices.numpy()  # [num_edges, 2]
    edges_source = adj_indices[:, 0]
    edges_target = adj_indices[:, 1]

    node_types_array = np.array(node_types)
    node_types_source = node_types_array[edges_source]
    node_types_target = node_types_array[edges_target]

    num_classes = len(class_names)
    num_layers = len(edge_weights)
    num_heads = len(edge_weights[0]) if num_layers > 0 else 0

    class_attention_avg_all = []
    class_attention_counts_all = []

    for layer in range(num_layers):
        class_attention_avg_layer = []
        class_attention_counts_layer = []
        for head in range(num_heads):
            attention_vals = edge_weights[layer][head].numpy()  # [E]

            class_attention_sum = np.zeros((num_classes, num_classes))
            class_pair_counts = np.zeros((num_classes, num_classes))
            class_attention_counts = np.zeros((num_classes, num_classes))

            # sum up
            np.add.at(class_attention_sum, (node_types_source, node_types_target), attention_vals)
            np.add.at(class_pair_counts, (node_types_source, node_types_target), 1)

            if threshold is not None:
                mask = attention_vals > threshold
                high_c1 = node_types_source[mask]
                high_c2 = node_types_target[mask]
                np.add.at(class_attention_counts, (high_c1, high_c2), 1)

            with np.errstate(divide='ignore', invalid='ignore'):
                class_attention_avg = np.divide(class_attention_sum, class_pair_counts)
                class_attention_avg[np.isnan(class_attention_avg)] = 0

            class_attention_avg_layer.append(class_attention_avg)
            class_attention_counts_layer.append(class_attention_counts)

            # Plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(class_attention_avg, annot=True, fmt='.4f', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Target Class')
            plt.ylabel('Source Class')
            plt.title(f'Layer {layer}, Head {head}: Avg Pairwise Attention')
            plt.show()

            if threshold is not None:
                plt.figure(figsize=(10, 8))
                sns.heatmap(class_attention_counts, annot=True, fmt='d', cmap='Reds',
                            xticklabels=class_names, yticklabels=class_names)
                plt.xlabel('Target Class')
                plt.ylabel('Source Class')
                plt.title(f'Layer {layer}, Head {head}: # of High Attention Edges (>{threshold})')
                plt.show()

        class_attention_avg_all.append(class_attention_avg_layer)
        class_attention_counts_all.append(class_attention_counts_layer)

    return class_attention_avg_all, class_attention_counts_all


def plot_confusion_matrix(cm, class_names):
    conf_matrix = cm.numpy()
    plt.figure()
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,  # <--
        yticklabels=class_names   # <--
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def compute_attention_score(Edge_index, node_types, class_names, Attention_weights,  degree_normalization=True):
    """
    Computes the attention score from one cell type to others using max-flow,
    normalized by the number of possible connections from source node type i
    in the first layer to target node type j in the last layer.
    """
    import networkx as nx

    num_layers = len(Attention_weights)
    num_heads = len(Attention_weights[0])
    num_nodes = node_types.shape[0]
    num_cell_types = len(class_names)

    Edge_index_np = Edge_index.numpy()
    node_types_np = np.array(node_types)

    M_list_l1 = []
    M_list_l2 = []

    # Count matrix
    C = np.zeros((num_cell_types, num_cell_types))
    for i in range(num_cell_types):
        nodes_i = np.where(node_types_np == i)[0]
        for j in range(num_cell_types):
            nodes_j = np.where(node_types_np == j)[0]
            C[i, j] = len(nodes_i) * len(nodes_j)

    for h in range(num_heads):
        # Build flow network for head h
        G = nx.DiGraph()

        # add nodes
        for v in range(num_nodes):
            for l in range(num_layers + 1):
                G.add_node((v, l))

        # add source/sink
        source_nodes = {}
        sink_nodes = {}
        for i in range(num_cell_types):
            s_node = f"S_{i}"
            t_node = f"T_{i}"
            source_nodes[i] = s_node
            sink_nodes[i] = t_node
            G.add_node(s_node)
            G.add_node(t_node)

        # connect sources to first layer
        for i in range(num_cell_types):
            s_node = source_nodes[i]
            nodes_of_type = np.where(node_types_np == i)[0]
            for v in nodes_of_type:
                G.add_edge(s_node, (v, 0), capacity=math.inf)

        # connect last layer to sinks
        for i in range(num_cell_types):
            t_node = sink_nodes[i]
            nodes_of_type = np.where(node_types_np == i)[0]
            for v in nodes_of_type:
                G.add_edge((v, num_layers), t_node, capacity=math.inf)

        # add edges with capacity from Attention_weights
        for l in range(num_layers):
            head_weights = Attention_weights[l][h].numpy()  # shape [E]
            for e_idx, (u, v) in enumerate(Edge_index_np):
                cap = head_weights[e_idx]
                if cap > 0:
                    G.add_edge((u, l), (v, l + 1), capacity=cap)

        # compute max-flow from each source to each sink
        M = np.zeros((num_cell_types, num_cell_types))
        for i in tqdm(range(num_cell_types), desc=f"Head {h+1}/{num_heads}"):
            for j in range(num_cell_types):
                try:
                    s_node = source_nodes[i]
                    t_node = sink_nodes[j]
                    flow_val, _ = nx.maximum_flow(G, s_node, t_node, capacity='capacity', flow_func=nx.algorithms.flow.shortest_augmenting_path)
                    M[i, j] = flow_val
                except (nx.NetworkXError, ValueError):
                    M[i, j] = 0

        M_div_C = np.zeros_like(M)
        nonzero = C != 0
        M_div_C[nonzero] = M[nonzero] / C[nonzero]

        # Normalize L1
        M_normed_L1 = normalize(M_div_C, axis=0, norm='l1')
        # Normalize L2
        M_normed_L2 = normalize(M_div_C, axis=0, norm='l2')

        M_list_l1.append(M_normed_L1)
        M_list_l2.append(M_normed_L2)

    return M_list_l1, M_list_l2


def plot_attention_matrices(M_list, class_names, output_dir, norm_type='none'):
    """
    Plots the attention matrices as heatmaps.
    """
    white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
    max_val = np.max([np.max(M) for M in M_list])
    for head_idx, M in enumerate(M_list):
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            M,
            xticklabels=class_names,
            yticklabels=class_names,
            cmap=white_red,
            vmin=0,
            vmax=max_val,
            annot=True
        )
        plt.title(f'Attention Head {head_idx + 1}_{norm_type}')
        plt.xlabel('Destination')
        plt.ylabel('Source')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'attention_heatmap_{norm_type}_head{head_idx + 1}.pdf'),
                    format='pdf')
        plt.show()


# -----------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------
if __name__ == '__main__':
    from utils import create_model, load_mouse_midbrain_data, acquire_sparse_data, delaunay_to_graph, evaluate_metrics
    from tensorflow import keras

    FILENAME = 'Dorsal_midbrain_cell_bin.h5ad'
    data_path = os.path.join(os.curdir, FILENAME)

    # 1) Load data
    adata, coordinates = load_mouse_midbrain_data(data_path, ['SS200000108BR_B1B2'])
    node_states = tf.sparse.to_dense(acquire_sparse_data(adata.X))
    adj_mat = delaunay_to_graph(coordinates)
    edges = tf.cast(adj_mat.indices, tf.int32)

    # 2) Prepare labels
    cell_types = tf.constant(adata.obs['Cell Types'])
    string_lookup_layer = keras.layers.StringLookup(num_oov_indices=0, vocabulary = ['Glu Neu', 'RGC', 'Fibro', 'Glu NeuB', 'NeuB', 'GABA Neu', 'Endo', 'GlioB', 'Ery', 'GABA NeuB', 'Micro'])
    # string_lookup_layer.adapt(cell_types)
    encoded_cell_types = string_lookup_layer(cell_types)
    labels = tf.cast(encoded_cell_types, tf.int32)
    num_classes = len(string_lookup_layer.get_vocabulary())

    # 3) Load the pretrained model
    model = create_model(
        input_dim=node_states.shape[1],
        hidden_units=256,
        num_heads=3,
        num_classes=num_classes
    )
    model.load_weights('GAT_node_classification_model.h5')


    test_metrics = evaluate_metrics(
        model,
        node_states_full=node_states,
        edges_full=edges,
        labels_full=labels,
        indices=tf.range(labels.shape[0], dtype=tf.int32),
        num_classes=num_classes,
        batch_size=512
    )

    metrics_keys = ["acc", "f1", "precision", "recall", "auc"]

    for k in metrics_keys:
        values = np.array(test_metrics[k])
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {k}: {mean_val * 100:.2f}% (+/- {std_val * 100:.2f}%)")



    # 4) Inference + Confusion Matrix
    preds = model([node_states, edges], training=False)
    preds = np.argmax(preds, axis=1)
    cfm = tf.math.confusion_matrix(labels, preds, dtype=tf.int32)
    print("Confusion Matrix:\n", cfm.numpy())
    plot_confusion_matrix(cfm, string_lookup_layer.get_vocabulary())

    # 5) Optional: Reset old logs
    for layer in model.layers:
        if isinstance(layer, GATConv):
            layer.reset_attention_scores()

    # 6) Rerun forward pass to fill logs
    _ = model([node_states, edges], training=False)

    # 7) Collect the attention from each GATConv
    #    Each forward pass logs exactly one Tensor -> shape [E, num_heads]
    #    We'll split it per head to match the shape your analysis expects: [E].
    attention_weights = []
    for layer in model.layers:
        if isinstance(layer, GATConv):
            # GATConv logs one entry in `get_attention_scores()` per call
            # so we might have multiple if you call model multiple times.
            layer_logs = layer.get_attention_scores()  # list of Tensors

            for alpha in layer_logs:
                # alpha shape = [E, num_heads]
                # We split along axis=1
                heads_list = tf.split(alpha, num_or_size_splits=layer.num_heads, axis=-1)
                # Now heads_list[i] -> shape [E, 1], so we squeeze to shape [E]
                heads_list = [tf.squeeze(h, axis=-1) for h in heads_list]
                # heads_list is a list: [ head_0, head_1, ..., head_{num_heads-1} ]
                attention_weights.append(heads_list)

    # Now attention_weights is a list of shape [num_layers_called], each an array of heads
    # e.g. attention_weights[layer_idx][head_idx] => shape [E]

    # 8) Use the attention in your existing analysis
    compute_and_visualize_entropy(adj_mat, attention_weights, output_dir='.')
    plot_pairwise_attention_weights(adj_mat, attention_weights, labels, string_lookup_layer.get_vocabulary())

    # Flow-based
    attention_scores_l1, attention_scores_l2 = compute_attention_score(
        adj_mat.indices, labels, string_lookup_layer.get_vocabulary(), attention_weights
    )
    plot_attention_matrices(attention_scores_l1, string_lookup_layer.get_vocabulary(), '.', 'L1')
    plot_attention_matrices(attention_scores_l2, string_lookup_layer.get_vocabulary(), '.', 'L2')
