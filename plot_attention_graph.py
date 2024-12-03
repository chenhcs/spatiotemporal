'''
@Author: Hongliang Zhou
@Description: Providing tools for the visualization of the attention weights and delauny graph.
'''

import math
import os

from matplotlib.colors import LinearSegmentedColormap
from pyvis.network import Network
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.preprocessing import normalize
import tensorflow as tf
import networkx as nx
import numpy as np
from tqdm import tqdm

os.environ['BROWSER'] = r'C:\Program Files\Mozilla Firefox\firefox.exe'  # Use raw string or double backslashes



def visualize_delauny_graph(node_positions, node_labels, adj_mat: tf.sparse.SparseTensor,
                            edge_weights=None, threshold=0.4,
                            node_color_labels=None,
                            node_color_scheme='node_labels'):
    '''
    Parameters:
    node_positions: a 2D tensor or array of shape [N_nodes, 2], representing the positions of each node.
    node_labels: array-like, the labels of the nodes in the graph.
    adj_mat: a tf.sparse.SparseTensor representing the adjacency matrix.
    edge_weights: a list of lists of arrays, where edge_weights[layer][head] is an array representing the attention weights of head in layer.
    threshold: float, an adjustable threshold for the attention scores.
    node_color_labels: array-like, labels used for coloring nodes (e.g., train/val/test labels).
    node_color_scheme: str, 'node_labels' or 'node_color_labels', determines how nodes are colored.
    '''

    # Convert inputs to numpy arrays if necessary
    if isinstance(node_positions, tf.Tensor):
        node_positions = node_positions.numpy()
    if isinstance(node_labels, tf.Tensor):
        node_labels = node_labels.numpy()
    if node_color_labels is not None and isinstance(node_color_labels, tf.Tensor):
        node_color_labels = node_color_labels.numpy()
    edge_index = adj_mat.indices.numpy().T
    num_nodes = adj_mat.shape[0]

    # Build the nodes data
    nodes = []
    for idx, pos in enumerate(node_positions):
        node_data = {
            'id': idx,
            'label': str(idx),
            'x': int(pos[0])*1000,
            'y': int(pos[1])*1000,
            'title': f'Node {idx}<br>Cell Type: {node_labels[idx]}'
        }
        nodes.append(node_data)

    # Assign Node Colors
    node_colors_dict = {}

    # For node_labels
    unique_labels = np.unique(node_labels)
    palette = sns.color_palette('hsv', len(unique_labels))
    label_color_map = dict(zip(unique_labels, palette))
    label_color_map_hex = {label: '#' + ''.join(f'{int(255 * c):02x}' for c in color)
                           for label, color in label_color_map.items()}
    node_colors_labels = [label_color_map_hex[node_labels[idx]] for idx in range(num_nodes)]
    node_colors_dict['node_labels'] = node_colors_labels

    # For node_color_labels (e.g., train/val/test)
    if node_color_labels is not None:
        unique_labels = np.unique(node_color_labels)
        palette = sns.color_palette('Set2', len(unique_labels))
        label_color_map = dict(zip(unique_labels, palette))
        label_color_map_hex = {label: '#' + ''.join(f'{int(255 * c):02x}' for c in color)
                               for label, color in label_color_map.items()}
        node_colors_color_labels = [label_color_map_hex[node_color_labels[idx]] for idx in range(num_nodes)]
        node_colors_dict['node_color_labels'] = node_colors_color_labels
    else:
        node_colors_dict['node_color_labels'] = node_colors_labels  # Default to node_labels

    # Set initial node colors
    node_colors = node_colors_dict[node_color_scheme]

    # Add colors to nodes
    for idx in range(num_nodes):
        nodes[idx]['color'] = node_colors[idx]

    # Collect edge data for all layers and heads
    edge_data = {}
    num_layers = len(edge_weights)
    num_heads = len(edge_weights[0])

    # Create a mapping from edges to their reverse counterparts
    edge_pairs = {}
    for i in range(edge_index.shape[1]):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        key = (min(u, v), max(u, v))
        if key not in edge_pairs:
            edge_pairs[key] = []
        edge_pairs[key].append((u, v, i))  # Store the index for weight retrieval

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            edges = []
            head_weights = edge_weights[layer_idx][head_idx].numpy()
            for key, edge_list in edge_pairs.items():
                u, v = key
                weights = []
                titles = []
                # Check for both directions
                for (src, dst, idx) in edge_list:
                    weight = head_weights[idx]
                    weights.append((src, dst, weight))
                    titles.append(f'Edge {src} -> {dst}: Weight {weight:.3f}')
                # Decide on edge attributes
                if len(weights) == 2:
                    # Two edges, create curved edges in opposite directions
                    for i, (src, dst, weight) in enumerate(weights):
                        edge = {
                            'from': src,
                            'to': dst,
                            'weight': float(weight),
                            'title': titles[i],
                            'smooth': {'enabled': True, 'type': 'curvedCCW' if i == 0 else 'curvedCW'},
                            'arrows': 'to',
                        }
                        edges.append(edge)
                else:
                    # Single edge, treat as undirected
                    src, dst, weight = weights[0]
                    edge = {
                        'from': src,
                        'to': dst,
                        'weight': float(weight),
                        'title': titles[0],
                        'smooth': False,
                    }
                    edges.append(edge)
            edge_data_key = f'layer{layer_idx}_head{head_idx}'
            edge_data[edge_data_key] = edges

    # Initialize the network
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
    net.barnes_hut()

    # Disable physics to fix node positions
    net.toggle_physics(False)

    # Add nodes
    for node in nodes:
        net.add_node(node['id'], label=node['label'], color=node['color'],
                     x=node['x'], y=node['y'], title=node['title'])

    # Generate the base HTML
    net_html = net.generate_html()

    # Prepare the edge data as a JavaScript variable
    edge_data_json = json.dumps(edge_data)
    node_colors_dict_json = json.dumps(node_colors_dict)

    # Prepare the controls HTML
    controls_html = '''
    <div style="position: fixed; top: 10px; left: 10px; z-index: 1000; background-color: rgba(255, 255, 255, 0.8); padding: 10px;">
        <label for="layer_select">Select Layer:</label>
        <select id="layer_select"></select>
        <label for="head_select">Select Head:</label>
        <select id="head_select"></select>
        <br>
        <label for="color_select">Select Node Coloring:</label>
        <select id="color_select"></select>
        <br>
        <label for="threshold_input">Threshold:</label>
        <input type="range" id="threshold_input" min="0" max="1" step="0.01" value="{threshold}">
        <span id="threshold_value">{threshold}</span>
    </div>
    '''.format(threshold=threshold)

    # Prepare the custom JavaScript code
    custom_js = f'''
    <script type="text/javascript">
    var allEdges = {edge_data_json};
    var nodeColors = {node_colors_dict_json};

    // Get the network instance
    var network = window.network;

    // Function to update edges based on threshold
    function updateEdges() {{
        var layer = document.getElementById('layer_select').value;
        var head = document.getElementById('head_select').value;
        var threshold = parseFloat(document.getElementById('threshold_input').value);
        document.getElementById('threshold_value').innerText = threshold.toFixed(2);
        var edgeKey = 'layer' + layer + '_head' + head;
        var edges = allEdges[edgeKey];

        // Update edge colors based on threshold
        var updatedEdges = edges.map(function(edge) {{
            var newEdge = Object.assign({{}}, edge);
            if (edge.weight >= threshold) {{
                newEdge.color = {{'color': 'red'}};
            }} else {{
                newEdge.color = {{'color': 'gray'}};
            }}
            return newEdge;
        }});

        // Update edges in the network
        network.setData({{nodes: network.body.data.nodes, edges: new vis.DataSet(updatedEdges)}});
    }}

    // Function to update node colors
    function updateNodeColors() {{
        var colorScheme = document.getElementById('color_select').value;
        var colors = nodeColors[colorScheme];

        var nodes = network.body.data.nodes.get();
        for (var i = 0; i < nodes.length; i++) {{
            nodes[i].color = colors[i];
        }}
        network.body.data.nodes.update(nodes);
    }}

    // Populate the layer and head dropdowns
    function populateControls() {{
        var layerSelect = document.getElementById('layer_select');
        var headSelect = document.getElementById('head_select');
        var colorSelect = document.getElementById('color_select');

        var numLayers = {num_layers};
        var numHeads = {num_heads};

        for (var i = 0; i < numLayers; i++) {{
            var option = document.createElement('option');
            option.value = i;
            option.text = i;
            layerSelect.add(option);
        }}

        for (var j = 0; j < numHeads; j++) {{
            var option = document.createElement('option');
            option.value = j;
            option.text = j;
            headSelect.add(option);
        }}

        var colorSchemes = Object.keys(nodeColors);
        for (var k = 0; k < colorSchemes.length; k++) {{
            var option = document.createElement('option');
            option.value = colorSchemes[k];
            option.text = colorSchemes[k];
            colorSelect.add(option);
        }}

        // Set event listeners
        layerSelect.addEventListener('change', function() {{
            updateEdges();
        }});
        headSelect.addEventListener('change', function() {{
            updateEdges();
        }});
        colorSelect.addEventListener('change', function() {{
            updateNodeColors();
        }});
        document.getElementById('threshold_input').addEventListener('input', function() {{
            updateEdges();
        }});
    }}

    // Call the function to populate controls and update edges and node colors
    window.addEventListener('load', function() {{
        populateControls();
        updateEdges();
        updateNodeColors();
    }});
    </script>
    '''

    # Inject the controls and custom JavaScript into the HTML
    net_html = net_html.replace('</body>', controls_html + custom_js + '</body>')

    # Write the modified HTML to file
    with open('graph.html', 'w') as f:
        f.write(net_html)



def compute_and_visualize_entropy(adj_mat: tf.sparse.SparseTensor,
                                  edge_weights, output_dir,
                                  num_bins=50):
    '''
    Computes and visualizes the entropy of each node's attention weights
    over its neighbors for each layer and head.

    Parameters:
    adj_mat: tf.sparse.SparseTensor representing the adjacency matrix.
    edge_weights: list of lists of tensors, where edge_weights[layer][head]
                  is a tensor representing the attention weights of that head in that layer.
    num_bins: int, number of bins for the histogram.

    Returns:
    None (plots histograms)
    '''
    # Convert adjacency matrix to indices
    edge_index = adj_mat.indices.numpy().T  # Shape: [2, num_edges]
    num_nodes = adj_mat.shape[0]
    num_layers = len(edge_weights)
    num_heads = len(edge_weights[0])

    # Build a neighbor list for each node
    neighbor_dict = {i: [] for i in range(num_nodes)}
    for idx in range(edge_index.shape[1]):
        src = int(edge_index[0, idx])
        dst = int(edge_index[1, idx])
        neighbor_dict[src].append((dst, idx))  # Store destination node and edge index

    # Initialize entropy storage
    entropies = {}  # Dictionary to store entropies for each layer and head

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            # Get the attention weights for this layer and head
            head_weights = edge_weights[layer_idx][head_idx].numpy()
            node_entropies = []
            max_entropies = []
            normalized_entropies = []

            for node in range(num_nodes):
                neighbors = neighbor_dict[node]
                degrees = len(neighbors)
                if degrees == 0:
                    continue  # Skip isolated nodes

                # Extract attention weights for this node over its neighbors
                attn_weights = np.array([head_weights[idx] for (_, idx) in neighbors])

                # Ensure the weights sum to 1 (in case of numerical errors)
                attn_weights /= attn_weights.sum()

                # Compute entropy
                node_entropy = entropy(attn_weights, base=np.e)  # Natural logarithm
                node_entropies.append(node_entropy)

                # Compute maximum possible entropy (uniform distribution)
                max_entropy = np.log(degrees)
                max_entropies.append(max_entropy)

                # Compute normalized entropy
                normalized_entropy = node_entropy / max_entropy if max_entropy > 0 else 0
                normalized_entropies.append(normalized_entropy)

            # Store entropies
            key = f'Layer {layer_idx + 1}, Head {head_idx + 1}'
            entropies[key] = {
                'entropy': node_entropies,
                'max_entropy': max_entropies,
                'normalized_entropy': normalized_entropies
            }

    # Visualization
    sns.set(style="whitegrid")
    num_plots = num_layers * num_heads
    cols = min(num_heads, 4)
    rows = (num_plots + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 4 * rows))
    plot_idx = 1

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            key = f'Layer {layer_idx + 1}, Head {head_idx + 1}'
            data = entropies[key]['entropy']
            max_data = entropies[key]['max_entropy']

            plt.subplot(rows, cols, plot_idx)
            sns.histplot(data, bins=num_bins, kde=True, color='blue', edgecolor='black', label = 'Entropy Distribution')
            sns.histplot(max_data, bins=num_bins, kde=True, color='red', edgecolor='yellow', label='Uniform Distribution Entropy')
            # plt.axvline(x=1.0, color='red', linestyle='--', label='Uniform Distribution Entropy')
            plt.title(key)
            plt.xlabel('Entropy')
            plt.ylabel('Frequency')
            plt.legend()
            plot_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_graph_entropy_distribution.pdf'), format='pdf')
    plt.show()

def plot_pairwise_attention_weights(adj_mat, edge_weights, node_types, class_names, threshold=None):
    '''
    Computes and visualizes the average attentional weights between each pair of classes for each layer and head.

    Parameters:
    adj_mat: tf.sparse.SparseTensor representing the adjacency matrix.
    edge_weights: list of lists of tensors, where edge_weights[layer][head]
                  is a tensor representing the attention weights of that head in that layer.
    node_types: tf.Tensor of int, where node_types[i] is the class type of node i
    class_names: list of strings, where class_names[i] is the name of class i
    threshold: float, optional threshold to count high attention weights

    Returns:
    class_attention_avg_all: A list of lists containing numpy arrays representing the average attention weights between classes for each layer and head
    class_attention_counts_all: A list of lists containing numpy arrays representing the counts of high attention weights between classes for each layer and head
    '''

    # Get the adjacency indices (edges)
    adj_indices = adj_mat.indices.numpy()  # Shape [num_edges, 2]
    edges_source = adj_indices[:, 0]  # node u
    edges_target = adj_indices[:, 1]  # node v

    # Get node types for source and target nodes
    node_types_array = np.array(node_types)
    node_types_source = node_types_array[edges_source]
    node_types_target = node_types_array[edges_target]

    num_classes = len(class_names)
    num_layers = len(edge_weights)
    num_heads = len(edge_weights[0]) if num_layers > 0 else 0

    # Initialize lists to store results for all layers and heads
    class_attention_avg_all = []
    class_attention_counts_all = []

    # Iterate over layers
    for layer in range(num_layers):
        class_attention_avg_layer = []
        class_attention_counts_layer = []
        # Iterate over heads
        for head in range(num_heads):
            attention_weights = edge_weights[layer][head].numpy()
            # print(attention_weights.shape)

            # Initialize matrices for this layer and head
            class_attention_sum = np.zeros((num_classes, num_classes))
            class_pair_counts = np.zeros((num_classes, num_classes))
            class_attention_counts = np.zeros((num_classes, num_classes))

            # Accumulate sum of attention weights into class_attention_sum
            np.add.at(class_attention_sum, (node_types_source, node_types_target), attention_weights)

            # Count the number of occurrences for each class pairing
            np.add.at(class_pair_counts, (node_types_source, node_types_target), 1)

            if threshold is not None:
                # Identify high attention weights
                high_attention_mask = attention_weights > threshold
                high_c1 = node_types_source[high_attention_mask]
                high_c2 = node_types_target[high_attention_mask]

                # Accumulate counts of high attention weights
                np.add.at(class_attention_counts, (high_c1, high_c2), 1)

            # Compute the average attention weights
            with np.errstate(divide='ignore', invalid='ignore'):
                class_attention_avg = np.divide(class_attention_sum, class_pair_counts)
                class_attention_avg[np.isnan(class_attention_avg)] = 0  # Replace NaN with zero

            # Append results for this head
            class_attention_avg_layer.append(class_attention_avg)
            class_attention_counts_layer.append(class_attention_counts)

            # Plot the heatmap of average attention weights for this layer and head
            plt.figure(figsize=(10, 8))
            sns.heatmap(class_attention_avg, annot=True, fmt='.4f', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Target Class')
            plt.ylabel('Source Class')
            plt.title(f'Layer {layer}, Head {head}: Average Pairwise Attention Weights between Classes')
            plt.show()

            if threshold is not None:
                # Plot the heatmap of high attention counts
                plt.figure(figsize=(10, 8))
                sns.heatmap(class_attention_counts, annot=True, fmt='d', cmap='Reds',
                            xticklabels=class_names, yticklabels=class_names)
                plt.xlabel('Target Class')
                plt.ylabel('Source Class')
                plt.title(f'Layer {layer}, Head {head}: Number of High Attention Edges (>{threshold}) between Classes')
                plt.show()

        # Append results for this layer
        class_attention_avg_all.append(class_attention_avg_layer)
        class_attention_counts_all.append(class_attention_counts_layer)

    return class_attention_avg_all, class_attention_counts_all

# def attention_rollout():
#
#
# def quantify_neighbor_attention(adj_mat, edge_weights, node_types, class_names, threshold=None):


def plot_confusion_matrix(cm, class_names):
    conf_matrix = cm.numpy()

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    # Add labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')
    plt.show()


def compute_attention_score(Edge_index, node_types, class_names, Attention_weights, degree_normalization = True):
    """
    Computes the attention score from one cell type to others using max-flow.

    Parameters:
    - Edge_index: tf.Tensor of shape [num_edges, 2], representing the edges between nodes.
    - node_types: tf.Tensor of shape [num_nodes], where node_types[i] is the class type of node i.
    - class_names: List of strings of length num_cell_types, names of each cell type.
    - Attention_weights: List of lists of tensors, where Attention_weights[layer][head]
      is a tensor representing the attention weights of that head in that layer.

    Returns:
    - M_list: A list of N x N numpy arrays, where each array corresponds to a head.
      M_list[h][i, j] indicates the relative importance of cell type i to cell type j
      for head h after applying softmax to each column.
    """
    num_layers = len(Attention_weights)
    num_heads = len(Attention_weights[0])
    num_nodes = node_types.shape[0]
    num_cell_types = len(class_names)

    # Convert tensors to numpy arrays for processing
    Edge_index_np = Edge_index.numpy()
    node_types_np = np.array(node_types)

    # Initialize the list of importance matrices per head
    M_list_l1 = []
    M_list_l2 = []

    # For each head, build the flow network and compute the importance matrix
    for h in range(num_heads):
        # Build the flow network G_h for head h
        G = nx.DiGraph()

        # Add nodes (v, l) for each node v and layer l
        for v in range(num_nodes):
            for l in range(num_layers + 1):
                node_name = (v, l)
                G.add_node(node_name)

        # Add source and sink nodes for each cell type
        source_nodes = {}
        sink_nodes = {}
        for i in range(num_cell_types):
            source_node = f"S_{i}"
            sink_node = f"T_{i}"
            source_nodes[i] = source_node
            sink_nodes[i] = sink_node
            G.add_node(source_node)
            G.add_node(sink_node)

        # Connect sources to nodes in the first layer
        for i in range(num_cell_types):
            source_node = source_nodes[i]
            nodes_of_type = np.where(node_types_np == i)[0]
            for v in nodes_of_type:
                node_name = (v, 0)
                G.add_edge(source_node, node_name, capacity=math.inf)

        # Connect nodes in the last layer to sinks
        for i in range(num_cell_types):
            sink_node = sink_nodes[i]
            nodes_of_type = np.where(node_types_np == i)[0]
            for v in nodes_of_type:
                node_name = (v, num_layers)
                G.add_edge(node_name, sink_node, capacity=math.inf)

        # Add edges between layers with capacities as attention weights for this head
        num_edges = Edge_index_np.shape[0]
        for l in range(num_layers):
            attention_weights = Attention_weights[l][h].numpy()  # Shape: [num_edges]
            for e in range(num_edges):
                u = Edge_index_np[e, 0]
                v = Edge_index_np[e, 1]
                capacity = attention_weights[e]
                source_node = (u, l)
                target_node = (v, l + 1)
                # Only add edge if capacity is greater than zero to optimize the graph
                if capacity > 0:
                    G.add_edge(source_node, target_node, capacity=capacity)

        # Initialize the importance matrix M for this head
        M = np.zeros((num_cell_types, num_cell_types))

        # Compute max-flow from each source to each sink

        for i in tqdm(range(num_cell_types), desc="Generating Attention Graph"):
            for j in range(num_cell_types):
                try:
                    source_node = source_nodes[i]
                    sink_node = sink_nodes[j]
                    # Compute max-flow between source and sink
                    flow_value, _ = nx.maximum_flow(G, source_node, sink_node, capacity='capacity')
                    M[i, j] = flow_value
                except ValueError:
                    # When the source and sink is disconnected, we accredit a score of 0 to it.
                    M[i, j] = 0

        M_normed_L1 = normalize(M, axis=0, norm="l1")
        M_normed_L2 = normalize(M, axis=0, norm="l2")

        # Apply softmax to each column of M
        # M_exp_L1 = np.exp(M_normed_L1)
        # M_softmax_L1 = M_exp_L1 / np.sum(M_exp_L1, axis=0, keepdims=True)

        # M_exp_L2 = np.exp(M_normed_L2)
        # M_softmax_L2 = M_exp_L2 / np.sum(M_exp_L2, axis=0, keepdims=True)
        # M_normed = M / M.max(axis=0)

        # M_normed_L1 = normalize(M, axis=0,norm="l1")
        # M_normed_L2 = normalize(M, axis=0,norm="l2")

        # Append the importance matrix for this head to the list
        M_list_l1.append(M_normed_L1)
        M_list_l2.append(M_normed_L2)

    return M_list_l1, M_list_l2

def plot_attention_matrices(M_list, class_names, output_dir, norm_type='none'):
    """
    Plots the attention matrices as heatmaps.

    Parameters:
    - M_list: List of N x N numpy arrays, where each array corresponds to a head.
    - class_names: List of strings of length N, names of each cell type.
    - norm_type: String, type of normalization to apply ('none', 'row', 'column', 'global').
    """
    # Create a custom colormap from white to red
    white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])

    # Determine the maximum value across all matrices for consistent scaling
    max_value = np.max([np.max(M) for M in M_list])
    # Plot each attention matrix in its own figure
    for head_idx, M in enumerate(M_list):
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            M,
            xticklabels=class_names,
            yticklabels=class_names,
            cmap=white_red,
            vmin=0,
            vmax=max_value,
            annot=True
        )
        plt.title(f'Attention Head {head_idx + 1}_{norm_type}')
        plt.xlabel('Destination Node')
        plt.ylabel('Source Node')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'attention_heatmap{norm_type}.pdf'), format = 'pdf')
        plt.show()
