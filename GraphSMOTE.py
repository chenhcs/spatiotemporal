'''
@Author Hongliang Zhou
@Description An implementation of the GraphSMOTE method.
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import legacy as optim_legacy
import scipy.sparse as sp


##############################################################################
# Utilities
##############################################################################
def visualize_data_distribution(vocabularies, counts):
    x = vocabularies
    y = counts

    # 4. Plot the distribution in a bar chart
    plt.bar(x, y)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.show()

def build_normalized_adjacency(edges, num_nodes):
    """
    Create a [N, N] row-normalized adjacency with self-loops.
    edges: np.array [E, 2]
    Returns a tf.sparse.SparseTensor of shape [N, N].
    """
    row = edges[:, 0]
    col = edges[:, 1]
    data = np.ones(len(edges), dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    # Add self-loops
    adj = adj + sp.eye(num_nodes)

    # Row-normalize
    rowsum = np.array(adj.sum(axis=1)).flatten()
    inv_rowsum = np.power(rowsum, -1, where=(rowsum != 0))
    D_inv = sp.diags(inv_rowsum)
    adj_norm = D_inv.dot(adj).tocoo()

    indices = np.stack([adj_norm.row, adj_norm.col], axis=1)
    values = adj_norm.data
    A_tf = tf.sparse.SparseTensor(indices, values, adj_norm.shape)
    return tf.sparse.reorder(A_tf)


class GraphConv(tf.keras.layers.Layer):
    """
    Minimal GCN layer: X' = A_norm * X * W
    We'll pass row-normalized adjacency as a tf.sparse.SparseTensor.
    """
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def build(self, input_shape):
        in_dim = int(input_shape[0][-1])
        self.w = self.add_weight(
            shape=(in_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='gcn_weight'
        )

    def call(self, inputs, training=False):
        x, a_norm = inputs  # x: [N, in_dim]
        xw = tf.matmul(x, self.w)  # [N, out_dim]
        out = tf.sparse.sparse_dense_matmul(a_norm, xw)
        return out


##############################################################################
# GraphSMOTE Modules (Encoder, Decoder, EdgeGen, Classifier)
##############################################################################

class GNNEncoder(Model):
    """Feature extractor (Encoder) using a 2-layer GCN."""
    def __init__(self, hidden_dim=64, emb_dim=32, dropout=0.1):
        super().__init__()
        self.gc1 = GraphConv(hidden_dim)
        self.gc2 = GraphConv(emb_dim)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.dropout = Dropout(dropout)

    def call(self, inputs, training=False):
        x, a_norm = inputs
        h = self.gc1([x, a_norm], training=training)
        h = self.bn1(h, training=training)
        h = tf.nn.relu(h)
        h = self.dropout(h, training=training)
        z = self.gc2([h, a_norm], training=training)
        z = self.bn2(z, training=training)
        return z


class FeatureDecoder(Model):
    """MLP that reconstructs original-dim node features from latent embeddings."""
    def __init__(self, out_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.dense1 = Dense(hidden_dim, activation='relu')
        self.bn1 = BatchNormalization()
        self.dropout = Dropout(dropout)
        self.dense2 = Dense(out_dim, activation=None)  # linear output

    def call(self, z, training=False):
        h = self.dense1(z, training=training)
        h = self.bn1(h, training=training)
        h = self.dropout(h, training=training)
        x_recon = self.dense2(h, training=training)
        return x_recon


class EdgeGenerator(Model):
    """
    Edge probability for pairs (z_u, z_v). We'll do an MLP on [z_u, z_v].
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.dense1 = Dense(hidden_dim, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, z_u, z_v, training=False):
        """
        z_u, z_v: [B, emb_dim] each
        returns: link probability in [0,1], shape [B,1]
        """
        concat_z = tf.concat([z_u, z_v], axis=-1)
        h = self.dense1(concat_z, training=training)
        score = self.dense2(h, training=training)
        return score


class GNNClassifier(Model):
    """
    Node classifier (2-layer GCN for demonstration).
    """
    def __init__(self, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.gc1 = GraphConv(hidden_dim)
        self.gc2 = GraphConv(num_classes)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.dropout = Dropout(dropout)

    def call(self, inputs, training=False):
        x, a_norm = inputs
        h = self.gc1([x, a_norm], training=training)
        h = self.bn1(h, training=training)
        h = tf.nn.relu(h)
        h = self.dropout(h, training=training)
        logits = self.gc2([h, a_norm], training=training)
        logits = self.bn2(logits, training=training)
        return logits


##############################################################################
# Helper Loss Functions (MSE recon, partial edge sampling)
##############################################################################

def reconstruction_loss(x_recon, x_real):
    return tf.reduce_mean(tf.reduce_mean((x_recon - x_real)**2, axis=-1))

def partial_edge_loss(z, edges_real, edge_gen, neg_size=1000):
    """
    Sample up to 'neg_size' real edges => positive examples.
    For each, sample a matching number of random negative edges => negative examples.
    Return average BCE.
    """
    E = tf.shape(edges_real)[0]
    if E > neg_size:
        indices = tf.random.shuffle(tf.range(E))[:neg_size]
        edges_sample = tf.gather(edges_real, indices)
    else:
        edges_sample = edges_real

    # Positive edges
    idx_u = edges_sample[:, 0]
    idx_v = edges_sample[:, 1]
    z_u = tf.gather(z, idx_u)
    z_v = tf.gather(z, idx_v)
    score_pos = edge_gen(z_u, z_v)  # [?, 1]
    loss_pos = -tf.math.log(score_pos + 1e-9)

    # Negative edges: same count as edges_sample
    count_sample = tf.shape(edges_sample)[0]
    neg_u = tf.random.uniform(shape=[count_sample], maxval=tf.shape(z)[0], dtype=tf.int32)
    neg_v = tf.random.uniform(shape=[count_sample], maxval=tf.shape(z)[0], dtype=tf.int32)
    z_un = tf.gather(z, neg_u)
    z_vn = tf.gather(z, neg_v)
    score_neg = edge_gen(z_un, z_vn)
    loss_neg = -tf.math.log(1. - score_neg + 1e-9)

    return tf.reduce_mean(loss_pos + loss_neg)


##############################################################################
# Generation Phase: "Generate Synthetic Nodes" (No Training Here)
##############################################################################

def generate_synthetic_nodes(
    encoder,
    decoder,
    edge_gen,
    X_aug,         # [N_aug, D]
    labs_aug,      # [N_aug,]
    edges_aug,     # [E_aug, 2]
    minority_classes,
    alpha_synthetic=0.5,
    oversampling_scale=1.0,
    k_new_edges=10
):
    """
    1) Embed existing augmented graph with 'encoder'
    2) For each minority class, do SMOTE-style interpolation => new latent vectors
    3) Decode those new vectors => new features
    4) Use top-k approach (k_new_edges) to generate new edges to existing nodes
    Return:
      x_new, labs_new, new_edges (the newly generated content)
      DOES NOT modify X_aug or edges_aug in-place.
    """
    # build adjacency
    A_aug_tf = build_normalized_adjacency(edges_aug.numpy(), X_aug.shape[0])
    # embed
    z_all = encoder([X_aug, A_aug_tf], training=False)

    new_z_list   = []
    new_lab_list = []
    # Oversampling
    for c in minority_classes:
        c_indices = tf.where(labs_aug == c)[:,0]
        c_len = tf.shape(c_indices)[0]
        # produce ~ oversampling_scale * c_len synthetic nodes
        count_new = int(c_len.numpy() * oversampling_scale)
        for _ in range(count_new):
            i, j = tf.random.shuffle(c_indices)[:2]
            zi = z_all[i]
            zj = z_all[j]
            z_new = alpha_synthetic*zi + (1.-alpha_synthetic)*zj
            new_z_list.append(z_new)
            new_lab_list.append(c)

    if len(new_z_list) == 0:
        # no new nodes => return empty
        return None, None, None

    new_z_tf   = tf.stack(new_z_list, axis=0)  # shape [B, emb_dim]
    new_labs_tf= tf.stack(new_lab_list, axis=0)

    # decode => x_new
    x_new_tf   = decoder(new_z_tf, training=False)  # shape [B, D]

    # build edges for each new node => top-k approach
    B = tf.shape(new_z_tf)[0]
    new_edges_list = []
    base_id = X_aug.shape[0]
    for i in range(B):
        z_i = new_z_tf[i:i+1]  # [1, emb_dim]
        # compare with z_all => get [N_aug, emb_dim]
        z_i_tile = tf.tile(z_i, [tf.shape(z_all)[0], 1])
        scores   = edge_gen(z_i_tile, z_all, training=False)  # [N_aug,1]
        scores   = tf.squeeze(scores, axis=-1)
        # pick top-k
        topvals, topinds = tf.math.top_k(scores, k=k_new_edges)
        new_node_id = base_id + i
        edges_f = tf.stack([tf.fill([k_new_edges], new_node_id), topinds], axis=-1)
        edges_r = tf.stack([topinds, tf.fill([k_new_edges], new_node_id)], axis=-1)
        new_edges_list.append(edges_f)
        new_edges_list.append(edges_r)

    new_edges_all = tf.concat(new_edges_list, axis=0) if new_edges_list else None
    return x_new_tf, new_labs_tf, new_edges_all


##############################################################################
# Training Phase: "Train on Augmented Graph" (No New Synthesis Here)
##############################################################################

def train_on_augmented_graph(
    encoder,
    decoder,
    edge_gen,
    classifier,
    opt,
    X_aug,        # [N_aug, D]
    labs_aug,     # [N_aug,]
    edges_aug,    # [E_aug, 2]
    real_edges,   # [E_real, 2] for partial edge loss on real adjacency
    epochs=5,
    lambda_edge=1.0,
    max_real_edges_sample=1000
):
    """
    Run multiple epochs of gradient updates on the augmented graph,
    updating all modules by L_node + lambda_edge*L_edge (with partial sampling).
    No new nodes are created here.
    """
    for ep in range(epochs):
        with tf.GradientTape() as tape:
            A_aug_tf = build_normalized_adjacency(edges_aug.numpy(), X_aug.shape[0])
            # forward pass
            z_aug = encoder([X_aug, A_aug_tf], training=True)
            logits = classifier([X_aug, A_aug_tf], training=True)
            # classification loss
            loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labs_aug, logits)
            loss_cls = tf.reduce_mean(loss_cls)

            # partial edge recon on real node embeddings only (first len(real_nodes))
            # if you want to also apply edge loss on synthetic edges, you can adapt
            loss_edge = partial_edge_loss(z_aug[:real_edges.shape[0]], real_edges, edge_gen,
                                          neg_size=max_real_edges_sample)

            loss_total = loss_cls + lambda_edge*loss_edge

            # plus any regularization
            reg_loss = (sum(encoder.losses) + sum(decoder.losses) +
                        sum(edge_gen.losses) + sum(classifier.losses))
            loss_total = loss_total + reg_loss

        vars_all = (encoder.trainable_variables +
                    decoder.trainable_variables +
                    edge_gen.trainable_variables +
                    classifier.trainable_variables)
        grads = tape.gradient(loss_total, vars_all)
        opt.apply_gradients(zip(grads, vars_all))

        if (ep+1)%1 == 0:  # print every epoch
            print(f"   Train epoch {ep+1}/{epochs}: L_cls={loss_cls.numpy():.4f}, L_edge={loss_edge.numpy():.4f}")


##############################################################################
# Putting It All Together: "graphsmote_knn_2phase" => Generate -> Train -> repeat
##############################################################################

def graphsmote_knn_2phase(
    node_states,
    edges,
    labels,
    num_classes,
    minority_classes,
    hidden_dim=128,
    emb_dim=128,
    decoder_hidden=128,
    dropout=0.1,
    lr=1e-3,
    lambda_edge=.8,
    k_new_edges=10,
    max_real_edges_sample=1000,
    pretrain_epochs=100,
    rounds=5,                # how many "generate -> train" cycles
    train_epochs_per_round=200,
    oversampling_scale=1.0,
    alpha_synthetic=0.5,
    encoder: GNNEncoder = None,
    decoder: FeatureDecoder = None,
    edge_generator: EdgeGenerator = None,
    classifier : GNNClassifier = None,
    train = False
):
    """
    A 2-phase iterative approach:
      (1) Pre-train on real data
      (2) for round in [1..rounds]:
           - generate_synthetic_nodes() (no training)
           - add them to the augmented graph
           - train_on_augmented_graph() (no new generation)
    """
    node_states = tf.constant(node_states, dtype=tf.float32)
    labels      = tf.constant(labels, dtype=tf.int32)
    edges       = tf.constant(edges, dtype=tf.int32)
    N, D        = node_states.shape

    # Generate the default encoder, decoder, edge geenrator, and classifier if not provided.
    if encoder is None:
    # Instantiate models
        encoder    = GNNEncoder(hidden_dim=hidden_dim, emb_dim=emb_dim, dropout=dropout)
    if decoder is None:
        decoder    = FeatureDecoder(out_dim=D, hidden_dim=decoder_hidden, dropout=dropout)
    if edge_generator is None:
        edge_gen   = EdgeGenerator(hidden_dim=hidden_dim)
    if classifier is None:
        classifier = GNNClassifier(hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)

    # Use legacy optimizer to avoid issues with new Keras optimizers
    opt = optim_legacy.Adam(lr)
    if train:
        # 1) Pretrain on real data if needed
        if pretrain_epochs > 0:
            print(">>> (1) Pre-training autoencoder + edge generator on real data")
            A_real = build_normalized_adjacency(edges.numpy(), N)
            for ep in range(pretrain_epochs):
                with tf.GradientTape() as tape:
                    z_real = encoder([node_states, A_real], training=True)
                    x_recon= decoder(z_real, training=True)
                    loss_recon = reconstruction_loss(x_recon, node_states)
                    loss_edge  = partial_edge_loss(z_real, edges, edge_gen, neg_size=max_real_edges_sample)
                    loss_total = loss_recon + loss_edge
                    # add reg
                    reg = (sum(encoder.losses)+sum(decoder.losses)+sum(edge_gen.losses))
                    loss_total = loss_total + reg
                vars_pt = encoder.trainable_variables + decoder.trainable_variables + edge_gen.trainable_variables
                grads = tape.gradient(loss_total, vars_pt)
                opt.apply_gradients(zip(grads, vars_pt))

                if (ep+1)%5 == 0:
                    print(f"   Pretrain epoch {ep+1}/{pretrain_epochs}, "
                          f"L_recon={loss_recon.numpy():.4f}, L_edge={loss_edge.numpy():.4f}")

        # Initialize augmented data with just the real part
        X_aug   = node_states.numpy()  # shape [N, D]
        labs_aug= labels.numpy()       # shape [N,]
        aug_edges = edges.numpy()      # shape [E,2]

        # 2) Repeated "Generation + Training" rounds
        print(">>> (2) Iterative Generation -> Training rounds")
        for rd in range(rounds):
            print(f"Round {rd+1}/{rounds}: GENERATION phase ...")
            # generate new samples
            x_new, labs_new, new_edges = generate_synthetic_nodes(
                encoder, decoder, edge_gen,
                X_aug, labs_aug,
                tf.constant(aug_edges, dtype=tf.int32),
                minority_classes,
                alpha_synthetic=alpha_synthetic,
                oversampling_scale=oversampling_scale,
                k_new_edges=k_new_edges
            )
            if x_new is not None:
                # add them to augmented data
                x_new_np = x_new.numpy()
                labs_new_np = labs_new.numpy()
                new_edges_np = new_edges.numpy()

                idx_start = X_aug.shape[0]
                idx_end   = idx_start + x_new_np.shape[0]
                # Append features & labels
                X_aug   = np.concatenate([X_aug, x_new_np], axis=0)
                labs_aug= np.concatenate([labs_aug, labs_new_np], axis=0)
                # Append edges
                aug_edges= np.concatenate([aug_edges, new_edges_np], axis=0)
                print(f"   Generated {x_new_np.shape[0]} new nodes, edges+={new_edges_np.shape[0]}")
            else:
                print("   No new synthetic nodes generated this round.")

            # TRAINING phase
            print(f"Round {rd+1}/{rounds}: TRAINING phase ...")
            train_on_augmented_graph(
                encoder,
                decoder,
                edge_gen,
                classifier,
                opt,
                X_aug,
                labs_aug,
                tf.constant(aug_edges, dtype=tf.int32),
                edges,  # real edges for partial edge recon
                epochs=train_epochs_per_round,
                lambda_edge=lambda_edge,
                max_real_edges_sample=max_real_edges_sample
            )
    else:
        # Initialize augmented data with just the real part
        X_aug = node_states.numpy()  # shape [N, D]
        labs_aug = labels.numpy()  # shape [N,]
        aug_edges = edges.numpy()  # shape [E,2]
        # generate new samples
        x_new, labs_new, new_edges = generate_synthetic_nodes(
            encoder, decoder, edge_gen,
            X_aug, labs_aug,
            tf.constant(aug_edges, dtype=tf.int32),
            minority_classes,
            alpha_synthetic=alpha_synthetic,
            oversampling_scale=oversampling_scale,
            k_new_edges=k_new_edges
        )
        # add them to augmented data
        x_new_np = x_new.numpy()
        labs_new_np = labs_new.numpy()
        new_edges_np = new_edges.numpy()

        idx_start = X_aug.shape[0]
        idx_end = idx_start + x_new_np.shape[0]
        # Append features & labels
        X_aug = np.concatenate([X_aug, x_new_np], axis=0)
        labs_aug = np.concatenate([labs_aug, labs_new_np], axis=0)
        # Append edges
        aug_edges = np.concatenate([aug_edges, new_edges_np], axis=0)
    # Done. Return final augmented data + models
    return (encoder, decoder, edge_gen, classifier,
            X_aug, labs_aug, aug_edges)


##############################################################################
# Test
##############################################################################

if __name__ == "__main__":
    import os
    from utils import *
    from tensorflow import keras
    FILENAME = 'Dorsal_midbrain_cell_bin.h5ad'
    # Load data
    data_path = os.path.join(os.curdir, FILENAME)
    # adata, coordinates = load_data(data_path)
    adata, coordinates = load_mouse_midbrain_data(data_path, 'SS200000131BL_C5C6')

    # Prepare labels
    cell_types = tf.constant(adata.obs['Cell Types'])
    string_lookup_layer = keras.layers.StringLookup(num_oov_indices=0)
    string_lookup_layer.adapt(cell_types)
    encoded_cell_types = string_lookup_layer(cell_types)
    labels = encoded_cell_types  # shape: (N,)
    labels = tf.cast(labels, tf.int32)
    num_classes = len(string_lookup_layer.get_vocabulary())

    import matplotlib.pyplot as plt
    print(string_lookup_layer.get_vocabulary())
    # 1. Convert to NumPy
    labels_np = labels.numpy()
    # 2. Count the occurrences of each label
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    # 3. Prepare x-axis as class names and y-axis as the counts
    visualize_data_distribution(unique_labels, counts)


    print("Class Vocabulary:", string_lookup_layer.get_vocabulary())

    # Prepare node features
    node_states = tf.sparse.to_dense(acquire_sparse_data(adata.X))

    # Build adjacency (edges)
    adj_mat = delaunay_to_graph(coordinates)

    edges = adj_mat.indices
    edges = tf.cast(edges, tf.int32)
    # edges_np = edges.numpy()
    print("Edges shape:", edges.shape)
    print("Node features shape:", node_states.shape)

    data_indices = np.arange(len(labels))


    # Let's say we have a minority class [2], and do 3 "Generate->Train" rounds
    (enc, dec, eg, clf,
     X_aug, y_aug, edges_aug) = graphsmote_knn_2phase(
        node_states, edges, labels,
        num_classes=num_classes,
        minority_classes=[7,8,9,10],
        hidden_dim=128,
        emb_dim=128,
        decoder_hidden=64,
        dropout=0.1,
        lr=1e-3,
        lambda_edge=1,
        k_new_edges=10,
        max_real_edges_sample=10000,
        pretrain_epochs=100,          # short pretrain
        rounds=1,                    # 3 Generate->Train rounds
        train_epochs_per_round=500,
        oversampling_scale=3,
        alpha_synthetic=0.5
    )

    print("Final augmented graph shapes:")
    print("X_aug:", X_aug.shape)
    print("y_aug:", y_aug.shape)
    print("edges_aug:", edges_aug.shape)


    # labels_np = y_aug.numpy()
    # 2. Count the occurrences of each label
    unique_labels, counts = np.unique(y_aug, return_counts=True)
    visualize_data_distribution(unique_labels, counts)
