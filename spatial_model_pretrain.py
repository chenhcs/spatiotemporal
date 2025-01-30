import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CallbackList
from tensorflow import keras
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
)

# Local imports
from utils import (
    acquire_sparse_data,
    closeness_graph,
    delaunay_to_graph,
    load_mouse_midbrain_data,
    top_k_accuracy_score,
    create_model
)
from GraphSMOTE import graphsmote_knn_2phase
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a GNN model with configurable parameters.')
    parser.add_argument('--graph_construction', type=str, default='delauny',
                        choices=['delauny', 'closeness'],
                        help='Types of graph construction.')
    parser.add_argument('--time_point', type=str, default='E12.5',
                        choices=['E12.5', 'E14.5', 'E16.5'],
                        help='Time point dataset to use.')
    parser.add_argument('--threshold', type=int, default=10,
                        help='Threshold for closeness graph.')
    parser.add_argument('--cross_validation', default=False, action='store_true',
                        help='Enable cross-validation.')
    parser.add_argument('--hidden_units', type=int, default=256,
                        help='Number of hidden units in the GNN.')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers in the GNN.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (L2 regularization) factor.')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds for cross-validation.')
    parser.add_argument('--analysis_mode', action='store_true',
                        help='Enable analysis mode for visualizations.')
    parser.add_argument('--save_path', type=str, default=os.path.curdir,
                        help='The location to save the model weights.')
    parser.add_argument('--balance_strategy', type=str, default='none',
                        choices=['weight', 'downsample', 'oversample', 'both', 'none'],
                        help="Approach to class imbalance: 'weight', 'downsample', 'oversample', 'both', or 'none'.")
    return parser.parse_args()


@tf.function
def train_step(model, optimizer, loss_fn, accuracy_fn, top5_acc_fn,
               node_states_train, edges_train, labels_train, batch_indices, class_weights):
    """
    Single gradient update on training data, updating training accuracy/top5 metrics.
    """
    with tf.GradientTape() as tape:
        predictions = model([node_states_train, edges_train], training=True)
        batch_preds = tf.gather(predictions, batch_indices)
        batch_labels = tf.gather(labels_train, batch_indices)

        per_example_loss = loss_fn(batch_labels, batch_preds)
        if class_weights is not None:
            cw = tf.gather(class_weights, tf.cast(batch_labels, tf.int32))
            loss = tf.reduce_mean(per_example_loss * cw) + sum(model.losses)
        else:
            loss = tf.reduce_mean(per_example_loss) + sum(model.losses)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy_fn.update_state(batch_labels, batch_preds)
    top5_acc_fn.update_state(batch_labels, batch_preds)

    return loss


@tf.function
def val_step(model, loss_fn, accuracy_fn, top5_acc_fn,
             node_states, edges, labels, batch_indices):
    """
    Validation step on non-oversampled (original) data.
    """
    predictions = model([node_states, edges], training=False)
    batch_preds = tf.gather(predictions, batch_indices)
    batch_labels = tf.gather(labels, batch_indices)

    per_example_loss = loss_fn(batch_labels, batch_preds)
    loss = tf.reduce_mean(per_example_loss) + sum(model.losses)

    accuracy_fn.update_state(batch_labels, batch_preds)
    top5_acc_fn.update_state(batch_labels, batch_preds)

    return loss


def evaluate_metrics(model, node_states, edges, labels, indices, num_classes, batch_size=1024):
    """
    Evaluate final metrics (accuracy, F1, precision, recall, AUC, top-k accuracy).
    """
    dataset = tf.data.Dataset.from_tensor_slices(indices).batch(batch_size)
    all_preds, all_true = [], []

    for batch_indices in dataset:
        preds = model([node_states, edges], training=False)
        batch_preds = tf.gather(preds, batch_indices)
        batch_labels = tf.gather(labels, batch_indices)
        all_preds.append(batch_preds.numpy())
        all_true.append(batch_labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    pred_labels = np.argmax(all_preds, axis=1)
    acc = accuracy_score(all_true, pred_labels)
    f1 = f1_score(all_true, pred_labels, average='macro')
    precision = precision_score(all_true, pred_labels, average='macro', zero_division=0)
    recall = recall_score(all_true, pred_labels, average='macro', zero_division=0)

    # Compute multi-class AUC
    try:
        all_true_onehot = tf.one_hot(all_true, depth=num_classes).numpy()
        auc_val = roc_auc_score(all_true_onehot, all_preds, average='macro', multi_class='ovr')
    except ValueError:
        auc_val = float('nan')

    top5_acc = top_k_accuracy_score(all_true, all_preds, k=2)

    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc_val,
        "top5_acc": top5_acc,
    }


def downsample_indices(train_indices, labels, classes, fraction=0.2):
    """
    Downsamples specified classes in the training set to the given fraction of their original size.
    """
    if not isinstance(train_indices, tf.Tensor):
        train_indices = tf.convert_to_tensor(train_indices, dtype=tf.int64)
    train_labels_np = tf.gather(labels, train_indices).numpy()

    rng = np.random.default_rng(seed=42)
    balanced_indices = []
    unique_classes, counts = np.unique(train_labels_np, return_counts=True)
    min_count = np.min(counts)

    for cls in unique_classes:
        if cls in classes:
            count_cls = np.count_nonzero(train_labels_np == cls)
            keep_count = int(count_cls * fraction)
            keep_count = max(keep_count, min_count)
            cls_indices = train_indices[train_labels_np == cls].numpy()
            downsampled_cls_indices = rng.choice(cls_indices, size=keep_count, replace=False)
            balanced_indices.append(downsampled_cls_indices)
        else:
            balanced_indices.append(train_indices[train_labels_np == cls].numpy())

    balanced_indices = np.concatenate(balanced_indices)
    rng.shuffle(balanced_indices)
    return balanced_indices


def create_subgraph(all_edges, train_indices):
    """
    Filters the global edges to keep only those among train_indices, then remaps node IDs.
    """
    train_indices = np.array(train_indices)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(train_indices)}

    edges_np = all_edges.numpy()
    mask = np.isin(edges_np[:, 0], train_indices) & np.isin(edges_np[:, 1], train_indices)
    filtered_edges = edges_np[mask]
    remapped = np.array([[old_to_new[u], old_to_new[v]] for (u, v) in filtered_edges], dtype=np.int32)

    return tf.constant(remapped, dtype=tf.int32), old_to_new


def prepare_train_data(train_indices, node_states_full, edges_full, labels, num_classes,
                       balance_strategy, minority_classes=None):
    """
    Creates a subgraph for train_indices, then applies optional oversampling/downsampling strategies.
    Returns node_states_train, labels_train, sub_edges_train (possibly modified).
    """
    sub_edges_train, old2new_map = create_subgraph(edges_full, train_indices)
    node_states_train = []
    labels_train = []

    for old_id in train_indices:
        node_states_train.append(node_states_full[old_id].numpy())
        labels_train.append(labels[old_id].numpy())

    node_states_train = tf.convert_to_tensor(node_states_train, dtype=tf.float32)
    labels_train = tf.convert_to_tensor(labels_train, dtype=tf.int32)

    # Apply oversampling if requested
    if 'oversample' in balance_strategy.lower():
        # Replace minority_classes=[7, 8, 9, 10] with whichever is relevant for your dataset
        (_enc, _dec, _eg, _clf,
         node_states_train, labels_train, sub_edges_train) = graphsmote_knn_2phase(
            node_states_train,
            sub_edges_train,
            labels_train,
            num_classes=num_classes,
            minority_classes=minority_classes or [7, 8, 9, 10],
            hidden_dim=128,
            emb_dim=128,
            decoder_hidden=64,
            dropout=0.1,
            lr=1e-3,
            lambda_edge=1,
            k_new_edges=10,
            max_real_edges_sample=10000,
            pretrain_epochs=100,
            rounds=1,
            train_epochs_per_round=500,
            oversampling_scale=3,
            alpha_synthetic=0.5,
            train=True
        )

    return node_states_train, labels_train, sub_edges_train


def run_training(
    train_idx,
    val_idx,
    node_states_train,
    edges_train,
    labels_train,
    node_states_real,
    edges_real,
    labels_real,
    num_classes,
    args,
    callbacks,
    strategy
):
    """
    Training loop (with optional class weighting), validation on original data.
    Returns final validation metrics, trained model, training history.
    """
    # Build class weights if specified
    class_weights_tf = None
    if 'weight' in args.balance_strategy.lower():
        from sklearn.utils.class_weight import compute_class_weight
        y_train = labels_train.numpy()
        unique_y = np.unique(y_train)
        cw_values = compute_class_weight(
            class_weight='balanced',
            classes=unique_y,
            y=y_train
        )
        class_weights_array = np.ones(num_classes, dtype=np.float32)
        for cls, w in zip(unique_y, cw_values):
            class_weights_array[int(cls)] = w
        class_weights_tf = tf.constant(class_weights_array, dtype=tf.float32)

    with strategy.scope():
        optimizer = keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # Metrics
        accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
        top5_acc_fn = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="train_top5_acc")
        val_accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")
        val_top5_acc_fn = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="val_top5_acc")

        # Create GNN model
        model = create_model(
            input_dim=node_states_train.shape[1],
            hidden_units=args.hidden_units,
            num_heads=3,
            num_classes=num_classes
        )
        model.optimizer = optimizer
        optimizer.build(model.trainable_variables)

    steps_per_epoch = int(np.ceil(len(train_idx) / args.batch_size))
    callback_params = {'epochs': args.num_epochs, 'steps': steps_per_epoch, 'verbose': 1}
    callback_list = CallbackList(callbacks)
    callback_list.set_model(model)
    callback_list.set_params(callback_params)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        np.arange(node_states_train.shape[0])
    ).shuffle(buffer_size=node_states_train.shape[0], seed=42).batch(args.batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_idx).batch(args.batch_size)

    # History
    history = {
        'train_loss': [], 'train_acc': [], 'train_top5_acc': [],
        'val_loss': [], 'val_acc': [], 'val_top5_acc': [], 'lr': []
    }

    best_val_acc = 0.0
    callback_list.on_train_begin(logs={})

    for epoch in range(1, args.num_epochs + 1):
        callback_list.on_epoch_begin(epoch, logs={})

        batch_losses = []
        accuracy_fn.reset_states()
        top5_acc_fn.reset_states()

        step = 0
        for batch_indices in train_dataset:
            callback_list.on_train_batch_begin(step, logs={})
            per_replica_loss = strategy.run(
                train_step,
                args=(
                    model, optimizer, loss_fn,
                    accuracy_fn, top5_acc_fn,
                    node_states_train, edges_train,
                    labels_train, batch_indices,
                    class_weights_tf
                )
            )
            reduced_loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None
            )
            batch_losses.append(reduced_loss)
            callback_list.on_train_batch_end(step, logs={})
            step += 1

        train_loss = tf.math.reduce_mean(batch_losses)
        train_acc = accuracy_fn.result().numpy()
        train_top5 = top5_acc_fn.result().numpy()

        # Validation
        val_losses = []
        val_accuracy_fn.reset_states()
        val_top5_acc_fn.reset_states()

        for batch_indices in val_dataset:
            val_loss_per_replica = strategy.run(
                val_step,
                args=(
                    model, loss_fn,
                    val_accuracy_fn, val_top5_acc_fn,
                    node_states_real, edges_real,
                    labels_real, batch_indices
                )
            )
            reduced_val_loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, val_loss_per_replica, axis=None
            )
            val_losses.append(reduced_val_loss)

        val_loss = tf.math.reduce_mean(val_losses)
        val_acc = val_accuracy_fn.result().numpy()
        val_top5 = val_top5_acc_fn.result().numpy()

        print(f"Epoch {epoch}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"train_top5={train_top5:.4f}, val_loss={val_loss:.4f}, "
              f"val_acc={val_acc:.4f}, val_top5={val_top5:.4f}")

        # Update best val & history
        best_val_acc = max(best_val_acc, val_acc)
        logs = {
            'loss': train_loss.numpy(),
            'acc': train_acc,
            'top5_acc': train_top5,
            'val_loss': val_loss.numpy(),
            'val_acc': val_acc,
            'val_top5_acc': val_top5,
            'lr': optimizer.lr.numpy(),
        }
        callback_list.on_epoch_end(epoch, logs=logs)

        history['train_loss'].append(train_loss.numpy())
        history['train_acc'].append(train_acc)
        history['train_top5_acc'].append(train_top5)
        history['val_loss'].append(val_loss.numpy())
        history['val_acc'].append(val_acc)
        history['val_top5_acc'].append(val_top5)
        history['lr'].append(optimizer.lr.numpy())

        if model.stop_training:
            print("Early stopping triggered.")
            break

    callback_list.on_train_end(logs={})
    final_metrics = {'val_acc': val_acc, 'val_top5_acc': val_top5}

    return final_metrics, model, history


def plot_training_curves(history, title_prefix=''):
    """
    Plots training/validation accuracy and loss from a single-run history dictionary.
    """
    train_acc = np.array(history['train_acc'])
    val_acc = np.array(history['val_acc'])
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])

    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b-', label='Train Acc')
    plt.plot(epochs, val_acc, 'g-', label='Val Acc')
    plt.title(f'{title_prefix}Train vs. Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'r-', label='Train Loss')
    plt.plot(epochs, val_loss, 'c-', label='Val Loss')
    plt.title(f'{title_prefix}Train vs. Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    args = parse_arguments()

    FILENAME = 'Dorsal_midbrain_cell_bin.h5ad'
    data_path = os.path.join(os.curdir, FILENAME)
    adata, coordinates = load_mouse_midbrain_data(data_path, ['SS200000131BL_C5C6'])

    # Prepare labels
    cell_types = tf.constant(adata.obs['Cell Types'])
    string_lookup_layer = keras.layers.StringLookup(num_oov_indices=0)
    string_lookup_layer.adapt(cell_types)
    encoded_cell_types = string_lookup_layer(cell_types)
    labels = tf.cast(encoded_cell_types, tf.int32)
    num_classes = len(string_lookup_layer.get_vocabulary())

    # Convert X to dense
    node_states_full = tf.sparse.to_dense(acquire_sparse_data(adata.X))

    # Build adjacency
    if args.graph_construction == 'delauny':
        adj_mat = delaunay_to_graph(coordinates)
    else:
        adj_mat = closeness_graph(coordinates, args.threshold)
    edges_full = tf.cast(adj_mat.indices, tf.int32)

    data_indices = np.arange(len(labels))

    # Setup multi-GPU strategy (if available)
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: ", strategy.num_replicas_in_sync)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_acc', patience=20, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=1, min_lr=1e-6)
    my_callbacks = [early_stopping, reduce_lr]

    if args.cross_validation:
        kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
        all_fold_metrics = []

        # For aggregated curve plotting
        all_train_accuracies, all_val_accuracies = [], []
        all_train_losses, all_val_losses = [], []

        for fold, (train_indices, test_indices) in enumerate(kfold.split(data_indices, labels)):
            print(f"\n===== Fold {fold+1}/{args.num_folds} =====")
            train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

            # Downsample if requested (for demonstration: downsample class 0 only)
            if 'downsample' in args.balance_strategy.lower():
                train_indices = downsample_indices(train_indices, labels, classes=[0], fraction=0.2)

            # Prepare subgraph + oversampling
            node_states_train, labels_train, sub_edges_train = prepare_train_data(
                train_indices, node_states_full, edges_full, labels,
                num_classes, balance_strategy=args.balance_strategy
            )

            # Train
            fold_metrics, model, history = run_training(
                train_idx=train_indices,
                val_idx=val_indices,
                node_states_train=node_states_train,
                edges_train=sub_edges_train,
                labels_train=labels_train,
                node_states_real=node_states_full,
                edges_real=edges_full,
                labels_real=labels,
                num_classes=num_classes,
                args=args,
                callbacks=my_callbacks,
                strategy=strategy
            )

            # Collect per-epoch metrics
            all_train_accuracies.append(history['train_acc'])
            all_val_accuracies.append(history['val_acc'])
            all_train_losses.append(history['train_loss'])
            all_val_losses.append(history['val_loss'])

            # Evaluate on test split
            test_metrics = evaluate_metrics(
                model,
                node_states=node_states_full,
                edges=edges_full,
                labels=labels,
                indices=test_indices,
                num_classes=num_classes,
                batch_size=args.batch_size
            )
            print(f"Fold {fold+1} Test metrics:", test_metrics)
            all_fold_metrics.append(test_metrics)

        # Report average CV metrics
        metrics_keys = ["acc", "f1", "precision", "recall", "auc"]
        results_dict = {k: [] for k in metrics_keys}
        for fm in all_fold_metrics:
            for k in metrics_keys:
                results_dict[k].append(fm[k])

        print("\nCross-validation results (Test set):")
        for k in metrics_keys:
            vals = np.array(results_dict[k])
            mean_val, std_val = np.mean(vals), np.std(vals)
            print(f"  {k}: {mean_val*100:.2f}% (+/- {std_val*100:.2f}%)")

        # Plot aggregated curves
        all_train_accuracies = np.array(all_train_accuracies)
        all_val_accuracies = np.array(all_val_accuracies)
        all_train_losses = np.array(all_train_losses)
        all_val_losses = np.array(all_val_losses)

        mean_train_acc = np.mean(all_train_accuracies, axis=0)
        std_train_acc = np.std(all_train_accuracies, axis=0)
        mean_val_acc = np.mean(all_val_accuracies, axis=0)
        std_val_acc = np.std(all_val_accuracies, axis=0)

        mean_train_loss = np.mean(all_train_losses, axis=0)
        std_train_loss = np.std(all_train_losses, axis=0)
        mean_val_loss = np.mean(all_val_losses, axis=0)
        std_val_loss = np.std(all_val_losses, axis=0)

        epochs = range(1, len(mean_train_acc) + 1)
        plt.figure(figsize=(12, 5))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, mean_train_acc, 'b-', label='Train Accuracy')
        plt.fill_between(epochs, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc,
                         color='b', alpha=0.2)
        plt.plot(epochs, mean_val_acc, 'g-', label='Val Accuracy')
        plt.fill_between(epochs, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc,
                         color='g', alpha=0.2)
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, mean_train_loss, 'r-', label='Train Loss')
        plt.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss,
                         color='r', alpha=0.2)
        plt.plot(epochs, mean_val_loss, 'c-', label='Val Loss')
        plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss,
                         color='c', alpha=0.2)
        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    else:
        # Simple train/val split (optionally test split)
        train_indices, val_indices = train_test_split(data_indices, test_size=0.1, random_state=42)

        # Downsample if requested
        if 'downsample' in args.balance_strategy.lower():
            train_indices = downsample_indices(train_indices, labels, [0], fraction=0.2)

        # Prepare subgraph + oversampling
        node_states_train, labels_train, sub_edges_train = prepare_train_data(
            train_indices, node_states_full, edges_full, labels,
            num_classes, balance_strategy=args.balance_strategy
        )

        # Train
        final_metrics, model, history = run_training(
            train_idx=train_indices,
            val_idx=val_indices,
            node_states_train=node_states_train,
            edges_train=sub_edges_train,
            labels_train=labels_train,
            node_states_real=node_states_full,
            edges_real=edges_full,
            labels_real=labels,
            num_classes=num_classes,
            args=args,
            callbacks=my_callbacks,
            strategy=strategy
        )
        print("Final Validation Metrics:", final_metrics)

        # Optionally evaluate on separate test_indices if desired
        # test_metrics = evaluate_metrics(...)
        # print("Test Metrics:", test_metrics)

        # Plot single-run training curves
        plot_training_curves(history)

        # Save weights
        model.save_weights(os.path.join(args.save_path, 'GAT_node_classification_model.h5'))


if __name__ == '__main__':
    main()
