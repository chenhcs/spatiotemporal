import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from plot_attention_graph import visualize_delauny_graph, compute_and_visualize_entropy, plot_confusion_matrix, \
    plot_pairwise_attention_weights, compute_attention_score, plot_attention_matrices
import os
from utils import load_data, encode_cell_types

os.environ['BROWSER'] = r'C:\Program Files\Mozilla Firefox\firefox.exe'  # Use raw string or double backslashes

# Import custom modules
from simulation import h5ad
from spatial_model import (
    delaunay_to_graph,
    GraphAttentionNetwork,
    GraphConvolutionalNetwork,
    GraphSAGENetwork, closeness_graph,
)


def build_model(node_states, adj_mat, output_dim, model_type='GraphSAGE', hidden_units=128, num_layers=3):
    """Build the GNN model based on the specified type."""
    if model_type == 'GraphSAGE':
        model = GraphSAGENetwork(node_states, adj_mat, hidden_units, num_layers, output_dim)
    elif model_type == 'GCN':
        model = GraphConvolutionalNetwork(node_states, adj_mat, hidden_units, num_layers, output_dim)
    elif model_type == 'GAT':
        model = GraphAttentionNetwork(
            node_states, adj_mat.indices, hidden_units, num_heads=2, num_layers=num_layers, output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model

def main():
    # Load data
    data_path = os.path.join(os.curdir, 'simulation/st_simu.h5ad')
    adata, coordinates = load_data(data_path)



    # Encode cell types
    cell_types, class_values = encode_cell_types(adata)
    output_dim = len(class_values)
    print(f"Number of classes: {output_dim}")

    # Convert gene expression data to tensor
    node_states = tf.convert_to_tensor(adata.X)

    # Construct adjacency matrix
    # adj_mat = delaunay_to_graph(coordinates)
    adj_mat = closeness_graph(coordinates)
    print("Edges shape:\t\t", adj_mat.indices.shape)
    print("Node features shape:", node_states.shape)

    # Tune to True if want to visualize the delauny graph and corresponding attention weights.
    analysis_mode = True

    # Select Model Type
    model_type = 'GAT'

    # Tuen to true if operating cross-validation
    cross_validation = False

    # Define hyperparameters
    HIDDEN_UNITS = 128
    NUM_LAYERS = 3
    NUM_EPOCHS = 200
    BATCH_SIZE = 256
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.01
    NUM_FOLDS = 5  # Number of folds for cross-validation

    # Prepare data indices for KFold
    num_samples = len(cell_types)
    data_indices = np.arange(num_samples)

    # Initialize KFold
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # Lists to store histories and test accuracies
    histories = []
    test_accuracies = []

    # Variables to track the best model
    best_test_accuracy = 0.0
    best_model_path = 'gnn_model/best_{}_model'.format(model_type)  # Path to save the best model

    if cross_validation:
        for fold, (train_indices, test_indices) in enumerate(kfold.split(data_indices)):
            print(f'Fold {fold + 1}/{NUM_FOLDS}')

            # Get train and test labels
            train_labels = cell_types[train_indices]
            test_labels = cell_types[test_indices]

            # Define optimizer, loss, and metrics
            optimizer = keras.optimizers.Adam(LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", min_delta=1e-5, patience=20, restore_best_weights=True
            )
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", min_delta=1e-5, factor=0.5, patience=10, verbose=1
            )

            # Build and compile the model
             # Change to 'GCN' or 'GraphSAGE' as needed
            gnn_model = build_model(
                node_states,
                adj_mat,
                output_dim,
                model_type=model_type,
                hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS,
            )
            gnn_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

            # Train the model
            history = gnn_model.fit(
                x=train_indices,
                y=train_labels,
                validation_data=(test_indices, test_labels),
                batch_size=BATCH_SIZE,
                epochs=NUM_EPOCHS,
                callbacks=[early_stopping, reduce_lr],
                verbose=2,
            )

            # Evaluate the model
            _, test_accuracy = gnn_model.evaluate(x=test_indices, y=test_labels, verbose=0)


            print("-" * 76 + f"\nTest Accuracy: {test_accuracy * 100:.1f}%")

            # Store history and test accuracy
            histories.append(history)
            test_accuracies.append(test_accuracy)

            # Save the model if it has the best test accuracy so far
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                gnn_model.save(best_model_path, save_format='tf')
                print(f"Best model updated and saved with accuracy: {best_test_accuracy * 100:.2f}%")

    # Cross-Validation takes a long time to train; we can just run the experiment once
    else:
        train_indices, test_indices = train_test_split(data_indices, test_size=.2, random_state=42)
        # Get train and test labels
        train_labels = cell_types[train_indices]
        test_labels = cell_types[test_indices]

        # Define optimizer, loss, and metrics
        optimizer = keras.optimizers.Adam(LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=1e-5, patience=20, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", min_delta=1e-5, factor=0.5, patience=10, verbose=1
        )

        # Build and compile the model
        # Change to 'GCN' or 'GraphSAGE' as needed
        gnn_model = build_model(
            node_states,
            adj_mat,
            output_dim,
            model_type=model_type,
            hidden_units=HIDDEN_UNITS,
            num_layers=NUM_LAYERS,
        )
        gnn_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

        # Train the model
        gnn_model.fit(
            x=train_indices,
            y=train_labels,
            validation_data=(test_indices, test_labels),
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            callbacks=[early_stopping, reduce_lr],
            verbose=2,
        )

        #Save the model to local
        gnn_model.save(best_model_path, save_format='tf')
        # Evaluate the model
        _, test_accuracy = gnn_model.evaluate(x=test_indices, y=test_labels, verbose=0)

        print("-" * 76 + f"\nTest Accuracy: {test_accuracy * 100:.1f}%")

    # Visualize the Neighborhood graph, only applicable to GAT model
    if analysis_mode and model_type == 'GAT':
        # edge_index = tf.transpose(adj_mat.indices)
        # edges = tf.concat([edge_index, tf.stack([edge_index[1], edge_index[0]])], axis = 1)
        # tf.concat([adj_mat.indices, tf.stack(adj_mat.indices[1, adj_mat.indices[0]])])
        predictions = gnn_model.predict_step(test_indices)
        predictions = np.argmax(predictions, axis=1)

        cfm = tf.math.confusion_matrix(
            test_labels,
            predictions,
            num_classes=None,
            weights=None,
            dtype=tf.dtypes.int32,
            name=None
        )
        print("Confusion Matrix: ", cfm)
        plot_confusion_matrix(cfm, class_values)

        attention_weights = gnn_model.attention_scores(node_states, adj_mat.indices)
        # print(attention_weights[0].shape)
        # visualize_delauny_graph(coordinates, adata.obs['Cell Types'], adj_mat, attention_weights)
        compute_and_visualize_entropy(adj_mat, attention_weights)
        plot_pairwise_attention_weights(adj_mat, attention_weights, cell_types, class_values)
        attention_scores_l1, attention_scores_l2 = compute_attention_score(adj_mat.indices, cell_types, class_values, attention_weights)
        # for attention_score in attention_scores:
        #     print(attention_score)
        plot_attention_matrices(attention_scores_l1, class_values, 'l1')
        plot_attention_matrices(attention_scores_l2, class_values, 'l2')



    plt.figure(figsize=(12, 4))
    # Plot training and validation loss and accuracy for each fold
    for i, history in enumerate(histories):
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        # plt.title(f'Fold {i + 1} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        # plt.title(f'Fold {i + 1} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # plt.show()
        plt.savefig('fig/training_curve_{}.png'.format(model_type))
        plt.show()

        # Plot test accuracies over folds
        plt.figure()
        plt.plot(range(1, NUM_FOLDS + 1), [acc * 100 for acc in test_accuracies], marker='o')
        plt.title('Test Accuracy over Folds')
        plt.xlabel('Fold')
        plt.ylabel('Test Accuracy (%)')
        plt.xticks(range(1, NUM_FOLDS + 1))
        plt.grid(True)

        plt.savefig('fig/restuls_{}.png'.format(model_type))
        plt.show()
if __name__ == "__main__":
    main()
