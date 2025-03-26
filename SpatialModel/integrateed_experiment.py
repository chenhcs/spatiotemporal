import numpy as np
import matplotlib.pyplot as plt

from model import create_model
import scanpy as sc
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras as keras

import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# Settings for the model and training.
HIDDEN_LAYERS = 4
HIDDEN_DIM = 128
BATCH_SIZE = 32
OUT_DIM = 6
regularizer = tf.keras.regularizers.l1(0.001)

# 5-fold cross validation setting.
n_splits = 5

# Assumed dimensions for the LR interaction feature vector.
# Here we assume the feature vector is built as an outer product of ligand and receptor expressions.
n_ligands = 4  # number of ligand genes
n_receptors = 6  # number of receptor genes
feature_vector_length = n_ligands * n_receptors

# Assume we have 4 sender cell types and 4 receiver cell types,
# with datasets named as "X_i_j.npy" and "y_i_j.npy" for i,j in {0,1,2,3}.
n_senders = 4
n_receivers = 4

# Dictionary to store the mean LR interaction counts (per LR pair) for each cell–cell pair.
cellpair_counts = {}

# Loop over each cell–cell pair dataset.
for i in range(n_senders):
    for j in range(n_receivers):
        print(f"Processing cell–cell pair: X_{i}_{j}")
        # Load the data.
        X = np.load(f"data/LR_interaction_data/X_{i}_{j}.npy")
        y = np.load(f"data/LR_interaction_data/y_{i}_{j}.npy")

        # Initialize a count vector to accumulate binary predictions from each fold.
        lr_count = np.zeros(feature_vector_length, dtype=int)

        # Create the KFold splitter.
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold_idx, (train_index, val_index) in enumerate(kf.split(X)):
            print(f"  Fold {fold_idx}")
            # Split data.
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Create a fresh model for this fold.
            mlp = create_model(OUT_DIM, HIDDEN_LAYERS, HIDDEN_DIM, regularizer=regularizer)
            optimizer = Adam(learning_rate=1e-3)
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
            ]
            mlp.compile(optimizer=optimizer,
                        loss=keras.losses.MeanAbsoluteError(),
                        metrics=[keras.metrics.MeanAbsoluteError()])

            # Train the model (suppressing per-epoch output).
            history = mlp.fit(X_train, y_train,
                              validation_data=(X_val, y_val),
                              batch_size=BATCH_SIZE,
                              epochs=100,
                              callbacks=callbacks,
                              verbose=0)

            # --- NEW CODE BELOW: Plot training curves for each fold ---
            # 1) Plot loss curves
            plt.figure()
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('CosineSimilarity Loss')
            plt.title(f"Training curve for X_{i}_{j}, fold {fold_idx}")
            plt.legend()
            plt.savefig(f"data/interaction_scores/training_curve_{i}_{j}_fold_{fold_idx}.png", dpi=300)
            plt.close()

            # 2) If you want to plot the CosineSimilarity metric as well:
            if 'cosine_similarity' in history.history:
                plt.figure()
                plt.plot(history.history['cosine_similarity'], label='Train CosSim')
                plt.plot(history.history['val_cosine_similarity'], label='Val CosSim')
                plt.xlabel('Epoch')
                plt.ylabel('Cosine Similarity')
                plt.title(f"Cosine Similarity for X_{i}_{j}, fold {fold_idx}")
                plt.legend()
                plt.savefig(f"data/training_curves/training_cossim_{i}_{j}_fold_{fold_idx}.png", dpi=300)
                plt.close()
            # --- END NEW CODE ---

            # --- Extract LR interaction predictions from the first layer ---
            # We assume the first Dense layer maps the LR interaction features.
            first_layer = mlp.layers[0]
            weights, biases = first_layer.get_weights()  # weights shape: (input_dim, hidden_dim)
            # Compute a single importance value for each input feature (LR pair)
            feature_importance = np.linalg.norm(np.abs(weights), axis=1)

            # --- NEW CODE BELOW: Plot feature importances for each fold ---
            plt.figure()
            plt.bar(range(len(feature_importance)), feature_importance)
            plt.xlabel("Feature index")
            plt.ylabel("Importance (L2 norm of abs(weights))")
            plt.title(f"Feature importance for X_{i}_{j}, fold {fold_idx}")
            plt.savefig(f"data/weight_distributions/feature_importance_{i}_{j}_fold_{fold_idx}.png", dpi=300)
            plt.close()
            # --- END NEW CODE ---

            # Define a threshold using the 90th percentile.
            threshold = np.percentile(feature_importance, 90)
            # A binary vector: 1 if the LR pair importance is above the threshold, else 0.
            significant = feature_importance >= threshold

            # Accumulate the significant predictions.
            lr_count += significant.astype(int)

        # Compute the mean count (per fold) for each LR pair for this cell–cell pair.
        mean_lr_count = lr_count / n_splits
        # Reshape the result to (n_ligands, n_receptors) for easier interpretation.
        mean_lr_matrix = mean_lr_count.reshape(n_ligands, n_receptors)
        cellpair_counts[f"X_{i}_{j}"] = mean_lr_matrix


# Determine global min/max for color scale
all_values = []
for i in range(n_senders):
    for j in range(n_receivers):
        matrix = cellpair_counts[f"X_{i}_{j}"]
        all_values.extend(matrix.flatten())
all_values = np.array(all_values)
vmin = all_values.min()
vmax = all_values.max()

# Set up a common normalization and color map
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap("viridis")
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig, axes = plt.subplots(n_senders, n_receivers, figsize=(16, 16))
for i in range(n_senders):
    for j in range(n_receivers):

        ax = axes[i, j]
        matrix = cellpair_counts[f"X_{i}_{j}"]
        # Save each interaction score matrix locally.
        np.save(f'data/interaction_scores/score_{i}_{j}.npy', matrix)
        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar=False,
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(f"Cell–cell pair X_{i}_{j}")
        ax.set_xlabel("Receptor")
        ax.set_ylabel("Ligand")

fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label='Mean Count (over 5 folds)')
plt.tight_layout()
plt.savefig("data/interaction_scores/cellcell_heatmaps.png", dpi=300)
plt.show()
