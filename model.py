# model.py

import re
import os
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from data_processing import create_results_mesh


# Trains an ANN model using k-fold cross-validation and early stopping.
def train_model_ANN(features_array, extra_features_array, labels_array, test_size=0.2, random_state=42, k=5, epochs=200, batch_size=32, verbose=1):

    # Splitting data
    X_train, X_test, extra_features, extra_features_test, y_train, y_test = train_test_split(features_array, extra_features_array, labels_array, test_size=test_size, random_state=random_state)
    feature_means = np.mean(X_train, axis=0)

    num_features = X_train.shape[1]
    num_labels = y_train.shape[1]

    # Initialize lists to store evaluation metrics
    train_losses = []
    val_losses = []

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Lists to store results for each fold
    fold_results = []

    # K-fold Cross-Validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(features_array)):

        # Split both feature sets and labels
        X_train, X_val = features_array[train_index], features_array[val_index]
        extra_train, extra_val = extra_features_array[train_index], extra_features_array[val_index]
        y_train, y_val = labels_array[train_index], labels_array[val_index]

        # Create the model using the Functional API
        input_iqe = tf.keras.Input(shape=(num_features,))   # Input for IQE features
        input_extra = tf.keras.Input(shape=(1,))  # Input for the extra feature

        # Branch for IQE features
        x1 = tf.keras.layers.Dense(256, activation='relu')(input_iqe)

        # Branch for the extra feature
        x2 = tf.keras.layers.Dense(16, activation='relu')(input_extra)

        # Combine the branches
        concatenated = tf.keras.layers.Concatenate()([x1, x2])

        # Common hidden layers
        x = tf.keras.layers.Dense(256, activation='relu')(concatenated)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        output = tf.keras.layers.Dense(num_labels, activation='sigmoid')(x)

        # Define the model with multiple inputs
        model = tf.keras.Model(inputs=[input_iqe, input_extra], outputs=output)

        # Compile the model with weight decay and learning rate
        learning_rate = 0.001
        weight_decay = 1e-4
        model.compile(Adam(learning_rate=learning_rate, decay=weight_decay),
                      loss='mean_squared_error',
                      metrics=['mse'])

        # Train the model with early stopping
        history = model.fit(
            [X_train, extra_train],  # Use both input sets
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([X_val, extra_val], y_val),  # Use both input sets for validation
            callbacks=[early_stopping],
            verbose=verbose
        )

        # Append evaluation metrics
        train_losses.append(history.history['loss'])
        val_losses.append(history.history['val_loss'])

        print(f"Fold {fold + 1}: Train Loss = {history.history['loss'][-1]}, Val Loss = {history.history['val_loss'][-1]}")

        # After training, get the best epoch
        best_epoch = early_stopping.stopped_epoch + 1  # EarlyStopping counts from 0

        # Predict and calculate metrics on validation set
        val_predictions = model.predict([X_val, extra_val])
        val_mse = mean_squared_error(y_val, val_predictions)
        val_r2 = r2_score(y_val, val_predictions)
        val_mae = mean_absolute_error(y_val, val_predictions)

        # Store fold results
        fold_results.append({
            "fold": fold + 1,
            "val_index": val_index,
            "epochs": best_epoch,
            "train_loss": history.history['loss'][-1],
            "val_loss": history.history['val_loss'][-1],
            "val_mse": val_mse,
            "val_r2": val_r2,
            "val_mae": val_mae
        })

    # After the loop, print results for each fold
    print("\nFold Results:")
    for result in fold_results:
        print(f"\nFold {result['fold']}:")
        print(f"  Chosen for Validation: Index {result['val_index']}")
        print(f"  Number of Epochs: {result['epochs']}")
        print(f"  Train Loss: {result['train_loss']:.4f}")
        print(f"  Val Loss: {result['val_loss']:.4f}")
        print(f"  Val MSE: {result['val_mse']:.4f}")
        print(f"  Val R^2: {result['val_r2']:.4f}")
        print(f"  Val MAE: {result['val_mae']:.4f}")

    # After cross-validation, retrain the best model on the entire dataset (optional)
    best_model = model

    # Evaluate and predict using both feature sets
    train_loss, train_mse = best_model.evaluate([X_train, extra_train], y_train)
    val_loss, val_mse = best_model.evaluate([X_val, extra_val], y_val)
    test_loss, test_mse = best_model.evaluate([X_test, extra_features_test], y_test)

    # Predict on train, val, and test sets
    train_predictions = best_model.predict([X_train, extra_train])
    val_predictions = best_model.predict([X_val, extra_val])
    test_predictions = best_model.predict([X_test, extra_features_test])

    # Calculate additional metrics
    train_r2 = r2_score(y_train, train_predictions)
    val_r2 = r2_score(y_val, val_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    train_mae = mean_absolute_error(y_train, train_predictions)
    val_mae = mean_absolute_error(y_val, val_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    print(f'Final Train Loss: {train_loss}, Train MSE: {train_mse}, Train R^2: {train_r2}, Train MAE: {train_mae}')
    print(f'Final Val Loss: {val_loss}, Val MSE: {val_mse}, Val R^2: {val_r2}, Val MAE: {val_mae}')
    print(f'Final Test Loss: {test_loss}, Test MSE: {test_mse}, Test R^2: {test_r2}, Test MAE: {test_mae}')

    # Return the best model, feature means, and evaluation metrics
    return best_model, feature_means, train_losses, val_losses


# Predicts SCE from an IQE file using the trained model.
def predict_with_model(base_directory, filename, model_path, feature_means_path, num_intervals=2400, default_device_length=100):
    # Load trained model
    try:
        loaded_model = tf.keras.models.load_model(model_path)
    except OSError:
        print(f"Error: Model not found at '{model_path}'. Please train the model first.")
        exit(1)

    # Load the feature means
    try:
        default_features = np.load(feature_means_path)
    except FileNotFoundError:
        print(f"Error: Feature means not found at '{feature_means_path}'. Please train the model first.")
        exit(1)

    # Read the input data (IQE) and handle NaN values
    input_data = np.array(genfromtxt(os.path.join(base_directory, filename), delimiter=',', usecols=(0, 1)))
    input_data = np.expand_dims(input_data[:, 1], axis=0)  # IQE Data
    nan_indices = np.isnan(input_data)
    input_data[nan_indices] = np.take(default_features, nan_indices.nonzero()[1])

    # Extract the device length from the filename
    match = re.search(r"_L_(\d+)\.csv$", filename)
    if match:
        device_length = int(match.group(1))
    else:
        device_length = default_device_length  # Default if not found

    # Prepare the device length as a separate input for the model
    device_length_input = np.array([[device_length]])

    # Predict using both the IQE data and the device length
    output_prediction = loaded_model.predict([input_data, device_length_input])  # Use loaded_model

    # Create the output mesh
    output_mesh = create_results_mesh(device_length / num_intervals, num_intervals)

    # Create SCE data structure
    predicted_SCE = np.column_stack((np.array(output_mesh), output_prediction[0]))

    return predicted_SCE
