# model.py

import os
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from data_processing import create_results_mesh


# Trains an ANN model using k-fold cross-validation and early stopping.
def train_model_ANN(features_array, labels_array, X_test, y_test, num_features, num_labels, k=5, epochs=200, batch_size=32, verbose=1):

    # Initialize lists to store evaluation metrics
    train_losses = []
    val_losses = []

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # K-fold Cross-Validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(features_array)):
        X_train, X_val = features_array[train_index], features_array[val_index]
        y_train, y_val = labels_array[train_index], labels_array[val_index]

        # Create the model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, input_shape=(num_features,), activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(num_labels, activation='sigmoid'))

        # Compile the model with weight decay and learning rate
        learning_rate = 0.001
        weight_decay = 1e-4
        model.compile(Adam(learning_rate=learning_rate, decay=weight_decay),
                      loss='mean_squared_error',
                      metrics=['mse'])

        # Train the model with early stopping
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=verbose)

        # Append evaluation metrics
        train_losses.append(history.history['loss'])
        val_losses.append(history.history['val_loss'])

        print(f"Fold {fold + 1}: Train Loss = {history.history['loss'][-1]}, Val Loss = {history.history['val_loss'][-1]}")

    # After cross-validation, retrain the best model on the entire dataset (optional)
    best_model = model

    # Evaluate the best model on the entire dataset
    train_loss, train_mse = best_model.evaluate(X_train, y_train)
    val_loss, val_mse = best_model.evaluate(X_val, y_val)
    test_loss, test_mse = best_model.evaluate(X_test, y_test)

    # Predict on train, val, and test sets
    train_predictions = best_model.predict(X_train)
    val_predictions = best_model.predict(X_val)
    test_predictions = best_model.predict(X_test)

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
    return best_model, train_losses, val_losses


# Predicts SCE from an IQE file using the trained model.
def predict_with_model(model, base_directory, filename, default_features, L, num_intervals):
    input_data = np.array(genfromtxt(os.path.join(base_directory, filename), delimiter=','))
    input_data = np.expand_dims(input_data[:, 1], axis=0)

    nan_indices = np.isnan(input_data)
    input_data[nan_indices] = np.take(default_features, nan_indices.nonzero()[1])

    output_prediction = model.predict(input_data)
    output_mesh = create_results_mesh(L / num_intervals, num_intervals)
    return np.array(output_mesh), output_prediction[0]