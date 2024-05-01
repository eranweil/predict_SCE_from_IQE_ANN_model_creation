# main.py

import os
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_processing import read_data, create_results_mesh, plot_predicted_vs_actual_SCE
from model import train_model_ANN, predict_with_model

np.set_printoptions(precision=3, suppress=True)


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings
    print("TensorFlow version:", tf.__version__)

    # Model default Parameters (TODO - take this from a CSV file)
    device_length = 100
    number_of_intervals_in_mesh = 2400

    # Base directory (replace with user's actual path)
    base_directory = "/Users/eranweil/PycharmProjects/MSc_ML/"

    # Read data from the generated database
    training_data_and_labels = read_data(base_directory)

    # Load and split data, handling potential errors
    features_array, labels_array = read_data(base_directory)
    if features_array is None or labels_array is None:
        print("Error loading data. Exiting.")
        exit(1)
    X_train, X_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=0.2, random_state=42)

    num_features = X_train.shape[1]
    num_labels = y_train.shape[1]

    feature_means = np.mean(X_train, axis=0)  # Calculate mean from training data only

    # Train the model with cross-validation and early stopping
    model, _, _ = train_model_ANN(X_train, y_train, X_test, y_test, num_features, num_labels)

    # Save the best model and feature_means
    model.save(base_directory + 'SCE_model')
    np.save(base_directory + 'feature_means.npy', feature_means)

    # Load trained model
    model_path = os.path.join(base_directory, "SCE_model")
    try:
        loaded_model = tf.keras.models.load_model(model_path)
    except OSError:
        print(f"Error: Model not found at '{model_path}'. Please train the model first.")
        exit(1)

    # Load the feature means
    feature_means_path = os.path.join(base_directory, 'feature_means.npy')
    try:
        default_features = np.load(feature_means_path)
    except FileNotFoundError:
        print(f"Error: Feature means not found at '{feature_means_path}'. Please train the model first.")
        exit(1)

    # Path to test files for display
    base_directory_test = os.path.join(base_directory, "IQE_results")
    test_filenames = [
        "IQE_bulk doping_1e+14_p0_1e+15_n0_1e+15_taup_1.000000e+01_taun_1.000000e+01_mup_5.000000e+02_mun_1.450000e+03_L_100.csv",
        "IQE_bulk_doping_1e+14_p0_1e+18_n0_1e+18_taup_1.000000e+02_taun_1.000000e+02_mup_1.077217e+03_mun_3.123930e+03_L_140.csv",
        "IQE_bulk_doping_1e+15_p0_6e+16_n0_6e+16_taup_1_taun_1_mup_3.871318e+02_mun_1.122682e+03_L_220.csv"
    ]

    # Prediction and Saving Results
    results_directory = os.path.join(base_directory, "Predicted_results")
    os.makedirs(results_directory, exist_ok=True)

    for filename in test_filenames:
        # Extract the device parameters string from Filename
        device_params = filename[4:-4]
        # Extract Device Length from Filename
        match = re.search(r"L_(\d+)\.csv", filename)
        if match:
            device_length = int(match.group(1))
        else:
            print(f"Warning: Device length not found in '{filename}'. Using default: {device_length}")

        # Prediction with Error Handling
        try:
            output_mesh, output_prediction = predict_with_model(loaded_model, base_directory_test, filename,
                                                                default_features, device_length,
                                                                number_of_intervals_in_mesh)

            # Prepare the full output to save
            predicted_SCE = np.column_stack((output_mesh, output_prediction))
        except (FileNotFoundError, ValueError) as e:
            print(f"Error predicting SCE for '{filename}': {e}")
            continue  # Skip to next file if error occurs

        # Save Results
        predicted_SCE_file = os.path.join(results_directory, f"predict_SCE_{device_params}.csv")
        np.savetxt(predicted_SCE_file, predicted_SCE, delimiter=",")
        print(f"Saved predictions for '{filename}' to '{predicted_SCE_file}'")
        plot_predicted_vs_actual_SCE(base_directory, device_params)

    # # Predict SCE with the trained model for lab measured IQE
    # test_filename = "SEGEV_IQE_minimized.csv"
    # predicted_SCE_file = os.path.join(results_directory, f"predict_SCE_SEGEV.csv")
    # device_length = 100
    # predicted_SCE = predict_with_model(loaded_model, base_directory_test, test_filename, default_features, device_length, number_of_intervals_in_mesh)
    # np.savetxt(predicted_SCE_file, predicted_SCE, delimiter=",")
    # print(f"Saved predictions for 'SEGEV_IQE.csv' to '{predicted_SCE_file}'")