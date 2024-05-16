# main.py

import os
import numpy as np
import tensorflow as tf
from data_processing import read_data, plot_predicted_vs_actual_SCE, plot_predicted_vs_actual_SCE_SEGEV
from model import train_model_ANN, predict_with_model

np.set_printoptions(precision=3, suppress=True)


if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)

    # Model default Parameters
    default_device_length = 100
    number_of_intervals_in_mesh = 2400

    # Directory and file paths
    base_directory = "/Users/eranweil/PycharmProjects/MSc_ML_code/"  # Base directory (replace with user's actual path)
    base_directory_test = os.path.join(base_directory, "IQE_results")  # Path to test files for display
    iqe_directory_segev = os.path.join(base_directory, "IQE_SEGEV")  # Path to Segev IQE file
    model_path = os.path.join(base_directory, "SCE_model")  # Path to trained model
    feature_means_path = os.path.join(base_directory, 'feature_means.npy')  # Path to feature means
    results_directory = os.path.join(base_directory, "Predicted_results")  # Path to results of model prediction

    # List of files to test and show graphs
    test_filenames = [
        "IQE_bulk_doping_1e+16_p0_1e+18_n0_1e+18_taup_10_taun_10_mup_500_mun_1450_L_100.csv",
        "IQE_bulk_doping_1e+14_p0_1e+18_n0_1e+18_taup_1.000000e+02_taun_1.000000e+02_mup_1.077217e+03_mun_3.123930e+03_L_140.csv",
        "IQE_bulk_doping_1e+15_p0_6e+16_n0_6e+16_taup_1_taun_1_mup_3.871318e+02_mun_1.122682e+03_L_220.csv",
    ]

    # Comment from here to avoid training model
    # Load and split data, handling potential errors
    features_array, extra_features_array, labels_array = read_data(base_directory)
    if features_array is None or extra_features_array is None or labels_array is None:
        print("Error loading data. Exiting.")
        exit(1)

    # Train the model with k-fold, cross-validation and early stopping
    model, feature_means, _, _ = train_model_ANN(features_array, extra_features_array, labels_array, test_size=0.2, random_state=42)

    # Save the best model and the feature mean values of training data for use as default values (instead of NaN)
    np.save(feature_means_path, feature_means)
    model.save(model_path)
    # Comment until here to avoid training model

    # Comment from here to avoid predicting test_filenames
    # Predict and plot predictions for chosen IQE files
    device_index = 0
    for filename in test_filenames:
        # Predict SCE with model
        predicted_SCE = predict_with_model(base_directory_test, filename, model_path, feature_means_path)

        # Text for plots
        device_params = filename[4:-4]

        # Save Results
        predicted_SCE_file = os.path.join(results_directory, f"predict_SCE_{device_params}.csv")
        np.savetxt(predicted_SCE_file, predicted_SCE, delimiter=",")
        print(f"Saved predictions for '{filename}' to '{predicted_SCE_file}'")
        plot_predicted_vs_actual_SCE(base_directory, device_params, device_index)
        device_index += 1
    # Comment until here to avoid predicting test_filenames

    # # Comment from here to avoid predicting Segev results
    # # Predict SCE with the trained model for lab measured IQE
    # test_filename = "IQE_SEGEV_minimized_L_250.csv"
    # device_params = test_filename[4:-4]
    # predicted_SCE = predict_with_model(iqe_directory_segev, test_filename, model_path, feature_means_path)
    # predicted_SCE_file = os.path.join(results_directory, f"predict_SCE_{device_params}.csv")
    # np.savetxt(predicted_SCE_file, predicted_SCE, delimiter=",")
    # print(f"Saved predictions for 'SEGEV_IQE.csv' to '{predicted_SCE_file}'")
    # plot_predicted_vs_actual_SCE_SEGEV(base_directory, device_params)
    # # Comment until here to avoid predicting Segev results
