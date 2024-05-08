# data_processing.py

import re
import os
import csv
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Lists files in a directory, excluding hidden files (starting with '.')
def listdir_nohidden(path):
    try:
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f  # Use generator for memory efficiency
    except FileNotFoundError:
        print(f"Error: Directory not found: '{path}'")
        return []  # Return empty list if directory doesn't exist


# Removes files from data_files1 that don't have corresponding pairs in data_files2.
def clean_empty_data_points(training_data_path, data_files1, data_files2):
    prefix_dataa_files2 = data_files2[0].split('_')[0]
    valid_files1 = []
    for file in data_files1:
        if prefix_dataa_files2 + '_' + '_'.join(file.split('_')[1:]) in data_files2:
            valid_files1.append(file)
    return valid_files1, data_files2


# Reads and preprocesses training and label data. Handles potential file errors.
def read_data(base_directory, noise_std_dev=0.001, default_device_length=100):
    try:
        # Get a list of all training and labels data
        training_data_path = os.path.join(base_directory, 'IQE_results')
        label_data_path = os.path.join(base_directory, 'SCE_results')

        training_files = list(listdir_nohidden(training_data_path))
        label_files = list(listdir_nohidden(label_data_path))

        # Handle database generation errors which have created discrepancy between training data and labels
        training_files, label_files = clean_empty_data_points(training_data_path, training_files, label_files)

        # Initialize lists to store the data
        features_array = []
        extra_features_array = []
        labels_array = []

        # Read training data (IQE) and extract device length from filename
        for file_name in sorted(training_files):
            try:
                # Load data from the CSV file (only first 2 columns needed)
                data = genfromtxt(os.path.join(training_data_path, file_name), delimiter=',', usecols=(0, 1))

                # Check if the two columns have the same length and if there are any NaN values
                if data.shape[1] != 2 or np.isnan(data).any():
                    if np.isnan(data).any():
                        # Replace NaN values with 0
                        data = np.nan_to_num(data, nan=0.0)
                    else:
                        raise ValueError(f"Invalid data format in file: {file_name}")  # Raise error if not NaN

                # Extract the second column (IQE data) to features_array
                features_array.append(data[:, 1])

                # Extract device length from file name, or use default
                match = re.search(r"_L_(\d+)\.csv$", file_name)
                if match:
                    extra_features_array.append(int(match.group(1)))
                else:
                    extra_features_array.append(default_device_length)  # Default value

            except (csv.Error, ValueError) as error:
                print(f"Error reading training file {file_name}: {error}")

        # Read labels data (SCE)
        for file_name in sorted(label_files):
            try:
                data = genfromtxt(os.path.join(label_data_path, file_name), delimiter=',')
                if data.ndim != 2 or data.shape[1] < 2 or not np.all(np.isfinite(data)):
                    raise ValueError(f"Invalid data format in file: {file_name}")
                labels_array.append(data[:, 1])  # Append the second column (SCE data)
            except (csv.Error, ValueError) as error:
                print(f"Error reading training file {file_name}: {error}")

        # Convert lists to NumPy arrays
        features_array = np.array(features_array)
        labels_array = np.array(labels_array)
        extra_features_array = np.array(extra_features_array)

        # Add Gaussian noise for robustness (if enabled)
        if noise_std_dev > 0:
            features_array += np.random.normal(0, noise_std_dev, features_array.shape)
            labels_array += np.random.normal(0, noise_std_dev, labels_array.shape)
            features_array = np.clip(features_array, 0, 1)  # Clip to valid range [0, 1]
            labels_array = np.clip(labels_array, 0, 1)

        return features_array, extra_features_array, labels_array

    except FileNotFoundError as error:
        print(f"File not found: {error}")
        return None, None, None
    except Exception as error:
        print(f"Unexpected error occurred: {error}")
        return None, None, None


# Creates an identical mesh to the one created by Comsol for the database
def create_results_mesh(size_of_mesh_element, number_of_intervals_in_mesh):
    section_0 = 10
    section_1 = 25
    section_2 = 50
    section_3 = 100
    section_4 = 200
    num_points = number_of_intervals_in_mesh + 1
    mesh_points = []

    for i in range(num_points):
        if (i < section_0) or (i > number_of_intervals_in_mesh - section_0):
            if i % 2 == 0:
                mesh_points.append(i * size_of_mesh_element)
        elif(i < section_1) or (i > number_of_intervals_in_mesh - section_1):
            if i % 5 == 0:
                mesh_points.append(i * size_of_mesh_element)
        elif(i < section_2) or (i > number_of_intervals_in_mesh - section_2):
            if i % 10 == 0:
                mesh_points.append(i * size_of_mesh_element)
        elif(i < section_3) or (i > number_of_intervals_in_mesh - section_3):
            if i % 20 == 0:
                mesh_points.append(i * size_of_mesh_element)
        elif(i < section_4) or (i > number_of_intervals_in_mesh - section_4):
            if i % 40 == 0:
                mesh_points.append(i * size_of_mesh_element)
        else:
            if i % 160 == 0:
                mesh_points.append(i * size_of_mesh_element)

    return mesh_points


# Plots predicted vs. actual SCE data for a given device parameters
def plot_predicted_vs_actual_SCE(base_directory, device_param):

    predicted_file = os.path.join(base_directory, "Predicted_results", f"predict_SCE_{device_param}.csv")
    actual_file = os.path.join(base_directory, "SCE_results", f"SCE_{device_param}.csv")
    iqe_file = os.path.join(base_directory, "IQE_results", f"IQE_{device_param}.csv")

    # Read Predicted SCE
    try:
        predicted_data = np.genfromtxt(predicted_file, delimiter=",")
    except FileNotFoundError:
        print(f"Error: Predicted data not found at '{predicted_file}'")
        return

    # Read Actual SCE
    try:
        with open(actual_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            actual_data = np.array([[float(row[0]), float(row[1])] for row in reader])
    except FileNotFoundError:
        print(f"Error: Actual data not found at '{actual_file}'")
        return

    # Read IQE
    try:
        iqe_data = np.genfromtxt(iqe_file, delimiter=",", usecols=(0, 1))
    except FileNotFoundError:
        print(f"Error: IQE data not found at '{iqe_file}'")
        return

    # Calculate Metrics
    mse_values = mean_squared_error(actual_data[:, 1], predicted_data[:, 1], multioutput='raw_values')
    max_mse = np.max(mse_values)
    r2 = r2_score(actual_data[:, 1], predicted_data[:, 1])
    mae_values = mean_absolute_error(actual_data[:, 1], predicted_data[:, 1], multioutput='raw_values')
    max_mae = np.max(mae_values)

    # Plot 1: Predicted vs. Actual SCE
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_data[:, 0], predicted_data[:, 1], label='Predicted SCE', marker='o', linestyle='-', color='blue')
    plt.plot(actual_data[:, 0], actual_data[:, 1], label='Actual SCE', marker='x', linestyle='--', color='orange')

    plt.xlabel('Position', fontsize=12)
    plt.ylabel('SCE', fontsize=12)
    plt.title(f'Predicted vs. Actual SCE for {device_param}', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Add Metric Annotations at the Top Center
    text_x = 0.5
    text_y_start = 0.95
    text_y_offset = 0.05

    plt.text(text_x, text_y_start, f"Max MSE: {max_mse:.3e}", transform=plt.gca().transAxes, ha='center')
    plt.text(text_x, text_y_start - text_y_offset, f"R-squared: {r2:.3f}", transform=plt.gca().transAxes, ha='center')
    plt.text(text_x, text_y_start - 2 * text_y_offset, f"Max MAE: {max_mae:.3e}", transform=plt.gca().transAxes, ha='center')

    # Plot 2: IQE
    plt.figure(figsize=(10, 6))
    plt.plot(iqe_data[:, 0], iqe_data[:, 1], label='IQE', marker='.', linestyle='-', color='green')
    plt.xlabel('Wavelength', fontsize=12)
    plt.ylabel('IQE', fontsize=12)
    plt.title(f'IQE for {device_param}', fontsize=14)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
