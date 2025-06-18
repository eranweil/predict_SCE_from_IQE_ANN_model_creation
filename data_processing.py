# data_processing.py

import re
import os
import csv
import numpy as np
from numpy import genfromtxt
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'  # Add siunitx for unit formatting
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
                features_array.append(data[:, 1])  # Choose wavelengths by granularity of 250[nm] by choosing indexes
                # features_array.append(data[4:-5, 1])  # Segev wavelengths

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
def create_results_mesh(device_length, number_of_intervals_in_mesh):
    section_0 = 10
    section_1 = 25
    section_2 = 50
    section_3 = 100
    section_4 = 200
    num_points = number_of_intervals_in_mesh + 1
    size_of_mesh_element = device_length / number_of_intervals_in_mesh
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
def plot_predicted_vs_actual_SCE(base_directory, device_param, device_index, graph_annotations, standalone_graph=True):

    greek_letterz = [chr(code) for code in range(945, 970)]
    # print(greek_letterz)

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

    # Setting default values for device parameters
    device_params = {
        "bulk_doping": 1e16,
        "hole_concentration": 1e18,
        "electron_concentration": 1e18,
        "hole_lifetime": 10,
        "electron_lifetime": 10,
        "hole_mobility": 500,
        "electron_mobility": 1450,
        "device_length": 250
    }

    # Begin SCE plot
    plt.figure(figsize=(10, 6))
    # Check if dimensions are the same
    if predicted_data.shape[0] == actual_data.shape[0]:
        # If dimensions are the same, use the original data for metrics
        mean_squared_error_values = mean_squared_error(actual_data[:, 1], predicted_data[:, 1], multioutput='raw_values')
        maximum_mean_squared_error = np.max(mean_squared_error_values)
        r_squared = r2_score(actual_data[:, 1], predicted_data[:, 1])
        mean_absolute_error_values = mean_absolute_error(actual_data[:, 1], predicted_data[:, 1], multioutput='raw_values')
        maximum_mean_absolute_error = np.max(mean_absolute_error_values)

        # Extract data from IQE filename
        iqe_components = device_param.split("_")
        device_params['bulk_doping'] = float(iqe_components[2])  # Extract bulk doping in cm^-3
        device_params['hole_concentration'] = float(iqe_components[4])  # Extract hole concentration in cm^-3
        device_params['electron_concentration'] = float(iqe_components[6])  # Extract electron concentration in cm^-3
        device_params['hole_lifetime'] = float(iqe_components[8])  # Extract hole lifetime in us
        device_params['electron_lifetime'] = float(iqe_components[10])  # Extract electron lifetime in us
        device_params['hole_mobility'] = float(iqe_components[12])  # Extract hole mobility in cm^2/Vs
        device_params['electron_mobility'] = float(iqe_components[14])  # Extract electron mobility in cm^2/Vs
        device_params['device_length'] = float(iqe_components[16].replace("L_", ""))  # Extract device length in um

        # Plot both original datasets
        plt.plot(predicted_data[:, 0], predicted_data[:, 1], label='Predicted SCE', marker='o', linestyle='-', color='blue')
        plt.plot(actual_data[:, 0], actual_data[:, 1], label='Actual SCE', marker='x', linestyle='--', color='orange')

    else:
        # If dimensions are different, interpolate to a common x-axis
        minimum_x_value = max(np.min(predicted_data[:, 0]), np.min(actual_data[:, 0]))
        maximum_x_value = min(np.max(predicted_data[:, 0]), np.max(actual_data[:, 0]))

        x_values_for_interpolation = np.linspace(minimum_x_value, maximum_x_value, num=100)
        predicted_values_interpolated = np.interp(x_values_for_interpolation, predicted_data[:, 0], predicted_data[:, 1])
        actual_values_interpolated = np.interp(x_values_for_interpolation, actual_data[:, 0], actual_data[:, 1])

        # Calculate Metrics using interpolated data
        mean_squared_error_values = mean_squared_error(actual_values_interpolated, predicted_values_interpolated, multioutput='raw_values')
        maximum_mean_squared_error = np.max(mean_squared_error_values)
        r_squared = r2_score(actual_values_interpolated, predicted_values_interpolated)
        mean_absolute_error_values = mean_absolute_error(actual_values_interpolated, predicted_values_interpolated, multioutput='raw_values')
        maximum_mean_absolute_error = np.max(mean_absolute_error_values)

        # Plot both original and interpolated datasets
        plt.plot(predicted_data[:, 0], predicted_data[:, 1], label='Predicted SCE', marker='o', linestyle='-', color='blue', alpha=0.5)
        plt.plot(actual_data[:, 0], actual_data[:, 1], label='Actual SCE', marker='x', linestyle='--', color='orange', alpha=0.5)
        plt.plot(x_values_for_interpolation, predicted_values_interpolated, label='Predicted SCE (Interpolated)', linestyle='-', color='blue')
        plt.plot(x_values_for_interpolation, actual_values_interpolated, label='Actual SCE (Interpolated)', linestyle='--', color='orange')

    plt.xlabel(f'X[{greek_letterz[11]}m]', fontsize=22, fontweight='bold')
    plt.ylabel('SCE(X)', fontsize=22, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Add either plot title or plot annotation
    if standalone_graph:
        plt.title(f'Predicted vs. Actual SCE for Device {device_index + 1}', fontsize=26, fontweight='bold')
    else:
        plt.text(-0.15, 1.05, graph_annotations[1], transform=plt.gca().transAxes,
                 ha='left', va='top', fontsize=30, fontweight='bold', color='darkblue')
    plt.legend()
    plt.grid(True)

    # Add Metric Annotations at the Top Center
    text_x_model_stats = 0.5
    text_y_model_stats = [0.9, 0.8, 0.9]
    text_y_offset = 0.05

    plt.text(text_x_model_stats, text_y_model_stats[device_index], f"MSE: {maximum_mean_squared_error:.3e}", transform=plt.gca().transAxes, ha='center', fontstyle='italic', fontweight='bold')
    plt.text(text_x_model_stats, text_y_model_stats[device_index] - text_y_offset, f"R-squared: {r_squared:.3f}", transform=plt.gca().transAxes, ha='center', fontstyle='italic', fontweight='bold')
    plt.text(text_x_model_stats, text_y_model_stats[device_index] - 2 * text_y_offset, f"MAE: {maximum_mean_absolute_error:.3e}", transform=plt.gca().transAxes, ha='center', fontstyle='italic', fontweight='bold')

    # Add device parameters annotation
    # device 1
    text_x_SCE_params = [0.4, 0.5, 0.65]
    text_y_SCE_params = [0.2, 0.22, 0.41]
    device_params_text = (
        f"L: {device_params['device_length']:.1f}[{greek_letterz[11]}m]\t"
        f"Bulk Doping: {device_params['bulk_doping']:.1e}[cm$^{-3}$]\n"
        f"P0: {device_params['hole_concentration']:.1e}[cm$^{-3}$]\t"
        f"N0: {device_params['electron_concentration']:.1e}[cm$^{-3}$]\n"
        f"τp: {device_params['hole_lifetime']:.1f}[{greek_letterz[11]}s]    "
        f"τn: {device_params['electron_lifetime']:.1f}[{greek_letterz[11]}s]\n"
        f"{greek_letterz[11]}n: {device_params['hole_mobility']:.1f}[cm²/Vs]    "
        f"{greek_letterz[11]}p: {device_params['electron_mobility']:.1f}[cm²/Vs]"
    )

    plt.text(text_x_SCE_params[device_index], text_y_SCE_params[device_index], device_params_text,
             transform=plt.gca().transAxes, ha='center', va='center', fontsize=14,
             fontstyle='italic', fontweight='bold')

    plt.tight_layout()

    # Plot 2: Internal Quantum Efficiency
    plt.figure(figsize=(10, 6))
    plt.plot(iqe_data[:, 0], iqe_data[:, 1], label='Internal Quantum Efficiency', marker='.', linestyle='-', color='green')
    plt.xlabel(f'Wavelength {greek_letterz[10]}[{greek_letterz[11]}m]', fontsize=22, fontweight='bold')
    plt.ylabel(f'IQE({greek_letterz[10]})', fontsize=22, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Add either plot title or plot annotation
    if standalone_graph:
        plt.title(f'IQE for Device {device_index + 1}', fontsize=26, fontweight='bold')
    else:
        plt.text(-0.15, 1.05, graph_annotations[0], transform=plt.gca().transAxes,
                 ha='left', va='top', fontsize=30, fontweight='bold', color='darkblue')
    plt.legend()

    # device 1
    text_x_IQE_params = [0.432, 0.46, 0.43]
    text_y_IQE_params = [0.335, 0.322, 0.34]

    plt.text(text_x_IQE_params[device_index], text_y_IQE_params[device_index], device_params_text,
             transform=plt.gca().transAxes, ha='center', va='center', fontsize=12,
             fontstyle='italic', fontweight='bold')
    plt.grid(True)

    plt.tight_layout()

    plt.show()


# Plots predicted vs. actual SCE data for a given device parameters
def plot_predicted_vs_actual_SCE_SEGEV(base_directory, device_param):

    greek_letterz = [chr(code) for code in range(945, 970)]
    # print(greek_letterz)

    predicted_file = os.path.join(base_directory, "Predicted_results", f"predict_SCE_{device_param}.csv")
    segev_file = os.path.join(base_directory, "SCE_SEGEV", f"SCE_{device_param}.csv")

    # Read Predicted SCE
    try:
        predicted_data = np.genfromtxt(predicted_file, delimiter=",")
    except FileNotFoundError:
        print(f"Error: Predicted data not found at '{predicted_file}'")
        return

    # Read Actual SCE
    try:
        with open(segev_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            actual_data = np.array([[float(row[0]), float(row[1])] for row in reader])
    except FileNotFoundError:
        print(f"Error: Actual data not found at '{segev_file}'")
        return

    # Begin SCE plot
    plt.figure(figsize=(10, 6))
    # Check if dimensions are the same
    if predicted_data.shape[0] == actual_data.shape[0]:
        # If dimensions are the same, use the original data for metrics
        mean_squared_error_values = mean_squared_error(actual_data[:, 1], predicted_data[:, 1], multioutput='raw_values')
        maximum_mean_squared_error = np.max(mean_squared_error_values)
        r_squared = r2_score(actual_data[:, 1], predicted_data[:, 1])
        mean_absolute_error_values = mean_absolute_error(actual_data[:, 1], predicted_data[:, 1], multioutput='raw_values')
        maximum_mean_absolute_error = np.max(mean_absolute_error_values)

        # Plot both original datasets
        plt.plot(predicted_data[:, 0], predicted_data[:, 1], label='Predicted SCE', marker='o', linestyle='-', color='blue')
        plt.plot(actual_data[:, 0], actual_data[:, 1], label='SEGEV SCE', marker='x', linestyle='--', color='orange')

    else:
        # If dimensions are different, interpolate to a common x-axis
        minimum_x_value = max(np.min(predicted_data[:, 0]), np.min(actual_data[:, 0]))
        maximum_x_value = min(np.max(predicted_data[:, 0]), np.max(actual_data[:, 0]))

        x_values_for_interpolation = np.linspace(minimum_x_value, maximum_x_value, num=100)
        predicted_values_interpolated = np.interp(x_values_for_interpolation, predicted_data[:, 0], predicted_data[:, 1])
        actual_values_interpolated = np.interp(x_values_for_interpolation, actual_data[:, 0], actual_data[:, 1])

        # Calculate Metrics using interpolated data
        mean_squared_error_values = mean_squared_error(actual_values_interpolated, predicted_values_interpolated, multioutput='raw_values')
        maximum_mean_squared_error = np.max(mean_squared_error_values)
        r_squared = r2_score(actual_values_interpolated, predicted_values_interpolated)
        mean_absolute_error_values = mean_absolute_error(actual_values_interpolated, predicted_values_interpolated, multioutput='raw_values')
        maximum_mean_absolute_error = np.max(mean_absolute_error_values)

        # Plot both original and interpolated datasets
        plt.plot(predicted_data[:, 0], predicted_data[:, 1], label='Predicted SCE', marker='o', linestyle='-', color='blue', alpha=0.5)
        plt.plot(actual_data[:, 0], actual_data[:, 1], label='SEGEV SCE', marker='x', linestyle='--', color='orange', alpha=0.5)
        plt.plot(x_values_for_interpolation, predicted_values_interpolated, label='Predicted SCE (Interpolated)', linestyle='-', color='blue')
        plt.plot(x_values_for_interpolation, actual_values_interpolated, label='SEGEV SCE (Interpolated)', linestyle='--', color='orange')

    plt.xlabel(f'X[{greek_letterz[11]}m]', fontsize=12, fontweight='bold')
    plt.ylabel('SCE(X)', fontsize=12, fontweight='bold')
    plt.title(f'Predicted SCE vs. SEGEV SCE', fontsize=26, fontweight='bold')
    plt.legend()
    plt.grid(True)

    # Add Metric Annotations at the Top Center
    text_x_model_stats = 0.5
    text_y_model_stats = 0.9
    text_y_offset = 0.05

    plt.text(text_x_model_stats, text_y_model_stats, f"MSE: {maximum_mean_squared_error:.3e}", transform=plt.gca().transAxes, ha='center', fontstyle='italic', fontweight='bold')
    plt.text(text_x_model_stats, text_y_model_stats - text_y_offset, f"R-squared: {r_squared:.3f}", transform=plt.gca().transAxes, ha='center', fontstyle='italic', fontweight='bold')
    plt.text(text_x_model_stats, text_y_model_stats - 2 * text_y_offset, f"MAE: {maximum_mean_absolute_error:.3e}", transform=plt.gca().transAxes, ha='center', fontstyle='italic', fontweight='bold')

    plt.grid(True)
    plt.tight_layout()
    plt.show()
