import os
import pandas as pd
import numpy as np

def concatenate_eval_csv_files(folder_path):
    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate over files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a CSV file
        if file_name.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)

            # Read the CSV file into a DataFrame and append to the list
            df = pd.read_csv(file_path)
            dfs.append(df)

    # Concatenate all DataFrames in the list along rows
    concatenated_df = pd.concat(dfs, ignore_index=True)

    return concatenated_df


def compute_ISDR(true_source, separated_signal):
    # Compute energy of true source
    true_source_energy = np.sum(np.square(np.abs(true_source)))

    # Compute energy of spatial distortion
    spatial_distortion = true_source - separated_signal
    spatial_distortion_energy = np.sum(np.square(np.abs(spatial_distortion)))

    # Compute ISDR
    ISDR = true_source_energy / spatial_distortion_energy

    return ISDR
