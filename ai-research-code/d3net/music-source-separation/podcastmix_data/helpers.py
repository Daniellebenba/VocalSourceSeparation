import os
import pandas as pd


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

