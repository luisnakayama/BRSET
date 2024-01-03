import subprocess
import os
import getpass
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np

def organize_dataset(output_dir):
    """
    Organizes the downloaded dataset by moving and renaming files and directories.

    This function performs the following tasks:
    1. Moves the 'labels.csv' file from its original location to the 'data/' directory.
    2. Renames the 'fundus_photos' directory to 'images' within the 'data/' directory.
    3. Removes the 'physionet.org' directory and its contents, cleaning up the directory structure.

    Parameters:
    output_dir (str): The path to the root directory where the dataset was downloaded.

    Example:
    organize_dataset("data/")

    This example would move 'labels.csv' to 'data/' and rename 'fundus_photos' to 'images' within 'data/'.
    It would also remove the 'physionet.org' directory and its contents.

    Note: Make sure to call this function after downloading the dataset using 'download_dataset'.
    """
    
    # Move labels.csv to the data directory
    shutil.move(os.path.join(output_dir, "physionet.org/files/brazilian-ophthalmological/1.0.0/labels.csv"), os.path.join(output_dir, "labels.csv"))

    # Rename fundus_photos to images
    shutil.move(os.path.join(output_dir, "physionet.org/files/brazilian-ophthalmological/1.0.0/fundus_photos"), os.path.join(output_dir, "images"))

    # Remove the physionet.org directory and its contents
    shutil.rmtree(os.path.join(output_dir, "physionet.org"))

def download_dataset(output_dir="data/", url="https://physionet.org/files/brazilian-ophthalmological/1.0.0/"):
    """
    Downloads a dataset from a specified URL and organizes it.

    This function performs the following tasks:
    1. Downloads a dataset from the provided URL using the 'wget' command.
    2. Prompts the user for their PhysioNet username and securely enters their password.
    3. Organizes the downloaded dataset by moving and renaming files and directories using the 'organize_dataset' function.
    
    Parameters:
    output_dir (str, optional): The directory where the dataset will be downloaded and organized. Defaults to "data/".
    url (str, optional): The URL of the dataset to be downloaded. Defaults to the Brazilian Ophthalmological Dataset on PhysioNet.

    Example:
    download_dataset(output_dir="data/", url="https://physionet.org/files/brazilian-ophthalmological/1.0.0/")

    This example would download the Brazilian Ophthalmological Dataset from the provided URL into the "data/" directory.
    It would prompt the user for their PhysioNet username and securely enter their password.
    After downloading, it would organize the dataset using the 'organize_dataset' function.

    Note: You need to have 'wget' and 'shutil' installed to use this function.
    """
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    username = input("Please provide your physionet's username: ")
    # Prompt the user for the password without displaying it
    password = getpass.getpass("Please provide your physionet's password: ")

    # Run the wget command to download the dataset
    command = f'wget -r -N -c -np --user {username} --password {password} {url} -P {output_dir}'
    try:
        subprocess.run(command, shell=True, check=True)
        print("Dataset downloaded successfully.")
        
        organize_dataset(output_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        

def check_columns(row, columns):
    for column in columns:
        if row[column] != 0:
            return 'abnormal'
    return 'normal'

        
def get_dataset(data_dir, download=False, info=False):
    """
    Reads the dataset CSV file and provides information about the DataFrame.

    Parameters:
    data_dir (str): The directory where the dataset is stored.
    download (bool, optional): Whether to download the dataset if it's not already available. Defaults to False.

    Returns:
    pd.DataFrame: The loaded DataFrame containing dataset information.

    Example:
    df = get_dataset("data/", download=True)

    This example would download the dataset if not already available, then load the 'labels.csv' file from the specified directory.
    The resulting DataFrame will contain information about the dataset.

    Note: Make sure to have the 'labels.csv' file in the specified directory.
    """
    
    if download:
        download_dataset(output_dir=data_dir)
        
    print(f'loading csv file in {data_dir}/labels.csv')
    df_path = os.path.join(data_dir, 'labels.csv')
    df = pd.read_csv(df_path)
    
    # Provide information about the DataFrame
    if info:
        print(f"Number of Rows: {df.shape[0]}")
        print(f"Number of Columns: {df.shape[1]}")
        print(f"Column Names: {', '.join(df.columns)}")
        print("\nInfo:")
        print(df.info())

        #print("\nDescription:")
        #print(df.describe())
    
    #
    columns = ['diabetic_retinopathy', 'macular_edema', 'scar', 'nevus',
               'amd', 'vascular_occlusion', 'hypertensive_retinopathy', 
               'drusens', 'hemorrhage', 'retinal_detachment',
               'myopic_fundus', 'increased_cup_disc', 'other'
              ]
    
    df['normality'] = df.apply(check_columns, args=(columns,),  axis=1)

    return df


def plot_labels_distribution(df, column, title=None):
    # Get value counts for the new categories
    category_counts = df[column].value_counts()

    # Plotting the value counts
    category_counts.plot(kind='bar', rot=0, color='skyblue')
    plt.xlabel(f'{column} Categories')
    plt.ylabel('Count')
    if title:
        plt.title(title)
    else:
        plt.title(f'Distribution of {column} Categories')
    #plt.show()

def split_data(df, column, split, undersample=False, undersample_ratio=1.0, random_state=42):
    """
    Split a DataFrame into training and testing sets while stratifying by a specified column.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to be split.
    - column: str
        The column used for stratification.
    - split: float
        The proportion of the dataset to include in the test split (0.0 to 1.0).
    - undersample: bool, default=False
        Whether to undersample the majority class in the training set.
    - undersample_ratio: float, default=1.0
        The percentage of undersampling relative to the minority class (1.0 means undersample to the size of the minority class).


    Returns:
    - train_data: pandas DataFrame
        Training set.
    - test_data: pandas DataFrame
        Testing set.
    """

    # Stratified split
    train_data, test_data = train_test_split(df, test_size=split, stratify=df[column])
    
    if undersample:
        # Determine the class with the fewest samples
        min_class_count = train_data[column].value_counts().min()
        # Determine the class with the most samples ang get the name
        max_class_label = train_data[column].value_counts().idxmax()
        # Downsample the majority classes to balance the dataset
        
        downsampled_classes = []
        for class_label in train_data[column].unique():
            if class_label == max_class_label:
                class_data = train_data[train_data[column] == class_label]
                downsampled_class = resample(class_data, replace=False, n_samples=int(min_class_count*undersample_ratio), random_state=random_state)
                downsampled_classes.append(downsampled_class)
            else:
                downsampled_class = train_data[train_data[column] == class_label]
                downsampled_classes.append(downsampled_class)
        
        # Combine the downsampled classes
        train_data = pd.concat(downsampled_classes)


    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_labels_distribution(train_data, column, title='Train Label Distribution')
    plt.subplot(1, 2, 2)
    plot_labels_distribution(test_data, column, title='Test Label Distribution')
    plt.show()

    return train_data, test_data


if __name__ == "__main__":
    data_dir = "data/"
    get_dataset(data_dir, download=True)
