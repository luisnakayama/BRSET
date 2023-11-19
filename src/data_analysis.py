from src.get_dataset import get_dataset

import os
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image

import missingno as msno
import matplotlib.pyplot as plt

import skimage
import cv2
import random

### Metadata:

# Missing values:
def show_missing(df):
    """
    Visualizes and reports missing values in a Pandas DataFrame.

    Parameters:
    - df (DataFrame): The input Pandas DataFrame to check for missing values.

    Returns:
    None

    This function performs the following tasks:
    1. Displays a missing data matrix plot to visualize the location of missing values in the DataFrame.
    2. Prints the percentage of missing values in each column with missing data.
    3. Creates a bar plot to visualize the percentage of missing values in each column.

    Example usage:
    show_missing(df)
    """
    
    # Visualize the percentage of missing data using a bar plot
    msno.matrix(df)
    
    
    # Print the percentage of missing values in the columns with missing
    features_with_nan = [features for features in df.columns if df[features].isnull().sum()>=1]
    print(f'{len(features_with_nan)} columns with missing values detected:')

    for features in features_with_nan:
        print(features , np.round(df[features].isnull().mean(),4) * 100 , '% missing values')

    
    # Create a bar plot to visualize missing values
    # Calculate the percentage of missing data in each column
    missing_percentage = df.isnull().mean() * 100
    missing_percentage
    
    plt.figure(figsize=(10, 6))
    missing_percentage.plot(kind='bar', color='skyblue')
    plt.title('Percentage of Missing Values per Column')
    plt.xlabel('Columns')
    plt.ylabel('Percentage Missing')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    os.makedirs('Profile', exist_ok=True)
    plt.savefig('Profile/Missing values')
    plt.show()
    


# Categorical Data:
def plot_categorical_column(df, column, save=False):
    """
    Generate pie charts and bar plots to visualize categorical columns in a Pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the categorical data.
    n (int, optional): The maximum number of unique values allowed in a column for it to be considered categorical.
                       Default is 10.
    categorical_columns (list of str, optional): A list of column names to be visualized. If not specified,
                                                 the function will automatically identify categorical columns.

    This function iterates through the specified or automatically identified categorical columns in the DataFrame
    and creates two plots for each column: a pie chart illustrating the distribution of categories as percentages
    and a bar plot showing the count of each category.

    The function utilizes the 'plot_categorical_column' function to generate the individual plots for each column.

    Args:
    df (pandas.DataFrame): The DataFrame containing the categorical data.
    n (int, optional): The maximum number of unique values allowed in a column for it to be considered categorical.
    categorical_columns (list of str, optional): A list of column names to be visualized.

    Returns:
    None
    """
    
    
    values = df[column].value_counts()

    print(df[column].value_counts(normalize=True))
    print('')
    
    if column == 'patient_sex':
        column = 'Patient Sex'

    # Plot a pie chart with improved quality
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121, aspect='equal')
    explode = [0.1] * len(values)  # Explode a slice for emphasis
    colors = plt.cm.Paired(range(len(values)))

    plt.pie(values, labels=values.index, autopct='%1.1f%%', startangle=140, 
            pctdistance=0.85, explode=explode, colors=colors)
    plt.title(f'Distribution of Column {column}')
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Plot a bar chart with improved quality
    plt.subplot(122)
    values.plot(kind='bar', color='skyblue')
    plt.title(f'Distribution of Column {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    if save:
        os.makedirs('Profile', exist_ok=True)
        plt.savefig(f'Profile/Plots {column}')
    
    plt.show()

def plot_categorical_columns(df, n=10, categorical_columns=None, save=False):
    """    
    Generate pie charts and bar plots to visualize categorical columns in a Pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the categorical data.
    n (int): The maximum number of unique values for a column to be considered categorical. Default is 10.
    categorical_columns (list or None): A list of column names to be considered as categorical columns.
        If None, the function automatically identifies categorical columns based on the 'n' parameter.

    This function creates pie charts and bar plots for the specified categorical columns. The function
    enhances the quality of the plots by customizing colors, labels, and formatting.

    Args:
    df (pandas.DataFrame): The DataFrame containing the categorical data.
    n (int): The maximum number of unique values for a column to be considered categorical. Default is 10.
    categorical_columns (list or None): A list of column names to be considered as categorical columns.
        If None, the function automatically identifies categorical columns based on the 'n' parameter.

    Returns:
    None

    """
    if not(categorical_columns):
        # Identify categorical columns with unique values less than or equal to n
        categorical_columns = []
        for column in df.columns:
            if df[column].nunique() <= n:
                categorical_columns.append(column)
    
    # Create pie charts and bar plots for each categorical column
    for column in categorical_columns:
        print('#'*90)
        print('#'*40, f' {column} ', '#'*40)
        print('#'*90)
        
        plot_categorical_column(df, column, save=save)
        
## Numeric data
def plot_continuous(df, column):
    """
    Plot the distribution of a continuous numerical variable in a DataFrame using a histogram.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - column (str): The name of the column to be visualized.

    Returns:
    None

    This function generates a histogram of the specified column's data distribution, calculates the mean of the data,
    and plots a vertical line representing the mean. It also displays descriptive statistics for the column.

    Example usage:
    plot_continuous(dataframe, 'column_name')

    Note:
    - The function uses Matplotlib for plotting and assumes that the necessary libraries (pandas, numpy, and Matplotlib)
      are imported and available in the current environment.

    """
    
    #Age description
    if column == 'patient_age':
        print('#'*90)
        print('#'*40, f' Age ', '#'*40)
        print('#'*90)
    else:
        print('#'*90)
        print('#'*40, f' {column} ', '#'*40)
        print('#'*90)
        
    print(df[column].describe())
    print()
    
    # Plot age distribution
    plt.figure(figsize=(20, 10))
    plt.hist(df[column], bins=70, edgecolor='black', color='skyblue', alpha=0.7)

    # Calculating the mean
    mean_age = np.mean(df['patient_age'])
    
    if column == 'patient_age':
        label = 'Age'
    else:
        label = column

    # Plotting the mean line
    plt.axvline(mean_age, color='red', linestyle='--', linewidth=2)
    plt.xlabel(label, fontdict={'fontsize': 20})
    plt.ylabel('Frequency', fontdict={'fontsize': 20})
    plt.title(f'{label} Distribution', fontdict={'fontsize': 25})
    plt.grid(True)

    # Adjusting tick labels and font sizes
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.show()


### Images:

def show_random_images(dataset_dir, image_folder, id_column, class_column, max_images_per_class, save=True):
    """
    Display random images from a dataset with their corresponding class labels.

    Parameters:
    dataset_dir (str): The directory where the dataset and labels are stored.
    image_folder (str): The subdirectory where the images are located.
    id_column (str): The column name containing unique image identifiers.
    class_column (str): The column name containing class labels.
    max_images_per_class (int): The maximum number of images to display per class.

    Example:
    show_random_images(
        dataset_dir="data/",
        image_folder="images",
        id_column="ID",
        class_column="Class",
        max_images_per_class=3
    )

    This example will display a random selection of images from the 'images' subdirectory of the 'data/' directory.
    It will use the 'ID' column as the unique image identifier and the 'Class' column for class labels.
    Up to 3 random images will be shown for each class.

    Note: Ensure that the required image files are present in the specified directory and subdirectory.
    """
    
    # Get the labels
    df = get_dataset(dataset_dir, download=False, info=False)
    
    # Get the images:
    image_folder = os.path.join(dataset_dir, image_folder)
    
    grouped_df = df.groupby(class_column)
    sample_ids = []

    for _, group_df in grouped_df:
        num_images = min(len(group_df), max_images_per_class)
        image_ids = random.sample(group_df[id_column].tolist(), num_images)

        sample_ids.extend(image_ids)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,10))

    for i, image_id in enumerate(sample_ids):
        image_path = os.path.join(image_folder, f'{image_id}.jpg')

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        subplot_row = i // 3
        subplot_col = i % 3

        ax = axes[subplot_row, subplot_col]

        class_value = df.loc[df[id_column] == image_id, class_column].values[0]
        file_name = os.path.basename(image_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Diagnosis: {class_value}')

    plt.tight_layout()
    if class_column == 'diabetic_retinopathy':
        column = 'Diabetic Retinopathy'
    else:
        column = class_column
    plt.suptitle(f'Sample Images with Class Labels for {column}', fontsize=16)
    
    if save:
        os.makedirs('Profile', exist_ok=True)
        plt.savefig(f'Profile/Sample images {column}')
    plt.show()


def show_image(image, size=8):
    """
    Display a single image and its shape.

    Parameters:
    image (numpy.ndarray): The image to display.
    size (int, optional): The size of the displayed image. Defaults to 8.

    Example:
    image = skimage.io.imread("image.jpg")
    show_image(image, size=10)

    This example will display the specified image with a size of 10.

    Note: The image parameter should be a NumPy array representing the image.
    """
    print(f'The shape of the image is: {image.shape}')
    plt.figure(figsize=(size, size))
    plt.imshow(image)
    plt.show()
    

def show_images(data_root, image_dir="images", n=5):
    """
    Display a random selection of images from a directory.

    Parameters:
    data_root (str): The root directory where the images are stored.
    image_dir (str, optional): The subdirectory where the images are located. Defaults to "images".
    n (int, optional): The number of random images to display. Defaults to 5.

    Example:
    show_images(data_root="data/", image_dir="images", n=3)

    This example will display 3 random images from the "images" subdirectory of the "data/" directory.

    Note: Ensure that the required image files are present in the specified directory and subdirectory.
    """
    images_path = os.path.join(data_root, image_dir)
    
    files = os.listdir(images_path)

    # select n random files
    files = np.random.choice(files, n)

    for file in files:
        # Get the path to the image
        image_path = os.path.join(images_path, file)
        print(f'---------- {file} ----------')
        
        # read the file
        image = skimage.io.imread(image_path)
        show_image(image)
        
        
        
def get_image_statistics(image_path):
    """
    Retrieve statistics for an image, including its dimensions and mean color values.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    (height, width, mean_color_channels): A tuple containing the image's height, width, and mean color values for each channel (BGR).

    If the image cannot be read or doesn't exist, None is returned.

    Example usage:
    height, width, mean_b, mean_g, mean_r = get_image_statistics('image.jpg')

    Note:
    - This function uses the OpenCV (cv2) library to read the image.
    - The mean color values are calculated for each channel (Blue, Green, and Red) as BGR.

    """
    image = cv2.imread(image_path)
    if image is not None:
        height, width, _ = image.shape
        mean_colors = np.mean(image, axis=(0, 1))
        return height, width, *mean_colors
    return None

def get_image_statistics_df(image_directory):
    """
    Generate a DataFrame containing statistics for multiple images in a specified directory.

    Parameters:
    - image_directory (str): The path to the directory containing image files.

    Returns:
    (pandas.DataFrame): A DataFrame with columns "Height," "Width," "Mean_R," "Mean_G," "Mean_B," and "image_id."

    This function iterates through image files in the specified directory, calculates statistics for each image (dimensions
    and mean color values for each channel), and stores the results in a DataFrame. The "image_id" column contains the
    filenames of the processed images.

    Example usage:
    image_stats_dataframe = get_image_statistics_df('image_directory')

    Note:
    - This function depends on the get_image_statistics function for individual image statistics.
    - Only the first 51 images in the directory (as determined by the "i > 50" condition) are processed, and this condition
      can be adjusted as needed.
    """
    # Format to store the results
    columns = ["Height", "Width", "Mean_R", "Mean_G", "Mean_B", "image_id"]
    stats_dict = {'Height': [], 'Width': [], 'Mean_R': [], 'Mean_G': [], 'Mean_B': [], "image_id": []}
    
    for i, filename in enumerate(os.listdir(image_directory)):
        if i % 1000 == 0:
            print(f'Image #{i}...')

        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_directory, filename)
            stats = get_image_statistics(image_path)
            height, width, mean_r, mean_g, mean_b = stats

            if stats is not None:
                for j, col in enumerate(columns):
                    if col == "image_id":
                        stats_dict[col].append(filename)
                    else:
                        stats_dict[col].append(stats[j])

    image_stats_df = pd.DataFrame(stats_dict)
    return image_stats_df


def plot_image_statistics(image_stats_df):
    """
    Create histograms to visualize and analyze image statistics.

    Parameters:
    - image_stats_df (pandas.DataFrame): DataFrame containing image statistics, including "Height," "Width," "Mean_R,"
      "Mean_G," and "Mean_B."

    Returns:
    None

    This function generates histograms for image statistics, providing insights into the distribution of image dimensions
    (height and width) and the mean color values in the Red (R), Green (G), and Blue (B) channels.

    Example usage:
    plot_image_statistics(image_stats_df)

    Note:
    - The input DataFrame should have the specified columns for image statistics.
    - The function uses Matplotlib for plotting and assumes that the necessary libraries (pandas and Matplotlib) are
      imported and available in the current environment.
    """
    
    print('Statistics: ')
    print(image_stats_df.describe())
    
    # Plot histograms
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
    fig.suptitle('Image Statistics Histograms', fontsize=16)

    # Plot Height histogram
    axes[0, 0].hist(image_stats_df['Height'], bins=10, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Height')
    axes[0, 0].set_xlabel('Height')
    axes[0, 0].set_ylabel('Frequency')

    # Plot Width histogram
    axes[0, 1].hist(image_stats_df['Width'], bins=10, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Width')
    axes[0, 1].set_xlabel('Width')
    axes[0, 1].set_ylabel('Frequency')

    # Separate the RGB channels for Mean_R, Mean_G, and Mean_B
    # Plot Mean_R histogram
    axes[0, 2].hist(image_stats_df['Mean_R'], bins=10, color='salmon', alpha=0.7, label='R')
    axes[0, 2].hist(image_stats_df['Mean_G'], bins=10, color='lightgreen', alpha=0.7, label='G')
    axes[0, 2].hist(image_stats_df['Mean_B'], bins=10, color='dodgerblue', alpha=0.7, label='B')
    axes[0, 2].set_title('RGB Mean')
    axes[0, 2].set_xlabel('RGB Mean')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()

    # Plot Mean_R histogram
    axes[1, 0].hist(image_stats_df['Mean_R'], bins=10, color='salmon', alpha=0.7)
    axes[1, 0].set_title('Mean Red')
    axes[1, 0].set_xlabel('Mean Red')
    axes[1, 0].set_ylabel('Frequency')

    # Plot Mean_G histogram
    axes[1, 1].hist(image_stats_df['Mean_G'], bins=10, color='lightgreen', alpha=0.7)
    axes[1, 1].set_title('Mean Green')
    axes[1, 1].set_xlabel('Mean Green')
    axes[1, 1].set_ylabel('Frequency')

    # Plot Mean_B histogram
    axes[1, 2].hist(image_stats_df['Mean_B'], bins=10, color='dodgerblue', alpha=0.7)
    axes[1, 2].set_title('Mean Blue')
    axes[1, 2].set_xlabel('Mean Blue')
    axes[1, 2].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Define the path to the image folder
def calculate_normalization_values(image_folder):
    # Initialize empty lists to store pixel values
    mean_values = [0, 0, 0]
    std_values = [0, 0, 0]

    # Create a transform to convert images to tensors
    transform = transforms.ToTensor()

    # Iterate through the images in the folder and accumulate pixel values
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]

    for image_path in image_paths:
        if not(image_path.endswith('.jpg')):
            continue
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        for i in range(3):  # Channels (R, G, B)
            mean_values[i] += image_tensor[i, :, :].mean().item()
            std_values[i] += image_tensor[i, :, :].std().item()

    # Calculate the mean and standard deviation
    num_images = len(image_paths)
    mean_values = [m / num_images for m in mean_values]
    std_values = [s / num_images for s in std_values]

    print("Mean values (R, G, B):", mean_values)
    print("Standard deviation (R, G, B):", std_values)
