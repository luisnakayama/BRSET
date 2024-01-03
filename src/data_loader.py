import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


import warnings
warnings.filterwarnings("ignore")

# Define a custom dataset to load images from a folder
class ImageFolderDataset(Dataset):
    """
    A custom PyTorch dataset class for loading and preprocessing images from a folder.
    
    This class is designed to be used with image data stored in a folder. It loads images, applies
    specified transformations, and returns them as PyTorch tensors.

    Parameters:
    - folder_path (str): The path to the folder containing the image files.
    - shape (tuple, optional): The desired shape for the images (height, width). Default is (224, 224).
    - transform (torchvision.transforms.Compose, optional): A composition of image transformations.
      Default is a set of common transformations, including resizing and normalization.

    Methods:
    - __len__(): Returns the total number of images in the dataset.
    - __getitem__(idx): Loads and preprocesses an image at the given index and returns it as a PyTorch tensor.

    Example Usage:
    ```python
    dataset = ImageFolderDataset(folder_path='/path/to/images', shape=(256, 256), transform=my_transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    ```

    Note:
    - Ensure that the image files in the specified folder are in common formats (jpg, jpeg, png, gif).
    - The default transform includes resizing the image to the specified shape and normalizing pixel values.
    - You can customize the image preprocessing by passing your own transform as an argument.
    - This dataset class is suitable for tasks like image classification, object detection, and more.

    Dependencies:
    - PyTorch
    - torchvision.transforms.Compose
    - PIL (Python Imaging Library)
    """
    def __init__(self, folder_path, shape=(224, 224), transform=None):
        """
        Initialize the ImageFolderDataset.

        Args:
        - folder_path (str): The path to the folder containing the image files.
        - shape (tuple, optional): The desired shape for the images (height, width). Default is (224, 224).
        - transform (torchvision.transforms.Compose, optional): A composition of image transformations.
          Default is a set of common transformations, including resizing and normalization.
        """
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'gif'))]
        self.shape = shape
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Load and preprocess an image at the specified index.

        Args:
        - idx (int): The index of the image to retrieve.

        Returns:
        tuple: A tuple containing the image file name and the preprocessed image as a PyTorch tensor.
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img_name, img
    


class BinaryBRSETDataset(Dataset):
    """
    A custom PyTorch dataset class for loading and preprocessing images with associated labels from a folder and a dataframe.

    This class is designed for tasks where you have image data stored in a folder and corresponding labels in a DataFrame.
    It loads images, applies specified transformations, and returns them as PyTorch tensors along with their labels.

    Parameters:
    - folder_path (str): The path to the folder containing the image files.
    - dataframe (pandas.DataFrame): A DataFrame containing image filenames and associated labels.
    - shape (tuple, optional): The desired shape for the images (height, width). Default is (224, 224).
    - transform (torchvision.transforms.Compose, optional): A composition of image transformations.
      Default is a set of common transformations, including resizing and normalization.
    - label_col (str, optional): The name of the DataFrame column containing the labels. Default is 'diabetic_retinopathy'.

    Methods:
    - __len__(): Returns the total number of images in the dataset.
    - __getitem__(idx): Loads and preprocesses an image and its corresponding label at the given index and returns them as PyTorch tensors.

    Example Usage:
    ```python
    dataset = CustomLabeledImageDataset(
        folder_path='/path/to/images',
        dataframe=my_dataframe,
        shape=(256, 256),
        transform=my_transforms,
        label_col='diabetic_retinopathy'
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    ```

    Note:
    - Ensure that the image files in the specified folder are in common formats (jpg, jpeg, png, gif).
    - The default transform includes resizing the image to the specified shape and normalizing pixel values.
    - You can customize the image preprocessing by passing your own transform as an argument.
    - This dataset class is suitable for supervised learning tasks with image data and associated labels.

    Dependencies:
    - PyTorch
    - torchvision.transforms.Compose
    - PIL (Python Imaging Library)
    - pandas (for handling the DataFrame)
    """
    def __init__(self, folder_path, dataframe, shape=(224, 224), transform=None, label_col='diabetic_retinopathy'):
        """
        Initialize the CustomLabeledImageDataset.

        Args:
        - folder_path (str): The path to the folder containing the image files.
        - dataframe (pandas.DataFrame): A DataFrame containing image filenames and associated labels.
        - shape (tuple, optional): The desired shape for the images (height, width). Default is (224, 224).
        - transform (torchvision.transforms.Compose, optional): A composition of image transformations.
          Default is a set of common transformations, including resizing and normalization.
        - label_col (str, optional): The name of the DataFrame column containing the labels. Default is 'diabetic_retinopathy'.
        """
        
        self.folder_path = folder_path
        self.dataframe = dataframe
        self.shape = shape
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
            # Imagenet Normalization:
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Dataset Normalization:
            #transforms.Normalize(mean=[0.5896205017400412, 0.29888971649817453, 0.1107679405196557], std=[0.28544273712830986, 0.15905456049750208, 0.07012281660980953])
        ])
        self.label_col = label_col
        self.labels = self.dataframe[self.label_col].values
        self.labels = np.expand_dims(self.labels, axis=1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Load and preprocess an image and its corresponding label at the specified index.

        Args:
        - idx (int): The index of the image and label to retrieve.

        Returns:
        tuple: A tuple containing the preprocessed image as a PyTorch tensor and its associated label.
        """
        img_name = self.dataframe['image_id'].iloc[idx] + ('.jpg')
        img_path = os.path.join(self.folder_path, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        label = self.labels[idx] #self.dataframe[self.label_col].iloc[idx]
        
        return {
            'image': torch.FloatTensor(img),
            'labels': torch.FloatTensor(label)
        }


# Function to process text labels and one-hot encode them
def process_labels(df, col='answer', mlb=None, train_columns=None):
    """
    Process text labels and perform one-hot encoding using MultiLabelBinarizer.

    Args:
    - df (pd.DataFrame): The DataFrame containing the labels.
    - col (str): The column name containing the labels.
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): The MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.

    Returns:
    pd.DataFrame: One-hot encoded labels.
    sklearn.preprocessing.MultiLabelBinarizer: MultiLabelBinarizer object.
    list: List of columns from the training set.

    Example:
    one_hot_labels, mlb, train_columns = process_labels(df, col='answer')
    """
    if mlb is None:
        mlb = MultiLabelBinarizer()
        if df[col].dtype == int:
            label = df[col]
        else:
            labels = df[col].apply(lambda x: set(x.split(', ')))

        if df[col].dtype == int and (len(df[col].unique()) == 2):
            train_columns = col
            one_hot_labels = label
        else:
            one_hot_labels = pd.DataFrame(mlb.fit_transform(labels), columns=mlb.classes_)
            # Save the columns from the training set
            train_columns = one_hot_labels.columns

        return one_hot_labels, mlb, train_columns

    else:
        if df[col].dtype == int:
            label = df[col]
        else:
            labels = df[col].apply(lambda x: set(x.split(', ')))

        if df[col].dtype == int and (len(df[col].unique()) == 2):
            one_hot_labels = label
        else:
            one_hot_labels = pd.DataFrame(mlb.transform(labels), columns=train_columns)

        return one_hot_labels


# Custom Dataset class for PyTorch
class BRSETDataset(Dataset):
    """
    Custom PyTorch Dataset for VQA (Visual Question Answering).

    Args:
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - image_cols (str): Column name containing the path to the images.
    - label_col (str): Column name containing labels.
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.

    Attributes:
    - image_data (np.ndarray): Array of image data.
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.
    - labels (np.ndarray): Array of one-hot encoded labels.

    Methods:
    - __len__(): Returns the length of the dataset.
    - __getitem__(idx): Returns a dictionary with 'text', 'image', and 'labels'.

    Example:
    dataset = BRSETDataset(df, image_cols='image1', label_col='answer', mlb=mlb, train_columns=train_columns)
    """
    def __init__(self, df, image_cols, images_dir, label_col, mlb, train_columns, shape=(224, 224), transform=None):

        # Images
        self.image_data = df[image_cols].values
        self.images_dir = images_dir
        self.shape = shape
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Labels
        self.mlb = mlb
        self.train_columns = train_columns
        self.labels = process_labels(df, col=label_col, mlb=mlb, train_columns=train_columns).values
        #print(self.labels.shape)
        if len(self.labels.shape) == 1:
            self.labels = np.expand_dims(self.labels, axis=1)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # Images:
        img_path = self.image_data[idx]

        img = Image.open( os.path.join(self.images_dir, img_path + '.jpg') ).convert("RGB")
        img = self.transform(img)

        return {
            'image': torch.FloatTensor(img),
            'labels': torch.FloatTensor(self.labels[idx])
        }