import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os

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
    


class CustomLabeledImageDataset(Dataset):
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
        
        label = self.dataframe[self.label_col].iloc[idx]
        
        return img, label
