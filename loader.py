from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from utils import resize_image
import numpy as np

class ImageData(Dataset):

    """
    Custom PyTorch Dataset for loading and preprocessing grayscale images.

    This dataset:
      - Loads images from provided file paths
      - Resizes and normalizes them
      - Optionally masks images using the label
      - Converts them to tensors with a single channel

    Attributes
    ----------
    image_paths : list of str
        List of file paths to image files.
    labels : list or array-like
        Corresponding labels for the images.
    mask_image : bool, optional
        If True, mask the image such that only non-zero pixels are preserved and scaled by the label.
        Set this to true to get array representations of masks images
    """

    def __init__(self, image_paths, labels, mask_image_paths=False):
        self.image_paths = image_paths
        self.labels = labels
        self.mask_image_paths = mask_image_paths

    def __len__(self):

        """
        Returns the total number of samples in the dataset.

        Returns
        -------
        int
            Number of images in the dataset.
        """

        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads and preprocesses an image and its label by index.

        Steps:
          - Loads the image and converts to grayscale
          - Resizes and normalizes pixel values to [0, 1]
          - Adds channel dimension (1, H, W)
          - Optionally masks the image with its label

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        image_tensor : torch.Tensor
            Preprocessed image tensor of shape (1, H, W).
        label_tensor : torch.Tensor
            Label tensor corresponding to the image.
        """
        label = self.labels[idx]  

        image = Image.open(self.image_paths[idx]).convert("L")
        resized_image = resize_image(image, ideal_size=100)
        image_array = np.array(resized_image, dtype=np.float32)/255
        image_array = np.expand_dims(image_array, axis=0)
        image_array = torch.from_numpy(image_array)

        if self.mask_image_paths:
            mask_image = Image.open(self.mask_image_paths[idx]).convert("L")
            resized_mask_image = resize_image(mask_image)
            mask_image_array = np.array(resized_mask_image, dtype=np.float32)/255
            mask_image_array = label*(mask_image_array != 0).astype("long")
            mask_image_array = torch.from_numpy(mask_image_array)
            return image_array, mask_image_array

        return image_array, torch.tensor(label, dtype=torch.long)
