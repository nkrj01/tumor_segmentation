import numpy as np
from PIL import Image, ImageOps
import os
import torch

def pad_to_square(image, fill_color=0):
        """Pads and image to make it square

        Args:
            image (PIL Image): image you want to pad
            fill_color (tuple, optional): pad using pixel of specific size. Defaults to (255, 255, 255).

        Returns:
            PIL image: Squared image
        """

        width, height = image.size
        max_side = max(width, height)

        # Calculate required padding
        left = (max_side - width) // 2
        right = max_side - width - left
        top = (max_side - height) // 2
        bottom = max_side - height - top

        # Pad the image
        padded_image = ImageOps.expand(image, border=(left, top, right, bottom), fill=fill_color)
        return padded_image


def resize_image(image, ideal_size=100):
    """This function converts your image to a desired size using resize and padding

    Args:
        image (PIL Image): A PIL image object
        ideal_size (int, optional): the desired size. Defaults to 224.

    Returns:
        PIL image: resized image
    """
    w, h  = image.size

    if h <= ideal_size and w <= ideal_size:
        resize_image = ImageOps.pad(image, (ideal_size, ideal_size))
    else:
        image = pad_to_square(image)
        resize_image = image.resize((ideal_size, ideal_size), resample=Image.BICUBIC)

    return resize_image  


def image_batch_generator(folder_path, batch_size=250, output_image_size = 224, flatten=False):
    """Create batches of matrix from batches of image.
    This is necessary because the data size is too big and cannot be processed
    all at once.  

    Args:
        folder_path (str): path to your images
        batch_size (int, optional): batch size. Defaults to 250.
        output_image_size (int, optional): size of output images. Defaults to 224 by 224.
        flatten (bool, optional): flatten the array or not. Default to False. 

    Yields:
        np_array: matrix of image representation
    """
    files = os.listdir(folder_path)
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batch = []
        for f in batch_files:
            image = Image.open(os.path.join(folder_path, f)).convert('L')
            image_resized = resize_image(image, ideal_size=output_image_size)
            if flatten:
                image_array = (np.array(image_resized, dtype=np.float32).flatten())/255
            else:
                image_array = np.array(image_resized, dtype=np.float32)/255
            batch.append(image_array)
        yield np.array(batch)


def per_class_accuracy(output, target, num_classes):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)  # [N, H, W]
        accuracies = []

        for class_id in range(num_classes):
            # Mask for where the ground truth is class_id
            mask = (target == class_id)

            # If class_id doesn't exist in this batch, skip
            total = mask.sum().item()
            if total == 0:
                accuracies.append(None)  # or 0.0, or np.nan
                continue

            correct = (pred[mask] == target[mask]).sum().item()
            acc = correct / total
            accuracies.append(acc)

    return accuracies
