import torch
from torch.utils.data import Dataset
import cv2

class FacadesDataset(Dataset):
    def __init__(self, list_file, image_size=256):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # Read the list of image filenames
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        self.image_size = image_size
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)
        if img_color_semantic is None:
            raise FileNotFoundError(f'Unable to read image: {img_name}')

        height, width, _ = img_color_semantic.shape
        midpoint = width // 2
        if midpoint == 0:
            raise ValueError(f'Image width is too small to split into paired samples: {img_name}')

        image_rgb = img_color_semantic[:, :midpoint]
        image_semantic = img_color_semantic[:, midpoint:]

        if self.image_size is not None:
            resize_shape = (self.image_size, self.image_size)
            image_rgb = cv2.resize(image_rgb, resize_shape, interpolation=cv2.INTER_AREA)
            image_semantic = cv2.resize(image_semantic, resize_shape, interpolation=cv2.INTER_AREA)

        # Convert the image to a PyTorch tensor
        image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        image_semantic = torch.from_numpy(image_semantic).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        return image_rgb, image_semantic
