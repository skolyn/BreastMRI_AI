import os

import numpy as np
from skimage.transform import resize
from typing import Tuple, Dict


class DICOMImageNormalizer:
    """A class for normalizing and resizing DICOM images using an existing DCMProcessor instance."""

    def __init__(self, dcm_processor, target_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Initialize the DICOMImageNormalizer with a DCMProcessor instance.
        Args:
            dcm_processor: An instance of DCMProcessor.
            target_size (Tuple[int, int], optional): Desired image size (height, width). Defaults to (224, 224).
        """
        self.dcm_processor = dcm_processor
        self.target_size = target_size

    def __load_image(self, img_name: str) -> np.ndarray:
        """
        Private method to load a single DICOM image from DCMProcessor.
        Args:
            img_name (str): Filename of the DICOM file.
        Returns:
            np.ndarray: Loaded image as a NumPy array.
        """
        return self.dcm_processor.load_dicom_image(img_name)

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        """
        Normalize an image to the range [0,1].
        Args:
            img (np.ndarray): Input image array.
        Returns:
            np.ndarray: Normalized image array.
        """
        img = img - np.min(img)
        if np.max(img) != 0:
            img = img / np.max(img)
        return img

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Resize an image to the target size.
        Args:
            img (np.ndarray): Input image array.
        Returns:
            np.ndarray: Resized image array.
        """
        return resize(img, self.target_size, anti_aliasing=True)

    def process_image(self, img_name: str) -> np.ndarray:
        """
        Public method to load, normalize, and resize a single DICOM image by name.
        Args:
            img_name (str): Filename of the DICOM file.
        Returns:
            np.ndarray: Processed image array.
        """
        img = self.__load_image(img_name)
        img = self.normalize(img)
        img = self.resize_image(img)
        return img

    def process_all(self) -> Dict[str, np.ndarray]:
        """
        Public method to process all DICOM images in the DCMProcessor's directory.
        Returns:
            Dict[str, np.ndarray]: Dictionary with filenames as keys and processed image arrays as values.
        """
        processed_dict = {}
        for f in sorted(os.listdir(self.dcm_processor.dicom_dir)):
            if f.endswith(".dcm"):
                processed_dict[f] = self.process_image(f)
        return processed_dict