import os
from typing import Tuple, Callable, List

import numpy as np
from skimage.transform import resize




class DCMImageNormalizer:
    """A class for normalizing and resizing DICOM images using an existing DCMProcessor instance."""

    def __init__(self, dcm_processor) -> None:
        """
        Initialize the DICOMImageNormalizer with a DCMProcessor instance.
        Args:
            dcm_processor: An instance of DCMProcessor.
            target_size (Tuple[int, int], optional): Desired image size (height, width). Defaults to (224, 224).
        """
        self.dcm_processor = dcm_processor

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
        - Subtracts the minimum pixel value from the entire image (so minimum becomes 0).
        - Divides by the maximum pixel value (so maximum becomes 1).
        Args:
            img (np.ndarray): Input image array.
        Returns:
            np.ndarray: Normalized image array in [0,1].
        """
        img = img - np.min(img)
        if np.max(img) != 0:
            img = img / np.max(img)
        return img

    @staticmethod
    def resize_image(self, img: np.ndarray,target_size:Tuple[int,int]=(224,224) -> np.ndarray:
        """
        Resize an image to the target size (default 224x224).
        Uses anti-aliasing for smoother results.
        Args:
            img (np.ndarray): Input image array.
        Returns:
            np.ndarray: Resized image array.
        """
        return resize(img, target_size, anti_aliasing=True)

    def process_image(self, img_name: str) -> np.ndarray:
        """
        Load, normalize, and resize a single DICOM image by filename.
        Pipeline:
            1. Load DICOM file as numpy array.
            2. Normalize pixel intensities to [0,1].
            3. Resize image to target size (e.g. 224x224).
        Args:
            img_name (str): Filename of the DICOM file.
        Returns:
            np.ndarray: Processed image array.
        """
        img = self.__load_image(img_name)
        img = self.normalize(img)
        img = self.resize_image(img)
        return img

    def process_all(self) -> list[np.ndarray]:
        """
        Process all DICOM images in the given directory.
        For each `.dcm` file:
            - Load image
            - Normalize
            - Resize
        Returns:
            list[np.ndarray]: List of processed image arrays.
        """
        processed_list = []
        for f in sorted(os.listdir(self.dcm_processor.dicom_dir)):
            if f.endswith(".dcm"):
                processed_list.append(self.process_image(f))
        return processed_list

    @staticmethod
    def apply_transforms(img: np.ndarray, transforms: List[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
        """
        Apply a sequence of transforms to an image (similar to torchvision.transforms.Compose).
        Each transform must be a function that takes and returns a numpy array.
        Example:
            def invert(x): return 1 - x
            def threshold(x): return (x > 0.5).astype(float)
            transforms = [invert, threshold]
            new_img = DICOMImageNormalizer.apply_transforms(img, transforms)
        Args:
            img (np.ndarray): Input image array.
            transforms (List[Callable]): List of functions to apply in sequence.
        Returns:
            np.ndarray: Transformed image.
        """
        for t in transforms:
            img = t(img)
        return img
