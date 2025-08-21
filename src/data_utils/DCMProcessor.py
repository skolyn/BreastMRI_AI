import os
from typing import List,Optional

import numpy as np
import pydicom as pdc
import matplotlib.pyplot as plt
import pandas as pd


class DCMProcessor:
    """A class for processing DICOM files from a directory or individual files."""

    def __init__(self, dicom_dir: str):
        """
        Initialize the DCMProcessor with a directory containing DICOM files.

        Args:
            dicom_dir (str): Path to the directory containing DICOM (.dcm) files.
        """
        self.dicom_dir: str = dicom_dir

    def __load_dicom_file(self, dicom_filename: str) -> pdc.FileDataset:
        """
        Load a single DICOM file.

        Args:
            dicom_filename (str): Filename of the DICOM file.

        Returns:
            pydicom.FileDataset: The loaded DICOM dataset object.
        """
        return pdc.dcmread(os.path.join(self.dicom_dir, dicom_filename))

    def load_dicom_file(self, dicom_filename: str) -> pdc.FileDataset:
        """
        Public method to load a single DICOM file.

        Args:
            dicom_filename (str): Filename of the DICOM file.

        Returns:
            pydicom.FileDataset: The loaded DICOM dataset object.
        """
        return self.__load_dicom_file(dicom_filename)

    def load_dicom_image(self, dicom_filename: str) -> np.ndarray:
        """
        Get the pixel array from a DICOM file.

        Args:
            dicom_filename (str): Filename of the DICOM file.

        Returns:
            np.ndarray: The image pixel array.
        """
        ds = self.__load_dicom_file(dicom_filename)
        return ds.pixel_array

    def load_dicom_dir_datasets(self) -> List[pdc.FileDataset]:
        """
        Load all DICOM files in the directory as datasets.

        Returns:
            List[pydicom.FileDataset]: A list of DICOM dataset objects.
        """
        datasets: List[pdc.FileDataset] = []
        for f in sorted(os.listdir(self.dicom_dir)):
            if f.endswith(".dcm"):
                ds = self.__load_dicom_file(f)
                datasets.append(ds)
        return datasets

    def load_dicom_dir_images(self) -> List[np.ndarray]:
        """
        Load all DICOM files in the directory as NumPy arrays.

        Returns:
            List[np.ndarray]: A list of pixel arrays for each DICOM file.
        """
        datasets = self.load_dicom_dir_datasets()
        return [ds.pixel_array for ds in datasets]

    def load_volume(self) -> np.ndarray:
        """
        Load all DICOM images in the directory as a 3D volume.

        Returns:
            np.ndarray: 3D array with shape (num_slices, height, width).
        """
        images = self.load_dicom_dir_images()
        if not images:
            raise ValueError("No DICOM files found in the directory.")
        return np.stack(images, axis=0)

    def visualize_volume(self, volume: np.ndarray = None):
        """
        Visualize the first slice of a 3D DICOM volume using matplotlib.

        Args:
            volume (np.ndarray, optional): 3D NumPy array. If None, loads volume from directory.
        """
        if volume is None:
            volume = self.load_volume()
        plt.imshow(volume[0], cmap='gray')
        plt.title("Slice 0")
        plt.axis('off')
        plt.show()

    @staticmethod
    def __extract_metadata(ds: pdc.FileDataset, tags: List[str] = None) -> dict:
        """
        Extract metadata from a single DICOM dataset.

        Args:
            ds (pydicom.FileDataset): The DICOM dataset.
            tags (List[str], optional): List of tag names to extract. If None, extracts all non-sequence tags.

        Returns:
            dict: Metadata dictionary for the dataset.
        """
        row = {}
        if tags is None:
            for elem in ds:
                if elem.VR != 'SQ':  # skip sequences
                    row[elem.name] = elem.value
        else:
            for tag in tags:
                row[tag] = getattr(ds, tag, None)
        return row

    def get_metadata_dataframe(self, tags: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get metadata of all DICOM files in the directory as a Pandas DataFrame.

        Args:
            tags (List[str], optional): List of DICOM tags to extract. If None, extracts all available non-sequence tags.

        Returns:
            pd.DataFrame: DataFrame where each row corresponds to a DICOM file and columns are metadata fields.
        """
        datasets = self.load_dicom_dir_datasets()
        rows = [self.__extract_metadata(ds, tags) for ds in datasets]
        return pd.DataFrame(rows)

