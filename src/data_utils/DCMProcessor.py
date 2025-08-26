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

    def load_dicom_dir_datasets(self,subdir:Optional[str]=None) -> List[pdc.FileDataset]:
        """
        Load all DICOM files in the directory as datasets.

        Returns:
            List[pydicom.FileDataset]: A list of DICOM dataset objects.
        """
        datasets: List[pdc.FileDataset] = []
        dir=os.path.join(self.dicom_dir, subdir) if subdir else self.dicom_dir
        for f in sorted(os.listdir(dir)):
            if f.endswith(".dcm"):
                ds = self.__load_dicom_file(f)
                datasets.append(ds)
        return datasets

    def load_dicom_dir_images(self,subdir=None) -> List[np.ndarray]:
        """
        Load all DICOM files in the directory as NumPy arrays.

        Returns:
            List[np.ndarray]: A list of pixel arrays for each DICOM file.
        """
        datasets = self.load_dicom_dir_datasets(subdir)
        return [ds.pixel_array for ds in datasets]

    def load_volume(self,subdir:str=None) -> np.ndarray:
        """
        Load all DICOM images in the directory as a 3D volume.

        Returns:
            np.ndarray: 3D array with shape (num_slices, height, width).
        """
        images = self.load_dicom_dir_images(subdir=subdir)
        if not images:
            raise ValueError("No DICOM files found in the directory.")
        return np.stack(images, axis=0)

    def visualize_volume(self, volume: Optional[np.ndarray] = None, group_size: int = 4):
        """
        Visualize DICOM slices in groups (e.g., 4 slices per study).

        Args:
            volume (np.ndarray, optional): Preloaded volume (num_slices, H, W).
            group_size (int): Number of slices per study (default=4).
        """
        if volume is None:
            volume = self.load_volume()

        num_slices = volume.shape[0]
        num_groups = num_slices // group_size

        for g in range(num_groups):
            group = volume[g * group_size:(g + 1) * group_size]

            fig, axes = plt.subplots(1, group_size, figsize=(12, 3))
            for i, ax in enumerate(axes):
                ax.imshow(group[i], cmap="gray")
                ax.set_title(f"Slice {g*group_size + i}")
                ax.axis("off")
            plt.suptitle(f"Group {g+1} (slices {g*group_size}–{g*group_size+group_size-1})")
            plt.show()

    def visualize_slice(self, slice_index: int, volume: Optional[np.ndarray] = None):
        """
        Visualize a single slice from the volume.

        Args:
            slice_index (int): Index of the slice to visualize.
            volume (np.ndarray, optional): Preloaded volume.
        """
        if volume is None:
            volume = self.load_volume()
        if slice_index >= volume.shape[0]:
            raise IndexError("Slice index out of range.")
        plt.imshow(volume[slice_index], cmap="gray")
        plt.title(f"Slice {slice_index}")
        plt.axis("off")
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



d=DCMProcessor("../../data/fastMRI_breast_IDS_150_300_DCM/fastMRI_breast_151_1_DCM")
