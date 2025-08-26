from typing import List,Tuple
import os

import torch
import numpy as np
from torch.utils.data import Dataset

from src.data_utils.DCMProcessor import DCMProcessor
from src.data_utils.DCMImageNormalizer import DCMImageNormalizer


class BreastMRIDataset(Dataset):
    """Breast DCE-MRI dataset for classification.

    Loads DICOM volumes from patient case directories, groups
    frames per slice, applies normalization and resizing, and
    returns tensors for EfficientNet training.

    Labels are expected to be:
        0 = Normal
        1 = Benign
        2 = Malignant
    """

    def __init__(self,base_dir:str, cases: List[str], labels: dict, transform=None):
        """Initialize the dataset.

        Args:
            cases (list[str]): List of paths to patient DICOM directories.
            labels (dict): Mapping {case_dir: label_int} where label_int ∈ {0,1,2}.
            transform (callable, optional): Optional transform applied after
                normalization and resizing. Defaults to None.
        """
        self.base_dir = base_dir
        self.cases = cases
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of patient cases.

        Returns:
            int: Number of patient case directories.
        """
        return len(self.cases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load one case and return its processed slices and label.

        Args:
            idx (int): Index of the patient case.

        Returns:
            tuple:
                - slices (torch.Tensor): Tensor of shape
                    (num_slices, 4, 224, 224), where each slice
                    has 4 frames (channels).
                - label (int): Case-level label (0=normal, 1=benign, 2=malignant).
        """
        case_dir = self.cases[idx]

        dcm_proc = DCMProcessor(os.path.join(self.base_dir,case_dir))
        volume = dcm_proc.load_volume()

        slices = []
        for s in range(0, volume.shape[0], 4):
            if s + 4 <= volume.shape[0]:
                four_frames = volume[s:s + 4]

                processed = []
                for f in four_frames:
                    img = DCMImageNormalizer.normalize(f)
                    img = DCMImageNormalizer.resize_image(img, (224, 224))
                    processed.append(img)
                processed = np.stack(processed, axis=0)

                tensor_slice = torch.tensor(processed, dtype=torch.float32)

                if self.transform:
                    tensor_slice = self.transform(tensor_slice)

                slices.append(tensor_slice)

        slices = torch.stack(slices, dim=0)
        label = self.labels[case_dir]

        return slices, label


