import os

import tempfile
from typing import Tuple
import pytest
import pandas as pd
import numpy as np
from PIL import Image
import torch

from src.model_utils.BreastMRIDataset import BreastMRIDataset

@pytest.fixture
def dummy_dataset() -> Tuple[BreastMRIDataset, tempfile.TemporaryDirectory]:
    """
    Create a temporary dataset with dummy images and labels for testing.

    Creates a temporary directory structure with:
    - 2 grayscale dummy images (64x64 pixels)
    - Corresponding CSV file with image filenames and binary labels

    Returns:
        Tuple[BreastMRIDataset, tempfile.TemporaryDirectory]:
            Dataset instance and temporary directory handle.
    """

    tmp_dir = tempfile.TemporaryDirectory()
    img_dir: str = os.path.join(tmp_dir.name, "images")
    os.makedirs(img_dir)

    for i in range(2):

        img = Image.new("L", (64, 64), color=i * 100)
        img.save(os.path.join(img_dir, f"img{i}.png"))

    label_file: str = os.path.join(tmp_dir.name, "labels.csv")
    df = pd.DataFrame({
        "filename": ["img0.png", "img1.png"],
        "label": [0, 1]
    })
    df.to_csv(label_file, index=False)

    dataset = BreastMRIDataset(
        image_dir=img_dir,
        label_file=label_file,
        transform=None
    )

    return dataset, tmp_dir


def test_dataset_length(dummy_dataset: Tuple[BreastMRIDataset, tempfile.TemporaryDirectory]) -> None:
    """
    Test that dataset returns correct number of samples.

    Args:
        dummy_dataset: Fixture providing test dataset and temp directory.
    """
    dataset, _ = dummy_dataset

    assert len(dataset) == 2, "Dataset should contain exactly 2 samples"


def test_dataset_item_type(dummy_dataset: Tuple[BreastMRIDataset, tempfile.TemporaryDirectory]) -> None:
    """
    Test that dataset returns correct data types for images and labels.

    Ensures that:
    - Images are returned as numpy arrays or torch tensors
    - Labels are returned as integers, numpy integers, or torch tensors

    Args:
        dummy_dataset: Fixture providing test dataset and temp directory.
    """
    dataset, _ = dummy_dataset
    image, label = dataset[0]

    assert isinstance(label, (int, np.integer, torch.Tensor)), \
        f"Label should be int, np.integer, or torch.Tensor, got {type(label)}"

    assert isinstance(image, (np.ndarray, torch.Tensor)), \
        f"Image should be np.ndarray or torch.Tensor, got {type(image)}"


def test_dataset_pairing(dummy_dataset: Tuple[BreastMRIDataset, tempfile.TemporaryDirectory]) -> None:
    """
    Test that dataset correctly pairs images with their corresponding labels.

    Verifies that the dataset maintains proper image-label correspondence
    as defined in the CSV file.

    Args:
        dummy_dataset: Fixture providing test dataset and temp directory.
    """
    dataset, _ = dummy_dataset

    _, label0 = dataset[0]
    _, label1 = dataset[1]

    assert label0 == 0, f"First sample should have label 0, got {label0}"
    assert label1 == 1, f"Second sample should have label 1, got {label1}"

def test_dataset_image_dimensions(dummy_dataset: Tuple[BreastMRIDataset, tempfile.TemporaryDirectory]) -> None:
    """
    Test that loaded images have expected dimensions.

    Verifies that images are loaded with correct spatial dimensions
    matching the dummy data specifications.

    Args:
        dummy_dataset: Fixture providing test dataset and temp directory.
    """
    dataset, _ = dummy_dataset
    image, _ = dataset[0]

    if isinstance(image, torch.Tensor):
        image_array = image.numpy()
    else:
        image_array = image

    assert image_array.shape[-2:] == (64, 64), \
        f"Image should have spatial dimensions 64x64, got {image_array.shape[-2:]}"


def test_dataset_index_bounds(dummy_dataset: Tuple[BreastMRIDataset, tempfile.TemporaryDirectory]) -> None:
    """
    Test that dataset handles index bounds correctly.

    Ensures that accessing valid indices works and invalid indices raise
    appropriate exceptions.

    Args:
        dummy_dataset: Fixture providing test dataset and temp directory.
    """
    dataset, _ = dummy_dataset

    _ = dataset[0]
    _ = dataset[1]
    with pytest.raises(IndexError):
        _ = dataset[2]

    with pytest.raises(IndexError):
        _ = dataset[-3]