import unittest
import os
import shutil
from typing import List

import numpy as np
import pydicom
import torch
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian

from src.model_utils.BreastMRIDataset import BreastMRIDataset

class TestBreastMRIDataset(unittest.TestCase):
    """Unit tests for the BreastMRIDataset class."""

    def setUp(self) -> None:
        """Set up a temporary directory with mock DICOM files for testing."""
        self.test_dir: str = "test_data"
        self.case1_dir: str = os.path.join(self.test_dir, "case1")
        self.case2_dir: str = os.path.join(self.test_dir, "case2")
        self.empty_case_dir: str = os.path.join(self.test_dir, "empty_case")

        # Create directories
        os.makedirs(self.case1_dir, exist_ok=True)
        os.makedirs(self.case2_dir, exist_ok=True)
        os.makedirs(self.empty_case_dir, exist_ok=True)

        # Create mock DICOM files for case1 (8 files = 2 slices)
        for i in range(8):
            self._create_mock_dcm(
                os.path.join(self.case1_dir, f"slice{i}.dcm"),
                (512, 512)
            )

        # Create mock DICOM files for case2 (4 files = 1 slice)
        for i in range(4):
            self._create_mock_dcm(
                os.path.join(self.case2_dir, f"slice{i}.dcm"),
                (512, 512)
            )

        self.cases: List[str] = ["case1", "case2", "empty_case"]
        self.labels: dict[str, int] = {"case1": 0, "case2": 1, "empty_case": 2}

        self.dataset = BreastMRIDataset(
            base_dir=self.test_dir,
            cases=self.cases,
            labels=self.labels
        )

    def tearDown(self) -> None:
        """Remove the temporary directory after tests are complete."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_mock_dcm(self, filename: str, shape: tuple[int, int]) -> None:
        """Helper function to create a mock DICOM file."""
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        file_meta.ImplementationClassUID = "1.2.3.4"
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = True

        # Add necessary DICOM tags
        ds.PatientName = "Test^Patient"
        ds.PatientID = "12345"
        ds.Modality = "MR"
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.Columns = shape[1]
        ds.Rows = shape[0]

        # Create random pixel data
        pixel_array = np.random.randint(0, 4096, size=shape, dtype=np.uint16)
        ds.PixelData = pixel_array.tobytes()

        ds.save_as(filename, write_like_original=False)


    def test_dataset_length(self) -> None:
        """Verify that the dataset returns the correct number of cases."""
        self.assertEqual(len(self.dataset), 3)

    def test_getitem_returns_correct_types(self) -> None:
        """Check if __getitem__ returns a tensor and an integer label."""
        slices, label = self.dataset[0]
        self.assertIsInstance(slices, torch.Tensor)
        self.assertIsInstance(label, int)

    def test_getitem_shape_for_case1(self) -> None:
        """
        Verify the output tensor shape for a case with 8 DICOM files (2 slices).
        """
        slices, _ = self.dataset[0]
        self.assertEqual(slices.shape, (2, 4, 224, 224))

    def test_getitem_shape_for_case2(self) -> None:
        """
        Verify the output tensor shape for a case with 4 DICOM files (1 slice).
        """
        slices, _ = self.dataset[1]
        self.assertEqual(slices.shape, (1, 4, 224, 224))

    def test_getitem_for_empty_case(self) -> None:
        """
        Test that an empty case directory results in a tensor with 0 slices.
        """
        slices, _ = self.dataset[2]
        self.assertEqual(slices.shape[0], 0)

    def test_label_pairing(self) -> None:
        """Confirm that images are correctly paired with their labels."""
        _, label1 = self.dataset[0]
        self.assertEqual(label1, self.labels["case1"])

        _, label2 = self.dataset[1]
        self.assertEqual(label2, self.labels["case2"])

        _, label3 = self.dataset[2]
        self.assertEqual(label3, self.labels["empty_case"])

    def test_transform_application(self) -> None:
        """Test if an optional transform is correctly applied."""
        # Define a simple transform that flips the tensor
        def test_transform(tensor_slice: torch.Tensor) -> torch.Tensor:
            return torch.flip(tensor_slice, [0])

        transformed_dataset = BreastMRIDataset(
            base_dir=self.test_dir,
            cases=self.cases,
            labels=self.labels,
            transform=test_transform
        )

        original_slices, _ = self.dataset[0]
        transformed_slices, _ = transformed_dataset[0]

        # Check that the transform was applied by comparing the first and last frames
        self.assertTrue(
            torch.equal(original_slices[0, 0], transformed_slices[0, 3])
        )


if __name__ == '__main__':
    unittest.main()