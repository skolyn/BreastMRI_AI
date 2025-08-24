import logging
import os
from typing import Dict, Optional, Tuple, Any, List

import cv2
import numpy as np

from data_utils.DCMProcessor import DCMProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DICOMPreprocessor:
    """
    DICOM fayllarını emal etmək üçün sinif.

    Funksionallıqlar:
        - Normalizasiya (z-score, min-max, robust, percentile)
        - Resize
        - Window/level tətbiqi
        - CLAHE ilə kontrast artırma
        - Batch emal
        - Statistikaların çıxarılması
    """

    def __init__(
        self,
        dicom_dir: str,
        output_dir: str,
        target_size: Tuple[int, int] = (224, 224),
        normalization_method: str = "z_score",
    ) -> None:
        """
        Preprocessor obyektini yaradın.

        Args:
            dicom_dir (str): DICOM fayllarının qovluğu.
            output_dir (str): Çıxış şəkillərinin saxlanacağı qovluq.
            target_size (Tuple[int, int], optional): Resize ölçüsü.
                Default (224, 224).
            normalization_method (str, optional): Normalizasiya metodu.
                Seçimlər: ["z_score", "min_max", "robust", "percentile"].

        Raises:
            ValueError: Əgər yanlış normalizasiya metodu verilsə.
        """
        self.dicom_dir: str = dicom_dir
        self.output_dir: str = output_dir
        self.target_size: Tuple[int, int] = target_size
        self.normalization_method: str = normalization_method

        os.makedirs(self.output_dir, exist_ok=True)

        self.dcm_processor: DCMProcessor = DCMProcessor()

        valid_methods = ["z_score", "min_max", "robust", "percentile"]
        if normalization_method not in valid_methods:
            raise ValueError(
                f"Invalid normalization method. Must be one of {valid_methods}"
            )

    def normalize_image(
        self,
        image: np.ndarray,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Verilən tibbi görüntünü normalizasiya edir.

        Args:
            image (np.ndarray): Giriş görüntüsü.
            method (Optional[str], optional): Normalizasiya metodu.
                Əgər `None` verilsə, default olaraq obyektin
                `self.normalization_method` dəyəri götürüləcək.

        Returns:
            np.ndarray: Normalizə edilmiş görüntü.

        Raises:
            ValueError: Əgər naməlum normalizasiya metodu verilsə.
        """
        if method is None:
            method = self.normalization_method

        image_float: np.ndarray = image.astype(np.float32)

        if method == "z_score":
            mean_val: float = np.mean(image_float)
            std_val: float = np.std(image_float)
            if std_val > 0:
                normalized = (image_float - mean_val) / std_val
            else:
                normalized = image_float - mean_val

        elif method == "min_max":
            min_val: float = np.min(image_float)
            max_val: float = np.max(image_float)
            if max_val > min_val:
                normalized = (image_float - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(image_float)

        elif method == "robust":
            median_val: float = np.median(image_float)
            q75, q25 = np.percentile(image_float, [75, 25])
            iqr: float = q75 - q25
            if iqr > 0:
                normalized = (image_float - median_val) / iqr
            else:
                normalized = image_float - median_val

        elif method == "percentile":
            p1, p99 = np.percentile(image_float, [1, 99])
            if p99 > p1:
                normalized = np.clip((image_float - p1) / (p99 - p1), 0, 1)
            else:
                normalized = np.zeros_like(image_float)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        Görüntünü verilən ölçüyə dəyişir.

        Args:
            image (np.ndarray): Giriş görüntüsü.
            target_size (Optional[Tuple[int, int]], optional): Resize ölçüsü.
                Default olaraq obyektin `self.target_size` dəyəri.
            interpolation (int, optional): İnterpolasiya metodu.
                Default `cv2.INTER_LINEAR`.

        Returns:
            np.ndarray: Resize edilmiş görüntü.
        """
        if target_size is None:
            target_size = self.target_size

        return cv2.resize(image, target_size, interpolation=interpolation)

    def apply_window_level(
        self,
        image: np.ndarray,
        window: float,
        level: float,
    ) -> np.ndarray:
        """
        Görüntüyə window və level tətbiq edir.

        Args:
            image (np.ndarray): Giriş görüntüsü.
            window (float): Window dəyəri.
            level (float): Level dəyəri.

        Returns:
            np.ndarray: Tətbiq edilmiş görüntü.
        """
        min_val: float = level - window / 2
        max_val: float = level + window / 2

        windowed: np.ndarray = np.clip(image, min_val, max_val)

        if max_val > min_val:
            windowed = (
                (windowed - min_val) / (max_val - min_val) * 255
            ).astype(np.uint8)
        else:
            windowed = np.zeros_like(windowed, dtype=np.uint8)

        return windowed

    def preprocess_single_image(
        self,
        dicom_path: str,
        output_filename: Optional[str] = None,
        apply_clahe: bool = True,
        window: Optional[float] = None,
        level: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Tək bir DICOM faylını emal edir.

        Addımlar:
            - Faylı oxuyur
            - (Opsional) Window/level tətbiq edir
            - Normalizasiya
            - CLAHE (opsional)
            - Resize
            - Faylı PNG kimi saxlayır

        Args:
            dicom_path (str): DICOM faylının yolu.
            output_filename (Optional[str], optional): Çıxış fayl adı.
            apply_clahe (bool, optional): CLAHE tətbiq edilsinmi.
                Default True.
            window (Optional[float], optional): Window dəyəri.
            level (Optional[float], optional): Level dəyəri.

        Returns:
            Optional[np.ndarray]: Emal edilmiş görüntü.
                Əgər səhv olarsa, `None`.
        """
        try:
            image_data: Optional[np.ndarray] = (
                self.dcm_processor.read_dicom_image(dicom_path)
            )
            if image_data is None:
                logger.error(f"Failed to read DICOM file: {dicom_path}")
                return None

            if window is not None and level is not None:
                image_data = self.apply_window_level(image_data, window, level)

            normalized_image: np.ndarray = self.normalize_image(image_data)

            if self.normalization_method in ["z_score", "robust"]:
                normalized_image = np.clip(normalized_image, -3, 3)
                display_image: np.ndarray = (
                    (normalized_image + 3) / 6 * 255
                ).astype(np.uint8)
            else:
                display_image = (normalized_image * 255).astype(np.uint8)

            if apply_clahe:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                display_image = clahe.apply(display_image)

            resized_image: np.ndarray = self.resize_image(display_image)

            if output_filename:
                output_path: str = os.path.join(
                    self.output_dir, output_filename
                )
                cv2.imwrite(output_path, resized_image)
                logger.info(f"Processed image saved: {output_path}")

            return resized_image

        except Exception as e:
            logger.error(f"Error processing image {dicom_path}: {str(e)}")
            return None

    def batch_process(
        self,
        file_extension: str = ".dcm",
        apply_clahe: bool = True,
        window: Optional[float] = None,
        level: Optional[float] = None,
    ) -> Dict[str, int]:
        """
        Qovluqdakı bütün DICOM fayllarını emal edir.

        Args:
            file_extension (str, optional): Fayl uzantısı. Default ".dcm".
            apply_clahe (bool, optional): CLAHE tətbiq edilsinmi.
                Default True.
            window (Optional[float], optional): Window dəyəri.
            level (Optional[float], optional): Level dəyəri.

        Returns:
            Dict[str, int]: Emal nəticələrinin statistikası:
                - processed
                - failed
                - total
        """
        results: Dict[str, int] = {"processed": 0, "failed": 0, "total": 0}

        dicom_files: List[str] = []
        for root, dirs, files in os.walk(self.dicom_dir):
            for file in files:
                if file.lower().endswith(file_extension.lower()):
                    dicom_files.append(os.path.join(root, file))

        results["total"] = len(dicom_files)
        logger.info(f"Found DICOM files: {results['total']}")

        for i, dicom_path in enumerate(dicom_files, 1):
            filename: str = os.path.splitext(os.path.basename(dicom_path))[0]
            output_filename: str = f"{filename}_processed.png"

            logger.info(f"Processing ({i}/{results['total']}): {filename}")

            processed_image: Optional[np.ndarray] = self.preprocess_single_image(
                dicom_path,
                output_filename,
                apply_clahe=apply_clahe,
                window=window,
                level=level,
            )

            if processed_image is not None:
                results["processed"] += 1
            else:
                results["failed"] += 1

        logger.info(
            f"Batch processing completed: {results['processed']} successful, "
            f"{results['failed']} failed"
        )
        return results

    def get_image_statistics(self, dicom_path: str) -> Optional[Dict[str, Any]]:
        """
        Verilən DICOM faylı üçün statistik göstəriciləri qaytarır.

        Args:
            dicom_path (str): DICOM faylının yolu.

        Returns:
            Optional[Dict[str, Any]]: Statistik məlumatlar:
                - shape
                - dtype
                - min
                - max
                - mean
                - std
                - median
                Əgər fayl oxunmazsa, `None`.
        """
        try:
            image_data: Optional[np.ndarray] = (
                self.dcm_processor.read_dicom_image(dicom_path)
            )
            if image_data is None:
                return None

            stats: Dict[str, Any] = {
                "shape": image_data.shape,
                "dtype": str(image_data.dtype),
                "min": float(np.min(image_data)),
                "max": float(np.max(image_data)),
                "mean": float(np.mean(image_data)),
                "std": float(np.std(image_data)),
                "median": float(np.median(image_data)),
            }

            return stats

        except Exception as e:
            logger.error(f"Error calculating statistics for {dicom_path}: {str(e)}")
            return None


def main() -> None:
    """
    Misal üçün istifadə.
    """
    dicom_directory: str = "path/to/dicom/files"
    output_directory: str = "path/to/output"

    preprocessor: DICOMPreprocessor = DICOMPreprocessor(
        dicom_dir=dicom_directory,
        output_dir=output_directory,
        target_size=(224, 224),
        normalization_method="percentile",
    )

    results: Dict[str, int] = preprocessor.batch_process(
        file_extension=".dcm",
        apply_clahe=True,
        window=400.0,
        level=200.0,
    )

    print(f"Results: {results}")


if __name__ == "__main__":
    main()
