from typing import Optional

import pandas as pd


def excel_to_csv(excel_input_path: str, csv_output_path: Optional[str] = None):
    """
    Convert an Excel (.xlsx) file to a CSV (.csv) file.

    Args:
        excel_input_path (str): Path to the input Excel file.
        csv_output_path (Optional[str], default=None): Path to save the output CSV file.
            If None, the output file will be saved in the same directory as the input file
            with the same name but with a `.csv` extension.

    Returns:
        None: The function writes the CSV file to disk and does not return anything.
    """
    df = pd.read_excel(excel_input_path)
    csv_output_path = csv_output_path if csv_output_path is not None else excel_input_path.replace(".xlsx", ".csv")
    df.to_csv(csv_output_path, index=False)

