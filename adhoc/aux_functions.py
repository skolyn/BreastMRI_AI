from typing import Optional

import pandas as pd


def excel_to_csv(excel_input_path: str, csv_output_path: Optional[str] = None):
    df = pd.read_excel(excel_input_path)
    csv_output_path = csv_output_path if csv_output_path is not None else excel_input_path.replace(".xlsx", ".csv")
    df.to_csv(csv_output_path, index=False)


