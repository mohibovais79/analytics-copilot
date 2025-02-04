import json

import pandas as pd


def load_params(param: str):
    with open("config.json", "r") as f:
        data = json.load(f)
    return data["params"][param]


def dataframe_to_markdown(file_path: str, num_samples=5) -> str:
    df = pd.read_csv(file_path)
    summary = df.describe()

    markdown = "# DataFrame Summary\n\n"
    markdown += f"location of dataframe= {file_path}\n\n"

    markdown += "## Numeric Columns Summary\n"
    markdown += summary.to_markdown()
    markdown += "\n"

    missing_values = df.isnull().sum()
    markdown += "## Missing Values Count\n"
    markdown += "| Column Name | Missing Values |\n"
    markdown += "|-------------|----------------|\n"
    for column, missing in missing_values.items():
        markdown += f"| {column} | {missing} |\n"

    markdown += "\n## Column Data Types\n"
    markdown += "| Column Name | Data Type |\n"
    markdown += "|-------------|-----------|\n"
    for column, dtype in df.dtypes.items():
        markdown += f"| {column} | {dtype} |\n"

    markdown += "\n## Sample Rows\n"
    markdown += df.head(num_samples).to_markdown()

    return markdown
