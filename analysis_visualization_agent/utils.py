import json

import pandas as pd


def load_params(param: str):
    with open("config.json", "r") as f:
        data = json.load(f)
    return data["params"][param]


def dataframe_to_markdown(file_path: str, num_samples: int = 5, max_unique_values: int = 10) -> str:
    df = pd.read_csv(file_path)
    summary = df.describe()

    markdown = "# DataFrame Summary\n\n"

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

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    if not categorical_columns.empty:
        markdown += "\n## Categorical Columns Unique Values\n"
        for column in categorical_columns:
            unique_values = df[column].unique()
            total_unique = len(unique_values)
            markdown += f"### {column} (Total Unique: {total_unique})\n"

            if total_unique > max_unique_values:
                markdown += ", ".join(map(str, unique_values[:max_unique_values])) + ", ... (truncated)\n"
            else:
                markdown += ", ".join(map(str, unique_values)) + "\n"

    markdown += "\n## Sample Rows\n"
    markdown += df.head(num_samples).to_markdown()

    return markdown
