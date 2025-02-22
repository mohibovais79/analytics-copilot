import ast
import sqlite3

import pandas as pd

from data.loader import db_conn


def execute_sql(query: str, db_conn=db_conn):
    if query.lower() == "none":
        return None

    if query.lower().startswith("insert") or query.lower().startswith("update") or query.lower().startswith("delete"):
        print("write operation not allowed")
        return None

    connection = db_conn()

    df = pd.read_sql_query(query, connection)
    markdown_output = df.to_markdown(index=False)
    return markdown_output


def analyze_sqlite_db(db_conn: sqlite3.Connection = db_conn(), table_names: list[str] = []) -> str:
    if isinstance(table_names, str):
        try:
            table_names = ast.literal_eval(table_names)
        except Exception as e:
            print(f"Error converting table_names from string to list: {e}")
            table_names = []
    cursor = db_conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    existing_table_names = [table[0] for table in tables]

    if table_names:
        valid_table_names = [name for name in table_names if name in existing_table_names]
        if valid_table_names:
            tables_to_analyze = [(t,) for t in valid_table_names]
            print(f"Using only the following tables: {valid_table_names}")
        else:
            print(f"None of the specified tables exist. Using all tables: {existing_table_names}")
            tables_to_analyze = tables
    else:
        print(f"Using all tables: {existing_table_names}")
        tables_to_analyze = tables

    analysis_results = []

    for table in tables_to_analyze:
        table_name = table[0]

        df = pd.read_sql_query(f"SELECT * FROM {table_name}", db_conn)

        markdown = f"## Table: {table_name}\n"
        markdown += f"**Number of Rows**: {len(df)}\n\n"

        for column in df.columns:
            markdown += f"### Column: {column}\n"
            markdown += f"**Column Type**: {df[column].dtype}\n"

            if pd.api.types.is_numeric_dtype(df[column]):
                markdown += f"**Statistics**:\n"
                markdown += f"- Mean: {df[column].mean()}\n"
                markdown += f"- Max: {df[column].max()}\n"
                markdown += f"- Min: {df[column].min()}\n"

            else:
                markdown += f"**Statistics**: None\n\n"

            if pd.api.types.is_object_dtype(df[column]):
                unique_values = df[column].unique()
                if len(unique_values) <= 20:
                    truncated_values = [
                        (value[:50] + "...") if len(str(value)) > 50 else str(value) for value in unique_values
                    ]
                    markdown += f"**Categorical Values**: {', '.join(truncated_values)}\n"
                markdown += f"**Unique Count**: {len(unique_values)}\n\n"

            else:
                markdown += f"**Categorical Values**: None\n"

        markdown += f"**Missing Values**: {df[column].isnull().sum()}\n\n"
        n_sample = 3
        sample_top = df.head(n_sample).copy()
        sample_bottom = df.tail(n_sample).copy()

        for col in df.select_dtypes(include=["object"]).columns:
            sample_top[col] = sample_top[col].apply(lambda x: str(x)[:50] + "..." if len(str(x)) > 50 else x)
            sample_bottom[col] = sample_bottom[col].apply(lambda x: str(x)[:50] + "..." if len(str(x)) > 50 else x)

        markdown += f"**Missing Values**: {df[column].isnull().sum()}\n\n"

        markdown += f"### First {n_sample} Rows\n"
        markdown += sample_top.to_markdown(index=False).replace("|", "").replace(":", "").replace("-", "") + "\n"
        markdown += f"### Last {n_sample} Rows\n"
        markdown += sample_bottom.to_markdown(index=False).replace("|", "").replace(":", "").replace("-", "") + "\n"

        analysis_results.append(markdown)

    cursor.close()
    final_schema = "\n".join(analysis_results)

    return final_schema
