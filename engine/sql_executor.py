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
