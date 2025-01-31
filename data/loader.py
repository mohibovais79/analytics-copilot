import json
import os
import sqlite3

import pandas as pd

from utils import db_conn, load_params


class CSVToSQLite:
    def __init__(self, db_name: str):
        """Initializes the SQLite database connection and creates the database if it doesn't exist."""
        self.db_name = db_name
        if not os.path.exists(db_name):
            print(f"Database '{db_name}' does not exist. Creating a new one.")
        self.connection = db_conn()
        self.cursor = self.connection.cursor()

    def create_table_from_csv(self, csv_file: str, table_name: str):
        """Converts a CSV file into an SQLite table."""
        df = pd.read_csv(csv_file)

        columns = ", ".join([f'"{col}" TEXT' for col in df.columns])
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"
        self.cursor.execute(create_table_query)

        df.to_sql(table_name, self.connection, if_exists="replace", index=False)
        print(f"Data from '{csv_file}' inserted into table '{table_name}'.")

    def convert_csvs_to_sqlite(self, csv_files: list[str]):
        """Converts a list of CSV files into SQLite tables."""
        for csv_file in csv_files:
            table_name = os.path.splitext(os.path.basename(csv_file))[0]
            self.create_table_from_csv(csv_file, table_name)

    def commit_and_close(self):
        """Commits the transaction and closes the connection."""
        self.connection.commit()
        self.connection.close()
        print(f"Database '{self.db_name}' saved and connection closed.")


if __name__ == "__main__":
    csv_files = [
        "data/name_basics.csv",
        "data/title_akas.csv",
        "data/title_basics.csv",
        "data/title_principals.csv",
        "data/title_ratings.csv",
    ]
    db_name = load_params("db_name")

    csv_to_sqlite = CSVToSQLite(db_name)

    csv_to_sqlite.convert_csvs_to_sqlite(csv_files)

    csv_to_sqlite.commit_and_close()
