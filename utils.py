import json

import pandas as pd
import sqlite3


def load_params(param: str) -> str:
    with open("config.json", "r") as f:
        data = json.load(f)
        return data["params"][param]


def db_conn(db_name: str = load_params("db_name")) -> sqlite3.Connection:
    conn = sqlite3.connect(db_name)
    return conn

def serialize_dataframe(df: pd.DataFrame, n_results: int = 20) -> str:
    summary = f"Total Rows: {df.shape[0]}\n\n"
    
    numeric_cols = df.select_dtypes(include=['number'])
    if not numeric_cols.empty:
        means = numeric_cols.mean()
        medians = numeric_cols.median()
        modes = numeric_cols.mode().iloc[0]  
        
        stats_summary = "Statistical Summary:\n"
        stats_summary += "Mean:\n"
        for col, val in means.items():
            stats_summary += f"  {col}: {val:.2f}\n"
        stats_summary += "\nMedian:\n"
        for col, val in medians.items():
            stats_summary += f"  {col}: {val:.2f}\n"
        stats_summary += "\nMode:\n"
        for col, val in modes.items():
            stats_summary += f"  {col}: {val}\n"
        stats_summary += "\n"
        summary += stats_summary
    
    if df.shape[0] > n_results:
        head_count = n_results // 2
        tail_count = n_results - head_count
        
        preview = "Sample Preview:\n"
        preview += f"  First {head_count} rows:\n"
        preview += df.head(head_count).to_string() + "\n\n"
        preview += f"  Last {tail_count} rows:\n"
        preview += df.tail(tail_count).to_string() + "\n"
        summary += preview
    else:
        summary += df.to_string()
    
    return summary