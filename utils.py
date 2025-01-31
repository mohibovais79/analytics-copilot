import json
import sqlite3


def load_params(param: str) -> str:
    with open("config.json", "r") as f:
        data = json.load(f)
        return data["params"][param]


def db_conn(db_name: str = load_params("db_name")):
    conn = sqlite3.connect(db_name)
    print("connection established wit: ", db_name)
    return conn
