import sqlite3


def execute_sql(query: str):
    if query.lower() == "none":
        return None

    if query.lower().startswith("insert") or query.lower().startswith("update") or query.lower().startswith("delete"):
        print("write operation not allowed")
        return None

    connection = sqlite3.connect("imdb_ijs.db")
    cursor = connection.cursor()

    cursor.execute(query)

    results = cursor.fetchall()
    print("results:", results)
    if results:
        connection.close()

        return results
    else:
        connection.close()

        return None
