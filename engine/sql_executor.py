import sqlite3


def execute_sql(query: str):
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
