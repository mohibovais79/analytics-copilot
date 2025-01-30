def get_system_message(db_info: str):
    prompt = f"""you are provided following information on relational tables {db_info}.
    Provide correct sqlite query for the analysis ask by user.
    Follow these instructions:
    1. only query nothing else query should be in stirng format.
    2. you will only return queries if it is a read operation, if write operation return None.
    """
    return prompt
