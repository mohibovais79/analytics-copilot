def get_system_message(db_info: str):
    prompt = f"""you are provided following information on relational tables {db_info}.
    Provide correct sqlite query for the analysis ask by user only query nothing else query should be in stirng format"""
    return prompt
