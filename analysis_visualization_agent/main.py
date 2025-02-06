import pandas as pd
from executor import CodeExecutor
from visualization_prompt import get_user_prompt, system_prompt

from llm import llm_visualize
from utils import dataframe_to_markdown

if __name__ == "__main__":
    dataset = ""  # dataset path here
    prompt = ""  # prompt
    datafarme_summary = dataframe_to_markdown(dataset)
    user_prompt = get_user_prompt(datafarme_summary, prompt)
    print(user_prompt)
    response = llm_visualize(system_prompt, user_prompt)
    if response.code is not None:
        print("response", response.code)
        df = pd.read_csv(dataset)
        executor = CodeExecutor(df)
        executor.execute_code(response.code)
    else:
        print(response.refusal)
