import pandas as pd
from executor import CodeExecutor
from visualization_prompt import get_explaination_prompt, get_user_prompt, system_prompt

from llm import llm_explain, llm_visualize
from utils import dataframe_to_markdown

if __name__ == "__main__":
    dataset = r""  # dataset path here
    prompt = ""  # prompt
    dataframe_summary = dataframe_to_markdown(dataset)
    user_prompt = get_user_prompt(dataframe_summary, prompt)

    response = llm_visualize(system_prompt, user_prompt)
    if response.code is not None:
        print(response.code)
        df = pd.read_csv(dataset)
        executor = CodeExecutor(df)
        final_df_path, clean_code = executor.execute_code(response.code)
        if final_df_path is not None:
            final_df = pd.read_csv(f"{final_df_path}.csv")
            if final_df.shape[0] <= 10:
                final_df = final_df.to_markdown()
            else:
                final_df = final_df.head(10).to_markdown()
        explanation_prompt = get_explaination_prompt(final_df, prompt, clean_code)
        for response_chunk in llm_explain(explanation_prompt):
            print(response_chunk, end="")

    else:
        print(response.refusal)
