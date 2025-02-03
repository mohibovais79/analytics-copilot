from openai import OpenAI

from analysis_visualization_agent.executor import CodeExecutor
from analysis_visualization_agent.llm import llm_visualize
from analysis_visualization_agent.utils import dataframe_to_markdown
from analysis_visualization_agent.visualization_prompt import get_user_prompt, system_prompt

if __name__ == "__main__":
    prompt = "i want to see rating distribution"
    datafarme_summary = dataframe_to_markdown("data/title_ratings.csv")
    user_prompt = get_user_prompt(datafarme_summary, prompt)
    response = llm_visualize(system_prompt, OpenAI, user_prompt)
