import ast
import importlib
import re
import textwrap
from typing import Optional, Tuple

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

from utils import load_params

# allowed_imports = load_params("allowed_modules")


class CodeExecutor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy(deep=True)
        self.safe_globals = {}

        # for module in allowed_imports:
        #     self.safe_globals[module] = importlib.import_module(module)
        self.safe_globals["df"] = self.df

    def filter_code(self, code: str) -> str:
        code = textwrap.dedent(code.replace("\t", "    "))

        try:
            tree = ast.parse(code)
        except IndentationError as e:
            print(f"Indentation Error in generated code: {e}")
            return ""
        except SyntaxError as e:
            print(f"Syntax Error in generated code: {e}")
            return ""

        # filtered_lines = []

        # for node in tree.body:
        # if isinstance(node, ast.Import):
        #     # filtered_names = [alias for alias in node.names if alias.name in allowed_imports]
        #     if filtered_names:
        #         new_node = ast.Import(names=filtered_names)
        #         filtered_lines.append(ast.unparse(new_node))

        # elif isinstance(node, ast.ImportFrom):
        #     if any(node.module == mod or node.module.startswith(mod + ".") for mod in allowed_imports):
        #         filtered_lines.append(ast.unparse(node))
        # else:
        #     filtered_lines.append(ast.unparse(node))

        # cleaned_code = "\n".join(filtered_lines)

        finalized_code = f"import seaborn as sns\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport numpy as np\n{code}\n\nplt.tight_layout()"

        return finalized_code

    def execute_code(self, code: str) -> Optional[Tuple[str, str]]:
        """Executes sanitized code in a controlled environment."""
        current_datetime = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.final_df: Optional[pd.DataFrame] = None
        self.chart = None
        final_df_path = f"outputs/output_{current_datetime}"

        self.locals = {"df": self.df, "chart": self.chart, "final_df": self.final_df}
        clean_code = self.filter_code(code)

        try:
            with open(f"outputs/code_{current_datetime}.py", "w") as f:
                f.write(clean_code)
            exec(clean_code, None, self.locals)
            self.chart = self.locals["chart"]
            self.final_df = self.locals["final_df"]

            if self.chart is not None:
                self.chart.savefig(f"outputs/output_{current_datetime}.png")
            if self.final_df is not None:
                self.final_df.to_csv(f"{final_df_path}.csv", index=False)
                return final_df_path, clean_code
            return "", clean_code

        except Exception as e:
            print(f"Error while executing code: {str(e)}")
            print("Traceback:", e.__traceback__)
