import ast
import importlib
import re
import sqlite3
import textwrap
from typing import Any, Optional, Tuple

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

from utils import db_conn, load_params

# allowed_imports = load_params("allowed_modules")


class CodeExecutor:
    def __init__(self):
        self.safe_globals = {}
        self.db_conn: sqlite3.Connection = db_conn()

        # for module in allowed_imports:
        #     self.safe_globals[module] = importlib.import_module(module)

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

    def execute_code(self, code: str, save=True) -> tuple[pd.DataFrame, Optional[Any]]:
        """Executes sanitized code in a controlled environment."""
        current_datetime = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.final_df: Optional[pd.DataFrame] = None
        self.chart = None
        path_prefix = f"analysis_visualization_agent/outputs/output_{current_datetime}"

        self.locals = {"db_conn": self.db_conn, "chart": self.chart, "final_df": self.final_df}
        clean_code = self.filter_code(code)

        try:
            with open(f"analysis_visualization_agent/outputs/code_{current_datetime}.py", "w") as f:
                f.write(clean_code)
            print(clean_code)
            exec(clean_code, None, self.locals)
            self.chart = self.locals["chart"]
            self.final_df: pd.DataFrame = self.locals["final_df"]
            if save:
                if self.chart is not None:
                    self.chart.savefig(f"{path_prefix}.png")
                if self.final_df is not None:
                    self.final_df.to_csv(f"{path_prefix}.csv", index=False)
            return self.final_df, self.chart
        except Exception as e:
            print(f"Error while executing code: {str(e)}")
            print("Traceback:", e.__traceback__)
