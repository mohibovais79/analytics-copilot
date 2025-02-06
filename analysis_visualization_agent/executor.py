import ast
import importlib
import re
import textwrap

import pandas as pd
from matplotlib import pyplot as plt

from utils import load_params

allowed_imports = load_params("allowed_modules")


class CodeExecutor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.safe_globals = {}

        for module in allowed_imports:
            self.safe_globals[module] = importlib.import_module(module)
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

        filtered_lines = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                filtered_names = [alias for alias in node.names if alias.name in allowed_imports]
                if filtered_names:
                    new_node = ast.Import(names=filtered_names)
                    filtered_lines.append(ast.unparse(new_node))

            elif isinstance(node, ast.ImportFrom):
                if any(node.module == mod or node.module.startswith(mod + ".") for mod in allowed_imports):
                    filtered_lines.append(ast.unparse(node))
            else:
                filtered_lines.append(ast.unparse(node))

        cleaned_code = "\n".join(filtered_lines)

        finalized_code = f"import seaborn as sns\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport numpy as np\n{cleaned_code}\n\nplt.tight_layout()\nplt.savefig('outputs/output.png')\nplt.show()"

        return finalized_code

    def execute_code(self, code: str):
        """Executes sanitized code in a controlled environment."""

        clean_code = self.filter_code(code)

        try:
            print(clean_code)
            exec(clean_code, self.safe_globals)

        except Exception as e:
            print(f"Error while executing code: {str(e)}")
            print("Traceback:", e.__traceback__)
