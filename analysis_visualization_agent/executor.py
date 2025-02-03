import ast
import importlib
import re

from matplotlib import pyplot as plt

from utils import load_params

allowed_imports = load_params("allowed_modules")


class CodeExecutor:
    def __init__(self):
        self.safe_globals = {}

        for module in allowed_imports:
            self.safe_globals[module] = importlib.import_module(module)

    def filter_code(self, code: str) -> str:
        """Removes disallowed imports from the provided code."""
        code = re.sub(r"```.*?```", "", code, flags=re.DOTALL)
        tree = ast.parse(code)
        filtered_lines = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                node.names = [alias for alias in node.names if alias.name in allowed_imports]
                if node.names:
                    filtered_lines.append(ast.unparse(node))

            elif isinstance(node, ast.ImportFrom):
                if node.module in allowed_imports:
                    filtered_lines.append(ast.unparse(node))

            else:
                filtered_lines.append(ast.unparse(node))

        return "\n".join(filtered_lines)

    def execute_code(self, code):
        """Executes sanitized code in a controlled environment."""

        clean_code = self.filter_code(code)

        try:
            exec(clean_code, self.safe_globals)
            if "plt" in self.safe_globals and plt:
                plt.show()
        except Exception as e:
            print(f"Error while executing code: {str(e)}")
            print("Traceback:", e.__traceback__)
