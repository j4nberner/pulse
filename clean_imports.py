import os
import subprocess


def sort_imports_in_directory(directory: str):
    """
    Recursively sort imports in all Python files within the given directory.
    Might need to pip install isort if this function fails.

    Args:
        directory (str): The parent directory to process.
    """
    for root, dirs, files in os.walk(directory):
        # Exclude .venv or other unwanted directories
        dirs[:] = [d for d in dirs if d not in {".venv", "__pycache__"}]
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Sorting imports in: {file_path}")
                subprocess.run(["isort", file_path], check=True)


if __name__ == "__main__":
    parent_directory = os.path.abspath(
        os.path.join(os.path.dirname(__file__))
    )  # Adjust to target the parent directory
    sort_imports_in_directory(parent_directory)
