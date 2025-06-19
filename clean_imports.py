import os
import subprocess

#############################
# pip install autoflake isort
#############################


def clean_and_sort_imports_in_directory(directory: str):
    """
    Recursively sort imports and remove unused imports in all Python files within the given directory.
    Requires 'isort' and 'autoflake' to be installed.
    """
    for root, dirs, files in os.walk(directory):
        # Exclude .venv or other unwanted directories
        dirs[:] = [d for d in dirs if d not in {".venv", "__pycache__", "pulse"}]
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Sorting imports in: {file_path}")
                subprocess.run(["isort", file_path], check=True)
                print(f"Removing unused imports in: {file_path}")
                subprocess.run(
                    [
                        "autoflake",
                        "--remove-all-unused-imports",
                        "--in-place",
                        "--remove-unused-variables",
                        file_path,
                    ],
                    check=True,
                )


if __name__ == "__main__":
    parent_directory = os.path.abspath(
        os.path.join(os.path.dirname(__file__))
    )  # Adjust to target the parent directory
    clean_and_sort_imports_in_directory(parent_directory)
