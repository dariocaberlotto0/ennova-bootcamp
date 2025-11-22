# Ennova Bootcamp
This repo contain all the examples done in Ennova Bootcamp.

## Setup
1. Install **Python 3.12+**. If your system's python is older, use `pyenv` / `pyenv-win` (or `uv` for a full, fast package and project manager).
2. Create and activate a virtual environment:
   
   ```bash
   python -m venv bootcamp
   source bootcamp/bin/activate   # (Linux/Mac)
   bootcamp\Scripts\activate    # (Windows)
   pip install -r requirements.txt
   ```

## List of files
1. generative_ai: practical session to explore OpenAI and Google APIs. Using this to implement all the tools I learn during the Bootcamp
2. text_cleaner: practical session to apply clean-code patterns in small utilities.
3. path_manager: practical session to manage file paths using dataclasses and pathlib
4. cli: practical session to learn how to write a command-line utility that "normalizes" its (string) input
5. text_retrival: example of a small RAG
6. async_handling: example of how to manage async and sync pipelines
7. config: example of different type of files to config a project. Also an example of logging
8. andres_separate_clients: example of separate classes and logging made by Andres
9. inheritance: examples of inheritance between more classes

TODO: run instructions for all examples