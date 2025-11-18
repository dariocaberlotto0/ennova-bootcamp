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
1. generative_ai_adaptor: practical session to explore OpenAI and Google APIs
2. text_cleaner: practical session to apply clean-code patterns in small utilities.
3. path_manager: practical session to manage file paths using dataclasses and pathlib
4. cli: practical session to learn how to write a command-line utility that "normalizes" its (string) input