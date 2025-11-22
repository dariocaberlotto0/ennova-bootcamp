import uvicorn
import yaml
import os
import re

def expand_env_vars(text):
    pattern = re.compile(r'\$\{([^}:]+)(?::-([^}]+))?\}')
    def replace(match):
        key = match.group(1)
        default_value = match.group(2)
        value = os.environ.get(key, default_value)
        if value is None: return ""
        return value
    return pattern.sub(replace, text)

def load_log_config(config_path='log_config.yaml'):
    if os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            content = f.read()
            content_expanded = expand_env_vars(content)
            return yaml.safe_load(content_expanded)
    return None

if __name__ == "__main__":
    log_config_dict = load_log_config()
    
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        log_config=log_config_dict,
        reload=True
    )