import logging
import toml
import yaml
import json

def toml_config():
    try:
        with open('config.toml', 'r') as f:
            config = toml.load(f)
    except FileNotFoundError:
        logging.error("failed to load configuration file.")
        config = {}
    
    # not knowing if logging config exists
    logging_config = config.get('logging', {})
    # if it didn't exist, default to {}
    # not knowing if log level exists
    log_level = logging_config.get('level', 'INFO').upper()
    # if it didn't exist, default to 'INFO'
    
    # logging.basicConfig(level=log_level)
    logging.getLogger().setLevel(log_level)


def yaml_config():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("failed to load configuration file.")
        config = {}
    
    # not knowing if logging config exists
    logging_config = config.get('logging', {})
    # if it didn't exist, default to {}
    # not knowing if log level exists
    log_level = logging_config.get('level', 'INFO').upper()
    # if it didn't exist, default to 'INFO'

    # logging.basicConfig(level=log_level)
    logging.getLogger().setLevel(log_level)


def json_config():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error("failed to load configuration file.")
        config = {}
    
    # not knowing if logging config exists
    logging_config = config.get('logging', {})
    # if it didn't exist, default to {}
    # not knowing if log level exists
    log_level = logging_config.get('level', 'INFO').upper()
    # if it didn't exist, default to 'INFO'

    # logging.basicConfig(level=log_level)
    logging.getLogger().setLevel(log_level)
