import os
import sys
import pickle
from typing import Any, Dict, List, Union

import yaml
from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: Any) -> None:
    """
    Save a Python object to disk using pickle
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Object saved to {file_path}")
            
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """
    Load a Python object from disk using pickle
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
            
        logging.info(f"Object loaded from {file_path}")
        return obj
            
    except Exception as e:
        logging.error(f"Error loading object: {e}")
        raise CustomException(e, sys)


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, create it if it doesn't
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        logging.info(f"Directory created or already exists: {directory_path}")
    except Exception as e:
        logging.error(f"Error creating directory: {e}")
        raise CustomException(e, sys)