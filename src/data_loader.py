# src/data_loader.py

import pandas as pd
import yaml

def load_config(config_path="config.yaml"):
    print(f"🔍 Looking for config at: {config_path}")
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            print("✅ YAML loaded as:", config)
            return config
    except FileNotFoundError:
        print("❌ config.yaml file not found!")
    except yaml.YAMLError as e:
        print("❌ YAML parsing error:", e)

def load_dataset(dataset_path):
    """
    Load the dataset from CSV using pandas.
    """
    df = pd.read_csv(dataset_path)
    return df

def get_data():
    """
    High-level function to load config, get dataset path, and return DataFrame.
    """
    config = load_config()
    dataset_path = config["dataset"]
    df = load_dataset(dataset_path)
    return df
