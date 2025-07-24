import csv
import os
from datetime import datetime

def log_experiment(result_dict, log_file="results/experiment_log.csv"):
    """
    Logs experiment metrics and metadata into a CSV file.
    """

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_exists = os.path.isfile(log_file)

    if file_exists:
        with open(log_file, mode="r") as f:
            reader = csv.reader(f)
            headers = next(reader)
    else:
        headers = list(result_dict.keys())

    result_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {key: result_dict.get(key, None) for key in headers}

    with open(log_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)

        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
