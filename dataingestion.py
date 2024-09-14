import pandas as pd

def load_data(file):
    file_type = file.type
    if file_type == "text/csv":
        df = pd.read_csv(file)
    elif file_type == "application/json":
        df = pd.read_json(file)
    elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type")
    return df
