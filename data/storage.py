import json
import os

FILE_PATH = "portfolios.json"

def load_portfolios():
    if not os.path.exists(FILE_PATH):
        return {}
    try:
        with open(FILE_PATH, "r") as f:
            return json.load(f)
    except:
        return {}

def save_portfolio(name, assets_data):
    data = load_portfolios()
    data[name] = assets_data
    with open(FILE_PATH, "w") as f:
        json.dump(data, f, indent=4)

def delete_portfolio(name):
    data = load_portfolios()
    if name in data:
        del data[name]
        with open(FILE_PATH, "w") as f:
            json.dump(data, f, indent=4)