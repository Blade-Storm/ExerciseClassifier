import json

def load_json(filepath):
    with open(filepath, 'r') as f:
        json_content = json.load(f)

    return json_content