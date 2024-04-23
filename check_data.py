import json

# Assuming the JSON file name and path
json_name = 'arxiv_abstracts.json'
artifact_dir = 'arxiv-abstracts-dataset'
json_path = f"artifacts/{artifact_dir}/{json_name}"

# Load JSON data from the downloaded file
with open(json_path, "r", encoding='utf-8') as f:
    data = json.load(f)

# Use the data as needed
print(data)