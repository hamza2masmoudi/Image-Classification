import pandas as pd
import json
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Function to parse vectors
def parse_vector(vector_str):
    try:
        return json.loads(vector_str)
    except json.JSONDecodeError:
        vector_str = vector_str.replace("np.float64(", "").replace(")", "")
        vector_str = vector_str.replace("[", "").replace("]", "").strip()
        return [float(x) for x in vector_str.split(",")]

# Load the data
ref_data = pd.read_csv("/app/data/ref_data.csv")
prod_data = pd.read_csv("/app/data/prod_data.csv")

# Process the vectors
ref_data["vector"] = ref_data["vector"].apply(parse_vector)
prod_data["vector"] = prod_data["vector"].apply(parse_vector)

# Add the 'prediction' column to ref_data
ref_data["prediction"] = ref_data["target"]

# Remove unused columns for the report
ref_data_for_evidently = ref_data.drop(columns=["vector"])
prod_data_for_evidently = prod_data.drop(columns=["vector"])

# Create the report with DataDrift and Classification presets
report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])

# Run the report
report.run(reference_data=ref_data_for_evidently, current_data=prod_data_for_evidently)

# Save the report locally
report.save_html('/app/data/evidently_report.html')
print("Evidently report saved as 'evidently_report.html'")

# Define the API URL for Evidently UI
evidently_ui_url = "http://localhost:8082/api"

# Create a project in the Evidently UI
project_payload = {
    "project_name": "Image Classification Monitoring",
    "datasets": ["ref_data", "prod_data"]
}
response = requests.post(f"{evidently_ui_url}/projects", json=project_payload)

if response.status_code == 200:
    print("Project created successfully in Evidently UI.")
else:
    print("Failed to create project:", response.text)

# Upload reference dataset
with open("/app/data/ref_data.csv", "rb") as ref_file:
    response = requests.post(f"{evidently_ui_url}/projects/Image%20Classification%20Monitoring/datasets/ref_data",
                             files={"file": ref_file})
    if response.status_code == 200:
        print("Reference dataset uploaded successfully.")
    else:
        print("Failed to upload reference dataset:", response.text)

# Upload production dataset
with open("/app/data/prod_data.csv", "rb") as prod_file:
    response = requests.post(f"{evidently_ui_url}/projects/Image%20Classification%20Monitoring/datasets/prod_data",
                             files={"file": prod_file})
    if response.status_code == 200:
        print("Production dataset uploaded successfully.")
    else:
        print("Failed to upload production dataset:", response.text)