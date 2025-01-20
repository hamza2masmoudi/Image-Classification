import pandas as pd

# Load and update ref_data.csv
ref_data = pd.read_csv("/Users/hamza/Downloads/image-classification-main/data/ref_data.csv")
ref_data.rename(columns={"label": "target"}, inplace=True)
ref_data.to_csv("/Users/hamza/Downloads/image-classification-main/data/ref_data.csv", index=False)

# Load and update prod_data.csv
prod_data = pd.read_csv("/Users/hamza/Downloads/image-classification-main/data/prod_data.csv")
prod_data.rename(columns={"real_label": "target"}, inplace=True)
prod_data.to_csv("/Users/hamza/Downloads/image-classification-main/data/prod_data.csv", index=False)

print("Column names updated successfully!")