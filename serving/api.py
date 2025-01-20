import os
import pickle
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the absolute path to the artifacts folder
current_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_path = os.path.join(current_dir, 'artifacts')

# Load the model
try:
    with open(os.path.join(artifacts_path, 'model.pkl'), 'rb') as model_file:
        model = pickle.load(model_file)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load the model: {e}")
    raise

# Load the scaler
try:
    with open(os.path.join(artifacts_path, 'scaler.pkl'), 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    logger.info("Scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load the scaler: {e}")
    raise

app = FastAPI()

class PredictionInput(BaseModel):
    data: list

class FeedbackInput(BaseModel):
    image: list  # Transformed image data (vectorized form)
    predicted_class: int  # Predicted class by the model
    actual_class: int  # True class provided by the user

@app.get("/")
def health_check():
    """
    Health check endpoint to confirm API is running.
    """
    return {"status": "API is up and running."}

@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Predict endpoint to get predictions from the model.
    """
    try:
        # Convert input data to NumPy array
        data = np.array(input_data.data).reshape(1, -1)
        logger.info(f"Input data: {data}")

        # Validate input dimensions
        if data.shape[1] != scaler.mean_.shape[0]:
            raise ValueError(
                f"Input data must have {scaler.mean_.shape[0]} features, but got {data.shape[1]}"
            )

        # Preprocess the data
        scaled_data = scaler.transform(data)
        logger.info(f"Scaled data: {scaled_data}")

        # Get predictions
        prediction = model.predict(scaled_data)
        logger.info(f"Prediction: {prediction}")

        return {"prediction": prediction.tolist()}

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

@app.post("/feedback")
def feedback(data: FeedbackInput):
    """
    Endpoint to collect user feedback for predictions.
    """
    try:
        # 1. Transform and scale the input image vector
        vector = np.array(data.image).reshape(1, -1)
        scaled_vector = scaler.transform(vector)

        # 2. Append the feedback to prod_data.csv
        feedback_entry = pd.DataFrame({
            "vector": [scaled_vector.tolist()],
            "actual": [data.actual_class],
            "prediction": [data.predicted_class]
        })

        prod_data_path = os.path.join(current_dir, 'prod_data.csv')
        header = not os.path.exists(prod_data_path)  # Add header only if the file doesn't exist
        feedback_entry.to_csv(prod_data_path, mode="a", header=header, index=False)
        logger.info("Feedback saved to prod_data.csv.")

        # 3. Trigger model retraining if needed
        k = 10  # Retrain after every 10 feedbacks
        prod_data = pd.read_csv(prod_data_path)
        if len(prod_data) % k == 0:
            train_new_model()

        return {"message": "Feedback received and recorded successfully."}

    except Exception as e:
        logger.error(f"Error in feedback endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

def train_new_model():
    """
    Retrain the model using the combined reference and production data.
    """
    try:
        # Load reference and production data
        ref_data_path = os.path.join(current_dir, 'ref_data.csv')
        prod_data_path = os.path.join(current_dir, 'prod_data.csv')

        ref_data = pd.read_csv(ref_data_path)
        prod_data = pd.read_csv(prod_data_path)

        # Combine datasets
        data = pd.concat([ref_data, prod_data])
        X = np.array([eval(v)[0] for v in data["vector"]])
        y = data["actual"]

        # Retrain the model
        from sklearn.ensemble import RandomForestClassifier
        new_model = RandomForestClassifier()
        new_model.fit(X, y)

        # Save the new model
        with open(os.path.join(artifacts_path, 'model.pkl'), 'wb') as model_file:
            pickle.dump(new_model, model_file)

        # Update the global model
        global model
        model = new_model
        logger.info("New model trained and deployed successfully.")

    except Exception as e:
        logger.error(f"Error during model retraining: {e}")
