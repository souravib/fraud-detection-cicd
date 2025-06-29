import joblib
import os
import pandas as pd
import json

def model_fn(model_dir):
    """Load the trained pipeline from disk"""
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    print("✅ Model loaded.")
    return model

def input_fn(request_body, request_content_type):
    """Parse input JSON into a DataFrame"""
    if request_content_type == "application/json":
        data = json.loads(request_body)

        # Support both single record and list of records
        if isinstance(data, dict):
            data = [data]

        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Run the model prediction"""
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, response_content_type):
    """Return prediction as JSON"""
    if response_content_type == "application/json":
        return json.dumps({"predictions": prediction.tolist()})
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
