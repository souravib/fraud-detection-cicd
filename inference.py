# inference.py

import joblib
import os
import numpy as np
import json

def model_fn(model_dir):
    """Load the trained model from disk"""
    return joblib.load(os.path.join(model_dir, "model.pkl"))

def input_fn(request_body, request_content_type):
    """Parse input data from the request"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data)
    elif request_content_type == "text/csv":
        return np.fromstring(request_body, sep=",").reshape(1, -1)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make a prediction"""
    return model.predict(input_data)

def output_fn(prediction, response_content_type):
    """Format prediction output"""
    if response_content_type == "application/json":
        return json.dumps(prediction.tolist())
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
