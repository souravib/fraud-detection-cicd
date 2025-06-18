# inference.py

import joblib
import os

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.pkl"))

def predict_fn(input_data, model):
    return model.predict(input_data)
