import joblib
import pandas as pd
import io

def model_fn(model_dir):
    return joblib.load(f"{model_dir}/fraud_model.pkl")

def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        return pd.read_csv(io.StringIO(request_body), header=None)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return str(prediction.tolist())
