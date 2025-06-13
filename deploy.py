import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import time

# Config
model_name = "fraud-model-v1"
endpoint_config_name = "fraud-detection-endpoint-config"
endpoint_name = "fraud-detection-endpoint"
role = "arn:aws:iam::377632750099:role/datazone_usr_role_5ih12zk69oqyvq_ce0yt0fsdqn7o6"
model_artifact = "s3://creditcarddata1204/model-output-1306/model.tar.gz"

# Initialize clients
sm = boto3.client("sagemaker")
session = sagemaker.Session()

# Clean up existing endpoint if it exists
def delete_if_exists():
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"Deleting existing endpoint: {endpoint_name}")
        sm.delete_endpoint(EndpointName=endpoint_name)
        # Wait until deletion
        while True:
            try:
                sm.describe_endpoint(EndpointName=endpoint_name)
                time.sleep(5)
            except sm.exceptions.ClientError as e:
                if "Could not find" in str(e):
                    break
                else:
                    raise
    except sm.exceptions.ClientError:
        pass

    try:
        sm.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"Deleting existing endpoint config: {endpoint_config_name}")
        sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    except sm.exceptions.ClientError:
        pass

    try:
        sm.describe_model(ModelName=model_name)
        print(f"Deleting existing model: {model_name}")
        sm.delete_model(ModelName=model_name)
    except sm.exceptions.ClientError:
        pass

# Begin
print("Cleaning up existing resources if any...")
delete_if_exists()

print("Registering new model...")
model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    entry_point="train.py",
    framework_version="0.23-1",
    sagemaker_session=session
)

print("Deploying new endpoint...")
predictor = model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1,
    endpoint_name=endpoint_name
)

print(f"Deployment successful! Endpoint name: {endpoint_name}")

import sys
print("✅ Model deployed successfully.")
sys.exit(0)

