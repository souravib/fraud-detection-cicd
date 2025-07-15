import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import get_execution_role, Session
import botocore.exceptions
import sys

# --- Config ---
model_name = "fraud-model-v1"
endpoint_name = "fraud-detection-endpoint"
endpoint_config_name = endpoint_name + "-config"
model_data_path = "s3://creditcarddata1204/model-output-1306/model.tar.gz"
role = get_execution_role()
session = Session()
sagemaker_client = session.sagemaker_client

def delete_existing_resources(endpoint_name, endpoint_config_name):
    """Delete existing endpoint and config if they exist."""
    try:
        sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"⚠️ Deleting existing endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        waiter = sagemaker_client.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=endpoint_name)
        print("✅ Endpoint deleted.")
    except botocore.exceptions.ClientError as e:
        if "Could not find" in str(e):
            print("ℹ️ No existing endpoint found.")
        else:
            raise

    try:
        sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"⚠️ Deleting endpoint config: {endpoint_config_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        print("✅ Endpoint config deleted.")
    except botocore.exceptions.ClientError as e:
        if "Could not find" in str(e):
            print("ℹ️ No existing endpoint config found.")
        else:
            raise

# --- Step 1: Clean up previous deployment ---
delete_existing_resources(endpoint_name, endpoint_config_name)

# --- Step 2: Deploy model ---
print("📦 Deploying model to SageMaker...")
model = SKLearnModel(
    model_data=model_data_path,
    role=role,
    entry_point="inference.py",  # ✅ Points to your inference handler
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name=endpoint_name,
    wait=False  # ✅ Fire-and-forget deployment
)

print(f"✅ Endpoint creation triggered for '{endpoint_name}'. Check SageMaker console for status.")
sys.exit(0)
