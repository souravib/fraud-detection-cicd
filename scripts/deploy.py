import boto3
from sagemaker import get_execution_role, Session
import botocore.exceptions
import sys

# --- Config ---
model_name = "fraud-model-v1"
endpoint_name = "fraud-detection-endpoint"
endpoint_config_name = endpoint_name + "-config"
model_data_path = "s3://creditcarddata1204/model-output-1306/model.tar.gz"

# --- Use SageMaker’s public scikit-learn container ---
region = boto3.Session().region_name
container_image_uri = f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

# --- Create session and client ---
session = Session()
sagemaker_client = session.sagemaker_client
role = get_execution_role()

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

# --- Step 2: Create model ---
print("📦 Creating SageMaker model...")
sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': container_image_uri,
        'ModelDataUrl': model_data_path,
        'Environment': {
            'SAGEMAKER_PROGRAM': 'inference.py'
        }
    },
    ExecutionRoleArn=role
)
print(f"✅ Model '{model_name}' created.")

# --- Step 3: Create endpoint config ---
print("⚙️ Creating endpoint configuration...")
sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.t2.medium',
            'InitialVariantWeight': 1
        }
    ]
)
print(f"✅ Endpoint configuration '{endpoint_config_name}' created.")

# --- Step 4: Create endpoint (async) ---
print("🚀 Creating endpoint asynchronously...")
sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)
print(f"✅ Endpoint creation triggered for '{endpoint_name}'. Check SageMaker console for status.")

# --- Exit immediately to avoid CodeBuild hang ---
sys.exit(0)
