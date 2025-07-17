import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.xgboost.model import XGBoostModel

# Set up SageMaker session and role
session = sagemaker.Session()
role = get_execution_role()

# Your model artifact location (S3)
model_data_path = "s3://creditcarddata1204/model-output-1306/model.tar.gz"

# Endpoint configuration
endpoint_name = "fraud-detection-endpoint"

# Delete existing endpoint and endpoint config (if any)
sm_client = boto3.client("sagemaker")

# Delete endpoint if it exists
try:
    sm_client.describe_endpoint(EndpointName=endpoint_name)
    print(f"⚠️ Deleting existing endpoint: {endpoint_name}")
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    waiter = sm_client.get_waiter("endpoint_deleted")
    waiter.wait(EndpointName=endpoint_name)
    print("✅ Endpoint deleted.")
except sm_client.exceptions.ClientError:
    print("ℹ️ No existing endpoint found.")

# Delete endpoint config if it exists
try:
    sm_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
    print(f"⚠️ Deleting existing endpoint config: {endpoint_name}")
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    print("✅ Endpoint config deleted.")
except sm_client.exceptions.ClientError:
    print("ℹ️ No existing endpoint config found.")

# Deploy the model using SageMaker XGBoost container
print("📦 Deploying model to SageMaker...")
model = XGBoostModel(
    model_data=model_data_path,
    role=role,
    entry_point="inference.py",  # Your inference script
    framework_version="1.7-1",   # SageMaker XGBoost version (adjust if needed)
    sagemaker_session=session
)

# Deploy as endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name=endpoint_name
)

print(f"🚀 Model deployed to endpoint: {endpoint_name}")
