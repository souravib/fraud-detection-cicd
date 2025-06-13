import boto3
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import Session
import botocore

# Configuration
model_name = "fraud-model-v1"
endpoint_name = "fraud-detection-endpoint"
model_data_path = "s3://creditcarddata1204/model-output-1306/model.tar.gz"
role = "arn:aws:iam::377632750099:role/datazone_usr_role_5ih12zk69oqyvq_ce0yt0fsdqn7o6"

# Create SageMaker session and client
session = Session()
sagemaker_client = boto3.client("sagemaker")

# Step 1: Delete existing endpoint and config if they exist
def delete_if_exists(endpoint_name):
    try:
        sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"⚠️ Endpoint '{endpoint_name}' already exists. Deleting...")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        waiter = sagemaker_client.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=endpoint_name)
        print(f"✅ Endpoint '{endpoint_name}' deleted.")
    except botocore.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e):
            print("ℹ️ No existing endpoint found.")
        else:
            raise

    try:
        print(f"Deleting endpoint config '{endpoint_name}'...")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print("✅ Endpoint config deleted.")
    except botocore.exceptions.ClientError as e:
        if "Could not find" in str(e):
            print("ℹ️ No existing endpoint config found.")
        else:
            raise

delete_if_exists(endpoint_name)

# Step 2: Define the SKLearn model
model = SKLearnModel(
    model_data=model_data_path,
    role=role,
    framework_version="0.23-1",
    py_version="py3",
    sagemaker_session=session,
)

# Step 3: Deploy the model
try:
    print(f"🚀 Deploying model to endpoint: {endpoint_name}")
    predictor = model.deploy(
        instance_type='ml.m5.large',
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        update_endpoint=False  # false since we delete and recreate
    )
    print("✅ Deployment successful.")
except Exception as e:
    print("❌ Deployment failed:")
    print(str(e))
