import boto3
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import Session, get_execution_role
import botocore.exceptions

# --- Config ---
model_name = "fraud-model-v1"
endpoint_name = "fraud-detection-endpoint"
model_data_path = "s3://creditcarddata1204/model-output-1306/model.tar.gz"

# --- Create session and client ---
session = Session()
sagemaker_client = session.sagemaker_client
role = get_execution_role()

def delete_existing_resources(endpoint_name):
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
        sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"⚠️ Deleting endpoint config: {endpoint_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print("✅ Endpoint config deleted.")
    except botocore.exceptions.ClientError as e:
        if "Could not find" in str(e):
            print("ℹ️ No existing endpoint config found.")
        else:
            raise

# --- Step 1: Clean up previous deployment ---
delete_existing_resources(endpoint_name)

# --- Step 2: Trigger deploy without waiting ---
model = SKLearnModel(
    model_data=model_data_path,
    role=role,
    entry_point="inference.py",  # must define input/output handling
    framework_version="1.2-1",   # match your scikit-learn version
    py_version="py3",
    sagemaker_session=session
)

try:
    print("🚀 Deploying model to SageMaker endpoint (async)...")
    model.deploy(
        instance_type='ml.t2.medium',
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        update_endpoint=False,
        wait=False  # ✅ Fire-and-forget
    )
    print(f"✅ Endpoint creation triggered for {endpoint_name}. Check SageMaker console for status.")
except Exception as e:
    print("❌ Deployment failed:")
    print(str(e))
