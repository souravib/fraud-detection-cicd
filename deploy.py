import boto3
from sagemaker.sklearn.model import SKLearnModel

sagemaker = boto3.client("sagemaker")

model_name = "fraud-model-v1"
endpoint_name = "fraud-detection-endpoint"
role = "arn:aws:iam::377632750099:role/datazone_usr_role_5ih12zk69oqyvq_ce0yt0fsdqn7o6"

# Create model
model = SKLearnModel(
    model_data="s3://creditcarddata1204/model-output-1306/model.tar.gz",
    role=role,
    entry_point="train.py",  # or your actual inference script
    framework_version="0.23-1",
    py_version="py3",
    sagemaker_session=None  # Optional unless you're customizing
)

# Deploy model with update
predictor = model.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1,
    endpoint_name=endpoint_name,
    update_endpoint=True  # This line allows redeploying to existing endpoint
)
