import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import boto3

# Use explicit S3 bucket to avoid needing s3:CreateBucket permissions
bucket = 'creditcarddata1204'  # ✅ Replace with your actual bucket if different
model_key = 'fraud-detection/fraud_model.tar.gz'
model_path = f's3://{bucket}/{model_key}'

# Set up SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Upload model to S3
sagemaker_session.upload_data(path='fraud_model.pkl', bucket=bucket, key_prefix='fraud-detection')

# Create SageMaker model
model = SKLearnModel(
    model_data=model_path,
    role=role,
    entry_point='inference.py',
    framework_version='0.23-1',
    sagemaker_session=sagemaker_session
)

# Deploy the model to a SageMaker endpoint
predictor = model.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1
)

print("✅ Deployed model to endpoint:", predictor.endpoint_name)
