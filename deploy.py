
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import boto3

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sagemaker_session.default_bucket()
model_path = f's3://{bucket}/fraud-detection/fraud_model.pkl'

# Upload model to S3
sagemaker_session.upload_data(path='fraud_model.pkl', bucket=bucket, key_prefix='fraud-detection')

model = SKLearnModel(
    model_data=model_path,
    role=role,
    entry_point='inference.py',
    framework_version='0.23-1',
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(instance_type='ml.m5.large', initial_instance_count=1)
print("Deployed model to endpoint:", predictor.endpoint_name)
