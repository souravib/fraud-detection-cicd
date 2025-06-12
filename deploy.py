import tarfile
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import os

# Set up SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create a tar.gz model archive that includes both the model and inference script
with tarfile.open('fraud_model.tar.gz', mode='w:gz') as tar:
    tar.add('fraud_model.pkl', arcname='fraud_model.pkl')
    tar.add('inference.py', arcname='inference.py')  # ✅ This line is essential

# Upload the tar.gz to S3
bucket = 'creditcarddata1204'
prefix = 'fraud-detection'
model_path = sagemaker_session.upload_data(path='fraud_model.tar.gz', bucket=bucket, key_prefix=prefix)

# Define and deploy the model
model = SKLearnModel(
    model_data=model_path,
    role=role,
    framework_version='0.23-1',
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1
)

print("✅ Deployed model to endpoint:", predictor.endpoint_name)
