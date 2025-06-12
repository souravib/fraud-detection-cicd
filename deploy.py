import tarfile
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import os

# Set up SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# 🔁 Clean up any existing tar.gz archive to avoid stale uploads
if os.path.exists('fraud_model.tar.gz'):
    os.remove('fraud_model.tar.gz')

# ✅ Create tar.gz archive including model and inference script
with tarfile.open('fraud_model.tar.gz', mode='w:gz') as tar:
    tar.add('fraud_model.pkl', arcname='fraud_model.pkl')
    tar.add('inference.py', arcname='inference.py')

# ✅ Upload to the correct S3 bucket/prefix
bucket = 'sagemaker-eu-west-1-377632750099'
prefix = 'fraud-detection'
model_path = sagemaker_session.upload_data(path='fraud_model.tar.gz', bucket=bucket, key_prefix=prefix)

print("📦 Model archive uploaded to:", model_path)

# ✅ Define and deploy model using SageMaker
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
