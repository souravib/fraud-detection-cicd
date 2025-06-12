import tarfile
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import os

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# ✅ Create a tar.gz archive with both model and inference script
with tarfile.open('fraud_model.tar.gz', mode='w:gz') as tar:
    tar.add('fraud_model.pkl', arcname='fraud_model.pkl')
    tar.add('inference.py', arcname='inference.py')

# ✅ Upload tar.gz to S3
bucket = 'sagemaker-eu-west-1-377632750099'
prefix = 'fraud-detection'
model_path = sagemaker_session.upload_data(path='fraud_model.tar.gz', bucket=bucket, key_prefix=prefix)

print("📦 Model archive uploaded to:", model_path)

# ✅ Deploy from the tar.gz
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
