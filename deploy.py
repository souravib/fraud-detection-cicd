import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import os

# Set up SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# ✅ Use existing tar.gz from buildspec
model_tarball = 'fraud_model.tar.gz'
assert os.path.exists(model_tarball), "❌ Tarball not found!"

# ✅ Upload to S3
bucket = 'sagemaker-eu-west-1-377632750099'
prefix = 'fraud-detection'
model_path = sagemaker_session.upload_data(path=model_tarball, bucket=bucket, key_prefix=prefix)

print("📦 Uploaded model to:", model_path)

# ✅ Define and deploy model
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
