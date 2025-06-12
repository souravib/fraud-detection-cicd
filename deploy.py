from sagemaker.sklearn.model import SKLearnModel
import sagemaker
import os

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

model = SKLearnModel(
    model_data='s3://creditcarddata1204/fraud-detection/fraud_model.tar.gz',  # Update if path changes
    role=role,
    entry_point='inference.py',
    framework_version='0.23-1',
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1
)

print("✅ Deployed model to endpoint:", predictor.endpoint_name)
