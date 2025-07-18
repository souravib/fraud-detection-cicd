import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import boto3

# SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# S3 path to model.tar.gz
model_data_path = "s3://creditcarddata1204/model-output-1306/model.tar.gz"

# Create SKLearnModel with dependencies
sklearn_model = SKLearnModel(
    model_data=model_data_path,
    role=role,
    entry_point="inference.py",        # ✅ Your custom inference script
    source_dir="scripts",              # ✅ Directory containing inference.py
    framework_version="1.2-1",         # ✅ Matches sklearn version used
    py_version="py3",
    sagemaker_session=sagemaker_session,
    dependencies=["requirements.txt"]  # ✅ Add custom requirements
)

# Deploy the model
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",  # Or ml.t2.medium for testing
)

print("✅ Model deployed successfully.")
print("Endpoint name:", predictor.endpoint_name)
