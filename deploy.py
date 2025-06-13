import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# Create a SageMaker session
sagemaker_session = sagemaker.Session()

# Define IAM role (same one used in training)
role = "arn:aws:iam::377632750099:role/datazone_usr_role_5ih12zk69oqyvq_ce0yt0fsdqn7o6"

# Define model location in S3
model_data = 's3://creditcarddata1204/model-output-1306/model.tar.gz'

# Create SKLearnModel object
model = SKLearnModel(
    model_data=model_data,
    role=role,
    entry_point='train.py',  # Needed only if your model has custom inference logic
    framework_version='0.23-1',
    py_version='py3',
    sagemaker_session=sagemaker_session
)

# Deploy to SageMaker endpoint
predictor = model.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1,
    endpoint_name='fraud-detection-endpoint'
)
